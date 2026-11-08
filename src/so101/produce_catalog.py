"""Resumable CPU-idle development curriculum worker; never runs training."""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import resource
import signal
import shutil
import threading

import cv2
import mujoco
import torch

from so101.catalog_sim import CatalogSimulation
from so101.cloud_checkpoint import atomic_json, sha256
from so101.collect_catalog_sim import load_appearance, record_pair
from so101.control_model.simulation_dataset import EntityTrajectoryDataset
from so101.curriculum_ledger import CurriculumLedger
from so101.episode_storage import EpisodeStorage, consolidate
from so101.lego_curriculum import load_catalog, prepare_scene


LOGGER = logging.getLogger(__name__)
CONTACT_SOLVER = "pyramidal-v1"
STATIONARITY_METRIC = "surface-speed-v2"
VELOCITY_TOLERANCE = .005
DISK_FREE_FLOOR = 8 * 1024**3
PAIR_WRITE_ALLOWANCE = 2 * 1024**3


class DiskBackpressure:
    """Keep in-flight simulation state in RAM while shared storage is pressured."""

    def __init__(self, root: Path):
        self.root = root
        self.status_path = root / "disk_status.json"
        self.wakeup = threading.Event()

    def wait(self, required_bytes: int, phase: str,
             stopping: Callable[[], bool] | None = None) -> bool:
        if required_bytes < 0:
            raise ValueError("Disk write allowance must be nonnegative")
        threshold = DISK_FREE_FLOOR + required_bytes
        paused = False
        while True:
            if stopping is not None and stopping():
                return False
            filesystem = os.statvfs(self.root)
            available = filesystem.f_bavail * filesystem.f_frsize
            if available >= threshold:
                if paused or phase != "recording":
                    atomic_json(self.status_path, {
                        "status": "ready", "phase": phase, "available_bytes": available,
                        "required_available_bytes": threshold,
                    })
                return True
            if not paused:
                LOGGER.warning("Curriculum disk pause during %s: %s bytes available, %s needed",
                               phase, available, threshold)
                atomic_json(self.status_path, {
                    "status": "resource_paused", "phase": phase,
                    "available_bytes": available, "required_available_bytes": threshold,
                    "preserves_inflight_state": True,
                })
                paused = True
            self.wakeup.wait(30)

    def before_frame(self) -> None:
        # Allow both camera images and metadata; check before any physics advancement.
        self.wait(1024**2, "recording")


def qualification_write_allowance(attempt: Path) -> int:
    # Count logical bytes (including hard links) conservatively, not reclaimed space.
    return 2 * sum(path.stat().st_size for path in attempt.rglob("*") if path.is_file()) + 64 * 1024**2


def scheduled_scene(sequence: int) -> tuple[str, str, int]:
    if type(sequence) is not int or sequence < 0:
        raise ValueError("Sequence must be a nonnegative integer")
    category = ("separated", "cluster", "overlap", "pile")[sequence % 4]
    task = "view" if sequence % 16 == 15 else ("grasp", "place", "push")[sequence % 3]
    count = {"separated": 2, "cluster": 4, "overlap": 6, "pile": 8}[category]
    return category, task, count


def qualify_pair(root: Path) -> dict:
    """Validate every actual paired SFT sample before immutable publication."""
    report = json.loads((root / "collection.json").read_bytes())
    if report["status"] != "complete":
        raise ValueError("Cannot qualify incomplete pair")
    storage_path = root / "episodes.sqlite"
    if not storage_path.exists():
        staging = root / ".qualified-episodes.sqlite"
        if not staging.exists():
            consolidate(root, staging)
        with EpisodeStorage(staging) as storage:
            storage.verify_sources(root)
            storage.verify()
        staging.chmod(0o444)
        os.link(staging, storage_path)
        staging.unlink()
    else:
        storage_path.chmod(0o444)
    with EpisodeStorage(storage_path) as storage:
        storage.verify_sources(root)
        population = storage.verify()
    dataset = EntityTrajectoryDataset(root, horizon=50, history_steps=4,
                                      storage_path=storage_path)
    try:
        if dataset.pair_groups != [[0, 1]]:
            raise ValueError("Recorded commands did not form an SFT pair")
        for index in range(len(dataset)):
            sample = dataset[index]
            if not all(torch.isfinite(value).all() for value in sample.values()
                       if isinstance(value, torch.Tensor)):
                raise ValueError(f"Non-finite SFT sample {index}")
        return {**population, "samples": len(dataset), "storage_sha256": sha256(storage_path)}
    finally:
        dataset.storage.close()


def recipe(catalog_path: Path, appearance_path: Path, part_ids: list[str], seed: int) -> dict:
    sources = sorted(Path(__file__).parent.rglob("*.py"))
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.relative_to(Path(__file__).parent).as_posix().encode())
        digest.update(source.read_bytes())
    return {
        "source_sha256": digest.hexdigest(), "catalog_sha256": sha256(catalog_path),
        "appearance_sha256": sha256(appearance_path), "part_ids": sorted(part_ids),
        "seed": seed, "split": "development", "mujoco_version": mujoco.__version__,
        "settling": {
            "contact_solver": CONTACT_SOLVER, "stationarity_metric": STATIONARITY_METRIC,
            "velocity_tolerance": VELOCITY_TOLERANCE,
        },
    }


def publish_pair(root: Path, attempt: Path, report: dict) -> Path:
    destination = root / "releases" / report["release_id"]
    if destination.exists():
        if sha256(destination / "episodes.sqlite") != sha256(attempt / "episodes.sqlite"):
            raise ValueError("Published release identity changed")
        return destination
    staging = root / "releases" / (report["release_id"] + ".building")
    staging.mkdir(parents=True, exist_ok=False)
    for name in ("collection.json", "episodes.sqlite", "qualification.json"):
        shutil.copyfile(attempt / name, staging / name)
    shutil.copyfile(attempt / "source" / "parts_catalog.json", staging / "parts_catalog.json")
    with EpisodeStorage(staging / "episodes.sqlite") as storage:
        storage.verify()
    for file in staging.iterdir():
        with file.open("rb") as stream:
            os.fsync(stream.fileno())
        file.chmod(0o444)
    staging.chmod(0o555)
    staging.rename(destination)
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def run(root: Path, catalog_path: Path, appearance_path: Path,
        part_ids: list[str], seed: int, stopping, *, ledger_path: Path | None = None,
        initial_sequence: int = 0) -> None:
    if type(initial_sequence) is not int or initial_sequence < 0:
        raise ValueError("Initial sequence must be nonnegative")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".producer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = recipe(catalog_path, appearance_path, part_ids, seed)
        state_path = root / "producer.json"
        if state_path.exists():
            state = json.loads(state_path.read_bytes())
            if state["recipe"] != identity:
                raise ValueError("Producer source/catalog/runtime changed; use a new root")
        else:
            state = {"recipe": identity, "next_sequence": initial_sequence, "attempts": []}
            atomic_json(state_path, state)
        catalog = load_catalog(catalog_path)
        if part_ids:
            requested = set(part_ids)
            if requested - {part.part_id for part in catalog.parts}:
                raise ValueError("Requested parts are not verified")
            catalog = replace(catalog, parts=tuple(
                part for part in catalog.parts if part.part_id in requested
            ))
        appearance = load_appearance(appearance_path)
        ledger_path = (ledger_path or root / "ledger.sqlite").resolve()
        if state.get("ledger_path", str(ledger_path)) != str(ledger_path):
            raise ValueError("Producer coordination ledger changed")
        state["ledger_path"] = str(ledger_path)
        ledger = CurriculumLedger(ledger_path, audit_seed=seed)
        disk = DiskBackpressure(root)
        try:
            while not stopping():
                if not disk.wait(PAIR_WRITE_ALLOWANCE, "admission", stopping):
                    break
                if recipe(catalog_path, appearance_path, part_ids, seed) != identity:
                    raise ValueError("Pinned producer inputs changed during execution")
                sequence = state.get("active_sequence")
                if sequence is None:
                    sequence = ledger.claim_sequence(minimum=state["next_sequence"])
                    state["active_sequence"] = sequence
                    atomic_json(state_path, state)
                # Feedback-visible generation is development-only, never sealed evaluation.
                category, task, object_count = scheduled_scene(sequence)
                plan = prepare_scene(catalog, seed=seed, sequence=sequence,
                                     category=category, task=task, object_count=object_count)
                family = hashlib.sha256(json.dumps(
                    ["new-catalog-development-v1", seed, sequence],
                    separators=(",", ":"),
                ).encode()).hexdigest()
                ledger.reserve_family(family, "development")
                plan.update(scene_family_id=family, split="development")
                attempt = root / f"attempt_{sequence:09d}"
                entry = {"sequence": sequence, "category": category, "task": task,
                         "object_count": object_count,
                         "source": str(attempt), "status": "interrupted",
                         "worker_source_sha256": identity["source_sha256"]}
                sim = None
                try:
                    if attempt.exists():
                        report_path = attempt / "collection.json"
                        if not report_path.exists():
                            raise ValueError("Interrupted attempt has no complete collection")
                        report = json.loads(report_path.read_bytes())
                    else:
                        sim = CatalogSimulation(
                            catalog, plan, appearance=appearance, contact_solver=CONTACT_SOLVER,
                            stationarity_metric=STATIONARITY_METRIC,
                            velocity_tolerance=VELOCITY_TOLERANCE,
                            before_recording_write=disk.before_frame,
                        )
                        disk.wait(PAIR_WRITE_ALLOWANCE, "archive")
                        report = record_pair(sim, attempt)
                    if report["status"] != "complete":
                        raise ValueError("Independent command pair failed physical/recording acceptance")
                    disk.wait(qualification_write_allowance(attempt), "qualification")
                    qualification = qualify_pair(attempt)
                    entry.update(status="qualified", qualification=qualification,
                                 audits=ledger.accept_pair(report))
                    atomic_json(attempt / "qualification.json", {
                        "scope": "development curriculum; not train/sealed-test",
                        "storage_sha256": qualification["storage_sha256"],
                        "release_id": report["release_id"], **qualification,
                    })
                    disk.wait(sum((attempt / name).stat().st_size for name in (
                        "collection.json", "episodes.sqlite", "qualification.json",
                        "source/parts_catalog.json",
                    )), "publication")
                    entry["release"] = str(publish_pair(root, attempt, report))
                except (ValueError, RuntimeError, mujoco.FatalError) as error:
                    LOGGER.error("Curriculum attempt %s rejected: %s", sequence, error)
                    entry.update(status="rejected", reason=str(error))
                finally:
                    if sim is not None:
                        sim.close()
                entry["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                state["attempts"].append(entry)
                state["next_sequence"] = sequence + 1
                state["active_sequence"] = None
                atomic_json(state_path, state)
                atomic_json(root / "pending_audits.json", ledger.pending_audits())
                print(json.dumps(entry), flush=True)
        finally:
            ledger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--appearance", type=Path, required=True)
    parser.add_argument("--part-id", action="append", default=[])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--initial-sequence", type=int, default=0)
    args = parser.parse_args()
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    os.nice(19)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    logging.basicConfig(level=logging.INFO)
    run(args.root, args.catalog, args.appearance, args.part_id, args.seed, lambda: stopped,
        ledger_path=args.ledger, initial_sequence=args.initial_sequence)


if __name__ == "__main__":
    main()
