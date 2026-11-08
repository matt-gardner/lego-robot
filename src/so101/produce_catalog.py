"""Resumable CPU-idle development curriculum worker; never runs training."""
from __future__ import annotations

import argparse
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


def qualify_pair(root: Path) -> dict:
    """Validate every actual paired SFT sample before immutable publication."""
    report = json.loads((root / "collection.json").read_bytes())
    if report["status"] != "complete":
        raise ValueError("Cannot qualify incomplete pair")
    storage_path = root / "episodes.sqlite"
    if not storage_path.exists():
        consolidate(root, storage_path)
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
        part_ids: list[str], seed: int, stopping) -> None:
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
            state = {"recipe": identity, "next_sequence": 0, "attempts": []}
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
        ledger = CurriculumLedger(root / "ledger.sqlite", audit_seed=seed)
        try:
            while not stopping():
                if recipe(catalog_path, appearance_path, part_ids, seed) != identity:
                    raise ValueError("Pinned producer inputs changed during execution")
                sequence = state["next_sequence"]
                # Feedback-visible generation is development-only, never sealed evaluation.
                category = ("separated", "cluster", "overlap", "pile")[sequence % 4]
                task = ("grasp", "view", "push", "place")[(sequence // 4) % 4]
                plan = prepare_scene(catalog, seed=seed, sequence=sequence,
                                     category=category, task=task, object_count=2)
                family = hashlib.sha256(json.dumps(
                    ["new-catalog-development-v1", seed, sequence],
                    separators=(",", ":"),
                ).encode()).hexdigest()
                ledger.reserve_family(family, "development")
                plan.update(scene_family_id=family, split="development")
                attempt = root / f"attempt_{sequence:09d}"
                entry = {"sequence": sequence, "category": category, "task": task,
                         "source": str(attempt), "status": "interrupted"}
                sim = None
                try:
                    if attempt.exists():
                        report_path = attempt / "collection.json"
                        if not report_path.exists():
                            raise ValueError("Interrupted attempt has no complete collection")
                        report = json.loads(report_path.read_bytes())
                    else:
                        sim = CatalogSimulation(catalog, plan, appearance=appearance)
                        report = record_pair(sim, attempt)
                    if report["status"] != "complete":
                        raise ValueError("Independent command pair failed physical/recording acceptance")
                    qualification = qualify_pair(attempt)
                    entry.update(status="qualified", qualification=qualification,
                                 audits=ledger.accept_pair(report))
                    atomic_json(attempt / "qualification.json", {
                        "scope": "development curriculum; not train/sealed-test",
                        "storage_sha256": qualification["storage_sha256"],
                        "release_id": report["release_id"], **qualification,
                    })
                    entry["release"] = str(publish_pair(root, attempt, report))
                except (ValueError, RuntimeError, mujoco.FatalError) as error:
                    LOGGER.error("Curriculum attempt %s rejected: %s", sequence, error)
                    entry.update(status="rejected", reason=str(error))
                finally:
                    if sim is not None:
                        sim.close()
                entry["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                state["attempts"].append(entry)
                state["next_sequence"] += 1
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
    run(args.root, args.catalog, args.appearance, args.part_id, args.seed, lambda: stopped)


if __name__ == "__main__":
    main()
