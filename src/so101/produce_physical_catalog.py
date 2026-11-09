"""Generate clean physical pairs and hand immutable trajectories to rendering."""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import threading
from time import perf_counter

import mujoco

from so101.catalog_sim import CatalogSimulation
from so101.cloud_checkpoint import atomic_json
from so101.collect_catalog_sim import record_pair
from so101.curriculum_ledger import CurriculumLedger
from so101.curriculum_workers import memory_state
from so101.lego_curriculum import load_catalog, prepare_scene
from so101.produce_catalog import (
    CONTACT_SOLVER, DiskBackpressure, PAIR_WRITE_ALLOWANCE, STATIONARITY_METRIC,
    VELOCITY_TOLERANCE, scheduled_scene, sha256,
)


LOGGER = logging.getLogger(__name__)
PENDING_QUEUE_LIMIT = 4 * 1024**3


def pending_queue_bytes(root: Path) -> int:
    total = 0
    for path in (root / "render_queue" / "ready").glob("*.json"):
        manifest = json.loads(path.read_bytes())
        acknowledgment = root / "render_queue" / "acknowledged" / path.name
        if acknowledgment.exists():
            receipt = json.loads(acknowledgment.read_bytes())
            if (receipt["release_id"] != manifest["release_id"]
                    or receipt["status"] not in ("published", "image_rejected")):
                raise ValueError("Invalid downstream acknowledgment")
            continue
        source = Path(manifest["pair_root"])
        total += sum((source / name).stat().st_size for name in manifest["files_sha256"])
    return total


def handoff_pair(attempt: Path, queue: Path, timings: dict) -> Path:
    report = json.loads((attempt / "collection.json").read_bytes())
    if report["status"] != "physically_validated" or len(report["successful_episodes"]) != 2:
        raise ValueError("Only independently validated physical pairs may enter the render queue")
    if any(row["outcome"] != "success" for row in report["attempts"]):
        raise ValueError("Render queue cannot contain a failed command prefix")
    files = {}
    for path in sorted(attempt.rglob("*")):
        if path.is_symlink():
            raise ValueError("Trajectory handoff cannot contain symlinks")
        if path.is_file():
            files[path.relative_to(attempt).as_posix()] = sha256(path)
    manifest = {
        "schema_version": 1, "kind": "lego-physical-pair",
        "status": "physically_validated", "images_pending": True, "sft_qualified": False,
        "pair_root": str(attempt.resolve()), "release_id": report["release_id"],
        "example_ids": [row["example_id"] for row in report["attempts"]],
        "scene_family_id": report["scene_family_id"], "split": report["split"],
        "collection": "collection.json", "files_sha256": files,
        "timings_seconds": {key: value for key, value in timings.items()
                            if key != "handoff_packaging"},
        "rendering_contract": "Restore each pre-action qpos; mj_forward only, never replay actions",
    }
    queue.mkdir(parents=True, exist_ok=True)
    target = queue / f"{report['release_id']}.json"
    if target.exists():
        if json.loads(target.read_bytes()) != manifest:
            raise ValueError("An immutable physical handoff identity changed")
        return target
    for path in attempt.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    atomic_json(target, manifest)
    target.chmod(0o444)
    return target


def run(root: Path, catalog_path: Path, part_ids: list[str], seed: int,
        initial_sequence: int, stopping: threading.Event, attempts: int = 0) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".physical-producer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        catalog = load_catalog(catalog_path)
        if part_ids:
            selected = set(part_ids)
            if selected - {part.part_id for part in catalog.parts}:
                raise ValueError("Requested physical parts are not verified")
            catalog = replace(catalog, parts=tuple(p for p in catalog.parts if p.part_id in selected))
        disk = DiskBackpressure(root)
        ledger = CurriculumLedger(root / "ledger.sqlite", audit_seed=seed)
        completed = 0
        try:
            for collection in sorted(root.glob("attempt_*/collection.json")):
                report = json.loads(collection.read_bytes())
                if report["status"] == "physically_validated":
                    sequence = int(collection.parent.name.removeprefix("attempt_"))
                    timing_path = root / f"timing_{sequence:09d}.json"
                    if not timing_path.exists():
                        raise ValueError("Validated interrupted pair has no stage timing journal")
                    handoff_pair(collection.parent, root / "render_queue" / "ready",
                                 json.loads(timing_path.read_bytes()))
            while not stopping.is_set() and (not attempts or completed < attempts):
                memory = memory_state()
                pending = pending_queue_bytes(root)
                if (pending + PAIR_WRITE_ALLOWANCE > PENDING_QUEUE_LIMIT
                        or memory["pressure"] or memory["available_bytes"] < PAIR_WRITE_ALLOWANCE):
                    atomic_json(root / "resource_status.json", {
                        "status": "paused", "pending_queue_bytes": pending, "memory": memory,
                    })
                    stopping.wait(1)
                    continue
                if not disk.wait(PAIR_WRITE_ALLOWANCE, "physical-admission", stopping.is_set):
                    break
                atomic_json(root / "resource_status.json", {"status": "ready"})
                sequence = ledger.claim_sequence(minimum=initial_sequence)
                timing = {}
                started = perf_counter()
                category, task, count = scheduled_scene(sequence)
                plan = prepare_scene(catalog, seed=seed, sequence=sequence,
                                     category=category, task=task, object_count=count)
                timing["scene_proposal"] = perf_counter() - started
                timing_path = root / f"timing_{sequence:09d}.json"
                atomic_json(timing_path, timing)
                family = hashlib.sha256(json.dumps(
                    ["new-catalog-development-v1", seed, sequence], separators=(",", ":"),
                ).encode()).hexdigest()
                ledger.reserve_family(family, "development")
                plan.update(scene_family_id=family, split="development")
                attempt = root / f"attempt_{sequence:09d}"
                sim = None
                stage = "scene_construction_and_settling"
                try:
                    started = perf_counter()
                    sim = CatalogSimulation(
                        catalog, plan, render=False, record_images=False,
                        contact_solver=CONTACT_SOLVER, stationarity_metric=STATIONARITY_METRIC,
                        velocity_tolerance=VELOCITY_TOLERANCE,
                        before_recording_write=disk.before_frame,
                    )
                    timing["scene_construction_and_settling"] = perf_counter() - started
                    atomic_json(timing_path, timing)
                    stage = "paired_execution_and_physical_validation"
                    started = perf_counter()
                    report = record_pair(sim, attempt, physical_only=True)
                    timing["paired_execution_and_physical_validation"] = perf_counter() - started
                    atomic_json(timing_path, timing)
                    if report["status"] != "physically_validated":
                        raise ValueError("Physical pair rejected; no render handoff")
                    stage = "handoff_packaging"
                    started = perf_counter()
                    target = handoff_pair(attempt, root / "render_queue" / "ready", timing)
                    timing["handoff_packaging"] = perf_counter() - started
                    atomic_json(timing_path, timing)
                    print(json.dumps({"sequence": sequence, "status": "physically_validated",
                                      "handoff": str(target), "timings_seconds": timing}), flush=True)
                except (ValueError, RuntimeError, mujoco.FatalError) as error:
                    timing[f"{stage}_failed_elapsed"] = perf_counter() - started
                    LOGGER.error("Physical attempt %s failed during %s: %s", sequence, stage, error)
                    atomic_json(root / f"rejection_{sequence:09d}.json",
                                {"sequence": sequence, "reason": str(error), "stage": stage,
                                 "status": "handoff_failed" if stage == "handoff_packaging"
                                 else "physical_rejected", "timings_seconds": timing})
                finally:
                    if sim is not None:
                        sim.close()
                completed += 1
        finally:
            ledger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--part-id", action="append", default=[])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initial-sequence", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=0)
    args = parser.parse_args()
    if args.attempts < 0:
        parser.error("--attempts must be nonnegative")
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    os.nice(19)
    logging.basicConfig(level=logging.INFO)
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    run(args.root, args.catalog, args.part_id, args.seed, args.initial_sequence,
        stopping, args.attempts)


if __name__ == "__main__":
    main()
