"""Build capped, progressively larger photo-style robot task buckets."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import shutil
import threading
import time
from tempfile import TemporaryDirectory

from so101.cloud_checkpoint import atomic_json
from so101.photo_pile_generator import RejectedPile, generate_scene, pin_catalog, verify_scene
from so101.photo_pile_tasks import build_problem
from so101.pile_pool import publish_pool
from so101.pile_task_runtime import load_problem
from so101.pile_task_generator import resource_check
from so101.robot_sim import MODEL_ROOT


LOGGER = logging.getLogger(__name__)


def validate_stages(stages: tuple[int, ...], quota: int, maximum_attempts: int) -> None:
    if (not stages or stages[0] < 20 or stages[-1] > 90
            or any(a >= b for a, b in zip(stages, stages[1:]))):
        raise ValueError("Stages must strictly increase from at least 20, up to 90 pieces")
    if not 1 <= quota <= 10 or maximum_attempts < quota:
        raise ValueError("Require 1..10 scenes per stage and an adequate attempt budget")


def run(root: Path, source: Path, *, stages: tuple[int, ...] = (30, 40, 60, 90),
        quota: int = 10, maximum_attempts: int = 100, seed: int = 20261112,
        model_root: Path = MODEL_ROOT, stopping: threading.Event | None = None,
        sampling: str = "original", prior_root: Path | None = None,
        catalog_path: Path | None = None) -> None:
    validate_stages(stages, quota, maximum_attempts)
    stopping = stopping or threading.Event()
    root.mkdir(parents=True, exist_ok=True)
    if sampling == "catalog-diverse-v1":
        if catalog_path is None:
            raise ValueError("Diverse sampling requires a catalog")
        catalog_path = pin_catalog(catalog_path, root / "catalog")
    config = {"stages": list(stages), "quota": quota, "maximum_attempts": maximum_attempts,
              "seed": seed, "source": str(source.resolve()), "model_root": str(model_root.resolve()),
              "sampling": sampling,
              "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest() if catalog_path else None,
              "prior_root": str(prior_root.resolve()) if prior_root else None,
              "source_sha256": hashlib.sha256((source / "model.xml").read_bytes()).hexdigest(),
              "generator_sha256": hashlib.sha256(
                  Path(__file__).with_name("photo_pile_generator.py").read_bytes()).hexdigest(),
              "task_builder_sha256": hashlib.sha256(
                  Path(__file__).with_name("photo_pile_tasks.py").read_bytes()).hexdigest(),
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (root / ".curriculum.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / "curriculum.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("Curriculum configuration changed; use a new root")
        if not path.exists():
            atomic_json(path, config)

        next_check = 0.

        def check_stop():
            nonlocal next_check
            if stopping.is_set() or (root / "STOP").exists() or (root.parent / "STOP").exists():
                raise InterruptedError("Curriculum stopped; completed tasks preserved")
            if time.monotonic() >= next_check:
                while True:
                    try:
                        resource_check(root)
                        break
                    except BlockingIOError as error:
                        atomic_json(root / "status.json", {"status": "paused", "reason": str(error)})
                        if stopping.wait(5) or (root / "STOP").exists() or (root.parent / "STOP").exists():
                            raise InterruptedError("Curriculum stopped while resource-paused")
                next_check = time.monotonic() + 2.

        try:
            for count in stages:
                bucket = root / f"pieces-{count}"
                scenes, tasks = bucket / "piles", bucket / "tasks"
                scenes.mkdir(parents=True, exist_ok=True)
                tasks.mkdir(exist_ok=True)
                if prior_root is not None:
                    for admission in sorted((prior_root / f"pieces-{count}" / "tasks").glob(
                        "scene_*/admission.json"
                    )):
                        destination = tasks / f"scene_carried_{admission.parent.name}"
                        if destination.exists():
                            continue
                        entry = json.loads(admission.read_text())
                        problem = json.loads((admission.parent / "problem.json").read_text())
                        for task in problem["tasks"]:
                            load_problem(entry, task["task_id"])
                        with TemporaryDirectory(prefix=".carry-", dir=tasks) as temporary:
                            staging = Path(temporary) / "scene"
                            shutil.copytree(admission.parent, staging)
                            entry["root"] = str(destination.resolve())
                            atomic_json(staging / "admission.json", entry)
                            staging.rename(destination)
                for sequence in range(maximum_attempts):
                    check_stop()
                    admitted = list(tasks.glob("scene_*/admission.json"))
                    if len(admitted) >= quota:
                        break
                    attempt_path = bucket / f"attempt_{sequence:09d}.json"
                    if attempt_path.exists():
                        continue
                    while True:
                        check_stop()
                        try:
                            resource_check(root)
                            break
                        except BlockingIOError as error:
                            atomic_json(root / "status.json",
                                        {"status": "paused", "reason": str(error)})
                            stopping.wait(5)
                    candidate_seed = seed + count * 10000 + sequence + (1000 if prior_root else 0)
                    color_mode = ("single", "few", "many")[len(admitted) % 3]
                    attempt = {"piece_count": count, "sequence": sequence, "seed": candidate_seed,
                               "color_mode": color_mode}

                    def progress(value):
                        atomic_json(root / "status.json",
                                    {"status": "generating", **attempt, **value})

                    pile = scenes / f"scene_{sequence:09d}"
                    try:
                        if not pile.exists():
                            generate_scene(source, pile, count=count, seed=candidate_seed,
                                           check_stop=check_stop, progress=progress,
                                           sampling=sampling, color_mode=color_mode,
                                           catalog_path=catalog_path)
                        else:
                            receipt = verify_scene(pile)
                            expected = {"piece_count": count, "seed": candidate_seed,
                                        "sampling": sampling}
                            if catalog_path is not None:
                                expected["catalog_sha256"] = config["catalog_sha256"]
                            if any(receipt.get(key) != value for key, value in expected.items()):
                                raise ValueError("Existing pile differs from curriculum configuration")
                            attempt["color_mode"] = receipt["color_mode"]
                        check_stop()
                        entry = build_problem(pile, tasks / f"scene_{sequence:09d}", model_root)
                        attempt.update(status="complete", entry=entry)
                    except ValueError as error:
                        if not isinstance(error, RejectedPile) and not str(error).startswith((
                            "Scene contact penetration exceeded limit:",
                            "Catalog scene did not settle", "Non-finite catalog scene physics",
                            "MuJoCo emitted a warning",
                        )):
                            raise
                        LOGGER.warning("%s-piece proposal %s rejected: %s", count, sequence, error)
                        attempt.update(status="rejected", reason=str(error))
                    atomic_json(attempt_path, attempt)
                    atomic_json(root / "status.json", attempt)
                manifest = publish_pool(tasks)
                admitted_count = manifest["scenes"]
                atomic_json(bucket / "handoff.json", {
                    **manifest, "piece_count": count, "task_count": admitted_count * count * 4,
                    "quota": quota, "complete": admitted_count == quota,
                    "policy_difficulty_measured": False,
                })
                if admitted_count != quota:
                    atomic_json(root / "status.json", {
                        "status": "blocked", "piece_count": count,
                        "reason": "Attempt budget exhausted before task-scene quota",
                        "accepted": admitted_count, "quota": quota,
                    })
                    return
                print(json.dumps({"status": "stage_complete", "pieces": count, **manifest}), flush=True)
                check_stop()
            atomic_json(root / "status.json", {"status": "complete", **config})
        except InterruptedError as error:
            atomic_json(root / "status.json", {"status": "stopped", "reason": str(error)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--stages", type=int, nargs="+", default=[30, 40, 60, 90])
    parser.add_argument("--quota", type=int, default=10)
    parser.add_argument("--maximum-attempts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--sampling", choices=("original", "common-basic-v1", "catalog-diverse-v1"), default="original")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--prior-root", type=Path)
    args = parser.parse_args()
    os.nice(19)
    logging.basicConfig(level=logging.INFO)
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    run(args.root, args.source, stages=tuple(args.stages), quota=args.quota,
        maximum_attempts=args.maximum_attempts, model_root=args.model_root, stopping=stopping,
        sampling=args.sampling, prior_root=args.prior_root, catalog_path=args.catalog,
        seed=args.seed)


if __name__ == "__main__":
    main()
