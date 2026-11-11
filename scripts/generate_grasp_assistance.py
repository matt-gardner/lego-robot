"""Freeze all admitted original GRASP tasks and construct resumable parent families."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from so101.cloud_checkpoint import atomic_json


LOGGER = logging.getLogger(__name__)


def read_hashed(path: Path, expected: str | None = None) -> tuple[dict, str]:
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"Source hash changed: {path}")
    return json.loads(content), digest


def inventory(pointer: Path, extra_roots: list[Path]) -> dict:
    selected, pointer_sha = read_hashed(pointer)
    entries, sources = {}, []
    for pool in selected["pools"]:
        path = Path(pool["manifest_path"])
        manifest, digest = read_hashed(path, pool["sha256"])
        sources.append({"path": str(path), "sha256": digest, "kind": "selected-pool"})
        for entry in manifest["scenes"]:
            previous = entries.setdefault(entry["root"], entry)
            if previous != entry:
                raise ValueError(f"Conflicting source entry: {entry['root']}")
    for root in extra_roots:
        for path in sorted(root.rglob("admission.json")):
            entry, digest = read_hashed(path)
            sources.append({"path": str(path), "sha256": digest, "kind": "additional-admission"})
            previous = entries.setdefault(entry["root"], entry)
            if previous != entry:
                raise ValueError(f"Conflicting source entry: {entry['root']}")
    jobs = {}
    for entry in entries.values():
        root = Path(entry["root"])
        if not root.is_absolute() or root.is_symlink():
            raise ValueError(f"Invalid original source root: {root}")
        scene, _ = read_hashed(root / "problem.json", entry["files_sha256"]["problem.json"])
        if scene["scene_id"] != entry["scene_id"]:
            raise ValueError(f"Scene identity mismatch: {root}")
        for task in scene["tasks"]:
            if task["task_type"] != "grasp" or task.get("parent_task_id"):
                continue
            if task.get("assistance"):
                raise ValueError(f"Assisted task masquerades as an original: {task['task_id']}")
            job_id = hashlib.sha256(task["task_id"].encode()).hexdigest()
            job = {"job_id": job_id, "task_id": task["task_id"],
                   "entry": entry, "physics_version": scene["physics_version"]}
            previous = jobs.setdefault(job_id, job)
            if previous != job:
                raise ValueError(f"Conflicting original GRASP task: {task['task_id']}")
    if not jobs:
        raise ValueError("No original GRASP tasks found")
    return {
        "schema_version": 1, "kind": "all-original-grasp-assistance-inventory",
        "pointer": {"path": str(pointer), "sha256": pointer_sha},
        "sources": sources, "scene_count": len(entries),
        "task_count": len(jobs),
        "split_counts": dict(Counter(job["entry"]["split"] for job in jobs.values())),
        "physics_counts": dict(Counter(job["physics_version"] for job in jobs.values())),
        "jobs": sorted(jobs.values(), key=lambda job: job["task_id"]),
    }


def summarize(output: Path, plan: dict, status: str) -> dict:
    counts = Counter()
    children = 0
    for job in plan["jobs"]:
        receipt = output / "receipts" / f"{job['job_id']}.json"
        if not receipt.exists():
            continue
        result = json.loads(receipt.read_text())
        counts[result["status"]] += 1
        children += result.get("child_count", 0)
    return {
        "schema_version": 1, "status": status, "total_parents": plan["task_count"],
        "finished_parents": sum(counts.values()), "parent_outcomes": dict(counts),
        "published_children": children, "scene_count": plan["scene_count"],
        "split_counts": plan["split_counts"], "updated_at_unix": time.time(),
    }


def run_job(output: Path, job: dict, timeout: float) -> dict:
    job_id = job["job_id"]
    receipt = output / "receipts" / f"{job_id}.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    family = output / "families" / job_id
    admission = family / "admission.json"
    if admission.exists():
        entry = json.loads(admission.read_text())
        scene, _ = read_hashed(family / "problem.json", entry["files_sha256"]["problem.json"])
        if ({task["parent_task_id"] for task in scene["tasks"]} != {job["task_id"]}
                or scene["provenance"]["parent_entry"] != job["entry"]
                or sorted(task["assistance"]["value"] for task in scene["tasks"])
                != [i / 20 for i in range(20)]):
            raise ValueError(f"Existing output does not match frozen job: {family}")
        result = {"status": "complete", "task_id": job["task_id"],
                  "admission_path": str(admission), "child_count": len(scene["tasks"])}
        atomic_json(receipt, result)
        return result
    entry_path = output / "entries" / f"{job_id}.json"
    atomic_json(entry_path, job["entry"])
    command = [
        sys.executable, "-m", "so101.pile_grasp_assistance",
        "--admission", str(entry_path), "--output", str(family),
        "--task-id", job["task_id"],
    ]
    log_path = output / "logs" / f"{job_id}.log"
    started = time.time()
    with log_path.open("wb") as stream:
        try:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            result = {"status": "timeout", "task_id": job["task_id"],
                      "reason": f"Exceeded {timeout}s; not a claim of task infeasibility"}
        else:
            if process.returncode == 0:
                if not admission.exists():
                    raise RuntimeError(f"Constructor returned success without admission: {log_path}")
                entry = json.loads(admission.read_text())
                scene, _ = read_hashed(family / "problem.json", entry["files_sha256"]["problem.json"])
                result = {"status": "complete", "task_id": job["task_id"],
                          "admission_path": str(admission), "child_count": len(scene["tasks"])}
            else:
                stream.flush()
                log = log_path.read_text(errors="replace")
                unavailable = "UnavailableGrasp: No original-pose GRASP families constructed" in log
                result = {"status": "unavailable" if unavailable else "error",
                          "task_id": job["task_id"], "returncode": process.returncode,
                          "reason": log[-6000:]}
    result.update(log_path=str(log_path), duration_seconds=time.time() - started)
    atomic_json(receipt, result)
    return result


def run(output: Path, *, workers: int, timeout: float) -> dict:
    if workers < 1 or timeout <= 0:
        raise ValueError("Workers and per-parent timeout must be positive")
    plan = json.loads((output / "inventory.json").read_text())
    import mujoco  # The inventory-only command does not need the simulator.
    if set(plan["physics_counts"]) != {mujoco.__version__}:
        raise ValueError(f"Inventory requires different physics versions: {plan['physics_counts']}")
    for name in ("receipts", "logs", "entries", "families"):
        (output / name).mkdir(exist_ok=True)
    atomic_json(output / "status.json", summarize(output, plan, "running"))
    remaining = iter(job for job in plan["jobs"]
                     if not (output / "receipts" / f"{job['job_id']}.json").exists())
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = set()
        while True:
            if (output / "STOP").exists():
                for future in pending:
                    future.result()
                result = summarize(output, plan, "stopped")
                atomic_json(output / "status.json", result)
                return result
            while len(pending) < workers:
                job = next(remaining, None)
                if job is None:
                    break
                pending.add(executor.submit(run_job, output, job, timeout))
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                LOGGER.info("%s: %s", result["task_id"], result["status"])
            atomic_json(output / "status.json", summarize(output, plan, "running"))
    result = summarize(output, plan, "complete")
    if result["parent_outcomes"].get("error") or result["parent_outcomes"].get("timeout"):
        result["status"] = "complete-with-errors"
    atomic_json(output / "status.json", result)
    admissions = [json.loads((output / "receipts" / f"{job['job_id']}.json").read_text())
                  for job in plan["jobs"]]
    atomic_json(output / "admissions.json", {
        "kind": "original-pose-grasp-admissions", "status": result["status"],
        "inventory_path": str(output / "inventory.json"),
        "admissions": [item["admission_path"] for item in admissions if item["status"] == "complete"],
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pointer", type=Path)
    parser.add_argument("--extra-root", action="append", type=Path, default=[])
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    os.nice(19)
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "inventory.json"
    if not plan_path.exists():
        if args.pointer is None:
            parser.error("--pointer is required for a new frozen inventory")
        atomic_json(plan_path, inventory(args.pointer, args.extra_root))
    elif args.pointer is not None or args.extra_root:
        parser.error("Inventory already exists; resume without source-selection arguments")
    if args.inventory_only:
        plan = json.loads(plan_path.read_text())
        print(json.dumps({key: value for key, value in plan.items() if key != "jobs"}))
        return
    with (args.output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = run(args.output, workers=args.workers, timeout=args.timeout)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            plan = json.loads(plan_path.read_text())
            atomic_json(args.output / "status.json",
                        {**summarize(args.output, plan, "failed"), "error": str(error)})
            raise
    print(json.dumps(result), flush=True)
    if result["status"] == "complete-with-errors":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
