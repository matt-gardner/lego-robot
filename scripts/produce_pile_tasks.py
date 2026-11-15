import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np

from so101.cloud_checkpoint import atomic_json
from so101.lego_curriculum import load_catalog
from so101.lego_parts_catalog import catalog_read_lock
from so101.photo_pile_generator import catalog_candidates
from so101.pile_task_generator import attempt_resource_limits, resource_check, scheduled_pile


def prepare(root, catalog):
    pinned = root/"catalog"
    frozen = root/"source"
    if not pinned.exists():
        with catalog_read_lock(catalog):
            raw = catalog.read_bytes()
            report = json.loads(raw)
            if report["converter_version"] != "surface-prisms-v5-zup":
                raise ValueError("Production requires the older prism representation")
            if report.get("selection_profile") == "catalog-diverse-v1-max32-per-family":
                chosen = report["parts"]
            else:
                candidates = [p for p in catalog_candidates(report)
                              if p["part_id"] not in {"3001", "3002", "2456"}]
                rng = np.random.default_rng(20261114183545)
                families = {}
                for part in candidates:
                    families.setdefault(part["family"], []).append(part)
                chosen = []
                for family in sorted(families):
                    entries = families[family]
                    chosen.extend(entries[int(i)] for i in rng.choice(
                        len(entries), min(32, len(entries)), replace=False))
            if len(chosen) < 64:
                raise ValueError(f"Insufficient eligible older catalog diversity: {len(chosen)}")
            pinned.mkdir()
            for part in chosen:
                for asset in part["assets"]:
                    relative = Path(asset["path"])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Catalog snapshot asset escapes root")
                    target = pinned/relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(catalog.parent/relative, target)
            atomic_json(pinned/"catalog.json", {
                **report, "parts": chosen,
                "parent_manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "selection_profile": "catalog-diverse-v1-max32-per-family",
            })
            load_catalog(pinned/"catalog.json")
    if not frozen.exists():
        shutil.copytree(Path("/mnt/dockyard/lego-robot/src"), frozen/"src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copyfile(Path(__file__), frozen/"controller.py")
        hashes = {str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in frozen.rglob("*") if p.is_file()}
        atomic_json(frozen/"files-sha256.json", hashes)
    return pinned/"catalog.json", frozen/"src"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--until", required=True)
    parser.add_argument("--maximum-attempts", type=int, default=100)
    parser.add_argument("--catalog", type=Path, default=Path(
        "/mnt/dockyard/lego-robot-data/data/lego_catalog/current/catalog.json"))
    parser.add_argument("--seed", type=int, default=20261114183545)
    parser.add_argument("--compact-stage-root", type=Path, action="append", default=[],
                        help="Earlier pool root whose admitted piles count toward the 5-per-size compact stage")
    args = parser.parse_args()
    preparation_started = time.time()
    args.root.mkdir(parents=True, exist_ok=True)
    catalog, source = prepare(args.root, args.catalog)
    piles = args.root/"piles"
    piles.mkdir(exist_ok=True)
    logs = Path("/tmp/lego-production-562")/args.root.name
    logs.mkdir(parents=True, exist_ok=True)
    deadline = datetime.datetime.fromisoformat(args.until).timestamp()
    previous = json.loads((args.root/"progress.json").read_bytes()) if (args.root/"progress.json").exists() else {}
    started = datetime.datetime.fromisoformat(previous["started_at"]).timestamp() if previous else preparation_started
    records = list(previous.get("attempts", []))
    env = {**os.environ, "PYTHONPATH":str(source), "MUJOCO_GL":"disable",
           "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1"}
    profile = {
        "generator":"native-pile-tasks-v1-airborne",
        "catalog":str(catalog), "source":str(source),
        "started_at":datetime.datetime.fromtimestamp(started, datetime.timezone.utc).isoformat(),
        "until":args.until,
        "controller_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "budget_profile":"size-scaled-8-cpu-seconds-per-piece-compact-18-full-catalog-v6",
        "preparation_CPU_seconds":previous.get("preparation_CPU_seconds", 0.)+sum(resource.getrusage(resource.RUSAGE_SELF)[:2]),
        "qualification":"TRAIN reset/task problems; no expert references or SFT qualification",
        "compact_stage_roots":[str(r) for r in args.compact_stage_root],
    }
    progress = {**previous, **profile, "status":"starting"}
    for _ in range(args.maximum_attempts):
        if time.time() >= deadline or (args.root/"STOP").exists() or (piles/"STOP").exists():
            break
        try:
            resource_check(piles, None)
        except BlockingIOError as error:
            progress.update(profile, status="paused", reason=str(error),
                            updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            atomic_json(args.root/"progress.json", progress)
            print(json.dumps(progress), flush=True)
            break
        sequence = 0
        while (piles/f"attempt_{sequence:09d}.json").exists():
            sequence += 1
        count, mode = scheduled_pile(sequence, airborne=True)
        sizes = [json.loads(a.read_bytes())["tags"]["piece_count"]
                 for root in (piles, *(r/"piles" for r in args.compact_stage_root))
                 for a in root.glob("scene_*/admission.json")]
        full_catalog = all(sizes.count(size) >= 5 for size in (48, 64, 80, 100))
        cpu_soft, cpu_hard, wall_limit = attempt_resource_limits(count, full_catalog)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        wall = time.monotonic()
        log = logs/f"attempt_{sequence:09d}.log"
        command = [
            sys.executable, "-c",
            f"import resource; resource.setrlimit(resource.RLIMIT_CPU,({cpu_soft},{cpu_hard})); "
            "from so101.pile_task_generator import main; main()",
            "--root", str(piles), "--catalog", str(catalog),
            "--seed", str(args.seed), "--attempts", "1", "--airborne",
        ] + (["--full-catalog"] if full_catalog else [])
        with log.open("w") as stream:
            try:
                child = subprocess.run(command, env=env, stdout=stream, stderr=stream,
                                       timeout=wall_limit, cwd="/mnt/dockyard/lego-robot")
                code = child.returncode
            except subprocess.TimeoutExpired:
                code = -999
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        record = {
            "sequence":sequence, "returncode":code,
            "piece_count":count, "color_mode":mode, "full_catalog":full_catalog,
            "cpu_soft_limit_seconds":cpu_soft, "cpu_hard_limit_seconds":cpu_hard,
            "wall_limit_seconds":wall_limit,
            "wall_seconds":time.monotonic()-wall,
            "CPU_seconds":after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        }
        receipt = piles/f"attempt_{sequence:09d}.json"
        if code != 0:
            # SIGKILL below the soft CPU limit is not RLIMIT_CPU; usually the cgroup OOM killer.
            exhausted = code == -24 or (code == -9 and record["CPU_seconds"] >= cpu_soft)
            record.update(status="CPU_budget_rejected" if exhausted else
                          "killed_externally" if code == -9 else "blocked",
                          reason=log.read_text()[-3000:])
            atomic_json(receipt, record)
        else:
            result = json.loads(receipt.read_bytes())
            record.update(status=result["status"], piece_count=result["piece_count"],
                          color_mode=result["color_mode"])
        records.append(record)
        admitted = list(piles.glob("scene_*/admission.json"))
        tasks = 0
        counts = []
        for admission in admitted:
            entry = json.loads(admission.read_bytes())
            count = entry["tags"]["piece_count"]
            counts.append(count)
            tasks += 4*count
        elapsed = time.time()-started
        progress = {
            **profile, "status":"running" if code==0 or code in (-24,-9) else "blocked",
            "admitted_piles":len(admitted), "piece_counts":counts, "task_count":tasks,
            "elapsed_wall_seconds":elapsed,
            "average_wall_seconds_per_admitted_pile":elapsed/len(admitted) if admitted else None,
            "attempts":records,
            "updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        atomic_json(args.root/"progress.json", progress)
        print(json.dumps({k:v for k,v in progress.items() if k!="attempts"}), flush=True)
        if code not in (0,-24,-9):
            raise RuntimeError(f"Pile producer blocked at attempt{sequence}; inspect {log}")
        if "did not settle" in record.get("reason", "") or "did not settle" in json.dumps(
                json.loads(receipt.read_bytes()).get("reason", "")):
            # Every unsettled pile is investigated before production continues.
            progress["status"] = "stopped_unsettled_pile"
            atomic_json(args.root/"progress.json", progress)
            print(f"Stopped: attempt {sequence} did not settle; investigate before resuming", flush=True)
            return
    if progress["status"] != "paused":
        progress["status"] = "bounded_run_complete"
    atomic_json(args.root/"progress.json", progress)


if __name__ == "__main__":
    main()
