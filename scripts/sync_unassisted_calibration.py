"""Continuously mirror and additively publish policy screening measurements."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading


def refresh(args, published):
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
           "-o", f"UserKnownHostsFile={args.known_hosts}", "-i", str(args.key)]
    with (args.local.parent / "calibration-publication.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        subprocess.run(["rsync", "-rtcz", "-e", shlex.join(ssh),
                        f"{args.host}:{args.remote}/", str(args.local) + "/"], check=True)
        pools = {}
        for kind in ("photo", "native"):
            for pool in getattr(args, kind + "_pool"):
                pools[hashlib.sha256(pool.read_bytes()).hexdigest()] = (kind, pool)
        for path in sorted(args.local.glob("results-*.json")):
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if published.get(path.name) == digest:
                continue
            result = json.loads(content)
            pool_sha = result["input_sha256"]["native_pool"]
            kind, pool = pools[pool_sha]
            if (result["optimizer"] is not None
                    or any(row["assistance"] != 0 for row in result["groups"])):
                raise ValueError("Unassisted mirror received training or assisted measurements")
            _, pieces, suffix = path.stem.split("-", 2)
            series_suffix = suffix if kind == "photo" else f"{pieces}-{suffix}"
            distance_start = any(
                row.get("task_assistance", {}).get("kind") == "direct-task-distance-v2"
                for row in result.get("plan", [])
            )
            prefix = "distance" if distance_start else "unassisted"
            label = "Direct distance-start pilot" if distance_start else "Zero assistance"
            command = [
                sys.executable, "-m", "so101.publish_photo_calibration",
                "--pool", str(pool), "--source-kind", kind, "--measurements", str(path),
                "--output", str(args.output), "--publication-id", f"{kind}-{pool_sha}",
                "--measurement-series-id", f"{prefix}-{args.job_id}-{series_suffix}",
                "--measurement-series-label", f"{label}, {pieces} pieces ({suffix})",
            ]
            subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True,
                           env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
            published[path.name] = digest
        status = json.loads((args.local / "status.json").read_text())
        print(json.dumps({"event": "dashboard_refresh", "phase": status["phase"],
                          "published_result_files": len(published)}), flush=True)
        if status["phase"] == "failed":
            raise RuntimeError(status["error"])
        return status["phase"] == "policy-rollouts-complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("local", "output", "key", "known-hosts"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("host", "remote", "job-id"):
        parser.add_argument("--" + name, required=True)
    for kind in ("photo", "native"):
        parser.add_argument("--" + kind + "-pool", type=Path, action="append", default=[])
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.local.mkdir(parents=True, exist_ok=True)
    published = {}
    while True:
        complete = refresh(args, published)
        if complete or args.once:
            break
        threading.Event().wait(20)


if __name__ == "__main__":
    main()
