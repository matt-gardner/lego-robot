#!/usr/bin/env python3
"""Mirror direct GPU logs into the dashboard's existing remote-run format."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import sched
import shlex
import subprocess
import sys
import tempfile
import time


REMOTE = """
import json, pathlib, subprocess, time
root = pathlib.Path('/home/ubuntu/lego-sweep/output')
running = subprocess.run(
    ['docker', 'ps', '--format', '{{.Names}}'], check=True,
    capture_output=True, text=True).stdout.splitlines()
logs = {}
for path in sorted(root.glob('*/sft_training.json')):
    logs[path.parent.name] = json.loads(path.read_text())
exports = [path.name for path in root.glob('*-export.json')]
print(json.dumps({'logs': logs, 'running': running, 'exports': exports,
                  'observed_at': time.time()}))
"""

ARMS = {
    "entity_smolvla_sft_b128_lr3e-06": "b128-lr3e6",
    "entity_smolvla_sft_b64_lr3e-06": "b64-lr3e6",
    "entity_smolvla_sft_b128_lr3e-05": "b128-lr3e5",
    "entity_smolvla_sft_b64_lr3e-05": "b64-lr3e5",
}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".sync-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream)
            stream.write("\n")
            os.fchmod(stream.fileno(), 0o644)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sync(snapshot, output, job_id):
    observed = snapshot["observed_at"]
    if not isinstance(observed, (int, float)) or abs(time.time() - observed) > 60:
        raise ValueError("Remote observation is not current")
    result = []
    for name, log in snapshot["logs"].items():
        match = re.fullmatch(r"entity_smolvla_sft_b([1-9][0-9]*)_lr([0-9]+e-[0-9]+)", name)
        if match is None:
            raise ValueError(f"Unrecognized GPU trial directory: {name}")
        canonical_arm = f"b{match[1]}-lr{match[2]}"
        arm = ARMS.get(name, canonical_arm)
        active = any(f"lego-sweep-{job_id}-{candidate}" in snapshot["running"]
                     for candidate in (arm, canonical_arm))
        state = "running" if active else (
            log["status"] if log["status"] in ("completed", "stopped") else "failed")
        phase = log["status"] if active and log["status"] in ("training", "evaluating") else None
        destination = output / name
        atomic_json(destination / "sft_training.json", log)
        atomic_json(destination / "remote_status.json", {
            "state": state, "observed_at": observed, "heartbeat": observed,
            "phase": phase, "phase_observed_at": observed if phase is not None else None,
            "job_id": job_id, "instance_id": "i-02ba0e9a61e1c68ef",
            "error": None if active or state in ("completed", "stopped") else
            "Trainer container is not running; inspect sweep runner before retrying.",
        })
        result.append({"arm": arm, "state": state, "updates": len(log["steps"])})
    if not result:
        raise ValueError("No GPU training logs found; dashboard was not updated")
    return result


def refresh(args):
    response = subprocess.run([
        "ssh", "-i", "/home/claude/.ssh/lego_gpu_ed25519",
        "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "UserKnownHostsFile=/home/claude/.ssh/lego_gpu_known_hosts",
        "-o", "ConnectTimeout=10", "ubuntu@172.31.50.157",
        "sudo -n python3 -c " + shlex.quote(REMOTE),
    ], check=True, capture_output=True, text=True, timeout=30)
    snapshot = json.loads(response.stdout)
    result = sync(snapshot, args.output, args.job_id)
    print(json.dumps(result), flush=True)
    return (
        set(snapshot["logs"]) == set(ARMS)
        and all(log["status"] == "completed" for log in snapshot["logs"].values())
        and {f"{arm}-export.json" for arm in ARMS.values()}.issubset(snapshot["exports"])
    )


def watch(args):
    scheduler = sched.scheduler()

    def tick():
        try:
            complete = refresh(args)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            print(f"Dashboard sync failed: {error}", file=sys.stderr, flush=True)
            atomic_json(args.output / "sync_status.json", {
                "state": "error", "observed_at": time.time(), "error": str(error),
            })
        else:
            atomic_json(args.output / "sync_status.json", {
                "state": "completed" if complete else "running",
                "observed_at": time.time(), "error": None,
            })
            if complete:
                return
        scheduler.enter(20, 1, tick)

    scheduler.enter(0, 1, tick)
    scheduler.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--watch", action="store_true",
                        help="Automatically sync every 20 seconds until all four exports complete")
    args = parser.parse_args()
    if args.watch:
        with (args.output / "dashboard_sync.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            watch(args)
    else:
        refresh(args)


if __name__ == "__main__":
    main()
