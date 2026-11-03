"""Root-pinned SFT container launch; no shell, cloud writes, or original output mutation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import time

from so101.aws_worker_bootstrap import load_manifest, require_guard
from so101.cloud_checkpoint import atomic_json, fsync_directory, sha256
from so101.control_broker import canonical, trusted_json, trusted_path


def relative_output(argv: list[str]) -> str:
    if (not isinstance(argv, list) or not argv
            or any(not isinstance(item, str) or "\x00" in item for item in argv)):
        raise ValueError("SFT arguments must be a fixed string vector")
    def value(flag):
        if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
            raise ValueError(f"Exactly one {flag} is required")
        return argv[argv.index(flag) + 1]
    output = value("--output")
    path = PurePosixPath(output)
    if (path.is_absolute() or ".." in path.parts or not output.startswith("outputs/")
            or str(path) != output or "," in output):
        raise ValueError("Output must preserve a safe recorded relative outputs path")
    if value("--training-state") != output + "/training_state.pt" or value("--device") != "cuda":
        raise ValueError("SFT launch must use CUDA and full-state recovery in the fresh output")
    return output


def launch(manifest_path: Path, manifest_sha256: str, plan_path: Path, plan_sha256: str,
           staged: Path, attempts: Path, *, job_id: str, spec_sha256: str,
           run=subprocess.run, wall=time.time) -> str:
    if os.geteuid() != 0:
        raise PermissionError("Worker launch requires the trusted root entrypoint")
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("Invalid worker job ID")
    manifest = load_manifest(manifest_path, manifest_sha256)
    plan = trusted_json(plan_path)
    if hashlib.sha256(canonical(plan)).hexdigest() != plan_sha256:
        raise ValueError("Launch plan differs from its root-pinned digest")
    if set(plan) != {"schema", "spec_sha256", "argv", "readonly_mounts"} or plan["schema"] != 1:
        raise ValueError("Unsupported launch plan")
    if (plan["spec_sha256"] != spec_sha256 or spec_sha256 != manifest["spec_sha256"]
            or plan["argv"] != manifest["spec"]["recipe"].get("argv")):
        raise ValueError("Launch arguments must match the exact approved recipe/specification")
    output = relative_output(plan["argv"])
    staged = trusted_path(staged)
    binding = trusted_json(staged / ".bootstrap.json")
    if (binding.get("phase") != "staged-idle" or binding.get("job_id") != job_id
            or binding.get("spec_sha256") != spec_sha256
            or binding.get("manifest_sha256") != manifest_sha256
            or sha256(staged / ".staged.json") != binding.get("staging_receipt_sha256")):
        raise ValueError("Launch requires the exact committed idle staging receipt")
    deadline = binding.get("deadline")
    if type(deadline) not in (int, float) or not wall() < deadline:
        raise ValueError("Worker lease is expired or invalid")
    require_guard(run)
    checkpoint = trusted_path(staged / "training_state.pt")
    if sha256(checkpoint) != manifest["spec"]["checkpoint"]["sha256"]:
        raise ValueError("Staged full-state checkpoint changed before launch")
    mounts = plan["readonly_mounts"]
    if not isinstance(mounts, dict) or not mounts:
        raise ValueError("Explicit staged read-only data/model mounts are required")
    args = [
        manifest["runtime"]["docker_cli"], "--host=unix:///var/run/docker.sock",
        "run", "--detach", "--name", "so101-" + job_id, "--restart=no",
        "--label", "so101-spec=" + spec_sha256, "--read-only", "--network=none",
        "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=1001:1001",
        "--gpus=device=0", "--shm-size=1g", "--tmpfs=/tmp:rw,nosuid,nodev,size=1g",
    ]
    for destination, source in mounts.items():
        if (not isinstance(destination, str)
                or not destination.startswith(("/opt/so101/data/", "/opt/so101/outputs/",
                                               "/work/cache/"))
                or ".." in PurePosixPath(destination).parts or "," in destination
                or destination == "/opt/so101/" + output
                or destination.startswith("/opt/so101/" + output + "/")
                or ("/opt/so101/" + output).startswith(destination.rstrip("/") + "/")):
            raise ValueError("Read-only mount must not mask source code or writable output")
        if not isinstance(source, str):
            raise ValueError("Input mount source must name a staged path")
        path = trusted_path(staged / source)
        if not path.is_relative_to(staged) or "," in str(path):
            raise ValueError("Input mount must stay inside root-trusted staged artifacts")
        args.extend(["--mount", f"type=bind,src={path},dst={destination},readonly"])
    attempt = trusted_path(attempts) / job_id
    attempt.mkdir(mode=0o700)
    # Intent persists before any container call. A repeated or ambiguous start cannot replay.
    atomic_json(attempt / "launch-intent.json", {
        "job_id": job_id, "spec_sha256": spec_sha256, "plan_sha256": plan_sha256,
    })
    fresh = attempt / "output"
    fresh.mkdir(mode=0o700)
    shutil.copyfile(checkpoint, fresh / "training_state.pt")
    with (fresh / "training_state.pt").open("rb") as stream:
        os.fsync(stream.fileno())
    os.chown(fresh / "training_state.pt", 1001, 1001)
    os.chown(fresh, 1001, 1001)
    args.extend(["--mount", f"type=bind,src={fresh},dst=/opt/so101/{output}",
                 manifest["runtime"]["image"], "python", "-u", "-m",
                 "so101.train_entity_sim", *plan["argv"]])
    if not wall() < deadline:
        raise ValueError("Lease expired while preparing launch")
    result = run(args, check=True, capture_output=True, text=True, timeout=60)
    container_id = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{64}", container_id):
        raise ValueError("Docker returned an invalid container identity; reconciliation required")
    atomic_json(attempt / "launched.json", {
        "job_id": job_id, "spec_sha256": spec_sha256, "container_id": container_id,
    })
    return container_id


def checkpoint_stop(attempts: Path, *, job_id: str, spec_sha256: str,
                    docker_cli: str, grace_seconds: int, run=subprocess.run) -> None:
    if os.geteuid() != 0:
        raise PermissionError("Checkpoint stop requires the trusted root entrypoint")
    if (not re.fullmatch(r"[0-9a-f]{32}", job_id)
            or not re.fullmatch(r"[0-9a-f]{64}", spec_sha256)
            or type(grace_seconds) is not int or not 1 <= grace_seconds <= 300):
        raise ValueError("Checkpoint stop requires exact identity and bounded grace")
    receipt = trusted_json(trusted_path(attempts) / job_id / "launched.json")
    container = receipt.get("container_id")
    if (receipt.get("job_id") != job_id or receipt.get("spec_sha256") != spec_sha256
            or not isinstance(container, str) or not re.fullmatch(r"[0-9a-f]{64}", container)):
        raise ValueError("Checkpoint stop differs from committed launch identity")
    cli = trusted_path(Path(docker_cli))
    output = attempts / job_id / "output"
    if output.is_symlink() or not output.is_dir() or output.stat().st_uid != 1001:
        raise ValueError("Checkpoint stop must target the new worker-owned output")
    stop = output / "STOP"
    try:
        with stop.open("xb") as stream:
            stream.write(b"checkpoint-stop\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if not stat.S_ISREG(stop.lstat().st_mode):
            raise ValueError("Worker STOP must be a regular file")
    fsync_directory(output)
    result = run([str(cli), "--host=unix:///var/run/docker.sock", "wait", container],
                 check=True, capture_output=True, text=True, timeout=grace_seconds)
    if result.stdout.strip() != "0":
        raise ValueError("Trainer did not finish a clean checkpoint stop; recovery required")
    # A clean exit alone is not publication or full-state recovery verification.
    atomic_json(attempts / job_id / "checkpoint-stop.json", {
        "job_id": job_id, "spec_sha256": spec_sha256, "phase": "exited-awaiting-validation",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "checkpoint-stop"))
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--spec-sha256", required=True)
    args = parser.parse_args()
    config = trusted_json(Path("/etc/so101-worker/execution.json"))
    fields = {"manifest", "manifest_sha256", "plan", "plan_sha256", "staging_root",
              "attempts_root", "docker_cli", "grace_seconds", "qualification"}
    if set(config) != fields:
        raise ValueError("Unsupported root worker execution configuration")
    attempts = Path(config["attempts_root"])
    if args.action == "start":
        qualification = trusted_json(Path(config["qualification"]))
        if (qualification.get("status") != "PASS"
                or qualification.get("spec_sha256") != args.spec_sha256):
            raise PermissionError("Exact specification has no root-installed qualification PASS")
        print(launch(
            Path(config["manifest"]), config["manifest_sha256"],
            Path(config["plan"]), config["plan_sha256"],
            Path(config["staging_root"]) / args.job_id, attempts,
            job_id=args.job_id, spec_sha256=args.spec_sha256,
        ))
    else:
        checkpoint_stop(
            attempts, job_id=args.job_id, spec_sha256=args.spec_sha256,
            docker_cli=config["docker_cli"], grace_seconds=config["grace_seconds"],
        )


if __name__ == "__main__":
    main()
