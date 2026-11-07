"""Fresh, identity-bound GPU host installation; never starts training."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import venv

from so101.aws_artifacts import worker_request
from so101.aws_worker_bootstrap import load_manifest, require_guard, stage_idle
from so101.aws_worker_guard import Metadata
from so101.aws_worker_prepare import prepare
from so101.cloud_checkpoint import atomic_json
from so101.control_broker import canonical, trusted_json, trusted_path


WORKER = Path("/opt/so101-worker")
GUARD = Path("/opt/lego-robot-worker")
STATE = Path("/var/lib/so101-worker")
CONFIG = Path("/etc/so101-worker")
UNITS = Path("/etc/systemd/system")
EXECUTION_FIELDS = {"manifest", "manifest_sha256", "plan", "plan_sha256", "staging_root",
                    "attempts_root", "docker_cli", "grace_seconds", "qualification"}


def identity(manifest: dict, metadata, *, wall=time.time) -> tuple[str, float]:
    token = metadata.token()
    actual = json.loads(metadata.read("dynamic/instance-identity/document", token))
    expected = {"accountId": manifest["account_id"], "region": "us-west-2",
                "instanceType": manifest["instance_type"], "imageId": manifest["ami"]}
    if not isinstance(actual, dict) or any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError("Installer is not on the exact approved GPU instance")
    tags = {name: metadata.read("meta-data/tags/instance/so101-" + name, token)
            for name in ("job", "spec", "broker")}
    if (not re.fullmatch(r"[0-9a-f]{32}", tags["job"])
            or tags["spec"] != manifest["spec_sha256"]
            or tags["broker"] != manifest["broker_id"]):
        raise ValueError("Installer job/spec/broker metadata mismatch")
    deadline = metadata.deadline()
    if not math.isfinite(deadline) or deadline <= wall():
        raise ValueError("Installer lease is expired or invalid")
    return tags["job"], deadline


def pull_image(manifest: dict, auth_root: Path, *, run=subprocess.run) -> None:
    registry = manifest["account_id"] + ".dkr.ecr.us-west-2.amazonaws.com"
    image = manifest["runtime"]["image"]
    if not re.fullmatch(re.escape(registry) + r"/lego-robot-runtime@sha256:[0-9a-f]{64}", image):
        raise ValueError("Worker pulls only its exact private Oregon project ECR digest")
    response = worker_request([
        manifest["aws_cli"], "ecr", "get-authorization-token",
        "--registry-ids", manifest["account_id"], "--region", "us-west-2",
        "--output", "json", "--no-cli-pager",
    ], run, timeout=30)
    records = response.get("authorizationData")
    if (not isinstance(records, list) or len(records) != 1
            or not isinstance(records[0], dict)
            or records[0].get("proxyEndpoint") != "https://" + registry):
        raise ValueError("ECR returned a different registry authorization")
    decoded = base64.b64decode(records[0]["authorizationToken"], validate=True).decode()
    username, separator, password = decoded.partition(":")
    if username != "AWS" or separator != ":" or not password or any(
        char.isspace() for char in password
    ):
        raise ValueError("Invalid ECR worker authorization")
    cli = manifest["runtime"]["docker_cli"]
    # Registry credentials exist only in this fresh private temporary directory.
    with tempfile.TemporaryDirectory(prefix="ecr-auth-", dir=auth_root) as directory:
        env = {"PATH": "/usr/bin:/bin", "HOME": str(STATE), "DOCKER_CONFIG": directory}
        run([cli, "--host=unix:///var/run/docker.sock", "login",
             "--username=AWS", "--password-stdin", registry],
            input=password, check=True, capture_output=True, text=True, timeout=30, env=env)
        run([cli, "--host=unix:///var/run/docker.sock", "pull", image],
            check=True, capture_output=True, text=True, timeout=3600, env=env)


def adopt_guard(source: Path, manifest: dict, job: str) -> None:
    expected_names = {"early.json", "early.py", "aws_worker_guard.py", "early-receipt.json"}
    if {path.name for path in trusted_path(GUARD).iterdir()} != expected_names:
        raise ValueError("Existing early guard has ambiguous contents")
    config = trusted_json(GUARD / "early.json")
    receipt = trusted_json(GUARD / "early-receipt.json")
    if (receipt.get("schema") != 1 or receipt.get("job_id") != job
            or receipt.get("spec_sha256") != manifest["spec_sha256"]
            or receipt.get("config_sha256") != hashlib.sha256(canonical(config)).hexdigest()
            or any(config.get(key) != manifest[key]
                   for key in ("account_id", "ami", "instance_type", "broker_id"))
            or config.get("region") != "us-west-2"
            or config.get("aws_cli") != manifest["aws_cli"]
            or config.get("source_sha256") != manifest["runtime"]["source_sha256"]
            or config.get("launcher_sha256") != hashlib.sha256(
                trusted_path(source / "scripts/so101-worker-install.py").read_bytes()).hexdigest()):
        raise ValueError("Existing early guard receipt does not bind this worker")
    for existing, original, key in (
        (GUARD / "early.py", source / "scripts/so101-worker-early.py", "early_sha256"),
        (GUARD / "aws_worker_guard.py", source / "src/so101/aws_worker_guard.py", "guard_sha256"),
        (UNITS / "lego-robot-worker-guard.service",
         source / "scripts/lego-robot-worker-guard.service", "guard_unit_sha256"),
    ):
        if (existing.is_symlink() or not trusted_path(existing).is_file()
                or hashlib.sha256(existing.read_bytes()).hexdigest() != config.get(key)
                or existing.read_bytes() != trusted_path(original).read_bytes()):
            raise ValueError("Existing early guard code/unit differs from pinned source")


def install(execution_sha256: str, source_sha256: str, *, metadata=None,
            run=subprocess.run, wall=time.time) -> dict:
    if os.geteuid() != 0:
        raise PermissionError("GPU host installation requires root")
    if sys.version_info < (3, 12):
        raise ValueError("Trusted AMI must supply Python 3.12 or newer")
    config = trusted_json(CONFIG / "execution.json")
    if (not re.fullmatch(r"[0-9a-f]{64}", execution_sha256)
            or hashlib.sha256(canonical(config)).hexdigest() != execution_sha256
            or set(config) != EXECUTION_FIELDS):
        raise ValueError("Execution configuration differs from its pinned canonical digest")
    expected_paths = {
        "manifest": str(CONFIG / "manifest.json"), "plan": str(CONFIG / "plan.json"),
        "staging_root": str(STATE / "staged"), "attempts_root": str(STATE / "attempts"),
        "qualification": str(CONFIG / "qualification.json"),
    }
    if (any(config.get(key) != value for key, value in expected_paths.items())
            or type(config["grace_seconds"]) is not int
            or not 1 <= config["grace_seconds"] <= 300):
        raise ValueError("Installer requires fixed private worker paths and bounded stop grace")
    manifest = load_manifest(Path(config["manifest"]), config["manifest_sha256"])
    plan = trusted_json(Path(config["plan"]))
    if (set(plan) != {"schema", "spec_sha256", "argv", "readonly_mounts", "archives"}
            or plan["schema"] != 1 or plan["spec_sha256"] != manifest["spec_sha256"]
            or plan["argv"] != manifest["spec"]["recipe"].get("argv")
            or hashlib.sha256(canonical(plan)).hexdigest() != config["plan_sha256"]
            or manifest["runtime"]["source_sha256"] != source_sha256
            or config["docker_cli"] != manifest["runtime"]["docker_cli"]):
        raise ValueError("Installer source/plan/runtime must match the exact root specification")
    for cli in (manifest["aws_cli"], config["docker_cli"], "/usr/bin/python3"):
        if not trusted_path(Path(cli)).is_file() or not os.access(cli, os.X_OK):
            raise ValueError("Trusted AMI must already supply executable AWS CLI, Docker, and Python")
    source = trusted_path(WORKER / "source")
    if {path.name for path in WORKER.iterdir()} != {"source"}:
        raise ValueError("Fresh worker code directory contains unexpected existing files")
    for parent in (GUARD.parent, STATE.parent, UNITS):
        trusted_path(parent)
    metadata = metadata if metadata is not None else Metadata()
    job, deadline = identity(manifest, metadata, wall=wall)
    early = GUARD.exists()
    if early:
        adopt_guard(source, manifest, job)
    if (STATE.exists() or (WORKER / "pyvenv.cfg").exists()
            or (CONFIG / "qualification.json").exists()):
        raise ValueError("Fresh-worker installer cannot overwrite existing state or qualification")
    assets = ("lego-robot-worker-guard.service", "so101-worker-spot.service",
              "so101-worker-spot.timer")
    if any((UNITS / name).exists() or (UNITS / name).is_symlink()
           for name in (assets[1:] if early else assets)):
        raise ValueError("Fresh-worker installer cannot replace existing service units")
    for name in assets:
        trusted_path(source / "scripts" / name)
    trusted_path(source / "src/so101/aws_worker_guard.py")

    def bounded_run(command, **options):
        remaining = deadline - wall()
        if remaining <= 0:
            raise TimeoutError("Worker lease expired during installation")
        options["timeout"] = min(options["timeout"], remaining)
        return run(command, **options)

    def systemctl(*arguments):
        bounded_run(["/usr/bin/systemctl", *arguments], check=True, capture_output=True,
                    text=True, timeout=30, env={"PATH": "/usr/bin:/bin"})

    if early:
        require_guard(bounded_run)
    STATE.mkdir(mode=0o700)
    atomic_json(STATE / "installation-intent.json", {
        "job_id": job, "execution_sha256": execution_sha256, "source_sha256": source_sha256,
    })
    if not early:
        GUARD.mkdir(mode=0o755)
        shutil.copyfile(source / "src/so101/aws_worker_guard.py", GUARD / "aws_worker_guard.py")
        (GUARD / "aws_worker_guard.py").chmod(0o644)
        shutil.copyfile(source / "scripts" / assets[0], UNITS / assets[0])
        (UNITS / assets[0]).chmod(0o644)
        systemctl("daemon-reload")
        systemctl("enable", "--now", assets[0])
    require_guard(bounded_run)

    venv.EnvBuilder(with_pip=False, symlinks=True).create(WORKER)
    sites = list((WORKER / "lib").glob("python*/site-packages"))
    if len(sites) != 1:
        raise ValueError("Private stdlib interpreter has an ambiguous site-packages directory")
    (sites[0] / "so101-worker.pth").write_text(str(source / "src") + "\n")
    for name in ("staged", "attempts"):
        (STATE / name).mkdir(mode=0o700)
    for name in assets[1:]:
        shutil.copyfile(source / "scripts" / name, UNITS / name)
        (UNITS / name).chmod(0o644)
    systemctl("daemon-reload")
    # Spot monitoring also protects image pull/staging on an otherwise idle worker.
    systemctl("enable", "--now", "so101-worker-spot.timer")
    runtimes = bounded_run([
        config["docker_cli"], "--host=unix:///var/run/docker.sock",
        "info", "--format={{json .Runtimes}}",
    ], check=True, capture_output=True, text=True, timeout=30,
        env={"PATH": "/usr/bin:/bin", "HOME": str(STATE)})
    if "nvidia" not in json.loads(runtimes.stdout):
        raise ValueError("Trusted AMI Docker daemon must already supply NVIDIA Container Toolkit")
    pull_image(manifest, STATE, run=bounded_run)
    staged = stage_idle(Path(config["manifest"]), config["manifest_sha256"],
                        STATE / "staged", metadata=metadata, run=bounded_run, wall=wall)
    prepare(Path(config["manifest"]), config["manifest_sha256"], Path(config["plan"]),
            config["plan_sha256"], staged)
    require_guard(bounded_run)
    if wall() >= deadline:
        raise TimeoutError("Worker lease expired during input preparation")
    receipt = {"schema": 1, "phase": "prepared-idle", "job_id": job,
               "spec_sha256": manifest["spec_sha256"], "source_sha256": source_sha256,
               "execution_sha256": execution_sha256, "deadline": deadline}
    atomic_json(STATE / "installed.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.execution_sha256, args.source_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
