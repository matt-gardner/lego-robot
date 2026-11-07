"""Isolated Spot qualification attempts and namespaces, never SFT resume artifacts."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import time

from so101.aws_checkpoint import CheckpointPublisher
from so101.aws_runtime import RuntimeImage
from so101.aws_worker_bootstrap import load_manifest, require_guard
from so101.cloud_checkpoint import atomic_json, sha256
from so101.control_broker import trusted_json, trusted_path


def start(config: dict, *, job_id: str, spec_sha256: str, run=subprocess.run,
          wall=time.time) -> dict:
    if os.geteuid() != 0 or not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise PermissionError("Disposable Spot qualification requires root and exact job")
    manifest = load_manifest(Path(config["manifest"]), config["manifest_sha256"])
    staged = trusted_path(Path(config["staging_root"]) / job_id)
    binding = trusted_json(staged / ".bootstrap.json")
    if (manifest["spec_sha256"] != spec_sha256 or binding.get("job_id") != job_id
            or binding.get("spec_sha256") != spec_sha256
            or binding.get("manifest_sha256") != config["manifest_sha256"]
            or binding.get("phase") != "staged-idle"
            or sha256(staged / ".staged.json") != binding.get("staging_receipt_sha256")
            or not wall() < binding["deadline"]):
        raise ValueError("Spot qualification requires the exact unexpired idle binding")
    original = trusted_path(staged / "training_state.pt")
    if sha256(original) != manifest["spec"]["checkpoint"]["sha256"]:
        raise ValueError("Original checkpoint changed before disposable qualification")
    padding = original.stat().st_size
    if not 1 <= padding <= 4 * 1024**3:
        raise ValueError("Representative qualification state exceeds bounded padding capacity")
    attempts = trusted_path(Path(config["attempts_root"]))
    if (attempts / job_id).exists():
        raise ValueError("Disposable qualification cannot overlap a training attempt")
    require_guard(run)
    runtime = manifest["runtime"]
    RuntimeImage(runtime["image"], runtime["commit"], runtime["source_sha256"]).verify(
        runtime["docker_cli"], run=run)
    root = attempts / "spot-qualification"
    root.mkdir(mode=0o700, exist_ok=True)
    trusted_path(root)
    attempt = root / job_id
    attempt.mkdir(mode=0o700)
    atomic_json(attempt / "intent.json", {
        "job_id": job_id, "spec_sha256": spec_sha256, "padding_bytes": padding,
        "original_sha256": manifest["spec"]["checkpoint"]["sha256"],
        "workload": "spot-qualification", "original_trial_advanced": False,
    })
    output = attempt / "output"
    output.mkdir(mode=0o700)
    os.chown(output, 1001, 1001)
    group = manifest["spec"]["log_group"]
    if not re.fullmatch(r"/so101/[A-Za-z0-9_./-]+", group):
        raise ValueError("Disposable qualification requires its approved CloudWatch group")
    duration = min(1800, int(binding["deadline"] - wall()))
    if duration < 1:
        raise TimeoutError("Qualification lease expired before Docker launch")
    result = run([
        runtime["docker_cli"], "--host=unix:///var/run/docker.sock", "run", "--detach",
        "--name=so101-spot-qual-" + job_id, "--restart=no", "--label=so101-spec=" + spec_sha256,
        "--label=so101-workload=spot-qualification", "--read-only", "--network=none",
        "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=1001:1001",
        "--gpus=device=0", "--tmpfs=/tmp:rw,nosuid,nodev,size=64m",
        "--env=CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "--log-driver=awslogs", "--log-opt=awslogs-region=us-west-2",
        "--log-opt=awslogs-group=" + group,
        "--log-opt=awslogs-stream=" + job_id + "-spot-qualification",
        "--log-opt=awslogs-create-stream=true", "--log-opt=mode=blocking",
        "--log-opt=awslogs-force-flush-interval-seconds=1",
        "--mount", f"type=bind,src={output},dst=/work/output",
        runtime["image"], "python", "-u", "-m", "so101.aws_spot_workload", "run",
        "--output", "/work/output", "--padding-bytes", str(padding), "--duration", str(duration),
    ], check=True, capture_output=True, text=True, timeout=min(60, duration))
    container = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{64}", container):
        raise ValueError("Ambiguous disposable qualification launch; reconciliation required")
    receipt = {"job_id": job_id, "spec_sha256": spec_sha256, "container_id": container,
               "phase": "spot-qualification-running", "original_trial_advanced": False}
    atomic_json(attempt / "launched.json", receipt)
    return receipt


def check_recovered(config: dict, *, job_id: str, spec_sha256: str, run=subprocess.run,
                    wall=time.time) -> dict:
    if os.geteuid() != 0 or not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise PermissionError("Recovered qualification requires root and exact job")
    manifest = load_manifest(Path(config["manifest"]), config["manifest_sha256"])
    staged = trusted_path(Path(config["staging_root"]) / job_id)
    binding = trusted_json(staged / ".bootstrap.json")
    if (manifest["spec_sha256"] != spec_sha256 or binding.get("job_id") != job_id
            or binding.get("spec_sha256") != spec_sha256
            or binding.get("manifest_sha256") != config["manifest_sha256"]
            or binding.get("phase") != "staged-idle"
            or sha256(staged / ".staged.json") != binding.get("staging_receipt_sha256")):
        raise ValueError("Recovered qualification requires the exact idle staging binding")
    remaining = binding["deadline"] - wall()
    if remaining <= 0:
        raise TimeoutError("Recovered qualification lease has expired")
    attempts = trusted_path(Path(config["attempts_root"]))
    if (attempts / job_id).exists() or (attempts / "spot-qualification" / job_id).exists():
        raise ValueError("Recovered GPU check cannot overlap training or disposable workload")
    require_guard(run)
    artifact = manifest["artifacts"].get("spot_recovered_state.pt")
    recovered = trusted_path(staged / "spot_recovered_state.pt")
    original = trusted_path(staged / "training_state.pt")
    if (not isinstance(artifact, dict) or recovered.is_symlink()
            or sha256(recovered) != artifact["sha256"]
            or recovered.stat().st_size != artifact["size"]
            or sha256(original) != manifest["spec"]["checkpoint"]["sha256"]):
        raise ValueError("Recovered GPU check requires exact verified new input copies")
    runtime = manifest["runtime"]
    RuntimeImage(runtime["image"], runtime["commit"], runtime["source_sha256"]).verify(
        runtime["docker_cli"], run=run)
    intent = attempts / (job_id + "-spot-recovery")
    intent.mkdir(mode=0o700)
    atomic_json(intent / "intent.json", {
        "job_id": job_id, "spec_sha256": spec_sha256, "artifact": artifact,
    })
    cli = runtime["docker_cli"]
    name = "so101-spot-recover-" + job_id
    try:
        result = run([
            cli, "--host=unix:///var/run/docker.sock", "run", "--rm", "--name=" + name,
            "--label=so101-spec=" + spec_sha256, "--gpus=device=0", "--read-only",
            "--network=none", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--user=0:0", "--env=CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "--mount", f"type=bind,src={recovered},dst=/check/state.pt,readonly",
            runtime["image"], "python", "-m", "so101.aws_spot_workload", "check-cuda",
            "--state", "/check/state.pt", "--padding-bytes", str(original.stat().st_size),
        ], check=True, capture_output=True, text=True, timeout=min(300, remaining))
    except subprocess.TimeoutExpired:
        inspected = run([cli, "--host=unix:///var/run/docker.sock", "inspect", name],
                        check=True, capture_output=True, text=True, timeout=10)
        container = json.loads(inspected.stdout)[0]
        if container["Config"]["Labels"].get("so101-spec") != spec_sha256:
            raise ValueError("Timed-out recovered qualification container is not owned")
        run([cli, "--host=unix:///var/run/docker.sock", "rm", "--force", container["Id"]],
            check=True, capture_output=True, text=True, timeout=30)
        raise
    body = json.loads(result.stdout)
    if (not isinstance(body, dict)
            or set(body) != {"step", "size", "sha256", "padding_bytes", "cuda_next_update_exact"}
            or type(body["step"]) is not int or body["step"] < 1
            or body.get("cuda_next_update_exact") is not True
            or body.get("padding_bytes") != original.stat().st_size
            or body.get("size") != artifact["size"] or body.get("sha256") != artifact["sha256"]
            or not wall() < binding["deadline"]):
        raise ValueError("Recovered GPU next update/reference/body verification failed")
    receipt = dict(body, job_id=job_id, spec_sha256=spec_sha256, artifact=artifact,
                   status="SPOT_CUDA_RECOVERY_PASS", original_trial_advanced=False,
                   aggregate_pass=False)
    atomic_json(intent / "recovered-runtime.json", receipt)
    return receipt


def publish(config: dict, *, job_id: str, spec_sha256: str, run=subprocess.run) -> dict:
    manifest = load_manifest(Path(config["manifest"]), config["manifest_sha256"])
    attempt = trusted_path(Path(config["attempts_root"]) / job_id)
    intent = trusted_json(attempt / "intent.json")
    if (manifest["spec_sha256"] != spec_sha256 or intent.get("job_id") != job_id
            or intent.get("spec_sha256") != spec_sha256
            or intent.get("workload") != "spot-qualification"
            or intent.get("original_trial_advanced") is not False):
        raise ValueError("Disposable publication differs from its isolated workload intent")
    cli, image = manifest["runtime"]["docker_cli"], manifest["runtime"]["image"]
    padding = intent["padding_bytes"]

    def validate(path):
        result = run([
            cli, "--host=unix:///var/run/docker.sock", "run", "--rm", "--network=none",
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=0:0",
            "--mount", f"type=bind,src={path},dst=/check/state.pt,readonly",
            image, "python", "-m", "so101.aws_spot_workload", "check",
            "--state", "/check/state.pt", "--padding-bytes", str(padding),
        ], check=True, capture_output=True, text=True, timeout=120)
        body = json.loads(result.stdout)
        if (set(body) != {"step", "size", "sha256", "padding_bytes", "cuda_next_update_exact"}
                or type(body["step"]) is not int or body["step"] < 1
                or body["size"] != path.stat().st_size or body["size"] < padding
                or body["sha256"] != sha256(path) or body["padding_bytes"] != padding
                or body["cuda_next_update_exact"] is not False):
            raise ValueError("Runtime returned invalid disposable state validation")
        return body

    state = attempt / "output/training_state.pt"
    if state.is_symlink() or not state.is_file():
        raise ValueError("Disposable checkpoint must be a regular new output")
    checked = validate(state)
    directory = attempt / "publication"
    directory.mkdir(mode=0o700)
    prefix = manifest["spec"]["output_uri"]
    publisher = CheckpointPublisher(
        owner=manifest["account_id"], aws_cli=manifest["aws_cli"],
        output_prefix=prefix + "qualification/", run=run, request_timeout=120,
    )

    def snapshot(path):
        if validate(path) != checked:
            raise ValueError("Disposable snapshot changed after validation")

    artifact = publisher.publish(
        state, directory, job_id=job_id, spec_sha256=spec_sha256, step=checked["step"],
        max_bytes=padding + 1024**2, validate_snapshot=snapshot,
    )
    spot = trusted_json(attempt.parent.parent / (job_id + "-spot.json"))
    if spot.get("job_id") != job_id or spot.get("spec_sha256") != spec_sha256:
        raise ValueError("Disposable publication requires an actual bound Spot notice intent")
    evidence = dict(spot, workload="spot-qualification", original_trial_advanced=False,
                    original_sha256=intent["original_sha256"], padding_bytes=padding,
                    state_manifest=asdict(artifact), state_bytes=checked["size"],
                    log_stream=job_id + "-spot-qualification")
    event = attempt / "evidence.json"
    atomic_json(event, evidence)
    event_root = attempt / "event-publication"
    event_root.mkdir(mode=0o700)
    events = CheckpointPublisher(
        owner=manifest["account_id"], aws_cli=manifest["aws_cli"],
        output_prefix=prefix + "qualification-events/", run=run, request_timeout=120,
    )
    event_manifest = events.publish(event, event_root, job_id=job_id,
                                    spec_sha256=spec_sha256, step=0, max_bytes=65536)
    receipt = {"phase": "spot-qualification-published", "job_id": job_id,
               "spec_sha256": spec_sha256, "manifest": asdict(artifact),
               "evidence_manifest": asdict(event_manifest), "state_bytes": checked["size"],
               "padding_bytes": padding, "original_trial_advanced": False}
    atomic_json(attempt / "recovery.json", receipt)
    return receipt


def stop(config: dict, *, job_id: str, spec_sha256: str, stop_deadline: int,
         run=subprocess.run, wall=time.time, monotonic=time.monotonic):
    # The launcher dispatches qualification through this module, creating an import cycle.
    from so101.aws_worker_run import stop_and_publish
    isolated = dict(config, attempts_root=str(Path(config["attempts_root"]) / "spot-qualification"))
    return stop_and_publish(isolated, job_id=job_id, spec_sha256=spec_sha256,
                            stop_deadline=stop_deadline, run=run, wall=wall,
                            monotonic=monotonic, publisher=publish)
