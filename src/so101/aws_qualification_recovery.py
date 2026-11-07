"""Read-only controller evidence retrieval; never authorizes training or cleanup."""
from __future__ import annotations

import json
import os
from pathlib import Path

from so101.aws_artifacts import Artifact, split_uri
from so101.aws_recovery import recover
from so101.cloud_checkpoint import fsync_directory, sha256
from so101.aws_spot_notice import notice_deadline


def evidence(request, root: Path, *, owner: str, prefix: str, job_id: str,
             spec_sha256: str, original_checkpoint: dict, log_group: str,
             termination_observed_at: float) -> dict:
    state = recover(request, root / "spot-state", owner=owner,
                    prefix=prefix + "qualification/", job_id=job_id, spec_sha256=spec_sha256)
    events = recover(request, root / "spot-events", owner=owner,
                     prefix=prefix + "qualification-events/", job_id=job_id,
                     spec_sha256=spec_sha256)
    artifact = Artifact(**events["checkpoint"])
    if not 0 < artifact.size <= 65536:
        raise ValueError("Qualification evidence must be a bounded JSON object")
    directory = root / "spot-events" / job_id
    path = directory / (artifact.sha256 + ".evidence.json")
    if path.is_symlink():
        raise ValueError("Qualification evidence cannot replace a symlink")
    if not path.exists():
        bucket, key = split_uri(artifact.uri)
        temporary = directory / (artifact.sha256 + ".evidence.part")
        with temporary.open("xb"):
            pass
        response = request(
            "s3api", "get-object", "--bucket", bucket, "--key=" + key,
            "--version-id=" + artifact.version, "--expected-bucket-owner", owner, str(temporary),
        )
        if (response.get("VersionId") != artifact.version
                or temporary.stat().st_size != artifact.size or sha256(temporary) != artifact.sha256):
            raise ValueError("Qualification evidence body differs from exact committed version")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        fsync_directory(directory)
    if path.stat().st_size != artifact.size or sha256(path) != artifact.sha256:
        raise ValueError("Local qualification evidence body changed")
    body = json.loads(path.read_bytes())
    if (body.get("job_id") != job_id or body.get("spec_sha256") != spec_sha256
            or body.get("workload") != "spot-qualification"
            or body.get("original_trial_advanced") is not False
            or body.get("original_sha256") != original_checkpoint["sha256"]
            or body.get("state_manifest") != state["manifest"]
            or body.get("log_stream") != job_id + "-spot-qualification"
            or body.get("state_bytes") != state["checkpoint"]["size"]
            or type(body.get("padding_bytes")) is not int
            or not 1 <= body["padding_bytes"] <= body.get("state_bytes", 0)
            or type(body.get("deadline")) is not int
            or not isinstance(body.get("notice"), dict)):
        raise ValueError("Qualification state/Spot evidence does not bind exact job/specification")
    if notice_deadline(body["notice"], body["deadline"] - 120, body["deadline"]) != body["deadline"]:
        raise ValueError("Qualification deadline exceeds its real Spot notice")
    bucket, key = split_uri(original_checkpoint["uri"])
    original = request(
        "s3api", "head-object", "--bucket", bucket, "--key=" + key,
        "--version-id=" + original_checkpoint["version"], "--expected-bucket-owner", owner,
    )
    if (original.get("VersionId") != original_checkpoint["version"]
            or type(original.get("ContentLength")) is not int or original["ContentLength"] < 1
            or body["padding_bytes"] < original["ContentLength"]):
        raise ValueError("Disposable checkpoint is smaller than the exact original byte population")
    logs = request("logs", "get-log-events", "--log-group-name", log_group,
                   "--log-stream-name", body["log_stream"], "--limit", "1000")
    markers = []
    for item in logs.get("events", []):
        message = item.get("message", "")
        if '"workload":' in message:
            marker = json.loads(message)
            if marker.get("workload") == "so101-spot-qualification-v1":
                markers.append(marker)
    return {
        "status": "QUALIFICATION_EVIDENCE", "job_id": job_id, "spec_sha256": spec_sha256,
        "checkpoint": state["checkpoint"], "evidence": body,
        "termination_observed_at": termination_observed_at,
        "termination_observed_by_deadline": termination_observed_at <= body["deadline"],
        "checkpoint_stop_log_verified": any(
            marker.get("checkpoint_stop") is True
            and marker.get("original_trial_advanced") is False for marker in markers),
        "original_trial_advanced": False, "aggregate_pass": False,
        "local_full_checkpoint_export_verified": False,
    }
