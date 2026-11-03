"""One-shot IMDSv2 Spot handling; timer driven, never restarts training."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.error import HTTPError

from so101.aws_worker_bootstrap import load_manifest
from so101.aws_worker_guard import Metadata, poweroff
from so101.aws_worker_run import stop_and_publish
from so101.cloud_checkpoint import atomic_json
from so101.control_broker import trusted_json, trusted_path


LOG = logging.getLogger(__name__)


def notice_deadline(notice: dict, now: float, lease: float) -> int:
    if not isinstance(notice, dict) or set(notice) != {"action", "time"}:
        raise ValueError("Malformed Spot interruption notice")
    if notice["action"] not in {"terminate", "stop", "hibernate"}:
        raise ValueError("Unknown Spot interruption action")
    deadline = datetime.fromisoformat(notice["time"].replace("Z", "+00:00"))
    if deadline.utcoffset() != timezone.utc.utcoffset(deadline):
        raise ValueError("Spot interruption time must be UTC")
    return int(min(deadline.timestamp(), now + 120, lease))


def tick(config: dict, *, metadata=None, wall=time.time, stop=stop_and_publish,
         shutdown=poweroff) -> bool:
    if os.geteuid() != 0:
        raise PermissionError("Spot monitor must run on the root-installed GPU worker")
    manifest = load_manifest(Path(config["manifest"]), config["manifest_sha256"])
    metadata = metadata if metadata is not None else Metadata()
    token = metadata.token()
    identity = json.loads(metadata.read("dynamic/instance-identity/document", token))
    expected = {"accountId": manifest["account_id"], "region": "us-west-2",
                "imageId": manifest["ami"], "instanceType": manifest["instance_type"]}
    if not isinstance(identity, dict) or any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Spot monitor identity differs from approved GPU worker")
    job = metadata.read("meta-data/tags/instance/so101-job", token)
    spec = metadata.read("meta-data/tags/instance/so101-spec", token)
    broker = metadata.read("meta-data/tags/instance/so101-broker", token)
    if (not re.fullmatch(r"[0-9a-f]{32}", job)
            or spec != manifest["spec_sha256"] or broker != manifest["broker_id"]):
        raise ValueError("Spot monitor job/spec/broker identity mismatch")
    try:
        notice = json.loads(metadata.read("meta-data/spot/instance-action", token))
    except HTTPError as error:
        if error.code == 404:
            return False
        raise
    deadline = notice_deadline(notice, wall(), metadata.deadline())
    attempts = trusted_path(Path(config["attempts_root"]))
    attempt = attempts / job
    intent = attempts / (job + "-spot.json")
    if intent.exists():
        previous = trusted_json(intent)
        if (previous.get("job_id") != job or previous.get("spec_sha256") != spec
                or type(previous.get("deadline")) is not int):
            raise ValueError("Invalid existing Spot interruption intent")
        deadline = min(deadline, previous["deadline"])
    if attempt.exists():
        atomic_json(attempt / "stop-deadline.json", {
            "job_id": job, "spec_sha256": spec, "deadline": deadline,
        })
    if intent.exists():
        if attempt.exists():
            descriptor = os.open(attempt / "stop.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "rb") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    LOG.warning("Spot deadline recorded; checkpoint operation still active")
                    return False
        LOG.warning("Spot operation already attempted; requesting termination without replay")
        shutdown()
        return True
    atomic_json(intent, {"job_id": job, "spec_sha256": spec, "deadline": deadline, "notice": notice})
    if not attempt.exists() or not (attempt / "launched.json").exists() or deadline <= wall():
        LOG.warning("Spot interrupt on idle/expired worker; requesting termination")
        shutdown()
        return True
    try:
        stop(config, job_id=job, spec_sha256=spec, stop_deadline=deadline)
    except BlockingIOError:
        LOG.warning("Existing checkpoint operation owns stop lock; Spot deadline shortened")
        return False
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        LOG.error("Spot checkpoint recovery failed; requesting termination: %s", error)
        shutdown()
        raise
    shutdown()
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    tick(trusted_json(Path("/etc/so101-worker/execution.json")))


if __name__ == "__main__":
    main()
