"""Root-manifest-bound idle AWS staging; deliberately no training entrypoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import time

from so101.aws_artifacts import Artifact, ArtifactStore
from so101.aws_runtime import RuntimeImage
from so101.aws_worker_guard import Metadata
from so101.cloud_checkpoint import atomic_json, sha256
from so101.control_broker import DIGEST, IDENTIFIER, canonical, trusted_json, trusted_path


FIELDS = {
    "schema", "broker_id", "spec_sha256", "account_id", "instance_type", "ami",
    "aws_cli", "prefixes", "max_bytes", "artifacts", "runtime", "spec",
}


def load_manifest(path: Path, expected_sha256: str) -> dict:
    if not isinstance(expected_sha256, str) or not DIGEST.fullmatch(expected_sha256):
        raise ValueError("A pinned canonical manifest SHA256 is required")
    manifest = trusted_json(path)
    if hashlib.sha256(canonical(manifest)).hexdigest() != expected_sha256:
        raise ValueError("Root manifest differs from the pinned canonical SHA256")
    if (set(manifest) != FIELDS or type(manifest["schema"]) is not int
            or manifest["schema"] != 2):
        raise ValueError("Unsupported root staging manifest schema")
    if not isinstance(manifest["broker_id"], str) or not IDENTIFIER.fullmatch(manifest["broker_id"]):
        raise ValueError("Invalid manifest broker ID")
    if not isinstance(manifest["spec_sha256"], str) or not DIGEST.fullmatch(manifest["spec_sha256"]):
        raise ValueError("Root manifest must pin the registered job specification")
    if (not isinstance(manifest["instance_type"], str)
            or manifest["instance_type"] not in {"g5.2xlarge", "g6.2xlarge"}):
        raise ValueError("Only the approved Oregon A10G/L4 worker sizes may stage")
    if not isinstance(manifest["ami"], str) or not re.fullmatch(r"ami-[0-9a-f]+", manifest["ami"]):
        raise ValueError("Root manifest must pin an AMI")
    if (not isinstance(manifest["prefixes"], list) or not manifest["prefixes"]
            or type(manifest["max_bytes"]) is not int or manifest["max_bytes"] <= 0
            or not isinstance(manifest["artifacts"], dict) or not manifest["artifacts"]):
        raise ValueError("Root manifest requires prefixes, byte ceiling, and artifacts")
    runtime = manifest["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "docker_cli", "image", "commit", "source_sha256",
    }:
        raise ValueError("Root manifest must pin the local runtime and source identity")
    RuntimeImage(runtime["image"], runtime["commit"], runtime["source_sha256"]).validate()
    spec = manifest["spec"]
    if (not isinstance(spec, dict)
            or hashlib.sha256(canonical(spec)).hexdigest() != manifest["spec_sha256"]
            or spec.get("image") != runtime["image"] or spec.get("commit") != runtime["commit"]
            or spec.get("ami") != manifest["ami"]
            or spec.get("instance_type") != manifest["instance_type"]):
        raise ValueError("Root runtime must match the exact registered specification contents")
    source = manifest["artifacts"].get("source.tar")
    if not isinstance(source, dict) or source.get("sha256") != runtime["source_sha256"]:
        raise ValueError("Root runtime source digest must match the source.tar artifact")
    checkpoint = spec.get("checkpoint")
    staged_checkpoint = manifest["artifacts"].get("training_state.pt")
    if (not isinstance(checkpoint, dict) or set(checkpoint) != {"uri", "version", "sha256"}
            or not isinstance(staged_checkpoint, dict)
            or any(staged_checkpoint.get(key) != value for key, value in checkpoint.items())):
        raise ValueError("Staged training state must match the exact registered checkpoint")
    return manifest


def require_guard(run) -> None:
    run(
        ["/usr/bin/systemctl", "is-active", "--quiet", "lego-robot-worker-guard.service"],
        check=True, timeout=10, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin"},
    )


def stage_idle(manifest_path: Path, manifest_sha256: str, root: Path, *,
               metadata=None, run=subprocess.run, wall=time.time,
               monotonic=time.monotonic) -> Path:
    manifest = load_manifest(manifest_path, manifest_sha256)
    root = trusted_path(root)
    if not root.is_dir():
        raise ValueError("Worker staging root must be an operator-owned directory")
    store = ArtifactStore(
        owner=manifest["account_id"], prefixes=tuple(manifest["prefixes"]),
        aws_cli=manifest["aws_cli"], run=run,
    )
    if not trusted_path(Path(manifest["aws_cli"])).is_file():
        raise ValueError("Worker AWS CLI must be a root-installed executable file")
    runtime = manifest["runtime"]
    if (not isinstance(runtime["docker_cli"], str) or not Path(runtime["docker_cli"]).is_absolute()
            or not trusted_path(Path(runtime["docker_cli"])).is_file()):
        raise ValueError("Worker Docker CLI must be an absolute root-installed file")
    artifacts = {}
    for name, value in manifest["artifacts"].items():
        if not isinstance(value, dict) or set(value) != {"uri", "version", "sha256", "size"}:
            raise ValueError("Root manifest artifacts require exact immutable reference fields")
        artifacts[name] = Artifact(**value)
        artifacts[name].validate()
    require_guard(run)
    metadata = metadata if metadata is not None else Metadata()
    token = metadata.token()
    identity = json.loads(metadata.read("dynamic/instance-identity/document", token))
    expected_identity = {
        "accountId": manifest["account_id"], "region": "us-west-2",
        "instanceType": manifest["instance_type"], "imageId": manifest["ami"],
    }
    if not isinstance(identity, dict) or any(
        identity.get(key) != value for key, value in expected_identity.items()
    ):
        raise ValueError("Worker instance identity differs from the root staging manifest")
    tags = {
        name: metadata.read("meta-data/tags/instance/so101-" + name, token)
        for name in ("job", "spec", "broker")
    }
    if (not re.fullmatch(r"[0-9a-f]{32}", tags["job"])
            or tags["spec"] != manifest["spec_sha256"]
            or tags["broker"] != manifest["broker_id"]):
        raise ValueError("Worker job/spec/broker tags differ from the root staging manifest")
    runtime_identity = RuntimeImage(
        runtime["image"], runtime["commit"], runtime["source_sha256"],
    ).verify(runtime["docker_cli"], run=run)
    started = monotonic()
    deadline = metadata.deadline()
    remaining = deadline - wall()
    if not math.isfinite(remaining) or remaining <= 0:
        raise ValueError("Worker lease is already expired or invalid")
    end = started + remaining
    destination = store.stage(root, tags["job"], artifacts, max_bytes=manifest["max_bytes"])
    require_guard(run)
    if wall() >= deadline or monotonic() >= end:
        raise ValueError("Worker lease expired during staging; no idle binding committed")
    atomic_json(destination / ".bootstrap.json", {
        "schema": 2, "phase": "staged-idle", "job_id": tags["job"],
        "manifest_sha256": manifest_sha256, "spec_sha256": manifest["spec_sha256"],
        "deadline": deadline, "staging_receipt_sha256": sha256(destination / ".staged.json"),
        "runtime": runtime_identity,
    })
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("AWS worker staging entrypoint must run as root")
    destination = stage_idle(args.manifest, args.manifest_sha256, args.root)
    logging.info("Verified artifact staging complete at %s; worker remains idle", destination)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
