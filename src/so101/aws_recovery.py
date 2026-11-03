"""Controller read-only discovery and verification of committed worker checkpoints."""
from __future__ import annotations

import base64
from dataclasses import asdict
import json
import os
from pathlib import Path
import re

from so101.aws_artifacts import Artifact, split_uri
from so101.cloud_checkpoint import atomic_json, sha256


def recover(request, root: Path, *, owner: str, prefix: str, job_id: str,
            spec_sha256: str) -> dict:
    if (not re.fullmatch(r"[0-9]{12}", owner)
            or not re.fullmatch(r"[0-9a-f]{32}", job_id)
            or not re.fullmatch(r"[0-9a-f]{64}", spec_sha256)
            or not prefix.endswith("/")):
        raise ValueError("Recovery requires exact owner/job/specification and approved output prefix")
    bucket, key = split_uri(prefix + job_id)
    key += "/"
    listed = request("s3api", "list-objects-v2", "--bucket", bucket, "--prefix", key,
                     "--expected-bucket-owner", owner)
    if listed.get("IsTruncated") is True or listed.get("NextContinuationToken"):
        raise ValueError("Recovery object population is incomplete; operator reconciliation required")
    contents = listed.get("Contents", [])
    if not isinstance(contents, list):
        raise ValueError("Invalid recovery object population")
    manifests = []
    for item in contents:
        name = item.get("Key") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name.startswith(key):
            raise ValueError("Recovery listing escaped the exact job namespace")
        match = re.fullmatch(re.escape(key) + r"step_([0-9]{9})/([0-9a-f]{64})\.json", name)
        if match:
            manifests.append((int(match[1]), match[2], name))
        elif not re.fullmatch(re.escape(key) + r"step_[0-9]{9}/[0-9a-f]{64}\.pt", name):
            raise ValueError("Unexpected object in recovery job namespace")
    if not manifests:
        raise ValueError("No committed checkpoint manifests in the exact recovery job namespace")
    step = max(item[0] for item in manifests)
    selected = [item for item in manifests if item[0] == step]
    if len(selected) != 1:
        raise ValueError("Multiple manifests for latest checkpoint; operator reconciliation required")
    _, digest, manifest_key = selected[0]

    def head(artifact: Artifact | None, object_key: str, expected_sha: str) -> Artifact:
        options = [] if artifact is None else ["--version-id=" + artifact.version]
        response = request(
            "s3api", "head-object", "--bucket", bucket, "--key=" + object_key,
            "--expected-bucket-owner", owner, "--checksum-mode", "ENABLED", *options,
        )
        result = Artifact("s3://" + bucket + "/" + object_key, response.get("VersionId"),
                          expected_sha, response.get("ContentLength"))
        result.validate()
        if (artifact is not None and result != artifact
                or response.get("ChecksumSHA256") != base64.b64encode(bytes.fromhex(expected_sha)).decode()):
            raise ValueError("Recovery version/size/checksum differs from committed identity")
        return result

    manifest = head(None, manifest_key, digest)
    if manifest.size > 65536:
        raise ValueError("Recovery manifest exceeds bounded metadata size")
    directory = root / job_id
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    for path in (root, directory):
        if (path.is_symlink() or path.stat().st_uid != os.geteuid()
                or path.stat().st_mode & 0o077):
            raise ValueError("Recovery storage must be a privately owned controller directory")
    temporary = directory / (digest + ".part")
    if temporary.is_symlink():
        raise ValueError("Recovery download must not overwrite a symlink")
    response = request(
        "s3api", "get-object", "--bucket", bucket, "--key=" + manifest_key,
        "--version-id=" + manifest.version, "--expected-bucket-owner", owner,
        "--checksum-mode", "ENABLED", str(temporary),
    )
    if (response.get("VersionId") != manifest.version
            or temporary.stat().st_size != manifest.size or sha256(temporary) != digest):
        raise ValueError("Downloaded recovery manifest differs from pinned version/checksum")
    body = json.loads(temporary.read_text())
    if (not isinstance(body, dict) or set(body) != {
        "schema", "job_id", "step", "spec_sha256", "checkpoint",
    } or type(body["schema"]) is not int or body["schema"] != 1
            or body["job_id"] != job_id or body["spec_sha256"] != spec_sha256
            or type(body["step"]) is not int or body["step"] != step
            or not isinstance(body["checkpoint"], dict)
            or set(body["checkpoint"]) != {"uri", "version", "sha256", "size"}):
        raise ValueError("Recovery manifest job/specification/update identity mismatch")
    checkpoint = Artifact(**body["checkpoint"])
    checkpoint.validate()
    expected_uri = prefix + job_id + f"/step_{step:09d}/" + checkpoint.sha256 + ".pt"
    if checkpoint.uri != expected_uri:
        raise ValueError("Recovery checkpoint escaped its committed job/update/hash namespace")
    _, checkpoint_key = split_uri(checkpoint.uri)
    head(checkpoint, checkpoint_key, checkpoint.sha256)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(directory / (digest + ".json"))
    receipt = {"job_id": job_id, "spec_sha256": spec_sha256, "step": step,
               "manifest": asdict(manifest), "checkpoint": asdict(checkpoint)}
    atomic_json(directory / "receipt.json", receipt)
    return receipt
