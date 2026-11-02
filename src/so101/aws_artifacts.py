"""Exact-version S3 staging primitives; no launch, extraction, or trainer start."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit

from so101.cloud_checkpoint import atomic_json, fsync_directory, sha256


def worker_request(args: list[str], run=subprocess.run) -> dict:
    result = run(
        args, check=True, capture_output=True, text=True, timeout=3600,
        env={
            "PATH": "/usr/bin:/bin", "HOME": "/var/lib/so101-worker",
            "AWS_CONFIG_FILE": "/dev/null", "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
            "AWS_EC2_METADATA_DISABLED": "false", "AWS_MAX_ATTEMPTS": "1",
            "AWS_METADATA_SERVICE_TIMEOUT": "2", "AWS_METADATA_SERVICE_NUM_ATTEMPTS": "1",
        },
    )
    response = json.loads(result.stdout)
    if not isinstance(response, dict):
        raise ValueError("Worker AWS request must return a JSON object")
    return response


@dataclass(frozen=True)
class Artifact:
    uri: str
    version: str
    sha256: str
    size: int

    def validate(self) -> None:
        split_uri(self.uri)
        if (not isinstance(self.version, str) or self.version == "null"
                or not re.fullmatch(r"[A-Za-z0-9._+/=-]{1,1024}", self.version)):
            raise ValueError("Artifact requires an explicit non-null S3 version")
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("Artifact requires a SHA256 digest")
        if type(self.size) is not int or self.size <= 0:
            raise ValueError("Artifact size must be a positive integer")


def split_uri(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str):
        raise ValueError("Artifact URI must be a string")
    parsed = urlsplit(uri)
    if (parsed.scheme != "s3" or parsed.query or parsed.fragment
            or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", parsed.netloc)):
        raise ValueError("Artifact requires a canonical S3 URI")
    key = parsed.path.removeprefix("/")
    if (uri != f"s3://{parsed.netloc}/{key}"
            or not key or any(part in {"", ".", ".."} for part in key.split("/"))
            or any(ord(char) < 32 or ord(char) > 126 for char in key)
            or "\\" in key):
        raise ValueError("Artifact requires a canonical S3 object key")
    return parsed.netloc, key


class ArtifactStore:
    def __init__(self, *, owner: str, prefixes: tuple[str, ...],
                 aws_cli: str, run=subprocess.run):
        if not isinstance(owner, str) or not re.fullmatch(r"[0-9]{12}", owner):
            raise ValueError("An exact bucket-owner account is required")
        if not isinstance(aws_cli, str) or not Path(aws_cli).is_absolute():
            raise ValueError("AWS CLI must be an absolute installed worker path")
        if not prefixes:
            raise ValueError("At least one scoped input/recovery prefix is required")
        for prefix in prefixes:
            if not isinstance(prefix, str) or not prefix.endswith("/"):
                raise ValueError("Artifact prefixes must end in /")
            split_uri(prefix[:-1])
        self.owner = owner
        self.prefixes = tuple(prefixes)
        self.aws_cli = aws_cli
        self.run = run

    def request(self, operation: str, artifact: Artifact, target: Path | None = None) -> dict:
        if (operation not in {"head-object", "get-object"}
                or (target is not None) != (operation == "get-object")):
            raise ValueError("Only versioned HEAD and file GET operations are supported")
        artifact.validate()
        if not any(artifact.uri.startswith(prefix) for prefix in self.prefixes):
            raise ValueError("Artifact is outside the operator-approved prefixes")
        bucket, key = split_uri(artifact.uri)
        args = [
            self.aws_cli, "s3api", operation, "--bucket", bucket, "--key=" + key,
            "--version-id=" + artifact.version, "--expected-bucket-owner", self.owner,
            "--region", "us-west-2", "--output", "json", "--no-cli-pager",
        ]
        if target is not None:
            args.append(str(target))
        # Worker instance-role credentials only; never inherit controller/user keys.
        response = worker_request(args, self.run)
        if (not isinstance(response, dict) or response.get("VersionId") != artifact.version
                or type(response.get("ContentLength")) is not int
                or response["ContentLength"] != artifact.size):
            raise ValueError("S3 response differs from the exact artifact version/size")
        return response

    def stage(self, root: Path, job_id: str, artifacts: dict[str, Artifact], *,
              max_bytes: int) -> Path:
        """Commit a receipt only after every object verifies in a new private attempt."""
        if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("An exact broker job ID is required")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("An explicit positive staging-byte ceiling is required")
        if not isinstance(artifacts, dict) or not 1 <= len(artifacts) <= 64:
            raise ValueError("Stage between one and 64 named artifacts")
        artifacts = dict(artifacts)
        for name, artifact in artifacts.items():
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", name):
                raise ValueError("Artifact destinations must be safe flat filenames")
            if not isinstance(artifact, Artifact):
                raise ValueError("Artifact entries must be explicit immutable references")
            artifact.validate()
            if not any(artifact.uri.startswith(prefix) for prefix in self.prefixes):
                raise ValueError("Artifact is outside the operator-approved prefixes")
        if sum(artifact.size for artifact in artifacts.values()) > max_bytes:
            raise ValueError("Artifacts exceed the reserved staging-byte ceiling")
        destination = root.resolve(strict=True) / job_id
        # Never reuse, overwrite, or clear STOP in an existing attempt.
        destination.mkdir(mode=0o700)
        fsync_directory(destination.parent)
        for name, artifact in artifacts.items():
            self.request("head-object", artifact)
            temporary = destination / ("." + name + ".part")
            self.request("get-object", artifact, temporary)
            if temporary.stat().st_size != artifact.size or sha256(temporary) != artifact.sha256:
                raise ValueError(f"Staged artifact failed checksum/size verification: {name}")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary.replace(destination / name)
            fsync_directory(destination)
        atomic_json(destination / ".staged.json", {
            "schema": 1, "job_id": job_id,
            "artifacts": {name: asdict(artifact) for name, artifact in artifacts.items()},
        })
        return destination
