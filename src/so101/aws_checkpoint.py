"""Stable opaque checkpoint publication; no trainer, deletion, or automatic retry."""
from __future__ import annotations

import base64
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Callable

from so101.aws_artifacts import Artifact, split_uri, worker_request
from so101.cloud_checkpoint import atomic_json, fsync_directory, sha256


class CheckpointPublisher:
    def __init__(self, *, owner: str, aws_cli: str, output_prefix: str, run=subprocess.run,
                 request_timeout: int = 3600):
        if not isinstance(owner, str) or not re.fullmatch(r"[0-9]{12}", owner):
            raise ValueError("Checkpoint publication requires the exact bucket owner")
        if not isinstance(aws_cli, str) or not Path(aws_cli).is_absolute():
            raise ValueError("Checkpoint publication requires an absolute installed AWS CLI")
        if type(request_timeout) is not int or not 1 <= request_timeout <= 3600:
            raise ValueError("Checkpoint request timeout must be bounded")
        if not isinstance(output_prefix, str) or not output_prefix.endswith("/"):
            raise ValueError("Checkpoint output prefix must end in /")
        _, key = split_uri(output_prefix[:-1])
        if key != "runs/lego-robot" and not key.startswith("runs/lego-robot/"):
            raise ValueError("Checkpoints must stay in the approved runs/lego-robot namespace")
        self.owner, self.aws_cli, self.output_prefix, self.run = owner, aws_cli, output_prefix, run
        self.request_timeout = request_timeout

    def request(self, operation: str, uri: str, *options: str) -> dict:
        if operation not in {"put-object", "head-object"} or not uri.startswith(self.output_prefix):
            raise ValueError("Only checkpoint publication/confirmation inside the output prefix is allowed")
        bucket, key = split_uri(uri)
        return worker_request([
            self.aws_cli, "s3api", operation, "--bucket", bucket, "--key=" + key,
            "--expected-bucket-owner", self.owner, "--region", "us-west-2",
            "--output", "json", "--no-cli-pager", *options,
        ], self.run, timeout=self.request_timeout)

    def put(self, path: Path, uri: str) -> Artifact:
        digest = sha256(path)
        checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
        response = self.request(
            "put-object", uri, "--body", str(path), "--if-none-match", "*",
            "--checksum-algorithm", "SHA256", "--checksum-sha256", checksum,
            "--server-side-encryption", "AES256",
        )
        artifact = Artifact(uri, response.get("VersionId"), digest, path.stat().st_size)
        artifact.validate()
        if response.get("ChecksumSHA256") != checksum:
            raise ValueError("S3 upload did not acknowledge the exact checkpoint checksum")
        head = self.request(
            "head-object", uri, "--version-id=" + artifact.version, "--checksum-mode", "ENABLED",
        )
        if (head.get("VersionId") != artifact.version
                or type(head.get("ContentLength")) is not int or head["ContentLength"] != artifact.size
                or head.get("ChecksumSHA256") != checksum):
            raise ValueError("S3 checkpoint version/size/checksum confirmation failed")
        return artifact

    def publish(self, source: Path, root: Path, *, job_id: str, step: int,
                spec_sha256: str, max_bytes: int,
                validate_snapshot: Callable[[Path], None] | None = None) -> Artifact:
        if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("An exact broker job ID is required")
        if type(step) is not int or step < 0:
            raise ValueError("Checkpoint step must be a nonnegative integer")
        if not isinstance(spec_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", spec_sha256):
            raise ValueError("Checkpoint publication must bind a registered specification digest")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("Checkpoint publication requires a positive byte ceiling")
        directory = root.resolve(strict=True) / job_id / f"step_{step:09d}"
        # The operator supplies a task-owned publication root, not original outputs.
        if directory.parent.is_symlink():
            raise ValueError("Publication job directory must not be a symlink")
        directory.parent.mkdir(mode=0o700, exist_ok=True)
        info = directory.parent.stat()
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("Publication job directory must be privately owned by the worker")
        fsync_directory(directory.parent.parent)
        directory.mkdir(mode=0o700)
        fsync_directory(directory.parent)
        snapshot = directory / "checkpoint.pt"
        digest = hashlib.sha256()
        size = 0
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as incoming, snapshot.open("xb") as outgoing:
            before = os.fstat(incoming.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("Checkpoint source must be a regular non-symlink file")
            if before.st_size <= 0 or before.st_size > max_bytes:
                raise ValueError("Checkpoint snapshot exceeds the declared byte ceiling or is empty")
            for block in iter(lambda: incoming.read(1024 * 1024), b""):
                size += len(block)
                if size > max_bytes:
                    raise ValueError("Growing checkpoint exceeds the declared byte ceiling")
                digest.update(block)
                outgoing.write(block)
            after = os.fstat(incoming.fileno())
            if (size != before.st_size or before.st_size != after.st_size
                    or before.st_mtime_ns != after.st_mtime_ns
                    or before.st_ctime_ns != after.st_ctime_ns):
                raise ValueError("Checkpoint changed in place during snapshot")
            outgoing.flush()
            os.fsync(outgoing.fileno())
        fsync_directory(directory)
        prefix = self.output_prefix + job_id + f"/step_{step:09d}/"
        uri = prefix + digest.hexdigest() + ".pt"
        atomic_json(directory / "intent.json", {
            "schema": 1, "job_id": job_id, "step": step, "spec_sha256": spec_sha256,
            "checkpoint_uri": uri, "sha256": digest.hexdigest(), "size": size,
        })
        if validate_snapshot is not None:
            validate_snapshot(snapshot)
            if snapshot.stat().st_size != size or sha256(snapshot) != digest.hexdigest():
                raise ValueError("Checkpoint snapshot changed during validation")
        artifact = self.put(snapshot, uri)
        manifest_path = directory / "manifest.json"
        atomic_json(manifest_path, {
            "schema": 1, "job_id": job_id, "step": step, "spec_sha256": spec_sha256,
            "checkpoint": asdict(artifact),
        })
        manifest_uri = prefix + sha256(manifest_path) + ".json"
        manifest = self.put(manifest_path, manifest_uri)
        atomic_json(directory / "published.json", {
            "schema": 1, "manifest": asdict(manifest), "checkpoint": asdict(artifact),
        })
        return manifest
