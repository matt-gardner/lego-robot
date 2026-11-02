"""Verify local immutable worker image identity without pulling or running it."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess


@dataclass(frozen=True)
class RuntimeImage:
    image: str
    commit: str
    source_sha256: str

    def validate(self) -> None:
        if (not isinstance(self.image, str)
                or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9./:_-]*@sha256:[0-9a-f]{64}", self.image)):
            raise ValueError("Worker runtime image must be pinned by SHA256")
        if not isinstance(self.commit, str) or not re.fullmatch(r"[0-9a-f]{40}", self.commit):
            raise ValueError("Worker runtime must declare an immutable source commit")
        if (not isinstance(self.source_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256)):
            raise ValueError("Worker runtime must declare a source archive SHA256")

    def verify(self, docker_cli: str, *, run=subprocess.run) -> dict:
        self.validate()
        if not isinstance(docker_cli, str) or not Path(docker_cli).is_absolute():
            raise ValueError("Docker CLI must be an absolute root-installed worker path")
        result = run(
            [docker_cli, "--host", "unix:///var/run/docker.sock", "image", "inspect", self.image],
            check=True, text=True, capture_output=True, timeout=30,
            env={"PATH": "/usr/bin:/bin", "HOME": "/var/lib/so101-worker"},
        )
        images = json.loads(result.stdout)
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
            raise ValueError("Expected exactly one local runtime image")
        image = images[0]
        digests = image.get("RepoDigests")
        if not isinstance(digests, list) or self.image not in digests:
            raise ValueError("Local image does not have the pinned repository digest")
        if image.get("Os") != "linux" or image.get("Architecture") != "amd64":
            raise ValueError("Worker runtime must be Linux amd64")
        config = image.get("Config")
        labels = config.get("Labels") if isinstance(config, dict) else None
        if (not isinstance(labels, dict)
                or labels.get("org.opencontainers.image.revision") != self.commit
                or labels.get("io.so101.source-sha256") != self.source_sha256):
            raise ValueError("Runtime source identity labels differ from the root manifest")
        return {
            "image": self.image, "commit": self.commit, "source_sha256": self.source_sha256,
            "os": "linux", "architecture": "amd64",
        }
