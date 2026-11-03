"""Offline immutable source export and hashed Linux CUDA wheel inventory."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile


def wheel_lock(directory: Path) -> str:
    wheels = sorted(directory.iterdir())
    if not wheels or any(not path.is_file() or path.is_symlink()
                         or path.suffix != ".whl" for path in wheels):
        raise ValueError("Wheelhouse must contain only regular, non-symlink wheels")
    packages = {}
    for path in wheels:
        if not re.fullmatch(
            r".+-(?:cp312-cp312|(?:cp3[2-9]|cp31[0-2])-abi3|"
            r"(?:py3|py312|py2\.py3)-none)-"
            r"(?:any|manylinux[^/]*x86_64|linux_x86_64)\.whl", path.name,
        ):
            raise ValueError(f"Not a Linux amd64 or portable wheel: {path.name}")
        with zipfile.ZipFile(path) as archive:
            metadata = [name for name in archive.namelist()
                        if name.count("/") == 1 and name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError(f"Wheel must have exactly one package metadata record: {path.name}")
            record = BytesParser().parsebytes(archive.read(metadata[0]))
        name, version = record.get("Name", ""), record.get("Version", "")
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version)):
            raise ValueError(f"Invalid wheel package identity: {path.name}")
        name = re.sub(r"[-_.]+", "-", name).lower()
        if name in packages:
            raise ValueError(f"Duplicate wheel package: {name}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        packages[name] = (version, digest)
    for name, version in {"torch": "2.11.0+cu130", "torchvision": "0.26.0+cu130"}.items():
        if packages.get(name, (None,))[0] != version:
            raise ValueError(f"Runtime requires {name}=={version}")
    return "".join(f"{name}=={version} --hash=sha256:{digest}\n"
                   for name, (version, digest) in sorted(packages.items()))


def prepare(repo: Path, commit: str, destination: Path, wheelhouse: Path) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Build source must name a full immutable commit SHA")
    lock = wheel_lock(wheelhouse)
    resolved = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", commit + "^{commit}"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if resolved != commit:
        raise ValueError("Commit identity mismatch")
    # Export only tracked runtime inputs, never the mutable checkout or local data.
    source = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", commit,
         "src/so101", "pyproject.toml", "uv.lock"],
        check=True, capture_output=True,
    ).stdout
    dockerfile = subprocess.run(
        ["git", "-C", str(repo), "show", commit + ":scripts/Dockerfile.gpu-worker"],
        check=True, capture_output=True,
    ).stdout
    destination.mkdir(mode=0o700)
    (destination / "source.tar").write_bytes(source)
    (destination / "Dockerfile").write_bytes(dockerfile)
    (destination / "requirements.wheels.txt").write_text(lock)
    shutil.copytree(wheelhouse, destination / "wheelhouse")
    if wheel_lock(destination / "wheelhouse") != lock:
        raise ValueError("Wheelhouse changed while preparing build inputs")
    manifest = {
        "schema": 1, "commit": commit,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "dockerfile_sha256": hashlib.sha256(dockerfile).hexdigest(),
        "wheel_lock_sha256": hashlib.sha256(lock.encode()).hexdigest(),
        "phase": "offline-inputs-not-qualified",
    }
    (destination / "build-inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.repo, args.commit, args.destination, args.wheelhouse)))


if __name__ == "__main__":
    main()
