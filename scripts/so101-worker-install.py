"""Run with /usr/bin/python3 -I after independently verifying this launcher's SHA256."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tarfile


def unpack(archive: Path, expected_sha256: str, worker: Path) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Source archive requires a pinned SHA256")
    with archive.open("rb") as stream:
        initial = os.fstat(stream.fileno())
        if (not stat.S_ISREG(initial.st_mode) or initial.st_size > 64 * 1024**2
                or hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256):
            raise ValueError("Source archive differs from its pinned bytes")
        stream.seek(0)
        worker.mkdir(mode=0o755)
        source = worker / "source"
        source.mkdir(mode=0o755)
        with tarfile.open(fileobj=stream, mode="r:") as bundle:
            members = bundle.getmembers()
            if len(members) > 10_000 or sum(member.size for member in members) > 32 * 1024**2:
                raise ValueError("Source bundle exceeds its bounded code-only size")
            for member in members:
                path = Path(member.name)
                if (path.is_absolute() or ".." in path.parts or "\\" in member.name
                        or not (member.isfile() or member.isdir()) or member.issparse()
                        or not (member.name.startswith(("src/so101/", "scripts/"))
                                or member.name in {"src", "src/so101", "scripts",
                                                       "pyproject.toml", "uv.lock"})):
                    raise ValueError("Source bundle contains unsupported non-code members")
            bundle.extractall(source, members=members, filter="data")
        stream.seek(0)
        final = os.fstat(stream.fileno())
        if (any(getattr(initial, field) != getattr(final, field) for field in
                ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))
                or hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256):
            raise ValueError("Source archive changed during bootstrap")
    for path in source.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--execution-sha256", required=True)
    args = parser.parse_args()
    if os.geteuid() != 0 or sys.version_info < (3, 12):
        raise PermissionError("Fresh GPU bootstrap requires root and AMI Python >=3.12")
    archive = args.source_archive
    for candidate in (archive, *archive.parents):
        info = candidate.lstat()
        if info.st_uid != 0 or info.st_mode & 0o022 or stat.S_ISLNK(info.st_mode):
            raise ValueError("Bootstrap archive and parents must be private root-trusted paths")
    for candidate in (Path("/opt"), Path("/")):
        info = candidate.lstat()
        if info.st_uid != 0 or info.st_mode & 0o022 or not stat.S_ISDIR(info.st_mode):
            raise ValueError("Worker installation parents must be root-trusted directories")
    source = unpack(archive, args.source_sha256, Path("/opt/so101-worker"))
    sys.path.insert(0, str(source / "src"))
    # Import root code only after its complete pinned archive has verified.
    from so101.aws_worker_install import install
    print(json.dumps(install(args.execution_sha256, args.source_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
