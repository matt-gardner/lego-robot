"""Bounded extraction into new read-only input copies; never touches original trees."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import time

from so101.cloud_checkpoint import atomic_json, fsync_directory, sha256


MAX_MEMBERS = 200_000


def extract(archive: Path, destination: Path, *, max_bytes: int, max_members: int,
            deadline: float | None = None, wall=time.time) -> dict:
    if (type(max_bytes) is not int or max_bytes <= 0
            or type(max_members) is not int or not 1 <= max_members <= MAX_MEMBERS):
        raise ValueError("Archive preparation requires explicit byte/member ceilings")
    def check_deadline():
        if deadline is not None and wall() >= deadline:
            raise TimeoutError("Worker lease expired during archive extraction")
    check_deadline()
    descriptor = os.open(archive, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        initial = os.fstat(source.fileno())
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError("Input archive must be a regular non-symlink file")
        digest = sha256(archive)
        destination.mkdir(mode=0o700)
        count, total = 0, 0
        files = {}
        with tarfile.open(fileobj=source, mode="r|*") as bundle:
            for member in bundle:
                check_deadline()
                count += 1
                if count == 1 or count % 10000 == 0:
                    logging.warning("Archive preparation members=%d bytes=%d archive=%s",
                                    count, total, archive.name)
                if count > max_members:
                    raise ValueError("Archive member ceiling exceeded")
                relative = PurePosixPath(member.name)
                if (relative.is_absolute() or ".." in relative.parts or "\\" in member.name
                        or str(relative) == ".extracted.json"
                        or member.issparse() or not (member.isfile() or member.isdir())):
                    raise ValueError("Archive contains traversal, links, sparse data or special files")
                if not relative.parts:
                    if member.isdir():
                        continue
                    raise ValueError("Archive file has no relative name")
                target = destination.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(mode=0o755, parents=True, exist_ok=True)
                    continue
                if member.size < 0 or member.size > max_bytes - total:
                    raise ValueError("Archive extracted byte ceiling exceeded")
                target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError("Archive regular file has no readable body")
                size, checksum = 0, hashlib.sha256()
                with stream, target.open("xb") as output:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        check_deadline()
                        size += len(chunk)
                        if size > member.size:
                            raise ValueError("Archive file exceeds its declared size")
                        checksum.update(chunk)
                        output.write(chunk)
                    if size != member.size:
                        raise ValueError("Archive file is truncated")
                    output.flush()
                    os.fsync(output.fileno())
                target.chmod(0o644)
                files[str(relative)] = {"size": size, "sha256": checksum.hexdigest()}
                total += size
        final = os.fstat(source.fileno())
        if (initial.st_size != final.st_size or initial.st_mtime_ns != final.st_mtime_ns
                or initial.st_ctime_ns != final.st_ctime_ns or sha256(archive) != digest):
            raise ValueError("Input archive changed during extraction")
    for directory in destination.rglob("*"):
        check_deadline()
        if directory.is_dir():
            directory.chmod(0o755)
            fsync_directory(directory)
    receipt = {"schema": 1, "archive_sha256": digest, "members": count,
               "bytes": total, "files": files}
    check_deadline()
    atomic_json(destination / ".extracted.json", receipt)
    destination.chmod(0o755)
    fsync_directory(destination)
    fsync_directory(destination.parent)
    return receipt
