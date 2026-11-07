"""Explicit, bounded archive production; never archive an entire HF_HOME."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile

from so101.cloud_checkpoint import fsync_directory, sha256
from so101.aws_extract import MAX_MEMBERS


_PRIVATE = {".git", ".ssh", ".aws", ".config", ".netrc", ".gitconfig",
            "token", "stored_tokens", "credentials", "id_rsa", "id_ed25519",
            "xet", ".locks"}


@dataclass(frozen=True)
class Entry:
    name: str
    source: Path
    identity: tuple[int, int, int, int, int]
    checksum: str


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("Only regular files may be archived")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _scan(path: Path) -> tuple[tuple[int, int, int, int, int], str]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = _identity(os.fstat(stream.fileno()))
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        if _identity(os.fstat(stream.fileno())) != before:
            raise ValueError("Archive source changed while measuring")
    return before, digest.hexdigest()


def inventory(root: Path, selections: list[str], *, hf_cache: bool,
              max_bytes: int, max_members: int) -> list[Entry]:
    if (type(max_bytes) is not int or max_bytes <= 0
            or type(max_members) is not int or not 1 <= max_members <= MAX_MEMBERS
            or not selections):
        raise ValueError("Explicit selections and positive byte/member ceilings are required")
    root = root.resolve(strict=True)
    entries: dict[str, Entry] = {}
    total = 0

    def visit(path: Path, boundary: Path) -> None:
        nonlocal total
        relative = path.relative_to(root)
        if set(relative.parts) & _PRIVATE or "\\" in str(relative):
            raise ValueError("Private/auth/cache-control paths cannot be packaged")
        if path.is_symlink():
            resolved = path.resolve(strict=True)
            if not hf_cache or not resolved.is_relative_to(boundary):
                raise ValueError("Only internal regular HF cache links may be flattened")
            target_parts = resolved.relative_to(boundary).parts
            if (not target_parts or target_parts[0] not in {"refs", "snapshots", "blobs"}
                    or set(target_parts) & _PRIVATE):
                raise ValueError("HF link targets must remain inside approved cache data")
            source = resolved
        elif path.is_dir():
            for child in sorted(path.iterdir()):
                visit(child, boundary)
            return
        else:
            source = path
        identity = _identity(source.stat())
        name = relative.as_posix()
        if name in entries:
            raise ValueError("Archive selections overlap")
        if len(entries) >= max_members or identity[2] > max_bytes - total:
            raise ValueError("Flattened archive exceeds its byte/member ceiling")
        measured, checksum = _scan(source)
        if measured != identity:
            raise ValueError("Archive source changed before measurement")
        entries[name] = Entry(name, source, measured, checksum)
        total += measured[2]

    for selection in selections:
        relative = PurePosixPath(selection)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in selection
                or not relative.parts or str(relative) != selection):
            raise ValueError("Select explicit normalized relative paths, not entire roots")
        selected = root.joinpath(*relative.parts)
        if (not selected.resolve(strict=True).is_relative_to(root)
                or any(parent.is_symlink() for parent in selected.parents if parent != root
                       and parent.is_relative_to(root))):
            raise ValueError("Selection traverses a symlinked parent or escapes its root")
        if hf_cache:
            if not re.fullmatch(r"hub/models--[A-Za-z0-9_.-]+--[A-Za-z0-9_.-]+", selection):
                raise ValueError("HF selections must name explicit hub/models--ORG--REPO trees")
            if selected.is_symlink() or not selected.is_dir():
                raise ValueError("Selected HF model tree must be a real directory")
            for part in ("refs", "snapshots", "blobs"):
                child = selected / part
                if child.exists() or child.is_symlink():
                    visit(child, selected.resolve(strict=True))
        else:
            # Data roots and selections must be the operator-approved robot namespace.
            if not selected.resolve(strict=True).is_relative_to(root):
                raise ValueError("Data selection escapes its approved root")
            visit(selected, root)
    if not entries:
        raise ValueError("Archive selection contains no regular files")
    return [entries[name] for name in sorted(entries)]


def produce(root: Path, selections: list[str], destination: Path, *, hf_cache: bool,
            max_bytes: int, max_members: int) -> dict:
    entries = inventory(root, selections, hf_cache=hf_cache,
                        max_bytes=max_bytes, max_members=max_members)
    if destination.resolve().is_relative_to(root.resolve(strict=True)):
        raise ValueError("Archive output must be outside its source root")
    with destination.open("xb") as output:
        with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as bundle:
            for entry in entries:
                identity, checksum = _scan(entry.source)
                if identity != entry.identity or checksum != entry.checksum:
                    raise ValueError("Archive source changed after inventory")
                descriptor = os.open(entry.source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, "rb") as stream:
                    if _identity(os.fstat(stream.fileno())) != identity:
                        raise ValueError("Archive source changed before packing")
                    member = tarfile.TarInfo(entry.name)
                    member.size, member.mode = identity[2], 0o644
                    bundle.addfile(member, stream)
                    if _identity(os.fstat(stream.fileno())) != identity:
                        raise ValueError("Archive source changed during packing")
                if _scan(entry.source) != (identity, checksum):
                    raise ValueError("Archive source body changed during packing")
        output.flush()
        os.fsync(output.fileno())
    fsync_directory(destination.parent)
    return {"schema": 1, "archive_sha256": sha256(destination),
            "archive_bytes": destination.stat().st_size,
            "flattened_bytes": sum(entry.identity[2] for entry in entries),
            "members": len(entries),
            "files": {entry.name: {"size": entry.identity[2], "sha256": entry.checksum}
                      for entry in entries}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--include", action="append", required=True)
    parser.add_argument("--hf-cache", action="store_true")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--max-members", type=int, required=True)
    args = parser.parse_args()
    options = dict(hf_cache=args.hf_cache, max_bytes=args.max_bytes, max_members=args.max_members)
    if args.destination is None:
        entries = inventory(args.root, args.include, **options)
        result = {"flattened_bytes": sum(entry.identity[2] for entry in entries),
                  "members": len(entries)}
    else:
        result = produce(args.root, args.include, args.destination, **options)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
