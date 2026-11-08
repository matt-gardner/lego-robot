"""Byte-preserving, read-only indexed episode container (format version 1)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading


FORMAT_VERSION = 1


class EpisodeStorage:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self._local = threading.local()
        if self.connection.execute("PRAGMA user_version").fetchone()[0] != FORMAT_VERSION:
            self.close()
            raise ValueError(f"Unsupported episode storage version: {path}")
        self.catalog = json.loads(self.connection.execute(
            "SELECT content FROM catalog"
        ).fetchone()[0])

    @property
    def connection(self) -> sqlite3.Connection:
        if getattr(self._local, "pid", None) != os.getpid():
            self._local.connection = sqlite3.connect(
                self.path.as_uri() + "?mode=ro&immutable=1", uri=True
            )
            self._local.connection.execute("PRAGMA cache_size=-16384")
            self._local.pid = os.getpid()
        return self._local.connection

    def __getstate__(self) -> dict:
        return {"path": self.path}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["path"])

    def read(self, source: Path, relative: str) -> bytes:
        row = self.connection.execute(
            "SELECT content FROM files WHERE source=? AND name=?",
            (str(source), relative),
        ).fetchone()
        if row is None:
            raise FileNotFoundError(f"{source}/{relative} in {self.path}")
        return row[0]

    def verify(self) -> dict[str, int]:
        if self.connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError(f"Corrupt episode container: {self.path}")
        count = size = 0
        for source, name, content, digest in self.connection.execute(
            "SELECT source,name,content,sha256 FROM files ORDER BY source,name"
        ):
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Content checksum mismatch: {source}/{name}")
            count += 1
            size += len(content)
        return {"files": count, "bytes": size}

    def verify_sources(self, collection: Path) -> None:
        catalog_bytes = self.connection.execute("SELECT content FROM catalog").fetchone()[0]
        if (collection / "collection.json").read_bytes() != catalog_bytes:
            raise ValueError("Catalog differs from original")
        for source in self.catalog["successful_episodes"]:
            expected = {
                row[0] for row in self.connection.execute(
                    "SELECT name FROM files WHERE source=?", (str(Path(source)),)
                )
            }
            actual = set()
            for file in Path(source).rglob("*"):
                if file.is_symlink():
                    raise ValueError(f"Symlink in episode: {file}")
                if file.is_file():
                    relative = file.relative_to(source).as_posix()
                    actual.add(relative)
                    if file.read_bytes() != self.read(Path(source), relative):
                        raise ValueError(f"Original differs: {file}")
            if actual != expected:
                raise ValueError(f"Original file population differs: {source}")

    def close(self) -> None:
        self.connection.close()
        del self._local.connection
        del self._local.pid

    def __enter__(self) -> EpisodeStorage:
        return self

    def __exit__(self, *args) -> None:
        self.close()


def consolidate(collection: Path, destination: Path) -> dict[str, int]:
    """Publish a new container exclusively, after rereading all stored bytes."""
    if destination.exists():
        raise FileExistsError(destination)
    catalog_bytes = (collection / "collection.json").read_bytes()
    catalog = json.loads(catalog_bytes)
    if catalog.get("status", "complete") != "complete":
        raise ValueError(f"Simulation dataset is not complete: {collection}")
    sources = catalog["successful_episodes"]
    if not sources or len({str(Path(source)) for source in sources}) != len(sources):
        raise ValueError("Catalog must contain distinct successful episodes")
    for source in sources:
        if destination.resolve().is_relative_to(Path(source).resolve()):
            raise ValueError("Destination must be outside original episode directories")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".episode-storage-", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with sqlite3.connect(temporary) as connection:
            connection.executescript("""
                PRAGMA journal_mode=DELETE;
                CREATE TABLE catalog(content BLOB NOT NULL);
                CREATE TABLE files(
                    source TEXT NOT NULL, name TEXT NOT NULL,
                    content BLOB NOT NULL, sha256 TEXT NOT NULL,
                    PRIMARY KEY(source,name)
                );
            """)
            connection.execute(f"PRAGMA user_version={FORMAT_VERSION}")
            connection.execute("INSERT INTO catalog VALUES (?)", (catalog_bytes,))
            for source_name in sources:
                source = Path(source_name)
                if source.is_symlink():
                    raise ValueError(f"Symlink episode directory: {source}")
                metadata = (source / "episode.json").read_bytes()
                rows = (source / "frames.jsonl").read_bytes()
                json.loads(metadata)
                for line in rows.splitlines():
                    image = Path(json.loads(line)["image"])
                    if image.is_absolute() or ".." in image.parts:
                        raise ValueError(f"Unsafe image path: {image}")
                    if not (source / image).is_file():
                        raise FileNotFoundError(source / image)
                for file in sorted(source.rglob("*")):
                    if file.is_symlink():
                        raise ValueError(f"Symlink in episode: {file}")
                    if not file.is_file():
                        continue
                    content = file.read_bytes()
                    if file.name == "episode.json" and file.parent == source and content != metadata:
                        raise ValueError(f"Episode changed during migration: {source}")
                    if file.name == "frames.jsonl" and file.parent == source and content != rows:
                        raise ValueError(f"Frames changed during migration: {source}")
                    connection.execute("INSERT INTO files VALUES (?,?,?,?)", (
                        str(source), file.relative_to(source).as_posix(),
                        content, hashlib.sha256(content).hexdigest(),
                    ))
        with EpisodeStorage(temporary) as storage:
            result = storage.verify()
            storage.verify_sources(collection)
        if (collection / "collection.json").read_bytes() != catalog_bytes:
            raise ValueError("Catalog changed during migration")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return {**result, "episodes": len(sources), "container_bytes": destination.stat().st_size}
    finally:
        temporary.unlink(missing_ok=True)
