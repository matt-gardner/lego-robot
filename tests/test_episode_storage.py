import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
from pathlib import Path
import sqlite3
import threading

import pytest

from so101.episode_storage import EpisodeStorage, consolidate


def collection(tmp_path):
    source = tmp_path / "episode"
    (source / "wrist").mkdir(parents=True)
    (source / "episode.json").write_bytes(b'{ "initial": {}, "task": "grasp" }\n')
    (source / "frames.jsonl").write_bytes(b'{"image":"wrist/0.jpg","phase":"approach"}\n')
    (source / "wrist/0.jpg").write_bytes(b"\x00original jpeg bytes\xff")
    root = tmp_path / "collection"
    root.mkdir()
    (root / "collection.json").write_text(json.dumps({
        "successful_episodes": [str(source)], "status": "complete",
    }))
    return root, source


def test_exact_bytes_readonly_and_exclusive_publish(tmp_path):
    root, source = collection(tmp_path)
    original = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in tmp_path.rglob("*") if p.is_file()}
    destination = tmp_path / "v1.sqlite"
    result = consolidate(root, destination)
    assert result["files"] == 3
    with EpisodeStorage(destination) as storage:
        assert storage.catalog == json.loads((root / "collection.json").read_bytes())
        for relative in ("episode.json", "frames.jsonl", "wrist/0.jpg"):
            assert storage.read(source, relative) == (source / relative).read_bytes()
        assert storage.verify()["files"] == 3
        storage.verify_sources(root)
        restored = pickle.loads(pickle.dumps(storage))
        assert restored.read(source, "episode.json") == storage.read(source, "episode.json")
        restored.close()
        with pytest.raises(sqlite3.OperationalError):
            storage.connection.execute("DELETE FROM files")
        with pytest.raises(FileNotFoundError):
            storage.read(source, "missing")
    with pytest.raises(FileExistsError):
        consolidate(root, destination)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
               for p, digest in original.items())


def test_failed_migration_never_publishes(tmp_path):
    root, source = collection(tmp_path)
    (source / "frames.jsonl").write_text('{"image":"../escape.jpg"}\n')
    destination = tmp_path / "v1.sqlite"
    with pytest.raises(ValueError, match="Unsafe"):
        consolidate(root, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".episode-storage-*"))


def test_corruption_detected(tmp_path):
    root, _ = collection(tmp_path)
    destination = tmp_path / "v1.sqlite"
    consolidate(root, destination)
    with sqlite3.connect(destination) as connection:
        connection.execute(
            "UPDATE blobs SET content=x'00' WHERE sha256="
            "(SELECT sha256 FROM files WHERE name='episode.json')"
        )
    with EpisodeStorage(destination) as storage:
        with pytest.raises(ValueError, match="checksum"):
            storage.verify()


def test_original_changes_detected(tmp_path):
    root, source = collection(tmp_path)
    destination = tmp_path / "v1.sqlite"
    consolidate(root, destination)
    (source / "wrist/0.jpg").write_bytes(b"changed")
    with EpisodeStorage(destination) as storage:
        with pytest.raises(ValueError, match="Original differs"):
            storage.verify_sources(root)


def test_shared_blobs_preserve_distinct_logical_files(tmp_path):
    root, source = collection(tmp_path)
    shared = b"compiled model bytes" * 65536
    (source / "model.mjb").write_bytes(shared)
    (source / "other-model.mjb").write_bytes(shared)
    destination = tmp_path / "v2.sqlite"
    result = consolidate(root, destination)
    with EpisodeStorage(destination) as storage:
        assert storage.format_version == 2
        assert result["files"] == 5
        assert result["bytes"] == sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
        assert storage.connection.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 4
        assert storage.read(source, "model.mjb") == shared
        assert storage.read(source, "other-model.mjb") == shared
        assert storage.connection.execute(
            "SELECT content FROM files WHERE name='model.mjb'"
        ).fetchone()[0] == shared
        metadata = storage.connection.execute(
            "SELECT source,content,sha256 FROM files WHERE name='episode.json' ORDER BY source"
        ).fetchall()
        assert metadata == [(str(source), (source / "episode.json").read_bytes(),
                             hashlib.sha256((source / "episode.json").read_bytes()).hexdigest())]
        frames = storage.connection.execute(
            "SELECT content,sha256 FROM files WHERE source=? AND name='frames.jsonl'",
            (str(source),),
        ).fetchone()
        assert frames[0] == (source / "frames.jsonl").read_bytes()
        assert hashlib.sha256(frames[0]).hexdigest() == frames[1]
    assert destination.stat().st_size < len(shared) * 1.1


def test_format1_remains_readable(tmp_path):
    root, source = collection(tmp_path)
    destination = tmp_path / "historical.sqlite"
    with sqlite3.connect(destination) as connection:
        connection.executescript("""
            PRAGMA user_version=1;
            CREATE TABLE catalog(content BLOB NOT NULL);
            CREATE TABLE files(
                source TEXT, name TEXT, content BLOB, sha256 TEXT,
                PRIMARY KEY(source,name)
            );
        """)
        connection.execute("INSERT INTO catalog VALUES (?)", ((root / "collection.json").read_bytes(),))
        for file in source.rglob("*"):
            if file.is_file():
                content = file.read_bytes()
                connection.execute("INSERT INTO files VALUES (?,?,?,?)", (
                    str(source), file.relative_to(source).as_posix(),
                    content, hashlib.sha256(content).hexdigest(),
                ))
    with EpisodeStorage(destination) as storage:
        assert storage.format_version == 1
        assert storage.read(source, "wrist/0.jpg") == (source / "wrist/0.jpg").read_bytes()
        assert storage.verify()["files"] == 3
        storage.verify_sources(root)


def test_missing_blob_never_silently_reduces_verified_population(tmp_path):
    root, _ = collection(tmp_path)
    destination = tmp_path / "v2.sqlite"
    consolidate(root, destination)
    with sqlite3.connect(destination) as connection:
        connection.execute(
            "DELETE FROM blobs WHERE sha256=(SELECT sha256 FROM files WHERE name='episode.json')"
        )
    with EpisodeStorage(destination) as storage:
        with pytest.raises(ValueError, match="Missing episode content"):
            storage.verify()


def test_destination_cannot_enter_original_episode(tmp_path):
    root, source = collection(tmp_path)
    with pytest.raises(ValueError, match="outside original"):
        consolidate(root, source / "v1.sqlite")
    assert not list(source.glob(".episode-storage-*"))


def test_worker_connections_read_concurrently_and_close_independently(tmp_path):
    root, source = collection(tmp_path)
    destination = tmp_path / "v1.sqlite"
    consolidate(root, destination)
    barrier = threading.Barrier(2)
    with EpisodeStorage(destination) as storage:
        expected = (source / "wrist/0.jpg").read_bytes()

        def read_window(_):
            barrier.wait(timeout=5)
            contents = [storage.read(source, "wrist/0.jpg") for _ in range(20)]
            storage.close()
            return contents

        with ThreadPoolExecutor(max_workers=2) as workers:
            for contents in workers.map(read_window, range(2)):
                assert contents == [expected] * 20
        assert storage.read(source, "wrist/0.jpg") == expected
