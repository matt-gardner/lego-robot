"""One transactional current dashboard inventory and its image bytes."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_INVENTORY_BYTES = 128 * 1024 * 1024


@contextmanager
def connection(root: Path, *, writable: bool = False):
    path = root / "current.sqlite3"
    if writable:
        root.mkdir(parents=True, exist_ok=True)
    try:
        db = sqlite3.connect(path if writable else path.resolve().as_uri() + "?mode=ro",
                             uri=not writable, timeout=60)
    except sqlite3.Error as error:
        raise OSError("Current dashboard database operation failed") from error
    try:
        db.execute("PRAGMA foreign_keys=ON")
        if writable:
            db.execute("PRAGMA auto_vacuum=FULL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS inventory (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    body TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS images (
                    sha256 TEXT PRIMARY KEY, body BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS image_references (
                    sha256 TEXT PRIMARY KEY REFERENCES images(sha256)
                );
                CREATE TABLE IF NOT EXISTS problem_images (
                    problem_id TEXT NOT NULL, image_index INTEGER NOT NULL,
                    sha256 TEXT NOT NULL REFERENCES images(sha256),
                    PRIMARY KEY (problem_id, image_index)
                );
            """)
        yield db
        if writable:
            db.commit()
    except sqlite3.Error as error:
        db.rollback()
        raise OSError("Current dashboard database operation failed") from error
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def inventory(db: sqlite3.Connection) -> dict:
    row = db.execute("SELECT body FROM inventory WHERE singleton=1").fetchone()
    if row is None:
        raise ValueError("Current dashboard inventory has not been initialized")
    return json.loads(row[0])


def current_inventory(root: Path) -> dict:
    with connection(root) as db:
        return inventory(db)


def current_response(root: Path) -> bytes:
    with connection(root) as db:
        row = db.execute("SELECT body FROM inventory WHERE singleton=1").fetchone()
        if row is None:
            raise ValueError("Current dashboard inventory has not been initialized")
        return row[0].encode()


def current_manifest(root: Path, channel: str) -> dict | None:
    if not (root / "current.sqlite3").exists():
        return None
    document = current_inventory(root)
    matches = [row for row in document["manifest"]["datasets"]
               if row.get("current_channel") == channel or row.get("dataset_id") == channel]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Multiple populations occupy one current dashboard channel")
    manifest = deepcopy(matches[0])
    revision = manifest["snapshot_id"]
    manifest["groups"] = [
        {k: v for k, v in row.items() if k != "dataset_snapshot_id"}
        for row in document["calibration"]["groups"]
        if row.get("dataset_snapshot_id") == revision
    ]
    return manifest


def _task_records(document: dict):
    yield from document["problems"]["problems"]
    for pilot in document.get("distance_pilots", []):
        yield from pilot["problems"]


def _records(document: dict):
    yield from _task_records(document)
    yield from document.get("admitted_piles", [])


def _image_bytes(path: Path, expected: str) -> bytes:
    with path.open("rb") as stream:
        body = stream.read(MAX_IMAGE_BYTES + 1)
    if len(body) > MAX_IMAGE_BYTES:
        raise ValueError("Dashboard image exceeds read limit")
    if hashlib.sha256(body).hexdigest() != expected:
        raise ValueError("Dashboard image hash mismatch")
    if not body.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")):
        raise ValueError("Dashboard image must be PNG or JPEG")
    return body


def _save(db: sqlite3.Connection, document: dict, root: Path) -> None:
    identities = [row["id"] for row in document["problems"]["problems"]]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate current dashboard problem")
    references = set()
    lookups = {}
    root = root.resolve()
    for record in _task_records(document):
        record["initial_images"] = []
        record.pop("initial_images_status", None)
    for record in _records(document):
        for index, image in enumerate(record.get("initial_images", [])):
            expected = image["sha256"]
            if (not isinstance(expected, str) or len(expected) != 64
                    or any(c not in "0123456789abcdef" for c in expected)):
                raise ValueError("Invalid dashboard image digest")
            stored = db.execute("SELECT 1 FROM images WHERE sha256=?", (expected,)).fetchone()
            if stored is None:
                relative = Path(image["path"])
                if not image.get("root_relative"):
                    relative = Path(record.get("snapshot_directory", "")) / relative
                path = (root / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(root):
                    raise ValueError("Dashboard image path escapes import root")
                body = _image_bytes(path, expected)
                db.execute("INSERT INTO images VALUES (?,?)", (expected, body))
            image["path"] = "images/" + expected
            image["root_relative"] = True
            references.add(expected)
            key = record["id"], index
            if key in lookups and lookups[key] != expected:
                raise ValueError("Conflicting current dashboard image identity")
            lookups[key] = expected
        record.pop("snapshot_directory", None)
    body = json.dumps(document, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if len(body.encode()) > MAX_INVENTORY_BYTES:
        raise ValueError("Current dashboard inventory exceeds byte limit")
    db.execute("INSERT OR REPLACE INTO inventory VALUES (1,?)", (body,))
    db.execute("DELETE FROM problem_images")
    db.execute("DELETE FROM image_references")
    db.executemany("INSERT INTO image_references VALUES (?)",
                   ((sha,) for sha in sorted(references)))
    db.executemany("INSERT INTO problem_images VALUES (?,?,?)",
                   ((identity, index, sha) for (identity, index), sha in lookups.items()))
    db.execute("DELETE FROM images WHERE sha256 NOT IN (SELECT sha256 FROM image_references)")


def import_current(root: Path, document: dict) -> Path:
    """Import a checked complete inventory; never merges historical generations."""
    with connection(root, writable=True) as db:
        db.execute("BEGIN IMMEDIATE")
        _save(db, deepcopy(document), root)
    return root / "current.sqlite3"


def current_image(root: Path, problem_id: str, index: int) -> tuple[bytes, str]:
    if index < 0:
        raise ValueError("Invalid dashboard image index")
    with connection(root) as db:
        row = db.execute(
            "SELECT images.sha256, images.body FROM problem_images "
            "JOIN images USING (sha256) WHERE problem_id=? AND image_index=?",
            (problem_id, index),
        ).fetchone()
        if row is None:
            raise ValueError("Unknown dashboard problem/image")
        expected, body = row
        if hashlib.sha256(body).hexdigest() != expected:
            raise ValueError("Current dashboard image hash mismatch")
        return body, "image/png" if body.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg"


def publish_current(root: Path, manifest: dict, asset_directory: Path,
                    channel: str = "native-piles") -> Path:
    """Replace one current population without changing unrelated measurements."""
    if manifest.get("kind") != "lego-data-buckets" or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported current dashboard population")
    if "problem_shards" in manifest:
        raise ValueError("Current writer requires decoded problem rows")
    with connection(root, writable=True) as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM inventory WHERE singleton=1").fetchone():
            document = inventory(db)
        else:
            document = {
                "manifest": {"datasets": []},
                "problems": {"problems": [], "buckets": []},
                "admitted_piles": [], "distance_pilots": [],
                "calibration": {"groups": [], "protocol": {}, "complete": False,
                                "qualification_status": "unmeasured"},
            }
        datasets = document["manifest"]["datasets"]
        if channel == "native-piles":
            selected = lambda row: "published_pile_snapshot_sha256" in row.get("provenance", {})
        else:
            pool = manifest.get("provenance", {}).get("native_pool_sha256")
            selected = lambda row: (
                row.get("current_channel") == channel
                or (pool is not None
                    and row.get("provenance", {}).get("native_pool_sha256") == pool)
            )
        previous = [row for row in datasets if selected(row)]
        if len(previous) > 1:
            raise ValueError("Multiple populations occupy one current dashboard channel")
        old_ids = {row["snapshot_id"] for row in previous}
        revision = manifest["snapshot_id"]
        metadata = {k: v for k, v in manifest.items()
                    if k not in ("problems", "groups", "buckets", "plan")}
        metadata["current_channel"] = channel
        document["manifest"]["datasets"] = [row for row in datasets if not selected(row)] + [metadata]
        if len(document["manifest"]["datasets"]) == 1:
            document["manifest"].update(deepcopy(metadata))
        problems = document["problems"]
        removed = {row["id"] for row in problems["problems"]
                   if row.get("dataset_snapshot_id") in old_ids}
        for key in ("problems",):
            problems[key] = [row for row in problems[key]
                             if row.get("dataset_snapshot_id") not in old_ids]
        relative = asset_directory.resolve().relative_to(root.resolve())
        incoming = [{**deepcopy(row), "bucket": row.get("bucket_id", row.get("bucket")),
                     "dataset_snapshot_id": revision, "snapshot_directory": str(relative)}
                    for row in manifest["problems"]]
        problems["problems"].extend(incoming)
        problems["buckets"] = [
            row for row in problems["buckets"]
            if not any(identity in removed for task in row.get("tasks", [])
                       for identity in task.get("problem_ids", []))
        ] + deepcopy(manifest.get("buckets", []))
        document["admitted_piles"] = [
            row for row in document.get("admitted_piles", [])
            if row.get("dataset_snapshot_id") not in old_ids
        ] + [{**deepcopy(row), "dataset_snapshot_id": revision,
              "snapshot_directory": str(relative)}
             for row in manifest.get("admitted_piles", [])]
        calibration = document["calibration"]
        calibration["groups"] = [row for row in calibration["groups"]
                                 if row.get("dataset_snapshot_id") not in old_ids]
        groups = manifest.get("groups", [])
        if manifest.get("measurement_series"):
            groups = [{**deepcopy(row), "measurement_series_id": series["series_id"],
                       "protocol": series["protocol"]}
                      for series in manifest["measurement_series"] for row in series["groups"]]
        calibration["groups"].extend({**deepcopy(row), "dataset_snapshot_id": revision}
                                     for row in groups)
        if len(document["manifest"]["datasets"]) == 1:
            calibration.update(protocol=manifest.get("protocol", {}),
                               complete=manifest.get("complete", False),
                               qualification_status=manifest.get("measurement_status", "unmeasured"))
        document["manifest"]["snapshot_id"] = hashlib.sha256(
            json.dumps(document["manifest"]["datasets"], sort_keys=True).encode()).hexdigest()
        _save(db, document, root)
    return root / "current.sqlite3"
