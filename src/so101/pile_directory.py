"""Canonical pile/task files; the dashboard is a read-only projection of these files."""
import hashlib
import io
import json
from pathlib import Path
import re

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
PILE_NAME = re.compile(r"pile_[0-9]{4,}")
TASK_NAME = re.compile(r"task_[0-9]{3,}\.json")
CLIENT_TASK_FIELDS = (
    "id", "task", "bucket", "bucket_id", "source", "tags", "feasibility",
    "measurement_status", "measurement_scope", "dataset_snapshot_id", "problem_index",
)


def contained(root: Path, path: Path) -> Path:
    current = root
    if root.is_symlink():
        raise ValueError("Canonical data root must not be a symlink")
    for component in path.relative_to(root).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("Canonical data path contains a symlink")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Canonical data path escapes its root or is a symlink")
    return path


def read_json(path: Path) -> dict:
    with path.open("rb") as stream:
        body = stream.read(MAX_JSON_BYTES + 1)
    if len(body) > MAX_JSON_BYTES:
        raise ValueError("Canonical JSON file exceeds byte limit")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("Canonical JSON record must be an object")
    return value


def _piles(root: Path):
    directory = root / "piles"
    if (directory / ".incomplete").exists():
        raise ValueError("Canonical pile import is incomplete")
    for path in sorted(directory.iterdir()):
        if PILE_NAME.fullmatch(path.name):
            contained(root, path)
            if not path.is_dir():
                raise ValueError("Canonical pile entry must be a directory")
            yield path


def _client_pile(directory: Path, pile: dict) -> dict:
    result = {
        "id": directory.name, "scene_id": pile["scene_id"], "tags": pile["tags"],
        "status": pile["qualification"], "image_kind": "admitted-pile-camera",
        "initial_images": [], **pile.get("piece_coordinates", {}),
    }
    image = pile.get("rendered")
    if image is not None:
        result["initial_images"] = [{**image, "path": "rendered.png"}]
    return result


def current_response(root: Path) -> bytes:
    """Stream task records into one response without retaining a decoded population."""
    metadata = read_json(contained(root, root / "piles" / "metadata.json"))
    if metadata["schema_version"] != 1:
        raise ValueError("Unsupported canonical pile schema")
    directories = list(_piles(root))
    fingerprint = hashlib.sha256()
    piles = []
    output = io.BytesIO()

    def emit(value):
        output.write(json.dumps(value, separators=(",", ":"), allow_nan=False).encode())

    output.write(b'{"problems":{"problems":[')
    first = True
    identities = set()
    for directory in directories:
        pile_path = contained(root, directory / "pile.json")
        before = pile_path.stat()
        pile = read_json(pile_path)
        if pile["id"] != directory.name:
            raise ValueError("Pile identity differs from directory")
        piles.append(_client_pile(directory, pile))
        fingerprint.update(pile_path.read_bytes())
        for path in sorted((directory / "tasks").iterdir()):
            if not TASK_NAME.fullmatch(path.name):
                continue
            contained(root, path)
            stat = path.stat()
            fingerprint.update(f"{directory.name}/{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
            task = read_json(path)
            if task["schema_version"] != 1:
                raise ValueError("Unsupported canonical task schema")
            if task["id"] in identities:
                raise ValueError("Duplicate canonical task identity")
            identities.add(task["id"])
            record = {key: task[key] for key in CLIENT_TASK_FIELDS if key in task}
            record["initial_images"] = []
            if not first:
                output.write(b",")
            emit(record)
            first = False
        after = pile_path.stat()
        if (before.st_ino, before.st_mtime_ns) != (after.st_ino, after.st_mtime_ns):
            raise ValueError("Pile metadata changed during read")
    output.write(b'],"buckets":')
    emit(metadata["buckets"])
    output.write(b'},"admitted_piles":')
    emit(piles)
    output.write(b',"calibration":')
    emit(metadata["calibration"])
    output.write(b',"distance_pilots":')
    emit(metadata["distance_pilots"])
    manifest = metadata["manifest"]
    fingerprint.update(json.dumps(metadata, sort_keys=True).encode())
    manifest["snapshot_id"] = fingerprint.hexdigest()
    output.write(b',"manifest":')
    emit(manifest)
    output.write(b"}")
    return output.getvalue()


def current_inventory(root: Path) -> dict:
    return json.loads(current_response(root))


def current_image(root: Path, pile_id: str, index: int) -> tuple[bytes, str]:
    if index != 0 or not PILE_NAME.fullmatch(pile_id):
        raise ValueError("Unknown canonical pile/image")
    directory = contained(root, root / "piles" / pile_id)
    pile = read_json(contained(root, directory / "pile.json"))
    image = pile.get("rendered")
    if image is None:
        raise ValueError("Pile image has not been rendered")
    path = contained(root, directory / "rendered.png")
    with path.open("rb") as stream:
        body = stream.read(MAX_IMAGE_BYTES + 1)
    if (len(body) > MAX_IMAGE_BYTES or not body.startswith(b"\x89PNG\r\n\x1a\n")
            or hashlib.sha256(body).hexdigest() != image["sha256"]):
        raise ValueError("Canonical pile image size, format or digest differs")
    return body, "image/png"
