"""Read-only, hash-pinned learning bucket publications."""
import hashlib
import json
from pathlib import Path


class BucketDashboard:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def read(self, relative: str, expected: str | None = None, limit: int = 32 * 1024 * 1024) -> bytes:
        path = (self.root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(self.root):
            raise ValueError("Bucket artifact path escapes publication root")
        with path.open("rb") as stream:
            body = stream.read(limit + 1)
        if len(body) > limit:
            raise ValueError("Bucket artifact exceeds read limit")
        if expected is not None and hashlib.sha256(body).hexdigest() != expected:
            raise ValueError("Bucket artifact hash mismatch")
        return body

    def snapshot(self) -> dict:
        pointer = json.loads(self.read("current.json", limit=1024 * 1024))
        relative = pointer["path"]
        manifest = json.loads(self.read(relative, pointer["sha256"]))
        if manifest.get("schema_version") != 1 or manifest.get("kind") != "lego-data-buckets":
            raise ValueError("Unsupported bucket manifest")
        if not isinstance(manifest.get("problems"), list):
            raise ValueError("Bucket problems population is missing")
        if not isinstance(manifest.get("groups"), list):
            raise ValueError("Bucket measurement groups are missing")
        problems = []
        identities = set()
        for problem in manifest["problems"]:
            identity = problem["id"]
            if not isinstance(identity, str) or identity in identities:
                raise ValueError("Bucket problem identity missing or duplicated")
            identities.add(identity)
            problems.append({**problem, "bucket": problem.get("bucket_id", problem.get("bucket"))})
        return {
            "manifest": {key: value for key, value in manifest.items()
                         if key not in ("problems", "groups", "buckets", "plan")},
            "problems": {"problems": problems, "buckets": manifest.get("buckets", [])},
            "calibration": {"groups": manifest["groups"], "protocol": manifest.get("protocol", {}),
                            "complete": manifest.get("complete", False),
                            "qualification_status": manifest.get("measurement_status", "unknown")},
            "snapshot_directory": str(Path(relative).parent),
        }

    def image(self, problem_id: str, index: int, snapshot_id: str | None = None) -> tuple[bytes, str]:
        snapshot = self.snapshot()
        if snapshot_id is not None and snapshot["manifest"].get("snapshot_id") != snapshot_id:
            raise ValueError("Bucket publication changed; refresh the page")
        problems = [row for row in snapshot["problems"]["problems"] if row["id"] == problem_id]
        if len(problems) != 1 or index < 0:
            raise ValueError("Unknown bucket problem/image")
        images = problems[0].get("initial_images", [])
        if index >= len(images):
            raise ValueError("Unknown bucket problem/image")
        image = images[index]
        relative = Path(snapshot["snapshot_directory"]) / image["path"]
        body = self.read(str(relative), image["sha256"], limit=8 * 1024 * 1024)
        if body.startswith(b"\x89PNG\r\n\x1a\n"):
            return body, "image/png"
        if body.startswith(b"\xff\xd8\xff"):
            return body, "image/jpeg"
        raise ValueError("Bucket image must be PNG or JPEG")
