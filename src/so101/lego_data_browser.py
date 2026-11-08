"""Read-only browsing of versioned part catalogs and episode-byte containers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from so101.episode_storage import EpisodeStorage


class LegoDataBrowser:
    def __init__(self, workspace: Path, catalog_root: Path | None = None):
        self.root = workspace.resolve() / "data"
        self.catalog_root = catalog_root.absolute() if catalog_root else self.root
        self.fingerprints: dict[Path, tuple[tuple[int, int, int, int], str]] = {}

    def artifact_hash(self, path: Path) -> str:
        stat = path.stat()
        revision = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self.fingerprints.get(path)
        if cached and cached[0] == revision:
            return cached[1]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if revision != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("Release changed while fingerprinting")
        self.fingerprints[path] = (revision, digest)
        return digest

    def contained(self, path: Path, root: Path | None = None) -> Path:
        root = root or self.root
        relative = path.relative_to(root)
        current = root
        if root.is_symlink():
            raise ValueError("Catalog root must not be a symlink")
        for component in relative.parts:
            current = current / component
            if current.is_symlink():
                raise ValueError("Catalog and release paths must not contain symlinks")
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Path outside dashboard data root")
        return path

    def releases(self) -> dict[str, dict]:
        releases = {}
        notices = self.notices()
        for pattern, kind in (("**/catalog.json", "parts"), ("**/*.sqlite", "examples")):
            root = self.catalog_root if kind == "parts" else self.root
            for path in sorted(root.glob(pattern)):
                if any(component.startswith(".") for component in path.relative_to(root).parts):
                    continue
                self.contained(path, root)
                if kind == "parts":
                    content = path.read_bytes()
                    metadata = json.loads(content)
                    if "parts" not in metadata:
                        continue
                    if metadata.get("schema_version") != 1:
                        raise ValueError(f"Unsupported part catalog schema: {path}")
                    version = metadata["catalog_id"]
                else:
                    with EpisodeStorage(path) as storage:
                        metadata = storage.catalog
                        content = storage.connection.execute(
                            "SELECT content FROM catalog"
                        ).fetchone()[0]
                    version = metadata.get("release_id", path.stem)
                digest = hashlib.sha256(content).hexdigest()
                artifact_digest = self.artifact_hash(path)
                relative = path.relative_to(root).as_posix()
                key = hashlib.sha256(f"{kind}:{relative}:{artifact_digest}".encode()).hexdigest()
                releases[key] = {
                    "id": key, "kind": kind, "version": version,
                    "path": relative, "catalog_sha256": digest,
                    "artifact_sha256": artifact_digest,
                    "metadata": metadata,
                    "qualification_notices": [
                        notice for notice in notices if kind == "parts"
                        and notice["catalog_id"] == version
                        and notice["catalog_sha256"] == digest
                    ],
                }
        return releases

    def notices(self) -> list[dict]:
        notices = []
        identities = set()
        for path in sorted((self.catalog_root / "notices").glob("*.json")):
            if path.name.startswith("."):
                continue
            self.contained(path, self.catalog_root)
            content = path.read_bytes()
            notice = json.loads(content)
            if not isinstance(notice, dict) or notice.get("schema_version") != 1:
                raise ValueError(f"Unsupported qualification notice schema: {path.name}")
            for field in ("notice_id", "catalog_id", "reason", "source_work_item"):
                if not isinstance(notice.get(field), str) or not notice[field].strip():
                    raise ValueError(f"Invalid qualification notice {field}: {path.name}")
            if (not isinstance(notice.get("catalog_sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", notice["catalog_sha256"])
                    or notice.get("status") not in ("invalidated", "superseded")
                    or not isinstance(notice.get("evidence"), dict)):
                raise ValueError(f"Invalid qualification notice: {path.name}")
            if notice["notice_id"] in identities:
                raise ValueError(f"Duplicate qualification notice: {notice['notice_id']}")
            identities.add(notice["notice_id"])
            notices.append({**notice, "notice_sha256": hashlib.sha256(content).hexdigest()})
        return notices

    def qualification_status(self, release: dict) -> str:
        statuses = {notice["status"] for notice in release["qualification_notices"]}
        if "invalidated" in statuses:
            return "invalidated"
        return "superseded" if "superseded" in statuses else "not_invalidated"

    def release(self, release_id: str) -> dict:
        release = self.releases().get(release_id)
        if release is None:
            raise FileNotFoundError("Catalog or immutable release not found")
        return release

    def rows(self, release: dict) -> list[dict]:
        if release["kind"] == "parts":
            rows = release["metadata"]["parts"]
            ids = [row["part_id"] for row in rows]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate part identity in catalog")
            if any(row["status"] not in ("imported", "prepared", "verified", "unavailable")
                   for row in rows):
                raise ValueError("Invalid catalog part status")
            qualification = self.qualification_status(release)
            return [{**row, "id": row["part_id"], "stored_status": row["status"],
                     "qualification_status": qualification,
                     "effective_status": qualification if qualification != "not_invalidated"
                     else row["status"]} for row in rows]
        rows = []
        with EpisodeStorage(self.contained(self.root / release["path"])) as storage:
            for source, content, digest in storage.connection.execute(
                "SELECT source,content,sha256 FROM files WHERE name='episode.json' ORDER BY source"
            ):
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError(f"Episode checksum mismatch: {source}")
                metadata = json.loads(content)
                rows.append({
                    **metadata,
                    "id": metadata.get("example_id", hashlib.sha256(source.encode()).hexdigest()),
                    "source": source,
                    "expert_acceptance": metadata.get(
                        "expert_acceptance",
                        "single-pass-v1" if metadata.get("initial", {}).get(
                            "expert_acceptance_version") == 1 else
                        "legacy final-success (may include retries)",
                    ),
                    "split": metadata.get("split", "not recorded"),
                    "visual_audit": metadata.get("visual_audit", "not recorded"),
                })
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate example identity in release")
        return rows

    def media(self, query: dict[str, str]) -> tuple[bytes, str]:
        release = self.release(query["release"])
        if release["kind"] != "examples":
            raise ValueError("Recorded frames require an episode release")
        rows = self.rows(release)
        row = next((row for row in rows if row["id"] == query["item"]), None)
        if row is None:
            raise FileNotFoundError("Example not found")
        frame_index = int(query["frame"])
        if frame_index < 0:
            raise ValueError("Frame must be nonnegative")
        with EpisodeStorage(self.contained(self.root / release["path"])) as storage:
            frames = storage.read(Path(row["source"]), "frames.jsonl").splitlines()
            if frame_index >= len(frames):
                raise FileNotFoundError("Recorded frame not found")
            frame = json.loads(frames[frame_index])
            view = query.get("camera", "wrist")
            if view not in ("wrist", "external"):
                raise ValueError("Camera must be wrist or external")
            image = frame.get("image" if view == "wrist" else "external_image")
            if not isinstance(image, str):
                raise FileNotFoundError(f"No recorded {view} frame; no reconstruction substituted")
            name = Path(image)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Unsafe recorded media path")
            content_type = {".png": "image/png", ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg"}.get(name.suffix.lower())
            if content_type is None:
                raise ValueError("Unsupported recorded image format")
            content = storage.read(Path(row["source"]), image)
            digest = storage.connection.execute(
                "SELECT sha256 FROM files WHERE source=? AND name=?",
                (row["source"], image),
            ).fetchone()[0]
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("Recorded image checksum mismatch")
            return content, content_type

    def list(self, query: dict[str, str]) -> dict:
        if not query.get("release"):
            return {"releases": [
                {key: value for key, value in release.items() if key != "metadata"}
                for release in self.releases().values()
            ], "feedback": {"available": False,
                            "reason": "Authenticated request-monitor delivery is not configured."}}
        release = self.release(query["release"])
        rows = self.rows(release)
        if "item" in query:
            row = next((row for row in rows if row["id"] == query["item"]), None)
            if row is None:
                raise FileNotFoundError("Part or example not found in release")
            frames = []
            if release["kind"] == "examples":
                with EpisodeStorage(self.contained(self.root / release["path"])) as storage:
                    found = storage.connection.execute(
                        "SELECT content,sha256 FROM files WHERE source=? AND name='frames.jsonl'",
                        (row["source"],),
                    ).fetchone()
                    if found:
                        if hashlib.sha256(found[0]).hexdigest() != found[1]:
                            raise ValueError("Frame metadata checksum mismatch")
                        for index, line in enumerate(found[0].splitlines()):
                            frame = json.loads(line)
                            frames.append({"frame_index": index, "timestamp": frame.get("timestamp"),
                                           "wrist": "image" in frame,
                                           "external": "external_image" in frame})
            return {"item": row, "frames": frames,
                    "release": {key: value for key, value in release.items()
                                           if key != "metadata"}}
        search = query.get("search", "").casefold()
        rows = [row for row in rows if not search or search in
                json.dumps(row, sort_keys=True).casefold()]
        for field in ("status", "family", "task", "category", "split",
                      "expert_acceptance", "visual_audit", "qualification_status"):
            if query.get(field):
                rows = [row for row in rows if str(row.get(field, "")) == query[field]]
        offset, limit = int(query.get("offset", 0)), int(query.get("limit", 24))
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("Expected offset >= 0 and limit 1..100")
        return {"items": rows[offset:offset + limit], "total": len(rows),
                "release": {key: value for key, value in release.items() if key != "metadata"},
                "offset": offset, "limit": limit}
