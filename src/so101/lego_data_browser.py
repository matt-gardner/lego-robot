"""Read-only browsing of versioned part catalogs and episode-byte containers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path

from so101.episode_storage import EpisodeStorage
from so101.generation_order import generation_order
from so101.part_preview import thumbnail, visual_geometry


def catalog_summary(path: Path) -> dict:
    """Read catalog headers and readiness counts without retaining the part array."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as stream:
        buffer = ""
        eof = False

        def fill() -> None:
            nonlocal buffer, eof
            chunk = stream.read(65536)
            eof = not chunk
            buffer += chunk
            if len(buffer) > 8 * 1024 * 1024:
                raise ValueError("Catalog entry exceeds summary reader limit")

        def whitespace() -> None:
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                fill()
                buffer = buffer.lstrip()

        def token(expected: str) -> None:
            nonlocal buffer
            whitespace()
            if not buffer.startswith(expected):
                raise ValueError(f"Invalid catalog JSON: expected {expected}")
            buffer = buffer[len(expected):]

        def value():
            nonlocal buffer
            whitespace()
            while True:
                try:
                    result, end = decoder.raw_decode(buffer)
                    if end == len(buffer) and not eof:
                        fill()
                        continue
                    buffer = buffer[end:]
                    return result
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError("Invalid or truncated catalog JSON") from None
                    fill()

        token("{")
        metadata = {}
        verified = 0
        while True:
            key = value()
            if not isinstance(key, str) or key in metadata:
                raise ValueError("Invalid or duplicate catalog field")
            token(":")
            if key == "parts":
                token("[")
                whitespace()
                if not buffer.startswith("]"):
                    while True:
                        part = value()
                        if not isinstance(part, dict):
                            raise ValueError("Invalid catalog part")
                        verified += part.get("status") == "verified"
                        whitespace()
                        if buffer.startswith("]"):
                            break
                        token(",")
                token("]")
                metadata[key] = True
            else:
                metadata[key] = value()
            whitespace()
            if buffer.startswith("}"):
                break
            token(",")
        token("}")
        whitespace()
        if buffer:
            raise ValueError("Trailing catalog JSON content")
        return {**metadata, "validated_parts": verified}


class LegoDataBrowser:
    def __init__(self, workspace: Path, catalog_root: Path | None = None,
                 episode_root: Path | None = None):
        self.root = workspace.resolve() / "data"
        self.catalog_root = catalog_root.absolute() if catalog_root else self.root
        self.episode_root = episode_root.absolute() if episode_root else self.root
        self.fingerprints: dict[Path, tuple[tuple[int, int, int, int], str]] = {}
        self.lock = threading.RLock()
        self.preview_catalogs: dict[str, dict] = {}
        self.release_summaries: dict[str, dict] = {}
        self.target_coverage: dict[str, set[str]] = {}

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

    def releases(self, selected: str | None = None) -> dict[str, dict]:
        with self.lock:
            return self._releases(selected)

    def _releases(self, selected: str | None = None) -> dict[str, dict]:
        releases = {}
        notices = self.notices()
        for kind in ("parts", "examples"):
            root = self.catalog_root if kind == "parts" else self.episode_root
            for path in self.release_paths(root, kind):
                if any(component.startswith(".") for component in path.relative_to(root).parts):
                    continue
                self.contained(path, root)
                artifact_digest = self.artifact_hash(path)
                relative = path.relative_to(root).as_posix()
                key = hashlib.sha256(f"{kind}:{relative}:{artifact_digest}".encode()).hexdigest()
                if selected is not None and key != selected:
                    continue
                summary = self.release_summaries.get(key)
                if summary is None or selected is not None:
                    if kind == "parts":
                        if selected is None:
                            metadata = catalog_summary(path)
                            content = None
                        else:
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
                    digest = artifact_digest if content is None else hashlib.sha256(content).hexdigest()
                    if self.artifact_hash(path) != artifact_digest:
                        raise ValueError("Release changed while reading metadata")
                    summary = {
                        "id": key, "kind": kind, "version": version,
                        "path": relative, "catalog_sha256": digest,
                        "artifact_sha256": artifact_digest,
                        "generated_at": metadata.get("generated_at") if kind == "examples" else None,
                        "validated_parts": (metadata["validated_parts"] if kind == "parts" and selected is None else
                                            sum(row.get("status") == "verified" for row in metadata["parts"])
                                            if kind == "parts" else 0),
                    }
                    self.release_summaries[key] = summary
                    if selected is None:
                        del metadata, content
                qualification = None
                if kind == "examples" and (path.parent / "qualification.json").is_file():
                    receipt_path = self.contained(path.parent / "qualification.json", root)
                    qualification = json.loads(receipt_path.read_bytes())
                    if qualification.get("storage_sha256") != artifact_digest:
                        raise ValueError("Episode qualification receipt storage hash mismatch")
                releases[key] = {
                    **summary,
                    "publication_mtime_ns": path.stat().st_mtime_ns,
                    "qualification_notices": [
                        notice for notice in notices if kind == "parts"
                        and notice["catalog_id"] == summary["version"]
                        and notice["catalog_sha256"] == summary["catalog_sha256"]
                    ],
                    "qualification": qualification,
                }
                if selected is not None:
                    releases[key]["metadata"] = metadata
                    return releases
        self.release_summaries = {key: self.release_summaries[key] for key in releases}
        return releases

    def release_paths(self, root: Path, kind: str) -> list[Path]:
        if not root.exists():
            return []
        paths = []
        def traversal_error(error: OSError) -> None:
            raise error
        for directory, children, files in os.walk(root, onerror=traversal_error):
            children[:] = [name for name in children if not name.startswith(".")]
            matches = ([name for name in files if name == "catalog.json"] if kind == "parts"
                       else [name for name in files if name.endswith(".sqlite")
                             and not name.startswith(".") and name != "ledger.sqlite"])
            paths.extend(Path(directory) / name for name in matches)
            if kind == "parts" and matches:
                children.clear()
        return sorted(paths)

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

    def footprint(self, row: dict) -> str:
        match = re.search(r"\b(\d+(?:\.\d+)?)\s+x\s+(\d+(?:\.\d+)?)\b", row.get("description", ""))
        if not match or row.get("family") not in ("brick", "plate", "tile", "round", "slope"):
            return "not specified"
        sides = sorted((float(match[1]), float(match[2])))
        return f"{sides[0]:g} x {sides[1]:g}"

    def release(self, release_id: str) -> dict:
        release = self.releases(release_id).get(release_id)
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
            self.preview_catalogs[release["id"]] = {
                "release": {key: value for key, value in release.items() if key != "metadata"},
                "parts": {
                    row["part_id"]: {key: row.get(key) for key in ("part_id", "mjcf", "assets")}
                    for row in rows if row.get("mjcf")
                },
            }
            while len(self.preview_catalogs) > 2:
                del self.preview_catalogs[next(iter(self.preview_catalogs))]
            return [{**row, "id": row["part_id"], "stored_status": row["status"],
                     "footprint": self.footprint(row),
                     "readiness": "ready" if qualification == "not_invalidated" and row["status"] == "verified" else "not_ready",
                     "qualification_status": qualification,
                     "effective_status": qualification if qualification != "not_invalidated"
                     else row["status"]} for row in rows]
        rows = []
        with EpisodeStorage(self.contained(self.episode_root / release["path"], self.episode_root)) as storage:
            for source, content, digest in storage.connection.execute(
                "SELECT source,content,sha256 FROM files WHERE name='episode.json' ORDER BY source"
            ):
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError(f"Episode checksum mismatch: {source}")
                metadata = json.loads(content)
                derivative = metadata.get("physical_pair_render")
                if derivative is not None:
                    qualification = release.get("qualification")
                    if (not isinstance(derivative, dict)
                            or derivative.get("kind") != "lego-physical-pair-render"
                            or derivative.get("source_has_images") is not False
                            or derivative.get("unique_training_data_claim") is not False
                            or derivative.get("derivative_release_id") != metadata.get("release_id")
                            or not isinstance(qualification, dict)
                            or qualification.get("image_qualified") is not True
                            or qualification.get("sft_qualified") is not True
                            or qualification.get("publication_status") != "published"
                            or qualification.get("release_id") != metadata.get("release_id")):
                        raise ValueError("Invalid or unqualified physical-pair render")
                rows.append({
                    **metadata,
                    "media_origin": "derived_cpu_raster" if derivative else "original_recordings",
                    "qualification": release.get("qualification") if derivative else None,
                    "category": metadata.get("category", metadata.get("tags", {}).get("category")),
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
        return sorted(rows, key=generation_order)

    def generated_targets(self) -> set[str]:
        releases = [row for row in self.releases().values() if row["kind"] == "examples"]
        self.target_coverage = {row["id"]: self.target_coverage[row["id"]]
                                for row in releases if row["id"] in self.target_coverage}
        for release in releases:
            if release["id"] in self.target_coverage:
                continue
            targets = set()
            path = self.contained(self.episode_root / release["path"], self.episode_root)
            with EpisodeStorage(path) as storage:
                for source in storage.catalog.get("successful_episodes", []):
                    files = {}
                    for name in ("episode.json", "scene_plan.json"):
                        found = storage.connection.execute(
                            "SELECT content,sha256 FROM files WHERE source=? AND name=?",
                            (source, name),
                        ).fetchone()
                        if found is None:
                            continue
                        if hashlib.sha256(found[0]).hexdigest() != found[1]:
                            raise ValueError(f"Coverage metadata checksum mismatch: {source}/{name}")
                        files[name] = json.loads(found[0])
                    metadata, plan = files.get("episode.json"), files.get("scene_plan.json")
                    if not metadata or not plan or not storage.catalog.get("catalog_sha256"):
                        continue
                    objects = [obj for obj in plan.get("objects", [])
                               if obj.get("instance_id") == metadata.get("target")]
                    if len(objects) > 1:
                        raise ValueError(f"Ambiguous supervised target identity: {source}")
                    if objects and isinstance(objects[0].get("part_id"), str):
                        targets.add(objects[0]["part_id"])
            self.target_coverage[release["id"]] = targets
        return set().union(*self.target_coverage.values()) if self.target_coverage else set()

    def geometry(self, query: dict[str, str]) -> dict:
        cached = self.preview_catalogs.get(query["release"])
        if cached is None:
            release = self.release(query["release"])
            if release["kind"] != "parts":
                raise ValueError("Geometry previews require a part catalog")
            self.rows(release)
            cached = self.preview_catalogs[query["release"]]
        release = cached["release"]
        if release["kind"] != "parts":
            raise ValueError("Geometry previews require a part catalog")
        catalog_path = self.contained(self.catalog_root / release["path"], self.catalog_root)
        if self.artifact_hash(catalog_path) != release["artifact_sha256"]:
            raise FileNotFoundError("Catalog changed; reload the pinned part identity")
        row = cached["parts"].get(query["item"])
        if row is None:
            raise FileNotFoundError("Part has no prepared geometry in this catalog")
        reference = row.get("mjcf")
        if not isinstance(reference, str):
            raise FileNotFoundError("Part has no prepared geometry")
        relative = Path(reference)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe MJCF reference")
        asset = next((asset for asset in row.get("assets", []) if asset["path"] == reference), None)
        if asset is None:
            raise ValueError("MJCF missing from hash-pinned assets")
        catalog_dir = (self.catalog_root / release["path"]).parent
        path = self.contained(catalog_dir / relative, self.catalog_root)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != asset["sha256"]:
            raise ValueError("Part geometry checksum mismatch")
        return {**visual_geometry(content), "part_id": row["part_id"],
                "asset_sha256": asset["sha256"], "release_id": release["id"]}

    def preview(self, query: dict[str, str]) -> bytes:
        return thumbnail(self.geometry(query))

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
        with EpisodeStorage(self.contained(self.episode_root / release["path"], self.episode_root)) as storage:
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
        inventory = query.get("inventory", "")
        if inventory not in ("", "parts") or (inventory and query.get("release")):
            raise ValueError("Inventory must be parts without a selected release")
        if not query.get("release"):
            releases = list(self.releases().values())
            if inventory == "parts":
                releases = [row for row in releases if row["kind"] == "parts"]
            releases = ([row for row in releases if row["kind"] == "parts"] +
                        sorted((row for row in releases if row["kind"] == "examples"),
                               key=generation_order))
            return {"releases": [
                {key: value for key, value in release.items() if key != "metadata"}
                for release in releases
            ], "feedback": {"available": False,
                            "reason": "Authenticated request-monitor delivery is not configured."}}
        release = self.release(query["release"])
        rows = self.rows(release)
        coverage = query.get("coverage", "")
        if coverage not in ("", "generated"):
            raise ValueError("Coverage must be empty or generated")
        if coverage:
            if release["kind"] != "parts":
                raise ValueError("Part coverage requires a parts catalog")
            targets = self.generated_targets()
            rows = [row for row in rows if row["part_id"] in targets]
        if "item" in query:
            row = next((row for row in rows if row["id"] == query["item"]), None)
            if row is None:
                raise FileNotFoundError("Part or example not found in release")
            base_part_id = (row.get("base_part_id") or
                            (row.get("part_id") if row.get("decoration") == "undecorated" else None))
            frames = []
            if release["kind"] == "examples":
                with EpisodeStorage(self.contained(self.episode_root / release["path"], self.episode_root)) as storage:
                    found = storage.connection.execute(
                        "SELECT content,sha256 FROM files WHERE source=? AND name='frames.jsonl'",
                        (row["source"],),
                    ).fetchone()
                    if found:
                        if hashlib.sha256(found[0]).hexdigest() != found[1]:
                            raise ValueError("Frame metadata checksum mismatch")
                        for index, line in enumerate(found[0].splitlines()):
                            frame = json.loads(line)
                            frames.append({"frame_index": index,
                                           "timestamp": frame.get("timestamp", frame.get("elapsed")),
                                           "wrist": "image" in frame,
                                           "external": "external_image" in frame})
            return {"item": row, "frames": frames,
                    "variants": [
                        {"part_id": candidate["part_id"], "description": candidate["description"],
                         "decoration": candidate.get("decoration", "not recorded"),
                         "status": candidate["status"],
                         "relationship": candidate.get("base_part_relationship")}
                        for candidate in release["metadata"].get("parts", [])
                        if base_part_id and (
                            candidate.get("base_part_id") or
                            (candidate["part_id"] if candidate.get("decoration") == "undecorated" else None)
                        ) == base_part_id
                        and candidate["part_id"] != row["part_id"]
                    ],
                    "release": {key: value for key, value in release.items()
                                           if key != "metadata"}}
        search = query.get("search", "").casefold()
        facets = {
            field: sorted({str(row.get(field, "not specified")) for row in rows})
            for field in ("family", "footprint", "decoration")
        } if release["kind"] == "parts" else {}
        rows = [row for row in rows if not search or search in
                json.dumps(row, sort_keys=True).casefold()]
        for field in ("status", "effective_status", "readiness", "family", "footprint", "decoration", "task", "category", "split",
                      "expert_acceptance", "visual_audit", "qualification_status"):
            if query.get(field):
                rows = [row for row in rows if str(row.get(field, "")) == query[field]]
        offset, limit = int(query.get("offset", 0)), int(query.get("limit", 24))
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("Expected offset >= 0 and limit 1..100")
        return {"items": rows[offset:offset + limit], "total": len(rows),
                "release": {key: value for key, value in release.items() if key != "metadata"},
                "facets": facets,
                "offset": offset, "limit": limit}
