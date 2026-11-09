"""Sequential fixed-loopback projection receipts for immutable rendered pairs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import signal
import tempfile
import threading
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

BASE = "http://127.0.0.1:8765/lego-robot"
ROOT = Path("/mnt/dockyard/lego-robot-data/data/lego_curriculum/rendered-physics-first-6d8de05")
EPISODE_ROOT = ROOT.parent
REQUEST_LIMIT = 64 * 1024
API_LIMIT = 2 * 1024 * 1024
RECEIPT_LIMIT = 8 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


def json_object(raw: bytes) -> dict[str, Any]:
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError(f"Duplicate JSON key: {key}")
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_path(path: Path, root: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(root):
        raise ValueError(f"Path outside fixed publication root: {path}")
    for entry in (path, *path.parents):
        if entry.is_symlink():
            raise ValueError(f"Symlink publication path: {entry}")
    return path


def read_object(path: Path, limit: int) -> tuple[dict[str, Any], bytes]:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"Oversized JSON: {path}")
    value = json_object(raw)
    return value, raw


def identity(value: Any, pattern: str = r"[0-9a-f]{64}") -> str:
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError("Invalid immutable identity")
    return value


def urls(release: str, example: str, dashboard: str) -> tuple[str, str, list[str]]:
    lego = BASE + "/api/lego?" + urlencode({"release": release, "item": example})
    data = BASE + "/api/data/all?" + urlencode({"view": "examples", "item": dashboard})
    media = [lego + "&frame=0&camera=" + camera for camera in ("wrist", "external")]
    return lego, data, media


def load_request(path: Path, root: Path) -> tuple[dict[str, Any], str]:
    checked_path(path, root / "publication_requests")
    request, raw = read_object(path, REQUEST_LIMIT)
    if (request.get("schema_version") != 1
            or request.get("kind") != "lego-physical-render-publication-request"):
        raise ValueError("Unsupported publication request schema")
    source = identity(request.get("source_release_id"))
    derivative = identity(request.get("derivative_release_id"))
    artifact = identity(request.get("dashboard_release_id"))
    archive_hash = identity(request.get("archive_sha256"))
    identity(request.get("qualification_sha256"))
    if path.name != source + ".json":
        raise ValueError("Request filename identity mismatch")
    publication = checked_path(Path(request["published_root"]), root)
    archive = checked_path(Path(request["archive_path"]), publication)
    if (archive != publication / "episodes.sqlite"
            or publication != root / "releases" / derivative):
        raise ValueError("Archive is not an exposed immutable publication")
    expected_artifact = hashlib.sha256(
        f"examples:{archive.relative_to(EPISODE_ROOT).as_posix()}:{archive_hash}".encode()
    ).hexdigest()
    if artifact != expected_artifact:
        raise ValueError("Dashboard artifact identity mismatch")
    examples = request.get("example_ids")
    items = request.get("items")
    if (not isinstance(examples, list) or len(examples) != 2
            or not isinstance(items, list) or len(items) != 2):
        raise ValueError("Expected exactly two rendered commands")
    examples = [identity(value, r"[0-9a-f]{64}(?::[01])?") for value in examples]
    if len(set(examples)) != 2:
        raise ValueError("Duplicate source example identity")
    dashboards = []
    for example, item in zip(examples, items, strict=True):
        if not isinstance(item, dict) or item.get("example_id") != example:
            raise ValueError("Item/example identity mismatch")
        dashboard = identity(item.get("dashboard_item_id"), r"[0-9a-f]{20}")
        if dashboard != hashlib.sha256(f"curriculum:{example}:derived:{artifact}".encode()).hexdigest()[:20]:
            raise ValueError("Dashboard item identity mismatch")
        dashboards.append(dashboard)
        count = item.get("frame_count")
        if type(count) is not int or not 1 <= count <= 10000:
            raise ValueError("Invalid bounded frame population")
        lego, data, media_urls = urls(artifact, example, dashboard)
        if item.get("lego_url") != lego or item.get("data_url") != data:
            raise ValueError("Noncanonical fixed-loopback metadata URL")
        media = item.get("media")
        if not isinstance(media, list) or len(media) != 2:
            raise ValueError("Expected two paired frame-zero images")
        for row, url in zip(media, media_urls, strict=True):
            if not isinstance(row, dict) or row.get("url") != url:
                raise ValueError("Noncanonical fixed-loopback image URL")
            identity(row.get("sha256"))
    if len(set(dashboards)) != 2 or source == derivative:
        raise ValueError("Invalid derivative command identities")
    return request, hashlib.sha256(raw).hexdigest()


def check_qualification(value: Any, request: dict[str, Any]) -> None:
    if (not isinstance(value, dict)
            or value.get("publication_status") != "published"
            or value.get("image_qualified") is not True
            or value.get("sft_qualified") is not True
            or value.get("release_id") != request["derivative_release_id"]
            or value.get("source_release_id") != request["source_release_id"]
            or value.get("storage_sha256") != request["archive_sha256"]
            or value.get("unique_training_data_claim") is not False):
        raise ValueError("Qualification is not the exact published derivative")


def verify_files(request: dict[str, Any], root: Path) -> None:
    archive = checked_path(Path(request["archive_path"]), root)
    qualification = checked_path(archive.parent / "qualification.json", root)
    if digest_file(archive) != request["archive_sha256"]:
        raise ValueError("Immutable archive checksum mismatch")
    value, raw = read_object(qualification, API_LIMIT)
    if hashlib.sha256(raw).hexdigest() != request["qualification_sha256"]:
        raise ValueError("Immutable qualification checksum mismatch")
    check_qualification(value, request)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError(f"Fixed-loopback redirect refused: HTTP{code}")


def fetch(url: str, binary: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    opener = build_opener(ProxyHandler({}), NoRedirect())
    def expired(signum, frame):
        raise TimeoutError(f"Fixed browser endpoint exceeded4s: {url}")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 4)
    try:
        with opener.open(url, timeout=4) as response:
            code = response.status
            content_type = response.headers.get_content_type()
            body = response.read(API_LIMIT + 1)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    if code != 200 or len(body) > API_LIMIT:
        raise ValueError(f"Invalid bounded response: {url} HTTP{code}")
    result = {"url": url, "http_status": code, "content_type": content_type,
              "elapsed_seconds": time.monotonic() - started}
    if binary:
        if content_type != "image/png" or not body.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Expected browser PNG: {url}")
        result["body_sha256"] = hashlib.sha256(body).hexdigest()
    else:
        if content_type != "application/json":
            raise ValueError(f"Expected JSON content type: {url}")
        value = json_object(body)
        if "error" in value:
            raise ValueError(f"Expected successful JSON object: {url}")
        result["body"] = value
    return result


def check_frames(frames: Any, count: int) -> None:
    if (not isinstance(frames, list) or len(frames) != count
            or any(not isinstance(frame, dict) or frame.get("frame_index") != index
                   or frame.get("wrist") is not True or frame.get("external") is not True
                   for index, frame in enumerate(frames))):
        raise ValueError("Browser paired frame population mismatch")


def check_metadata(lego: dict[str, Any], data: dict[str, Any],
                   item: dict[str, Any], request: dict[str, Any]) -> None:
    release = lego.get("release")
    metadata = lego.get("item")
    if (not isinstance(release, dict) or release.get("id") != request["dashboard_release_id"]
            or release.get("artifact_sha256") != request["archive_sha256"]
            or not isinstance(metadata, dict) or metadata.get("id") != item["example_id"]
            or metadata.get("example_id") != item["example_id"]
            or metadata.get("release_id") != request["derivative_release_id"]
            or metadata.get("source_release_id") != request["source_release_id"]
            or metadata.get("media_origin") != "derived_cpu_raster"):
        raise ValueError("LEGO browser derivative identity mismatch")
    check_qualification(release.get("qualification"), request)
    check_qualification(metadata.get("qualification"), request)
    provenance = metadata.get("physical_pair_render")
    if (not isinstance(provenance, dict)
            or provenance.get("source_release_id") != request["source_release_id"]
            or provenance.get("derivative_release_id") != request["derivative_release_id"]
            or provenance.get("unique_training_data_claim") is not False):
        raise ValueError("Derivative physical provenance mismatch")
    if (data.get("id") != item["dashboard_item_id"]
            or data.get("archive_release") != request["dashboard_release_id"]
            or data.get("archive_item") != item["example_id"]
            or data.get("media_origin") != "derived_cpu_raster"
            or data.get("original_recording") is not False
            or data.get("paired_camera_images") is not True
            or data.get("movie_available") is not True
            or data.get("metadata") != metadata):
        raise ValueError("Data browser derivative projection mismatch")
    check_qualification(data.get("qualification"), request)
    check_frames(lego.get("frames"), item["frame_count"])
    check_frames(data.get("frames"), item["frame_count"])


def check_receipt(receipt: dict[str, Any], request: dict[str, Any], request_hash: str,
                  qualification: dict[str, Any]) -> None:
    if (receipt.get("schema_version") != 1
            or receipt.get("kind") != "lego-physical-render-browser-receipt"
            or receipt.get("status") != "verified"
            or receipt.get("request_sha256") != request_hash):
        raise ValueError("Stale or invalid browser receipt")
    timestamp = datetime.fromisoformat(receipt["verified_at"])
    if timestamp.tzinfo is None or timestamp.timestamp() > time.time() + 60:
        raise ValueError("Invalid receipt verification time")
    responses = receipt.get("responses")
    if not isinstance(responses, list) or len(responses) != 8:
        raise ValueError("Expected eight browser responses")
    expected = [url for item in request["items"]
                for url in (item["lego_url"], item["data_url"], *(row["url"] for row in item["media"]))]
    by_url = {row["url"]: row for row in responses}
    if (len(by_url) != 8 or set(by_url) != set(expected)
            or any(row.get("http_status") != 200 for row in responses)):
        raise ValueError("Browser response identity/status mismatch")
    for item in request["items"]:
        lego = by_url[item["lego_url"]]["body"]
        data = by_url[item["data_url"]]["body"]
        check_metadata(lego, data, item, request)
        if any(value != qualification for value in (
                lego["release"]["qualification"], lego["item"]["qualification"], data["qualification"])):
            raise ValueError("Browser qualification differs from immutable receipt")
        if (qualification.get("example_ids") != request["example_ids"]
                or any(lego["item"].get(key) != qualification.get(key)
                       for key in ("scene_family_id", "split"))):
            raise ValueError("Source example population, family or split mismatch")
        for media in item["media"]:
            row = by_url[media["url"]]
            if row.get("body_sha256") != media["sha256"] or row.get("content_type") != "image/png":
                raise ValueError("Browser paired PNG checksum mismatch")


def verify(path: Path, root: Path = ROOT) -> dict[str, Any]:
    request, request_hash = load_request(path, root)
    verify_files(request, root)
    responses = []
    for item in request["items"]:
        lego = fetch(item["lego_url"])
        data = fetch(item["data_url"])
        check_metadata(lego["body"], data["body"], item, request)
        responses.extend((lego, data))
        for media in item["media"]:
            response = fetch(media["url"], binary=True)
            if response["body_sha256"] != media["sha256"]:
                raise ValueError("Browser paired PNG checksum mismatch")
            responses.append(response)
    verify_files(request, root)
    if load_request(path, root)[1] != request_hash:
        raise ValueError("Request changed during browser verification")
    receipt = {"schema_version": 1, "kind": "lego-physical-render-browser-receipt", "status": "verified",
               "request_sha256": request_hash,
               "verified_at": datetime.now(timezone.utc).isoformat(),
               "responses": responses}
    qualification, _ = read_object(Path(request["archive_path"]).parent / "qualification.json", API_LIMIT)
    check_receipt(receipt, request, request_hash, qualification)
    if len(json.dumps(receipt, allow_nan=False).encode()) > RECEIPT_LIMIT:
        raise ValueError("Browser receipt exceeds8MiB")
    return receipt


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    checked_path(path, path.parent)
    content = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(content) > RECEIPT_LIMIT:
        raise ValueError("Oversized receipt")
    fd, name = tempfile.mkstemp(prefix=".browser-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def run(root: Path, stopping: threading.Event, once: bool = False) -> None:
    for name in ("publication_requests", "browser_receipts", "browser_failures"):
        checked_path(root / name, root)
    with (root / "browser_receipts" / ".verifier.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validated: dict[str, tuple[Any, ...]] = {}
        while not stopping.is_set():
            for path in sorted((root / "publication_requests").glob("*.json")):
                if stopping.is_set():
                    break
                if not re.fullmatch(r"[0-9a-f]{64}\.json", path.name):
                    LOGGER.error("Rejected invalid request filename: %s", path.name)
                    continue
                receipt_path = root / "browser_receipts" / path.name
                failure_path = root / "browser_failures" / path.name
                request_hash = None
                try:
                    request, request_hash = load_request(path, root)
                    stamp_paths = (path, receipt_path, Path(request["archive_path"]),
                                   Path(request["archive_path"]).parent / "qualification.json")
                    stamp = tuple((entry.stat().st_ino, entry.stat().st_size,
                                   entry.stat().st_mtime_ns, entry.stat().st_ctime_ns)
                                  if entry.exists() else None for entry in stamp_paths)
                    if validated.get(path.name) == stamp:
                        continue
                    if receipt_path.exists():
                        receipt, _ = read_object(checked_path(receipt_path, root), RECEIPT_LIMIT)
                        verify_files(request, root)
                        qualification, _ = read_object(Path(request["archive_path"]).parent / "qualification.json",
                                                       API_LIMIT)
                        check_receipt(receipt, request, request_hash, qualification)
                        validated[path.name] = stamp
                        continue
                    if failure_path.exists():
                        failure, _ = read_object(checked_path(failure_path, root), REQUEST_LIMIT)
                        if (failure.get("request_sha256") == request_hash
                                and failure.get("retry_after", 0) > time.time()):
                            continue
                    receipt = verify(path, root)
                    atomic_json(receipt_path, receipt)
                    if len(validated) >= 256:
                        validated.pop(next(iter(validated)))
                    validated[path.name] = tuple((entry.stat().st_ino, entry.stat().st_size,
                                                   entry.stat().st_mtime_ns, entry.stat().st_ctime_ns)
                                                  for entry in stamp_paths)
                    LOGGER.info("Browser projection verified: %s", request["source_release_id"])
                except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
                    LOGGER.exception("Browser projection failed: %s", path.name)
                    atomic_json(failure_path, {
                        "schema_version": 1, "status": "failed", "request_sha256": request_hash,
                        "error": str(error), "retry_after": time.time() + 60,
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                    })
            if once:
                return
            stopping.wait(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    run(ROOT, stopping, args.once)


if __name__ == "__main__":
    main()
