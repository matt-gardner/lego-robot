import copy
import hashlib
import json
from pathlib import Path
import threading
from unittest.mock import patch

import pytest

from so101 import verify_rendered_browser as verifier


@pytest.fixture
def candidate(tmp_path):
    root = tmp_path / "rendered"
    for name in ("publication_requests", "browser_receipts", "browser_failures"):
        (root / name).mkdir(parents=True)
    source, derivative = "a" * 64, "b" * 64
    published = root / "releases" / derivative
    published.mkdir(parents=True)
    archive = published / "episodes.sqlite"
    archive.write_bytes(b"immutable archive fixture")
    archive_hash = verifier.digest_file(archive)
    artifact = hashlib.sha256(
        f"examples:{archive.relative_to(root.parent).as_posix()}:{archive_hash}".encode()
    ).hexdigest()
    qualification = {
        "publication_status": "published", "image_qualified": True, "sft_qualified": True,
        "release_id": derivative, "source_release_id": source,
        "storage_sha256": archive_hash, "unique_training_data_claim": False,
    }
    qpath = published / "qualification.json"
    qpath.write_text(json.dumps(qualification))
    examples = [source + ":0", source + ":1"]
    qualification.update(example_ids=examples, scene_family_id="family", split="development")
    qpath.write_text(json.dumps(qualification))
    items, replies = [], {}
    for example in examples:
        dashboard = hashlib.sha256(f"curriculum:{example}:derived:{artifact}".encode()).hexdigest()[:20]
        lego_url, data_url, media_urls = verifier.urls(artifact, example, dashboard)
        media = [{"url": url, "sha256": "d" * 64} for url in media_urls]
        item = {"example_id": example, "dashboard_item_id": dashboard, "frame_count": 2,
                "lego_url": lego_url, "data_url": data_url, "media": media}
        items.append(item)
        metadata = {"id": example, "example_id": example, "release_id": derivative,
                    "source_release_id": source, "media_origin": "derived_cpu_raster",
                    "qualification": qualification, "scene_family_id": "family", "split": "development",
                    "physical_pair_render": {"source_release_id": source, "derivative_release_id": derivative,
                                             "unique_training_data_claim": False}}
        frames = [{"frame_index": i, "wrist": True, "external": True} for i in range(2)]
        lego = {"item": metadata, "frames": frames,
                "release": {"id": artifact, "artifact_sha256": archive_hash, "qualification": qualification}}
        data = {"id": dashboard, "archive_release": artifact, "archive_item": example,
                "media_origin": "derived_cpu_raster", "original_recording": False,
                "paired_camera_images": True, "movie_available": True, "metadata": metadata,
                "qualification": qualification, "frames": frames}
        replies[lego_url] = {"url": lego_url, "http_status": 200, "body": lego}
        replies[data_url] = {"url": data_url, "http_status": 200, "body": data}
        for url in media_urls:
            replies[url] = {"url": url, "http_status": 200,
                            "body_sha256": "d" * 64, "content_type": "image/png"}
    request = {"schema_version": 1, "kind": "lego-physical-render-publication-request",
               "source_release_id": source, "derivative_release_id": derivative,
               "dashboard_release_id": artifact, "archive_path": str(archive),
               "archive_sha256": archive_hash, "qualification_sha256": verifier.digest_file(qpath),
               "published_root": str(published), "example_ids": examples, "items": items}
    path = root / "publication_requests" / (source + ".json")
    path.write_text(json.dumps(request))
    with patch.object(verifier, "EPISODE_ROOT", root.parent):
        yield root, path, request, replies


def test_exact_pair_verifies_all_eight_responses_and_atomic_receipt(candidate):
    root, path, request, replies = candidate
    with patch.object(verifier, "fetch", side_effect=lambda url, binary=False: copy.deepcopy(replies[url])) as fetch:
        result = verifier.verify(path, root)
    assert result["status"] == "verified"
    assert result["request_sha256"] == verifier.digest_file(path)
    assert len(result["responses"]) == fetch.call_count == 8
    target = root / "browser_receipts" / path.name
    verifier.atomic_json(target, result)
    assert json.loads(target.read_text()) == result


@pytest.mark.parametrize("mutation", [
    "url", "archive_escape", "symlink", "archive_hash", "duplicate", "artifact_id",
])
def test_unsafe_request_never_performs_network(candidate, mutation):
    root, path, request, replies = candidate
    if mutation == "url":
        request["items"][0]["lego_url"] = "http://127.0.0.1:8765/private"
    elif mutation == "archive_escape":
        request["archive_path"] = "/tmp/private.sqlite"
    elif mutation == "symlink":
        archive = Path(request["archive_path"])
        other = archive.with_suffix(".original")
        archive.rename(other)
        archive.symlink_to(other)
    elif mutation == "archive_hash":
        Path(request["archive_path"]).write_bytes(b"changed")
    elif mutation == "duplicate":
        request["example_ids"][1] = request["example_ids"][0]
    else:
        request["dashboard_release_id"] = "f" * 64
    path.write_text(json.dumps(request))
    with patch.object(verifier, "fetch") as fetch, pytest.raises(ValueError):
        verifier.verify(path, root)
    fetch.assert_not_called()


@pytest.mark.parametrize("mutation", ["original", "missing_camera", "frame_count", "qualification", "png"])
def test_mismatched_projection_never_writes_success(candidate, mutation):
    root, path, request, replies = candidate
    first = request["items"][0]
    data = replies[first["data_url"]]["body"]
    if mutation == "original":
        data["original_recording"] = True
    elif mutation == "missing_camera":
        data["frames"][1]["external"] = False
    elif mutation == "frame_count":
        data["frames"].pop()
    elif mutation == "qualification":
        data["qualification"]["publication_status"] = "staged"
    else:
        replies[first["media"][0]["url"]]["body_sha256"] = "e" * 64
    with patch.object(verifier, "fetch", side_effect=lambda url, binary=False: copy.deepcopy(replies[url])):
        verifier.run(root, threading.Event(), once=True)
    assert not (root / "browser_receipts" / path.name).exists()
    assert json.loads((root / "browser_failures" / path.name).read_text())["status"] == "failed"


def test_changed_request_during_verification_is_rejected(candidate):
    root, path, request, replies = candidate
    def fetch(url, binary=False):
        path.write_text(json.dumps(request) + " ")
        return copy.deepcopy(replies[url])
    with patch.object(verifier, "fetch", side_effect=fetch), pytest.raises(ValueError, match="Request changed"):
        verifier.verify(path, root)


def test_timeout_is_visible_and_does_not_create_receipt(candidate):
    root, path, _, _ = candidate
    with patch.object(verifier, "fetch", side_effect=TimeoutError("fixed endpoint timed out")):
        verifier.run(root, threading.Event(), once=True)
    failure = json.loads((root / "browser_failures" / path.name).read_text())
    assert "timed out" in failure["error"]
    assert not (root / "browser_receipts" / path.name).exists()


def test_stale_success_receipt_cannot_ack_changed_request(candidate):
    root, path, request, replies = candidate
    with patch.object(verifier, "fetch", side_effect=lambda url, binary=False: copy.deepcopy(replies[url])):
        receipt = verifier.verify(path, root)
    verifier.atomic_json(root / "browser_receipts" / path.name, receipt)
    path.write_text(json.dumps(request) + " ")
    with patch.object(verifier, "fetch") as fetch:
        verifier.run(root, threading.Event(), once=True)
    fetch.assert_not_called()
    assert "Stale" in json.loads((root / "browser_failures" / path.name).read_text())["error"]


def test_idle_service_does_not_refetch_or_rehash_unchanged_archive(candidate):
    root, _, _, replies = candidate
    class StopAfterTwoScans(threading.Event):
        scans = 0
        def wait(self, timeout=None):
            self.scans += 1
            if self.scans == 2:
                self.set()
            return self.is_set()
    stopping = StopAfterTwoScans()
    with patch.object(verifier, "fetch", side_effect=lambda url, binary=False: copy.deepcopy(replies[url])) as fetch, \
            patch.object(verifier, "digest_file", wraps=verifier.digest_file) as digest:
        verifier.run(root, stopping)
        assert fetch.call_count == 8
        assert digest.call_count == 2
        with patch.object(verifier, "digest_file", wraps=verifier.digest_file) as digest:
            verifier.run(root, threading.Event(), once=True)
            assert fetch.call_count == 8
            assert digest.call_count == 1


@pytest.mark.parametrize("raw", [b'{"id":1,"id":2}', b'{"x":NaN}', b'[]'])
def test_invalid_json_is_rejected(raw):
    with pytest.raises(ValueError):
        verifier.json_object(raw)
