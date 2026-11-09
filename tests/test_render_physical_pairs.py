import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import pytest

from so101 import render_physical_pairs as consumer
from so101.cloud_checkpoint import atomic_json
from so101.episode_storage import consolidate


@pytest.fixture
def handoff(tmp_path, monkeypatch):
    root = tmp_path / "producer"
    pair = root / "pair"
    queue = root / "render_queue"
    ready = queue / "ready"
    ready.mkdir(parents=True)
    release = "a" * 64
    attempts = []
    ids = ["scene:0", "scene:1"]
    for index in range(2):
        source = pair / f"command_{index}"
        source.mkdir(parents=True)
        metadata = {
            "example_id": ids[index], "release_id": release,
            "scene_family_id": "family", "split": "development",
            "physical_state_dimensions": {"qpos": 1},
            "initial": {"qpos": [0.]},
        }
        atomic_json(source / "episode.json", metadata)
        (source / "frames.jsonl").write_text(json.dumps({"qpos": [0.]}) + "\n")
        attempts.append({"source": str(source), "example_id": ids[index],
                         "outcome": "success", "verification": {"success": True}})
    report = {
        "release_id": release, "scene_family_id": "family", "split": "development",
        "status": "physically_validated", "attempts": attempts,
        "successful_episodes": [row["source"] for row in attempts],
    }
    atomic_json(pair / "collection.json", report)
    manifest = {
        "schema_version": 1, "kind": "lego-physical-pair", "release_id": release,
        "scene_family_id": "family", "split": "development",
        "status": "physically_validated", "images_pending": True, "sft_qualified": False,
        "pair_root": str(pair), "collection": "collection.json", "example_ids": ids,
        "files_sha256": {
            path.relative_to(pair).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in pair.rglob("*") if path.is_file()
        },
    }
    path = ready / f"{release}.json"
    atomic_json(path, manifest)
    monkeypatch.setattr(consumer, "validate_recording", lambda *args, **kwargs: None)
    return path, manifest


def test_handoff_checks_complete_population_and_hashes(handoff):
    path, manifest = handoff
    consumer.verify_handoff(path)
    source = Path(manifest["pair_root"]) / "command_0" / "frames.jsonl"
    source.write_text('{"qpos": [1.0]}\n')
    with pytest.raises(ValueError, match="checksum"):
        consumer.verify_handoff(path)
    source.write_text('{"qpos": [0.0]}\n')
    (source.parent / "extra").write_text("unmanifested")
    with pytest.raises(ValueError, match="population"):
        consumer.verify_handoff(path)


def test_handoff_rejects_promoted_input_and_path_escape(handoff):
    path, manifest = handoff
    atomic_json(path, {**manifest, "sft_qualified": True})
    with pytest.raises(ValueError, match="unrendered"):
        consumer.verify_handoff(path)
    with pytest.raises(ValueError, match="Unsafe"):
        consumer.safe_file(Path(manifest["pair_root"]), "../outside")


def test_independent_pair_initial_pose_required(handoff):
    path, manifest = handoff
    source = Path(manifest["pair_root"]) / "command_1" / "frames.jsonl"
    source.write_text('{"qpos": [1.0]}\n')
    manifest["files_sha256"]["command_1/frames.jsonl"] = consumer.sha256(source)
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="independently"):
        consumer.verify_handoff(path)


@pytest.mark.parametrize("error,terminal", [
    (consumer.ImageRejected("uniform image"), True),
    (RuntimeError("renderer unavailable"), False),
])
def test_queue_ack_only_image_rejection_not_processing_failure(handoff, tmp_path, monkeypatch,
                                                              error, terminal):
    path, _ = handoff
    monkeypatch.setattr(consumer, "memory_state",
                        lambda: {"pressure": False, "available_bytes": consumer.MEMORY_ALLOWANCE})
    monkeypatch.setattr(consumer.DiskBackpressure, "wait", lambda *args: True)

    def fail(*args):
        raise error

    monkeypatch.setattr(consumer, "render_pair", fail)
    output = tmp_path / "derivatives"
    consumer.run(path.parent.parent, output, threading.Event(), once=True)
    ack = path.parent.parent / "acknowledged" / path.name
    assert ack.exists() is terminal
    journal = json.loads((output / path.name).read_bytes())
    if terminal:
        assert json.loads(ack.read_bytes())["status"] == "image_rejected"
    else:
        assert journal["status"] == "processing_failed"
        assert len(journal["attempts"]) == 1
    assert path.exists()


def test_qualified_stage_does_not_ack_publication(handoff, tmp_path, monkeypatch):
    path, _ = handoff
    monkeypatch.setattr(consumer, "memory_state",
                        lambda: {"pressure": False, "available_bytes": consumer.MEMORY_ALLOWANCE})
    monkeypatch.setattr(consumer.DiskBackpressure, "wait", lambda *args: True)
    monkeypatch.setattr(consumer, "render_pair", lambda *args: {"release_id": "b" * 64})
    output = tmp_path / "derivatives"
    consumer.run(path.parent.parent, output, threading.Event(), once=True)
    assert not (path.parent.parent / "acknowledged" / path.name).exists()
    journal = json.loads((output / path.name).read_bytes())
    assert journal["publication_status"] == "awaiting_browser_projection"


def test_uniform_and_malformed_images_rejected():
    with pytest.raises(consumer.ImageRejected):
        consumer.check_image(np.zeros((240, 320, 3), dtype=np.uint8))
    with pytest.raises(consumer.ImageRejected):
        consumer.check_image(np.zeros((240, 320), dtype=np.uint8))
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[0, 0] = 255
    consumer.check_image(image)


@pytest.fixture
def published_pair(handoff, tmp_path):
    path, manifest = handoff
    output = tmp_path / "derivatives"
    stage = output / ".staging" / "pair"
    stage.mkdir(parents=True)
    derivative = "b" * 64
    sources = []
    images = {}
    provenance = {
        "source_release_id": manifest["release_id"],
        "source_manifest_sha256": consumer.sha256(path),
        "derivative_release_id": derivative,
    }
    for index, example in enumerate(manifest["example_ids"]):
        source = stage / f"command_{index}"
        source.mkdir()
        sources.append(str(source))
        metadata = {"example_id": example, "release_id": derivative,
                    "physical_pair_render": provenance}
        atomic_json(source / "episode.json", metadata)
        row = {"image": "wrist.png", "external_image": "external.png"}
        (source / "frames.jsonl").write_text(json.dumps(row) + "\n")
        for name in row.values():
            (source / name).write_bytes(b"test image")
            images[f"command_{index}/{name}"] = consumer.sha256(source / name)
    catalog = stage / "command_0" / "source"
    catalog.mkdir()
    atomic_json(catalog / "parts_catalog.json", {})
    atomic_json(stage / "collection.json", {
        "status": "complete", "successful_episodes": sources, "release_id": derivative,
    })
    atomic_json(stage / "physical_pair_render.json", provenance)
    consolidate(stage, stage / "episodes.sqlite")
    atomic_json(stage / "qualification.json", {
        **provenance, "release_id": derivative, "status": "image_qualified",
        "image_qualified": True, "sft_qualified": True, "publication_status": "staged",
        "storage_sha256": consumer.sha256(stage / "episodes.sqlite"),
        "images_sha256": images, "example_ids": manifest["example_ids"],
    })
    published = consumer.publish_derivative(stage, output)
    request_path = consumer.publication_request(published, tmp_path, output)
    request = json.loads(request_path.read_bytes())
    qualification = json.loads((published / "qualification.json").read_bytes())
    responses = []
    for item in request["items"]:
        frames = [{"frame_index": 0, "wrist": True, "external": True}]
        metadata = {
            "example_id": item["example_id"], "release_id": derivative,
            "media_origin": "derived_cpu_raster", "physical_pair_render": provenance,
            "qualification": qualification,
        }
        responses.append({"url": item["lego_url"], "http_status": 200, "body": {
            "item": metadata, "frames": frames, "release": {
                "id": request["dashboard_release_id"], "artifact_sha256": request["archive_sha256"],
            },
        }})
        responses.append({"url": item["data_url"], "http_status": 200, "body": {
            "id": item["dashboard_item_id"], "archive_release": request["dashboard_release_id"],
            "archive_item": item["example_id"], "media_origin": "derived_cpu_raster",
            "original_recording": False, "paired_camera_images": True, "movie_available": True,
            "qualification": qualification, "frames": frames,
        }})
        responses.extend({"url": media["url"], "http_status": 200,
                          "body_sha256": media["sha256"], "content_type": "image/png"}
                         for media in item["media"])
    receipt_path = output / "browser_receipts" / path.name
    receipt_path.parent.mkdir()
    atomic_json(receipt_path, {
        "schema_version": 1, "kind": "lego-physical-render-browser-receipt", "status": "verified",
        "request_sha256": consumer.sha256(request_path),
        "verified_at": "2026-11-09T04:00:00+00:00", "responses": responses,
    })
    return path.parent.parent, stage, published, request_path, receipt_path


def test_publication_preserves_archive_and_is_idempotent(published_pair):
    queue, stage, published, request, receipt = published_pair
    assert consumer.sha256(stage / "episodes.sqlite") == consumer.sha256(published / "episodes.sqlite")
    assert consumer.publish_derivative(stage, published.parent.parent) == published
    assert not (queue / "acknowledged").exists()
    result = consumer.acknowledge_browser(queue, request, receipt)
    assert result["status"] == "published"
    assert consumer.acknowledge_browser(queue, request, receipt) == result


@pytest.mark.parametrize("damage", ["original", "missing_camera", "wrong_pixels", "stale_request"])
def test_browser_ack_requires_exact_derived_pair_evidence(published_pair, damage):
    queue, _, _, request, receipt = published_pair
    evidence = json.loads(receipt.read_bytes())
    if damage == "original":
        evidence["responses"][1]["body"]["original_recording"] = True
    elif damage == "missing_camera":
        evidence["responses"][0]["body"]["frames"][0]["external"] = False
    elif damage == "wrong_pixels":
        evidence["responses"][2]["body_sha256"] = "c" * 64
    else:
        evidence["request_sha256"] = "c" * 64
    atomic_json(receipt, evidence)
    with pytest.raises(ValueError):
        consumer.acknowledge_browser(queue, request, receipt)
    assert not (queue / "acknowledged").exists()


@pytest.mark.parametrize("invalid_receipt", [False, True])
def test_browser_receipt_serviced_during_other_pair_memory_pause(
        published_pair, monkeypatch, invalid_receipt):
    queue, stage, published, request, receipt = published_pair
    output = published.parent.parent
    journal_path = output / request.name
    atomic_json(journal_path, {
        "status": "browser_pending", "derivative_root": str(stage),
        "publication_request": str(request), "published_root": str(published),
    })
    if invalid_receipt:
        evidence = json.loads(receipt.read_bytes())
        evidence["request_sha256"] = "c" * 64
        atomic_json(receipt, evidence)
    active = output / ".staging" / "later-pair"
    active.mkdir()
    partial = active / "frames.jsonl"
    partial.write_text('{"qpos": [0.]}\n')
    monkeypatch.setattr(consumer, "memory_state",
                        lambda: {"pressure": True, "available_bytes": consumer.MEMORY_ALLOWANCE})
    stopping = threading.Event()
    verified = set()

    def service():
        consumer.service_browser_receipts(queue, output, verified)
        stopping.set()

    assert consumer.wait_memory(active, stopping, service) is False
    ack = queue / "acknowledged" / request.name
    assert ack.exists() is not invalid_receipt
    journal = json.loads(journal_path.read_bytes())
    assert journal["status"] == ("browser_verification_failed" if invalid_receipt else "published")
    assert partial.read_text() == '{"qpos": [0.]}\n'
    monkeypatch.setattr(consumer, "acknowledge_browser",
                        lambda *args: pytest.fail("Unchanged receipt was processed twice"))
    consumer.service_browser_receipts(queue, output, verified)


def test_qualified_pair_published_during_other_pair_memory_pause(published_pair, monkeypatch):
    queue, stage, published, request, receipt = published_pair
    output = published.parent.parent
    receipt_bytes = receipt.read_bytes()
    request.unlink()
    receipt.unlink()
    journal_path = output / request.name
    atomic_json(journal_path, {"status": "image_qualified", "derivative_root": str(stage)})
    monkeypatch.setattr(consumer, "memory_state",
                        lambda: {"pressure": True, "available_bytes": consumer.MEMORY_ALLOWANCE})
    monkeypatch.setattr(consumer.DiskBackpressure, "wait", lambda *args: True)
    stopping = threading.Event()
    verified = set()

    def service():
        consumer.service_browser_receipts(queue, output, verified, stopping)
        stopping.set()

    active = output / ".staging" / "later-pair"
    active.mkdir()
    assert consumer.wait_memory(active, stopping, service) is False
    assert request.exists()
    assert json.loads(journal_path.read_bytes())["status"] == "browser_pending"
    assert not (queue / "acknowledged" / request.name).exists()
    receipt.write_bytes(receipt_bytes)
    consumer.service_browser_receipts(queue, output, verified)
    assert json.loads((queue / "acknowledged" / request.name).read_bytes())["status"] == "published"


def test_publication_disk_stop_preserves_qualified_journal(published_pair, monkeypatch):
    queue, stage, published, request, receipt = published_pair
    output = published.parent.parent
    request.unlink()
    receipt.unlink()
    journal_path = output / request.name
    atomic_json(journal_path, {"status": "image_qualified", "derivative_root": str(stage)})
    stopping = threading.Event()

    def stop_wait(self, required, operation, stop):
        assert required > 0 and operation == "publication"
        stopping.set()
        assert stop()
        return False

    monkeypatch.setattr(consumer.DiskBackpressure, "wait", stop_wait)
    consumer.service_browser_receipts(queue, output, set(), stopping)
    assert json.loads(journal_path.read_bytes())["status"] == "image_qualified"
    assert not request.exists()
    assert not (queue / "acknowledged" / request.name).exists()
