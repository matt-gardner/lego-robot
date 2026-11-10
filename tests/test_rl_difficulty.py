import hashlib
import json
from unittest.mock import Mock

import pytest

from so101.cloud_checkpoint import atomic_json
from so101.pile_task_worker import validate_request
from so101.rl_difficulty import DifficultyController, HarderDataRequests


def observe(controller, successes=8, total=8, maximum=0., key="view"):
    event = None
    for step in range(4):
        event = controller.record(key, successes, total, maximum, .05, step)
        if event is not None:
            return event
    return event


def test_zero_assistance_advances_only_after_admission_and_resumes_partial_window():
    controller = DifficultyController()
    assert observe(controller, maximum=.05)["action"] == "reduce-assistance"
    assert observe(controller)["action"] == "request"
    assert observe(controller) is None
    controller.admit("view", 1, "a" * 64)
    assert observe(controller)["action"] == "advance"
    assert controller.slice("view", 0.)["level"] == 1
    assert controller.record("view", 0, 8, 0., .05, 10) is None
    resumed = DifficultyController(json.loads(json.dumps(controller.state)))
    assert resumed.record("view", 0, 8, 0., .05, 11) is None
    assert resumed.record("view", 0, 8, 0., .05, 12) is None
    assert resumed.record("view", 0, 8, 0., .05, 13)["action"] == "backoff"
    assert resumed.slice("view", 0.)["level"] == 0
    with pytest.raises(ValueError, match="changed"):
        resumed.admit("view", 1, "b" * 64)


def test_success_band_and_slices_are_independent_and_weighted_by_actual_population():
    controller = DifficultyController()
    assert observe(controller, successes=4, maximum=.5) is None
    assert controller.slice("view", .5)["assistance"] == .5
    assert observe(controller, successes=0, maximum=.5, key="grasp") is None
    assert controller.slice("view", .5)["assistance"] == .5
    controller = DifficultyController()
    for _ in range(3):
        controller.record("view", 1, 1, 0., .05, 1)
    assert controller.record("view", 0, 100, 0., .05, 2) is None
    assert controller.state["events"] == []


def test_generation_failures_retry_with_a_bound_not_silent_saturation():
    controller = DifficultyController()
    for attempt in range(1, 4):
        assert observe(controller)["action"] == "request"
        controller.slice("view", 0.)["pending_request_id"] = str(attempt)
        controller.failed_request({
            "slice": "view", "level": 1, "status": "blocked",
            "request_id": str(attempt), "error": "resource pressure",
        })
    assert observe(controller) is None
    assert controller.state["events"][-1]["retry"] is False
    assert controller.slice("view", 0.)["pending"] == -1


def test_requests_match_real_worker_contract_and_admission_is_hash_bound(tmp_path):
    queue = HarderDataRequests(tmp_path / "requests", tmp_path / "catalog.json")
    path = queue.request("view", 1, "view", 123, ["held-out"], {
        "piece_count": 6, "color_mode": "many",
    })
    request = json.loads(path.read_text())
    validate_request(request)
    assert request["excluded_scene_families"] == ["held-out"]
    assert queue.request("view", 1, "view", 123, ["held-out"], request["tag_constraints"]) == path
    assert queue.results()[0]["status"] == "blocked"
    manifest = path.parent / "pool" / "manifest.json"
    manifest.parent.mkdir()
    atomic_json(manifest, {"request_id": request["request_id"]})
    result = {
        "request_id": request["request_id"], "status": "complete",
        "request_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "manifest_path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    atomic_json(path.with_name(f"{request['request_id']}.result.json"), result)
    assert queue.results()[0]["sha256"] == result["sha256"]
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        queue.results()


def test_only_one_bounded_generator_is_dispatched_and_worker_failure_is_explicit(tmp_path, monkeypatch):
    process = Mock(pid=123)
    process.poll.return_value = None
    start = Mock(return_value=process)
    monkeypatch.setattr("so101.rl_difficulty.subprocess.Popen", start)
    queue = HarderDataRequests(tmp_path, tmp_path / "catalog.json", ["worker"])
    for key in ("view", "grasp"):
        queue.request(key, 1, key, 123, [], {"piece_count": 6, "color_mode": "many"})
    assert sorted(row["status"] for row in queue.results()) == ["queued", "running"]
    queue.results()
    assert start.call_count == 1
    resumed = HarderDataRequests(tmp_path, tmp_path / "catalog.json", ["worker"])
    resumed.results()
    assert start.call_count == 1
    process.poll.return_value = 1
    process.returncode = 1
    with pytest.raises(RuntimeError, match="without result"):
        queue.results()
