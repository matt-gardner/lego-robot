import json

import pytest

from so101.collect_catalog_sim import validate_recording
from so101.produce_physical_catalog import handoff_pair, pending_queue_bytes


def test_physical_validation_requires_states_not_images(tmp_path):
    metadata = {"outcome": "success", "fps": 10,
                "initial": {"expert_acceptance_version": 1},
                "physical_state_dimensions": {"qvel": 1, "ctrl": 1, "act": 0}}
    rows = [{
        "elapsed": i / 10, "state": [0.] * 6, "action": [0.] * 6,
        "qpos": [0.], "qvel": [0.], "ctrl": [0.], "act": [],
        "command_entity_ids": ["object"], "command_roles": ["object"],
        "action_interval_max_contact_penetration_m": 0.,
        "action_interval_max_table_penetration_m": 0., "phase": "lift",
    } for i in range(2)]
    validate_recording(tmp_path, metadata, rows, 1, physical_only=True)
    rows[0]["qvel"] = [float("nan")]
    with pytest.raises(ValueError, match="qvel"):
        validate_recording(tmp_path, metadata, rows, 1, physical_only=True)
    rows[0]["qvel"] = [0.]
    rows[0]["image"] = "not-rendered.jpg"
    with pytest.raises(ValueError, match="rendered output"):
        validate_recording(tmp_path, metadata, rows, 1, physical_only=True)


@pytest.mark.parametrize("status", ["physically_validated", "incomplete"])
def test_queue_only_accepts_complete_physical_pairs(tmp_path, status):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    report = {
        "status": status, "successful_episodes": ["command_0", "command_1"],
        "attempts": [{"example_id": str(i), "outcome": "success"} for i in range(2)],
        "release_id": "physical-pair", "scene_family_id": "family", "split": "development",
    }
    (attempt / "collection.json").write_text(json.dumps(report))
    queue = tmp_path / "queue"
    if status == "incomplete":
        with pytest.raises(ValueError, match="validated physical pairs"):
            handoff_pair(attempt, queue, {})
        assert not queue.exists()
    else:
        path = handoff_pair(attempt, queue, {})
        manifest = json.loads(path.read_text())
        assert manifest["status"] == "physically_validated"
        assert manifest["images_pending"] and not manifest["sft_qualified"]
        assert manifest["files_sha256"]["collection.json"]
        assert handoff_pair(attempt, queue, {}) == path


def test_downstream_acknowledgment_releases_queue_budget_without_deleting_sources(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "states").write_bytes(b"trajectory")
    queue = tmp_path / "render_queue"
    (queue / "ready").mkdir(parents=True)
    manifest = {"release_id": "pair", "pair_root": str(source),
                "files_sha256": {"states": "unused-for-budget"}}
    (queue / "ready/pair.json").write_text(json.dumps(manifest))
    assert pending_queue_bytes(tmp_path) == 10
    (queue / "acknowledged").mkdir()
    acknowledgment = queue / "acknowledged/pair.json"
    acknowledgment.write_text(json.dumps({"release_id": "pair", "status": "published"}))
    assert pending_queue_bytes(tmp_path) == 0
    assert (source / "states").exists()
    acknowledgment.write_text(json.dumps({"release_id": "different", "status": "published"}))
    with pytest.raises(ValueError, match="acknowledgment"):
        pending_queue_bytes(tmp_path)
