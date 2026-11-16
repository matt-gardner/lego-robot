import hashlib
import json

import pytest

from so101.pile_directory import current_image, current_inventory, current_response


def fixture(root):
    directory = root / "piles" / "pile_0001"
    (directory / "tasks").mkdir(parents=True)
    metadata = {"schema_version": 1, "manifest": {"datasets": []}, "buckets": [],
                "calibration": {"groups": [{"problem_id": "original-task", "attempts": 4}]},
                "distance_pilots": []}
    (root / "piles/metadata.json").write_text(json.dumps(metadata))
    body = b"\x89PNG\r\n\x1a\npile"
    (directory / "rendered.png").write_bytes(body)
    pile = {"id": directory.name, "scene_id": "scene", "tags": {"piece_count": 1},
            "qualification": "unmeasured", "rendered": {"sha256": hashlib.sha256(body).hexdigest()}}
    (directory / "pile.json").write_text(json.dumps(pile))
    task = {"schema_version": 1, "id": "original-task", "task": "grasp",
            "source": {"scene_id": "scene"}, "initial_arm_position": {"joint_a": 0.1},
            "assistance_checkpoints": [], "assistance_checkpoint_status": "not_persisted"}
    path = directory / "tasks/task_001.json"
    path.write_text(json.dumps(task))
    return directory, path, task, body


def test_directory_is_live_source_and_metadata_is_not_a_second_task_inventory(tmp_path):
    _, path, task, _ = fixture(tmp_path)
    first = current_inventory(tmp_path)
    task["task"] = "place"
    path.write_text(json.dumps(task))
    second = current_inventory(tmp_path)
    assert first["problems"]["problems"][0]["task"] == "grasp"
    assert second["problems"]["problems"][0]["task"] == "place"
    assert second["manifest"]["snapshot_id"] != first["manifest"]["snapshot_id"]
    assert second["calibration"]["groups"] == [{"problem_id": "original-task", "attempts": 4}]
    assert (tmp_path / "piles/metadata.json").stat().st_size < 500
    assert isinstance(current_response(tmp_path), bytes)


def test_original_pile_bytes_are_served_and_corruption_is_explicit(tmp_path):
    directory, _, _, body = fixture(tmp_path)
    assert current_image(tmp_path, "pile_0001", 0) == (body, "image/png")
    (directory / "rendered.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest differs"):
        current_image(tmp_path, "pile_0001", 0)
    with pytest.raises(ValueError, match="Unknown"):
        current_image(tmp_path, "../elsewhere", 0)


def test_incomplete_import_and_symlinked_tasks_are_not_served_as_success(tmp_path):
    directory, path, _, _ = fixture(tmp_path)
    marker = tmp_path / "piles/.incomplete"
    marker.touch()
    with pytest.raises(ValueError, match="incomplete"):
        current_inventory(tmp_path)
    marker.unlink()
    original = directory / "original.json"
    path.rename(original)
    path.symlink_to(original)
    with pytest.raises(ValueError, match="symlink"):
        current_inventory(tmp_path)
