import hashlib
import json

import pytest
from PIL import Image

from so101.bucket_dashboard import BucketDashboard
from so101.publish_photo_calibration import publish


def inputs(tmp_path):
    scene_root = tmp_path / "scene"
    scene_root.mkdir()
    scene = {
        "scene_id": "scene", "scene_family_id": "family", "tags": {"piece_count": 20},
        "tasks": [{"task_id": "scene:target:grasp", "task_type": "grasp",
                   "target_instance_id": "target", "initial_target_position": [0., 0., 0.],
                   "goal": {"lift_m": .05}, "solvability": "unmeasured", "grader_version": "v1"}],
    }
    problem = scene_root / "problem.json"
    problem.write_text(json.dumps(scene))
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({
        "kind": "lego-pile-task-pool", "status": "complete",
        "scenes": [{"root": str(scene_root), "split": "train",
                    "files_sha256": {"problem.json": hashlib.sha256(problem.read_bytes()).hexdigest()}}],
    }))
    reference = tmp_path / "references.json"
    data = {
        "kind": "pile-expert-references", "success_condition": "terminal-native-dwell-v1",
        "pool_sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
        "collector_sha256": "collector", "complete": False,
        "attempts": [{"source": str(scene_root / "tasks" / "scene:target:grasp"), "task": "grasp"}],
        "references": [],
    }
    reference.write_text(json.dumps(data))
    output = tmp_path / "dashboard"
    return pool, reference, output, data, scene_root


def test_new_photo_progress_is_not_a_policy_rate_and_rejects_other_pool(tmp_path):
    pool, reference, output, data, scene_root = inputs(tmp_path)
    result = publish(pool, reference, output)
    snapshot = BucketDashboard(output).snapshot()
    assert result["problems"] == 1
    assert snapshot["calibration"]["groups"] == []
    assert snapshot["manifest"]["measurement_status"] == "unmeasured"
    assert snapshot["manifest"]["progress"]["policy_rollouts"] == 0
    assert snapshot["manifest"]["progress"]["reference_attempts"] == 1
    assert snapshot["problems"]["problems"][0]["source"]["kind"] == "photo"
    images = tmp_path / "images"
    images.mkdir()
    image = images / "initial.png"
    Image.new("RGB", (5, 3)).save(image)
    identity = snapshot["problems"]["problems"][0]["id"]
    receipt = {
        "episode_source": data["attempts"][0]["source"], "pool_sha256": data["pool_sha256"],
        "problem_id": identity, "target_instance_id": "target",
        "image_file": image.name, "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "width": 5, "height": 3, "physics_unchanged": True, "assistance": 0.,
        "camera": "external", "backend": "mujoco", "target_bbox_xyxy": [1, 0, 4, 3],
    }
    annotation = images / "receipt.json"
    annotation.write_text(json.dumps(receipt))
    result = publish(pool, reference, output, images_paths=[images])
    assert result["problems_with_images"] == 1
    assert BucketDashboard(output).image(identity, 0)[0] == image.read_bytes()
    assert BucketDashboard(output).snapshot()["problems"]["problems"][0]["initial_images"][0][
        "target_bbox_xyxy"] == [1, 0, 4, 3]
    previous_pointer = (output / "current.json").read_bytes()
    receipt["width"] = 6
    annotation.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="unchanged reset receipt"):
        publish(pool, reference, output, images_paths=[images])
    assert (output / "current.json").read_bytes() == previous_pointer
    data["pool_sha256"] = "other-pool"
    reference.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="immutable task pool"):
        publish(pool, reference, output)
    assert (output / "current.json").read_bytes() == previous_pointer


def test_policy_counts_only_real_rollouts_and_rejects_changed_policy_iteration(tmp_path):
    pool, reference, output, data, _ = inputs(tmp_path)
    measurements = tmp_path / "measurements.json"
    row = {
        "task": "grasp", "source": data["attempts"][0]["source"], "policy_step": 0,
        "assistance": .5, "success_interval_95": [.04, .7],
        "rollouts": [{"success": index == 0, "invalid_action_error": "boundary" if index == 1 else None,
                      "seed": index} for index in range(4)],
    }
    value = {"input_sha256": {"native_pool": data["pool_sha256"]}, "optimizer": None,
             "policy_sha256_before": "initial-policy", "repeats": 16,
             "groups": [row], "complete": False}
    measurements.write_text(json.dumps(value))
    publish(pool, reference, output, measurements)
    snapshot = BucketDashboard(output).snapshot()
    group = snapshot["calibration"]["groups"][0]
    assert (group["attempts"], group["successes"], group["invalid"]) == (4, 1, 1)
    assert group["planned_attempts"] == 16
    assert snapshot["manifest"]["progress"]["policy_rollouts"] == 4
    assert snapshot["manifest"]["progress"]["reference_attempts"] == 1
    previous_pointer = (output / "current.json").read_bytes()
    row["policy_step"] = 1
    measurements.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="bound task"):
        publish(pool, reference, output, measurements)
    assert (output / "current.json").read_bytes() == previous_pointer
