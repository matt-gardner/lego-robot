import hashlib
import json
from pathlib import Path

import pytest

from scripts import import_pile_directories as migration
from so101.bucket_store import import_current
from so101.pile_directory import current_image, current_inventory


def source_fixture(tmp_path, monkeypatch, override=None):
    source = tmp_path / "source"
    source.mkdir()
    definition = {"task_id": "scene:brick_0:grasp", "task_type": "grasp",
                  "target_instance_id": "brick_0", "goal": {"lift_height": 0.02}}
    if override is not None:
        definition["initial_state"] = override
    raw = {"scene_id": "scene", "tasks": [definition],
           "instances": [{"instance_id": "brick_0", "part_id": "3005"}], "tags": {}}
    for name, value in (("problem.json", raw), ("scene_plan.json", {}),
                        ("integration_state.json", {"scene_id": "scene", "qpos": [0.1]})):
        (source / name).write_text(json.dumps(value))
    (source / "model.mjb").write_bytes(b"compiled")
    (source / "model.xml").write_text("<mujoco/>")
    monkeypatch.setattr(migration, "arm_joint_addresses", lambda path: {"shoulder": 0})
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    image = b"\x89PNG\r\n\x1a\noriginal"
    (legacy / "pile.png").write_bytes(image)
    record = {
        "id": "original-task", "task": "grasp", "bucket": "current:1-pieces",
        "source": {"scene_id": "scene", "task_id": definition["task_id"],
                   "episode_source": str(source / "tasks" / definition["task_id"])},
        "tags": {}, "initial_images": [],
    }
    document = {
        "manifest": {"datasets": []}, "problems": {"problems": [record], "buckets": []},
        "admitted_piles": [{"id": "old-pile", "scene_id": "scene", "initial_images": [
            {"path": "pile.png", "root_relative": True, "sha256": hashlib.sha256(image).hexdigest()}],
            "pieces": [{"id": "brick_0", "target_bbox_xyxy": [1, 2, 3, 4]}]}],
        "calibration": {"groups": []}, "distance_pilots": [],
    }
    import_current(legacy, document)
    return source, definition, legacy, image


def test_export_preserves_original_bytes_and_arm_positions_but_does_not_invent_checkpoints(tmp_path, monkeypatch):
    source, definition, legacy, image = source_fixture(tmp_path, monkeypatch)
    assistance = tmp_path / "assistance"
    (assistance / "receipts").mkdir(parents=True)
    receipt = {
        "status": "complete", "inventory_sha256": "static-protocol",
        "source_binding": {"scene_id": "scene", "task_id": definition["task_id"],
                           "task_sha256": migration.identity(definition)},
        "levels": [{"value": index / 20} for index in range(20)],
    }
    (assistance / "receipts/job.json").write_text(json.dumps(receipt))
    root = tmp_path / "canonical"
    result = migration.export(legacy, root, [assistance])
    assert not (root / ".lego-data.lock").exists()
    assert result["piles"] == result["tasks"] == result["rendered"] == 1
    assert result["assistance_receipts"] == 1
    directory = root / "piles/pile_0001"
    task = json.loads((directory / "tasks/task_001.json").read_bytes())
    assert task["initial_arm_position"] == {"shoulder": 0.1}
    assert task["part_id"] == "3005"
    assert task["assistance_checkpoints"] == []
    assert not task["assistance_breakdowns"][0]["checkpoint_states_persisted"]
    assert task["assistance_breakdowns"][0]["measurement_inventory_sha256"] == "static-protocol"
    assert (directory / "model.mjb").stat().st_ino == (source / "model.mjb").stat().st_ino
    assert current_image(root, "pile_0001", 0) == (image, "image/png")
    assert current_inventory(root)["problems"]["problems"][0]["id"] == "original-task"


def test_task_specific_initial_arm_state_is_not_replaced_by_pile_default(tmp_path, monkeypatch):
    _, _, legacy, _ = source_fixture(tmp_path, monkeypatch, {"scene_id": "scene", "qpos": [0.7]})
    root = tmp_path / "canonical"
    migration.export(legacy, root, [])
    task = json.loads((root / "piles/pile_0001/tasks/task_001.json").read_bytes())
    assert task["initial_arm_position"] == {"shoulder": 0.7}


def test_inherited_baseline_requires_admission_hashes_and_exact_task_override(tmp_path, monkeypatch):
    source, _, legacy, _ = source_fixture(
        tmp_path, monkeypatch, {"scene_id": "scene", "qpos": [0.7]})
    state = {"scene_id": "parent", "qpos": [0.1],
             "model_xml_sha256": migration.digest(source / "model.xml")}
    (source / "integration_state.json").write_text(json.dumps(state))
    (source / "scene_plan.json").write_text(json.dumps({"scene_id": "parent"}))
    admission = {"files_sha256": {name: migration.digest(source / name)
                                  for name in migration.SOURCE_FILES}}
    (source / "admission.json").write_text(json.dumps(admission))
    root = tmp_path / "canonical"
    migration.export(legacy, root, [])
    task = json.loads((root / "piles/pile_0001/tasks/task_001.json").read_bytes())
    assert task["initial_arm_position"] == {"shoulder": 0.7}
    admission["files_sha256"]["model.mjb"] = "different"
    (source / "admission.json").write_text(json.dumps(admission))
    with pytest.raises(ValueError, match="without exact task overrides"):
        migration.export(legacy, tmp_path / "invalid", [])


def test_cross_pile_initial_state_aborts_before_exposing_inventory(tmp_path, monkeypatch):
    _, _, legacy, _ = source_fixture(tmp_path, monkeypatch, {"scene_id": "other", "qpos": [0.7]})
    root = tmp_path / "canonical"
    with pytest.raises(ValueError, match="different pile"):
        migration.export(legacy, root, [])
    assert (root / "piles/.incomplete").exists()
    assert not (root / "piles/metadata.json").exists()


def test_saved_assistance_state_is_attached_to_exact_parent_without_requalification(tmp_path, monkeypatch):
    source, definition, legacy, _ = source_fixture(tmp_path, monkeypatch)
    document = migration.current_inventory(legacy)
    variant = {
        "id": "saved-child",
        "source": {"scene_id": "scene", "task_id": definition["task_id"],
                   "path": str(source / "problem.json")},
        "tags": {"parent_task_id": definition["task_id"],
                 "task_assistance": {"kind": "direct-task-distance-v2",
                                     "initial_task_state": {"lifted_once": False}}},
    }
    document["distance_pilots"] = [{"problems": [variant]}]
    monkeypatch.setattr(migration, "current_inventory", lambda path: document)
    root = tmp_path / "canonical"
    result = migration.export(legacy, root, [])
    task = json.loads((root / "piles/pile_0001/tasks/task_001.json").read_bytes())
    checkpoint = task["assistance_checkpoints"][0]
    assert result["assistance_checkpoints"] == 1
    assert checkpoint["initial_state"] == {"scene_id": "scene", "qpos": [0.1]}
    assert checkpoint["task_assistance"] == variant["tags"]["task_assistance"]
    assert checkpoint["qualification"] is None
    assert task["assistance_checkpoint_status"] == "persisted"
    variant["tags"]["parent_task_id"] = "missing-parent"
    with pytest.raises(ValueError, match="no primary parent"):
        migration.export(legacy, tmp_path / "invalid", [])


def test_refresh_attaches_only_finalized_receipts_idempotently_without_changing_task_state(tmp_path, monkeypatch):
    _, definition, legacy, _ = source_fixture(tmp_path, monkeypatch)
    root = tmp_path / "canonical"
    migration.export(legacy, root, [])
    task_path = root / "piles/pile_0001/tasks/task_001.json"
    original = json.loads(task_path.read_bytes())
    assistance = tmp_path / "fast-v4"
    receipts = assistance / "receipts"
    receipts.mkdir(parents=True)
    receipt = {
        "status": "unavailable", "inventory_sha256": "older-measurement-inventory",
        "retained_from": {"inventory_sha256": "original"},
        "source_binding": {"scene_id": "scene", "task_id": definition["task_id"],
                           "task_sha256": migration.identity(definition)},
        "levels": [],
    }
    (receipts / "job.json").write_text(json.dumps(receipt))
    (receipts / "inflight.json.tmp").write_text("unfinished")
    assert migration.refresh_assistance(root, [assistance]) == {
        "receipts": 1, "attached": 1, "tasks_updated": 1}
    updated = json.loads(task_path.read_bytes())
    reference = updated.pop("assistance_breakdowns")[0]
    original.pop("assistance_breakdowns")
    assert updated == original
    assert reference["measurement_inventory_sha256"] == "older-measurement-inventory"
    assert reference["retained_from"] == receipt["retained_from"]
    assert reference["status"] == "unavailable"
    assert not reference["checkpoint_states_persisted"]
    previous_mtime = task_path.stat().st_mtime_ns
    assert migration.refresh_assistance(root, [assistance])["attached"] == 0
    assert task_path.stat().st_mtime_ns == previous_mtime
    receipt["source_binding"]["task_sha256"] = "different"
    (receipts / "bad.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="different source task"):
        migration.refresh_assistance(root, [assistance])
