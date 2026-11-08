import hashlib
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np
import pytest
import torch

from so101.collect_catalog_sim import load_appearance, record_pair, validate_recording
from so101.control_model.simulation_dataset import EntityTrajectoryDataset
from so101.episode_storage import EpisodeStorage, consolidate
from so101.renderer_appearance_v3 import DEFAULT_PROFILE_V3
from so101.renderer_appearance_v4 import DEFAULT_PROFILE_V4
from so101.catalog_sim import CatalogSimulation
from so101.robot_sim import BeamSimulation


def fixture_sim(tmp_path, *, recovered=False):
    manifest = tmp_path / "fixture_catalog.json"
    manifest.write_text(json.dumps({"schema_version": 1, "catalog_id": "fixture", "parts": []}))
    sim = SimpleNamespace(
        xml="<mujoco/>", settled_snapshot={"fixture": True},
        catalog=SimpleNamespace(catalog_id="fixture", manifest_path=manifest,
                                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()),
        plan={
            "scene_id": "fixture-scene", "seed": 1, "pair_id": "fixture-pair",
            "tags": {"category": "separated"},
            "commands": [
                {"command_id": f"fixture-{i}", "task": "grasp",
                 "object_instance_id": f"object_{i}", "destination_id": None}
                for i in range(2)
            ],
        },
        initial={"expert_acceptance_version": 1, "simulation_backend": "catalog-v1",
                 "scene_version": 4},
        target_labels={"object_0": "fixture0", "object_1": "fixture1"},
        fps=10, model=mujoco.MjModel.from_xml_string("<mujoco/>"), rows=[],
        appearance=None, data=None,
    )
    def restore():
        sim.rows = []
        sim.episode_root = None
    def demonstrate(command):
        for i in range(2):
            for folder in ("wrist", "external"):
                image = np.zeros((240, 320, 3), dtype=np.uint8)
                image[:, 160:] = 255
                assert cv2.imwrite(str(sim.episode_root / folder / f"{i:06}.jpg"),
                                   image)
            sim.rows.append({
                "state": [0.] * 6, "action": [0.] * 6, "qpos": [0.] * sim.model.nq,
                "elapsed": i / 10, "phase": "grasp_retry" if recovered else "approach",
                "command_entity_ids": [command["object_instance_id"]],
                "command_roles": ["object"], "command_boxes": [[0.] * 5],
                "command_visibility": [1], "command_depth_m": [.1],
                "command_depth_valid": [True],
                "action_interval_max_contact_penetration_m": 0.,
                "action_interval_max_table_penetration_m": 0.,
                "image": f"wrist/{i:06}.jpg", "external_image": f"external/{i:06}.jpg",
            })
        return {"success": True}
    sim.restore_initial = restore
    sim.observe_recovery_start = lambda *args: None
    sim.demonstrate_scene_command = demonstrate
    return sim


def test_pair_publication_requires_two_clean_recordings(tmp_path):
    output = tmp_path / "new-pair"
    report = record_pair(fixture_sim(tmp_path), output)
    assert report["status"] == "complete"
    assert len(report["successful_episodes"]) == 2
    for source in report["successful_episodes"]:
        root = Path(source)
        metadata = json.loads((root / "episode.json").read_text())
        assert metadata["release_id"] == report["release_id"]
        assert metadata["scene_model"] == "model.xml"
        assert (root / "model.mjb").read_bytes() == (output / "model.mjb").read_bytes()
        assert (root / "source" / "hashes.json").is_file()
    assert json.loads((output / "collection.json").read_text()) == report
    with pytest.raises(FileExistsError):
        record_pair(fixture_sim(tmp_path), output)


def test_recovered_success_never_enters_clean_pair(tmp_path, monkeypatch):
    monkeypatch.setattr("so101.collect_catalog_sim.capture_scene_state",
                        lambda *args, **kwargs: {"status": "failed-attempt"})
    output = tmp_path / "rejected-pair"
    report = record_pair(fixture_sim(tmp_path, recovered=True), output)
    assert report["status"] == "incomplete"
    assert report["successful_episodes"] == []
    assert "generated_at" not in report
    assert all(attempt["outcome"] == "failed" for attempt in report["attempts"])
    for index in range(2):
        metadata = json.loads((output / f"command_{index}" / "episode.json").read_text())
        assert metadata["outcome"] == "failed"
        assert "retry" in metadata["verification"]["reason"]
        assert (output / f"command_{index}" / "frames.jsonl").exists()
        assert metadata["failure_state"] == "failure_state.json"
        assert "generated_at" not in metadata


def test_generation_dates_capture_successful_completion_and_survive_storage(tmp_path, monkeypatch):
    moments = [
        datetime(2026, 11, 8, 18, 0, second, tzinfo=timezone.utc)
        for second in (1, 2, 3)
    ]
    clock = iter(moments)
    def now(zone):
        assert zone is timezone.utc
        return next(clock)
    monkeypatch.setattr("so101.collect_catalog_sim.datetime", SimpleNamespace(now=now))
    output = tmp_path / "pair"
    report = record_pair(fixture_sim(tmp_path), output)
    assert report["generated_at"] == moments[2].isoformat()
    for index, source in enumerate(report["successful_episodes"]):
        metadata = json.loads((Path(source) / "episode.json").read_bytes())
        assert metadata["generated_at"] == moments[index].isoformat()
    container = tmp_path / "pair.sqlite"
    consolidate(output, container)
    with EpisodeStorage(container) as storage:
        assert storage.catalog["generated_at"] == report["generated_at"]
        for source in report["successful_episodes"]:
            assert storage.read(Path(source), "episode.json") == (Path(source) / "episode.json").read_bytes()


def test_reserved_family_membership_is_preserved_in_both_commands(tmp_path):
    sim = fixture_sim(tmp_path)
    sim.plan.update(scene_family_id="reserved-parent-family", split="development")
    report = record_pair(sim, tmp_path / "pair")
    assert report["scene_family_id"] == "reserved-parent-family"
    assert report["split"] == "development"
    for source in report["successful_episodes"]:
        metadata = json.loads((Path(source) / "episode.json").read_bytes())
        assert metadata["scene_family_id"] == report["scene_family_id"]
        assert metadata["split"] == report["split"]


@pytest.mark.parametrize("field,value", [
    ("command_boxes", [[0.] * 4]),
    ("command_visibility", [[1]]),
    ("action_interval_max_table_penetration_m", float("nan")),
    ("elapsed", float("nan")),
])
def test_recording_rejects_invalid_label_shapes_and_nonfinite_evidence(tmp_path, field, value):
    sim = fixture_sim(tmp_path)
    record_pair(sim, tmp_path / "pair")
    root = tmp_path / "pair" / "command_1"
    metadata = json.loads((root / "episode.json").read_text())
    sim.rows[0][field] = value
    with pytest.raises(ValueError):
        validate_recording(root, metadata, sim.rows, sim.model.nq)


def test_recording_rejects_uniform_camera_output(tmp_path):
    sim = fixture_sim(tmp_path)
    record_pair(sim, tmp_path / "pair")
    root = tmp_path / "pair" / "command_1"
    assert cv2.imwrite(str(root / sim.rows[0]["image"]),
                       np.zeros((240, 320, 3), dtype=np.uint8))
    metadata = json.loads((root / "episode.json").read_text())
    with pytest.raises(ValueError, match="Uniform catalog image"):
        validate_recording(root, metadata, sim.rows, sim.model.nq)


def test_recorded_pair_loads_through_existing_immutable_sft_reader(tmp_path):
    output = tmp_path / "pair"
    report = record_pair(fixture_sim(tmp_path), output)
    container = tmp_path / "pair.sqlite"
    consolidate(output, container)
    with EpisodeStorage(container) as storage:
        storage.verify_sources(output)
        for source in report["successful_episodes"]:
            assert storage.read(Path(source), "model.mjb") == (output / "model.mjb").read_bytes()
    dataset = EntityTrajectoryDataset(output, horizon=3, history_steps=4,
                                      storage_path=container)
    try:
        assert len(dataset.episodes) == 2
        assert dataset.pair_groups == [[0, 1]]
        for index in range(len(dataset)):
            sample = dataset[index]
            assert sample["images"].shape == (4, 3, 240, 320)
            assert sample["actions"].shape == (3, 6)
            assert all(torch.isfinite(value).all() for value in sample.values()
                       if isinstance(value, torch.Tensor))
    finally:
        dataset.storage.close()


def test_v3_recording_archives_the_exact_inherited_profile_and_sources(tmp_path):
    sim = fixture_sim(tmp_path)
    sim.appearance = load_appearance(DEFAULT_PROFILE_V3)
    sim.initial["renderer"] = sim.appearance.identity()
    output = tmp_path / "pair"
    record_pair(sim, output)
    source = output / "command_0" / "source"
    hashes = json.loads((source / "hashes.json").read_text())
    assert hashes["renderer_profiles/wrist_soft_v2.json"] == sim.initial["renderer"]["parent_profile_sha256"]
    assert (source / "renderer_appearance_v3.py").is_file()
    assert (source / "renderer_appearance_v2.py").is_file()
    assert (source / "renderer_appearance.py").is_file()
    assert json.loads((source / "parts_catalog.json").read_bytes())["catalog_id"] == "fixture"


def test_recording_rejects_catalog_manifest_changed_after_loading(tmp_path):
    sim = fixture_sim(tmp_path)
    sim.catalog.manifest_path.write_text("changed")
    with pytest.raises(ValueError, match="manifest changed"):
        record_pair(sim, tmp_path / "pair")


def test_v4_recording_preserves_both_parent_profiles_and_all_implementations(tmp_path):
    sim = fixture_sim(tmp_path)
    sim.appearance = load_appearance(DEFAULT_PROFILE_V4)
    sim.initial["renderer"] = sim.appearance.identity()
    output = tmp_path / "pair"
    record_pair(sim, output)
    source = output / "command_0" / "source"
    hashes = json.loads((source / "hashes.json").read_bytes())
    for version in (2, 3, 4):
        name = f"renderer_profiles/wrist_soft_v{version}.json"
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == hashes[name]
    for suffix in ("", "_v2", "_v3", "_v4"):
        name = f"renderer_appearance{suffix}.py"
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == hashes[name]


def test_producer_publishes_cross_uid_readable_archive(tmp_path):
    from so101.produce_catalog import qualify_pair

    output = tmp_path / "pair"
    record_pair(fixture_sim(tmp_path), output)
    result = qualify_pair(output)
    assert result["samples"] == 4
    archive = output / "episodes.sqlite"
    assert archive.stat().st_mode & 0o777 == 0o444
    assert not (output / ".qualified-episodes.sqlite").exists()
    assert qualify_pair(output) == result


def test_catalog_view_preflights_equivalent_heading_before_recorded_motion(monkeypatch):
    sim = CatalogSimulation.__new__(CatalogSimulation)
    sim.position = lambda target: np.array([.2, 0., .003])
    sim.object_yaw = lambda target: 0.
    sim.yaws = {}
    calls = []
    def ik(position, heading):
        calls.append((position.copy(), heading))
        if heading < math.pi:
            raise RuntimeError("Unreachable heading")
        return np.zeros(6)
    sim.ik = ik
    def demonstrate(instance, target):
        assert len(calls) == 3
        assert instance.yaws[target] == pytest.approx(math.pi)
        return {"success": True}
    monkeypatch.setattr(BeamSimulation, "demonstrate_view", demonstrate)
    assert sim.demonstrate_view("target") == {"success": True}
    assert calls[-1][0] == pytest.approx([.21, -.01, .065])
