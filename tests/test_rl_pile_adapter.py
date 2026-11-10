import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import torch

from so101.lego_scene import capture_scene_state
from so101.pile_task_runtime import GRADER_VERSION
from so101.rl_pile_adapter import NativePileSimulation, load_native_pool
from so101.robot_sim import BeamSimulation, MODEL_ROOT, scene_xml
from so101.simulation_corpus import TASKS
from so101.train_trajectory_rl import rollout


@pytest.fixture
def native_pool(tmp_path, monkeypatch):
    class Renderer:
        def __init__(self, *args, **kwargs):
            self._gl_context = SimpleNamespace(make_current=lambda: None)
            self._mjr_context = object()

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", Renderer)
    if not (MODEL_ROOT / "so101.xml").is_file():
        pytest.skip("pinned robot assets unavailable")
    root = ET.fromstring(scene_xml(MODEL_ROOT))
    world = root.find("worldbody")
    for index in range(2):
        body = ET.SubElement(world, "body", name=f"instance_{index}", pos=f".22 {index * .05} .02")
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "geom", name=f"instance_geom_{index}", type="box",
                      size=".01 .01 .01", pos=".003 0 0", mass=".01")
    xml = ET.tostring(root, encoding="unicode")
    sim = BeamSimulation(model=mujoco.MjModel.from_xml_string(xml), render=False)
    try:
        mujoco.mj_forward(sim.model, sim.data)
        tasks = []
        for task in TASKS:
            goal = {"dwell_steps": 2, "minimum_pixels": 64, "center_radius": .2}
            if task == "grasp":
                goal["lift_m"] = .05
            if task in ("push", "place"):
                goal.update(destination_entity="goal_left",
                            destination_xy=sim.position("goal_left")[:2].tolist(),
                            tolerance_m=.025, maximum_surface_speed_m_s=.01)
            tasks.append({
                "task_id": task, "task_type": task, "target_instance_id": "instance_0",
                "grader_version": GRADER_VERSION, "goal": goal,
                "initial_target_position": sim.data.body("instance_0").xipos.tolist(),
            })
        scene = {
            "scene_id": "scene", "scene_family_id": "family", "seed": 123,
            "instances": [
                {"instance_id": f"instance_{index}", "body_name": f"instance_{index}",
                 "geom_names": [f"instance_geom_{index}"]} for index in range(2)
            ],
            "tasks": tasks, "capabilities": {
                "reset": "mujoco-integration-v1", "target_labels": "instance-segmentation-v1",
                "expert_references": False, "sft_qualified": False,
            },
        }
        snapshot = capture_scene_state(sim.model, sim.data, xml, {
            "scene_id": "scene", "catalog_sha256": "catalog",
        }, status="settled")
        mujoco.mj_saveModel(sim.model, str(tmp_path / "model.mjb"), None)
        (tmp_path / "model.xml").write_text(xml)
        for name, value in (("problem.json", scene), ("integration_state.json", snapshot),
                            ("scene_plan.json", {})):
            (tmp_path / name).write_text(json.dumps(value))
        entry = {
            "scene_id": "scene", "scene_family_id": "family", "split": "train", "root": str(tmp_path),
            "files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in tmp_path.iterdir()},
        }
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": 1, "kind": "lego-pile-task-pool", "status": "complete",
            "expert_references": False, "scenes": [entry],
        }))
        yield manifest, sim
    finally:
        sim.close()


def test_native_adapter_exact_reset_com_action_grading_and_no_legacy_object_lookup(native_pool, monkeypatch):
    manifest, original = native_pool
    pool = load_native_pool(manifest, set())
    episode = pool["view"][0]
    sim = NativePileSimulation(episode, MODEL_ROOT)
    try:
        np.testing.assert_array_equal(sim.data.qpos, original.data.qpos)
        np.testing.assert_array_equal(sim.model.actuator_forcerange, original.model.actuator_forcerange)
        np.testing.assert_array_equal(sim.position("instance_0"), sim.native.position("instance_0"))
        assert not np.array_equal(sim.position("instance_0"), sim.data.body("instance_0").xpos)
        # Label fixture deliberately does not claim GL occlusion qualification.
        monkeypatch.setattr(sim.native, "instance_box", lambda target: {
            "visible": True, "visible_pixels": 100, "box_xyxy": [.4, .4, .6, .6],
        })
        controller = SimpleNamespace(
            reset=lambda *args, **kwargs: None, command_inputs=lambda *args: (),
            observe=lambda *args: None, device="cpu",
            action_mean=sim.state(), action_std=np.ones(6),
            config=SimpleNamespace(history_steps=4),
            policy=SimpleNamespace(
                eval=lambda: None, config=SimpleNamespace(chunk_size=4),
                model=SimpleNamespace(sample_actions_entity=lambda: (torch.zeros(1, 4, 6), None)),
            ),
        )
        original._native_simulation = sim
        trajectory = rollout(original, controller, episode, 123, 10, 4,
                             {"optimal_steps": 5., "jerk_p10": 1.}, True)
        assert trajectory.success and trajectory.steps == 2
        assert not trajectory.wrong_object
        sim.reset_native(episode)
        np.testing.assert_array_equal(sim.data.qpos, original.data.qpos)
        with pytest.raises(ValueError, match="held-out"):
            load_native_pool(manifest, {"family"})
    finally:
        sim.close()


def test_native_gpu_pathtrace_uses_native_visual_topology_without_changing_physics(native_pool, monkeypatch):
    manifest, original = native_pool
    captured = []

    class Tracer:
        def __init__(self, model, profile, *, variant):
            captured.append((model, variant))
            self.profile = profile
            self.mi = SimpleNamespace(__version__="fixture")

        def render(self, data, frame):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    monkeypatch.setattr("so101.rl_renderer.FrameRenderer", Tracer)
    episode = load_native_pool(manifest, set())["view"][0]
    sim = NativePileSimulation(episode, MODEL_ROOT, backend="gpu-pathtrace", samples=2)
    try:
        monkeypatch.setattr(sim, "_activate_context", lambda: None)
        assert captured[0][1] == "cuda_ad_rgb"
        assert sim.rgb_model.body("instance_0").id == sim.model.body("instance_0").id
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(sim.model, spec))
        after = np.empty_like(before)
        mujoco.mj_getState(sim.model, sim.data, before, spec)
        assert sim.image().shape == (240, 320, 3)
        mujoco.mj_getState(sim.model, sim.data, after, spec)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_allclose(sim.rgb_data.cam_xpos, sim.data.cam_xpos, atol=1e-12)
        np.testing.assert_allclose(sim.rgb_data.cam_xmat, sim.data.cam_xmat, atol=1e-12)
        assert sim.rgb_identity["scene_xml_sha256"] == hashlib.sha256(
            (Path(episode.metadata["native_entry"]["root"]) / "model.xml").read_bytes(),
        ).hexdigest()
    finally:
        sim.close()
