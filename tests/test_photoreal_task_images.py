import json
import os
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from so101.pathtrace_pile import DEFAULT_PROFILE, render_pathtrace
from so101.photoreal_task_images import (
    capture_external_labels, export_scene, group_task_resets, mono_camera, world_mesh, write_ply,
)
from so101.task_image_roles import role_annotations


def test_world_mesh_uses_compiled_center_and_world_rotation_once():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><asset><mesh name="offset" vertex="1 0 0  2 0 0  1 1 0  1 0 1"
      face="0 2 1  0 1 3  0 3 2  1 2 3"/></asset>
      <worldbody><body pos=".2 .3 .4" euler="0 0 90">
      <geom type="mesh" mesh="offset"/></body></worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    vertices, normals = world_mesh(model, data, 0)
    expected = np.array([[.2, 1.3, .4], [.2, 2.3, .4], [-.8, 1.3, .4], [.2, 1.3, 1.4]])
    for point in expected:
        assert np.min(np.linalg.norm(vertices - point, axis=1)) < 1e-6
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)


def test_reset_grouping_distinguishes_full_state_not_scene_or_qpos(monkeypatch):
    states = {"first": np.array([.1, .2, 0.]), "second": np.array([.1, .2, 1.]),
              "same-as-first": np.array([.1, .2, 0.])}
    sim = SimpleNamespace(state=None)
    monkeypatch.setattr("so101.photoreal_task_images.load_problem", lambda entry, task: task)
    monkeypatch.setattr("so101.photoreal_task_images.reset_problem",
                        lambda sim, task: setattr(sim, "state", states[task]))
    monkeypatch.setattr("so101.photoreal_task_images.integration_state", lambda sim: sim.state.copy())
    tasks = [{"task_id": identity} for identity in states]
    groups = group_task_resets(sim, {"scene_id": "shared"}, tasks)
    assert [[task["task_id"] for task in group] for group in groups.values()] == [
        ["first", "same-as-first"], ["second"],
    ]


def test_binary_world_meshes_share_storage_only_for_identical_world_geometry(tmp_path):
    vertices = np.eye(3)
    normals = np.tile([0., 0., 1.], (3, 1))
    shared = tmp_path / "meshes"
    first, same, moved = [tmp_path / name for name in ("first.ply", "same.ply", "moved.ply")]
    write_ply(first, vertices, normals, shared)
    write_ply(same, vertices, normals, shared)
    write_ply(moved, vertices + [1, 0, 0], normals, shared)
    assert first.samefile(same)
    assert not first.samefile(moved)
    assert len(list(shared.glob("*.ply"))) == 2
    assert b"format binary_little_endian 1.0" in first.read_bytes()


def test_source_and_destination_visibility_are_independent():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><body name="target"><geom type="box" size=".01 .01 .01"/></body>
      <body name="goal" pos=".2 .1 0"><geom type="cylinder" size=".02 .001"/></body>
      </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    task = {"task_type": "push", "target_instance_id": "target",
            "goal": {"destination_entity": "goal", "destination_xy": [.2, .1]}}
    segmentation = np.full((2, 3, 2), -1, dtype=int)
    segmentation[0, 0] = [0, mujoco.mjtObj.mjOBJ_GEOM]
    source, destination = role_annotations(model, data, task, segmentation)
    assert source["target_bbox_xyxy"] == [0, 0, 1, 1]
    assert destination["target_bbox_status"] == "not_visible"
    assert destination["target_bbox_xyxy"] is None
    segmentation[1, 2] = [1, mujoco.mjtObj.mjOBJ_GEOM]
    source, destination = role_annotations(model, data, task, segmentation)
    assert destination["target_bbox_xyxy"] == [2, 1, 3, 2]
    assert destination["entity_kind"] == "goal_marker"
    assert "not tolerance-region/amodal" in destination["bbox_semantics"]

@pytest.mark.skipif(not os.environ.get("LEGO_PATHTRACE_PYTHON"), reason="Optional CPU backend absent")
def test_actual_export_matches_segmentation_and_rejects_wrong_camera(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    xml = """<mujoco><visual><global offwidth="64" offheight="48"/></visual>
      <worldbody><geom name="table" type="plane" size="2 2 .1"/>
      <body pos=".02 .01 .03"><geom type="box" size=".03 .02 .03" rgba=".2 .5 .8 1"/>
      </body></worldbody></mujoco>"""
    (source / "model.xml").write_text(xml)
    (source / "integration_state.json").write_text("{}")
    (source / "problem.json").write_text("{}")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = data.qpos.copy()
    with mujoco.Renderer(model, height=48, width=64) as renderer:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.02, .01, .03]
        camera.distance, camera.azimuth, camera.elevation = .4, 135, -65
        renderer.update_scene(data, camera=camera)
        geometry_camera = mono_camera(renderer.scene)
        np.testing.assert_allclose(
            geometry_camera["origin"], np.mean([eye.pos for eye in renderer.scene.camera], axis=0),
        )
        rgb = renderer.render().copy()
        samples = model.vis.quality.offsamples
        segmentation = capture_external_labels(
            SimpleNamespace(model=model, data=data, renderer=renderer), camera,
        )
        assert model.vis.quality.offsamples == samples
        np.testing.assert_array_equal(renderer.render(), rgb)
        assert np.any(segmentation[:, :, 0] == 1)
    np.savez_compressed(tmp_path / "segmentation.npz", segmentation=segmentation)
    scene_path = export_scene(SimpleNamespace(model=model, data=data), source,
                              tmp_path / "export", geometry_camera)
    profile = json.loads(DEFAULT_PROFILE.read_text())
    profile.update(width=64, height=48, samples=4)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    receipt = render_pathtrace(scene_path, tmp_path / "correct", profile_path)
    assert receipt["segmentation_alignment"]["mismatch_fraction"] <= .01
    np.testing.assert_array_equal(data.qpos, before)
    description = json.loads(scene_path.read_text())
    description["camera"]["target"][0] += .2
    scene_path.write_text(json.dumps(description))
    with pytest.raises(ValueError, match="alignment failed"):
        render_pathtrace(scene_path, tmp_path / "wrong", profile_path)
