import os

import mujoco
import numpy as np
import pytest

from so101.native_pile_dashboard_images import boxed_image
import so101.native_pile_dashboard_images as renderer_module
from so101.pile_task_runtime import PileSimulation
from so101.task_image_roles import role_annotations


def test_boxes_are_drawn_at_exclusive_bounds_and_invisible_roles_stay_unboxed():
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    rows = [
        {"role": "source_object", "target_bbox_xyxy": [2, 2, 6, 6]},
        {"role": "destination", "target_bbox_xyxy": None},
    ]
    image = np.asarray(boxed_image(rgb, rows))
    assert image[2, 2].tolist() == [255, 221, 0]
    assert image[5, 5].tolist() == [255, 221, 0]
    assert image[6, 6].tolist() == [0, 0, 0]
    assert not rgb.any()


def test_destination_box_has_distinct_color():
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    image = np.asarray(boxed_image(rgb, [
        {"role": "destination", "target_bbox_xyxy": [1, 1, 5, 5]},
    ]))
    assert image[1, 1].tolist() == [40, 220, 255]


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") != "osmesa", reason="CPU GL integration")
def test_actual_same_camera_segmentation_produces_visible_source_and_destination_boxes():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <body name="target" pos="-.08 0 0"><geom type="box" size=".04 .04 .04"/></body>
      <body name="goal" pos=".08 0 0"><geom type="box" size=".03 .03 .01"/></body>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = 0
    camera.distance, camera.azimuth, camera.elevation = .5, 90, -90
    model.vis.quality.offsamples = 0
    task = {"task_type": "push", "target_instance_id": "target",
            "goal": {"destination_entity": "goal", "destination_xy": [.08, 0]}}
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
    annotations = role_annotations(model, data, task, segmentation)
    assert all(row["target_visible_pixels"] > 100 for row in annotations)
    assert annotations[0]["target_bbox_xyxy"] != annotations[1]["target_bbox_xyxy"]
    image = np.asarray(boxed_image(rgb, annotations))
    for row, color in zip(annotations, ([255, 221, 0], [40, 220, 255])):
        x, y, _, _ = row["target_bbox_xyxy"]
        assert image[y, x].tolist() == color


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") != "osmesa", reason="CPU GL integration")
def test_large_compiled_extent_does_not_clip_admitted_task_geometry(tmp_path, monkeypatch):
    xml = """<mujoco><statistic extent="57"/>
    <worldbody><geom name="table" type="plane" size="1 1 .01"/>
    <body name="target" pos=".22 0 .02"><geom type="box" size=".02 .02 .02"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sim = PileSimulation(model, data, None, {"target": model.body("target").id})
    (tmp_path / "model.xml").write_text(xml)
    (tmp_path / "integration_state.json").write_text("{}")
    problem = {"root": tmp_path, "task": {
        "task_id": "view", "task_type": "view", "target_instance_id": "target"}}
    monkeypatch.setattr(renderer_module, "load_problems", lambda entry: [problem])
    monkeypatch.setattr(renderer_module, "create_simulation", lambda problem: sim)
    monkeypatch.setattr(renderer_module, "restore_scene", lambda *args: None)
    monkeypatch.setattr(renderer_module, "reset_problem", lambda *args: None)
    receipt = renderer_module.render_scene({"split": "train", "files_sha256": {}}, tmp_path / "images")
    assert receipt["tasks"][0]["target_visible_pixels"] > 100
    assert receipt["tasks"][0]["target_bbox_status"] == "visible"
    assert receipt["tasks"][0]["camera_provenance"]["near_clip"] < .002
