import json
import os

import mujoco
import numpy as np
import pytest

from so101.native_pile_dashboard_images import boxed_image
import so101.native_pile_dashboard_images as renderer_module
from so101.pile_task_runtime import PileSimulation
from so101.task_image_roles import role_annotations
from so101.bucket_dashboard import BucketDashboard


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


def test_compact_manifest_boundary_preserves_last_valid_publication(tmp_path, monkeypatch):
    manifest = {"kind": "lego-data-buckets", "schema_version": 1, "snapshot_id": "original",
                "problems": [], "groups": [], "buckets": []}
    size = len(json.dumps(manifest, separators=(",", ":")).encode())
    monkeypatch.setattr(renderer_module, "MAX_MANIFEST_BYTES", size)
    output = tmp_path / "current.json"
    path = renderer_module.publish_manifest(manifest, tmp_path, tmp_path, output)
    assert path.stat().st_size == size
    assert BucketDashboard(tmp_path).publication("current.json")["manifest"]["snapshot_id"] == "original"
    pointer = output.read_bytes()
    with pytest.raises(ValueError, match="current publication unchanged"):
        renderer_module.publish_manifest({**manifest, "snapshot_id": "original+"},
                                         tmp_path, tmp_path, output)
    assert output.read_bytes() == pointer


def test_partitioned_population_preserves_images_and_rejects_oversized_activation(tmp_path, monkeypatch):
    image = b"\x89PNG\r\n\x1a\ncamera"
    (tmp_path / "task.png").write_bytes(image)
    rows = [{"id": f"task{i}", "initial_images": [
        {"path": "task.png", "sha256": renderer_module.digest(image)}]} for i in range(12)]
    manifest = {"kind": "lego-data-buckets", "schema_version": 1, "snapshot_id": "partitioned",
                "problems": rows, "groups": [], "buckets": []}
    monkeypatch.setattr(renderer_module, "MAX_MANIFEST_BYTES", 1024)
    monkeypatch.setattr(renderer_module, "MAX_PROBLEM_SHARD_BYTES", 700)
    output = tmp_path / "current.json"
    path = renderer_module.publish_manifest(manifest, tmp_path, tmp_path, output, partition=True)
    header = json.loads(path.read_bytes())
    assert path.stat().st_size <= 1024
    assert header["problems"] == []
    assert len(header["problem_shards"]) > 1
    browser = BucketDashboard(tmp_path)
    assert [r["id"] for r in browser.snapshot()["problems"]["problems"]] == [r["id"] for r in rows]
    assert browser.image("task0", 0) == (image, "image/png")
    pointer = output.read_bytes()
    monkeypatch.setattr(renderer_module, "MAX_PROBLEM_SHARD_TOTAL_BYTES", 1)
    with pytest.raises(ValueError, match="total byte limit"):
        renderer_module.publish_manifest(manifest, tmp_path, tmp_path, output, partition=True)
    assert output.read_bytes() == pointer
    assert browser.image("task0", 0) == (image, "image/png")
    shard = tmp_path / header["problem_shards"][0]["path"]
    shard.write_bytes(shard.read_bytes().replace(b"task0", b"TASK0"))
    with pytest.raises(ValueError, match="hash mismatch"):
        browser.snapshot()


@pytest.mark.parametrize("changed_rendering", [False, True])
def test_legacy_receipts_reused_only_for_identical_rendering(tmp_path, monkeypatch, changed_rendering):
    entry = {"split": "train", "files_sha256": {}}
    image = b"immutable rendered image"
    (tmp_path / "image.png").write_bytes(image)
    receipt = {"contract": {
        "admission_sha256": renderer_module.digest(json.dumps(entry, sort_keys=True).encode()),
        "implementation_sha256": renderer_module.LEGACY_IMPLEMENTATION_SHA,
        "mujoco_version": mujoco.__version__},
        "pile": {"image_file": "image.png", "sha256": renderer_module.digest(image)}, "tasks": []}
    (tmp_path / "scene.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(renderer_module, "load_problems", lambda entry: [{}])
    if changed_rendering:
        monkeypatch.setattr(renderer_module, "render_source_sha", lambda: "changed rendering")
        with pytest.raises(ValueError, match="render contract changed"):
            renderer_module.render_scene(entry, tmp_path)
    else:
        assert renderer_module.render_scene(entry, tmp_path) == receipt


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
