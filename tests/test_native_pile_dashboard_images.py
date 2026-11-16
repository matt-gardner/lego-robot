import json
import os

import mujoco
import pytest

import so101.native_pile_dashboard_images as renderer_module
from so101.bucket_dashboard import BucketDashboard
from so101.pile_task_runtime import PileSimulation


def test_current_manifest_failure_preserves_last_valid_publication(tmp_path, monkeypatch):
    manifest = {"kind": "lego-data-buckets", "schema_version": 1, "snapshot_id": "original",
                "problems": [], "groups": [], "buckets": []}
    output = tmp_path / "current.sqlite3"
    assert renderer_module.publish_manifest(manifest, tmp_path, tmp_path, output) == output
    before = BucketDashboard(tmp_path).snapshot()
    monkeypatch.setattr("so101.bucket_store.MAX_INVENTORY_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds byte limit"):
        renderer_module.publish_manifest({**manifest, "snapshot_id": "changed"},
                                         tmp_path, tmp_path, output)
    assert BucketDashboard(tmp_path).snapshot() == before


@pytest.mark.parametrize("changed_rendering", [False, True])
def test_receipts_reused_only_for_identical_rendering(tmp_path, monkeypatch, changed_rendering):
    entry = {"split": "train", "files_sha256": {}}
    image = b"immutable rendered image"
    (tmp_path / "image.png").write_bytes(image)
    receipt = {"contract": {
        "admission_sha256": renderer_module.digest(json.dumps(entry, sort_keys=True).encode()),
        "implementation_sha256": renderer_module.render_source_sha(),
        "mujoco_version": mujoco.__version__},
        "pile": {"image_file": "image.png", "sha256": renderer_module.digest(image)},
        "pieces": []}
    (tmp_path / "scene.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(renderer_module, "load_problems", lambda entry: [{}])
    if changed_rendering:
        monkeypatch.setattr(renderer_module, "render_source_sha", lambda: "changed rendering")
        with pytest.raises(ValueError, match="render contract changed"):
            renderer_module.render_scene(entry, tmp_path)
    else:
        assert renderer_module.render_scene(entry, tmp_path) == receipt


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") != "osmesa", reason="CPU GL integration")
def test_large_extent_pile_capture_has_piece_coordinates_and_no_task_images(tmp_path, monkeypatch):
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
    problem = {"root": tmp_path, "task": {"task_id": "view", "task_type": "view"},
               "scene": {"instances": [{"instance_id": "target", "body_name": "target"}]}}
    monkeypatch.setattr(renderer_module, "load_problems", lambda entry: [problem])
    monkeypatch.setattr(renderer_module, "create_simulation", lambda problem: sim)
    monkeypatch.setattr(renderer_module, "restore_scene", lambda *args: None)
    receipt = renderer_module.render_scene({"split": "train", "files_sha256": {}}, tmp_path / "images")
    assert receipt["pieces"][0]["target_visible_pixels"] > 100
    assert receipt["pieces"][0]["target_bbox_status"] == "visible"
    assert receipt["pile"]["camera_provenance"]["near_clip"] < .002
    assert "tasks" not in receipt
    assert len(list((tmp_path / "images").glob("*.png"))) == 1
