import json
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from so101.render_pile import DEFAULT_PROFILE, render_pile, smooth_mesh_normals


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") not in ("osmesa", "egl"),
                    reason="Requires a headless rendering backend")
def test_saved_pile_render_preserves_physics_and_frames_geometry(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    xml = """<mujoco><compiler angle="radian"/><visual><global offwidth="128" offheight="96"/></visual>
<asset><texture name="table_texture" type="2d" builtin="checker" width="32" height="32"
rgb1=".7 .7 .7" rgb2=".8 .8 .8"/><material name="table_material" texture="table_texture"/>
<mesh name="brick" vertex="-.01 -.01 -.01 -.01 -.01 .01 -.01 .01 -.01 -.01 .01 .01
.01 -.01 -.01 .01 -.01 .01 .01 .01 -.01 .01 .01 .01"
face="0 2 6 0 6 4 1 5 7 1 7 3 0 1 3 0 3 2 4 6 7 4 7 5 0 4 5 0 5 1 2 3 7 2 7 6"/></asset>
<worldbody><geom type="plane" size="1 1 .01" material="table_material"/>
<body name="brick" pos="0 0 .02"><freejoint/><geom type="mesh" mesh="brick"
rgba=".8 .2 .1 1"/></body></worldbody></mujoco>"""
    (source / "model.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    np.savez_compressed(source / "settled-state.npz", qpos=data.qpos, qvel=data.qvel)
    original_state = (source / "settled-state.npz").read_bytes()
    config = json.loads(DEFAULT_PROFILE.read_text())
    config.update(image_width=96, image_height=72, supersampling=1, light_samples=4)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(config))
    monkeypatch.setattr(mujoco, "mj_step", lambda *args: pytest.fail("Rendering advanced physics"))
    output = tmp_path / "render"
    report = render_pile(source, output, profile)
    assert (source / "settled-state.npz").read_bytes() == original_state
    image = cv2.imread(str(output / "pile.png"))
    assert image.shape == (72, 96, 3)
    assert report["frame_occupancy"] == pytest.approx(config["frame_occupancy"], abs=.04)
    x0, y0, x1, y1 = report["object_bbox_px"]
    assert 0 < x0 < x1 < 96 and 0 < y0 < y1 < 72
    with np.load(output / "synchronized_labels.npz") as labels:
        assert labels["segmentation"].shape == (72, 96, 2)
        assert labels["depth_m"].shape == (72, 96)
        assert np.isfinite(labels["depth_m"]).all()


def test_invalid_resolution_rejected_before_loading_scene(tmp_path):
    config = json.loads(DEFAULT_PROFILE.read_text())
    config["image_width"] = -1
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="positive integers"):
        render_pile(tmp_path / "missing", tmp_path / "output", profile)


def test_shading_smooths_shallow_angles_without_changing_vertices_or_hard_edges():
    shallow = ET.Element("mesh", vertex="0 0 0 1 0 0 0 1 0 -.9238795 0 .3826834",
                         face="0 1 2 0 2 3")
    original = shallow.attrib.copy()
    normals = smooth_mesh_normals(shallow, 30)
    np.testing.assert_allclose(normals[0], normals[3], atol=1e-10)
    assert 0 < normals[0, 0] < .3826834
    assert shallow.get("vertex") == original["vertex"] and shallow.get("face") == original["face"]
    hard = ET.Element("mesh", vertex="0 0 0 1 0 0 0 1 0 0 0 1", face="0 1 2 0 2 3")
    normals = smooth_mesh_normals(hard, 30)
    np.testing.assert_allclose(normals[0], [0, 0, 1])
    np.testing.assert_allclose(normals[3], [1, 0, 0])
