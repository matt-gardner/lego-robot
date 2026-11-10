from datetime import datetime, timezone
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from so101.pathtrace_pile import DEFAULT_PROFILE
from so101.pathtrace_recordings import FrameRenderer, cad_render_xml, eligible, restore_pose, visible_geoms


def test_date_selection_uses_pacific_midnight_not_utc_date():
    cutoff = datetime(2026, 11, 8, 8, tzinfo=timezone.utc)
    assert not eligible({"generated_at": "2026-11-08T07:59:59+00:00"}, cutoff)
    assert eligible({"generated_at": "2026-11-08T00:00:00-08:00"}, cutoff)
    with pytest.raises(ValueError, match="generated_at"):
        eligible({}, cutoff)
    with pytest.raises(ValueError, match="time zones"):
        eligible({"generated_at": "2026-11-08T12:00:00"}, cutoff)


def test_cad_restore_changes_only_visual_jaws():
    xml = """<mujoco><worldbody>
    <body name="gripper">
      <geom class="visual" material="wrist_roll_follower_so101_v1_material" mesh="sleeve"/>
      <geom name="contact" type="box" size=".01 .01 .01"/>
    </body>
    <body name="moving_jaw_so101_v1">
      <geom class="visual" material="moving_jaw_so101_v1_material" mesh="moving_sleeve"/>
    </body></worldbody></mujoco>"""
    original = xml.replace('mesh="sleeve"', 'mesh="cad" pos=".1 0 0"').replace(
        'mesh="moving_sleeve"', 'mesh="moving_cad"',
    )
    result = ET.fromstring(cad_render_xml(xml, original))
    assert result.find(".//body[@name='gripper']/geom").get("mesh") == "cad"
    assert result.find(".//geom[@name='contact']").attrib == ET.fromstring(xml).find(
        ".//geom[@name='contact']",
    ).attrib


def test_pose_restoration_preserves_saved_objects_without_simulation():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".01"/>'
        '</body></worldbody></mujoco>',
    )
    data = mujoco.MjData(model)
    pose = [.2, .3, .04, 1, 0, 0, 0]
    restore_pose(model, data, {"qpos": pose})
    np.testing.assert_array_equal(data.qpos, pose)
    np.testing.assert_allclose(data.xpos[1], pose[:3])
    assert data.time == 0
    with pytest.raises(ValueError, match="pose"):
        restore_pose(model, data, {"qpos": pose[:-1]})


def test_render_geometry_excludes_all_hidden_collision_groups():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="sphere" size=".01" group="2"/>'
        '<geom type="sphere" size=".01" group="3"/>'
        '<geom type="sphere" size=".01" group="4" rgba="1 0 0 1"/>'
        '</worldbody></mujoco>',
    )
    assert visible_geoms(model) == [0]


def test_reused_mesh_render_matches_rebuild_after_motion_and_material_changes():
    pytest.importorskip("mitsuba")
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><asset><mesh name="piece" vertex="
      -.08 -.08 -.08 .08 -.08 -.08 .08 .08 -.08 -.08 .08 -.08
      -.08 -.08 .08 .08 -.08 .08 .08 .08 .08 -.08 .08 .08"/></asset>
      <worldbody><camera name="wrist_cam" pos="0 0 1"/>
        <geom name="table" type="plane" size="1 1 .01"/>
        <body name="piece" pos="0 0 .12">
          <geom name="piece" type="mesh" mesh="piece" rgba=".7 .2 .1 1"/>
        </body></worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    profile = json.loads(DEFAULT_PROFILE.read_bytes())
    profile.update(samples=1, threads=1)
    reference = FrameRenderer(model, profile)
    cached = FrameRenderer(model, profile, reuse_meshes=True)
    geom = model.geom("piece").id
    mesh_object = None
    previous = None
    for frame in range(3):
        if frame:
            model.body("piece").pos[0] += .08
            model.geom("piece").rgba[:3] = [.1, .7, .2]
            profile["plastic_roughness"] = .1
            profile["plastic_ior"] = 1.6
        mujoco.mj_forward(model, data)
        expected = reference.render(data, frame)
        actual = cached.render(data, frame)
        np.testing.assert_array_equal(actual, expected)
        if mesh_object is not None:
            assert cached.mesh_objects[geom][0] is mesh_object
            assert not np.array_equal(actual, previous)
        mesh_object = cached.mesh_objects[geom][0]
        previous = actual
    assert reference.mesh_objects == {}
