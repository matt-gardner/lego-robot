from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from so101.pathtrace_recordings import cad_render_xml, eligible, restore_pose, visible_geoms


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
