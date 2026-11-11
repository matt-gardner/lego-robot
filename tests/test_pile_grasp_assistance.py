import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from so101.pile_grasp_assistance import GraspConstructor, UnavailableGrasp, robot_pose
from so101.pile_task_runtime import PileSimulation, reset_problem
from so101.robot_table_clearance import require_robot_above_table


def table_model(*, visual=False):
    visual_mesh = """
      <asset><mesh name="visual" vertex="-.01 -.01 -.02 .01 -.01 -.02
        .01 .01 -.02 -.01 .01 -.02 -.01 -.01 .02 .01 -.01 .02
        .01 .01 .02 -.01 .01 .02"/></asset>""" if visual else ""
    geom = ('<geom type="mesh" mesh="visual" group="2" contype="0" conaffinity="0"/>'
            if visual else '<geom type="box" size=".01 .01 .01"/>')
    model = mujoco.MjModel.from_xml_string(f"""<mujoco>{visual_mesh}<worldbody>
      <geom name="table" type="plane" size="1 1 .01"/>
      <body name="base" pos="0 0 .015">
        <joint type="slide" axis="0 0 1"/>{geom}
      </body></worldbody></mujoco>""")
    return model, mujoco.MjData(model)


def test_table_guard_rejects_visual_mesh_intersection_without_collision_contacts():
    model, data = table_model(visual=True)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0
    with pytest.raises(ValueError, match="Robot intersects table"):
        require_robot_above_table(model, data)
    data.qpos[0] = .01
    mujoco.mj_forward(model, data)
    assert require_robot_above_table(model, data)["minimum_clearance_m"] == pytest.approx(.005)


def test_table_guard_checks_descendants_and_actual_plane_pose():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="table" type="plane" size="1 1 .01" pos="0 0 .1"/>
      <body name="base" pos="0 0 .2"><geom type="sphere" size=".01"/>
        <body name="jaw" pos="0 0 -.095"><geom type="box" size=".01 .01 .01"/></body>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with pytest.raises(ValueError, match="jaw"):
        require_robot_above_table(model, data)


def test_intermediate_retreat_pose_cannot_cross_table_even_with_clear_endpoints():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="table" type="plane" size="1 1 .01"/>
      <body name="base" pos="0 0 .06"><joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 .08 0 0" size=".005"/>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    for angle in (0., np.pi):
        data.qpos[0] = angle
        mujoco.mj_forward(model, data)
        require_robot_above_table(model, data)
    data.qpos[0] = np.pi / 2
    mujoco.mj_forward(model, data)
    constructor = GraspConstructor.__new__(GraspConstructor)
    constructor.sim = SimpleNamespace(model=model, data=data)
    constructor.native = None
    with pytest.raises(UnavailableGrasp, match="intersects table"):
        constructor.validate({"target_instance_id": "unused"}, .4)


def test_reset_rechecks_geometry_instead_of_trusting_clearance_metadata(tmp_path, monkeypatch):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="table" type="plane" size="1 1 .01"/>
      <body name="base" pos="0 0 .005"><geom type="box" size=".01 .01 .01"/></body>
      <body name="gripper" pos="0 0 1"><geom type="sphere" size=".01"/></body>
      <body name="moving_jaw_so101_v1" pos="0 0 2"><geom type="sphere" size=".01"/></body>
      <body name="piece" pos="0 0 3"><freejoint/><geom type="sphere" size=".01"/></body>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    (tmp_path / "integration_state.json").write_text(json.dumps({}))
    (tmp_path / "model.xml").write_text("")
    monkeypatch.setattr("so101.pile_task_runtime.restore_scene",
                        lambda model, data, *_: mujoco.mj_forward(model, data))
    task = {
        "task_type": "grasp", "target_instance_id": "piece", "parent_task_id": "parent",
        "initial_task_state": {"lifted_once": False},
        "assistance": {"kind": "original-pose-grasp-v1", "value": .8,
                       "parent_task_id": "parent",
                       "robot_table_clearance": {"minimum_clearance_m": 1.}},
    }
    native = PileSimulation(model, data, None, {"piece": model.body("piece").id})
    with pytest.raises(ValueError, match="Robot intersects table"):
        reset_problem(native, {"root": tmp_path, "task": task})


def test_open_partial_and_grasped_poses_preserve_arm_alignment():
    original = np.array([-.2, .4, .1, -.3, .2, .6])
    closed = np.array([.3, .1, .2, .4, -.1, .15])
    opened = robot_pose(original, closed, 1., .8)
    partial = robot_pose(original, closed, 1., .85)
    grasped = robot_pose(original, closed, 1., .9)
    np.testing.assert_array_equal(opened[:5], closed[:5])
    np.testing.assert_array_equal(partial[:5], opened[:5])
    np.testing.assert_array_equal(grasped[:5], opened[:5])
    assert partial[-1] == pytest.approx((opened[-1] + grasped[-1]) / 2)
    np.testing.assert_allclose(robot_pose(original, closed, 1., .4),
                               (original + opened) / 2)
    np.testing.assert_array_equal(robot_pose(original, closed, 1., 0.), original)
