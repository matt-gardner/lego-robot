import copy
import json

import mujoco
import pytest

from so101.pile_other_reset import PROFILE, validate_reset
from so101.pile_task_runtime import (
    GRADER_VERSION, PileSimulation, reset_problem, target_contact_state, task_predicate,
)
from so101.robot_control import JOINTS


def released_place():
    robot = ""
    for index, joint in enumerate(JOINTS):
        body = "base" if index == 0 else (
            "gripper" if index == 4 else "moving_jaw_so101_v1" if index == 5 else f"link{index}")
        position = "0 0 .5" if index == 0 else ".04 0 0"
        robot += (f'<body name="{body}" pos="{position}">'
                  f'<joint name="{joint}" range="-2 2"/>'
                  '<geom type="sphere" size=".005" mass=".1"/>')
    robot += "</body>" * len(JOINTS)
    model = mujoco.MjModel.from_xml_string(f"""<mujoco><worldbody>
      <geom name="table" type="plane" size="1 1 .01" pos="0 0 .1"/>
      {robot}
      <body name="piece" pos="-.1 0 .104995"><freejoint/>
        <geom type="box" size=".01 .01 .005"/></body>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    native = PileSimulation(model, data, None, {"piece": model.body("piece").id})
    snapshot = {"scene_id": "published", "qpos": data.qpos.tolist(),
                "runtime_model_sha256": "same-runtime", "model_xml_sha256": "same-xml",
                "physics_version": mujoco.__version__}
    original = {
        "task_id": "original", "task_type": "place", "target_instance_id": "piece",
        "grader_version": GRADER_VERSION, "initial_state": copy.deepcopy(snapshot),
        "initial_target_position": [-.1, 0, .104995],
        "goal": {"destination_xy": [.2, 0], "tolerance_m": .01,
                 "maximum_surface_speed_m_s": .03, "dwell_steps": 10},
    }
    data.qpos[6] = .2
    mujoco.mj_forward(model, data)
    task = {
        **original, "task_id": "child", "parent_task_id": "original",
        "initial_state": {**snapshot, "qpos": data.qpos.tolist()},
        "initial_task_state": {"lifted_once": True},
        "assistance": {"kind": PROFILE, "schema_version": 1, "value": .95,
                       "parent_task_id": "original", "independently_displaced_other_pieces": []},
    }
    return native, {"task": task}, {"task": original}


def test_released_place_progress_is_not_discarded_on_native_reset(tmp_path, monkeypatch):
    sim, problem, parent = released_place()
    (tmp_path / "integration_state.json").write_text(json.dumps({}))
    (tmp_path / "model.xml").write_text("")
    problem.update(root=tmp_path, scene={
        "scene_id": "published", "provenance": {"parent_entry": {}}})
    monkeypatch.setattr("so101.pile_task_runtime.restore_scene",
                        lambda model, data, *_: mujoco.mj_forward(model, data))
    monkeypatch.setattr("so101.pile_task_runtime.load_problem", lambda *_: parent)
    reset_problem(sim, problem)
    assert sim.lifted_once
    assert task_predicate(sim, problem)
    sim.lifted_once = False
    assert not task_predicate(sim, problem)


@pytest.mark.parametrize("mutation,error", [
    ("goal", "original task field"),
    ("physics", "physics identity"),
    ("progress", "progress disagrees"),
    ("held", "lacks bilateral"),
    ("missing-release", "support/release"),
])
def test_task_specific_reset_rejects_bad_progress_physics_or_release(mutation, error):
    sim, problem, parent = released_place()
    task = problem["task"]
    if mutation == "goal":
        task["goal"] = {**task["goal"], "destination_xy": [.3, 0]}
    elif mutation == "physics":
        task["initial_state"]["runtime_model_sha256"] = "changed"
    elif mutation == "progress":
        task["initial_task_state"] = {"lifted_once": False}
    elif mutation == "held":
        task["assistance"]["value"] = .9
    else:
        sim.data.qpos[8] += .02
        mujoco.mj_forward(sim.model, sim.data)
    with pytest.raises(ValueError, match=error):
        validate_reset(sim, problem, parent, target_contact_state(sim, "piece"),
                       target_contact_state)


def test_zero_assistance_cannot_change_even_the_robot_start():
    sim, problem, parent = released_place()
    sim.data.qpos[:] = parent["task"]["initial_state"]["qpos"]
    sim.data.qpos[0] += .01
    mujoco.mj_forward(sim.model, sim.data)
    problem["task"]["assistance"]["value"] = 0.
    problem["task"]["initial_task_state"] = {"lifted_once": False}
    with pytest.raises(ValueError, match="Zero assistance"):
        validate_reset(sim, problem, parent, target_contact_state(sim, "piece"),
                       target_contact_state)


def test_missing_progress_cannot_bypass_other_task_reset_validation(tmp_path, monkeypatch):
    sim, problem, _ = released_place()
    del problem["task"]["initial_task_state"]
    (tmp_path / "integration_state.json").write_text(json.dumps({}))
    (tmp_path / "model.xml").write_text("")
    problem.update(root=tmp_path, scene={"scene_id": "published"})
    monkeypatch.setattr("so101.pile_task_runtime.restore_scene",
                        lambda model, data, *_: mujoco.mj_forward(model, data))
    with pytest.raises(ValueError, match="missing initial task progress"):
        reset_problem(sim, problem)


def test_legacy_place_profile_still_requires_a_physically_held_piece(tmp_path, monkeypatch):
    sim, problem, _ = released_place()
    problem["task"]["assistance"]["kind"] = "direct-task-distance-v2"
    (tmp_path / "integration_state.json").write_text(json.dumps({}))
    (tmp_path / "model.xml").write_text("")
    problem.update(root=tmp_path, scene={"scene_id": "published"})
    monkeypatch.setattr("so101.pile_task_runtime.restore_scene",
                        lambda model, data, *_: mujoco.mj_forward(model, data))
    with pytest.raises(ValueError, match="requires a physically held piece"):
        reset_problem(sim, problem)
