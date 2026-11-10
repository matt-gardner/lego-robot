import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from so101 import pile_task_generator as generator
from so101 import pile_task_runtime as runtime
from so101 import pile_task_worker as worker
from so101.lego_curriculum import Catalog, Part
from so101.lego_scene import capture_scene_state


def test_random_catalog_sampling_and_monochrome_are_reproducible():
    parts = tuple(Part(str(index), "brick", (.008, .016, .01), "", ())
                  for index in range(20))
    catalog = Catalog("fixture", "abc", parts, Path("/unused"))
    plan = generator.propose_pile(catalog, 123, 0, 8, "monochrome")
    assert plan == generator.propose_pile(catalog, 123, 0, 8, "monochrome")
    assert len({obj["part_id"] for obj in plan["objects"]}) == 8
    assert len({obj["color"] for obj in plan["objects"]}) == 1
    assert plan["destinations"][0]["position_relative_to_scene_m"] == [0., -.14, .00002]
    other = generator.propose_pile(catalog, 123, 1, 8, "many")
    assert {obj["part_id"] for obj in plan["objects"]} != {
        obj["part_id"] for obj in other["objects"]
    }


def test_hidden_target_has_no_box_and_identity_survives_unburying():
    hidden = runtime.box_from_mask(np.zeros((10, 10), dtype=bool), "buried_7")
    assert hidden["box_xyxy"] is None and not hidden["box_valid"]
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 6:9] = True
    revealed = runtime.box_from_mask(mask, "buried_7")
    assert revealed["instance_id"] == hidden["instance_id"] == "buried_7"
    assert revealed["box_xyxy"] == [.6, .2, .9, .4]
    assert revealed["visible_pixels"] == 6


def test_instance_segmentation_filters_geometry_type_and_never_switches_target(monkeypatch):
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody>
        <body name="target"><geom name="target_geom" size=".01"/></body>
        <body name="other"><geom name="other_geom" size=".01"/></body>
      </worldbody></mujoco>""")
    segmentation = np.full((4, 4, 2), -1, dtype=int)
    segmentation[0:2, 0:2] = [model.geom("other_geom").id, int(mujoco.mjtObj.mjOBJ_GEOM)]
    calls = []
    renderer = SimpleNamespace(
        _gl_context=SimpleNamespace(make_current=lambda: None), _mjr_context=object(),
        update_scene=lambda *args, **kwargs: None,
        enable_segmentation_rendering=lambda: None,
        disable_segmentation_rendering=lambda: calls.append("disabled"),
        render=lambda: segmentation,
    )
    monkeypatch.setattr(mujoco, "mjr_setBuffer", lambda *_: None)
    sim = runtime.PileSimulation(model, mujoco.MjData(model), renderer,
                                 {"target": model.body("target").id})
    assert sim.instance_box("target")["box_valid"] is False
    segmentation[3, 3] = [model.geom("target_geom").id, int(mujoco.mjtObj.mjOBJ_GEOM)]
    assert sim.instance_box("target")["visible_pixels"] == 1
    assert calls == ["disabled", "disabled"]


def test_native_binary_reset_roundtrip_and_file_tampering(tmp_path):
    xml = """<mujoco><worldbody><geom name="table" type="plane" size="1 1 .1"/>
      <body name="piece" pos="0 0 .1"><freejoint/><geom name="piece_geom" size=".01"/>
      </body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    snapshot = capture_scene_state(model, data, xml, {"scene_id": "scene",
                                   "catalog_sha256": "catalog"}, status="settled")
    task = {"task_id": "task", "task_type": "view", "target_instance_id": "piece",
            "grader_version": runtime.GRADER_VERSION, "goal": {"dwell_steps": 2}}
    problem = {"scene_id": "scene", "scene_family_id": "family", "tasks": [task],
               "instances": [{"instance_id": "piece", "body_name": "piece",
                              "geom_names": ["piece_geom"]}]}
    mujoco.mj_saveModel(model, str(tmp_path / "model.mjb"), None)
    (tmp_path / "model.xml").write_text(xml)
    for name, value in (("integration_state.json", snapshot), ("problem.json", problem),
                        ("scene_plan.json", {})):
        (tmp_path / name).write_text(json.dumps(value))
    entry = {"root": str(tmp_path), "scene_id": "scene", "scene_family_id": "family",
             "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in tmp_path.iterdir()}}
    loaded = runtime.load_problem(entry, "task")
    sim = runtime.create_simulation(loaded)
    try:
        np.testing.assert_array_equal(sim.data.qpos, data.qpos)
        mujoco.mj_step(sim.model, sim.data)
        runtime.reset_problem(sim, loaded)
        np.testing.assert_array_equal(sim.data.qvel, data.qvel)
        assert sim.data.time == data.time
    finally:
        sim.close()
    (tmp_path / "problem.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        runtime.load_problem(entry, "task")


def request(tmp_path):
    return {
        "schema_version": 1, "kind": "pile-task-generation", "request_id": "request1",
        "split": "train", "catalog": str(tmp_path / "catalog.json"),
        "output_root": str(tmp_path / "pool"), "seed": 123, "max_scenes": 2,
        "tag_constraints": {"piece_count": 8, "color_mode": "monochrome"},
        "excluded_scene_families": [], "task_types": ["view", "grasp", "push", "place"],
    }


def test_worker_rejects_evaluation_and_unknown_tags_without_generation(tmp_path, monkeypatch):
    value = request(tmp_path)
    value["split"] = "development"
    path = tmp_path / "request.json"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(worker, "generate_scene",
                        lambda *args, **kwargs: pytest.fail("Must not generate"))
    result = worker.process_request(path)
    assert result["status"] == "rejected" and "TRAIN-only" in result["error"]
    assert not (tmp_path / "pool").exists()
    assert worker.process_request(path) == result
    value["split"] = "train"
    value["tag_constraints"]["policy_success_rate"] = .5
    with pytest.raises(ValueError, match="Unsupported tag"):
        worker.validate_request(value)


def test_worker_records_explicit_resource_block_and_detects_reused_identity(tmp_path, monkeypatch):
    value = request(tmp_path)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(worker, "resource_check",
                        lambda *_: (_ for _ in ()).throw(BlockingIOError("memory pressure")))
    result = worker.process_request(path)
    assert result["status"] == "blocked" and result["error"] == "memory pressure"
    value["seed"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="reused"):
        worker.process_request(path)


def test_operational_errors_are_not_classified_as_bad_piles():
    assert generator.candidate_rejection(ValueError("Catalog scene did not settle before physics timeout"))
    assert not generator.candidate_rejection(ValueError("Part asset changed after catalog validation"))


def test_native_grader_sustained_view_and_hidden_target_reset_stability():
    goal = {"minimum_pixels": 64, "center_radius": .2, "dwell_steps": 2}
    problem = {"task": {"task_type": "view", "target_instance_id": "buried",
                       "grader_version": runtime.GRADER_VERSION, "goal": goal}}
    label = {"visible": True, "visible_pixels": 80, "box_xyxy": [.4, .4, .6, .6]}
    sim = SimpleNamespace(
        position=lambda _: np.zeros(3), registry={"buried": 1},
        instance_box=lambda _: label,
    )
    first = runtime.task_success(sim, problem, 1, 0)
    assert first == {"success": False, "stable": 1}
    assert runtime.task_success(sim, problem, 2, first["stable"])["success"]
    label.update(visible=False, visible_pixels=0, box_xyxy=None)
    assert runtime.task_success(sim, problem, 3, 1) == {"success": False, "stable": 0}


def test_native_position_uses_catalog_center_of_mass():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><body name="piece"><freejoint/>
        <geom size=".01" pos=".02 0 0"/>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sim = runtime.PileSimulation(model, data, None, {"piece": model.body("piece").id})
    np.testing.assert_array_equal(sim.position("piece"), [.02, 0, 0])


def test_generated_target_positions_match_post_integration_reset():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody>
        <body name="piece" pos="0 0 .1"><freejoint/><geom size=".01"/></body>
        <body name="goal_left" pos="0 -.14 0"/>
        <body name="goal_right" pos="0 .14 0"/>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    body = model.body("piece").id
    assert data.xipos[body, 2] != data.qpos[2]
    sim = SimpleNamespace(
        model=model, data=data,
        catalog=SimpleNamespace(parts=[
            SimpleNamespace(part_id="part", dimensions_m=(.02, .02, .02))
        ]),
        position=lambda name: data.body(name).xipos.copy(),
    )
    plan = {"scene_id": "scene", "color_mode": "monochrome", "objects": [
        {"instance_id": "piece", "part_id": "part", "color": "red"}
    ]}
    _, tasks, _ = generator.describe_scene(sim, plan)
    assert tasks[0]["initial_target_position"][2] == data.qpos[2]


def test_worker_excludes_families_and_reuses_completed_result(tmp_path, monkeypatch):
    value = request(tmp_path)
    value["excluded_scene_families"] = [
        hashlib.sha256(b"pile-v1:123:0").hexdigest()
    ]
    path = tmp_path / "request.json"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(worker, "resource_check", lambda *_: None)
    calls = []

    def generate(_catalog, output, **kwargs):
        calls.append(kwargs["sequence"])
        return {"root": str(output), "scene_family_id": "new-train-family"}

    monkeypatch.setattr(worker, "generate_scene", generate)
    result = worker.process_request(path)
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    assert result["status"] == "complete" and result["generated_scenes"] == 1
    assert manifest["rejections"] == [{"sequence": 0, "reason": "excluded_family"}]
    assert calls == [1]
    assert worker.process_request(path) == result
    assert calls == [1]


def test_interrupted_scene_is_not_silently_admitted(tmp_path):
    output = tmp_path / "scene"
    output.mkdir()
    (output / "model.mjb").write_bytes(b"partial")
    with pytest.raises(BlockingIOError, match="without admission"):
        generator.recover_entry(output)


@pytest.mark.parametrize("task_type", ["push", "place"])
def test_native_destination_requires_support_and_place_requires_prior_lift(task_type):
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody>
        <geom name="table" type="plane" size="1 1 .1"/>
        <body name="piece" pos="0 0 .01"><freejoint/><geom size=".01"/></body>
        <body name="gripper" pos="1 0 1"><geom size=".01"/></body>
        <body name="moving_jaw_so101_v1" pos="-1 0 1"><geom size=".01"/></body>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    for _ in range(500):
        mujoco.mj_step(model, data)
    sim = runtime.PileSimulation(model, data, None, {"piece": model.body("piece").id})
    problem = {"task": {
        "task_type": task_type, "target_instance_id": "piece",
        "initial_target_position": sim.position("piece").tolist(),
        "grader_version": runtime.GRADER_VERSION,
        "goal": {"destination_xy": [0., 0.], "tolerance_m": .025,
                 "maximum_surface_speed_m_s": .01},
    }}
    assert runtime.task_predicate(sim, problem) == (task_type == "push")
    sim.lifted_once = True
    assert runtime.task_predicate(sim, problem)
    data.qpos[2] += .1
    mujoco.mj_forward(model, data)
    assert not runtime.task_predicate(sim, problem)
