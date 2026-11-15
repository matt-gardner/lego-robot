import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from scripts import produce_pile_tasks as production
from so101 import pile_task_generator as generator
from so101 import pile_pool as pool
from so101 import pile_task_runtime as runtime
from so101 import pile_task_worker as worker
from so101.lego_curriculum import Catalog, Part
from so101.lego_scene import capture_scene_state, settle_data


DRIFT_XML = """<mujoco><option gravity="0 0 0"/><worldbody>
<body name="still"><freejoint/><geom size=".01"/></body>
<body name="drift" pos=".1 0 0"><freejoint/><geom size=".01"/></body>
</worldbody></mujoco>"""


@pytest.mark.parametrize("allowance,limit,settles", [(0, .005, False), (1, .005, True),
                                                     (1, .001, False)])
def test_final_shift_allowance_accepts_only_few_bounded_movers(allowance, limit, settles):
    model = mujoco.MjModel.from_xml_string(DRIFT_XML)
    data = mujoco.MjData(model)
    data.qvel[6] = .002
    plan = {"scene_id": "s", "catalog_sha256": "c", "objects": [
        {"instance_id": name, "part_id": "p"} for name in ("still", "drift")]}
    settle = lambda: settle_data(
        model, data, DRIFT_XML, plan, maximum_seconds=2., stationary_seconds=1.,
        velocity_tolerance=.001, stationarity_metric="displacement-v3",
        final_shift_allowance=allowance, final_shift_limit_m=limit)
    if settles:
        assert settle()[2]["stationarity"]["final_shifting_pieces"] == 1
    else:
        with pytest.raises(ValueError, match="did not settle"):
            settle()


@pytest.mark.parametrize("extensions,settles", [(1, False), (3, True)])
def test_repeated_final_extensions_wait_out_slow_creep(extensions, settles):
    xml = DRIFT_XML.replace('<body name="drift" pos=".1 0 0"><freejoint/>',
                            '<body name="drift" pos=".1 0 0"><joint type="free" damping=".004"/>')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qvel[6] = .01
    plan = {"scene_id": "s", "catalog_sha256": "c", "objects": [
        {"instance_id": name, "part_id": "p"} for name in ("still", "drift")]}
    settle = lambda: settle_data(
        model, data, xml, plan, maximum_seconds=2., stationary_seconds=1.,
        velocity_tolerance=.001, stationarity_metric="displacement-v3",
        final_extension_seconds=1., final_extension_count=extensions)
    if settles:
        assert settle()[2]["stationarity"]["final_extensions_used"] == 2
    else:
        with pytest.raises(ValueError, match="did not settle"):
            settle()


@pytest.mark.parametrize("sequence,limits", [(0, (384, 385, 768)),
                                           (2, (640, 641, 1280)),
                                           (3, (800, 801, 1600))])
def test_controller_applies_size_scaled_child_limits(tmp_path, monkeypatch, sequence, limits):
    piles = tmp_path / "piles"
    piles.mkdir()
    for previous in range(sequence):
        (piles / f"attempt_{previous:09d}.json").write_text("{}")
    monkeypatch.setattr(production, "prepare", lambda *_: (Path("/unused"), Path("/unused")))
    monkeypatch.setattr(production, "resource_check", lambda *_: None)
    monkeypatch.setattr(production.sys, "argv", [
        "produce_pile_tasks", "--root", str(tmp_path), "--until", "2099-01-01T00:00:00+00:00",
        "--maximum-attempts", "1"])
    original_run = production.subprocess.run

    def run(command, **kwargs):
        assert kwargs["timeout"] == limits[2]
        prefix = command[2].split("from so101.pile_task_generator")[0]
        probe = original_run([command[0], "-c", prefix + "print(resource.getrlimit(resource.RLIMIT_CPU))"],
                             capture_output=True, text=True, check=True)
        assert probe.stdout.strip() == str(limits[:2])
        count, mode = generator.scheduled_pile(sequence, airborne=True)
        (piles / f"attempt_{sequence:09d}.json").write_text(json.dumps({
            "status": "rejected", "piece_count": count, "color_mode": mode}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(production.subprocess, "run", run)
    production.main()
    attempt = json.loads((tmp_path / "progress.json").read_text())["attempts"][0]
    assert (attempt["cpu_soft_limit_seconds"], attempt["cpu_hard_limit_seconds"],
            attempt["wall_limit_seconds"]) == limits


@pytest.mark.parametrize("airborne", [False, True])
def test_continuous_schedule_revisits_small_piles_after_largest_range(tmp_path, monkeypatch, airborne):
    for sequence in range(11):
        (tmp_path / f"attempt_{sequence:09d}.json").write_text("{}")
    proposals = []

    def generate(_catalog, _output, **kwargs):
        proposals.append((kwargs["count"], kwargs["color_mode"]))
        raise ValueError("Settled proposal is not a table-connected contacting pile")

    budgets = []
    monkeypatch.setattr(generator, "resource_check", lambda _root, budget: budgets.append(budget))
    monkeypatch.setattr(generator, "generate_scene", generate)
    generator._run(tmp_path, Path("/unused"), seed=123, attempts=36,
                   stopping=threading.Event(), airborne=airborne)
    assert [count for count, _ in proposals[:3]] == ([100, 48, 64] if airborne else [64, 3, 12])
    if airborne:
        assert {count for count, _ in proposals} == {48, 64, 80, 100}
    assert len(set(proposals)) == (12 if airborne else 27)
    assert sum(count <= 8 for count, _ in proposals) == (0 if airborne else 18)
    assert budgets == [None if airborne else generator.MAXIMUM_POOL_BYTES] * 36
    assert json.loads((tmp_path / "attempt_000000012.json").read_text())["status"] == "rejected"


def test_sequential_drop_does_not_turn_flat_piece_width_into_drop_height(monkeypatch):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="a"><freejoint/><geom type="box" size=".15 .02 .001"/></body>
      <body name="b"><freejoint/><geom type="box" size=".15 .02 .001"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    sim = SimpleNamespace(
        model=model, data=data, xml="", scene_origin=np.array([.22, 0., generator.TABLE_Z]),
        plan={"objects": [{"instance_id": name, "part_id": "flat",
                           "position_relative_to_scene_m": [0., 0., 1.]}
                          for name in ("a", "b")]},
        part_metadata={"flat": SimpleNamespace(
            vertical_bounds_m=(-.001, .001), dimensions_m=(.30, .04, .002))},
    )
    drops = []

    def settle(*_args, **_kwargs):
        index = len(drops)
        drops.append(float(data.qpos[index * 7 + 2]))
        data.qpos[index * 7 + 2] = generator.TABLE_Z + .001 + index * .002

    monkeypatch.setattr(generator, "settle_data", settle)
    generator.drop_pieces_sequentially(sim)
    np.testing.assert_allclose(drops, [generator.TABLE_Z + .0011, generator.TABLE_Z + .0031])


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


@pytest.mark.parametrize("distant_peak", [False, True])
def test_airborne_batches_preserve_moving_state_without_per_piece_settling(monkeypatch, distant_peak):
    bodies = "".join(
        f'<body name="piece{i}" pos="{10+i} 0 1"><freejoint/>'
        '<geom type="mesh" mesh="visual" group="2" contype="0" conaffinity="0"/>'
        f'<geom name="collision{i}" type="box" group="3" size=".004 .004 .004"/></body>'
        for i in range(6))
    xml = (
        '<mujoco><asset><mesh name="visual" vertex="-.004 -.004 -.004 '
        '.004 -.004 -.004 -.004 .004 -.004 .004 .004 -.004 '
        '-.004 -.004 .004 .004 -.004 .004 -.004 .004 .004 .004 .004 .004"/>'
        f'</asset><worldbody><geom name="table" type="plane" size="1 1 .1" '
        f'friction=".7 .005 .0001" pos="0 0 {generator.TABLE_Z}"/>'
        f'{bodies}</worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    sim = SimpleNamespace(
        model=model, data=mujoco.MjData(model), xml=xml,
        scene_origin=np.array([0., 0., generator.TABLE_Z]),
        plan={"seed": 123, "sequence": 0, "objects": [
            {"instance_id": f"piece{i}", "part_id": "fixture"} for i in range(6)]},
    )
    original_step = mujoco.mj_step
    previous = []

    def step(model, data):
        original_step(model, data)
        if distant_peak:
            poses = data.qpos.reshape(-1, 7)
            poses[:, 0] = np.arange(len(poses)) + 1.
            poses[:, 2] = .5
        previous[:] = [data.qpos.copy(), data.qvel.copy(), data.qacc_warmstart.copy()]

    monkeypatch.setattr(mujoco, "mj_step", step)
    monkeypatch.setattr(generator, "settle_data",
                        lambda *_args, **_kwargs: pytest.fail("Per-piece settling invoked"))
    generator.drop_pieces_airborne(sim)
    assert sim.model.nbody == 7
    assert sim.data.time == pytest.approx(.5)
    np.testing.assert_array_equal(sim.data.qpos[:len(previous[0])], previous[0])
    np.testing.assert_array_equal(sim.data.qvel[:len(previous[1])], previous[1])
    np.testing.assert_array_equal(sim.data.qacc_warmstart[:len(previous[2])], previous[2])
    assert np.all(sim.model.body_gravcomp == 0)
    assert np.all(sim.data.tree_asleep < 0)
    assert sim.model.npair == 6
    np.testing.assert_allclose(sim.model.pair_friction,
                               np.tile([1., 1., .005, .0001, .0001], (6, 1)))
    assert [event["count"] for event in sim.plan["airborne_spawn_events"]] == list(range(1, 7))
    assert not np.allclose(sim.data.qpos.reshape(6, 7)[:, 3:], [1, 0, 0, 0])
    if distant_peak:
        event = sim.plan["airborne_spawn_events"][-1]
        assert event["highest_existing_visual_z_m"] > .5
        placement = event["placements"][0]
        assert placement["local_surface_top_m"] == generator.TABLE_Z
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, np.array(placement["quaternion"]))
        bottom = placement["position_m"][2] - .004 * np.abs(matrix.reshape(3, 3)[2]).sum()
        assert bottom == pytest.approx(generator.TABLE_Z + .01)


def test_large_airborne_stratum_separates_size_from_specialized_shapes():
    parts = tuple(Part(str(i), "brick" if i < 10 else "round",
                       (.008, .016 if i < 5 else .032, .01), "", ()) for i in range(20))
    catalog = Catalog("fixture", "abc", parts, Path("/unused"))
    large = generator.propose_pile(catalog, 123, 0, 24, "many", airborne=True)
    assert len(large["objects"]) == 24
    assert all(int(obj["part_id"]) < 10 for obj in large["objects"])
    assert any(int(obj["part_id"]) >= 5 for obj in large["objects"])
    high_count = generator.propose_pile(catalog, 123, 0, 48, "many", airborne=True)
    assert len(high_count["objects"]) == 48
    assert all(int(obj["part_id"]) < 5 for obj in high_count["objects"])
    largest = generator.propose_pile(catalog, 123, 0, 100, "many", airborne=True)
    assert len(largest["objects"]) == len({obj["instance_id"] for obj in largest["objects"]}) == 100
    assert all(int(obj["part_id"]) < 5 for obj in largest["objects"])
    small = generator.propose_pile(catalog, 123, 0, 8, "many", airborne=True)
    assert any(int(obj["part_id"]) >= 10 for obj in small["objects"])
    oversized = Catalog("fixture", "abc", parts[5:10], Path("/unused"))
    with pytest.raises(ValueError, match="no compact basic"):
        generator.propose_pile(oversized, 123, 0, 48, "many", airborne=True)


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
        width=4, height=4,
        update_scene=lambda *args, **kwargs: None,
        enable_segmentation_rendering=lambda: None,
        disable_segmentation_rendering=lambda: calls.append("disabled"),
        render=lambda: segmentation,
    )
    monkeypatch.setattr(mujoco, "mjr_setBuffer", lambda *_: None)
    samples = model.vis.quality.offsamples

    def label_renderer(model, **kwargs):
        assert model.vis.quality.offsamples == 0
        assert kwargs == {"width": 4, "height": 4}
        return renderer

    monkeypatch.setattr(mujoco, "Renderer", label_renderer)
    sim = runtime.PileSimulation(model, mujoco.MjData(model), renderer,
                                 {"target": model.body("target").id})
    assert sim.instance_box("target")["box_valid"] is False
    segmentation[3, 3] = [model.geom("target_geom").id, int(mujoco.mjtObj.mjOBJ_GEOM)]
    assert sim.instance_box("target")["visible_pixels"] == 1
    assert calls == ["disabled", "disabled"]
    assert model.vis.quality.offsamples == samples


def test_label_renderer_failure_preserves_rgb_sampling(monkeypatch):
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    samples = model.vis.quality.offsamples
    sim = runtime.PileSimulation(
        model, mujoco.MjData(model), SimpleNamespace(width=320, height=240), {})

    def unavailable(model, **kwargs):
        assert model.vis.quality.offsamples == 0
        raise RuntimeError("Rendering context unavailable")

    monkeypatch.setattr(mujoco, "Renderer", unavailable)
    with pytest.raises(RuntimeError, match="Rendering context unavailable"):
        sim.segmentation_image()
    assert sim.label_renderer is None
    assert model.vis.quality.offsamples == samples


def test_native_binary_reset_roundtrip_and_file_tampering(tmp_path, monkeypatch):
    tmp_path = tmp_path / "scene_test"
    tmp_path.mkdir()
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
    problem = {"scene_id": "scene", "scene_family_id": "family",
               "tasks": [task, {**task, "task_id": "second_task"}],
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
    entry["split"] = "train"
    (tmp_path / "admission.json").write_text(json.dumps(entry))
    reads = {}
    original_read = Path.read_bytes

    def read(path):
        reads[path.name] = reads.get(path.name, 0) + 1
        return original_read(path)

    with monkeypatch.context() as context:
        context.setattr(Path, "read_bytes", read)
        assert pool.publish_pool(tmp_path.parent)["scenes"] == 1
    assert all(reads[name] == 1 for name in entry["files_sha256"])
    problem["tasks"][1]["grader_version"] = "unsupported"
    (tmp_path / "problem.json").write_text(json.dumps(problem))
    entry["files_sha256"]["problem.json"] = hashlib.sha256(
        (tmp_path / "problem.json").read_bytes()).hexdigest()
    (tmp_path / "admission.json").write_text(json.dumps(entry))
    with pytest.raises(ValueError, match="unsupported grader"):
        pool.publish_pool(tmp_path.parent)
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
    assert generator.candidate_rejection(ValueError(
        "Pile constraint arena exhausted: mj_stackAlloc: out of memory"
    ))


def test_resource_admission_uses_available_memory_not_nonzero_psi(tmp_path, monkeypatch):
    monkeypatch.setattr(generator.os, "statvfs", lambda _: SimpleNamespace(
        f_bavail=20 * 1024**3, f_frsize=1,
    ))
    state = {"pressure": True, "available_bytes": 3 * 1024**3}
    monkeypatch.setattr(generator, "memory_state", lambda: state)
    generator.resource_check(tmp_path)
    with (tmp_path / "large.mjb").open("wb") as stream:
        stream.truncate(4 * 1024**3)
    with pytest.raises(BlockingIOError, match="pool budget"):
        generator.resource_check(tmp_path)
    generator.resource_check(tmp_path, None)
    state["available_bytes"] = 2 * 1024**3 - 1
    with pytest.raises(BlockingIOError, match="memory admission"):
        generator.resource_check(tmp_path, None)
    state["available_bytes"] = 3 * 1024**3
    monkeypatch.setattr(generator.os, "statvfs", lambda _: SimpleNamespace(
        f_bavail=9 * 1024**3, f_frsize=1,
    ))
    with pytest.raises(BlockingIOError, match="10GiB"):
        generator.resource_check(tmp_path, None)


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


def test_view_walls_added_outside_settled_pile_preserve_piece_state():
    xml = """<mujoco><worldbody>
      <body name="piece" pos=".22 0 .1"><freejoint/><geom size=".03"/></body>
      <geom name="pile_view_obstruction_0" type="box" size=".035 .006 .045" pos="0 0 -10"/>
      <geom name="pile_view_obstruction_1" type="box" size=".035 .006 .045" pos="0 0 -11"/>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    before = data.qpos.copy()
    sim = SimpleNamespace(model=model, data=data, xml=xml)
    generator.add_settled_view_walls(sim, {"objects": [{"instance_id": "piece"}]})
    np.testing.assert_array_equal(data.qpos, before)
    assert model.geom_pos[model.geom("pile_view_obstruction_0").id, 1] < -.036
    assert model.geom_pos[model.geom("pile_view_obstruction_1").id, 1] > .036
    rebuilt = mujoco.MjModel.from_xml_string(sim.xml)
    np.testing.assert_array_equal(model.geom_pos, rebuilt.geom_pos)


def test_sequential_drop_keeps_future_pieces_out_of_column_and_restores_gravity(monkeypatch):
    xml = """<mujoco><worldbody>
      <body name="a"><freejoint/><geom size=".01"/></body>
      <body name="b"><freejoint/><geom size=".01"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    plan = {"objects": [
        {"instance_id": name, "part_id": "part", "position_relative_to_scene_m": [0., 0., 10.]}
        for name in ("a", "b")
    ]}
    sim = SimpleNamespace(
        model=model, data=data, plan=plan, xml=xml,
        scene_origin=np.array([.22, 0., generator.TABLE_Z]),
        part_metadata={"part": SimpleNamespace(
            vertical_bounds_m=(-.01, .01), dimensions_m=(.02, .02, .02),
        )},
    )
    calls = []

    def settle(_model, _data, _xml, active_plan, **kwargs):
        calls.append(len(active_plan["objects"]))
        assert data.qpos[2] < generator.TABLE_Z + .1
        if len(calls) == 1:
            assert data.qpos[7] > 10
            assert model.body_gravcomp[model.body("b").id] == 1
        else:
            assert data.qpos[9] < generator.TABLE_Z + .1

    monkeypatch.setattr(generator, "settle_data", settle)
    generator.drop_pieces_sequentially(sim)
    assert calls == [1, 2]
    np.testing.assert_array_equal(model.body_gravcomp, 0)


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


def test_obstacles_are_deterministic_and_reach_the_drop_region():
    kinds = set()
    for sequence in range(40):
        obstacles = generator.sample_obstacles(7, sequence, 64)
        assert obstacles == generator.sample_obstacles(7, sequence, 64)
        kinds.add(obstacles[0]["kind"])
        for obstacle in obstacles:
            low, high = generator.obstacle_bounds(obstacle, (.22, 0., generator.TABLE_Z))
            assert low[2] >= generator.TABLE_Z - 1e-9
            assert np.hypot(*(np.clip((.22, 0.), low[:2], high[:2]) - (.22, 0.))) < .05
    assert kinds == set(generator.OBSTACLE_KINDS)
