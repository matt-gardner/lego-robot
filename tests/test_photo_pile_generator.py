import hashlib
import json
import os
from pathlib import Path
import threading

import mujoco
import numpy as np
import pytest

from so101 import photo_pile_generator as generator
from so101 import photo_pile_tasks
from so101 import photo_pile_curriculum
from so101 import pile_task_assistance
from so101 import pile_task_runtime
from so101.photo_pile_curriculum import validate_stages
from so101.pile_pool import publish_pool
from so101.pile_task_runtime import create_simulation, load_problem, reset_problem, task_predicate
from so101.rl_pile_adapter import NativePileSimulation, load_native_pool


@pytest.fixture
def photo_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    bodies = "".join(
        f'<body name="brick_{index}" pos="0 0 .1"><freejoint/>'
        '<geom type="box" size=".004 .004 .004" mass=".002" '
        'solref=".005 1" friction=".5 .005 .0001"/></body>'
        for index in range(90)
    )
    xml = (
        '<mujoco><option timestep=".0005" iterations="100" noslip_iterations="20" '
        'tolerance="1e-10" cone="elliptic" impratio="10" gravity="0 0 -9.81" '
        'integrator="implicitfast"/><worldbody>'
        '<geom type="plane" size="1 1 .01"/>'
        f'{bodies}</worldbody></mujoco>'
    )
    (source / "model.xml").write_text(xml)
    (source / "receipt.json").write_text(json.dumps({
        "model_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "catalog_sha256": "test-catalog",
    }))
    return source


def test_fresh_twenty_settles_without_source_saved_state(photo_source, tmp_path):
    output = tmp_path / "scene"
    stages = []
    receipt = generator.generate_scene(
        photo_source, output, count=20, seed=20261111, progress=stages.append, color_mode="single")
    verified = generator.verify_scene(output)
    assert receipt == verified
    additions = [stage for stage in stages if stage["phase"] == "adding"]
    assert [stage["inserted"] for stage in additions] == list(range(1, 21))
    assert additions[-1]["simulated_seconds"] < 20 * .16
    assert stages[-1]["phase"] == "final_settling"
    assert receipt["restored_original_integration_state"] is False
    assert receipt["stationarity"]["maximum_final_speed"] <= .0005
    assert receipt["maximum_insertion_penetration_m"] <= .005
    model = mujoco.MjModel.from_xml_path(str(output / "model.xml"))
    assert model.nbody == 21
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.noslip_iterations == 20
    assert np.array_equal(model.opt.gravity, [0., 0., -9.81])
    assert len({tuple(model.geom_rgba[g]) for g in range(model.ngeom)
                if model.geom_bodyid[g] != 0}) == 1


def test_collective_forty_uses_scaled_cpu_budget_and_strict_final_checks(
    photo_source, tmp_path, monkeypatch,
):
    clock = iter([0.])
    monkeypatch.setattr(generator.time, "process_time", lambda: next(clock, 150.))
    receipt = generator.generate_scene(
        photo_source, tmp_path / "forty", count=40, seed=20261111)
    assert receipt["stationarity"]["maximum_final_speed"] <= .0005
    assert receipt["maximum_penetration_m"] <= .005
    assert receipt["maximum_insertion_penetration_m"] <= .005
    clock = iter([0.])
    with pytest.raises(generator.RejectedPile, match="120-second CPU budget"):
        generator.generate_scene(photo_source, tmp_path / "twenty", count=20, seed=20261111)


def test_collective_final_settling_honors_stop(photo_source, tmp_path):
    settling = False

    def progress(value):
        nonlocal settling
        settling = value["phase"] == "final_settling"

    def stop():
        if settling:
            raise InterruptedError("stop during final settling")

    output = tmp_path / "stopped"
    with pytest.raises(InterruptedError, match="final settling"):
        generator.generate_scene(photo_source, output, count=20, seed=20261111,
                                 progress=progress, check_stop=stop)
    assert not output.exists()


def test_minimum_count_and_source_hash_protection(photo_source):
    with pytest.raises(ValueError, match="20 and 90"):
        generator.prepare_model(photo_source, 19, 1)
    with (photo_source / "model.xml").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="does not match"):
        generator.prepare_model(photo_source, 20, 1)


def test_rejections_resume_and_stop_preserve_attempts(photo_source, tmp_path, monkeypatch):
    root = tmp_path / "pool"
    seeds = []
    monkeypatch.setattr(generator, "resource_check", lambda root: None)

    def rejected(source, output, *, seed, **kwargs):
        seeds.append(seed)
        raise generator.RejectedPile("expected physical timeout")

    monkeypatch.setattr(generator, "generate_scene", rejected)
    generator.run(root, photo_source, seed=11, count=20, attempts=1)
    first = (root / "attempt_000000000.json").read_bytes()
    generator.run(root, photo_source, seed=11, count=20, attempts=1)
    assert seeds == [11, 12]
    assert (root / "attempt_000000000.json").read_bytes() == first
    assert json.loads((root / "attempt_000000001.json").read_text())["status"] == "rejected"
    stopping = threading.Event()
    stopping.set()
    generator.run(root, photo_source, seed=11, count=20, attempts=1, stopping=stopping)
    assert seeds == [11, 12]
    assert json.loads((root / "status.json").read_text())["status"] == "stopped"
    with pytest.raises(ValueError, match="different generator"):
        generator.run(root, photo_source, seed=11, count=21, attempts=1)


def test_robot_task_pool_resets_and_adapter(photo_source, tmp_path, monkeypatch):
    source = tmp_path / "fresh"
    generator.generate_scene(photo_source, source, count=20, seed=20261111)
    root = tmp_path / "tasks"
    root.mkdir()
    def reject_step(*args, **kwargs):
        raise AssertionError("Adding the robot must not step the settled pile")

    with monkeypatch.context() as context:
        context.setattr(mujoco, "mj_step", reject_step)
        entry = photo_pile_tasks.build_problem(source, root / "scene_000000000")
    snapshot = json.loads((root / "scene_000000000" / "integration_state.json").read_bytes())
    assert snapshot["robot_integration"]["physics_steps"] == 0
    receipt = json.loads((source / "receipt.json").read_bytes())
    model = mujoco.MjModel.from_binary_path(str(root / "scene_000000000" / "model.mjb"))
    for obj in receipt["objects"]:
        joint = int(model.body_jntadr[model.body(obj["instance_id"]).id])
        q = int(model.jnt_qposadr[joint])
        expected = np.asarray(obj["qpos"]).copy()
        expected[:3] += [.22, 0., photo_pile_tasks.TABLE_Z]
        assert np.array_equal(snapshot["qpos"][q:q + 7], expected)
    scene = json.loads((root / "scene_000000000" / "problem.json").read_text())
    assert len(scene["tasks"]) == 80
    assert len({task["target_instance_id"] for task in scene["tasks"]}) == 20
    starts = np.array([task["start"]["tool_position_m"] for task in scene["tasks"]])
    assert len(np.unique(starts, axis=0)) == 80
    distances = np.linalg.norm(starts[:, None] - starts[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= .005
    assert all(1 <= task["target_tags"]["height_rank_from_top"] <= 20 for task in scene["tasks"])
    assert all("not measured visibility" in task["target_tags"]["depth_method"]
               for task in scene["tasks"])
    assert scene["provenance"]["receipt_sha256"] == hashlib.sha256(
        (source / "receipt.json").read_bytes()).hexdigest()
    manifest = publish_pool(root)
    episodes = load_native_pool(Path(manifest["manifest_path"]), set())
    assert {kind: len(values) for kind, values in episodes.items()} == {
        "view": 20, "grasp": 20, "push": 20, "place": 20,
    }
    task = next(task for task in scene["tasks"] if task["task_type"] == "grasp")
    problem = load_problem(entry, task["task_id"])
    sim = create_simulation(problem)
    try:
        original = sim.data.qpos.copy()
        assert not task_predicate(sim, problem)
        sim.data.qpos[:] += .001
        reset_problem(sim, problem)
        assert np.array_equal(sim.data.qpos, original)
        for item in scene["tasks"]:
            reset_problem(sim, {**problem, "task": item})
            assert np.array_equal(sim.data.qpos, item["initial_state"]["qpos"])
            integration = np.empty(mujoco.mj_stateSize(
                sim.model, mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(sim.model, sim.data, integration,
                              mujoco.mjtState.mjSTATE_INTEGRATION)
            assert np.array_equal(integration, item["initial_state"]["integration_state"])
            assert np.allclose(sim.position(item["target_instance_id"]),
                               item["initial_target_position"], atol=1e-10, rtol=0)
        reset_problem(sim, problem)
        other = next(item for item in scene["tasks"] if item["task_id"] != task["task_id"])
        other_problem = {**problem, "task": other}
        reset_problem(sim, other_problem)
        assert not np.array_equal(sim.data.qpos, original)
        assert np.array_equal(sim.data.qpos, other["initial_state"]["qpos"])
        reset_problem(sim, problem)
        assert np.array_equal(sim.data.qpos, original)
        bad = {**task, "initial_state": {**task["initial_state"], "scene_id": "wrong"}}
        with pytest.raises(ValueError, match="different scene"):
            reset_problem(sim, {**problem, "task": bad})
    finally:
        sim.close()
    assert photo_pile_tasks.build_problem(source, root / "scene_000000000") == entry
    revised = photo_pile_tasks.retask_scene(entry, tmp_path / "revisions" / "scene_000000000")
    assert revised["scene_family_id"] == entry["scene_family_id"]
    assert revised["scene_id"] != entry["scene_id"]
    for name in ("model.mjb", "model.xml", "integration_state.json"):
        assert revised["files_sha256"][name] == entry["files_sha256"][name]
    revision_pool = publish_pool(tmp_path / "revisions")
    selected = photo_pile_tasks.select_current_pools(
        tmp_path, [Path(revision_pool["manifest_path"])])
    assert selected["pools"][0]["task_count"] == 80
    before = (tmp_path / "current-task-pools.json").read_bytes()
    with pytest.raises(ValueError, match="duplicates"):
        photo_pile_tasks.select_current_pools(
            tmp_path, [Path(revision_pool["manifest_path"])] * 2)
    assert (tmp_path / "current-task-pools.json").read_bytes() == before
    legacy = {key: value for key, value in scene.items() if key != "provenance"}
    legacy_path = Path(entry["root"]) / "problem.json"
    legacy_path.write_text(json.dumps(legacy))
    entry["files_sha256"]["problem.json"] = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
    native_revision = photo_pile_tasks.retask_scene(entry, tmp_path / "native_revision")
    assert native_revision["scene_family_id"] == entry["scene_family_id"]


def test_progression_allows_twenty_and_rejects_invalid_order_and_unbounded_quota():
    validate_stages((30, 40, 60, 90), 10, 100)
    validate_stages((20,), 10, 100)
    for stages in ((19, 30), (40, 30), (30, 30), (30, 100), ()):
        with pytest.raises(ValueError, match="Stages"):
            validate_stages(stages, 10, 100)
    with pytest.raises(ValueError, match="1..10"):
        validate_stages((30, 40), 11, 100)


def test_diverse_catalog_pins_assets_and_samples_distinct_families(photo_source, tmp_path, monkeypatch):
    source_xml = (photo_source / "model.xml").read_text().replace(
        "<worldbody>", '<asset><material name="table_material" rgba=".5 .5 .5 1"/></asset><worldbody>'
    ).replace('<geom type="plane"', '<geom material="table_material" type="plane"')
    (photo_source / "model.xml").write_text(source_xml)
    source_receipt = json.loads((photo_source / "receipt.json").read_bytes())
    source_receipt["model_xml_sha256"] = hashlib.sha256(source_xml.encode()).hexdigest()
    (photo_source / "receipt.json").write_text(json.dumps(source_receipt))
    root = tmp_path / "catalog-source"
    root.mkdir()
    entries = []
    for index in range(35):
        xml = (
            '<mujoco><compiler angle="radian"/><worldbody><body name="part">'
            '<freejoint name="free"/><geom name="shape" type="box" '
            'size=".004 .004 .004" mass=".002"/></body></worldbody></mujoco>'
        )
        name = f"part_{index}.xml"
        (root / name).write_text(xml)
        entries.append({
            "part_id": str(index), "family": ("brick", "plate", "tile", "slope", "round")[index % 5],
            "status": "verified", "description": "Synthetic basic part",
            "decoration": "undecorated", "ldraw_org": ["Part UPDATE 2026-01"],
            "dimensions_m": [.008, .008, .008], "mjcf": name,
            "assets": [{"path": name, "sha256": hashlib.sha256(xml.encode()).hexdigest()}],
            "qualification": {"collision_geom_count": 1},
            "geometry": {"frame_ldraw_to_mujoco": [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]],
                         "metres_per_ldu": .0004},
        })
    excluded = {**entries[0], "part_id": "excluded", "description": "Duplo Brick"}
    report = {"schema_version": 1, "catalog_id": "test-diversity", "parts": entries + [excluded]}
    manifest = root / "catalog.json"
    manifest.write_text(json.dumps(report))
    pinned = generator.pin_catalog(manifest, tmp_path / "pinned")
    assert len(json.loads(pinned.read_bytes())["parts"]) == 35
    _, original, placements = generator.prepare_model(
        photo_source, 20, 42, "catalog-diverse-v1", pinned)
    identities = [obj["part_id"] for obj in placements]
    assert len(set(identities)) == 20
    assert {entries[int(identity)]["family"] for identity in identities} == {
        "brick", "plate", "tile", "slope", "round"}
    assert generator.prepare_model(photo_source, 20, 42, "catalog-diverse-v1", pinned)[2] == placements
    assert generator.prepare_model(photo_source, 20, 43, "catalog-diverse-v1", pinned)[2] != placements
    assert original["catalog_sha256"] == hashlib.sha256(pinned.read_bytes()).hexdigest()
    (root / "part_0.xml").write_text("changed source")
    assert generator.pin_catalog(manifest, pinned.parent) == pinned
    clock = iter([0.])
    monkeypatch.setattr(generator.time, "process_time", lambda: next(clock, 150.))
    receipt = generator.generate_scene(
        photo_source, tmp_path / "diverse", count=20, seed=42,
        sampling="catalog-diverse-v1", catalog_path=pinned, color_mode="many")
    assert receipt["profile"] == generator.CATALOG_PROFILE
    assert receipt["unique_part_count"] == 20
    assert receipt["stationarity"]["maximum_final_speed"] <= .0005
    assert receipt["maximum_insertion_penetration_m"] <= .005
    assert generator.verify_scene(tmp_path / "diverse") == receipt
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "diverse" / "model.xml"))
    colors = [tuple(model.geom_rgba[model.body_geomadr[body]]) for body in range(1, model.nbody)]
    assert any(first != second for first, second in zip(colors, colors[6:]))


def test_direct_assistance_links_parent_without_a_trajectory(photo_source, tmp_path, monkeypatch):
    source = tmp_path / "fresh"
    generator.generate_scene(photo_source, source, count=20, seed=20261111)
    tasks = tmp_path / "original-tasks"
    tasks.mkdir()
    entry = photo_pile_tasks.build_problem(source, tasks / "scene_000000000")
    original_bytes = (Path(entry["root"]) / "problem.json").read_bytes()
    original = json.loads(original_bytes)
    parent = next(task for task in original["tasks"] if task["task_type"] == "grasp")
    parents = {task["task_id"]: task for task in original["tasks"]
               if task["target_instance_id"] == parent["target_instance_id"]
               and task["task_type"] in ("grasp", "place", "push")}

    def reject_step(*args, **kwargs):
        raise AssertionError("Distance variants must not require a setup trajectory")

    push_ik = pile_task_assistance.PlacementSimulation.push_ik
    push_calls = 0

    def reject_first_push(sim, *args, **kwargs):
        nonlocal push_calls
        push_calls += 1
        if push_calls == 1:
            raise RuntimeError("No fixed-wrist push IK solution for this candidate")
        return push_ik(sim, *args, **kwargs)

    monkeypatch.setattr(mujoco, "mj_step", reject_step)
    monkeypatch.setattr(pile_task_assistance.PlacementSimulation, "push_ik", reject_first_push)
    result = pile_task_assistance.generate(
        entry, tmp_path / "assistance" / "scene_000000000", task_ids=list(parents))
    revised = json.loads((Path(result["root"]) / "problem.json").read_bytes())
    assert (Path(entry["root"]) / "problem.json").read_bytes() == original_bytes
    assert revised["scene_family_id"] == original["scene_family_id"]
    assert revised["scene_id"] != original["scene_id"]
    assert result["tags"]["task_start_profile"] == pile_task_assistance.PROFILE
    assert revised["task_families"][0]["parent_task_id"] == parent["task_id"]
    assert revised["tasks"]
    assert any(task["initial_task_state"]["lifted_once"] for task in revised["tasks"])
    assert any(task["task_type"] == "push" and task["assistance"]["distance"]["active_stage"] == "push"
               for task in revised["tasks"])
    assert any("No fixed-wrist push IK" in failure["reason"]
               for group in revised["task_families"] for failure in group["unavailable"])
    for task in revised["tasks"]:
        parent = parents[task["parent_task_id"]]
        group = next(group for group in revised["task_families"]
                     if group["parent_task_id"] == parent["task_id"])
        distance = task["assistance"]["distance"]
        baseline = group["original_distance"]
        assert (distance["remaining_stage_count"] < baseline["remaining_stage_count"]
                or distance["within_stage"]["value"] < baseline["within_stage"]["value"])
        assert task["goal"] == parent["goal"]
        assert task["initial_target_position"] == parent["initial_target_position"]
        assert task["assistance"]["kind"] == pile_task_assistance.PROFILE
        assert max(task["assistance"]["physical_predicates"]["jaw_normal_forces_n"].values(),
                   default=0.) < 1.
        if task["task_type"] == "push" and distance["active_stage"] == "push":
            physical = task["assistance"]["physical_predicates"]
            assert physical["table_supported"]
            assert abs(physical["pushing_contact_gap_m"]) <= physical["pushing_contact_tolerance_m"]
        problem = load_problem(result, task["task_id"])
        sim = create_simulation(problem)
        try:
            assert not task_predicate(sim, problem)
            if not task["initial_task_state"]["lifted_once"]:
                spoofed = {**task, "initial_task_state": {"lifted_once": True}}
                with pytest.raises(ValueError, match="physically held"):
                    reset_problem(sim, {**problem, "task": spoofed})
            else:
                measured = pile_task_runtime.target_contact_state(sim, task["target_instance_id"])
                with monkeypatch.context() as patch:
                    patch.setattr(pile_task_runtime, "target_contact_state", lambda *args: {
                        **measured, "jaw_normal_forces_n": {"fixed": 2., "moving": 2.}})
                    with pytest.raises(ValueError, match="preload cap"):
                        reset_problem(sim, problem)
            reset_problem(sim, load_problem(entry, parent["task_id"]))
            assert not sim.lifted_once
            np.testing.assert_array_equal(sim.position(parent["target_instance_id"]),
                                          parent["initial_target_position"])
        finally:
            sim.close()
    approach = pile_task_assistance.distance_label("place", "approach", .05, "tool_to_target", "m")
    held = pile_task_assistance.distance_label("place", "transport", .1, "held_target_to_destination_xy", "m")
    assert approach["remaining_stage_count"] > held["remaining_stage_count"]
    assert held["remaining_stages"] == ["transport", "lower", "release", "dwell"]


def test_near_terminal_starts_measure_native_goal_gaps(photo_source, tmp_path, monkeypatch):
    source = tmp_path / "fresh"
    generator.generate_scene(photo_source, source, count=20, seed=20261111)
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    entry = photo_pile_tasks.build_problem(source, tasks / "scene")
    original_bytes = (Path(entry["root"]) / "problem.json").read_bytes()
    parents = json.loads(original_bytes)["tasks"]
    selected = [next(task["task_id"] for task in parents if task["task_type"] == kind)
                for kind in ("grasp", "push")]

    def reject_step(*args, **kwargs):
        raise AssertionError("Near-terminal construction must not execute a trajectory")

    monkeypatch.setattr(mujoco, "mj_step", reject_step)
    result = pile_task_assistance.generate(
        entry, tmp_path / "near" / "scene", task_ids=selected, near_terminal=True)
    revised = json.loads((Path(result["root"]) / "problem.json").read_bytes())
    assert (Path(entry["root"]) / "problem.json").read_bytes() == original_bytes
    assert revised["provenance"]["sampling_recipe"] == "near-terminal-v1"
    assert {task["task_type"] for task in revised["tasks"]} == {"grasp", "push"}
    for kind in ("grasp", "push"):
        children = [task for task in revised["tasks"] if task["task_type"] == kind]
        assert len(children) == 4
        for task in children:
            parent = next(parent for parent in parents if parent["task_id"] == task["parent_task_id"])
            assert task["goal"] == parent["goal"]
            assert task["initial_target_position"] == parent["initial_target_position"]
            assert task["assistance"]["sampling_recipe"] == "near-terminal-v1"
            requested = task["assistance"]["requested_goal_gap"]["value"]
            assert requested in (.0005, .001, .002, .005)
            problem = load_problem(result, task["task_id"])
            sim = create_simulation(problem)
            try:
                position = sim.position(task["target_instance_id"])
                if kind == "grasp":
                    actual = parent["initial_target_position"][2] + parent["goal"]["lift_m"] - position[2]
                else:
                    actual = (np.linalg.norm(position[:2] - parent["goal"]["destination_xy"])
                              - parent["goal"]["tolerance_m"])
                assert actual > 0
                assert actual == pytest.approx(requested, abs=.00001)
                assert task["assistance"]["distance"]["within_stage"]["value"] == pytest.approx(actual)
                assert not task_predicate(sim, problem)
            finally:
                sim.close()


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") != "osmesa", reason="Requires native CPU OSMesa rendering")
def test_near_view_uses_native_pixels_and_bbox_boundary(photo_source, tmp_path, monkeypatch):
    source = tmp_path / "fresh"
    generator.generate_scene(photo_source, source, count=20, seed=20261111)
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    entry = photo_pile_tasks.build_problem(source, tasks / "scene")
    parents = json.loads((Path(entry["root"]) / "problem.json").read_bytes())["tasks"]
    parent = next(task for task in parents if task["task_type"] == "view")

    def reject_step(*args, **kwargs):
        raise AssertionError("Native VIEW boundary construction must not execute a trajectory")

    monkeypatch.setattr(mujoco, "mj_step", reject_step)
    result = pile_task_assistance.generate(
        entry, tmp_path / "near-view" / "scene", task_ids=[parent["task_id"]], near_terminal=True)
    children = json.loads((Path(result["root"]) / "problem.json").read_bytes())["tasks"]
    assert len(children) == 4
    for task in children:
        problem = load_problem(result, task["task_id"])
        sim = create_simulation(problem, renderbackend="mujoco")
        try:
            rgb_renderer = sim.renderer
            rgb_renderer.update_scene(sim.data, camera="wrist_cam")
            before = rgb_renderer.render()
            label = sim.instance_box(task["target_instance_id"])
            adapter = NativePileSimulation.__new__(NativePileSimulation)
            adapter.native_backend = "mujoco"
            adapter.native = sim
            adapter.beam_geoms = {task["target_instance_id"]: np.flatnonzero(
                sim.model.geom_bodyid == sim.registry[task["target_instance_id"]])}
            adapter.destination_geoms = {}
            assert np.count_nonzero(adapter.entity_mask(task["target_instance_id"])) == label["visible_pixels"]
            assert sim.renderer is rgb_renderer
            assert sim.model.vis.quality.offsamples == 4
            assert sim.label_renderer._mjr_context.offSamples == 0
            rgb_renderer._gl_context.make_current()
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, rgb_renderer._mjr_context)
            rgb_renderer.update_scene(sim.data, camera="wrist_cam")
            np.testing.assert_array_equal(rgb_renderer.render(), before)
            assert label == task["assistance"]["physical_predicates"]["native_view_label"]
            assert label["visible_pixels"] >= parent["goal"]["minimum_pixels"]
            box = label["box_xyxy"]
            excess = np.linalg.norm((np.array(box[:2]) + box[2:]) / 2 - .5) - parent["goal"]["center_radius"]
            assert excess > 0
            assert excess == pytest.approx(task["assistance"]["requested_goal_gap"]["value"], abs=.0025)
            distance = task["assistance"]["distance"]["within_stage"]
            assert distance["metric"] == "native_bbox_center_radius_excess"
            assert distance["value"] == pytest.approx(excess)
            assert not task_predicate(sim, problem)
            assert task["goal"] == parent["goal"]
        finally:
            sim.close()


def test_progression_caps_each_bucket_and_resumes(photo_source, tmp_path, monkeypatch):
    generated = []
    colors = []
    monkeypatch.setattr(photo_pile_curriculum, "resource_check", lambda root: None)

    def generate(source, output, *, count, **kwargs):
        generated.append(count)
        colors.append(kwargs["color_mode"])
        output.mkdir()

    def build(source, output, model_root):
        output.mkdir()
        (output / "admission.json").write_text("{}")
        (output / "problem.json").write_text('{"tasks": [{"task_id": "test"}]}')
        return {"root": str(output)}

    def publish(root):
        return {"scenes": len(list(root.glob("scene_*/admission.json"))),
                "manifest_path": str(root / "pool.json"), "sha256": "test"}

    monkeypatch.setattr(photo_pile_curriculum, "generate_scene", generate)
    monkeypatch.setattr(photo_pile_curriculum, "build_problem", build)
    monkeypatch.setattr(photo_pile_curriculum, "publish_pool", publish)
    root = tmp_path / "curriculum"
    photo_pile_curriculum.run(root, photo_source, stages=(30, 40), quota=2, maximum_attempts=3)
    assert generated == [30, 30, 40, 40]
    monkeypatch.setattr(photo_pile_curriculum, "load_problem", lambda *args: None)
    photo_pile_curriculum.run(tmp_path / "carried", photo_source, stages=(30, 40),
                              quota=2, maximum_attempts=3, prior_root=root)
    assert generated == [30, 30, 40, 40]
    assert len(list((tmp_path / "carried").glob("pieces-*/tasks/scene_*/admission.json"))) == 4
    assert colors == ["single", "few", "single", "few"]
    assert json.loads((root / "status.json").read_text())["status"] == "complete"
    photo_pile_curriculum.run(root, photo_source, stages=(30, 40), quota=2, maximum_attempts=3)
    assert generated == [30, 30, 40, 40]


def test_common_bias_has_ninety_percent_basics_and_unique_instances(photo_source, monkeypatch):
    digest = hashlib.sha256((photo_source / "model.xml").read_bytes()).hexdigest()
    monkeypatch.setattr(generator, "PHOTO_TEMPLATE_SHA256", digest)
    xml, _, placements = generator.prepare_model(photo_source, 30, 14, "common-basic-v1")
    assert sum(p["part_id"] in generator.PHOTO_PART_IDS[:6] for p in placements) == 27
    assert len({p["instance_id"] for p in placements}) == 30
    assert mujoco.MjModel.from_xml_string(xml).nbody == 31
    assert generator.prepare_model(photo_source, 30, 14, "common-basic-v1")[0] == xml
