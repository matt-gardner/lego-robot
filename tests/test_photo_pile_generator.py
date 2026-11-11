import hashlib
import json
from pathlib import Path
import threading

import mujoco
import numpy as np
import pytest

from so101 import photo_pile_generator as generator
from so101 import photo_pile_tasks
from so101 import photo_pile_curriculum
from so101.photo_pile_curriculum import validate_stages
from so101.pile_pool import publish_pool
from so101.pile_task_runtime import create_simulation, load_problem, reset_problem, task_predicate
from so101.rl_pile_adapter import load_native_pool


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


def test_robot_task_pool_resets_and_adapter(photo_source, tmp_path):
    source = tmp_path / "fresh"
    generator.generate_scene(photo_source, source, count=20, seed=20261111)
    root = tmp_path / "tasks"
    root.mkdir()
    entry = photo_pile_tasks.build_problem(source, root / "scene_000000000")
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


def test_progression_rejects_twenty_repeat_and_unbounded_quota():
    validate_stages((30, 40, 60, 90), 10, 100)
    for stages in ((20, 30), (40, 30), (30, 30), (30, 100), ()):
        with pytest.raises(ValueError, match="Stages"):
            validate_stages(stages, 10, 100)
    with pytest.raises(ValueError, match="1..10"):
        validate_stages((30, 40), 11, 100)


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
