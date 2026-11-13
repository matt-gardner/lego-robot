from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from so101.grasp_window_demonstrations import diverse_pilot, freeze_intake, main, sha256, window_tasks
from so101.simulation_corpus import load_corpus
import so101.grasp_window_demonstrations as demonstrations


@pytest.mark.parametrize("backend", [None, "gpu-pathtrace"])
@pytest.mark.parametrize("workers", [1, 2])
def test_render_routes_explicit_backend_and_preserves_cpu_default(
    tmp_path, monkeypatch, backend, workers,
):
    intake = tmp_path / "intake.json"
    intake.write_text("{}")
    trace = tmp_path / "trace.json"
    trace.write_text("{}")
    status = tmp_path / "physical.json"
    status.write_text(json.dumps({
        "complete": True, "intake_sha256": sha256(intake),
        "reports": [{"accepted": True, "trace": str(trace), "trace_sha256": sha256(trace)}],
    }))
    selected = []

    def render(path, output, model_root, rgb_renderer):
        selected.append((path, rgb_renderer))
        return {"source": str(output)}

    monkeypatch.setattr(demonstrations, "render_window", render)
    def executor(max_workers, mp_context):
        assert max_workers == workers
        assert mp_context.get_start_method() == "spawn"
        return ThreadPoolExecutor(max_workers=max_workers)

    monkeypatch.setattr(demonstrations, "ProcessPoolExecutor", executor)
    argv = ["windows", "render", "--traces", str(status), "--intake", str(intake),
            "--output", str(tmp_path / "collection"), "--model-root", str(tmp_path),
            "--workers", str(workers)]
    if backend:
        argv += ["--rgb-renderer", backend]
    monkeypatch.setattr("sys.argv", argv)
    main()
    assert selected == [(trace, backend or "cpu-pathtrace")]


@pytest.mark.parametrize("corruption", [None, "trace", "frames", "image", "backend"])
def test_render_resume_preserves_completed_episode_and_checks_artifacts(
    tmp_path, monkeypatch, corruption,
):
    intake = tmp_path / "intake.json"
    intake.write_text("{}")
    trace = tmp_path / "trace.json"
    trace.write_text("{}")
    physical = {"complete": True, "intake_sha256": sha256(intake), "reports": [
        {"accepted": True, "trace": str(trace), "trace_sha256": sha256(trace)},
    ]}
    status = tmp_path / "physical.json"
    status.write_text(json.dumps(physical))
    output = tmp_path / "collection"
    episode = output / "trace"
    episode.mkdir(parents=True)
    (output / "intake.json").write_text("{}")
    (output / "physical-generation.json").write_text(json.dumps(physical))
    (episode / "physical-trace.json").write_text("{}")
    (episode / "frames.jsonl").write_text('{"image":"frame.png"}\n')
    (episode / "frame.png").write_bytes(b"retained-image")
    metadata = {
        "source_trace_sha256": sha256(trace), "trace_sha256": sha256(episode / "physical-trace.json"),
        "frames_sha256": sha256(episode / "frames.jsonl"), "rgb_renderer": "gpu-pathtrace",
        "maximum_replay_qpos_error": 0.0,
    }
    if corruption in ("trace", "frames"):
        (episode / ("physical-trace.json" if corruption == "trace" else "frames.jsonl")).write_text("changed")
    elif corruption == "image":
        (episode / "frame.png").unlink()
    elif corruption == "backend":
        metadata["rgb_renderer"] = "cpu-pathtrace"
    (episode / "episode.json").write_text(json.dumps(metadata))
    report = {"source": str(episode), "frames": 1}
    (output / "status.json").write_text(json.dumps({"complete": False, "episodes": [report]}))
    monkeypatch.setattr("sys.argv", [
        "windows", "render", "--traces", str(status), "--intake", str(intake),
        "--output", str(output), "--model-root", str(tmp_path),
        "--rgb-renderer", "gpu-pathtrace", "--resume", "--workers", "2",
    ])
    monkeypatch.setattr(demonstrations, "render_window",
                        lambda *args: pytest.fail("Completed episode was rendered again"))
    if corruption:
        with pytest.raises(ValueError):
            main()
    else:
        main()
        assert json.loads((output / "status.json").read_text()) == {
            "complete": True, "episodes": [report],
        }
        assert (episode / "frame.png").read_bytes() == b"retained-image"


def family(kind):
    return {"tasks": [
        {"task_type": kind, "parent_task_id": "original",
         "assistance": {"value": tick / 20}}
        for tick in reversed(range(20))
    ]}


@pytest.mark.parametrize("kind,expected", [
    ("grasp", [.7, .75, .8, .85, .9]),
    ("place", [.35, .4, .45, .5]),
])
def test_window_selects_ordered_approach_and_closure_not_transport(kind, expected):
    selected = window_tasks(family(kind))
    assert [task["assistance"]["value"] for task in selected] == expected


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "other_parent", "other_task"])
def test_incomplete_or_mixed_anchor_families_fail(corruption):
    scene = family("grasp")
    if corruption == "missing":
        scene["tasks"] = [t for t in scene["tasks"] if t["assistance"]["value"] != .85]
    elif corruption == "duplicate":
        scene["tasks"].append(scene["tasks"][0].copy())
    elif corruption == "other_parent":
        scene["tasks"][0]["parent_task_id"] = "different"
    else:
        scene["tasks"][0]["task_type"] = "place"
    with pytest.raises(ValueError):
        window_tasks(scene)


def test_pilot_covers_distinct_scenes_before_more_targets_in_same_scene():
    families = [
        {"task": kind, "parent_task_id": f"{scene}:{target}",
         "entry": {"scene_family_id": scene}}
        for kind in ("grasp", "place")
        for scene in ("a", "b", "c")
        for target in range(3)
    ]
    selected = diverse_pilot(families, 3)
    for kind in ("grasp", "place"):
        assert {f["entry"]["scene_family_id"] for f in selected if f["task"] == kind} == {"a", "b", "c"}


def test_window_collection_is_not_a_full_task_rollout_reference(tmp_path):
    (tmp_path / "collection.json").write_text(json.dumps({
        "kind": "grasp-window-sft-v1", "status": "complete", "successful_episodes": [],
    }))
    with pytest.raises(ValueError, match="not full-task"):
        load_corpus(tmp_path)


def test_freeze_rejects_problem_changed_since_admission(tmp_path):
    (tmp_path / "problem.json").write_text("{}")
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({
        "root": str(tmp_path), "files_sha256": {"problem.json": "incorrect"},
    }))
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"admissions": [str(admission)]}))
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"jobs": []}))
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    with pytest.raises(ValueError, match="Changed admitted problem"):
        freeze_intake(prior, inventory, receipts)


@pytest.mark.parametrize("changed_intake", [False, True])
def test_resume_preserves_completed_families_and_rejects_changed_intake(
    tmp_path, monkeypatch, changed_intake,
):
    intake = tmp_path / "intake.json"
    intake.write_text(json.dumps({"families": [{"task": "grasp", "parent_task_id": "parent"}]}))
    output = tmp_path / "generation"
    output.mkdir()
    report = {"task": "grasp", "parent_task_id": "parent", "accepted": False, "reason": "table"}
    (output / "status.json").write_text(json.dumps({
        "complete": False, "intake_sha256": "wrong" if changed_intake else sha256(intake),
        "planned_parents": ["parent"], "reports": [report],
    }))
    monkeypatch.setattr("sys.argv", [
        "windows", "generate", "--intake", str(intake), "--output", str(output), "--resume",
    ])
    if changed_intake:
        with pytest.raises(ValueError, match="same frozen intake"):
            main()
    else:
        main()
        status = json.loads((output / "status.json").read_text())
        assert status["complete"] is True
        assert status["reports"] == [report]
