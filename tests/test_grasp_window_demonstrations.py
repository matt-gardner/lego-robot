import json

import pytest

from so101.grasp_window_demonstrations import diverse_pilot, freeze_intake, main, sha256, window_tasks
from so101.simulation_corpus import load_corpus


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
