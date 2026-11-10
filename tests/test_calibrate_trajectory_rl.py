from types import SimpleNamespace

import numpy as np

from so101.calibrate_trajectory_rl import finished_cell, save_progress, select_problems, summarize


def test_problem_selection_prefers_different_families_and_reproduces():
    episodes = [
        SimpleNamespace(source=f"{family}:{target}", metadata={"scene_family_id": family})
        for family in ("a", "b", "c") for target in range(4)
    ]
    first = select_problems(episodes, 3, np.random.default_rng(42))
    assert len({episode.metadata["scene_family_id"] for episode in first}) == 3
    assert [item.source for item in first] == [
        item.source for item in select_problems(episodes, 3, np.random.default_rng(42))
    ]


def test_targets_require_repeated_within_problem_mixing_not_aggregate_half_success():
    rows = []
    for bucket, assistance, successes in [("easy", 0., [4, 0, 4]), ("target", .5, [2, 1, 3])]:
        for index, count in enumerate(successes):
            rows.append({
                "task": "grasp", "bucket": bucket, "assistance": assistance,
                "source": str(index), "scene_family": str(index),
                "successes": count, "invalid_actions": 0,
            })
    result = summarize(rows)
    targets = result["targets"]["grasp"]["candidates"]
    assert len(targets) == 1
    assert targets[0]["bucket"] == "target" and targets[0]["assistance"] == .5
    assert targets[0]["rollouts"] == 12 and targets[0]["mixed_group_fraction"] == 1.
    assert result["targets"]["view"]["status"] == "no-repeated-mixed-groups-yet"


def test_five_problems_use_extra_targets_when_only_two_scene_families_exist():
    episodes = [
        SimpleNamespace(source=f"{family}:{target}", metadata={"scene_family_id": family})
        for family in ("a", "b") for target in range(4)
    ]
    selected = select_problems(episodes, 5, np.random.default_rng(42))
    assert len({episode.source for episode in selected}) == 5
    assert len({episode.metadata["scene_family_id"] for episode in selected}) == 2
    assert len(select_problems(episodes[:4], 5, np.random.default_rng(42))) == 4


def test_partial_results_do_not_stop_five_problem_calibration_and_table_updates(tmp_path):
    rows = [{
        "task": "grasp", "bucket": "test", "assistance": 0., "source": str(index),
        "scene_family": str(index), "successes": 2, "invalid_actions": 0,
    } for index in range(5)]
    assert not finished_cell(summarize(rows[:2])["cells"], "grasp", "test", 5)
    assert finished_cell(summarize(rows)["cells"], "grasp", "test", 5)
    output = tmp_path / "calibration.json"
    result = {"plan": rows, "groups": rows[:2], "complete": False}
    save_progress(output, result)
    assert "Testing 0: 2/2 mixed (2/5 problems)" in output.with_suffix(".md").read_text()
    result.update(groups=rows, complete=True)
    save_progress(output, result)
    assert "**0** (5/5 mixed)" in output.with_suffix(".md").read_text()
