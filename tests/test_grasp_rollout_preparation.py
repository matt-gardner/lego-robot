import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "prepare_grasp_rollouts", Path(__file__).parents[1] / "scripts/prepare_grasp_rollouts.py",
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def admitted(parent, family, size=3, split="train"):
    return (
        {"root": parent, "split": split, "tags": {"piece_count": size}},
        {"tasks": [{"parent_task_id": parent, "task_type": "grasp",
                    "assistance": {"kind": "original-pose-grasp-v1", "value": tick / 20}}
                   for tick in range(20)],
         "provenance": {"parent_entry": {"scene_family_id": family}}},
    )


def test_selects_distinct_families_before_additional_parents_and_preserves_small_groups():
    rows = [admitted("a", "one"), admitted("b", "one"), admitted("c", "two"),
            admitted("d", "three"), admitted("small", "four", size=8),
            admitted("heldout", "five", split="validation")]
    selected = prepare.select_entries(rows, 3, 42)
    roots = {entry["root"] for entry in selected}
    assert len(roots & {"a", "b"}) == 1
    assert {"c", "d", "small"} <= roots
    assert "heldout" not in roots
    assert prepare.select_entries(rows, 3, 42) == selected


def test_incomplete_levels_are_not_admitted_as_paired_measurements():
    entry, scene = admitted("a", "one")
    scene["tasks"].pop()
    with pytest.raises(ValueError, match="complete"):
        prepare.select_entries([(entry, scene)], 4, 42)


def test_duplicate_original_parents_do_not_inflate_sample_size():
    with pytest.raises(ValueError, match="Duplicate original parent"):
        prepare.select_entries([admitted("a", "one"), admitted("a", "two")], 4, 42)


def test_summary_keeps_static_assistance_and_pile_sizes_separate():
    from so101.probe_grasp_assistance import summarize

    rows = summarize([
        {"level": .95, "piece_count": 3, "parent_task_id": "a",
         "rollouts": [{"success": True}, {"success": False}]},
        {"level": .95, "piece_count": 5, "parent_task_id": "b",
         "rollouts": [{"success": False}]},
        {"level": 0., "piece_count": 3, "parent_task_id": "a",
         "rollouts": [{"success": False}]},
    ])
    cells = {(row["assistance"], row["piece_count"]): row for row in rows}
    assert cells[.95, None]["total"] == 3
    assert cells[.95, None]["success_rate"] == pytest.approx(1 / 3)
    assert cells[.95, None]["distinct_parents"] == 2
    assert cells[.95, 3]["total"] == 2
    assert cells[0., None]["total"] == 1
    assert cells[0., None]["success_rate"] == 0.
    assert cells[0., None]["success_interval_95"][1] > 0.
