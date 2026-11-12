import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from so101.probe_grasp_assistance import select_parent_families


spec = importlib.util.spec_from_file_location(
    "prepare_grasp_rollouts", Path(__file__).parents[1] / "scripts/prepare_grasp_rollouts.py",
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_parent_subset_preserves_original_seed_indices():
    families = [{level: SimpleNamespace(metadata={"parent_task_id": f"parent-{index}"})
                 for level in (.75, .8)} for index in range(10)]
    selected = select_parent_families(families, ["parent-7", "parent-2"])
    assert [index for index, family in selected] == [2, 7]
    assert [17551000 + index for index, family in selected] == [17551002, 17551007]
    assert selected[-1][1] is families[7]
    assert len(select_parent_families(families, None)) == 10
    for invalid in ([], ["missing"], ["parent-7", "parent-7"]):
        with pytest.raises(ValueError, match="frozen parents"):
            select_parent_families(families, invalid)


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


def test_recovery_level_selection_retains_paired_frozen_semantics():
    from so101.probe_grasp_assistance import measurement_levels

    frozen = [tick / 20 for tick in range(20)]
    assert measurement_levels(frozen, [.8, .7, .75]) == [.7, .75, .8]
    assert measurement_levels(frozen, None) == frozen
    for invalid in ([], [.8, .8], [.81], [1.]):
        with pytest.raises(ValueError, match="frozen levels"):
            measurement_levels(frozen, invalid)


def test_recovery_holdout_is_by_original_scene_not_target_or_admission():
    from so101.probe_grasp_assistance import recovery_partition

    assert recovery_partition("shared-scene", ["shared-scene"]) == "recovery-training"
    assert recovery_partition("different-scene", ["shared-scene"]) == "recovery-heldout"
    assert recovery_partition("different-scene", []) == "unpartitioned"
