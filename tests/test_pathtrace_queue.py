import pytest

from so101.pathtrace_queue import select_tasks


def test_view_exclusion_preserves_nonview_pairs_and_original_manifest():
    grasp = {"release_id": "grasp", "episodes": [
        {"task": "grasp", "frames": 5}, {"task": "grasp", "frames": 8},
    ]}
    view = {"release_id": "view", "episodes": [
        {"task": "view", "frames": 2}, {"task": "view", "frames": 3},
    ]}
    original = {"releases": [grasp, view], "episodes": 4, "frames": 18}
    selected = select_tasks(original, {"view"})
    assert selected["releases"] == [grasp]
    assert selected["excluded_releases"] == [view]
    assert selected["episodes"] == 2
    assert selected["frames"] == 13
    assert original["releases"] == [grasp, view]


def test_filter_rejects_splitting_a_command_pair():
    with pytest.raises(ValueError, match="paired release"):
        select_tasks({"releases": [{"episodes": [
            {"task": "grasp", "frames": 5}, {"task": "view", "frames": 8},
        ]}]}, {"view"})
