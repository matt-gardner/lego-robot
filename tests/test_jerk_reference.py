import json

import numpy as np
import pytest

from so101.experiments.experiment_jerk import action_jerk_gradient, pathwise_backward
from so101.jerk_reference import ReferenceJerkMargins, calibrate_margins, reference_window, problem_margin


def test_reference_window_matches_assistance_and_holds_final_command():
    actions = np.zeros((6, 6), dtype=np.float32)
    actions[:, 0] = np.arange(6) ** 2
    actions[:, 5] = np.arange(6) ** 4
    window = reference_window(actions, 6, .5)
    np.testing.assert_array_equal(window[:, 0], [9, 16, 25, 25, 25, 25])
    assert window.shape == (6, 5)
    actions[:3, :5] = 1000
    np.testing.assert_array_equal(reference_window(actions, 6, .5), window)


def test_frozen_fallback_uses_only_successful_references_and_preserves_problem_margin(tmp_path):
    sources = []
    for index, outcome in enumerate(("success", "success", "failed")):
        path = tmp_path / str(index)
        path.mkdir()
        (path / "episode.json").write_text(json.dumps({"task": "push", "outcome": outcome, "fps": 10}))
        actions = np.zeros((8, 6))
        actions[:, 0] = (index + 1) * (-1.) ** np.arange(8)
        (path / "frames.jsonl").write_text("".join(
            json.dumps({"action": row.tolist()}) + "\n" for row in actions))
        sources.append(str(path))
    margins = ReferenceJerkMargins({"push": sources}, 8, 10, hold_final=False)
    frozen = margins.calibrate_fallback()
    assert frozen["reference_count"] == 2 and frozen["excluded_count"] == 1
    assert frozen["margin"] == pytest.approx(np.percentile(
        [margins.summary(source, 0.)["margin"] for source in sources[:2]], 75))
    assert margins.summary_with_fallback(sources[0], 0., frozen) == margins.summary(sources[0], 0.)
    fallback = margins.summary_with_fallback("new_problem", 0., frozen)
    assert fallback["margin"] == frozen["margin"]
    assert fallback["reference_scope"] == "frozen_reference_pool"
    assert frozen["references"][0]["sha256"]["frames.jsonl"]
    with pytest.raises(ValueError):
        margins.summary_with_fallback("new_problem", 0., {**frozen, "margin": float("nan")})


def test_each_problem_uses_its_own_reference_and_disables_only_missing_references(tmp_path):
    sources = []
    for index, (amplitude, outcome, fps) in enumerate(((1, "success", 10), (3, "success", 10),
                                                     (100, "failed", 10), (100, "success", 20))):
        source = tmp_path / str(index)
        source.mkdir()
        (source / "episode.json").write_text(json.dumps({"task": "view", "outcome": outcome, "fps": fps}))
        actions = np.zeros((8, 6))
        actions[:, 0] = amplitude * (-1.) ** np.arange(8)
        (source / "frames.jsonl").write_text("".join(json.dumps({"action": row.tolist()}) + "\n" for row in actions))
        sources.append(str(source))
    report = calibrate_margins({"view": sources, "push": []}, 8, 0, 10)
    unit_jerk = 8 / np.sqrt(5)
    assert problem_margin(report, sources[0]) == pytest.approx(unit_jerk)
    assert problem_margin(report, sources[1]) == pytest.approx(unit_jerk * 3)
    assert problem_margin(report, sources[2]) is None
    assert problem_margin(report, sources[3]) is None
    assert report["problems"][sources[2]]["disabled_reason"] == "not_successful"
    assert report["problems"][sources[3]]["disabled_reason"] == "missing_or_mismatched_fps"
    assert problem_margin(report, "unknown") is None
    assert pathwise_backward(None, None, None, None, 1, 6, None) is None


def test_margin_zeroes_pressure_below_budget_and_preserves_a_measured_zero_budget():
    actions = np.random.default_rng(42).normal(size=(32, 6)).astype(np.float32)
    gradient, cost = action_jerk_gradient(actions, 6)
    jerk = np.sqrt(cost) * 6
    below, below_cost = action_jerk_gradient(actions, 6, jerk * 2)
    assert below_cost == 0 and not below.any()
    above, above_cost = action_jerk_gradient(actions, 6, jerk / 2)
    assert above_cost == pytest.approx(cost / 4, rel=1e-5)
    np.testing.assert_allclose(above.numpy(), gradient.numpy() / 2, atol=1e-6)
    report = {"problems": {"source": {"margin": 0, "reference_count": 1}}}
    zero_budget_gradient, zero_budget_cost = action_jerk_gradient(actions, 6, problem_margin(report, "source"))
    assert zero_budget_cost > 0 and zero_budget_gradient.any()
    quiet, quiet_cost = action_jerk_gradient(np.zeros((32, 6)), 6, .1)
    assert quiet_cost == 0 and np.isfinite(quiet.numpy()).all() and not quiet.any()


def test_production_margins_match_variable_suffix_without_artificial_hold(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "episode.json").write_text(json.dumps({"task": "view", "outcome": "success", "fps": 10}))
    actions = np.zeros((10, 6), dtype=np.float32)
    actions[:, 0] = np.arange(10) ** 2
    (source / "frames.jsonl").write_text("".join(json.dumps({"action": row.tolist()}) + "\n" for row in actions))
    production = ReferenceJerkMargins({"view": [str(source)]}, 64, 10, hold_final=False)
    fixed = ReferenceJerkMargins({"view": [str(source)]}, 64, 10)
    assert production.summary(source, .5)["margin"] == 0
    assert fixed.summary(source, .5)["margin"] > 0
    short = production.summary(source, .8)
    assert short["margin"] is None and short["reference_count"] == 0
    assert short["disabled_reason"] == "reference_window_too_short"
    assert production.summary("unreferenced", .5)["margin"] is None
    assert len(production.snapshot()["profiles"]) == 3
    assert production.summary(source, 0)["margin"] == 0


def test_problem_reference_matches_assistance_and_does_not_borrow_a_command_margin(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "episode.json").write_text(json.dumps({"task": "push", "outcome": "success", "fps": 10}))
    actions = np.zeros((12, 6), dtype=np.float32)
    actions[:6, 0] = (-1.) ** np.arange(6)
    (source / "frames.jsonl").write_text("".join(json.dumps({"action": row.tolist()}) + "\n" for row in actions))
    references = ReferenceJerkMargins({"push": [str(source), "missing"]}, 8, 10, hold_final=False)
    assert references.summary(source, 0)["margin"] > 0
    assert references.summary(source, .5)["margin"] == 0
    assert references.summary("missing", 0)["margin"] is None
    assert references.summary(source, 0)["reference_count"] == 1
