import pytest

from so101.analysis.analyze_jerk_experiment import analyze, keyed


def row(source, noise, jerk, steps=64):
    return {"source": source, "noise_seed": noise, "rms_jerk": jerk, "steps": steps}


def experiment(before, after):
    return {
        "run_config": {"horizon": 64, "assistance": .8},
        "fixed_problems": ["a", "b"], "steps": [{}],
        "probes": [
            {"trajectories": rows, "success_rate": .5, "full_horizon_rate": .75}
            for rows in (before, after)
        ],
    }


def test_seed_pairing_excludes_truncation_at_either_endpoint():
    before = [row("a", 1, 10), row("a", 2, 20, 60), row("b", 1, 4), row("b", 2, 4)]
    after = [row("b", 2, 2), row("a", 2, 0, 63), row("a", 1, 5), row("b", 1, 2)]
    result = analyze(experiment(before, after))
    assert result["paired_complete_rollouts"] == 3
    assert result["problems_improved"] == 2
    assert result["pooled_paired_complete_fractional_change"] == -.5
    assert result["problem_balanced_paired_complete_fractional_change"] == -.5
    assert result["problem_cluster_bootstrap_95pct_fractional_change"] == [-.5, -.5]


def test_unbalanced_valid_pairs_do_not_mislabel_pooled_estimate_as_problem_balanced():
    before = [row("a", 1, 2), row("a", 2, 2), row("b", 1, 2)]
    after = [row("a", 1, 1), row("a", 2, 1), row("b", 1, 4)]
    result = analyze(experiment(before, after))
    assert result["pooled_paired_complete_fractional_change"] == 0
    assert result["problem_balanced_paired_complete_fractional_change"] == .25


def test_legacy_repeat_order_and_duplicate_noise_detection():
    result = keyed([{"source": "a"}, {"source": "a"}])
    assert list(result) == [("a", 0), ("a", 1)]
    with pytest.raises(ValueError, match="Duplicate"):
        keyed([row("a", 1, 2), row("a", 1, 3)])
