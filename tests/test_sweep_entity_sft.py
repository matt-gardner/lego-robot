from pathlib import Path

import pytest

from so101.sweep_entity_sft import command, common_success_jerk, trials


def test_sweep_matches_exposure_and_evaluation_points_not_update_count():
    plan = trials(24576, Path("outputs/sweep"))
    assert len(plan) == 6
    assert {(t["batch_size"], t["learning_rate"]) for t in plan} == {
        (batch, rate) for batch in (64, 128) for rate in (3e-6, 1e-5, 3e-5)
    }
    for trial in plan:
        assert trial["steps"] * trial["batch_size"] == 24576
        assert [step * trial["batch_size"] for step in range(
            0, trial["steps"] + 1, trial["validation_interval"])
        ] == [0, 6144, 12288, 18432, 24576]
        assert trial["validation_interval"] % 3 == 0
        args = command(trial)
        assert args[args.index("--learning-rate-schedule") + 1] == "warmup-constant"
        assert "--supervised-validation-probe" in args
        assert "--retain-validation-checkpoints" in args
        assert int(args[args.index("--checkpoint-interval") + 1]) > 0


@pytest.mark.parametrize("samples", [0, 10000, 24577])
def test_sweep_rejects_unmatched_or_unbalanced_budgets(samples):
    with pytest.raises(ValueError, match="multiple of 1536"):
        trials(samples, Path("outputs/sweep"))


def test_jerk_comparison_excludes_failed_and_unshared_problems():
    left = {"output": "left", "rollouts": [
        {"source": "a", "success": True, "rms_jerk": 3},
        {"source": "b", "success": False, "rms_jerk": .1},
    ]}
    right = {"output": "right", "rollouts": [
        {"source": "a", "success": True, "rms_jerk": 2},
        {"source": "b", "success": True, "rms_jerk": 4},
    ]}
    comparison = common_success_jerk([left, right])[0]
    assert comparison["problems"] == 1
    assert comparison["left_rms_jerk"] == 3
    assert comparison["right_rms_jerk"] == 2


def test_cuda_device_selection_preserves_every_recipe_argument():
    for trial in trials(24576, Path("outputs/sweep")):
        original = command(trial)
        cuda = command(trial, "cuda")
        assert original[original.index("--device") + 1] == "mps"
        cuda[cuda.index("--device") + 1] = "mps"
        assert cuda == original
        migrated = command(trial, "cuda", True, 61000)
        assert migrated[:-3] == command(trial, "cuda")
        assert migrated[-3:] == ["--allow-device-rng-migration", "--device-rng-seed", "61000"]


@pytest.mark.parametrize("seed", [None, True, -1, 2**64])
def test_sweep_migration_requires_valid_explicit_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        command(trials(24576, Path("outputs/sweep"))[0], "cuda", True, seed)


def test_storage_flag_preserves_original_recipe_and_device_migration():
    trial = trials(24576, Path("outputs/sweep"))[1]
    original = command(trial, "cuda", True, 61000)
    indexed = command(trial, "cuda", True, 61000, Path("/data/episodes-v1.sqlite"))
    assert indexed[:-2] == original
    assert indexed[-2:] == ["--episode-storage", "/data/episodes-v1.sqlite"]
