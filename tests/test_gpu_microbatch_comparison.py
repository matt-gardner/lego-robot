import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "gpu_microbatch_comparison", Path(__file__).parents[1] / "scripts/compare_gpu_microbatches.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
runner_spec = importlib.util.spec_from_file_location(
    "gpu_microbatch_runner", Path(__file__).parents[1] / "scripts/run_microbatch_tuning.py")
runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner)


def test_parameter_comparison_reports_population_and_normalized_difference():
    result = module.parameter_difference(
        {"weights": torch.tensor([3., 4.])}, {"weights": torch.tensor([3., 5.])})
    assert result["tensor_count"] == 1
    assert result["maximum_absolute_difference"] == 1
    assert result["relative_l2_difference"] == pytest.approx(.2)


def test_nonfinite_parameters_cannot_produce_successful_comparison():
    with pytest.raises(ValueError, match="Invalid compared parameter"):
        module.parameter_difference(
            {"weights": torch.tensor([1.])}, {"weights": torch.tensor([float("nan")])})


def test_fastest_baseline_still_compares_a_changed_feasible_microbatch():
    candidates = [
        {"status": "completed", "microbatch": 16, "steady_mean_seconds": 12,
         "max_reserved_bytes": 50, "total_memory_bytes": 100},
        {"status": "completed", "microbatch": 32, "steady_mean_seconds": 13,
         "max_reserved_bytes": 80, "total_memory_bytes": 100},
        {"status": "out_of_memory", "microbatch": 64},
    ]
    assert runner.select_candidates(candidates) == (16, 32)


def test_short_comparison_rejects_different_sample_populations():
    receipts = [
        {"status": "completed", "base_sha256": "same", "sample_fingerprints": ["a"] * 6},
        {"status": "completed", "base_sha256": "same", "sample_fingerprints": ["b"] * 6},
    ]
    with pytest.raises(ValueError, match="sample-matched"):
        module.short_comparison({}, {}, receipts)
