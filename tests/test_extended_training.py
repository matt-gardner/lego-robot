import importlib.util
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from so101.train_entity_sim import scheduled_learning_rate


spec = importlib.util.spec_from_file_location(
    "prepare_extended_training",
    Path(__file__).parents[1] / "scripts/prepare_extended_training.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def reference(tmp_path):
    path = tmp_path / "source.pt"
    torch.save({
        "step": 384,
        "model": {"weight": torch.tensor([1., 2.])},
        "optimizer": {
            "state": {0: {"step": torch.tensor(384.), "exp_avg": torch.tensor([.2, .3]),
                          "exp_avg_sq": torch.tensor([.04, .09])}},
            "param_groups": [{"params": [0], "lr": 3e-5}],
        },
        "rng": {
            "torch": torch.random.get_rng_state(),
            "device": torch.tensor([1, 2], dtype=torch.uint8), "backend": "cuda",
            "sampling": np.random.default_rng(1).bit_generator.state,
            "python": random.Random(1).getstate(), "numpy": np.random.RandomState(1).get_state(),
        },
        "log": {
            "steps": [{"step": i, "loss": .5, "samples_seen": i * 64} for i in range(1, 385)],
            "validation": [{"step": 384, "rollouts": [
                {"task": f"task-{i}", "source": f"source-{i}", "seed": 961000 + i}
                for i in range(40)]}],
            "dataset": {"normalization": {}},
            "supervised_probe": {"samples": 160},
            "run_config": {
                "steps": 384, "batch_size": 64, "learning_rate": 3e-5,
                "learning_rate_schedule": "warmup-constant", "warmup_fraction": .1,
                "device": "cuda", "gradient_accumulation": True, "micro_batch_size": 8,
                "validation_interval": 96, "output": "original",
                "training_state": "original/training_state.pt",
            },
        },
    }, path)
    return path


def test_extension_preserves_history_adam_rng_and_original_schedule(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "validate_state", lambda state: None)
    source = reference(tmp_path)
    original_bytes = source.read_bytes()
    result = module.prepare(source, tmp_path / "extended", 16384)
    extended = torch.load(tmp_path / "extended/training_state.pt", weights_only=False)
    original = torch.load(source, weights_only=False)
    for container, key in (("model", "weight"), ("rng", "torch")):
        assert torch.equal(extended[container][key], original[container][key])
    for key in ("step", "exp_avg"):
        assert torch.equal(extended["optimizer"]["state"][0][key],
                           original["optimizer"]["state"][0][key])
    assert extended["log"]["steps"] == original["log"]["steps"]
    assert extended["log"]["validation"] == original["log"]["validation"]
    assert extended["log"]["continuations"][0]["previous_run_config"] == original["log"]["run_config"]
    for step in range(385):
        assert scheduled_learning_rate(step, 384, 3e-5, "warmup-constant", .1) == (
            scheduled_learning_rate(step, 16384, 3e-5, "warmup-constant", result["warmup_fraction"]))
    assert scheduled_learning_rate(16384, 16384, 3e-5, "warmup-constant",
                                   result["warmup_fraction"]) == 3e-5
    assert source.read_bytes() == original_bytes


@pytest.mark.parametrize("setting,value", [("learning_rate", 3e-4), ("steps", 768)])
def test_extension_rejects_wrong_recipe_or_incomplete_source(monkeypatch, tmp_path, setting, value):
    monkeypatch.setattr(module, "validate_state", lambda state: None)
    source = reference(tmp_path)
    state = torch.load(source, weights_only=False)
    state["log"]["run_config"][setting] = value
    torch.save(state, source)
    with pytest.raises(ValueError, match="completed B64"):
        module.prepare(source, tmp_path / "invalid", 16384)


def checker():
    spec = importlib.util.spec_from_file_location(
        "publish_long_gpu_checkpoint",
        Path(__file__).parents[1] / "scripts/publish_long_gpu_checkpoint.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_recovery_validation_accepts_extended_state_without_losing_frozen_history(tmp_path):
    source = reference(tmp_path)
    module.prepare(source, tmp_path / "extended", 16384)
    snapshot = tmp_path / "extended/training_state.pt"
    state = torch.load(snapshot, weights_only=False)
    state["step"] = 385
    state["log"]["steps"].append({"step": 385, "loss": .4, "samples_seen": 385 * 64})
    state["optimizer"]["state"][0]["step"] = torch.tensor(385.)
    torch.save(state, snapshot)
    assert checker().check(snapshot, source)["step"] == 385
    state["log"]["validation"][0]["rollouts"][0]["seed"] += 1
    torch.save(state, snapshot)
    with pytest.raises(ValueError, match="rollout population"):
        checker().check(snapshot, source)


def test_recovery_validation_rejects_recipe_change_even_when_declared_in_journal(tmp_path):
    source = reference(tmp_path)
    module.prepare(source, tmp_path / "extended", 16384)
    snapshot = tmp_path / "extended/training_state.pt"
    state = torch.load(snapshot, weights_only=False)
    state["log"]["run_config"]["history_steps"] = 3
    state["log"]["continuations"][0]["new_run_config"]["history_steps"] = 3
    torch.save(state, snapshot)
    with pytest.raises(ValueError, match="authorized constant-rate"):
        checker().check(snapshot, source)
