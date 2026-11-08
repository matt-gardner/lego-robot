import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "prepare_adaptive_cell", Path(__file__).parents[1] / "scripts/prepare_adaptive_gpu_cell.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def reference(tmp_path):
    path = tmp_path / "reference.pt"
    torch.save({
        "step": 0, "model": {"weights": torch.tensor([1., 2.])},
        "optimizer": {"state": {}, "param_groups": [{"lr": 3e-5}]},
        "rng": {"torch": torch.tensor([1, 2], dtype=torch.uint8)},
        "log": {
            "steps": [], "status": "training",
            "validation": [{"step": 0, "rollouts": [{}] * 40, "evaluation_seconds": 123}],
            "supervised_probe": {"samples": 160},
            "run_config": {"device": "cuda", "batch_size": 64, "learning_rate": 3e-5},
        },
    }, path)
    return path


def test_preparation_reuses_only_untrained_model_and_initial_evaluation(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "validate_state", lambda state, **kwargs: None)
    source = reference(tmp_path)
    before = source.read_bytes()
    output = tmp_path / "next"
    result = module.prepare(source, output, 32, 1e-4)
    prepared = torch.load(output / "training_state.pt", weights_only=False)
    original = torch.load(source, weights_only=False)
    assert result["steps"] == 768 and result["validation_interval"] == 192
    torch.testing.assert_close(prepared["model"]["weights"], original["model"]["weights"])
    torch.testing.assert_close(prepared["rng"]["torch"], original["rng"]["torch"])
    assert prepared["optimizer"]["state"] == {}
    assert prepared["optimizer"]["param_groups"][0]["lr"] == 1e-4
    assert prepared["log"]["validation"][0]["evaluation_seconds"] == 0
    assert prepared["log"]["validation"][0]["cached_initial_evaluation"]["source_evaluation_seconds"] == 123
    assert (output / "checkpoints/training_state_000000.pt").is_file()
    assert source.read_bytes() == before


def test_trained_state_cannot_be_relabelled_as_fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "validate_state", lambda state, **kwargs: None)
    source = reference(tmp_path)
    state = torch.load(source, weights_only=False)
    state["step"] = 1
    torch.save(state, source)
    with pytest.raises(ValueError, match="step zero"):
        module.prepare(source, tmp_path / "invalid", 64, 1e-4)
