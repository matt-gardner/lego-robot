import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "publish_rl_checkpoint",
    Path(__file__).parents[1] / "scripts/publish_rl_checkpoint.py",
)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


@pytest.mark.parametrize("mutation,error", [
    ("objective", "objective"), ("parameters", "Unexpected trainable"),
])
def test_recovery_rejects_mismatched_training_contract(tmp_path, monkeypatch, mutation, error):
    state = {
        "step": 1,
        "model": {"model.action_in_proj.weight": torch.ones(2)},
        "optimizer": {"state": {0: {"exp_avg": torch.ones(2)}}},
        "rng": {key: None for key in ("python", "numpy", "sampling", "torch", "device")},
        "log": {
            "steps": [{"step": 1}], "validation": [{"step": 0}],
            "rl_estimator": "denoising-grpo",
            "training_objective": "task-reward-plus-reference-jerk",
            "jerk_regularizer": {"weight": .02, "scale": 6},
            "assistance_curriculum": {"tasks": {}}, "run_config": {"seed": 51000},
        },
    }
    monkeypatch.setattr(publisher, "plan_device_rng", lambda *args: None)
    path = tmp_path / "state.pt"
    torch.save(state, path)
    assert publisher.check(path)["step"] == 1
    state["log"]["validation"] = []
    torch.save(state, path)
    with pytest.raises(ValueError, match="history"):
        publisher.check(path)
    state["log"]["run_config"].update(initial_validation=False, validation_interval=5)
    torch.save(state, path)
    assert publisher.check(path)["step"] == 1
    state["step"] = 5
    state["log"]["steps"] = [{"step": step} for step in range(1, 6)]
    torch.save(state, path)
    with pytest.raises(ValueError, match="history"):
        publisher.check(path)
    state["step"] = 1
    state["log"]["steps"] = [{"step": 1}]
    state["log"]["validation"] = [{"step": 0}]
    if mutation == "objective":
        state["log"]["training_objective"] = "reward-only"
    else:
        state["model"]["model.backbone.weight"] = torch.ones(2)
    torch.save(state, path)
    with pytest.raises(ValueError, match=error):
        publisher.check(path)
    if mutation == "parameters":
        state["model"] = {
            "model.action_in_proj.weight": torch.ones(2),
            "model.vlm_with_expert.vlm.weight": torch.ones(3),
        }
        state["log"].update(trainable_scope="all", trainable_parameter_count=5)
        torch.save(state, path)
        assert publisher.check(path)["step"] == 1
        state["model"].pop("model.vlm_with_expert.vlm.weight")
        torch.save(state, path)
        with pytest.raises(ValueError, match="Incomplete"):
            publisher.check(path)
