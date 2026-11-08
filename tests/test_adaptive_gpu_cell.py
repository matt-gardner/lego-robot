import importlib.util
import json
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


def test_launcher_uses_prepared_trial_recipe_and_own_zero_snapshot(monkeypatch, tmp_path):
    launcher_spec = importlib.util.spec_from_file_location(
        "start_adaptive_cell", Path(__file__).parents[1] / "scripts/start_adaptive_gpu_cell.py")
    launcher = importlib.util.module_from_spec(launcher_spec)
    launcher_spec.loader.exec_module(launcher)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    logical = tmp_path / "logical"
    logical.mkdir()
    old = "entity_smolvla_sft_b64_lr3e-05"
    argv = ["--output", f"outputs/entity_smolvla_sft_lr_sweep/{old}",
            "--batch-size", "64", "--learning-rate", "3e-5", "--steps", "384",
            "--validation-interval", "96", "--resume-state", f"outputs/{old}/training_state.pt"]
    (inputs / "initialization-commands.json").write_text(
        json.dumps([{"id": "b64-lr3e5", "gpu_argv": argv}]))
    prepared = tmp_path / "prepared"
    (prepared / "checkpoints").mkdir(parents=True)
    (prepared / "training_state.pt").write_bytes(b"pristine")
    (prepared / "checkpoints/training_state_000000.pt").write_bytes(b"zero")
    (prepared / "initialization.json").write_text(json.dumps({
        "batch_size": 32, "learning_rate": 1e-4, "steps": 768,
        "source_step": 0, "empty_adam": True,
    }))
    mapping = {"/home/ubuntu/lego-sweep/inputs": inputs,
               "/home/ubuntu/lego-sweep/output": logical}
    monkeypatch.setattr(launcher, "Path", lambda value: mapping.get(str(value), Path(value)))
    commands = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs: commands.append(command))
    root = tmp_path / "runtime"
    launcher.run(root, "image", "job", 32, 1e-4, prepared / "training_state.pt")
    command = commands[0]
    assert command[command.index("--batch-size") + 1] == "32"
    assert command[command.index("--steps") + 1] == "768"
    assert command[command.index("--validation-interval") + 1] == "192"
    assert command[command.index("--micro-batch-size") + 1] == "8"
    assert old not in " ".join(command)
    output = root / "output/entity_smolvla_sft_b32_lr1e-04"
    assert (output / "checkpoints/training_state_000000.pt").read_bytes() == b"zero"
