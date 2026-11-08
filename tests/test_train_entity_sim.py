import copy
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from so101.control_model.modeling_entity_smolvla import EntityModelConfig
from so101.control_model.simulation_dataset import EntityTrajectoryDataset
from so101.evaluate_entity_sim import EntityPolicy
import so101.evaluate_entity_sim as evaluator
import so101.train_entity_sim as trainer
from so101.resume_rng import plan_device_rng


def test_storage_cli_changes_only_storage_selection(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(trainer, "train", lambda *args: calls.append(args))
    argv = ["train", "--collection", str(tmp_path), "--output", str(tmp_path / "output")]
    monkeypatch.setattr(sys, "argv", argv)
    trainer.main()
    monkeypatch.setattr(sys, "argv", argv + ["--episode-storage", str(tmp_path / "v1.sqlite")])
    trainer.main()
    assert calls[0][:-1] == calls[1][:-1]
    assert calls[0][-1] is None
    assert str(calls[1][-1]) == str(tmp_path / "v1.sqlite")


def test_triangular_schedule_peaks_at_early_warmup_then_decays():
    values = [trainer.scheduled_learning_rate(step, 5000, 1e-5, "triangular", .1)
              for step in range(5001)]
    assert values[0] == values[-1] == 0
    assert max(values) == 1e-5
    assert values.index(max(values)) == 500
    assert all(left <= right for left, right in zip(values[:500], values[1:501], strict=True))
    assert all(left >= right for left, right in zip(values[500:-1], values[501:], strict=True))
    assert trainer.scheduled_learning_rate(750, 5000, 1e-5, "triangular", .1) == values[750]
    assert trainer.scheduled_learning_rate(750, 5000, 1e-5, "constant") == 1e-5


def test_sft_full_state_migration_preserves_model_adam_sampling_and_cpu_rng(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    sampling = np.random.default_rng(42)
    config = {"device": "cpu", "learning_rate": 1e-5, "validation_interval": 96}
    path = tmp_path / "training_state.pt"
    trainer.save_training_state(path, model, optimizer, 304, sampling, {
        "run_config": config, "steps": [{"step": 304, "samples_seen": 19456}],
    })
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["rng"]["backend"] == "cpu"
    original_model = {key: value.clone() for key, value in state["model"].items()}
    original_optimizer = copy.deepcopy(state["optimizer"])
    expected_sampling = np.random.default_rng()
    expected_sampling.bit_generator.state = state["rng"]["sampling"]
    # Legacy MPS states identify their backend through run_config.device.
    state["rng"].pop("backend")
    state["log"]["run_config"]["device"] = "mps"
    state["rng"]["device"] = torch.tensor([1, 2, 3], dtype=torch.uint8)
    with pytest.raises(ValueError, match="explicitly allow"):
        plan_device_rng(state, "cuda")
    target = torch.nn.Linear(2, 1)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    target_sampling = np.random.default_rng(99)
    seeded = []
    monkeypatch.setattr(torch.cuda, "manual_seed", seeded.append)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda *args: pytest.fail("foreign RNG applied"))
    log = trainer.restore_training_state(
        state, target, target_optimizer, target_sampling, {**config, "device": "cuda"},
        plan_device_rng(state, "cuda", True, 61000),
    )
    assert seeded == [61000]
    assert log["steps"] == [{"step": 304, "samples_seen": 19456}]
    assert log["run_config"]["device"] == "cuda"
    assert log["device_rng_migrations"][-1]["bit_exact"] is False
    for name, value in target.named_parameters():
        torch.testing.assert_close(value, original_model[name], rtol=0, atol=0)
    actual_optimizer = target_optimizer.state_dict()
    for key, original in original_optimizer["state"].items():
        for name, value in original.items():
            torch.testing.assert_close(actual_optimizer["state"][key][name], value, rtol=0, atol=0)
    assert target_optimizer.param_groups[0]["lr"] == 1e-5
    assert target_sampling.random() == expected_sampling.random()
    assert random.getstate() == state["rng"]["python"]
    np.testing.assert_array_equal(np.random.get_state()[1], state["rng"]["numpy"][1])
    torch.testing.assert_close(torch.random.get_rng_state(), state["rng"]["torch"], rtol=0, atol=0)
    assert torch.load(path, map_location="cpu", weights_only=False)["log"]["run_config"]["device"] == "cpu"


@pytest.mark.parametrize("initial_updates", [0, 1])
def test_sft_same_backend_recovery_reproduces_next_optimizer_update(tmp_path, initial_updates):
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    sampling = np.random.default_rng(20)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    config = {"device": "cpu", "learning_rate": 1e-5, "validation_interval": 96}

    def update(policy, adam, generator):
        adam.zero_grad()
        batch = torch.randn(4, 2) + generator.random() + random.random() + np.random.random()
        loss = policy(batch).square().mean()
        loss.backward()
        adam.step()
        return loss.detach()

    if initial_updates:
        update(model, optimizer, sampling)
    path = tmp_path / "training_state.pt"
    history = [{"step": 1, "samples_seen": 64}] if initial_updates else []
    trainer.save_training_state(path, model, optimizer, initial_updates, sampling, {
        "run_config": config, "steps": history,
    })
    expected_loss = update(model, optimizer, sampling)
    target = torch.nn.Linear(2, 1)
    target_adam = torch.optim.AdamW(target.parameters(), lr=1e-3)
    target_rng = np.random.default_rng(999)
    state = torch.load(path, map_location="cpu", weights_only=False)
    trainer.validate_state(state, allow_untrained=True)
    if not initial_updates:
        assert state["optimizer"]["state"] == {}
        with pytest.raises(ValueError):
            trainer.validate_state(state)
    log = trainer.restore_training_state(
        state, target, target_adam, target_rng, config, plan_device_rng(state, "cpu"),
    )
    actual_loss = update(target, target_adam, target_rng)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for actual, expected in zip(target.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(target_adam.state.values(), optimizer.state.values(), strict=True):
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert log["steps"] == history
    assert "device_rng_migrations" not in log


@pytest.mark.parametrize("invalid", [
    "nonfinite_model", "nonfinite_optimizer", "mismatched_history",
    "nonzero_without_history", "initialized_adam", "missing_groups", "invalid_group", "invalid_step",
])
def test_sft_initial_checkpoint_validation_rejects_malformed_state(tmp_path, invalid):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    path = tmp_path / "training_state.pt"
    trainer.save_training_state(path, model, optimizer, 0, np.random.default_rng(42), {
        "run_config": {"device": "cpu"}, "steps": [],
    })
    state = torch.load(path, map_location="cpu", weights_only=False)
    if invalid == "nonfinite_model":
        state["model"]["weight"][0, 0] = float("nan")
    elif invalid == "nonfinite_optimizer":
        state["optimizer"]["param_groups"][0]["invalid"] = torch.tensor(float("nan"))
    elif invalid == "mismatched_history":
        state["log"]["steps"] = [{"step": 1}]
    elif invalid == "nonzero_without_history":
        state["step"] = 1
    elif invalid == "initialized_adam":
        state["optimizer"]["state"] = {0: {"step": torch.tensor(1)}}
    elif invalid == "missing_groups":
        state["optimizer"]["param_groups"] = []
    elif invalid == "invalid_group":
        state["optimizer"]["param_groups"] = [{}]
    else:
        state["step"] = False
    with pytest.raises(ValueError):
        trainer.validate_state(state, allow_untrained=True)


@pytest.mark.parametrize("warmup", [0, 1, float("nan")])
def test_triangular_schedule_rejects_invalid_warmup(warmup):
    with pytest.raises(ValueError, match="warmup"):
        trainer.scheduled_learning_rate(0, 5000, 1e-5, "triangular", warmup)


def test_frozen_sources_are_excluded_before_normalization_and_pairing(tmp_path):
    paths = []
    for index, value in enumerate((0., 10000., 2., 3.)):
        source = tmp_path / f"episode_{index}"
        source.mkdir()
        paths.append(str(source))
        (source / "episode.json").write_text(json.dumps({"task": "grasp"}))
        row = {
            "state": [value] * 6, "action": [value] * 6,
            "command_boxes": [[.5, .5, .1, 0., 0.]], "command_visibility": [1],
            "command_depth_m": [.2], "command_depth_valid": [1], "command_roles": ["object"],
        }
        (source / "frames.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "collection.json").write_text(json.dumps({"successful_episodes": paths}))
    dataset = EntityTrajectoryDataset(tmp_path, 50, 4, excluded_sources=(paths[1],))
    assert [str(source) for source, _, _ in dataset.episodes] == [paths[0], paths[2], paths[3]]
    np.testing.assert_allclose(dataset.action_mean, np.full(6, 5 / 3), atol=1e-6)
    assert [[str(dataset.episodes[index][0]) for index in group]
            for group in dataset.pair_groups] == [[paths[2], paths[3]]]


def test_evaluation_restores_training_mode_and_all_training_rng(monkeypatch):
    policy = torch.nn.Linear(1, 1).train()
    controller = SimpleNamespace(policy=policy, device="cpu")
    random.seed(8)
    np.random.seed(9)
    torch.manual_seed(10)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()
    def evaluate(*args):
        assert not policy.training
        random.random()
        np.random.random()
        torch.rand(1)
        return {"success_rate": .5}
    monkeypatch.setattr(trainer, "validation_metrics", evaluate)
    assert trainer.evaluate_without_training_rng(controller, None, [], {}, 961000) == {"success_rate": .5}
    assert policy.training and random.getstate() == python_rng
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
    torch.testing.assert_close(torch.random.get_rng_state(), torch_rng, rtol=0, atol=0)


def test_unpaired_difficulty_episodes_do_not_form_false_adjacent_pairs(tmp_path):
    paths = []
    for index, (command, scene_seed) in enumerate(
        (("grasp", 1), ("grasp", 1), ("grasp", 2), ("view", 3),
         ("place", 4), ("place", 5))
    ):
        source = tmp_path / f"episode_{index}"
        source.mkdir()
        paths.append(str(source))
        (source / "episode.json").write_text(json.dumps({
            "task": command, "scene_seed": scene_seed,
        }))
        row = {
            "state": [float(index)] * 6, "action": [float(index)] * 6,
            "command_boxes": [[.5, .5, .1, 0., 0.]], "command_visibility": [1],
            "command_depth_m": [.2], "command_depth_valid": [1], "command_roles": ["object"],
        }
        (source / "frames.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "collection.json").write_text(json.dumps({"successful_episodes": paths}))
    dataset = EntityTrajectoryDataset(tmp_path, 50, 4)
    assert len(dataset.episodes) == 6
    assert dataset.pair_groups == [[0, 1]]
    np.testing.assert_allclose(dataset.action_mean, np.full(6, 2.5), atol=1e-6)


def test_live_sft_evaluation_reuses_policy_without_loading_another_model(monkeypatch):
    def unexpected(*args):
        pytest.fail("Evaluation must not load another model into unified memory")
    monkeypatch.setattr(evaluator, "load_entity_checkpoint", unexpected)
    policy = SimpleNamespace(model=SimpleNamespace(
        vlm_with_expert=SimpleNamespace(processor=SimpleNamespace(tokenizer=object()))))
    normalization = {name: [1.] * 6 for name in ("state_mean", "state_std", "action_mean", "action_std")}
    controller = EntityPolicy(None, "cpu", True,
                              loaded_policy=(policy, EntityModelConfig(), normalization))
    assert controller.policy is policy


def test_checkpointed_microbatches_preserve_full_batch_masked_loss_and_adam_update():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(3, 2)
        def forward_entity(self, images, masks, tokens, token_masks, state, noise):
            value = self.projection(state + images[0] + noise)
            return tuple(value * scale for scale in range(1, 6))
    native = Model()
    chunked = Model()
    chunked.load_state_dict(native.state_dict())
    inputs = ([torch.randn(128, 3)], [torch.ones(128, dtype=torch.bool)],
              torch.ones(128, 3), torch.ones(128, 3), torch.randn(128, 3), torch.randn(128, 3))
    weights = (torch.arange(128) % 7 + 1).float().unsqueeze(1)
    native_optimizer = torch.optim.AdamW(native.parameters(), lr=1e-5)
    chunked_optimizer = torch.optim.AdamW(chunked.parameters(), lr=1e-5)
    expected = native.forward_entity(*inputs)
    actual = trainer.forward_supervised_batch(chunked, inputs)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right)
    expected_loss = sum((value.square() * weights).sum() / (weights.sum() * 2) for value in expected)
    actual_loss = sum((value.square() * weights).sum() / (weights.sum() * 2) for value in actual)
    torch.testing.assert_close(actual_loss, expected_loss)
    expected_loss.backward()
    actual_loss.backward()
    for left, right in zip(chunked.parameters(), native.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad)
    torch.nn.utils.clip_grad_norm_(native.parameters(), 10)
    torch.nn.utils.clip_grad_norm_(chunked.parameters(), 10)
    native_optimizer.step()
    chunked_optimizer.step()
    for left, right in zip(chunked.parameters(), native.parameters(), strict=True):
        torch.testing.assert_close(left, right)


def test_resuming_with_new_validation_interval_preserves_history_and_records_change():
    saved = {"run_config": {"validation_interval": 250, "learning_rate": 1e-5},
             "steps": [{"step": 157, "loss": .5}], "validation": [{"step": 0}]}
    result = trainer.resume_training_log(
        saved, {"validation_interval": 50, "learning_rate": 1e-5}, 157
    )
    assert result["steps"] == [{"step": 157, "loss": .5}]
    assert result["validation"] == [{"step": 0}]
    assert result["run_config"]["validation_interval"] == 50
    assert result["configuration_changes"] == [
        {"step": 157, "setting": "validation_interval", "from": 250, "to": 50}
    ]


def test_validation_interval_change_does_not_allow_learning_rate_change_on_resume():
    saved = {"run_config": {"validation_interval": 250, "learning_rate": 1e-5}}
    with pytest.raises(ValueError, match="configuration must match"):
        trainer.resume_training_log(
            saved, {"validation_interval": 50, "learning_rate": 2e-5}, 157
        )


def test_resume_accepts_removed_disabled_objective_without_rewriting_history():
    saved = {
        "run_config": {"validation_interval": 50, "learning_rate": 1e-5,
                       "contrast_coefficient": 0},
        "steps": [{"step": 157, "loss": .5, "contrast_loss": .2}],
    }
    result = trainer.resume_training_log(
        saved, {"validation_interval": 50, "learning_rate": 1e-5}, 157
    )
    assert result["run_config"] == {"validation_interval": 50, "learning_rate": 1e-5}
    assert result["steps"] == [{"step": 157, "loss": .5, "contrast_loss": .2}]


def test_resume_rejects_checkpoint_with_removed_objective_enabled():
    saved = {"run_config": {"validation_interval": 50, "contrast_coefficient": .1}}
    with pytest.raises(ValueError, match="removed contrast objective enabled"):
        trainer.resume_training_log(saved, {"validation_interval": 50}, 157)


def test_warmup_constant_holds_learning_rate_and_matches_sample_progress():
    for step in (0, 1, 19, 48, 192):
        assert trainer.scheduled_learning_rate(step, 192, 1e-5, "warmup-constant") == (
            trainer.scheduled_learning_rate(step * 2, 384, 1e-5, "warmup-constant")
        )
    assert trainer.scheduled_learning_rate(19, 192, 1e-5, "warmup-constant") == 1e-5
    assert trainer.scheduled_learning_rate(192, 192, 1e-5, "warmup-constant") == 1e-5


def test_shared_supervised_loss_preserves_padding_weighted_mean_and_gradients(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.))
            self.vlm_with_expert = SimpleNamespace(processor=SimpleNamespace(tokenizer=None))
        def sample_noise(self, shape, device):
            return torch.randn(shape, device=device)
        def sample_time(self, batch, device):
            return torch.rand(batch, device=device)
        def forward_entity(self, *inputs):
            actions = inputs[5]
            shape = (*actions.shape[:2], 4)
            anchor = self.scale * 0
            return (actions.square() * self.scale, torch.zeros(*shape, 5) + anchor,
                    torch.zeros(shape) + anchor, torch.zeros(shape) + anchor, actions)
    model = Model()
    policy = SimpleNamespace(model=model, config=SimpleNamespace(
        max_state_dim=6, max_action_dim=6, tokenizer_max_length=3))
    monkeypatch.setattr(trainer, "prepare_history_images",
                        lambda policy, images: ([images], [torch.ones(2, dtype=torch.bool)]))
    monkeypatch.setattr(trainer, "tokenize", lambda tokenizer, texts, length, device: (
        torch.ones(len(texts), length, dtype=torch.long),
        torch.ones(len(texts), length, dtype=torch.bool)))
    batch = {
        "task": ["grasp", "place"], "images": torch.zeros(2, 1),
        "history_state": torch.zeros(2, 4, 6), "history_action": torch.zeros(2, 4, 6),
        "actions": torch.tensor([1., 3.])[:, None, None].expand(2, 2, 6),
        "roles": [["object"] * 2] + [["padding"] * 2 for _ in range(3)],
        "entity_mask": torch.tensor([[True, False, False, False]] * 2),
        "entity_box": torch.zeros(2, 4, 5), "entity_visibility": torch.ones(2, 4),
        "entity_depth_m": torch.ones(2, 4), "entity_depth_valid": torch.ones(2, 4),
        "entity_camera_position": torch.zeros(2, 4, 3),
        "action_is_pad": torch.tensor([[False, False], [False, True]]),
        "future_visibility": torch.ones(2, 2, 4),
        "future_center_valid": torch.ones(2, 2, 4),
        "future_box": torch.zeros(2, 2, 4, 5),
        "future_depth_m": torch.ones(2, 2, 4), "future_depth_valid": torch.ones(2, 2, 4),
    }
    losses, _, _ = trainer.supervised_batch_losses(
        policy, batch, EntityModelConfig(), "cpu", "known", False, 1., 1., 1., .1)
    torch.testing.assert_close(losses["action_loss"], torch.tensor(11 / 3))
    torch.testing.assert_close(losses["loss"], torch.tensor(11 / 3 + .1 * np.log(2)), check_dtype=False)
    losses["loss"].backward()
    torch.testing.assert_close(model.scale.grad, torch.tensor(11 / 3))


def test_held_out_probe_reuses_loss_and_restores_rng_without_gradients(monkeypatch):
    policy = torch.nn.Linear(1, 1).train()
    def losses(policy, batch, *args):
        assert not policy.training and not torch.is_grad_enabled()
        value = policy.weight.sum() + torch.rand(()) + random.random() + np.random.random()
        return {"loss": value}, None, None
    monkeypatch.setattr(trainer, "supervised_batch_losses", losses)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    args = (policy, [torch.ones(1), torch.zeros(1)], [0, 1],
            EntityModelConfig(), "cpu", "known", 1061000, (1., 1., .1))
    first = trainer.supervised_validation_probe(*args)
    assert trainer.supervised_validation_probe(*args) == first
    assert policy.training and policy.weight.grad is None and random.getstate() == python_rng
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
    torch.testing.assert_close(torch.random.get_rng_state(), torch_rng, rtol=0, atol=0)


def test_probe_setting_cannot_be_changed_when_resuming():
    saved = {"run_config": {"validation_interval": 50}}
    with pytest.raises(ValueError, match="configuration must match"):
        trainer.resume_training_log(
            saved, {"validation_interval": 50, "supervised_validation_probe": True}, 10)


def test_failed_probe_restores_training_mode_and_rng(monkeypatch):
    policy = torch.nn.Linear(1, 1).train()
    before = torch.random.get_rng_state()
    def losses(*args):
        return {"loss": torch.tensor(float("nan"))}, None, None
    monkeypatch.setattr(trainer, "supervised_batch_losses", losses)
    with pytest.raises(FloatingPointError, match="held-out supervised loss"):
        trainer.supervised_validation_probe(
            policy, [torch.ones(1)], [0], EntityModelConfig(), "cpu", "known",
            1061000, (1., 1., .1))
    assert policy.training
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
