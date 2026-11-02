import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from so101.control_model.modeling_entity_smolvla import EntityModelConfig
from so101.control_model.simulation_dataset import EntityTrajectoryDataset
from so101.evaluate_entity_sim import EntityPolicy
import so101.evaluate_entity_sim as evaluator
import so101.train_entity_sim as trainer


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
