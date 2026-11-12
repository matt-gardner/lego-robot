from dataclasses import asdict
import hashlib
import json
from types import MethodType, SimpleNamespace

import pytest
import torch

from so101.control_model import modeling_entity_smolvla as modeling
import so101.fit_grasp_recovery as fitter
from so101.fit_grasp_recovery import (
    current_grasp_loss, load_current_grasp_labels, recovery_batch, recovery_loss,
    recovery_parameters,
)


def sample(valid):
    return {"inputs": ([torch.ones(1, 3, 2, 2)], [torch.ones(1, dtype=torch.bool)],
                       *[torch.ones(1, 2) for _ in range(13)]),
            "actions": torch.ones(1, 3, 6), "valid_steps": valid}


def test_recorded_input_order_and_future_action_padding():
    inputs, valid = recovery_batch([sample(1), sample(3)], 32, "cpu")
    assert len(inputs) == 16
    assert inputs[0][0].shape == (2, 3, 2, 2)
    assert inputs[5].shape == (2, 3, 32)
    assert torch.all(inputs[5][..., :6] == 1)
    assert torch.all(inputs[5][..., 6:] == 0)
    assert valid.tolist() == [[True, False, False], [True, True, True]]


def test_loss_ignores_padded_steps_and_non_robot_dimensions():
    losses = torch.ones(2, 3, 32, requires_grad=True)
    valid = torch.tensor([[True, False, False], [True, True, True]])
    with torch.no_grad():
        losses[0, 1:] = 999
        losses[..., 6:] = 999
    loss = recovery_loss(losses, valid)
    assert loss.item() == 1.
    loss.backward()
    assert torch.all(losses.grad[0, 1:] == 0)
    assert torch.all(losses.grad[..., 6:] == 0)
    assert losses.grad[..., :6].sum().item() == pytest.approx(1.)


@pytest.mark.parametrize("valid", [0, 4])
def test_rejects_empty_or_out_of_range_teacher_targets(valid):
    with pytest.raises(ValueError, match="future-action length"):
        recovery_batch([sample(valid), sample(1)], 32, "cpu")


@pytest.mark.parametrize("scope", ["action-output", "action-expert"])
def test_recovery_scope_updates_only_selected_action_layers(scope):
    policy = torch.nn.Module()
    policy.model = torch.nn.Module()
    model = policy.model
    model.vision = torch.nn.Linear(2, 2)
    model.entity_features = torch.nn.Linear(2, 2)
    model.vlm_with_expert = torch.nn.Module()
    model.vlm_with_expert.lm_expert = torch.nn.Linear(2, 2)
    model.action_in_proj = torch.nn.Linear(2, 2)
    model.action_out_proj = torch.nn.Linear(2, 6)
    for parameter in policy.parameters():
        torch.nn.init.constant_(parameter, .2)
    before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    optimizer = torch.optim.SGD(recovery_parameters(policy, scope), lr=.1)
    features = model.vision(torch.ones(2, 2)) + model.entity_features(torch.ones(2, 2))
    output = model.action_out_proj(model.vlm_with_expert.lm_expert(model.action_in_proj(features)))
    output.square().mean().backward()
    optimizer.step()
    for name, parameter in policy.named_parameters():
        expected = name.startswith("model.action_out_proj.") or (
            scope == "action-expert" and name.startswith((
                "model.action_in_proj.", "model.vlm_with_expert.lm_expert.",
            ))
        )
        assert parameter.requires_grad == expected
        assert (not torch.equal(before[name], parameter)) == expected


def test_unknown_recovery_scope_is_rejected():
    with pytest.raises(ValueError, match="Unknown recovery trainable scope"):
        recovery_parameters(torch.nn.Linear(2, 2), "vision")


def grasp_policy():
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
            self.expert_hidden_size = 8
            self.language = torch.nn.Embedding(10, 8)
            self.prefix = torch.nn.Linear(8, 8)
            self.lm_expert = torch.nn.Linear(8, 8)

        def embed_language_tokens(self, tokens):
            return self.language(tokens)

        def forward(self, *, attention_mask, inputs_embeds, **kwargs):
            prefix, suffix = inputs_embeds
            if suffix is None and not kwargs.get("use_cache"):
                raise ValueError("Cross-attention expert requires prefix prefill for observation-only input")
            tokens = prefix if suffix is None else torch.cat((prefix, suffix), dim=1)
            weights = attention_mask.float()
            features = self.prefix(weights @ tokens / weights.sum(-1, keepdim=True).clamp_min(1))
            return ((features[:, :prefix.shape[1]],
                     None if suffix is None else self.lm_expert(features[:, prefix.shape[1]:])), None)

    model = modeling.EntityVLAFlowMatching.__new__(modeling.EntityVLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.vlm_with_expert = Backbone()
    model.config = SimpleNamespace(chunk_size=3, use_cache=False)
    model.action_in_proj = torch.nn.Linear(6, 8)
    model.action_out_proj = torch.nn.Linear(8, 6)
    model.init_entity_modules(modeling.EntityModelConfig())

    def prefix(self, *inputs):
        entity = self._entity_embeddings(*inputs[7:])
        history = self.history_motion_proj(torch.cat((inputs[5], inputs[6]), dim=-1))
        history = history + self.history_time_embedding(torch.arange(4))[None]
        image = inputs[0][0].mean((1, 2, 3))[:, None, None].expand(-1, 1, 8)
        values = torch.cat((image, history, entity), dim=1)
        pad = torch.cat((torch.ones(values.shape[0], 5, dtype=torch.bool), inputs[-1]), dim=1)
        attention = torch.ones_like(pad)
        attention[:, 0] = False
        return values, pad, attention, entity

    def suffix(self, actions, time):
        values = self.action_in_proj(actions)
        mask = torch.ones(actions.shape[:2], dtype=torch.bool)
        attention = torch.zeros_like(mask)
        attention[:, 0] = True
        return values, mask, attention

    model._prefix_with_entities = MethodType(prefix, model)
    model.embed_suffix = MethodType(suffix, model)
    policy = torch.nn.Module()
    policy.model = model
    observation = (
        [torch.ones(2, 3, 2, 2)], [torch.ones(2, dtype=torch.bool)],
        torch.ones(2, 2, dtype=torch.long), torch.ones(2, 2, dtype=torch.bool),
        torch.ones(2, 6), torch.ones(2, 4, 6), torch.ones(2, 4, 6),
        torch.ones(2, 4, 2, dtype=torch.long), torch.ones(2, 4, 2, dtype=torch.bool),
        torch.ones(2, 4, 5), torch.ones(2, 4), torch.ones(2, 4), torch.ones(2, 4),
        torch.ones(2, 4, 3), torch.tensor([[True, False, False, False]] * 2),
    )
    return policy, observation


def test_current_grasp_is_future_action_invariant_and_updates_shared_prefix_only():
    policy, observation = grasp_policy()
    model = policy.model
    original = {key: value.clone() for key, value in policy.state_dict().items()}
    config = model.enable_current_grasp_prediction()
    assert config.current_grasp_prediction
    for key, value in original.items():
        torch.testing.assert_close(policy.state_dict()[key], value, rtol=0, atol=0)
    parameters = recovery_parameters(policy, "action-expert-grasp")
    future = torch.zeros(2, 3, 6, requires_grad=True)
    first = model.forward_entity(*observation[:5], future, *observation[5:],
                                 noise=torch.zeros_like(future), time=torch.full((2,), .5))
    logits = model.current_grasp_logits(observation)
    changed = model.forward_entity(*observation[:5], future + 10, *observation[5:],
                                   noise=torch.zeros_like(future), time=torch.full((2,), .5))
    assert not torch.equal(first[-1], changed[-1])
    torch.testing.assert_close(model.current_grasp_logits(observation), logits, rtol=0, atol=0)
    labels = torch.zeros_like(logits)
    known = observation[-1][..., None].expand_as(labels)
    loss = current_grasp_loss(logits, labels, known)
    before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    loss.backward()
    assert future.grad is None
    assert model.vlm_with_expert.prefix.weight.grad is None
    assert model.vlm_with_expert.lm_expert.weight.grad is None
    assert model.action_out_proj.weight.grad is None
    for module in (model.entity_numeric_proj, model.history_motion_proj, model.current_grasp_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    torch.optim.SGD(parameters, lr=.1).step()
    assert not torch.equal(before["model.entity_numeric_proj.0.weight"],
                           model.entity_numeric_proj[0].weight)
    with pytest.raises(ValueError, match="observation-only"):
        model.current_grasp_logits((*observation, future))


def test_current_grasp_loss_excludes_padding_and_rejects_empty_or_invalid_known_labels():
    logits = torch.tensor([[[0., 0.], [float("nan"), float("nan")]]], requires_grad=True)
    targets = torch.tensor([[[1., 0.], [float("nan"), float("nan")]]])
    mask = torch.tensor([[[True, True], [False, False]]])
    loss = current_grasp_loss(logits, targets, mask)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.)).item())
    loss.backward()
    assert torch.equal(logits.grad[0, 1], torch.zeros(2))
    with pytest.raises(ValueError, match="no known"):
        current_grasp_loss(logits, targets, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="finite"):
        current_grasp_loss(logits, targets, torch.ones_like(mask))


@pytest.mark.parametrize("enabled", [False, True])
def test_current_grasp_checkpoint_roundtrip_and_missing_head_rejection(tmp_path, monkeypatch, enabled):
    policy, observation = grasp_policy()
    if enabled:
        policy.model.enable_current_grasp_prediction()
        optimizer = torch.optim.SGD(recovery_parameters(policy, "action-expert-grasp"), lr=.1)
        policy.model.current_grasp_logits(observation).square().mean().backward()
        optimizer.step()
    path = tmp_path / "checkpoint"
    modeling.save_entity_checkpoint(path, policy, policy.model.entity_config, "fake", {})
    if not enabled:
        assert "current_grasp_prediction" not in json.loads((path / "entity_config.json").read_text())
        legacy = asdict(policy.model.entity_config)
        legacy.pop("current_grasp_prediction")
        (path / "entity_config.json").write_text(json.dumps(legacy))
    monkeypatch.setattr(modeling.SmolVLAPolicy, "from_pretrained", lambda _: grasp_policy()[0])
    monkeypatch.setattr(modeling, "attach_entity_modules",
                        lambda restored, config: restored.model.init_entity_modules(config))
    restored, config, _ = modeling.load_entity_checkpoint(path, "cpu")
    assert config.current_grasp_prediction == enabled
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    if enabled:
        torch.testing.assert_close(restored.model.current_grasp_logits(observation),
                                   policy.model.current_grasp_logits(observation), rtol=0, atol=0)
        state = torch.load(path / "model.pt", weights_only=True)
        del state["model.current_grasp_head.weight"]
        torch.save(state, path / "model.pt")
        with pytest.raises(ValueError, match="compatible trained head"):
            modeling.load_entity_checkpoint(path, "cpu")
    else:
        with pytest.raises(ValueError, match="not enabled"):
            restored.model.current_grasp_logits(observation)


def label_fixture(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    scene = tmp_path / "scene"
    scene.mkdir()
    scene_path = scene / "problem.json"
    scene_path.write_text(json.dumps({"tasks": [{
        "task_id": "task", "parent_task_id": "parent", "target_instance_id": "target",
    }]}))
    report = {"source": str(scene / "tasks/task"), "parent_task_id": "parent",
              "trace_sha256": "trace", "steps": 2, "samples": 2}
    manifest_path = data / "results.json"
    manifest_path.write_text(json.dumps({"reports": [report]}))
    samples = [sample(1), sample(1)]
    for index, row in enumerate(samples):
        row.update(step=index, parent_task_id="parent")
        inputs = list(row["inputs"])
        inputs[-1] = torch.tensor([[True, False]])
        row["inputs"] = tuple(inputs)
    bundle = {
        "kind": "exact-native-cpu-current-state-labels", "schema_version": 1,
        "samples_sha256": "samples",
        "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "qualifications": [{
            **report, "maximum_qpos_error": 0., "maximum_target_position_error": 0.,
            "qualified_states": 3, "recorded_steps": 2, "model_sha256": "model",
            "scene_files_sha256": {"model.mjb": "model",
                                  "problem.json": hashlib.sha256(scene_path.read_bytes()).hexdigest()},
        }],
        "samples": [{
            "sample_index": index, "report_index": 0, "input_step": index,
            "parent_task_id": "parent", "target_instance_id": "target",
            "entity_instance_ids": ["target", None],
            "target_native_current_predicate": bool(index),
            "pieces": {"target": {"bilateral_grip": True}},
        } for index in range(2)],
    }
    return samples, data, bundle


@pytest.mark.parametrize("corruption", [None, "time", "identity", "qualification"])
def test_current_labels_bind_input_time_identity_and_qualified_replay(tmp_path, corruption):
    samples, data, bundle = label_fixture(tmp_path)
    if corruption == "time":
        bundle["samples"][0]["input_step"] += 1
    elif corruption == "identity":
        bundle["samples"][0]["entity_instance_ids"] = [None, "target"]
    elif corruption == "qualification":
        bundle["qualifications"][0]["maximum_qpos_error"] = 1e-4
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(bundle))
    if corruption:
        with pytest.raises(ValueError):
            load_current_grasp_labels(samples, data, path, "samples")
    else:
        targets, known = load_current_grasp_labels(samples, data, path, "samples")
        assert targets[:, 0].tolist() == [[1., 0.], [1., 1.]]
        assert known[:, 0].all() and not known[:, 1].any()


def test_non_target_native_state_is_unknown_not_a_negative_label(tmp_path):
    samples, data, bundle = label_fixture(tmp_path)
    for sample_row, label in zip(samples, bundle["samples"], strict=True):
        sample_row["entity_instance_ids"] = ["other", "target"]
        sample_row["inputs"][-1][:] = True
        label["entity_instance_ids"] = ["other", "target"]
        label["pieces"]["other"] = {"bilateral_grip": False}
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(bundle))
    targets, known = load_current_grasp_labels(samples, data, path, "samples")
    assert known[:, :, 0].all()
    assert not known[:, 0, 1].any() and known[:, 1, 1].all()


def test_joint_recovery_fit_uses_bound_current_labels_and_exports_head(tmp_path, monkeypatch):
    _, data, bundle = label_fixture(tmp_path)
    policy, observation = grasp_policy()
    policy.config = SimpleNamespace(max_action_dim=6)
    policy.model.sample_noise = lambda shape, device: torch.zeros(shape, device=device)
    policy.model.sample_time = lambda batch, device: torch.full((batch,), .5, device=device)
    samples = []
    for index in range(2):
        inputs = tuple(
            [item[index:index + 1] for item in value] if isinstance(value, list)
            else value[index:index + 1]
            for value in observation
        )
        samples.append({"inputs": inputs, "actions": torch.ones(1, 3, 6), "valid_steps": 3,
                        "parent_task_id": "parent", "step": index})
        bundle["samples"][index]["entity_instance_ids"] = ["target", None, None, None]
    torch.save(samples, data / "samples.pt")
    manifest_path = data / "results.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(complete=True, samples=2, policy_sha256=fitter.policy_digest(policy))
    manifest["reports"][0].update(replay_success=True, maximum_qpos_error=0.)
    manifest_path.write_text(json.dumps(manifest))
    bundle["dataset_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    bundle["samples_sha256"] = hashlib.sha256((data / "samples.pt").read_bytes()).hexdigest()
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(bundle))
    checkpoint = tmp_path / "initial"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(json.dumps({
        "source_checkpoint": "fake", "normalization": {},
    }))
    controller = SimpleNamespace(policy=policy, config=policy.model.entity_config)
    def controller_factory(*args, **kwargs):
        assert kwargs["full_state_checkpoint"]
        return controller
    monkeypatch.setattr(fitter, "EntityPolicy", controller_factory)
    monkeypatch.setattr(fitter, "trainable_parameters", lambda *args: None)
    output = tmp_path / "fit"
    fitter.fit(SimpleNamespace(
        steps=1, batch_size=2, learning_rate=.001, output=output, data=data,
        checkpoint=checkpoint, device="cpu", full_state_checkpoint=True,
        trainable_scope="action-expert-grasp",
        grasp_labels=labels, grasp_loss_weight=.2, seed=10,
    ))
    log = json.loads((output / "sft_training.json").read_text())
    assert log["status"] == "completed" and not log["run_config"]["action_only"]
    row = log["steps"][0]
    assert row["loss"] == pytest.approx(row["action_loss"] + .2 * row["current_grasp_loss"])
    assert json.loads((output / "checkpoint/entity_config.json").read_text())["current_grasp_prediction"]
    state = torch.load(output / "training_state.pt", weights_only=False)
    assert "model.current_grasp_head.weight" in state["model"]


@pytest.mark.parametrize("enabled", [False, True])
def test_full_state_checkpoint_needs_configs_not_pretrained_weights(tmp_path, monkeypatch, enabled):
    path = tmp_path / "checkpoint"
    policy, _ = grasp_policy()
    if enabled:
        policy.model.enable_current_grasp_prediction()
    modeling.save_entity_checkpoint(path, policy, policy.model.entity_config, "fake", {})
    config = SimpleNamespace(load_vlm_weights=True, device="cuda")
    monkeypatch.setattr(modeling.SmolVLAConfig, "from_pretrained", lambda _: config)

    class ConfigOnlyPolicy:
        def __new__(cls, requested):
            assert requested is config and not requested.load_vlm_weights
            assert requested.device == "cpu"
            return grasp_policy()[0]

        @classmethod
        def from_pretrained(cls, *_):
            pytest.fail("Full-state load must not read pretrained weights")

    monkeypatch.setattr(modeling, "SmolVLAPolicy", ConfigOnlyPolicy)
    monkeypatch.setattr(modeling, "attach_entity_modules",
                        lambda restored, config: restored.model.init_entity_modules(config))
    restored, _, _ = modeling.load_entity_checkpoint(path, "cpu", full_state=True)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    state = torch.load(path / "model.pt", weights_only=True)
    del state[next(iter(state))]
    torch.save(state, path / "model.pt")
    with pytest.raises(ValueError, match="every model tensor"):
        modeling.load_entity_checkpoint(path, "cpu", full_state=True)


def test_default_partial_checkpoint_keeps_pretrained_missing_weights(tmp_path, monkeypatch):
    policy, _ = grasp_policy()
    path = tmp_path / "checkpoint"
    modeling.save_entity_checkpoint(path, policy, policy.model.entity_config, "fake", {})
    state = torch.load(path / "model.pt", weights_only=True)
    missing = next(name for name in state if "vlm_with_expert" in name)
    del state[missing]
    torch.save(state, path / "model.pt")
    pretrained, _ = grasp_policy()
    with torch.no_grad():
        pretrained.state_dict()[missing].fill_(.75)
    monkeypatch.setattr(modeling.SmolVLAPolicy, "from_pretrained", lambda _: pretrained)
    monkeypatch.setattr(modeling, "attach_entity_modules",
                        lambda restored, config: restored.model.init_entity_modules(config))
    restored, _, _ = modeling.load_entity_checkpoint(path, "cpu")
    torch.testing.assert_close(restored.state_dict()[missing],
                               torch.full_like(restored.state_dict()[missing], .75))
