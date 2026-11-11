from argparse import Namespace
from copy import deepcopy
import json
import hashlib
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

import pytest

from so101.train_trajectory_rl import (
    DecisionSample,
    Trajectory,
    curriculum_key,
    curriculum_maximum,
    CorpusEpisode,
    initialize_mastery_curriculum,
    mastery_assistance,
    record_mastery_group,
    record_target_success_group,
    resolve_mastery_check,
    recover_zero_success_group,
    replay_gold_prefix,
    assign_group_rewards,
    group_advantages,
    completion_values,
    load_auxiliary_view,
    record_auxiliary_admission,
    historical_auxiliary_training,
    initialize_solved_problems,
    record_solved_problem,
    sample_solved_problem,
    policy_optimizer_step,
    restore_resume_state,
    save_resume_state,
    trainable_parameters,
    trajectory_backward,
    trajectory_metrics,
    train,
)
from so101.control_model.flow_rl import (
    FlowTransition,
    clipped_policy_loss,
    finish_velocity_backward,
    gaussian_log_prob,
    reparameterize_flow_path,
    sample_flow_path,
    transition_distribution,
)
from lerobot.policies.common.flow_matching import euler_integrate
from transformers.cache_utils import DynamicCache
from so101.control_model.modeling_entity_smolvla import (
    EntityModelConfig, EntityVLAFlowMatching, FrozenObservation, load_entity_checkpoint,
    save_entity_checkpoint,
)


def test_native_initial_curriculum_preserves_validation_and_resumes_without_legacy_fallback(
        tmp_path, monkeypatch):
    import so101.train_trajectory_rl as trainer

    pool, references, specification = (tmp_path / name for name in ("pool", "references", "curriculum"))
    pool.write_text("pool")
    references.write_text("references")
    native = CorpusEpisode(tmp_path / "native-view", {"task": "view"})
    specification.write_text(json.dumps({"view": [{"source": str(native.source), "assistance": .5}]}))
    legacy = {task: [CorpusEpisode(tmp_path / task, {"task": task})] for task in trainer.TASKS}
    heldout = CorpusEpisode(tmp_path / "heldout", {"task": "view", "scene_family_id": "heldout-family"})
    all_episodes = {**legacy, "view": [*legacy["view"], heldout]}
    monkeypatch.setattr(trainer, "load_corpus", lambda _: all_episodes)
    monkeypatch.setattr(trainer, "split_corpus", lambda *_: (legacy, [heldout]))
    monkeypatch.setattr(trainer, "expert_bounds", lambda _: {
        task: {"optimal_steps": 10., "jerk_p10": 1.} for task in trainer.TASKS
    })
    monkeypatch.setattr(trainer, "load_native_pool", lambda path, excluded: {"view": [native]})
    monkeypatch.setattr(trainer, "attach_native_references",
                        lambda *_, **__: native.metadata.update(native_reference={"qualified": True}))
    captured = []
    with monkeypatch.context() as context:
        context.setattr(sys, "argv", [
            "train", "--checkpoint", str(tmp_path / "base"), "--collection", str(tmp_path / "collection"),
            "--output", str(tmp_path / "run"), "--device", "cpu",
            "--initial-native-pool", str(pool), "--initial-native-references", str(references),
            "--initial-curriculum", str(specification), "--training-tasks", "view",
            "--jerk-regularizer-weight", "0", "--action-boundary-weight", "0",
        ])
        context.setattr(trainer, "train", captured.append)
        trainer.main()
    args = captured[0]

    class IntakeResolved(Exception):
        pass

    def controller(*_):
        raise IntakeResolved

    monkeypatch.setattr(trainer, "EntityPolicy", controller)
    with pytest.raises(IntakeResolved):
        trainer.train(args)
    split = json.loads((args.output / "data_split.json").read_text())
    assert split["training"] == {task: [str(native.source)] if task == "view" else []
                                 for task in trainer.TASKS}
    assert split["validation"] == [str(heldout.source)]
    args.resume = True
    with pytest.raises(IntakeResolved):
        trainer.train(args)
    assert json.loads((args.output / "data_split.json").read_text()) == split
    references.write_text("changed")
    with pytest.raises(ValueError, match="native inputs or training tasks changed"):
        trainer.train(args)
    args.auxiliary_view_manifest = tmp_path / "auxiliary"
    with pytest.raises(ValueError, match="cannot append auxiliary legacy"):
        trainer.train(args)


def test_denoising_backward_does_not_reuse_cropped_suffix_graph() -> None:
    parameter = torch.nn.Parameter(torch.tensor(.4))
    cache = DynamicCache()
    prefix = torch.ones(1, 1, 1, 1)

    def prefix_forward(**kwargs):
        cache.update(prefix, prefix, 0)
        return None, cache

    def denoise(pad, past, latent, timestep):
        suffix = (latent * parameter).reshape(1, 1, 1, 1)
        keys, _ = past.update(suffix, suffix, 0)
        result = keys.square().sum().expand_as(latent)
        past.crop(1)
        return result

    model = SimpleNamespace(
        _prefix_with_entities=lambda: (
            prefix, torch.ones(1, 1, dtype=torch.bool),
            torch.zeros(1, 1, dtype=torch.bool), None,
        ),
        vlm_with_expert=SimpleNamespace(forward=prefix_forward),
        config=SimpleNamespace(use_cache=True),
        _denoise_step_entity=denoise,
    )
    velocity = EntityVLAFlowMatching._rl_velocity_function(model, ())
    for _ in range(3):
        velocity(torch.ones(1, 1, 1), torch.ones(1)).sum().backward()
    torch.testing.assert_close(parameter.grad, torch.tensor(2.4))
    assert not cache.layers[0].keys.requires_grad
    _, path = sample_flow_path(velocity, torch.ones(1, 1, 1), 10, .1, record_noise=True)
    parameter.grad = None
    reparameterize_flow_path(velocity, path, torch.device("cpu"), checkpoint_velocity=False).sum().backward()
    expected = parameter.grad.clone()
    parameter.grad = None
    reparameterize_flow_path(velocity, path, torch.device("cpu")).sum().backward()
    torch.testing.assert_close(parameter.grad, expected)
    assert cache.get_seq_length() == 1
    assert not cache.layers[0].keys.requires_grad


@pytest.fixture
def full_model_replay_policy():
    class Prefix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.7))

        def forward(self, *, inputs_embeds, **kwargs):
            keys = (inputs_embeds[0] * self.weight)[:, None]
            cache = DynamicCache()
            cache.update(keys, keys, 0)
            return None, cache

    class Model(torch.nn.Module):
        _rl_velocity_function = EntityVLAFlowMatching._rl_velocity_function
        denoising_log_probs_entity = EntityVLAFlowMatching.denoising_log_probs_entity
        replay_prefix_batch = EntityVLAFlowMatching.replay_prefix_batch

        def __init__(self):
            super().__init__()
            self.vision = torch.nn.Parameter(torch.tensor(.3))
            self.history_motion_proj = torch.nn.Parameter(torch.tensor(.2))
            self.action_out_proj = torch.nn.Parameter(torch.tensor(.5))
            self.vlm_with_expert = Prefix()
            self.config = SimpleNamespace(use_cache=True)
            self.batch_sizes = []

        def _prefix_with_entities(self, image, history, unused, mask, state):
            self.batch_sizes.append(len(image))
            value = (image * self.vision + history * self.history_motion_proj)[:, None, :]
            return value, mask, torch.zeros_like(mask), None

        def _denoise_step_entity(self, pad, cache, latent, timestep):
            return (cache.layers[0].keys.sum(dim=(1, 2, 3))[:, None, None]
                    * self.action_out_proj + latent * self.action_out_proj)

    policy = torch.nn.Module()
    policy.model = Model().bfloat16()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


@pytest.mark.parametrize("objective", ["independent", "pathwise", "likelihood"])
def test_full_model_replay_reuses_prefix_and_preserves_full_gradients(objective, full_model_replay_policy):
    policy = full_model_replay_policy
    parameters = trainable_parameters(SimpleNamespace(policy=policy))
    assert all(parameter.requires_grad and parameter.dtype == torch.float32 for parameter in parameters)
    assert {id(parameter) for parameter in parameters} == {id(parameter) for parameter in policy.parameters()}
    batch = (torch.tensor([[1., 2.], [3., 4.]]), torch.ones(2, 2),
             torch.zeros(2, 1), torch.ones(2, 1, dtype=torch.bool), torch.ones(2, 1))
    captured = []
    with torch.no_grad():
        velocity = policy.model._rl_velocity_function(
            batch, capture=lambda pad, cache: captured.append(FrozenObservation.from_row(pad, cache, 1, batch)),
        )
        sampled = velocity(torch.ones(2, 1, 1), torch.ones(2))
    row_inputs = tuple(value[1:2] for value in batch)
    replay = policy.model._rl_velocity_function(row_inputs, captured[0])
    reference = deepcopy(policy.model)

    def reference_velocity(latent, timestep):
        prefix, pad, _, _ = reference._prefix_with_entities(*batch)
        _, cache = reference.vlm_with_expert.forward(inputs_embeds=[prefix, None])
        return reference._denoise_step_entity(
            pad[1:2], FrozenObservation.copy_cache(cache, "cpu", 1, detach=False), latent, timestep,
        )

    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    before = [parameter.detach().clone() for parameter in parameters]
    if objective in ("pathwise", "likelihood"):
        with torch.no_grad():
            _, path = sample_flow_path(
                reference_velocity, torch.ones(1, 1, 1), 10, .1, record_noise=True,
            )
        if objective == "pathwise":
            expected = reparameterize_flow_path(
                reference_velocity, path, torch.device("cpu"), checkpoint_velocity=False,
            )
            actual = reparameterize_flow_path(replay, path, torch.device("cpu"))
            torch.testing.assert_close(actual, expected)
            actual.sum().backward()
            expected.sum().backward()
        else:
            for value, transition in zip(
                policy.model.denoising_log_probs_entity(row_inputs, path, captured[0]),
                path, strict=True,
            ):
                timestep = torch.full((1,), transition.time)
                mean, std = transition_distribution(
                    transition.latent, reference_velocity(transition.latent, timestep),
                    transition.time, transition.step_size, transition.noise_scale,
                )
                expected = gaussian_log_prob(
                    transition.next_latent, mean, std, transition.active_shape,
                )
                torch.testing.assert_close(value, expected)
                (-value.mean()).backward()
                (-expected.mean()).backward()
    else:
        for _ in range(3):
            value = replay(torch.ones(1, 1, 1), torch.ones(1))
            torch.testing.assert_close(value, sampled[1:2], rtol=0., atol=1e-4)
            value.sum().backward()
            reference_velocity(torch.ones(1, 1, 1), torch.ones(1)).sum().backward()
    if objective != "likelihood":
        finish_velocity_backward(replay)
    assert policy.model.batch_sizes == [2] * 2
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in parameters)
    for actual, expected in zip(parameters, reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)
    optimizer.step()
    assert all(not torch.equal(old, parameter) for old, parameter in zip(before, parameters, strict=True))
    captured[0].cache.layers[0].keys += .1
    replay(torch.ones(1, 1, 1), torch.ones(1)).sum().backward()
    with pytest.raises(AssertionError):
        finish_velocity_backward(replay)


@pytest.mark.parametrize("objective", ["independent", "pathwise", "likelihood"])
def test_shared_prefix_backward_combines_rows_without_mixing_original_batches(full_model_replay_policy, objective):
    policy = full_model_replay_policy
    trainable_parameters(SimpleNamespace(policy=policy))
    reference = deepcopy(policy)
    batch = (torch.tensor([[1., 2.], [3., 4.]]), torch.ones(2, 2),
             torch.zeros(2, 1), torch.ones(2, 1, dtype=torch.bool), torch.ones(2, 1))
    other_batch = tuple(value.clone() for value in batch)
    captured = []
    with torch.no_grad():
        for original in (batch, other_batch):
            policy.model._rl_velocity_function(
                original, capture=lambda pad, cache: captured.extend(
                    FrozenObservation.from_row(pad, cache, row, original) for row in (0, 1)
                ),
            )
    observations = [captured[1], captured[0], captured[1], captured[3]]
    with policy.model.replay_prefix_batch():
        for observation, weight in zip(observations, [.7, -.2, .3, .5], strict=True):
            inputs = tuple(value[observation.row:observation.row + 1] for value in observation.replay_inputs)
            actual_velocity = policy.model._rl_velocity_function(inputs, observation)
            expected_velocity = reference.model._rl_velocity_function(inputs, observation)
            if objective != "independent":
                with torch.no_grad():
                    _, path = sample_flow_path(expected_velocity, torch.ones(1, 1, 1), 10, .1,
                                               record_noise=True)
            if objective == "likelihood":
                for actual, expected in zip(
                    policy.model.denoising_log_probs_entity(inputs, path, observation),
                    reference.model.denoising_log_probs_entity(inputs, path, observation),
                    strict=True,
                ):
                    (weight * actual.mean()).backward()
                    (weight * expected.mean()).backward()
                continue
            if objective == "pathwise":
                actual = reparameterize_flow_path(actual_velocity, path, torch.device("cpu"))
                expected = reparameterize_flow_path(expected_velocity, path, torch.device("cpu"))
            else:
                actual = actual_velocity(torch.ones(1, 1, 1), torch.ones(1))
                expected = expected_velocity(torch.ones(1, 1, 1), torch.ones(1))
            (weight * actual.square().sum()).backward()
            (weight * expected.square().sum()).backward()
            finish_velocity_backward(actual_velocity)
            finish_velocity_backward(expected_velocity)
    assert policy.model.batch_sizes == [2] * 4
    assert reference.model.batch_sizes == [2] * 4
    for actual, expected in zip(policy.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)
    assert policy.model._rl_pending_prefix is None
    captured[1].cache.layers[0].keys += .1
    with pytest.raises(AssertionError), policy.model.replay_prefix_batch():
        velocity = policy.model._rl_velocity_function(batch, captured[1])
        velocity(torch.ones(1, 1, 1), torch.ones(1)).sum().backward()
        finish_velocity_backward(velocity)
    assert policy.model._rl_pending_prefix is None


def args() -> Namespace:
    return Namespace(
        curriculum_decrement=.05,
        curriculum_mastery_groups=3,
        curriculum_min=0.,
        obstacle_curriculum_max=.8,
        push_curriculum_max=.95,
        place_curriculum_max=.9,
        grasp_curriculum_max=.95,
        view_curriculum_max=.95,
    )


def test_curriculum_selects_task_and_obstacle_levels() -> None:
    config = args()
    assert curriculum_key("push", False) == "push"
    assert curriculum_key("grasp", True) == "obstacle_grasp"
    assert curriculum_maximum("push", False, config) == .95
    assert curriculum_maximum("place", False, config) == .9
    assert curriculum_maximum("grasp", False, config) == .95
    assert curriculum_maximum("view", False, config) == .95
    assert curriculum_maximum("grasp", True, config) == .8


@pytest.mark.parametrize("successes,expected", [(3, .9), (2, .95), (1, 1.)])
def test_target_success_uses_mixed_groups_and_waits_for_window(successes, expected):
    state = initialize_mastery_curriculum({}, args())
    mastery_assistance(state, "push", .95)
    for step in range(3):
        assert record_target_success_group(state, "push", successes, 4, 1., step) is None
    event = record_target_success_group(state, "push", successes, 4, 1., 3)
    assert state["tasks"]["push"]["level"] == expected
    assert (event is None) == (successes == 2)
    if event:
        assert event["total"] == 16
        assert state["tasks"]["push"]["success_window"] == []


def test_target_success_bounds_and_resume_window():
    state = initialize_mastery_curriculum({}, args())
    mastery_assistance(state, "view", 0.)
    for step in range(4):
        assert record_target_success_group(state, "view", 4, 4, 0., step) is None
    mastery_assistance(state, "push", .95)
    record_target_success_group(state, "push", 0, 4, .95, 1)
    resumed = json.loads(json.dumps(state))
    for step in range(2, 5):
        assert record_target_success_group(resumed, "push", 0, 4, .95, step) is None
    assert resumed["tasks"]["push"]["level"] == .95


def test_reference_free_completion_rewards_shorter_successes_without_expert_scores():
    rows = [
        SimpleNamespace(success=True, steps=20, completion_efficiency=0., smoothness=0.),
        SimpleNamespace(success=True, steps=40, completion_efficiency=0., smoothness=0.),
        SimpleNamespace(success=False, steps=1, completion_efficiency=1., smoothness=0.),
    ]
    np.testing.assert_allclose(completion_values(rows, True), [.5, 0., 0.])
    advantages, components = group_advantages(rows, 1., .5, 0., True)
    assert components["completion"][0] > 0 > components["completion"][1]
    assert components["completion"][2] == 0
    assign_group_rewards(rows, 1., .5, 0., True)
    assert rows[0].reward > rows[1].reward > rows[2].reward
    assert np.isfinite(advantages).all()


def test_auxiliary_transition_requires_hash_and_preserves_history():
    old = {"manifest": "old.json", "sha256": "old"}
    new = {"manifest": "new.json", "sha256": "new"}
    log = {"auxiliary_view": old, "auxiliary_view_validation": [{"successes": 4}]}
    with pytest.raises(ValueError, match="exact previous"):
        record_auxiliary_admission(log, new, None, 253)
    assert log["auxiliary_view"] == old
    record_auxiliary_admission(log, new, "old", 253)
    assert log["auxiliary_view_transitions"] == [
        {"step": 253, "previous": old, "current": new},
    ]
    assert log["auxiliary_view_validation"] == [{"successes": 4}]
    record_auxiliary_admission(log, new, "old", 254)
    assert len(log["auxiliary_view_transitions"]) == 1
    with pytest.raises(ValueError, match="existing admission"):
        record_auxiliary_admission({}, new, "old", 0)


def test_historical_auxiliary_preserves_solved_sources_and_split_guards(monkeypatch, tmp_path):
    old_train = CorpusEpisode(tmp_path / "old_train", {"scene_family_id": "train"})
    old_dev = CorpusEpisode(tmp_path / "old_dev", {"scene_family_id": "dev"})
    splits = {"train": [old_train], "development": [old_dev]}
    monkeypatch.setattr("so101.train_trajectory_rl.load_auxiliary_view",
                        lambda path, validation: (splits, "bound"))
    admission = {"manifest": "old.json", "sha256": "bound"}
    log = {"auxiliary_view": admission,
           "auxiliary_view_transitions": [{"previous": admission}]}
    current = {"train": [], "development": []}
    assert historical_auxiliary_training(log, [], current) == [old_train]
    with pytest.raises(ValueError, match="historical split"):
        historical_auxiliary_training(log, [], {"train": [old_dev], "development": []})
    admission["sha256"] = "changed"
    with pytest.raises(ValueError, match="manifest changed"):
        historical_auxiliary_training(log, [], current)


def test_auxiliary_view_verifies_hashes_and_excludes_heldout_families(tmp_path):
    entries = []
    for seed, split in [(1, "train"), (2, "development")]:
        source = tmp_path / str(seed)
        source.mkdir()
        family = hashlib.sha256(json.dumps([seed, 1]).encode()).hexdigest()
        metadata = {
            "task": "view", "outcome": "success", "initial": {"seed": seed},
            "reference_replay": {"success": True, "maximum_qpos_error": 1e-8},
            "scene_family_id": family, "split": split,
        }
        (source / "episode.json").write_text(json.dumps(metadata))
        (source / "frames.jsonl").write_text("{}\n")
        entries.append({
            "path": str(source), "split": split, "scene_family_id": family,
            "sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                       for name in ("episode.json", "frames.jsonl")},
        })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "status": "complete",
                               "kind": "harder-view-references", "episodes": entries}))
    admitted, digest = load_auxiliary_view(path, [])
    assert len(admitted["train"]) == len(admitted["development"]) == 1
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    heldout = CorpusEpisode(tmp_path / "heldout", {"initial": {"seed": 1}})
    with pytest.raises(ValueError, match="held-out"):
        load_auxiliary_view(path, [heldout])
    (tmp_path / "1/frames.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_auxiliary_view(path, [])


def test_assistance_only_decreases_after_perfect_mastery_window() -> None:
    state = initialize_mastery_curriculum({}, args())
    assert mastery_assistance(state, "push", .95) == .95
    assert not record_mastery_group(state, "push", 8, 8)
    assert not record_mastery_group(state, "push", 7, 8)
    assert not record_mastery_group(state, "push", 8, 8)
    assert not record_mastery_group(state, "push", 8, 8)
    assert record_mastery_group(state, "push", 8, 8)
    event = resolve_mastery_check(state, "push", 10, 10, 3, 8)
    assert event == {
        "step": 3,
        "task": "push",
        "previous": .95,
        "current": .9,
        "mastery_groups": 3,
        "mastery_rollouts": 24,
    }
    assert mastery_assistance(state, "push", .95) == .9


def test_curriculum_state_is_resume_safe() -> None:
    log = {}
    state = initialize_mastery_curriculum(log, args())
    mastery_assistance(state, "place", .9)
    record_mastery_group(state, "place", 8, 8)
    resumed = initialize_mastery_curriculum(log, args())
    assert resumed["tasks"]["place"]["perfect_group_streak"] == 1

    changed = args()
    changed.curriculum_decrement = .1
    with pytest.raises(ValueError, match="decrement"):
        initialize_mastery_curriculum(log, changed)


def test_failed_task_wide_check_does_not_reduce_assistance() -> None:
    state = initialize_mastery_curriculum({}, args())
    mastery_assistance(state, "push", .95)
    for _ in range(3):
        ready = record_mastery_group(state, "push", 8, 8)
    assert ready
    assert resolve_mastery_check(state, "push", 9, 10, 2, 8) is None
    assert mastery_assistance(state, "push", .95) == .95
    assert state["tasks"]["push"]["perfect_group_streak"] == 0


def test_zero_success_recovery_is_bounded_and_resets_stale_mastery():
    state = initialize_mastery_curriculum({}, args())
    mastery_assistance(state, "view", .95)
    state["tasks"]["view"].update(level=.93, perfect_group_streak=2)
    event = recover_zero_success_group(state, "view", 0, .95, 8)
    assert event["previous"] == .93 and event["current"] == .95
    assert event["kind"] == "zero_success_recovery"
    assert state["tasks"]["view"]["perfect_group_streak"] == 0
    assert recover_zero_success_group(state, "view", 0, .95, 9) is None
    assert mastery_assistance(state, "view", .95) == .95


def test_recovery_does_not_change_groups_with_task_reward_signal():
    state = initialize_mastery_curriculum({}, args())
    mastery_assistance(state, "view", .95)
    state["tasks"]["view"]["level"] = 0.
    assert recover_zero_success_group(state, "view", 1, .95, 8) is None
    assert mastery_assistance(state, "view", .95) == 0.
    event = recover_zero_success_group(state, "view", 0, .95, 8)
    assert event["current"] == .05


@pytest.mark.parametrize("task", ["grasp", "view", "push", "place"])
def test_gold_prefix_assists_all_tasks(tmp_path, task):
    rows = [{"action": [index]} for index in range(10)]
    (tmp_path / "frames.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    executed = []
    sim = Namespace(
        position=lambda target: [0, 0, len(executed) * .01],
        bounded_policy_action=lambda action: (action, None),
        step=lambda goal, phase: executed.append(goal),
    )
    episode = CorpusEpisode(tmp_path, {"task": task})
    steps, lift = replay_gold_prefix(sim, episode, "target", .5)
    assert steps == 5
    assert executed == [[0], [1], [2], [3], [4]]
    assert lift == pytest.approx(.05)
    assert replay_gold_prefix(sim, episode, "target", 0) == (0, 0.)
    assert len(executed) == 5


def test_atomic_resume_state_restores_model_optimizer_log_and_rng(tmp_path):
    model = torch.nn.Linear(2, 1)
    controller = Namespace(policy=model, device="cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    rng = np.random.default_rng(42)
    log = {"steps": [{"step": 7}], "assistance_curriculum": {"tasks": {"grasp": {"level": .9}}}}
    save_resume_state(tmp_path, controller, optimizer, rng, log)
    expected = (rng.random(), random.random(), np.random.random(), torch.rand(1))
    expected_parameters = [parameter.clone() for parameter in model.parameters()]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    state = torch.load(tmp_path / "latest_state.pt", weights_only=False)
    restored = restore_resume_state(state, controller, optimizer, rng, 2e-4)
    assert restored == log
    for parameter, saved in zip(model.parameters(), expected_parameters, strict=True):
        torch.testing.assert_close(parameter, saved)
    assert optimizer.param_groups[0]["lr"] == 2e-4
    assert optimizer.state[next(model.parameters())]["step"] == 1
    assert rng.random() == expected[0]
    assert random.random() == expected[1]
    assert np.random.random() == expected[2]
    torch.testing.assert_close(torch.rand(1), expected[3])
    assert not (tmp_path / "latest_state.pt.tmp").exists()
    state["step"] = 6
    with pytest.raises(ValueError, match="steps do not match"):
        restore_resume_state(state, controller, optimizer, rng, 1e-4)


def test_zero_noise_flow_path_matches_existing_ode_sampler():
    torch.manual_seed(12)
    initial = torch.randn(2, 5, 6)
    velocity = lambda value, time: value * .2 + time[:, None, None]
    actions, path = sample_flow_path(velocity, initial, 10, 0.)
    torch.testing.assert_close(
        actions, euler_integrate(velocity, initial, 10), rtol=0, atol=0
    )
    assert path == []


def test_recorded_path_has_exact_joint_gaussian_likelihoods():
    torch.manual_seed(13)
    velocity = lambda value, time: value * .3
    _, path = sample_flow_path(velocity, torch.randn(2, 4, 6), 10, .1)
    assert len(path) == 10
    for transition in path:
        mean, std = transition_distribution(
            transition.latent, velocity(transition.latent, None),
            transition.time, transition.step_size, transition.noise_scale,
        )
        expected = torch.distributions.Normal(mean, std).log_prob(
            transition.next_latent
        ).flatten(start_dim=1).sum(dim=1)
        torch.testing.assert_close(transition.old_log_prob, expected)
        assert std > 0
        assert not transition.next_latent.requires_grad


def test_reverse_sde_preserves_gaussian_rectified_flow_marginal():
    torch.manual_seed(14)
    initial = torch.randn(32768, 1)

    def velocity(value, time):
        variance = (1 - time).square() + time.square()
        return ((2 * time - 1) / variance)[:, None] * value

    actions, _ = sample_flow_path(velocity, initial, 100, .7)
    assert abs(float(actions.mean())) < .03
    assert abs(float(actions.var()) - 1.) < .07


def test_likelihood_gradient_matches_expected_squared_jerk_gradient():
    torch.manual_seed(15)
    mean = torch.tensor([[0., .1, .4, .6]], requires_grad=True)
    third_difference = torch.tensor([-1., 3., -3., 1.])
    samples = mean.detach() + .4 * torch.randn(262144, 4)
    reward = -(samples @ third_difference).square()
    baseline = -(
        (mean.detach() @ third_difference).square()
        + .4 ** 2 * third_difference.square().sum()
    )
    loss = -(
        (reward - baseline).detach()
        * gaussian_log_prob(samples, mean, .4)
    ).mean()
    loss.backward()
    expected = 2 * (mean.detach() @ third_difference)[:, None] * third_difference
    torch.testing.assert_close(mean.grad, expected, rtol=.08, atol=.04)
    updated = mean.detach() - .01 * mean.grad
    assert (updated @ third_difference).square() < (
        mean.detach() @ third_difference
    ).square()


@pytest.mark.parametrize("ratio,advantage", [(1.5, 1.), (.5, -1.)])
def test_policy_clipping_blocks_outward_updates(ratio, advantage):
    log_prob = torch.tensor(np.log(ratio), requires_grad=True)
    loss = clipped_policy_loss(log_prob, torch.tensor(0.), advantage, .2)
    loss.backward()
    assert log_prob.grad == 0


def test_recorded_policy_gradient_sums_decisions_including_invalid_proposals():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mean = torch.nn.Parameter(torch.zeros(1, 1))

        def denoising_log_probs_entity(self, inputs, path):
            for transition in path:
                yield gaussian_log_prob(transition.next_latent, self.mean, 1.)

    policy = torch.nn.Module()
    policy.model = Model()
    transition = FlowTransition(
        torch.zeros(1, 1), torch.ones(1, 1),
        gaussian_log_prob(torch.ones(1, 1), torch.zeros(1, 1), 1.),
        1., .1, .1,
    )
    decision = DecisionSample(
        (torch.zeros(1),), torch.zeros(1, 6), 0, [transition]
    )
    trajectory = Trajectory("view", False, False, 0, 0., 0., 0., 0., [decision, decision])
    trajectory_backward(Namespace(policy=policy, device="cpu"), trajectory, .5)
    torch.testing.assert_close(policy.model.mean.grad, torch.tensor([[-1.]]))


def test_final_transition_marginalizes_only_unused_output_noise():
    torch.manual_seed(19)
    initial = torch.randn(1, 4, 6)
    torch.manual_seed(20)
    full_actions, full_path = sample_flow_path(
        lambda value, time: .2 * value, initial, 3, .1,
    )
    torch.manual_seed(20)
    actions, path = sample_flow_path(
        lambda value, time: .2 * value, initial, 3, .1,
        final_action_shape=(2, 3),
    )
    torch.testing.assert_close(actions, full_actions, rtol=0, atol=0)
    for full, masked in zip(full_path, path, strict=True):
        torch.testing.assert_close(full.next_latent, masked.next_latent, rtol=0, atol=0)
    assert [item.active_shape for item in path] == [None, None, (2, 3)]
    for tr in path:
        old_mean, std = transition_distribution(
            tr.latent, .2 * tr.latent, tr.time, tr.step_size, tr.noise_scale
        )
        mean = old_mean.detach().clone().requires_grad_()
        log_prob = gaussian_log_prob(tr.next_latent, mean, .1, tr.active_shape)
        log_prob.sum().backward()
        if tr.active_shape is not None:
            assert mean.grad[:, 2:].count_nonzero() == 0
            assert mean.grad[:, :, 3:].count_nonzero() == 0
            assert mean.grad[:, :2, :3].count_nonzero() > 0
        else:
            assert mean.grad[:, 2:, 3:].count_nonzero() > 0
        torch.testing.assert_close(
            tr.old_log_prob,
            gaussian_log_prob(tr.next_latent, old_mean, std, tr.active_shape),
        )


def test_policy_scale_is_outside_joint_probability_ratio():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mean = torch.nn.Parameter(torch.zeros(1, 1, 2))

        def denoising_log_probs_entity(self, inputs, path):
            for tr in path:
                yield gaussian_log_prob(tr.next_latent, self.mean, 1.)

    policy = torch.nn.Module()
    policy.model = Model()
    sample = torch.ones(1, 1, 2)
    tr = FlowTransition(
        torch.zeros_like(sample), sample,
        gaussian_log_prob(sample, torch.zeros_like(sample), 1.), 1., .1, .1,
    )
    decision = DecisionSample((torch.zeros(1),), sample[0], 1, [tr, tr])
    trajectory = Trajectory("view", True, False, 1, 0., 0., 0., 0., [decision])
    trajectory_backward(Namespace(policy=policy, device="cpu"), trajectory, 1.)
    torch.testing.assert_close(policy.model.mean.grad, torch.full_like(sample, -.5))
    joint = torch.tensor([.15, .15], requires_grad=True)
    clipped_policy_loss(joint.sum(), torch.tensor(0.), 1., .2).backward()
    assert joint.grad.count_nonzero() == 0


def test_checkpoint_preserves_expert_precision_before_loading_weights(tmp_path, monkeypatch):
    def policy(dtype):
        result = torch.nn.Module()
        result.model = torch.nn.Module()
        result.model.vlm_with_expert = torch.nn.Module()
        result.model.vlm_with_expert.lm_expert = torch.nn.Linear(1, 1, dtype=dtype)
        return result

    original = policy(torch.float32)
    with torch.no_grad():
        original.model.vlm_with_expert.lm_expert.weight.fill_(1.003123)
    path = tmp_path / "checkpoint"
    save_entity_checkpoint(path, original, EntityModelConfig(), "fake", {})
    monkeypatch.setattr(
        'so101.control_model.modeling_entity_smolvla.SmolVLAPolicy.from_pretrained',
        lambda source: policy(torch.bfloat16),
    )
    monkeypatch.setattr(
        'so101.control_model.modeling_entity_smolvla.attach_entity_modules',
        lambda model, config: None,
    )
    restored, _, _ = load_entity_checkpoint(path, "cpu")
    expert = restored.model.vlm_with_expert.lm_expert
    assert expert.weight.dtype == torch.float32
    torch.testing.assert_close(
        expert.weight, original.model.vlm_with_expert.lm_expert.weight, rtol=0, atol=0
    )
    metadata = json.loads((path / "metadata.json").read_text())
    del metadata["action_expert_dtype"]
    (path / "metadata.json").write_text(json.dumps(metadata))
    legacy, _, _ = load_entity_checkpoint(path, "cpu")
    assert legacy.model.vlm_with_expert.lm_expert.weight.dtype == torch.bfloat16


def test_reward_only_update_averages_policy_gradient_without_imitation():
    parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
    before = parameter.detach().clone()
    optimizer = torch.optim.AdamW([parameter], lr=.1, weight_decay=.01)
    parameter.grad = torch.tensor([6., 8.])
    metrics = policy_optimizer_step(
        optimizer, [parameter], 2, 1.,
    )
    assert metrics["gradient_norm"] == pytest.approx(5.)
    assert metrics["policy_gradient_norm"] == pytest.approx(5.)
    assert metrics["replay_gradient_norm"] == 0
    assert metrics["optimizer_update_applied"]
    reference = torch.nn.Parameter(before.clone())
    other = torch.optim.AdamW(
        [reference], lr=.1, weight_decay=.01
    )
    reference.grad = torch.tensor([.6, .8])
    other.step()
    torch.testing.assert_close(parameter, reference)
    assert optimizer.state[parameter]["step"] == 1


def test_reward_only_all_failure_batch_does_not_apply_adam_momentum():
    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.AdamW([parameter], lr=.1, weight_decay=.01)
    parameter.grad = torch.tensor(.5)
    optimizer.step()
    before = parameter.detach().clone()
    state = deepcopy(optimizer.state_dict())
    parameter.grad = torch.zeros_like(parameter)

    metrics = policy_optimizer_step(
        optimizer, [parameter], 0, 1.,
    )
    assert not metrics["optimizer_update_applied"]
    assert metrics["optimizer_skip_reason"] == "no_accepted_groups"
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    for key, value in optimizer.state_dict()["state"][0].items():
        torch.testing.assert_close(value, state["state"][0][key], rtol=0, atol=0)


def test_absolute_jerk_score_survives_group_assignment_and_drives_advantage():
    def trajectory(jerk):
        actions = [np.full(5, value) for value in (0., 0., 0., jerk)]
        measured, completion, smoothness = trajectory_metrics(
            "view", True, 4, actions, {"optimal_steps": 2., "jerk_p10": 1.},
        )
        return Trajectory(
            "view", True, False, 4, measured, completion, smoothness, 0., [],
        )

    trajectories = [trajectory(jerk) for jerk in (2., 4., 8.)]
    assign_group_rewards(trajectories)
    np.testing.assert_allclose([t.smoothness for t in trajectories], [.5, .25, .125])
    np.testing.assert_allclose([t.reward for t in trajectories], [1.75, 1.5, 1.375])
    advantage, components = group_advantages(trajectories, 1., .5, 1.)
    expected = (np.array([.5, .25, .125]) - np.mean([.5, .25, .125])) / (
        np.std([.5, .25, .125]) + 1e-6
    )
    np.testing.assert_allclose(components["smoothness"], expected, atol=1e-6)
    np.testing.assert_allclose(advantage, expected, atol=1e-6)
    assert not components["success"].any()
    assert not components["completion"].any()
    assign_group_rewards([trajectories[1]], 1., .5, 2.)
    assert trajectories[1].smoothness == .25
    assert trajectories[1].reward == 1.75
    failure = Trajectory("view", False, False, 4, 8., 0., 0., 0., [])
    _, mixed = group_advantages([*trajectories, failure], 1., .5, 1.)
    assert mixed["success"][-1] < 0
    assert all(value > 0 for value in mixed["success"][:-1])
    assert mixed["smoothness"][-1] == 0


def test_validation_retains_matched_problem_noise_and_failure_summaries(tmp_path, monkeypatch):
    import so101.train_trajectory_rl as trainer

    episodes = [CorpusEpisode(tmp_path / task, {"task": task}) for task in trainer.TASKS]
    def rollout(sim, controller, episode, seed, *args, **kwargs):
        result = Trajectory(episode.task, episode.task != "view", False, 10, 2., .5, .1, 0., [])
        if episode.task == "view":
            result.invalid_action_error = "Policy target exceeds a joint boundary by 6 units"
        return result
    monkeypatch.setattr(trainer, "rollout", rollout)
    result = trainer.validation_metrics(None, None, episodes, {task: {} for task in trainer.TASKS},
                                        61000, 300, 16, .1)
    for index, (episode, row) in enumerate(zip(episodes, result["rollouts"], strict=True)):
        assert (row["task"], row["source"], row["seed"]) == (episode.task, str(episode.source), 61000 + index)
    view = next(row for row in result["rollouts"] if row["task"] == "view")
    assert not view["success"] and view["invalid_action_error"]
    assert result["by_task"]["view"]["success_rate"] == 0


def test_solved_bucket_tracks_exact_assistance_and_only_samples_prior_steps(tmp_path):
    episode = CorpusEpisode(tmp_path, {"task": "push"})
    bucket = {}
    record_solved_problem(bucket, episode, .65, 7, 8, 1)
    assert sample_solved_problem(bucket, np.random.default_rng(1), 2) is None
    record_solved_problem(bucket, episode, .65, 8, 8, 2)
    record_solved_problem(bucket, episode, .65, 8, 8, 3)
    record_solved_problem(bucket, episode, .60, 8, 8, 3)
    assert len(bucket) == 2
    assert sample_solved_problem(bucket, np.random.default_rng(1), 2) is None
    sampled = sample_solved_problem(bucket, np.random.default_rng(1), 3)
    assert sampled == {
        "source": str(tmp_path), "task": "push",
        "start_proximity": .65, "solved_at_step": 2,
    }


def test_historical_solved_bucket_excludes_validation_and_partial_success(tmp_path):
    episode = CorpusEpisode(tmp_path / "training", {"task": "grasp"})
    def group(source, successes):
        return {
            "source": str(source), "successes": successes,
            "start_proximity": .8, "trajectories": [{}] * 8,
        }
    log = {"steps": [
        {"step": 1, "groups": [
            group(episode.source, 7), group(tmp_path / "validation", 8),
        ]},
        {"step": 2, "groups": [group(episode.source, 8)]},
    ]}
    bucket = initialize_solved_problems(log, {"grasp": [episode]})
    assert len(bucket) == 1
    assert next(iter(bucket.values()))["solved_at_step"] == 2
    assert initialize_solved_problems(log, {"grasp": [episode]}) is bucket


@pytest.mark.parametrize("initial_validation", [True, False])
@pytest.mark.parametrize("mixed_success_only", [True, False])
@pytest.mark.parametrize("calibrated", [True, False])
def test_training_adds_one_fresh_solved_group_and_persists_bucket(
    tmp_path, monkeypatch, initial_validation, mixed_success_only, calibrated,
):
    import so101.train_trajectory_rl as trainer

    training = {
        task: [CorpusEpisode(tmp_path / task, {"task": task, "target": "wide"})]
        for task in trainer.TASKS
    }
    policy = torch.nn.Linear(1, 1)
    if calibrated:
        if not mixed_success_only:
            pytest.skip("Calibration requires the mixed-success objective")
        policy.model = Namespace(vlm_with_expert=Namespace(lm_expert=torch.nn.Linear(1, 1)))
        policy.config = Namespace(num_steps=1, chunk_size=1, max_action_dim=1)
        monkeypatch.setattr(trainer, "full_model_gradient_probes", lambda policy: {})
    original_parameters = [parameter.detach().clone() for parameter in policy.parameters()]
    controller = Namespace(policy=policy, device="cpu")
    monkeypatch.setattr(trainer, "load_corpus", lambda collection: training)
    monkeypatch.setattr(trainer, "split_corpus", lambda *args: (training, []))
    monkeypatch.setattr(trainer, "expert_bounds", lambda episodes: {
        task: {"optimal_steps": 10., "jerk_p10": 1.} for task in trainer.TASKS
    })
    monkeypatch.setattr(trainer, "EntityPolicy", lambda *args: controller)
    monkeypatch.setattr(trainer, "trainable_parameters", lambda value, scope: list(policy.parameters()))
    monkeypatch.setattr(trainer, "create_simulation", lambda *args, **kwargs: Namespace(close=lambda: None))
    monkeypatch.setattr(trainer, "validation_metrics", lambda *args: {})
    calls = []
    captures = []

    def rollout(sim, controller, episode, seed, horizon, action_steps, bounds,
                store_decisions, start_proximity, noise, capture_path=None):
        calls.append((episode.source, start_proximity, seed))
        if capture_path is not None:
            captures.append(capture_path)
        return Trajectory(
            episode.task, True, False, 10, 2. + seed % 8,
            .2 + (seed % 8) * .1, 1. / (2. + seed % 8), 0., [],
        )

    def backward(controller, trajectory, coefficient, policy_clip):
        (policy(torch.ones(1, 1)).sum() * coefficient).backward()
        return coefficient

    monkeypatch.setattr(trainer, "rollout", rollout)
    monkeypatch.setattr(trainer, "trajectory_backward", backward)
    calibration_calls = {}
    def grouped(sims, controller, episode, seeds, horizon, action_steps, bounds, assistance, *rest):
        attempts = calibration_calls.setdefault(episode.task, [])
        attempts.append(assistance)
        successes = 4 if len(attempts) == 1 else 0 if len(attempts) == 2 else 2
        trajectories = [
            rollout(sim, controller, episode, seed, horizon, action_steps, bounds,
                    True, assistance, .1)
            for sim, seed in zip(sims, seeds, strict=True)
        ]
        for index, trajectory in enumerate(trajectories):
            trajectory.success = index < successes
        return trajectories
    monkeypatch.setattr(trainer, "rollout_group", grouped)
    config = Namespace(
        checkpoint=tmp_path / "base", collection=tmp_path / "corpus",
        output=tmp_path / "run", resume=False, refresh_corpus=False,
        steps=2, group_size=8, groups_per_task=2, max_group_attempts_per_task=8,
        horizon=300, action_steps=16, validation_per_task=1, validation_interval=2,
        initial_validation=initial_validation,
        rl_estimator="flow-mse", denoising_noise_scale=.1, policy_clip=.2,
        success_advantage_weight=1., completion_advantage_weight=.5,
        smoothness_advantage_weight=1., learning_rate=1e-4, gradient_clip=1.,
        seed=61000, device="cpu", model_root=tmp_path, save_final_checkpoint=False,
        curriculum_start_step=0, curriculum_mode="mastery", curriculum_end_step=0,
        curriculum_decrement=.05, curriculum_mastery_groups=100,
        curriculum_min=0., push_curriculum_max=.95, place_curriculum_max=.95,
        grasp_curriculum_max=.95, view_curriculum_max=.95, obstacle_curriculum_max=.95,
        record_rollouts=True, rollout_retention_steps=20,
        mixed_success_only=mixed_success_only,
    )
    if calibrated:
        config.group_size = 4
        config.groups_per_task = 1
        config.calibrate_assistance = config.batch_rollouts = True
        config.rl_estimator = "denoising-grpo"
        config.jerk_penalty_weight = 0.
    train(config)
    state = torch.load(config.output / "latest_state.pt", weights_only=False)
    assert [row["step"] for row in state["log"]["validation"]] == (
        [0, 2] if initial_validation else [2]
    )
    first, second = state["log"]["steps"]
    if calibrated:
        assert first["accepted_groups"] == second["accepted_groups"] == 4
        calibration = state["log"]["assistance_calibration"]
        assert all(row["optimizer_update_applied"] for row in (first, second))
        assert sum(task["all_passed"] + task["all_failed"]
                   for task in calibration["tasks"].values()) == 8
        for attempts in calibration_calls.values():
            assert attempts[0:3] == pytest.approx([.475, .425, .475])
            assert attempts[3] == pytest.approx(.4686)
        assert all(
            group["successes"] == 2
            for step in (first, second) for group in step["groups"]
            if not group["discarded"]
        )
        config.resume = True
        config.steps = 3
        train(config)
        resumed = torch.load(config.output / "latest_state.pt", weights_only=False)
        assert resumed["log"]["steps"][-1]["optimizer_update_applied"]
        assert all(
            resumed["log"]["assistance_calibration"]["tasks"][task]["groups"] == record["groups"] + 1
            for task, record in calibration["tasks"].items()
        )
        return
    if mixed_success_only:
        assert first["accepted_groups"] == second["accepted_groups"] == 0
        assert len(first["groups"]) == 4 * config.max_group_attempts_per_task
        assert len(second["groups"]) == 4 * config.max_group_attempts_per_task + 1
        assert all(
            group["discarded"] and group["discard_reason"] == "all_passed"
            for step in (first, second) for group in step["groups"]
        )
        assert not first["optimizer_update_applied"] and not second["optimizer_update_applied"]
        for before, after in zip(original_parameters, policy.parameters(), strict=True):
            torch.testing.assert_close(before, after)
        assert state["optimizer"]["state"] == {}
        assert len(state["log"]["solved_problems"]) == 4
        config.resume = True
        config.steps = 3
        train(config)
        resumed = torch.load(config.output / "latest_state.pt", weights_only=False)
        assert resumed["step"] == 3 and resumed["optimizer"]["state"] == {}
        assert not resumed["log"]["steps"][-1]["optimizer_update_applied"]
        for before, after in zip(original_parameters, policy.parameters(), strict=True):
            torch.testing.assert_close(before, after)
        return
    assert first["accepted_groups"] == 8
    assert not any(g["sampling_bucket"] == "solved" for g in first["groups"])
    assert second["accepted_groups"] == 9
    solved = [g for g in second["groups"] if g["sampling_bucket"] == "solved"]
    assert len(solved) == 1
    assert solved[0]["start_proximity"] == .95
    assert not np.any(solved[0]["advantage_components"]["success"])
    assert np.any(solved[0]["advantage_components"]["smoothness"])
    assert len(calls) == (8 + 9) * 8
    assert len(captures) == 8 + 9
    assert len(set(captures)) == len(captures)
    assert len({seed for _, _, seed in calls}) == len(calls)
    assert len(state["log"]["solved_problems"]) == 4
    assert all(
        task["groups_at_level"] == 4
        for task in state["log"]["assistance_curriculum"]["tasks"].values()
    )
    assert state["log"]["reward_definition"]["weights"]["smoothness"] == 1.
    # Resume with the saved RNG and bucket; no extra model is loaded.
    config.resume = True
    config.steps = 3
    train(config)
    resumed = torch.load(config.output / "latest_state.pt", weights_only=False)
    assert len(resumed["log"]["solved_problems"]) == 4
    assert sum(
        g["sampling_bucket"] == "solved"
        for g in resumed["log"]["steps"][-1]["groups"]
    ) == 1
    assert sorted(path.name for path in (config.output / "generations").glob("*.pt")) == [
        "step_000000002.pt", "step_000000003.pt",
    ]
    assert not (config.output / "latest_model.pt").exists()
    assert not (config.output / "latest_optimizer.pt").exists()
    assert not list(config.output.glob("delta_step_*.pt"))

    class Requests:
        def __init__(self):
            self.completed = {}

        def results(self):
            return list(self.completed.values())

        def request(self, key, level, *args):
            identity = f"{key}-{level}"
            path = tmp_path / identity / "request.json"
            path.parent.mkdir(exist_ok=True)
            manifest = path.parent / "manifest.json"
            manifest.write_text("{}")
            self.completed[identity] = {
                "request_id": identity, "status": "complete", "slice": key, "level": level,
                "manifest_path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
            return path

        def close(self):
            pass

    queue = Requests()
    monkeypatch.setattr(trainer, "HarderDataRequests", lambda *args: queue)
    monkeypatch.setattr(trainer, "load_native_pool", lambda *args: {
        task: [CorpusEpisode(tmp_path / "native" / task, {"task": task, "target": "instance_0",
                                                        "native_entry": {"root": "/unused"}})]
        for task in trainer.TASKS
    })
    config.resume = False
    config.output = tmp_path / "automatic"
    config.groups_per_task = 4
    config.automatic_difficulty = True
    config.target_success_curriculum = True
    config.difficulty_catalog = tmp_path / "catalog.json"
    for task in trainer.TASKS:
        setattr(config, f"{task}_curriculum_max", 0.)
    train(config)
    adaptive = torch.load(config.output / "latest_state.pt", weights_only=False)
    assert all(item["level"] == 1 for item in adaptive["log"]["difficulty_curriculum"]["slices"].values())
    assert any(group["source"].startswith(str(tmp_path / "native"))
               for group in adaptive["log"]["steps"][-1]["groups"])
    config.resume = True
    config.steps = 4
    train(config)
    adaptive = torch.load(config.output / "latest_state.pt", weights_only=False)
    assert all(item["level"] == 2 for item in adaptive["log"]["difficulty_curriculum"]["slices"].values())
    assert any(item["source"].startswith(str(tmp_path / "native"))
               for item in adaptive["log"]["solved_problems"].values())


def test_new_scene_constraints_reject_collision_even_when_goal_is_reached(monkeypatch, tmp_path):
    import so101.train_trajectory_rl as trainer
    sim = Namespace(
        fps=1, obstacle_contact_events=[],
        position=lambda name: np.array([.2, 0., 0.]),
        bounded_policy_action=lambda action: (action, 0.),
        encode=lambda action: action,
    )
    sim.step = lambda goal, phase: sim.obstacle_contact_events.append({"body": "gripper"})
    model = Namespace(sample_actions_entity=lambda *args: (torch.zeros(1, 1, 6),))
    controller = Namespace(
        reset=lambda *a, **k: None, command_inputs=lambda *a: (),
        policy=Namespace(eval=lambda: None, model=model),
        action_std=np.ones(6), action_mean=np.zeros(6), observe=lambda *a: None,
        config=Namespace(history_steps=4),
    )
    monkeypatch.setattr(trainer, "reset_from_episode", lambda *a: ("short", None, [("short", "object")]))
    monkeypatch.setattr(trainer, "replay_gold_prefix", lambda *a: (0, 0.))
    monkeypatch.setattr(trainer, "control_phase", lambda *a: "acquire")
    monkeypatch.setattr(trainer, "wrong_object_lift", lambda *a: False)
    monkeypatch.setattr(trainer, "successful_step", lambda *a: True)
    episode = CorpusEpisode(tmp_path, {"task": "grasp", "constraints": {"avoid_obstacles": True}})
    result = trainer.rollout(sim, controller, episode, 1, 3, 1,
                             {"optimal_steps": 1., "jerk_p10": .1}, False)
    assert result.constraint_violation and not result.success and result.steps == 1


@pytest.mark.parametrize("store", [False, True])
def test_rejected_policy_action_reports_reason_without_relaxing_limits(monkeypatch, tmp_path, capsys, store):
    import so101.train_trajectory_rl as trainer

    def reject(action):
        raise ValueError("Policy target exceeds a joint boundary by 6 units")

    sim = Namespace(fps=1, obstacle_contact_events=[],
                    position=lambda name: np.array([.2, 0., 0.]),
                    bounded_policy_action=reject)
    controller = Namespace(
        reset=lambda *a, **k: None, command_inputs=lambda *a: (),
        policy=Namespace(eval=lambda: None, config=Namespace(chunk_size=1), model=Namespace(
            sample_actions_entity_sde=lambda *a, **k: (torch.zeros(1, 1, 6), []))),
        action_std=np.ones(6), action_mean=np.zeros(6),
        config=Namespace(history_steps=4),
    )
    monkeypatch.setattr(trainer, "reset_from_episode", lambda *a: ("short", None, []))
    monkeypatch.setattr(trainer, "replay_gold_prefix", lambda *a: (0, 0.))
    monkeypatch.setattr(trainer, "control_phase", lambda *a: "acquire")
    episode = CorpusEpisode(tmp_path, {"task": "view"})
    result = trainer.rollout(sim, controller, episode, 1, 300, 1,
                             {"optimal_steps": 1., "jerk_p10": .1}, store,
                             denoising_noise_scale=.1, record_path_noise=store)
    assert not result.success and result.steps == 0
    assert "boundary by 6" in trainer.serialized_trajectory(result)["invalid_action_error"]
    assert json.loads(capsys.readouterr().out)["event"] == "policy_action_rejected"
    if store:
        assert len(result.decisions) == 1
        assert result.decisions[0].executed_steps == 0
        assert result.decisions[0].attempted_steps == 1


def test_rollout_renders_only_history_consumed_by_next_decision(monkeypatch, tmp_path):
    from collections import deque
    import so101.train_trajectory_rl as trainer

    frames = deque([0] * 4, maxlen=4)
    observed = []
    decisions = []
    sim = Namespace(
        fps=10, frame=0, obstacle_contact_events=[],
        position=lambda name: np.zeros(3),
        bounded_policy_action=lambda action: (action, None),
        encode=lambda action: action,
    )

    def step(goal, phase):
        sim.frame += 1

    def observe(sim, action):
        observed.append(sim.frame)
        frames.append(sim.frame)

    def inputs(*args):
        decisions.append(list(frames))
        return ()

    sim.step = step
    controller = Namespace(
        config=Namespace(history_steps=4),
        reset=lambda *a, **k: None, command_inputs=inputs, observe=observe,
        action_std=np.ones(6), action_mean=np.zeros(6),
        policy=Namespace(eval=lambda: None, model=Namespace(
            sample_actions_entity=lambda *args: (torch.zeros(1, 6, 6),))),
    )
    monkeypatch.setattr(trainer, "reset_from_episode", lambda *a: ("short", None, []))
    monkeypatch.setattr(trainer, "replay_gold_prefix", lambda *a: (0, 0.))
    monkeypatch.setattr(trainer, "control_phase", lambda *a: "acquire")
    monkeypatch.setattr(trainer, "wrong_object_lift", lambda *a: False)
    monkeypatch.setattr(trainer, "successful_step", lambda *a: False)
    result = trainer.rollout(
        sim, controller, CorpusEpisode(tmp_path, {"task": "grasp"}),
        1, 15, 6, {"optimal_steps": 1., "jerk_p10": .1}, False,
    )
    assert result.steps == 15
    assert decisions == [[0] * 4, [3, 4, 5, 6], [9, 10, 11, 12]]
    assert observed == [3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15]
    assert list(frames) == [12, 13, 14, 15]


@pytest.mark.parametrize("record_noise", [False, True])
def test_flow_replay_transfers_after_sampling_without_changing_random_draws(monkeypatch, record_noise):
    from so101.control_model.flow_rl import sample_flow_path

    calls = []
    original = torch.Tensor.cpu

    def cpu(value, *args, **kwargs):
        calls.append(tuple(value.shape))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    torch.manual_seed(7)
    initial = torch.ones(1, 4, 6)
    result, path = sample_flow_path(
        lambda value, time: value * .25, initial, 4, .1, (2, 6), record_noise,
    )
    assert len(calls) == 2
    assert len(path) == 4 and path[-1].active_shape == (2, 6)
    assert all(item.latent.device.type == "cpu" for item in path)
    assert torch.equal(path[0].latent, initial)
    assert torch.equal(path[-1].next_latent, result)
    assert all(torch.equal(first.next_latent, second.latent)
               for first, second in zip(path, path[1:]))
    if record_noise:
        from so101.control_model.flow_rl import reparameterize_flow_path
        replayed = reparameterize_flow_path(
            lambda value, time: value * .25, path, torch.device("cpu"),
            checkpoint_velocity=False,
        )
        assert torch.equal(replayed, result)


@pytest.mark.parametrize("store_decisions", [True, False])
def test_batched_rollouts_preserve_independent_noise_histories_and_early_termination(
    monkeypatch, tmp_path, store_decisions,
):
    from collections import deque
    import so101.train_trajectory_rl as trainer
    from so101.control_model.flow_rl import independent_noise_like, sample_flow_path

    sizes = []
    monkeypatch.setattr(trainer, "activate_render_context", lambda sim: None)

    def sample(inputs, noise_scale, active_shape, record_noise=False, generators=None, capture=None):
        sizes.append(inputs[0].shape[0])
        initial = independent_noise_like(torch.empty(inputs[0].shape[0], 4, 6), generators)
        return sample_flow_path(lambda value, time: value * .25, initial, 3,
                                noise_scale, active_shape, record_noise, generators)

    class Controller:
        device = "cpu"
        config = Namespace(history_steps=4)
        policy = Namespace(eval=lambda: None, config=Namespace(chunk_size=4),
                           model=Namespace(sample_actions_entity_sde=sample))
        action_std = np.ones(6)
        action_mean = np.zeros(6)

        def reset(self, sim, seed, **kwargs):
            torch.manual_seed(seed)
            sim.seed = seed
            sim.frame = 0
            self.history = deque([0] * 4, maxlen=4)

        def command_inputs(self, *args):
            return (torch.tensor([list(self.history)], dtype=torch.float32),)

        def observe(self, sim, action):
            self.history.append(sim.frame)

    def simulation():
        sim = Namespace(
            fps=1, frame=0, obstacle_contact_events=[],
            position=lambda name: np.zeros(3),
            bounded_policy_action=lambda action: (action, None), encode=lambda action: action,
        )
        sim.step = lambda *args: setattr(sim, "frame", sim.frame + 1)
        return sim

    monkeypatch.setattr(trainer, "reset_from_episode", lambda *a: ("short", None, []))
    monkeypatch.setattr(trainer, "replay_gold_prefix", lambda *a: (0, 0.))
    monkeypatch.setattr(trainer, "control_phase", lambda *a: "acquire")
    monkeypatch.setattr(trainer, "wrong_object_lift", lambda *a: False)
    monkeypatch.setattr(trainer, "successful_step",
                        lambda sim, *args: sim.seed % 2 == 0 and sim.frame >= 2)
    episode = CorpusEpisode(tmp_path, {"task": "grasp"})
    bounds = {"optimal_steps": 1., "jerk_p10": .1}
    controller = Controller()
    expected = [
        trainer.rollout(simulation(), controller, episode, seed, 9, 4,
                        bounds, store_decisions, 0., .1, record_path_noise=True)
        for seed in (2, 3, 4)
    ]
    sizes.clear()
    actual = trainer.rollout_group(
        [simulation() for _ in range(3)], controller, episode, [2, 3, 4],
        9, 4, bounds, 0., .1, True, store_decisions=store_decisions,
    )
    assert sizes == [3, 1, 1]
    assert [item.steps for item in actual] == [2, 9, 2]
    for serial, batched in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(serial.actions, batched.actions)
        assert serial.success == batched.success
        assert len(serial.decisions) == len(batched.decisions)
        for first, second in zip(serial.decisions, batched.decisions, strict=True):
            assert torch.equal(first.inputs[0], second.inputs[0])
            for old, new in zip(first.denoising_path, second.denoising_path, strict=True):
                assert torch.equal(old.noise, new.noise)
                assert torch.equal(old.old_log_prob, new.old_log_prob)


def test_frozen_batch_observation_preserves_sampled_row_without_cache_aliasing():
    from transformers.cache_utils import DynamicCache
    from so101.control_model.modeling_entity_smolvla import FrozenObservation

    cache = DynamicCache()
    keys = torch.arange(3 * 2 * 5 * 4, dtype=torch.float32).reshape(3, 2, 5, 4)
    cache.update(keys, keys + 1, 0)
    pad = torch.ones(3, 5, dtype=torch.bool)
    frozen = FrozenObservation.from_row(pad, cache, 1)
    expected = keys[1:2].clone()
    cache.layers[0].keys.zero_()
    pad.zero_()
    assert torch.equal(frozen.cache.layers[0].keys, expected)
    assert frozen.pad.all()
    restored = FrozenObservation.copy_cache(frozen.cache, "cpu")
    restored.layers[0].keys.zero_()
    assert torch.equal(frozen.cache.layers[0].keys, expected)


def test_new_scene_destination_precision_is_enforced():
    from so101.train_trajectory_rl import successful_step
    positions = {"short": np.array([.2, 0., 0.]), "goal_right": np.array([.23, 0., 0.])}
    sim = Namespace(position=positions.__getitem__, contacts=lambda name: set(),
                    episode_constraints={"destination_tolerance_m": .025})
    assert not successful_step(sim, "place", "short", "goal_right", positions, True, 40)
    sim.episode_constraints = {}
    assert successful_step(sim, "place", "short", "goal_right", positions, True, 40)


def test_difficulty_scene_restores_exact_poses_perception_and_constraints(monkeypatch):
    from so101.robot_sim import BeamSimulation
    from so101.collect_difficulty_sim import configure_scene
    from so101.train_trajectory_rl import reset_from_episode
    from pathlib import Path
    sim = BeamSimulation(render=False)
    try:
        monkeypatch.setattr(sim, "target_oriented_box", lambda *args: [0.] * 6)
        monkeypatch.setattr(sim, "entity_mask", lambda *args: np.zeros((2, 2), dtype=np.uint8))
        config = configure_scene(sim, "place", 2, 122000)
        config["perception"]["hide_unseen_centers"] = True
        initial = deepcopy(sim.initial)
        expected = sim.state().copy()
        metadata = {**config, "task": "place", "initial": initial, "color_targets": True}
        sim.reset(20)
        target, destination, entities = reset_from_episode(sim, CorpusEpisode(Path("unused"), metadata))
        np.testing.assert_allclose(sim.state(), expected, atol=1e-5)
        for name, position in initial["objects"].items():
            np.testing.assert_allclose(sim.position(name), position)
        np.testing.assert_allclose(sim.position("goal_right"), initial["appearance"]["destinations"]["goal_right"])
        assert target == "short" and destination == "goal_right"
        assert sim.hide_unseen_centers
        assert sim.episode_constraints["destination_tolerance_m"] == .03
        assert entities == [("short", "object"), ("goal_right", "destination")]
    finally:
        sim.close()


def test_search_scenes_do_not_leak_hidden_target_coordinates(monkeypatch):
    from so101.robot_sim import BeamSimulation
    sim = BeamSimulation(render=False)
    try:
        sim.reset(90000)
        monkeypatch.setattr(sim, "entity_mask", lambda *args: np.zeros((2, 2), dtype=np.uint8))
        assert sim.entity_camera_depth("short")[1]
        sim.hide_unseen_centers = True
        assert sim.projected_entity_center("short") == ([0., 0.], False)
        assert sim.entity_camera_depth("short") == (0., False)
        assert sim.target_oriented_box("short")[:2] == [0., 0.]
    finally:
        sim.close()


def test_difficulty_collection_reports_missing_cases_and_resumes_without_overwriting(tmp_path, monkeypatch):
    import so101.collect_difficulty_sim as collector
    monkeypatch.setattr(collector, "BeamSimulation", lambda *args: Namespace(close=lambda: None))
    def invalid(*args):
        raise ValueError("No feasible camera pose")
    monkeypatch.setattr(collector, "configure_scene", invalid)
    root = tmp_path / "scenes"
    first = collector.collect(root, 1, 90000, attempts_per_case=1)
    assert len(first["missing_cases"]) == 16 and len(first["attempts"]) == 16
    assert all(row["outcome"] == "invalid_scene" and row["error"] for row in first["attempts"])
    second = collector.collect(root, 1, 90000, attempts_per_case=1, resume=True)
    assert second["attempts"] == first["attempts"]
    with pytest.raises(ValueError, match="original seed"):
        collector.collect(root, 1, 90001, resume=True)
    with pytest.raises(ValueError, match="non-overlapping seed"):
        collector.collect(tmp_path / "invalid", 101, 90000)
