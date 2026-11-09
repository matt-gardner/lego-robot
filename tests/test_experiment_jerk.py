from types import SimpleNamespace

import numpy as np
import pytest
import torch

from so101.experiments.experiment_jerk import action_jerk_gradient, jerk_advantages, problem_batch
from so101.control_model.flow_rl import reparameterize_flow_path, sample_flow_path
from so101.train_trajectory_rl import CorpusEpisode, Trajectory


@pytest.mark.parametrize("representation", ["negative-rms", "negative-square", "negative-log"])
def test_jerk_only_uses_failures_and_does_not_reward_early_invalid_termination(representation):
    rows = [Trajectory("view", False, False, steps, jerk, 0., 0., 0., [])
            for steps, jerk in ((64, 2.), (64, 8.), (1, 0.))]
    advantages, _ = jerk_advantages(rows, representation, False, 64)
    assert advantages[0] > advantages[1] > advantages[2]


def test_batches_have_exactly_eight_distinct_problems_two_per_task():
    pool = {task: [f"{task}_{index}" for index in range(10)]
            for task in ("grasp", "view", "push", "place")}
    rng = np.random.default_rng(42)
    first, second = problem_batch(pool, rng), problem_batch(pool, rng)
    assert len(first) == len(set(first)) == 8 and first != second
    assert all(sum(source.startswith(task) for source in first) == 2 for task in pool)


def test_fixed_horizon_measurement_cannot_improve_by_terminating_at_success(monkeypatch, tmp_path):
    import so101.train_trajectory_rl as trainer
    sim = SimpleNamespace(
        fps=1, obstacle_contact_events=[],
        position=lambda name: np.array([.2, 0., 0.]),
        bounded_policy_action=lambda action: (action, 0.),
        encode=lambda action: action, step=lambda *args: None,
    )
    model = SimpleNamespace(sample_actions_entity=lambda *args: (torch.zeros(1, 1, 6),))
    controller = SimpleNamespace(
        reset=lambda *a, **k: None, command_inputs=lambda *a: (),
        policy=SimpleNamespace(eval=lambda: None, model=model),
        action_std=np.ones(6), action_mean=np.zeros(6), observe=lambda *a: None,
        config=SimpleNamespace(history_steps=4),
    )
    monkeypatch.setattr(trainer, "reset_from_episode", lambda *a: ("short", None, [("short", "object")]))
    monkeypatch.setattr(trainer, "replay_gold_prefix", lambda *a: (0, 0.))
    monkeypatch.setattr(trainer, "control_phase", lambda *a: "view")
    monkeypatch.setattr(trainer, "wrong_object_lift", lambda *a: False)
    monkeypatch.setattr(trainer, "successful_step", lambda *a: True)
    episode = CorpusEpisode(tmp_path, {"task": "view"})
    bounds = {"optimal_steps": 1., "jerk_p10": .1}
    ordinary = trainer.rollout(sim, controller, episode, 1, 5, 1, bounds, False)
    fixed = trainer.rollout(sim, controller, episode, 1, 5, 1, bounds, False, fixed_horizon=True)
    assert ordinary.steps == 1 and fixed.steps == 5
    assert fixed.first_success_step == 1 and fixed.completion_efficiency == 1.


def test_recording_path_noise_preserves_actions_and_rng_sequence():
    initial = torch.ones(1, 8, 6)
    velocity = lambda value, time: value * time[:, None, None]
    torch.manual_seed(31)
    ordinary, _ = sample_flow_path(velocity, initial, 10, .1)
    following = torch.randn(4)
    torch.manual_seed(31)
    recorded, path = sample_flow_path(velocity, initial, 10, .1, record_noise=True)
    torch.testing.assert_close(recorded, ordinary, rtol=0, atol=0)
    torch.testing.assert_close(torch.randn(4), following, rtol=0, atol=0)
    torch.testing.assert_close(reparameterize_flow_path(velocity, path, torch.device("cpu")).detach(), ordinary)


def test_full_path_reparameterization_matches_sample_and_numerical_gradient():
    torch.manual_seed(3)
    parameter = torch.nn.Parameter(torch.tensor(.2))
    velocity = lambda value, time: parameter * value
    output, path = sample_flow_path(velocity, torch.randn(1, 8, 6), 10, .1, record_noise=True)
    replay = reparameterize_flow_path(velocity, path, torch.device("cpu"))
    torch.testing.assert_close(replay.detach(), output)
    loss = torch.diff(replay[0, :, :5], n=3, dim=0).square().mean()
    loss.backward()
    analytic = parameter.grad.item()
    values = []
    for value in (.201, .199):
        with torch.no_grad():
            parameter.fill_(value)
        replay = reparameterize_flow_path(velocity, path, torch.device("cpu"), checkpoint_velocity=False)
        values.append(torch.diff(replay[0, :, :5], n=3, dim=0).square().mean().item())
    assert analytic == pytest.approx((values[0] - values[1]) / .002, rel=.002)


def test_action_gradient_includes_chunk_boundary_and_excludes_gripper():
    actions = np.zeros((32, 6), dtype=np.float32)
    actions[16:, 0] = 2.
    actions[:, 5] = np.arange(32) ** 2
    gradient, cost = action_jerk_gradient(actions, 6.)
    assert cost > 0
    assert gradient[15, 0] != 0 and gradient[16, 0] != 0
    assert not gradient[:, 5].any()
    improved = actions - .1 * gradient.numpy()
    assert action_jerk_gradient(improved, 6.)[1] < cost
