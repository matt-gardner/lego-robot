from types import SimpleNamespace

import pytest
import torch

import so101.control_model.jerk_regularizer as regularizer
from so101.train_trajectory_rl import policy_optimizer_step


@pytest.mark.parametrize("component", ["jerk", "boundary"])
def test_zero_success_groups_cannot_advance_gradients_or_old_adam_momentum(monkeypatch, component):
    parameter = torch.nn.Parameter(torch.tensor([2.]))
    optimizer = torch.optim.AdamW([parameter], lr=.01)
    parameter.square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = parameter.detach().clone()
    before_step = optimizer.state[parameter]["step"].clone()
    def unexpected(*args, **kwargs):
        pytest.fail("A discarded group must not replay any regularizer gradient")
    monkeypatch.setattr(regularizer, "pathwise_backward", unexpected)
    monkeypatch.setattr(regularizer, "action_boundary_backward", unexpected)
    trajectories = [SimpleNamespace(steps=10, success=False), SimpleNamespace(steps=10, success=False)]
    if component == "jerk":
        losses = regularizer.group_jerk_backward(None, trajectories, None, None, .02, 6, .3)
    else:
        losses = regularizer.group_action_boundary_backward(None, trajectories, None, None, .02, 6, 5)
    assert losses == [None, None] and parameter.grad is None
    active, _ = regularizer.normalize_gradient_components([parameter], torch.zeros(1), 0, 0)
    assert not policy_optimizer_step(optimizer, [parameter], int(active), 1)["optimizer_update_applied"]
    torch.testing.assert_close(parameter.detach(), before)
    torch.testing.assert_close(optimizer.state[parameter]["step"], before_step)


def test_mixed_success_groups_keep_jerk_gradients_for_failed_rollouts(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([2.]))
    calls = []
    def backward(controller, trajectory, low, high, coefficient, scale, margin):
        calls.append((trajectory.success, coefficient))
        loss = parameter.square().sum()
        (loss * coefficient).backward()
        return float(loss.detach())
    monkeypatch.setattr(regularizer, "pathwise_backward", backward)
    trajectories = [SimpleNamespace(steps=10, success=True), SimpleNamespace(steps=10, success=False)]
    losses = regularizer.group_jerk_backward(None, trajectories, None, None, .02, 6, .3)
    assert losses == [4., 4.] and calls == [(True, .01), (False, .01)]
    jerk_sum = regularizer.gradient_vector([parameter])
    active, metrics = regularizer.normalize_gradient_components([parameter], jerk_sum, 0, 1)
    assert active and metrics["weighted_task_gradient_norm"] == 0
    assert metrics["weighted_jerk_gradient_norm"] > 0
    optimizer = torch.optim.AdamW([parameter], lr=.01)
    update = policy_optimizer_step(optimizer, [parameter], int(active), 1)
    assert update["optimizer_update_applied"] and parameter.item() < 2
    calls.clear()
    assert regularizer.group_jerk_backward(None, trajectories, None, None, .02, 6, None) == [None, None]
    assert not calls


def test_task_and_jerk_gradients_have_independent_group_denominators():
    parameter = torch.nn.Parameter(torch.tensor([1., -1.]))
    for _ in range(2):
        (parameter * torch.tensor([1., 2.])).sum().backward()
    before = regularizer.gradient_vector([parameter])
    for _ in range(4):
        (parameter * torch.tensor([1.5, 2.])).sum().backward()
    jerk_sum = regularizer.gradient_vector([parameter]) - before
    active, metrics = regularizer.normalize_gradient_components([parameter], jerk_sum, 2, 4)
    torch.testing.assert_close(parameter.grad, torch.tensor([2.5, 4.]))
    assert metrics["weighted_task_gradient_norm"] == pytest.approx(5 ** .5)
    assert metrics["weighted_jerk_gradient_norm"] == pytest.approx(2.5)
    optimizer = torch.optim.SGD([parameter], lr=.1)
    policy_optimizer_step(optimizer, [parameter], int(active), 100)
    torch.testing.assert_close(parameter.detach(), torch.tensor([.75, -1.4]))


def test_inactive_margin_does_not_create_zero_gradients_that_advance_old_adam_momentum():
    parameter = torch.nn.Parameter(torch.tensor([2.]))
    optimizer = torch.optim.AdamW([parameter], lr=.01)
    parameter.square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = parameter.detach().clone()
    active, _ = regularizer.normalize_gradient_components([parameter], torch.zeros(1), 0, 3)
    assert not active and parameter.grad is None
    result = policy_optimizer_step(optimizer, [parameter], int(active), 1)
    assert not result["optimizer_update_applied"]
    torch.testing.assert_close(parameter.detach(), before)


def test_boundary_gradient_corrects_rejected_unexecuted_arm_action_without_clipping(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([[[110., 50., 50., 50., 50., 1000.]]]))
    controller = SimpleNamespace(
        device="cpu", action_std=[1.] * 6, action_mean=[0.] * 6,
        policy=SimpleNamespace(eval=lambda: None, model=SimpleNamespace(
            _rl_velocity_function=lambda inputs: None)),
    )
    decision = SimpleNamespace(normalized_actions=parameter[0].detach().clone(),
                               executed_steps=0, attempted_steps=1, inputs=(), denoising_path=[])
    trajectory = SimpleNamespace(decisions=[decision])
    monkeypatch.setattr(regularizer, "reparameterize_flow_path", lambda *args: parameter)
    loss = regularizer.action_boundary_backward(controller, trajectory, torch.zeros(6),
                                                torch.full((6,), 100.), 1., 6., 5.)
    assert loss == pytest.approx(25 / 36)
    assert parameter.grad[0, 0, 0] > 0
    assert not parameter.grad[0, 0, 1:].any()
    optimizer = torch.optim.SGD([parameter], lr=1.)
    optimizer.step()
    assert parameter[0, 0, 0] < 110


def test_guard_accepted_clipping_and_gripper_excursions_do_not_trigger_boundary_replay(monkeypatch):
    controller = SimpleNamespace(action_std=[1.] * 6, action_mean=[0.] * 6)
    trajectory = SimpleNamespace(decisions=[SimpleNamespace(
        normalized_actions=torch.tensor([[105.] + [50.] * 4 + [1000.]]),
        executed_steps=1, attempted_steps=1,
    )])
    def unexpected(*args, **kwargs):
        pytest.fail("No denoising replay should occur for an inactive arm-boundary loss")
    monkeypatch.setattr(regularizer, "replay_command_gradients", unexpected)
    assert regularizer.action_boundary_backward(controller, trajectory, torch.zeros(6),
                                                torch.full((6,), 100.), 1., 6., 5.) == 0


def test_boundary_penalty_only_credits_the_actual_rejected_proposal(monkeypatch):
    controller = SimpleNamespace(device="cpu", action_std=[1.] * 6, action_mean=[0.] * 6)
    trajectory = SimpleNamespace(decisions=[
        SimpleNamespace(normalized_actions=torch.tensor([[104.] + [50.] * 5]),
                        executed_steps=1, attempted_steps=1),
        SimpleNamespace(normalized_actions=torch.tensor([[106.] + [50.] * 5]),
                        executed_steps=0, attempted_steps=1),
    ])
    recorded = []
    monkeypatch.setattr(regularizer, "replay_command_gradients",
                        lambda *args, **kwargs: recorded.append(args[5].clone()))
    cost = regularizer.action_boundary_backward(controller, trajectory, torch.zeros(6),
                                                torch.full((6,), 100.), 1., 6., 5.)
    assert cost == pytest.approx(1 / 36)
    assert not recorded[0][0].any()
    assert recorded[0][1, 0] > 0
    assert not recorded[0][1, 1:].any()


def test_boundary_gradients_use_their_own_eligible_group_count():
    parameter = torch.nn.Parameter(torch.tensor([1., -1.]))
    parameter.grad = torch.tensor([7., -1.])
    active, metrics = regularizer.normalize_gradient_components(
        [parameter], torch.tensor([2., 1.]), 1, 2,
        boundary_sum=torch.tensor([4., -4.]), boundary_groups=4,
    )
    assert active
    torch.testing.assert_close(parameter.grad, torch.tensor([3., 1.5]))
    assert metrics["weighted_boundary_gradient_norm"] == pytest.approx(2 ** .5)
    assert metrics["boundary_regularized_groups"] == 4
