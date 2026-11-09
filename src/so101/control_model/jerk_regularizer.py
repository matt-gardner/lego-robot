"""Local command-smoothness gradients; observations and simulator feedback stay fixed."""
from __future__ import annotations

import numpy as np
import torch

from .flow_rl import reparameterize_flow_path, to_device


def action_jerk_gradient(actions: np.ndarray, scale: float, margin: float = 0.):
    if len(actions) < 4 or not np.isfinite([scale, margin]).all() or scale <= 0 or margin < 0:
        raise ValueError("Jerk derivatives require four actions, a positive scale and a nonnegative margin")
    values = torch.tensor(actions, dtype=torch.float32, requires_grad=True)
    squared_jerk = torch.diff(values[:, :5], n=3, dim=0).square().mean()
    cost = (torch.relu(squared_jerk.clamp_min(1e-12).sqrt() - margin).square()
            if margin else squared_jerk) / scale ** 2
    cost.backward()
    return values.grad, float(cost.detach())


def replay_command_gradients(controller, trajectory, low, high, coefficient, gradients,
                             *, bounded: bool, attempted: bool):
    position = 0
    controller.policy.eval()
    for decision in trajectory.decisions:
        count = (decision.attempted_steps if attempted and decision.attempted_steps is not None
                 else decision.executed_steps)
        weights = gradients[position:position + count]
        position += count
        if not count or not weights.any():
            continue
        inputs = to_device(decision.inputs, controller.device)
        observation = getattr(decision, "frozen_observation", None)
        velocity = controller.policy.model._rl_velocity_function(
            inputs, **({"frozen_observation": observation} if observation is not None else {}),
        )
        actions = reparameterize_flow_path(velocity, decision.denoising_path,
                                          torch.device(controller.device))
        torch.testing.assert_close(actions[0, :count, :6].detach().cpu(),
                                   decision.normalized_actions[:count, :6], rtol=1e-4, atol=1e-4)
        physical = actions[0, :count, :6] * torch.as_tensor(controller.action_std, device=controller.device) + torch.as_tensor(controller.action_mean, device=controller.device)
        commands = torch.maximum(torch.minimum(physical, high), low) if bounded else physical
        (commands * weights.to(controller.device) * coefficient).sum().backward()
    if position != len(gradients):
        raise ValueError("Recorded decisions and command gradients differ")


def pathwise_backward(controller, trajectory, low, high, coefficient: float, scale: float,
                      margin: float | None):
    if margin is None:
        return None
    action_gradient, cost = action_jerk_gradient(trajectory.actions, scale, margin)
    if cost == 0:
        return cost
    replay_command_gradients(controller, trajectory, low, high, coefficient, action_gradient,
                             bounded=True, attempted=False)
    return cost


def action_boundary_backward(controller, trajectory, low, high, coefficient: float, scale: float,
                             correction_limit: float):
    if not np.isfinite([scale, correction_limit]).all() or scale <= 0 or correction_limit < 0:
        raise ValueError("Action-boundary scale must be positive and correction limit nonnegative and finite")
    recorded = [
        decision.normalized_actions[:decision.attempted_steps if decision.attempted_steps is not None
                                    else decision.executed_steps, :6]
        for decision in trajectory.decisions
    ]
    if not recorded or not sum(len(actions) for actions in recorded):
        return None
    physical = (torch.cat(recorded).float() * torch.as_tensor(controller.action_std)
                + torch.as_tensor(controller.action_mean)).detach().requires_grad_()
    if not torch.isfinite(physical).all():
        raise FloatingPointError("Non-finite recorded policy proposal")
    arm = physical[:, :5]
    low, high = low.detach().cpu(), high.detach().cpu()
    maximum_correction = torch.maximum(low[:5] - arm, arm - high[:5]).clamp_min(0).amax()
    loss = torch.relu(maximum_correction - correction_limit).square() / scale ** 2
    cost = float(loss.detach())
    if cost:
        loss.backward()
        replay_command_gradients(controller, trajectory, low.to(controller.device), high.to(controller.device),
                                 coefficient, physical.grad, bounded=False, attempted=True)
    return cost


def gradient_vector(parameters):
    return torch.cat([
        parameter.grad.detach().flatten() if parameter.grad is not None
        else parameter.new_zeros(parameter.numel())
        for parameter in parameters
    ])


def group_jerk_backward(controller, trajectories, low, high, weight, scale, margin):
    if not any(trajectory.success for trajectory in trajectories):
        return [None] * len(trajectories)
    return [
        pathwise_backward(controller, trajectory, low, high,
                          weight / len(trajectories), scale, margin)
        if margin is not None and trajectory.steps >= 4 else None
        for trajectory in trajectories
    ]


def group_action_boundary_backward(controller, trajectories, low, high, weight, scale, correction_limit):
    if not any(trajectory.success for trajectory in trajectories):
        return [None] * len(trajectories)
    return [
        action_boundary_backward(controller, trajectory, low, high,
                                 weight / len(trajectories), scale, correction_limit)
        for trajectory in trajectories
    ]


def normalize_gradient_components(parameters, jerk_sum, policy_groups: int, jerk_groups: int,
                                  *, boundary_sum=None, boundary_groups: int = 0):
    active = any(parameter.grad is not None for parameter in parameters)
    task = (gradient_vector(parameters) - jerk_sum
            - (boundary_sum if boundary_sum is not None else 0)) / max(policy_groups, 1)
    jerk = jerk_sum / max(jerk_groups, 1)
    boundary = boundary_sum / max(boundary_groups, 1) if boundary_sum is not None else 0
    combined = task + jerk + boundary
    position = 0
    if active:
        for parameter in parameters:
            value = combined[position:position + parameter.numel()].view_as(parameter)
            if parameter.grad is not None:
                parameter.grad.copy_(value)
            elif bool(value.any()):
                parameter.grad = value.clone()
            position += parameter.numel()
    task_norm, jerk_norm = float(task.norm()), float(jerk.norm())
    metrics = {
        "weighted_task_gradient_norm": task_norm,
        "weighted_jerk_gradient_norm": jerk_norm,
        "task_jerk_gradient_cosine": (
            float(torch.dot(task, jerk)) / (task_norm * jerk_norm) if task_norm and jerk_norm else None
        ),
        "jerk_regularized_groups": jerk_groups,
        "gradient_component_scope": "separately group-averaged, before shared clipping and Adam preconditioning",
    }
    if boundary_sum is not None:
        metrics.update(weighted_boundary_gradient_norm=float(boundary.norm()),
                       boundary_regularized_groups=boundary_groups)
    return active, metrics
