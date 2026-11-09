from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor


def to_device(value: Any, device: str) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    raise TypeError(f"Unsupported input type: {type(value)}")


@dataclass(frozen=True)
class FlowTransition:
    latent: Tensor
    next_latent: Tensor
    old_log_prob: Tensor
    time: float
    step_size: float
    noise_scale: float
    active_shape: tuple[int, int] | None = None
    noise: Tensor | None = None


def transition_distribution(
    latent: Tensor,
    velocity: Tensor,
    time: float,
    step_size: float,
    noise_scale: float,
) -> tuple[Tensor, float]:
    if not 0 < time <= 1 or not 0 < step_size <= time + 1e-8:
        raise ValueError("Invalid reverse-flow time or step size")
    if not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("Flow noise scale must be finite and nonnegative")
    # Reverse rectified-flow SDE, with sigma(t) = noise_scale * sqrt(t).
    drift = velocity + noise_scale ** 2 / 2 * (
        latent + (1 - time) * velocity
    )
    return (
        latent - step_size * drift,
        noise_scale * math.sqrt(time * step_size),
    )


def active_coordinates(value: Tensor, shape: tuple[int, int] | None) -> Tensor:
    if shape is None:
        return value
    steps, dimensions = shape
    if (
        value.ndim != 3 or not 0 < steps <= value.shape[1]
        or not 0 < dimensions <= value.shape[2]
    ):
        raise ValueError("Invalid active action shape")
    return value[:, :steps, :dimensions]


def gaussian_log_prob(
    sample: Tensor, mean: Tensor, std: float,
    active_shape: tuple[int, int] | None = None,
) -> Tensor:
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Gaussian transition standard deviation must be positive")
    residual = active_coordinates(sample.detach() - mean, active_shape)
    return (
        -.5 * (residual / std).square()
        - math.log(std) - .5 * math.log(2 * math.pi)
    ).flatten(start_dim=1).sum(dim=1)


@torch.no_grad()
def sample_flow_path(
    velocity: Callable[[Tensor, Tensor], Tensor],
    initial_noise: Tensor,
    num_steps: int,
    noise_scale: float,
    final_action_shape: tuple[int, int] | None = None,
    record_noise: bool = False,
) -> tuple[Tensor, list[FlowTransition]]:
    if num_steps <= 0:
        raise ValueError("Denoising steps must be positive")
    if not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("Flow noise scale must be finite and nonnegative")
    latent = initial_noise
    path = []
    step_size = 1. / num_steps
    for step in range(num_steps):
        time = 1. - step * step_size
        timestep = torch.full(
            (latent.shape[0],), time, dtype=torch.float32, device=latent.device
        )
        mean, std = transition_distribution(
            latent, velocity(latent, timestep), time, step_size, noise_scale
        )
        if noise_scale == 0:
            latent = mean
            continue
        noise = torch.randn_like(mean)
        next_latent = mean + std * noise
        active_shape = final_action_shape if step == num_steps - 1 else None
        path.append(FlowTransition(
            latent.detach(),
            next_latent.detach(),
            gaussian_log_prob(next_latent, mean, std, active_shape).detach(),
            time,
            step_size,
            noise_scale,
            active_shape,
            noise.detach() if record_noise else None,
        ))
        latent = next_latent
    if path:
        # Transfer after denoising, not between GPU steps; keep replay paths on CPU.
        tensors = [item.latent for item in path] + [item.next_latent for item in path]
        if record_noise:
            for item in path:
                if item.noise is None:
                    raise ValueError("Recorded flow transition is missing its sampled noise")
                tensors.append(item.noise)
        states = torch.stack(tensors).cpu()
        probabilities = torch.stack([item.old_log_prob for item in path]).cpu()
        count = len(path)
        path = [
            FlowTransition(
                states[index], states[count + index], probabilities[index],
                item.time, item.step_size, item.noise_scale, item.active_shape,
                states[2 * count + index] if record_noise else None,
            )
            for index, item in enumerate(path)
        ]
    return latent, path


def reparameterize_flow_path(
    velocity: Callable[[Tensor, Tensor], Tensor],
    path: list[FlowTransition],
    device: torch.device,
    checkpoint_velocity: bool = True,
) -> Tensor:
    """Replay recorded Gaussian draws with gradients through every denoising step."""
    from torch.utils.checkpoint import checkpoint

    if not path or any(item.noise is None for item in path):
        raise ValueError("Reparameterization requires explicitly recorded transition noise")
    latent = path[0].latent.to(device).detach().requires_grad_(True)
    for transition in path:
        timestep = torch.full((latent.shape[0],), transition.time, device=device, dtype=torch.float32)
        def evaluate(value, time_value=timestep):
            return velocity(value, time_value)
        # Reentrant checkpointing completes cache cleanup during every recomputation.
        predicted = checkpoint(evaluate, latent, use_reentrant=True, preserve_rng_state=False) if checkpoint_velocity else evaluate(latent)
        mean, std = transition_distribution(latent, predicted, transition.time,
                                            transition.step_size, transition.noise_scale)
        latent = mean + std * transition.noise.to(device)
    return latent


def clipped_policy_loss(
    log_prob: Tensor, old_log_prob: Tensor, advantage: float, clip: float
) -> Tensor:
    if not 0 < clip < 1:
        raise ValueError("Policy ratio clip must lie between zero and one")
    log_ratio = log_prob - old_log_prob.detach()
    ratio = log_ratio.exp()
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("Non-finite denoising policy ratio")
    return -torch.minimum(
        ratio * advantage,
        ratio.clamp(1 - clip, 1 + clip) * advantage,
    ).mean()
