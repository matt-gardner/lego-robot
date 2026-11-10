from __future__ import annotations

import argparse
from collections.abc import Generator
import copy
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import mujoco
import torch

from so101.control_model.command import ROLE_PROMPTS
from so101.control_model.flow_rl import (
    FlowTransition, clipped_policy_loss, to_device,
)
from so101.control_model.jerk_regularizer import (
    gradient_vector, group_action_boundary_backward, group_jerk_backward, normalize_gradient_components,
)
from so101.jerk_reference import ReferenceJerkMargins
from so101.control_model.modeling_entity_smolvla import (
    FrozenObservation,
    save_entity_checkpoint,
)
from so101.evaluate_entity_sim import EntityPolicy
from so101.robot_sim import (
    BeamSimulation,
    MODEL_ROOT,
    POLICY_ARM_CORRECTION_LIMIT,
    OBJECT_NAMES,
    restore_episode,
)
from so101.cloud_checkpoint import publish as publish_resume_checkpoint
from so101.cloud_checkpoint import atomic_json
from so101.rollout_evidence import RolloutRecorder, prune_evidence
from so101.resume_rng import plan_device_rng, restore_device_rng
from so101.rl_renderer import create_simulation
from so101.rl_pile_adapter import NativePileSimulation, load_native_pool
from so101.rl_difficulty import DifficultyController, HarderDataRequests
from so101.simulation_corpus import CorpusEpisode, TASKS, episode_actions, expert_bounds, load_corpus


@dataclass
class DecisionSample:
    inputs: tuple[Any, ...]
    normalized_actions: torch.Tensor
    executed_steps: int
    denoising_path: list[FlowTransition] | None = None
    attempted_steps: int | None = None
    frozen_observation: FrozenObservation | None = None


@dataclass
class SamplingRequest:
    inputs: tuple[Any, ...]
    active_shape: tuple[int, int]
    noise_scale: float
    record_noise: bool
    generator: torch.Generator | None
    frozen_observation: FrozenObservation | None = None


@dataclass
class Trajectory:
    task: str
    success: bool
    wrong_object: bool
    steps: int
    rms_jerk: float
    completion_efficiency: float
    smoothness: float
    reward: float
    decisions: list[DecisionSample]
    evidence: str | None = None
    constraint_violation: bool = False
    actions: np.ndarray | None = None
    first_success_step: int | None = None
    invalid_action_error: str | None = None
    timings: dict[str, float] | None = None


def cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    raise TypeError(f"Unsupported stored input type: {type(value)}")


def split_corpus(
    episodes: dict[str, list[CorpusEpisode]],
    validation_per_task: int,
    seed: int,
) -> tuple[dict[str, list[CorpusEpisode]], list[CorpusEpisode]]:
    rng = np.random.default_rng(seed)
    training = {}
    validation = []
    for task in TASKS:
        values = episodes[task].copy()
        rng.shuffle(values)
        if len(values) <= validation_per_task:
            raise ValueError(f"Not enough {task} episodes for validation")
        validation.extend(values[:validation_per_task])
        training[task] = values[validation_per_task:]
    return training, validation


def reset_from_episode(
    sim: BeamSimulation,
    episode: CorpusEpisode,
) -> tuple[str, str | None, list[tuple[str, str]]]:
    if "native_entry" in episode.metadata:
        return sim.reset_native(episode)
    return restore_episode(sim, episode.metadata)


def replay_gold_prefix(
    sim: BeamSimulation,
    episode: CorpusEpisode,
    target: str,
    fraction: float,
) -> tuple[int, float]:
    if fraction <= 0:
        return 0, 0.
    if "native_entry" in episode.metadata:
        raise ValueError("Reference-free native tasks do not support gold-prefix assistance")
    rows = [
        json.loads(line)
        for line in (episode.source / "frames.jsonl").read_text().splitlines()
    ]
    prefix_steps = min(
        int(len(rows) * fraction),
        max(len(rows) - 1, 0),
    )
    initial_z = float(sim.position(target)[2])
    maximum_lift = 0.
    for row in rows[:prefix_steps]:
        goal, _ = sim.bounded_policy_action(row["action"])
        sim.step(goal, "trajectory_rl_gold_prefix")
        maximum_lift = max(
            maximum_lift,
            float(sim.position(target)[2] - initial_z),
        )
    return prefix_steps, maximum_lift


def control_phase(
    sim: BeamSimulation,
    task: str,
    target: str,
    destination: str | None,
    initial_target_z: float,
    lifted_once: bool,
) -> str:
    contacts = sim.contacts(target)
    target_lift = float(sim.position(target)[2] - initial_target_z)
    if task == "grasp":
        return "lift" if contacts == {"fixed", "moving"} else "acquire"
    if task == "view":
        return "view"
    if task == "push":
        gripper_xy = sim.data.site_xpos[sim.site, :2]
        target_xy = sim.position(target)[:2]
        if contacts:
            return "push"
        if np.linalg.norm(gripper_xy - target_xy) < .06:
            return "lower"
        return "approach"
    if destination is None:
        raise ValueError("Place requires a destination")
    destination_error = float(np.linalg.norm(
        sim.position(target)[:2] - sim.position(destination)[:2]
    ))
    if contacts == {"fixed", "moving"}:
        if target_lift < .025 and destination_error > .06:
            return "lift"
        if destination_error > .055:
            return "carry"
        if target_lift > .025:
            return "lower"
        return "release"
    return "release" if lifted_once else "acquire"


def successful_step(
    sim: BeamSimulation,
    task: str,
    target: str,
    destination: str | None,
    initial_positions: dict[str, np.ndarray],
    lifted_once: bool,
    executed: int,
) -> bool:
    target_lift = float(
        sim.position(target)[2] - initial_positions[target][2]
    )
    if task == "grasp":
        return (
            target_lift >= .05
            and sim.contacts(target) == {"fixed", "moving"}
        )
    if task == "view":
        actual = sim.target_oriented_box(target)
        return (
            executed >= 30
            and bool(actual[5])
            and np.linalg.norm(np.asarray(actual[:2]) - .5) < .3
        )
    if destination is None:
        raise ValueError(f"{task} requires a destination")
    destination_error = float(np.linalg.norm(
        sim.position(target)[:2] - sim.position(destination)[:2]
    ))
    if task == "push":
        displacement = float(np.linalg.norm(
            sim.position(target)[:2] - initial_positions[target][:2]
        ))
        return (
            executed >= 30
            and destination_error <= getattr(sim, "episode_constraints", {}).get("destination_tolerance_m", .07)
            and displacement >= .02
        )
    return (
        lifted_once
        and destination_error <= getattr(sim, "episode_constraints", {}).get("destination_tolerance_m", .04)
        and sim.contacts(target) != {"fixed", "moving"}
    )


def wrong_object_lift(
    sim: BeamSimulation,
    target: str,
    initial_positions: dict[str, np.ndarray],
) -> bool:
    return any(
        name != target
        and sim.position(name)[2] - initial_positions[name][2] >= .04
        and sim.contacts(name) == {"fixed", "moving"}
        for name in getattr(sim, "object_names", OBJECT_NAMES)
    )


def trajectory_metrics(
    task: str,
    success: bool,
    steps: int,
    actions: list[np.ndarray],
    bounds: dict[str, float],
) -> tuple[float, float, float]:
    action_array = np.asarray(actions, dtype=np.float32)
    jerk = (
        np.diff(action_array[:, :5], n=3, axis=0)
        if len(action_array) > 3
        else np.zeros((0, 5), dtype=np.float32)
    )
    rms_jerk = (
        float(np.sqrt(np.mean(jerk ** 2))) if jerk.size else 0.
    )
    completion = (
        min(bounds["optimal_steps"] / max(steps, 1), 1.)
        if success else 0.
    )
    smoothness = (
        min(bounds["jerk_p10"] / max(rms_jerk, 1e-6), 1.)
        if success else 0.
    )
    return rms_jerk, completion, smoothness


def _rollout(
    sim: BeamSimulation,
    controller: EntityPolicy,
    episode: CorpusEpisode,
    rollout_seed: int,
    horizon: int,
    action_steps: int,
    bounds: dict[str, float],
    store_decisions: bool,
    start_proximity: float = 0.,
    denoising_noise_scale: float | None = None,
    capture_path: Path | None = None,
    fixed_horizon: bool = False,
    record_path_noise: bool = False,
    independent_rng: bool = False,
) -> Generator[SamplingRequest, tuple[torch.Tensor, list[FlowTransition]], Trajectory]:
    if "native_entry" in episode.metadata:
        cached = getattr(sim, "_native_simulation", None)
        root = Path(episode.metadata["native_entry"]["root"])
        if cached is None or cached.problem["root"] != root:
            if cached is not None:
                cached.close()
                sim._native_simulation = None
            cached = NativePileSimulation(
                episode, sim.model_root,
                backend=getattr(sim, "pathtrace_backend", "mujoco"),
                samples=getattr(sim, "pathtrace_samples", 16),
                threads=getattr(sim, "pathtrace_threads", 4),
            )
            sim._native_simulation = cached
        sim = cached
    return (yield from _rollout_on_sim(
        sim, controller, episode, rollout_seed, horizon, action_steps, bounds,
        store_decisions, start_proximity, denoising_noise_scale, capture_path,
        fixed_horizon, record_path_noise, independent_rng,
    ))


def _rollout_on_sim(
    sim, controller, episode, rollout_seed, horizon, action_steps, bounds,
    store_decisions, start_proximity=0., denoising_noise_scale=None, capture_path=None,
    fixed_horizon=False, record_path_noise=False, independent_rng=False,
):
    rollout_start = time.monotonic()
    target, destination, entities = reset_from_episode(sim, episode)
    initial_positions = {
        name: sim.position(name).copy() for name in getattr(sim, "object_names", OBJECT_NAMES)
    }
    prefix_steps, prefix_maximum_lift = replay_gold_prefix(
        sim,
        episode,
        target,
        start_proximity,
    )
    controller.reset(sim, rollout_seed, zero_noise=False)
    generator = None
    if independent_rng:
        generator = torch.Generator(device=controller.device)
        generator.set_state(
            torch.cuda.get_rng_state() if controller.device == "cuda"
            else torch.random.get_rng_state()
        )
    reset_seconds = time.monotonic() - rollout_start
    observation_seconds = 0.
    inference_seconds = 0.
    simulation_seconds = 0.
    decisions = []
    executed_actions = []
    executed = 0
    stable = 0
    lifted_once = prefix_maximum_lift >= .03
    success = False
    first_success_step = None
    wrong_object = False
    invalid_action_error = None
    recorder = (RolloutRecorder(capture_path, sim, rollout_seed, str(episode.source), start_proximity)
                if capture_path is not None else None)
    avoid_obstacles = episode.metadata.get("constraints", {}).get("avoid_obstacles", False)
    constraint_violation = bool(avoid_obstacles and sim.obstacle_contact_events)
    while executed < horizon and (fixed_horizon or not success and not wrong_object) and not constraint_violation:
        phase = control_phase(
            sim,
            episode.task,
            target,
            destination,
            float(initial_positions[target][2]),
            lifted_once,
        )
        observation_start = time.monotonic()
        inputs = controller.command_inputs(
            sim, episode.task, entities, phase
        )
        observation_seconds += time.monotonic() - observation_start
        inference_start = time.monotonic()
        controller.policy.eval()
        denoising_path = None
        frozen_observation = None
        if denoising_noise_scale is None:
            with torch.no_grad():
                normalized = controller.policy.model.sample_actions_entity(
                    *inputs
                )[0][0].detach()
        else:
            request = SamplingRequest(
                inputs,
                (min(action_steps, horizon - executed, controller.policy.config.chunk_size), 6),
                denoising_noise_scale, record_path_noise, generator,
            )
            sampled, denoising_path = yield request
            frozen_observation = request.frozen_observation
            normalized = sampled[0].detach()
        actions = (
            normalized[:, :6].float().cpu().numpy()
            * controller.action_std
            + controller.action_mean
        )
        inference_seconds += time.monotonic() - inference_start
        chunk_steps = 0
        chunk_attempts = 0
        invalid = False
        chunk_length = min(len(actions), action_steps, horizon - executed)
        history_steps = controller.config.history_steps
        for action in actions[:chunk_length]:
            simulation_start = time.monotonic()
            chunk_attempts += 1
            try:
                goal, _ = sim.bounded_policy_action(action)
            except ValueError as error:
                invalid = True
                invalid_action_error = str(error)
                print(json.dumps({
                    "event": "policy_action_rejected", "command": episode.task,
                    "source": str(episode.source), "seed": rollout_seed,
                    "executed_steps": executed, "error": invalid_action_error,
                }), flush=True)
                break
            sim.step(goal, "trajectory_rl")
            encoded = sim.encode(goal)
            simulation_seconds += time.monotonic() - simulation_start
            # Only the trailing history survives until the next policy decision.
            if chunk_steps >= chunk_length - history_steps:
                observation_start = time.monotonic()
                controller.observe(sim, encoded)
                observation_seconds += time.monotonic() - observation_start
            executed_actions.append(encoded)
            if recorder is not None:
                recorder.step(sim, encoded)
            executed += 1
            chunk_steps += 1
            target_lift = float(
                sim.position(target)[2] - initial_positions[target][2]
            )
            lifted_once |= target_lift >= .03
            wrong_object = wrong_object or wrong_object_lift(
                sim, target, initial_positions
            )
            if isinstance(sim, NativePileSimulation):
                success, stable = sim.native_success(executed, stable)
            else:
                condition = successful_step(
                    sim, episode.task, target, destination, initial_positions, lifted_once, executed,
                )
                stable = stable + 1 if condition else 0
                success = stable >= sim.fps
            if success and first_success_step is None:
                first_success_step = executed
            constraint_violation = bool(avoid_obstacles and sim.obstacle_contact_events)
            if constraint_violation:
                success = False
            if constraint_violation or not fixed_horizon and (success or wrong_object):
                break
        if store_decisions and (chunk_steps or denoising_path is not None):
            decisions.append(DecisionSample(
                cpu_copy(inputs),
                normalized.cpu(),
                chunk_steps,
                denoising_path,
                chunk_attempts,
                frozen_observation,
            ))
        if invalid or chunk_steps == 0:
            break
    curriculum_bounds = dict(bounds)
    curriculum_bounds["optimal_steps"] = max(
        1.,
        bounds["optimal_steps"] * (1 - start_proximity),
    )
    if fixed_horizon and (executed < horizon or wrong_object):
        success = False
    rms_jerk, completion, smoothness = trajectory_metrics(
        episode.task,
        success,
        first_success_step if fixed_horizon and first_success_step is not None else executed,
        executed_actions,
        curriculum_bounds,
    )
    if "native_entry" in episode.metadata:
        smoothness = 1. / (1. + rms_jerk) if success else 0.
    result = Trajectory(
        episode.task,
        success,
        wrong_object,
        executed,
        rms_jerk,
        completion,
        smoothness,
        0.,
        decisions,
    )
    result.invalid_action_error = invalid_action_error
    result.constraint_violation = constraint_violation
    result.first_success_step = first_success_step
    result.timings = {
        "reset_and_prefix_seconds": reset_seconds,
        "observation_seconds": observation_seconds,
        "inference_seconds": inference_seconds,
        "physics_step_seconds": simulation_seconds,
        "total_seconds": time.monotonic() - rollout_start,
    }
    if store_decisions:
        result.actions = np.asarray(executed_actions, dtype=np.float32)
    if recorder is not None:
        result.evidence = recorder.finish(sim, {
            key: value for key, value in serialized_trajectory(result).items()
            if key not in ("reward", "evidence")
        })
    return result


def rollout(
    sim: BeamSimulation,
    controller: EntityPolicy,
    episode: CorpusEpisode,
    rollout_seed: int,
    horizon: int,
    action_steps: int,
    bounds: dict[str, float],
    store_decisions: bool,
    start_proximity: float = 0.,
    denoising_noise_scale: float | None = None,
    capture_path: Path | None = None,
    fixed_horizon: bool = False,
    record_path_noise: bool = False,
) -> Trajectory:
    iterator = _rollout(
        sim, controller, episode, rollout_seed, horizon, action_steps, bounds,
        store_decisions, start_proximity, denoising_noise_scale, capture_path,
        fixed_horizon, record_path_noise,
    )
    try:
        request = next(iterator)
        while True:
            with torch.no_grad():
                sampled = controller.policy.model.sample_actions_entity_sde(
                    request.inputs, request.noise_scale, request.active_shape,
                    **({"record_noise": True} if request.record_noise else {}),
                )
            request = iterator.send(sampled)
    except StopIteration as finished:
        return finished.value
    finally:
        iterator.close()


def batch_inputs(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values, dim=0)
    if isinstance(first, (tuple, list)):
        if any(len(value) != len(first) for value in values):
            raise ValueError("Batched policy inputs have inconsistent structures")
        result = [batch_inputs([value[index] for value in values])
                  for index in range(len(first))]
        return tuple(result) if isinstance(first, tuple) else result
    raise TypeError(f"Unsupported batched policy input: {type(first)}")


def activate_render_context(sim: BeamSimulation) -> None:
    renderer = sim.renderer
    if renderer is None or renderer._gl_context is None or renderer._mjr_context is None:
        raise ValueError("Batched visual rollouts require a live GL renderer per simulator")
    # MuJoCo's Renderer exposes no public context API; restore its offscreen FBO.
    renderer._gl_context.make_current()
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, renderer._mjr_context)


def rollout_group(
    sims: list[BeamSimulation],
    controller: EntityPolicy,
    episode: CorpusEpisode,
    seeds: list[int],
    horizon: int,
    action_steps: int,
    bounds: dict[str, float],
    start_proximity: float,
    noise_scale: float,
    record_path_noise: bool,
    capture_path: Path | None = None,
) -> list[Trajectory]:
    if len(sims) != len(seeds):
        raise ValueError("Batched rollouts require one independent simulator per trajectory")
    iterators = [
        _rollout(
            sim, copy.copy(controller), episode, seed, horizon, action_steps,
            bounds, True, start_proximity, noise_scale,
            capture_path if index == 0 else None,
            record_path_noise=record_path_noise, independent_rng=True,
        )
        for index, (sim, seed) in enumerate(zip(sims, seeds, strict=True))
    ]
    results: dict[int, Trajectory] = {}
    pending: dict[int, SamplingRequest] = {}
    try:
        for index, iterator in enumerate(iterators):
            try:
                activate_render_context(sims[index])
                pending[index] = next(iterator)
            except StopIteration as finished:
                results[index] = finished.value
        while pending:
            indices = list(pending)
            requests = list(pending.values())
            first = requests[0]
            if any(request.active_shape != first.active_shape for request in requests):
                raise ValueError("Batched trajectories reached inconsistent action horizons")
            generators = []
            for request in requests:
                if request.generator is None:
                    raise ValueError("Batched trajectory lost its independent sampling RNG")
                generators.append(request.generator)
            with torch.no_grad():
                combined_inputs = batch_inputs([request.inputs for request in requests])
                replay_inputs = (
                    to_device(combined_inputs, "cpu")
                    if getattr(controller.policy.model, "rl_train_full_model", False) else None
                )
                def capture(pad, cache):
                    for row, request in enumerate(requests):
                        request.frozen_observation = FrozenObservation.from_row(
                            pad, cache, row, replay_inputs,
                        )

                actions, path = controller.policy.model.sample_actions_entity_sde(
                    combined_inputs,
                    noise_scale, first.active_shape, record_noise=record_path_noise,
                    generators=generators,
                    capture=capture,
                )
            following = {}
            for row, index in enumerate(indices):
                individual_path = [
                    FlowTransition(
                        item.latent[row:row + 1], item.next_latent[row:row + 1],
                        item.old_log_prob[row:row + 1], item.time, item.step_size,
                        item.noise_scale, item.active_shape,
                        item.noise[row:row + 1] if item.noise is not None else None,
                    )
                    for item in path
                ]
                try:
                    activate_render_context(sims[index])
                    following[index] = iterators[index].send(
                        (actions[row:row + 1], individual_path)
                    )
                except StopIteration as finished:
                    results[index] = finished.value
            pending = following
        return [results[index] for index in range(len(iterators))]
    finally:
        for iterator in iterators:
            iterator.close()


def load_auxiliary_view(path: Path, validation: list[CorpusEpisode]):
    manifest = json.loads(path.read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "complete"
            or manifest.get("kind") != "harder-view-references"):
        raise ValueError("Auxiliary VIEW manifest is not a completed reference collection")
    heldout = {
        (item.metadata["initial"]["seed"], item.metadata["initial"].get("scene_version", 1))
        for item in validation
    }
    splits = {"train": [], "development": []}
    families: dict[str, str] = {}
    seen = set()
    for entry in manifest["episodes"]:
        source = Path(entry["path"])
        if not source.is_absolute() or str(source) in seen or entry["split"] not in splits:
            raise ValueError("Invalid auxiliary VIEW episode path or split")
        seen.add(str(source))
        for name in ("episode.json", "frames.jsonl"):
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != entry["sha256"][name]:
                raise ValueError(f"Auxiliary VIEW artifact hash mismatch: {source / name}")
        metadata = json.loads((source / "episode.json").read_text())
        initial = metadata["initial"]
        family = (initial["seed"], initial.get("scene_version", 1))
        expected_family = hashlib.sha256(json.dumps(list(family)).encode()).hexdigest()
        if (family in heldout or metadata["task"] != "view"
                or metadata["outcome"] != "success"
                or not metadata["reference_replay"]["success"]
                or not np.isfinite(metadata["reference_replay"]["maximum_qpos_error"])
                or metadata["reference_replay"]["maximum_qpos_error"] > 1e-7
                or entry["scene_family_id"] != expected_family
                or metadata["scene_family_id"] != expected_family
                or metadata["split"] != entry["split"]):
            raise ValueError("Auxiliary VIEW admission or held-out exclusion failed")
        previous = families.setdefault(expected_family, entry["split"])
        if previous != entry["split"]:
            raise ValueError("Auxiliary VIEW scene family crosses train/development")
        splits[entry["split"]].append(CorpusEpisode(source, metadata))
    if not all(splits.values()):
        raise ValueError("Auxiliary VIEW requires both train and development references")
    return splits, hashlib.sha256(path.read_bytes()).hexdigest()


def historical_auxiliary_training(log: dict, validation: list[CorpusEpisode],
                                  current: dict) -> list[CorpusEpisode]:
    admissions = [event["previous"] for event in log.get("auxiliary_view_transitions", [])]
    if log.get("auxiliary_view"):
        admissions.append(log["auxiliary_view"])
    references = {}
    for previous in admissions:
        splits, digest = load_auxiliary_view(Path(previous["manifest"]), validation)
        if digest != previous["sha256"]:
            raise ValueError("Previously admitted auxiliary VIEW manifest changed")
        for old_split, new_split in (("train", "development"), ("development", "train")):
            families = {item.metadata["scene_family_id"] for item in splits[old_split]}
            if any(item.metadata["scene_family_id"] in families for item in current[new_split]):
                raise ValueError("Auxiliary VIEW transition crosses historical split families")
        references.update((str(item.source), item) for item in splits["train"])
    return list(references.values())


def record_auxiliary_admission(log: dict, admission: dict, previous_sha: str | None,
                               step: int) -> None:
    previous = log.get("auxiliary_view")
    if previous is not None and previous != admission:
        if previous_sha != previous["sha256"]:
            raise ValueError("Auxiliary VIEW transition requires the exact previous manifest SHA256")
        log.setdefault("auxiliary_view_transitions", []).append({
            "step": step, "previous": previous, "current": admission,
        })
    elif previous is None and previous_sha is not None:
        raise ValueError("Auxiliary VIEW transition requires an existing admission")
    log["auxiliary_view"] = admission


def assign_group_rewards(
    trajectories: list[Trajectory],
    success_weight: float = 1.,
    completion_weight: float = .5,
    smoothness_weight: float = 1.,
    reference_free_completion: bool = False,
) -> None:
    completion = completion_values(trajectories, reference_free_completion)
    for item, score in zip(trajectories, completion, strict=True):
        item.reward = (
            success_weight
            + completion_weight * float(score)
            + smoothness_weight * item.smoothness
            if item.success else 0.
        )


def completion_values(trajectories: list[Trajectory], reference_free: bool) -> np.ndarray:
    if not reference_free:
        return np.asarray([item.completion_efficiency for item in trajectories], dtype=np.float32)
    longest = max((item.steps for item in trajectories if item.success), default=1)
    if longest <= 0:
        raise ValueError("Successful completion requires a positive rollout length")
    return np.asarray([
        1. - item.steps / longest if item.success else 0.
        for item in trajectories
    ], dtype=np.float32)


def normalized_component_advantage(
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    advantages = np.zeros_like(values, dtype=np.float32)
    selected = np.ones(len(values), dtype=bool) if mask is None else mask
    selected_values = values[selected]
    if len(selected_values) <= 1 or selected_values.std() < 1e-6:
        return advantages
    advantages[selected] = (
        selected_values - selected_values.mean()
    ) / (selected_values.std() + 1e-6)
    return advantages


def group_advantages(
    trajectories: list[Trajectory],
    success_weight: float,
    completion_weight: float,
    smoothness_weight: float,
    reference_free_completion: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    success = np.asarray([
        item.success for item in trajectories
    ], dtype=np.float32)
    successful = success.astype(bool)
    components = {
        "success": normalized_component_advantage(success),
        "completion": normalized_component_advantage(
            completion_values(trajectories, reference_free_completion),
            successful,
        ),
        "smoothness": normalized_component_advantage(
            np.asarray([
                item.smoothness for item in trajectories
            ], dtype=np.float32),
            successful,
        ),
    }
    combined = (
        success_weight * components["success"]
        + completion_weight * components["completion"]
        + smoothness_weight * components["smoothness"]
    )
    return np.clip(combined, -3, 3), components


def serialized_trajectory(item: Trajectory) -> dict[str, Any]:
    return {
        "success": item.success,
        "wrong_object": item.wrong_object,
        "steps": item.steps,
        "completion_efficiency": item.completion_efficiency,
        "smoothness": item.smoothness,
        "rms_jerk": item.rms_jerk,
        "reward": item.reward,
        "evidence": item.evidence,
        "constraint_violation": item.constraint_violation,
        "invalid_action_error": item.invalid_action_error,
        "timings": item.timings,
    }


def save_delta_checkpoint(
    path: Path,
    controller: EntityPolicy,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    state = {
        name: parameter.detach().cpu()
        for name, parameter in controller.policy.named_parameters()
        if parameter.requires_grad
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "step": step,
        "model": state,
        "optimizer": optimizer.state_dict(),
    }, temporary)
    temporary.replace(path)


def save_resume_state(
    output: Path,
    controller: EntityPolicy,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    log: dict[str, Any],
    checkpoint_generations: int = 0,
) -> None:
    publish_resume_checkpoint(output, {
        "step": log["steps"][-1]["step"],
        "model": {
            name: parameter.detach().cpu()
            for name, parameter in controller.policy.named_parameters()
            if parameter.requires_grad
        },
        "optimizer": optimizer.state_dict(),
        "log": log,
        "rng": {
            "backend": controller.device,
            "sampling": rng.bit_generator.state,
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "device": (
                torch.mps.get_rng_state()
                if controller.device == "mps"
                else torch.cuda.get_rng_state()
                if controller.device == "cuda"
                else None
            ),
        },
    }, checkpoint_generations)


def restore_resume_state(
    state: dict[str, Any],
    controller: EntityPolicy,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    learning_rate: float,
    allow_device_rng_migration: bool = False,
    device_rng_seed: int | None = None,
) -> dict[str, Any]:
    if state["step"] != state["log"]["steps"][-1]["step"]:
        raise ValueError("Resume model and log steps do not match")
    device_rng_plan = plan_device_rng(
        state, controller.device, allow_device_rng_migration, device_rng_seed,
    )
    parameters = dict(controller.policy.named_parameters())
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if set(state["model"]) != expected:
        raise ValueError("Resume trainable parameters differ; start a fresh RL run when changing scope")
    with torch.no_grad():
        for name, value in state["model"].items():
            parameters[name].copy_(value.to(parameters[name].device))
    optimizer.load_state_dict(state["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    rng.bit_generator.state = state["rng"]["sampling"]
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.random.set_rng_state(state["rng"]["torch"].cpu())
    restore_device_rng(device_rng_plan, state, torch)
    return state["log"]


def trajectory_backward(
    controller: EntityPolicy,
    trajectory: Trajectory,
    coefficient: float,
    policy_clip: float = .2,
) -> float:
    if not trajectory.decisions or coefficient == 0:
        return 0.
    total = 0.
    controller.policy.train()
    per_decision = coefficient / len(trajectory.decisions)
    for decision in trajectory.decisions:
        inputs = to_device(decision.inputs, controller.device)
        if decision.denoising_path is not None:
            controller.policy.eval()
            if not decision.denoising_path:
                raise ValueError("Policy gradients require a stochastic path")
            normalizer = (
                len(decision.denoising_path)
                * decision.denoising_path[0].latent[0].numel()
            )
            log_probs = controller.policy.model.denoising_log_probs_entity(
                inputs, decision.denoising_path,
                **({"frozen_observation": decision.frozen_observation}
                   if decision.frozen_observation is not None else {}),
            )
            for log_prob, transition in zip(
                log_probs, decision.denoising_path, strict=True
            ):
                loss = clipped_policy_loss(
                    log_prob, transition.old_log_prob.to(controller.device),
                    coefficient, policy_clip,
                )
                scaled_loss = loss / normalizer
                scaled_loss.backward()
                total += float(scaled_loss.detach())
            continue
        target_actions = decision.normalized_actions.unsqueeze(0).to(
            controller.device
        )
        losses = controller.policy.model.forward_entity(
            *inputs[:5],
            target_actions,
            *inputs[5:],
        )[0][:, :decision.executed_steps, :6]
        loss = losses.mean()
        (loss * per_decision).backward()
        total += float(loss.detach()) / len(trajectory.decisions)
    return total


def full_model_gradient_probes(policy):
    probes = {}
    for group, pattern in {
        "vision": "vision_model.", "language_backbone": "text_model.layers.",
        "history": "history_motion_proj.", "entity": "entity_numeric_proj.",
        "action_expert": "lm_expert.", "action_input": "action_in_proj.",
        "action_output": "action_out_proj.",
    }.items():
        for name, parameter in policy.named_parameters():
            if pattern not in name or parameter.grad is None:
                continue
            gradient = parameter.grad.detach().flatten()
            index = int(gradient.abs().argmax())
            if gradient[index] != 0:
                if not torch.isfinite(gradient[index]):
                    raise ValueError(f"Nonfinite full-model gradient in {group}")
                probes[group] = (name, parameter, index, float(parameter.detach().flatten()[index]))
                break
        if group not in probes:
            raise ValueError(f"Full-model RL has no effective gradient through {group}")
    return probes


def trainable_parameters(
    controller: EntityPolicy,
    scope: str = "all",
) -> list[torch.nn.Parameter]:
    if scope not in ("all", "action-projections"):
        raise ValueError(f"Unknown RL trainable scope: {scope}")
    controller.policy.model.rl_train_full_model = scope == "all"
    if scope == "all":
        # AdamW needs FP32 master weights: small updates round away in BF16 parameters.
        controller.policy.float()
        parameters = list(controller.policy.parameters())
        for parameter in parameters:
            parameter.requires_grad_(True)
        return parameters
    for parameter in controller.policy.parameters():
        parameter.requires_grad_(False)
    parameters = []
    for module in (
        controller.policy.model.action_in_proj,
        controller.policy.model.action_out_proj,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            parameters.append(parameter)
    return parameters


def policy_optimizer_step(
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    accepted_groups: int,
    gradient_clip: float,
) -> dict[str, Any]:
    if not accepted_groups:
        return {
            "gradient_norm": 0.,
            "policy_gradient_norm": 0.,
            "replay_gradient_norm": 0.,
            "optimizer_update_applied": False,
            "optimizer_skip_reason": "no_accepted_groups",
        }
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(accepted_groups)
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(
        parameters, gradient_clip
    ))
    if not math.isfinite(gradient_norm):
        raise FloatingPointError(
            "Non-finite gradients; latest completed checkpoint preserved"
        )
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in parameters):
        raise FloatingPointError(
            "Non-finite policy parameters; latest completed checkpoint preserved"
        )
    return {
        "optimizer_update_applied": True,
        "gradient_norm": gradient_norm,
        "policy_gradient_norm": gradient_norm,
        "replay_gradient_norm": 0.,
    }


def validation_metrics(
    sim: BeamSimulation,
    controller: EntityPolicy,
    episodes: list[CorpusEpisode],
    bounds: dict[str, dict[str, float]],
    seed: int,
    horizon: int,
    action_steps: int,
    denoising_noise_scale: float | None = None,
) -> dict[str, Any]:
    trajectories = []
    for index, episode in enumerate(episodes):
        started = time.monotonic()
        trajectory = rollout(
            sim,
            controller,
            episode,
            seed + index,
            horizon,
            action_steps,
            bounds[episode.task],
            False,
            denoising_noise_scale=denoising_noise_scale,
        )
        trajectories.append(trajectory)
        print(json.dumps({
            "event": "validation_rollout_completed",
            "completed": index + 1, "total": len(episodes),
            "task": episode.task, "source": str(episode.source),
            "seconds": time.monotonic() - started, "success": trajectory.success,
        }), flush=True)
    successful = [item for item in trajectories if item.success]
    return {
        "sampler": "ode" if denoising_noise_scale is None else "sde",
        "denoising_noise_scale": denoising_noise_scale,
        "episodes": len(trajectories),
        "success_rate": sum(item.success for item in trajectories)
        / len(trajectories),
        "completion_efficiency": (
            float(np.mean([
                item.completion_efficiency for item in successful
            ]))
            if successful else 0.
        ),
        "rms_jerk": float(np.mean([
            item.rms_jerk for item in trajectories
        ])),
        "wrong_object_rate": sum(
            item.wrong_object for item in trajectories
        ) / len(trajectories),
        "rollouts": [
            {"task": episode.task, "source": str(episode.source), "seed": seed + index,
             **serialized_trajectory(item)}
            for index, (episode, item) in enumerate(zip(episodes, trajectories, strict=True))
        ],
        "by_task": {
            task: {
                "success_rate": float(np.mean([
                    item.success
                    for item in trajectories
                    if item.task == task
                ])),
                "completion_efficiency": (
                    float(np.mean([
                        item.completion_efficiency
                        for item in trajectories
                        if item.task == task and item.success
                    ]))
                    if any(
                        item.task == task and item.success
                        for item in trajectories
                    ) else 0.
                ),
                "rms_jerk": float(np.mean([
                    item.rms_jerk
                    for item in trajectories
                    if item.task == task
                ])),
            }
            for task in TASKS
        },
    }


def write_log(output: Path, payload: dict[str, Any]) -> None:
    temporary = output / "trajectory_rl.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(output / "trajectory_rl.json")


def curriculum_key(task: str, has_obstacles: bool) -> str:
    return f"obstacle_{task}" if has_obstacles else task


def curriculum_maximum(
    task: str,
    has_obstacles: bool,
    args: argparse.Namespace,
) -> float:
    if has_obstacles:
        return args.obstacle_curriculum_max
    if task == "push":
        return args.push_curriculum_max
    if task == "place":
        return args.place_curriculum_max
    if task == "grasp":
        return args.grasp_curriculum_max
    if task == "view":
        return args.view_curriculum_max
    return 0.


def initialize_mastery_curriculum(
    log: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    state = log.setdefault("assistance_curriculum", {
        "mode": "mastery",
        "decrement": args.curriculum_decrement,
        "mastery_groups": args.curriculum_mastery_groups,
        "minimum": args.curriculum_min,
        "tasks": {},
        "events": [],
        "checks": [],
    })
    state.setdefault("checks", [])
    if state["mode"] != "mastery":
        raise ValueError(
            "Cannot resume a non-mastery assistance curriculum in "
            "mastery mode"
        )
    expected = {
        "decrement": args.curriculum_decrement,
        "mastery_groups": args.curriculum_mastery_groups,
        "minimum": args.curriculum_min,
    }
    for name, value in expected.items():
        if state[name] != value:
            raise ValueError(
                f"Resume curriculum {name} is {state[name]}, not {value}"
            )
    return state


def mastery_assistance(
    state: dict[str, Any],
    key: str,
    maximum: float,
) -> float:
    task_state = state["tasks"].setdefault(key, {
        "level": maximum,
        "perfect_group_streak": 0,
        "groups_at_level": 0,
        "rollouts_at_level": 0,
        "successes_at_level": 0,
    })
    return float(task_state["level"])


def record_mastery_group(
    state: dict[str, Any],
    key: str,
    successes: int,
    group_size: int,
) -> bool:
    task_state = state["tasks"][key]
    task_state["groups_at_level"] += 1
    task_state["rollouts_at_level"] += group_size
    task_state["successes_at_level"] += successes
    if successes == group_size:
        task_state["perfect_group_streak"] += 1
    else:
        task_state["perfect_group_streak"] = 0
    return bool(
        task_state["perfect_group_streak"] >= state["mastery_groups"]
        and task_state["level"] > state["minimum"]
    )


def resolve_mastery_check(
    state: dict[str, Any],
    key: str,
    successes: int,
    total: int,
    step: int,
    group_size: int,
) -> dict[str, Any] | None:
    task_state = state["tasks"][key]
    check = {
        "step": step,
        "task": key,
        "level": float(task_state["level"]),
        "successes": successes,
        "total": total,
        "pass_rate": successes / total,
    }
    state["checks"].append(check)
    if successes != total:
        task_state["perfect_group_streak"] = 0
        return None
    previous = float(task_state["level"])
    current = round(max(
        float(state["minimum"]),
        previous - float(state["decrement"]),
    ), 6)
    event = {
        "step": step,
        "task": key,
        "previous": previous,
        "current": current,
        "mastery_groups": state["mastery_groups"],
        "mastery_rollouts": state["mastery_groups"] * group_size,
    }
    state["events"].append(event)
    task_state.update({
        "level": current,
        "perfect_group_streak": 0,
        "groups_at_level": 0,
        "rollouts_at_level": 0,
        "successes_at_level": 0,
    })
    return event


def recover_zero_success_group(state: dict[str, Any], key: str, successes: int,
                               maximum: float, step: int) -> dict[str, Any] | None:
    task_state = state["tasks"][key]
    previous = float(task_state["level"])
    if successes or previous >= maximum:
        return None
    current = round(min(maximum, previous + float(state["decrement"])), 6)
    event = {
        "step": step, "task": key, "previous": previous, "current": current,
        "kind": "zero_success_recovery",
    }
    state["events"].append(event)
    task_state.update(level=current, perfect_group_streak=0, groups_at_level=0,
                      rollouts_at_level=0, successes_at_level=0)
    return event


def record_target_success_group(
    state: dict[str, Any], key: str, successes: int, total: int,
    maximum: float, step: int,
) -> dict[str, Any] | None:
    task = state["tasks"][key]
    window = task.setdefault("success_window", [])
    window.append({"successes": successes, "total": total})
    del window[:-4]
    task["groups_at_level"] += 1
    task["rollouts_at_level"] += total
    task["successes_at_level"] += successes
    if len(window) < 4:
        return None
    rate = sum(row["successes"] for row in window) / sum(
        row["total"] for row in window
    )
    task["rolling_success_rate"] = rate
    previous = float(task["level"])
    direction = -1 if rate > .6 else 1 if rate < .4 else 0
    current = round(min(maximum, max(
        float(state["minimum"]),
        previous + direction * float(state["decrement"]),
    )), 6)
    if current == previous:
        return None
    event = {
        "step": step, "task": key, "previous": previous, "current": current,
        "kind": "target_success", "success_rate": rate,
        "successes": sum(row["successes"] for row in window),
        "total": sum(row["total"] for row in window),
    }
    state["events"].append(event)
    task.update(level=current, perfect_group_streak=0, groups_at_level=0,
                rollouts_at_level=0, successes_at_level=0, success_window=[])
    return event


def record_solved_problem(
    bucket: dict[str, Any],
    episode: CorpusEpisode,
    start_proximity: float,
    successes: int,
    total: int,
    step: int,
) -> None:
    if total <= 0 or successes != total:
        return
    key = json.dumps([str(episode.source), start_proximity])
    bucket.setdefault(key, {
        "source": str(episode.source),
        "task": episode.task,
        "start_proximity": start_proximity,
        "solved_at_step": step,
    })


def initialize_solved_problems(
    log: dict[str, Any],
    training: dict[str, list[CorpusEpisode]],
) -> dict[str, Any]:
    if "solved_problems" not in log:
        bucket: dict[str, Any] = {}
        by_source = {
            str(item.source): item
            for episodes in training.values() for item in episodes
        }
        for row in log["steps"]:
            for group in row["groups"]:
                episode = by_source.get(group["source"])
                if episode is not None:
                    record_solved_problem(
                        bucket, episode, group["start_proximity"],
                        group["successes"], len(group["trajectories"]),
                        row["step"],
                    )
        log["solved_problems"] = bucket
    return log["solved_problems"]


def sample_solved_problem(
    bucket: dict[str, Any],
    rng: np.random.Generator,
    step: int,
) -> dict[str, Any] | None:
    eligible = [
        item for item in bucket.values() if item["solved_at_step"] < step
    ]
    return eligible[int(rng.integers(len(eligible)))] if eligible else None


def admit_difficulty_pool(difficulty, pools, excluded_families, training_by_source,
                          key, level, path, digest):
    identity = f"{key}:{level}"
    admitted = difficulty.state.setdefault("pool_manifests", {})
    previous = admitted.get(identity)
    if previous is not None and previous["sha256"] != digest:
        return False
    if (key, level) not in pools:
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Admitted native pool changed: {path}")
        pool = next((
            pool for (known_key, known_level), pool in pools.items()
            if admitted[f"{known_key}:{known_level}"]["sha256"] == digest
        ), None)
        if pool is None:
            pool = load_native_pool(path, excluded_families)
        difficulty.admit(key, level, digest)
        admitted[identity] = {"slice": key, "level": level, "path": str(path), "sha256": digest}
        pools[(key, level)] = pool
        training_by_source.update(
            (str(item.source), item) for episodes in pool.values() for item in episodes
        )
    return True


def refresh_difficulty_pools(difficulty, requests, pools, excluded_families, training_by_source, log,
                             continuous_manifest=None, output=None):
    for previous in list(difficulty.state.get("pool_manifests", {}).values()):
        admit_difficulty_pool(
            difficulty, pools, excluded_families, training_by_source, previous["slice"],
            previous["level"], Path(previous["path"]), previous["sha256"],
        )
    for result in requests.results():
        identity = result["request_id"]
        statuses = log.setdefault("harder_data_requests", {})
        if statuses.get(identity) != result:
            statuses[identity] = result
            print(json.dumps({"event": "harder_data_request", **result}), flush=True)
            if result["status"] in ("blocked", "rejected"):
                difficulty.failed_request(result)
        if result["status"] != "complete":
            continue
        selected = admit_difficulty_pool(
            difficulty, pools, excluded_families, training_by_source, result["slice"],
            result["level"], Path(result["manifest_path"]), result["sha256"],
        )
        if not selected:
            statuses[identity]["admission"] = "not-selected-existing-immutable-tier"
    if continuous_manifest is None:
        return
    content = continuous_manifest.read_bytes()
    source_sha = hashlib.sha256(content).hexdigest()
    if source_sha == difficulty.state.get("continuous_source_sha256"):
        return
    manifest = json.loads(content)
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "lego-pile-task-pool"
            or manifest.get("status") != "complete" or manifest.get("expert_references") is not False):
        raise ValueError("Continuous native source is not an admitted problem pool")
    tiers = defaultdict(list)
    for scene in manifest["scenes"]:
        count = scene["tags"]["piece_count"]
        if type(count) is not int or not 2 <= count <= 128:
            raise ValueError("Continuous pool piece counts exceed supported generator bounds")
        tiers[max(1, (count - 3) // 2)].append(scene)
    directory = output / "difficulty-pools"
    directory.mkdir(exist_ok=True)
    for level, scenes in sorted(tiers.items()):
        keys = (*TASKS, *(f"obstacle_{task}" for task in TASKS))
        if all(f"{key}:{level}" in difficulty.state.get("pool_manifests", {}) for key in keys):
            continue
        snapshot = {
            "schema_version": 1, "kind": "lego-pile-task-pool", "status": "complete",
            "expert_references": False, "source_sha256": source_sha,
            "scenes": sorted(scenes, key=lambda scene: scene["scene_id"])[:32],
        }
        name = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
        path = directory / f"{name}.json"
        if not path.exists():
            atomic_json(path, snapshot)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for key in keys:
            admit_difficulty_pool(
                difficulty, pools, excluded_families, training_by_source, key, level, path, digest,
            )
    difficulty.state["continuous_source_sha256"] = source_sha
    log.setdefault("continuous_pool_admissions", []).append({
        "source": str(continuous_manifest), "sha256": source_sha,
        "scenes": len(manifest["scenes"]), "available_tiers": sorted(tiers),
        "qualification": "hash-bound native reset/goal capability, not expert or policy solvability",
    })


def train(args: argparse.Namespace) -> None:
    automatic_difficulty = getattr(args, "automatic_difficulty", False)
    reference_free_completion = automatic_difficulty or getattr(args, "reference_free_completion", False)
    reference_jerk_fallback = automatic_difficulty or getattr(args, "reference_jerk_fallback", False)
    if automatic_difficulty and (
        args.curriculum_mode != "mastery" or args.curriculum_min != 0
        or not getattr(args, "target_success_curriculum", False)
        or getattr(args, "difficulty_catalog", None) is None
    ):
        raise ValueError("Automatic difficulty requires target-success mastery, minimum 0 and --difficulty-catalog")
    recovery_enabled = getattr(args, "recover_zero_success_groups", False)
    if recovery_enabled and args.curriculum_mode != "mastery":
        raise ValueError("Zero-success recovery requires mastery curriculum mode")
    jerk_weight = getattr(args, "jerk_regularizer_weight", 0.)
    batch_rollouts = getattr(args, "batch_rollouts", False)
    if batch_rollouts and (args.rl_estimator != "denoising-grpo" or args.device not in ("cpu", "cuda")):
        raise ValueError("Batched rollouts require denoising-grpo on CPU or CUDA")
    jerk_scale = getattr(args, "jerk_scale", 6.)
    jerk_enabled = jerk_weight > 0
    boundary_weight = getattr(args, "action_boundary_weight", 0.)
    boundary_scale = getattr(args, "action_boundary_scale", 6.)
    boundary_enabled = boundary_weight > 0
    regularizers_enabled = jerk_enabled or boundary_enabled
    if not math.isfinite(boundary_weight) or boundary_weight < 0 or not math.isfinite(boundary_scale) or boundary_scale <= 0:
        raise ValueError("Action-boundary weight must be finite and nonnegative; scale must be positive")
    if boundary_enabled and args.rl_estimator != "denoising-grpo":
        raise ValueError("Action-boundary regularization requires denoising-grpo")
    if not math.isfinite(jerk_weight) or jerk_weight < 0 or not math.isfinite(jerk_scale) or jerk_scale <= 0:
        raise ValueError("Jerk weight must be finite and nonnegative; jerk scale must be positive")
    if jerk_enabled and (args.rl_estimator != "denoising-grpo" or args.smoothness_advantage_weight != 0):
        raise ValueError("Reference jerk regularization requires denoising-grpo and smoothness advantage weight 0")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if not math.isfinite(args.gradient_clip) or args.gradient_clip <= 0:
        raise ValueError("Gradient clip must be finite and positive")
    if args.refresh_corpus and not args.resume:
        raise ValueError("--refresh-corpus requires --resume")
    if (
        not math.isfinite(args.denoising_noise_scale)
        or args.denoising_noise_scale <= 0
    ):
        raise ValueError("Denoising noise scale must be finite and positive")
    if not 0 < args.policy_clip < 1:
        raise ValueError("Policy clip must lie between zero and one")
    sampling_noise = (
        args.denoising_noise_scale
        if args.rl_estimator == "denoising-grpo" else None
    )
    if min(
        args.steps,
        args.group_size,
        args.groups_per_task,
        args.max_group_attempts_per_task,
        args.horizon,
        args.action_steps,
        args.validation_interval,
    ) <= 0:
        raise ValueError("Training counts must be positive")
    if not (
        0 <= args.push_curriculum_max <= 1
        and 0 <= args.place_curriculum_max <= 1
        and 0 <= args.grasp_curriculum_max <= 1
        and 0 <= args.view_curriculum_max <= 1
        and 0 <= args.obstacle_curriculum_max <= 1
        and 0 <= args.curriculum_min <= 1
    ):
        raise ValueError("Curriculum fractions must be within [0, 1]")
    if not 0 < args.curriculum_decrement <= 1:
        raise ValueError("Curriculum decrement must be within (0, 1]")
    if args.curriculum_mastery_groups <= 0:
        raise ValueError("Curriculum mastery groups must be positive")
    if args.curriculum_min > min(
        args.push_curriculum_max,
        args.place_curriculum_max,
    ):
        raise ValueError(
            "Curriculum minimum cannot exceed a task maximum"
        )
    if min(
        args.success_advantage_weight,
        args.completion_advantage_weight,
        args.smoothness_advantage_weight,
    ) < 0:
        raise ValueError("Advantage weights must be nonnegative")
    if (
        args.success_advantage_weight
        + args.completion_advantage_weight
        + args.smoothness_advantage_weight
    ) == 0 and not jerk_enabled:
        raise ValueError("At least one advantage weight must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output = args.output
    all_episodes = load_corpus(args.collection)
    resume_state = None
    if args.resume and (output / "latest_state.pt").exists():
        resume_state = torch.load(
            output / "latest_state.pt",
            map_location="cpu",
            weights_only=False,
        )
    if args.resume:
        if not output.is_dir():
            raise ValueError(f"Resume output does not exist: {output}")
        split_payload = json.loads(
            (output / "data_split.json").read_text()
        )
        by_source = {
            str(item.source): item
            for values in all_episodes.values()
            for item in values
        }
        training = {
            task: [by_source[source] for source in sources]
            for task, sources in split_payload["training"].items()
        }
        validation = [
            by_source[source] for source in split_payload["validation"]
        ]
        bounds = split_payload["expert_bounds"]
        if args.refresh_corpus:
            held_out = set(split_payload["validation"])
            training = {
                task: [
                    item for item in values
                    if str(item.source) not in held_out
                ]
                for task, values in all_episodes.items()
            }
            bounds = expert_bounds(training)
            split_payload.update({
                "training": {
                    task: [str(item.source) for item in values]
                    for task, values in training.items()
                },
                "expert_bounds": bounds,
            })
            (output / "data_split.json").write_text(json.dumps(
                split_payload, indent=2
            ))
    else:
        output.mkdir(parents=True, exist_ok=False)
        training, validation = split_corpus(
            all_episodes, args.validation_per_task, args.seed
        )
        bounds = expert_bounds(training)
        split_payload = {
            "training": {
                task: [str(item.source) for item in values]
                for task, values in training.items()
            },
            "validation": [str(item.source) for item in validation],
            "expert_bounds": bounds,
        }
        (output / "data_split.json").write_text(json.dumps(
            split_payload, indent=2
        ))
    auxiliary_view = None
    auxiliary_path = getattr(args, "auxiliary_view_manifest", None)
    if auxiliary_path is not None:
        auxiliary_view, auxiliary_sha = load_auxiliary_view(auxiliary_path, validation)
        if resume_state is not None:
            previous_training = historical_auxiliary_training(
                resume_state["log"], validation, auxiliary_view,
            )
            existing = {str(item.source) for item in training["view"]}
            training["view"].extend(
                item for item in previous_training if str(item.source) not in existing
            )
        existing = {str(item.source) for item in training["view"]}
        training["view"].extend(
            item for item in auxiliary_view["train"] if str(item.source) not in existing
        )
    jerk_references = (ReferenceJerkMargins(
        {task: [str(item.source) for item in episodes] for task, episodes in training.items()},
        args.horizon, 10, hold_final=False,
    ) if jerk_enabled else None)
    if jerk_references is not None:
        print(json.dumps({"event": "jerk_reference_protocol",
                          **jerk_references.protocol()}), flush=True)
    controller = EntityPolicy(args.checkpoint, args.device, True)
    if args.resume and resume_state is None:
        controller.policy.load_state_dict(torch.load(
            output / "latest_model.pt",
            map_location=args.device,
            weights_only=True,
        ))
    if sampling_noise is not None:
        controller.policy.model.vlm_with_expert.lm_expert.float()
    trainable_scope = getattr(args, "trainable_scope", "all")
    parameters = trainable_parameters(controller, trainable_scope)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=1e-6,
    )
    if args.resume and resume_state is None:
        if trainable_scope == "all":
            raise ValueError("Full-model RL resume requires latest_state.pt with optimizer/RNG and trainable scope")
        optimizer_path = output / "latest_optimizer.pt"
        if optimizer_path.exists():
            optimizer.load_state_dict(torch.load(
                optimizer_path,
                map_location=args.device,
                weights_only=False,
            )["optimizer"])
        else:
            delta_paths = sorted(output.glob("delta_step_*.pt"))
            if delta_paths:
                optimizer.load_state_dict(torch.load(
                    delta_paths[-1],
                    map_location=args.device,
                    weights_only=False,
                )["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
    renderer_options = {
        "backend": getattr(args, "rgb_renderer", "mujoco"),
        "samples": getattr(args, "pathtrace_samples", 16),
        "threads": getattr(args, "pathtrace_threads", 4),
    }
    sim = create_simulation(args.model_root, **renderer_options)
    rollout_sims = [sim]
    log: dict[str, Any] = (
        resume_state["log"]
        if resume_state is not None
        else json.loads((output / "trajectory_rl.json").read_text())
        if args.resume
        else {
            "algorithm": "trajectory group-relative flow matching",
            "group_size": args.group_size,
            "groups_per_task": args.groups_per_task,
            "advantage_weights": {
                "success": args.success_advantage_weight,
                "completion": args.completion_advantage_weight,
                "smoothness": args.smoothness_advantage_weight,
            },
            "steps": [],
            "validation": [],
            "zero_success_groups": 0,
            "groups": 0,
            "zero_success_by_task": {task: 0 for task in TASKS},
            "success_count_histogram": {
                str(index): 0 for index in range(args.group_size + 1)
            },
        }
    )
    if resume_state is not None:
        log = restore_resume_state(
            resume_state, controller, optimizer, rng, args.learning_rate,
            getattr(args, "allow_device_rng_migration", False),
            getattr(args, "device_rng_seed", None),
        )
        # Model and optimizer now own their tensors; release the CPU recovery copy.
        del resume_state
    previous_estimator = log.get("rl_estimator", "flow-mse")
    if previous_estimator != args.rl_estimator:
        log.setdefault("estimator_changes", []).append({
            "start_step": log["steps"][-1]["step"] + 1 if log["steps"] else 1,
            "previous": previous_estimator,
            "current": args.rl_estimator,
            "denoising_noise_scale": sampling_noise,
        })
    log["rl_estimator"] = args.rl_estimator
    log["trainable_scope"] = trainable_scope
    log["trainable_parameter_count"] = sum(parameter.numel() for parameter in parameters)
    if {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} != {
        id(parameter) for parameter in controller.policy.parameters() if parameter.requires_grad
    }:
        raise ValueError("Optimizer membership does not match trainable policy parameters")
    if sampling_noise is not None:
        optimization = {
            "version": ("normalized-fp32-task-pg-plus-action-regularizers-v5" if boundary_enabled
                        else "normalized-fp32-task-pg-plus-reference-jerk-v4" if jerk_enabled
                        else "normalized-raoblackwell-fp32-reward-only-v3"),
            "expert_dtype": "float32",
            "policy_loss_scale": 1. / (
                controller.policy.config.num_steps * controller.policy.config.chunk_size
                * controller.policy.config.max_action_dim
            ),
            "optimizer_update": "full-adamw",
            "kl_guard": False,
            "zero_success_groups": "discard all learning gradients",
        }
        if log.get("policy_optimization") != optimization:
            log.setdefault("policy_optimization_changes", []).append({
                "start_step": log["steps"][-1]["step"] + 1 if log["steps"] else 1,
                **optimization,
            })
        log["policy_optimization"] = optimization
    log["algorithm"] = (
        "trajectory denoising-path GRPO"
        if sampling_noise is not None
        else "trajectory group-relative flow matching"
    )
    training_objective = "task-reward-plus-reference-jerk" if jerk_enabled else "reward-only"
    if boundary_enabled:
        training_objective = (training_objective + "-plus-action-boundary" if jerk_enabled
                              else "task-reward-plus-action-boundary")
    if log.get("training_objective") != training_objective:
        log.setdefault("training_objective_changes", []).append({
            "start_step": log["steps"][-1]["step"] + 1 if log["steps"] else 1,
            "previous": log.get(
                "training_objective", "reward-plus-supervised" if log["steps"] else None
            ),
            "current": training_objective,
        })
    log["training_objective"] = training_objective
    if log.get("auxiliary_view") and auxiliary_view is None:
        raise ValueError("Resume requires the previously admitted auxiliary VIEW manifest")
    if auxiliary_view is not None:
        admission = {"manifest": str(auxiliary_path), "sha256": auxiliary_sha,
                     "train": len(auxiliary_view["train"]),
                     "development": len(auxiliary_view["development"])}
        previous = log.get("auxiliary_view")
        if previous is not None and previous != admission:
            previous_splits, previous_digest = load_auxiliary_view(
                Path(previous["manifest"]), validation,
            )
            if previous_digest != previous["sha256"]:
                raise ValueError("Previously admitted auxiliary VIEW manifest changed")
            for old_split, new_split in (("train", "development"), ("development", "train")):
                old_families = {
                    item.metadata["scene_family_id"] for item in previous_splits[old_split]
                }
                if any(item.metadata["scene_family_id"] in old_families
                       for item in auxiliary_view[new_split]):
                    raise ValueError("Auxiliary VIEW transition crosses historical split families")
        record_auxiliary_admission(
            log, admission, getattr(args, "auxiliary_view_previous_sha256", None),
            len(log["steps"]),
        )
    if jerk_references is not None:
        if reference_jerk_fallback:
            if "frozen_jerk_calibration" not in log:
                log["frozen_jerk_calibration"] = jerk_references.calibrate_fallback()
            atomic_json(output / "frozen_jerk_calibration.json", log["frozen_jerk_calibration"])
        jerk_definition = {
            "weight": jerk_weight, "scale": jerk_scale, **jerk_references.protocol(),
            "loss": "max(RMS_command_jerk - problem_reference_jerk, 0)^2 / scale^2",
            "scope": "successful and failed rollouts with >=4 commands in groups with at least one success; all-failure groups discarded",
            "missing_references": "disable jerk loss for that problem; retain task policy gradients",
            "gradient": "reparameterized denoising; recorded observations fixed, no simulator gradients",
        }
        if reference_jerk_fallback:
            jerk_definition["missing_references"] = "frozen 75th-percentile successful-training-reference margin; never calibrated from RL"
            jerk_definition["fallback_calibration"] = "frozen_jerk_calibration.json"
        if log.get("jerk_regularizer") != jerk_definition:
            log.setdefault("jerk_regularizer_changes", []).append({
                "start_step": log["steps"][-1]["step"] + 1 if log["steps"] else 1,
                "previous": log.get("jerk_regularizer"), "current": jerk_definition,
            })
        log["jerk_regularizer"] = jerk_definition
    if boundary_enabled or "action_boundary_regularizer" in log:
        boundary_definition = {
            "version": 2, "weight": boundary_weight, "scale": boundary_scale, "units": "degrees",
            "correction_limit": POLICY_ARM_CORRECTION_LIMIT,
            "loss": "max(maximum attempted arm correction - rejection correction limit, 0)^2 / scale^2",
            "scope": "attempted commands in groups with at least one success, including rejected/unexecuted commands; gripper excluded; all-failure groups discarded",
            "gradient": "reparameterized denoising; recorded observations fixed; raw proposals, no clipping",
            "execution_guard": "unchanged; arm excursions over 5 degrees still reject the rollout",
        }
        if log.get("action_boundary_regularizer") != boundary_definition:
            log.setdefault("action_boundary_regularizer_changes", []).append({
                "start_step": log["steps"][-1]["step"] + 1 if log["steps"] else 1,
                "previous": log.get("action_boundary_regularizer"), "current": boundary_definition,
            })
        log["action_boundary_regularizer"] = boundary_definition
    log["sft_coefficient"] = 0.
    log["run_config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ("sft_coefficient", "sft_batch_size", "policy_target_kl")
    }
    step_times = [
        row["step_seconds"] for row in log["steps"]
    ]
    start_step = log["steps"][-1]["step"] + 1 if log["steps"] else 1
    reward_definition = {
        "completion": (
            "1 - steps / longest_successful_steps within the task group; successful-only normalized advantage"
            if reference_free_completion
            else "reference-relative completion efficiency"
        ),
        "smoothness": "legacy reference ratio; native reference-free tasks use 1 / (1 + rms_jerk)",
        "smoothness_scope": "successful trajectories",
        "advantage": "independently normalized absolute-score components",
        "weights": {
            "success": args.success_advantage_weight,
            "completion": args.completion_advantage_weight,
            "smoothness": args.smoothness_advantage_weight,
        },
        "solved_groups_per_step": 1,
    }
    if jerk_enabled:
        reward_definition["smoothness_scope"] = "diagnostic only; optimization uses separate reference-margin regularizer"
        reward_definition["jerk_regularizer"] = log["jerk_regularizer"]
    if log.get("reward_definition") != reward_definition:
        log.setdefault("reward_changes", []).append({
            "start_step": start_step,
            **reward_definition,
        })
    log["reward_definition"] = reward_definition
    solved_problems = initialize_solved_problems(log, training)
    training_by_source = {
        str(item.source): item
        for episodes in training.values() for item in episodes
    }
    difficulty = None
    difficulty_requests = None
    difficulty_pools = {}
    excluded_families = {
        item.metadata.get("scene_family_id", item.metadata.get("initial", {}).get(
            "scene_family_id", hashlib.sha256(json.dumps([
                item.metadata.get("initial", {}).get("seed"),
                item.metadata.get("initial", {}).get("scene_version", 1),
            ]).encode()).hexdigest(),
        ))
        for item in validation + (auxiliary_view["development"] if auxiliary_view else [])
    }
    if automatic_difficulty:
        difficulty = DifficultyController(log.setdefault("difficulty_curriculum", {
            "schema": 1, "slices": {}, "events": [], "admissions": {},
        }))
        excluded_families.update(difficulty.state.get("excluded_families", []))
        difficulty.state["excluded_families"] = sorted(excluded_families)
        difficulty_requests = HarderDataRequests(
            output / "harder-data-requests", args.difficulty_catalog,
            [sys.executable, "-m", "so101.pile_task_worker"],
            getattr(args, "difficulty_request_timeout", 3600.),
        )
        refresh_difficulty_pools(
            difficulty, difficulty_requests, difficulty_pools, excluded_families, training_by_source, log,
            getattr(args, "native_pool_manifest", None), output,
        )
    elif log.get("difficulty_curriculum"):
        raise ValueError("Cannot disable an existing automatic difficulty curriculum on resume")
    if any(
        item["source"] not in training_by_source
        for item in solved_problems.values()
    ):
        raise ValueError("Solved bucket contains a non-training episode")
    if (output / "STOP").exists():
        raise ValueError("Remove the acknowledged STOP file before resuming")
    mastery_curriculum = (
        initialize_mastery_curriculum(log, args)
        if args.curriculum_mode == "mastery"
        else None
    )
    if mastery_curriculum is not None:
        for task in TASKS:
            maximum = curriculum_maximum(task, False, args)
            if maximum > 0:
                mastery_assistance(mastery_curriculum, task, maximum)
        if mastery_curriculum.get("recover_zero_success_groups", False) != recovery_enabled:
            log.setdefault("curriculum_recovery_changes", []).append({
                "start_step": start_step,
                "enabled": recovery_enabled,
                "reason": "Increase existing gold-prefix assistance when a group supplies no task-reward gradient",
            })
        mastery_curriculum["recover_zero_success_groups"] = recovery_enabled
        target_success = getattr(args, "target_success_curriculum", False)
        if mastery_curriculum.get("target_success", False) != target_success:
            log.setdefault("curriculum_strategy_changes", []).append({
                "start_step": start_step, "target_success": target_success,
                "window_groups": 4, "success_band": [.4, .6],
            })
            for task_state in mastery_curriculum["tasks"].values():
                task_state["success_window"] = []
        mastery_curriculum["target_success"] = target_success
    if jerk_references is not None:
        atomic_json(output / "jerk_reference_margins.json", jerk_references.snapshot())
    renderer_identity = getattr(sim, "rgb_identity", {"backend": "mujoco"})
    if log.get("observation_renderer") != renderer_identity:
        log.setdefault("observation_renderer_history", []).append({
            "start_step": start_step,
            "previous": log.get("observation_renderer", {"backend": "mujoco"}),
            "selected": renderer_identity,
        })
    log["observation_renderer"] = renderer_identity
    write_log(output, log)
    try:
        if batch_rollouts:
            for _ in range(1, args.group_size):
                rollout_sims.append(create_simulation(args.model_root, **renderer_options))
        if not args.resume and getattr(args, "initial_validation", True):
            baseline_start = time.monotonic()
            baseline = validation_metrics(
                sim,
                controller,
                validation,
                bounds,
                args.seed + 900_000,
                args.horizon,
                args.action_steps,
                sampling_noise,
            )
            baseline["step"] = 0
            baseline["validation_seconds"] = (
                time.monotonic() - baseline_start
            )
            log["validation"].append(baseline)
            write_log(output, log)
            print(json.dumps({
                "event": "baseline_validation",
                **baseline,
            }), flush=True)
        elif not args.resume:
            print(json.dumps({
                "event": "initial_validation_skipped",
                "validation_interval": args.validation_interval,
            }), flush=True)
        else:
            log["advantage_weights"] = {
                "success": args.success_advantage_weight,
                "completion": args.completion_advantage_weight,
                "smoothness": args.smoothness_advantage_weight,
            }
            log.setdefault("objective_changes", []).append({
                "start_step": start_step,
                "advantage": "independently normalized components",
                "weights": log["advantage_weights"],
            })
            log.setdefault("learning_rate_changes", []).append({
                "start_step": start_step,
                "learning_rate": args.learning_rate,
            })
            log.setdefault("sft_coefficient_changes", []).append({
                "start_step": start_step,
                "sft_coefficient": 0.,
            })
            print(json.dumps({
                "event": "resume",
                "start_step": start_step,
                "target_step": args.steps,
            }), flush=True)
        if auxiliary_view is not None and not any(
            item.get("manifest_sha256") == auxiliary_sha
            for item in log.get("auxiliary_view_validation", [])
        ):
            development = validation_metrics(
                sim, controller, auxiliary_view["development"], bounds,
                args.seed + 1_900_000, args.horizon, args.action_steps,
                sampling_noise,
            )
            development["step"] = start_step - 1
            development["phase"] = "before_auxiliary_training"
            development["manifest_sha256"] = auxiliary_sha
            log.setdefault("auxiliary_view_validation", []).append(development)
            write_log(output, log)
        for step in range(start_step, args.steps + 1):
            if difficulty is not None:
                refresh_difficulty_pools(
                    difficulty, difficulty_requests, difficulty_pools, excluded_families, training_by_source, log,
                    getattr(args, "native_pool_manifest", None), output,
                )
            step_start = time.monotonic()
            phase_seconds = {"rollouts": 0., "jerk_backward": 0., "task_backward": 0., "optimizer": 0.}
            optimizer.zero_grad(set_to_none=True)
            accepted_groups = 0
            jerk_groups = 0
            component_device = "cpu" if trainable_scope == "all" else None
            jerk_gradient_sum = (
                torch.zeros_like(gradient_vector(parameters, component_device))
                if regularizers_enabled else None
            )
            boundary_groups = 0
            boundary_gradient_sum = torch.zeros_like(jerk_gradient_sum) if boundary_enabled else None
            group_rows = []
            problems: list[tuple[str, dict[str, Any] | None]] = [
                (task, None) for task in TASKS
            ]
            solved_problem = sample_solved_problem(solved_problems, rng, step)
            if solved_problem is not None:
                problems.append((solved_problem["task"], solved_problem))
            for task_index, (task, solved_problem) in enumerate(problems):
                accepted_for_task = 0
                attempts = 0
                while (
                    accepted_for_task < (
                        1 if solved_problem is not None else args.groups_per_task
                    )
                    and attempts < (
                        1 if solved_problem is not None
                        else args.max_group_attempts_per_task
                    )
                ):
                    attempt = attempts
                    attempts += 1
                    episode = (
                        training_by_source[solved_problem["source"]]
                        if solved_problem is not None
                        else (
                            auxiliary_view["train"][int(rng.integers(len(auxiliary_view["train"])))]
                            if auxiliary_view is not None and task == "view"
                            else training[task][int(rng.integers(len(training[task])))]
                        )
                    )
                    start_proximity = 0.
                    has_obstacles = bool(
                        episode.metadata.get("obstacles")
                    )
                    curriculum_max = curriculum_maximum(
                        task, has_obstacles, args
                    )
                    assistance_key = curriculum_key(
                        task, has_obstacles
                    )
                    if difficulty is not None and solved_problem is None:
                        slice_state = difficulty.slice(assistance_key, curriculum_max)
                        if slice_state["level"] > 0:
                            pool = difficulty_pools[(assistance_key, slice_state["level"])][task]
                            episode = pool[int(rng.integers(len(pool)))]
                            curriculum_max = 0.
                        start_proximity = slice_state["assistance"]
                    if (
                        solved_problem is None
                        and difficulty is None
                        and curriculum_max > 0
                        and step >= args.curriculum_start_step
                    ):
                        if args.curriculum_mode == "mastery":
                            start_proximity = mastery_assistance(
                                mastery_curriculum,
                                assistance_key,
                                curriculum_max,
                            )
                        elif (
                            args.curriculum_end_step
                            > args.curriculum_start_step
                        ):
                            curriculum_scale = max(
                                0.,
                                min(
                                    1.,
                                    (
                                        args.curriculum_end_step - step
                                    )
                                    / (
                                        args.curriculum_end_step
                                        - args.curriculum_start_step
                                    ),
                                ),
                            )
                            start_proximity = (
                                curriculum_scale * curriculum_max
                            )
                    if solved_problem is not None:
                        start_proximity = solved_problem["start_proximity"]
                    jerk_profile = (jerk_references.summary(episode.source, start_proximity)
                                    if jerk_references is not None else None)
                    if jerk_references is not None and reference_jerk_fallback:
                        jerk_profile = jerk_references.summary_with_fallback(
                            episode.source, start_proximity, log["frozen_jerk_calibration"],
                        )
                    jerk_margin = jerk_profile["margin"] if jerk_profile is not None else None
                    capture_path = (output / "rollouts" / f"step_{step:06}" / f"group_{task_index}_{attempt}.json"
                                    if getattr(args, "record_rollouts", False) else None)
                    rollout_started = time.monotonic()
                    trajectories = (rollout_group(
                        rollout_sims, controller, episode,
                        [args.seed + step * 100_000 + task_index * 10_000
                         + attempt * args.group_size + index for index in range(args.group_size)],
                        args.horizon, args.action_steps, bounds[task], start_proximity,
                        sampling_noise,
                        boundary_enabled or jerk_enabled and jerk_margin is not None,
                        capture_path,
                    ) if batch_rollouts else [
                        rollout(
                            sim,
                            controller,
                            episode,
                            args.seed
                            + step * 100_000
                            + task_index * 10_000
                            + attempt * args.group_size
                            + rollout_index,
                            args.horizon,
                            args.action_steps,
                            bounds[task],
                            True,
                            start_proximity,
                            sampling_noise,
                            **({"record_path_noise": True}
                               if boundary_enabled or jerk_enabled and jerk_margin is not None else {}),
                            **({
                                "capture_path": output / "rollouts" / f"step_{step:06}" / f"group_{task_index}_{attempt}.json",
                            } if getattr(args, "record_rollouts", False) and rollout_index == 0 else {}),
                        )
                        for rollout_index in range(args.group_size)
                    ])
                    phase_seconds["rollouts"] += time.monotonic() - rollout_started
                    success_count = sum(
                        item.success for item in trajectories
                    )
                    if (difficulty is not None and solved_problem is None
                            and step >= args.curriculum_start_step):
                        difficulty_event = difficulty.record(
                            assistance_key, success_count, len(trajectories), curriculum_max,
                            args.curriculum_decrement, step,
                        )
                        mastery_assistance(mastery_curriculum, assistance_key, curriculum_max)
                        mastery_curriculum["managed_by"] = "difficulty_curriculum"
                        current_slice = difficulty.slice(assistance_key, curriculum_max)
                        mastery_curriculum["tasks"][assistance_key].update(
                            level=current_slice["assistance"], difficulty_level=current_slice["level"],
                            success_window=current_slice["window"],
                        )
                        if difficulty_event is not None:
                            print(json.dumps({"event": "automatic_difficulty", **difficulty_event}), flush=True)
                            if difficulty_event["action"] == "request":
                                level = difficulty.slice(assistance_key, curriculum_max)["pending"]
                                slice_state = difficulty.slice(assistance_key, curriculum_max)
                                path = difficulty_requests.request(
                                    assistance_key, level, task, args.seed + level * 1_000_000 + task_index * 10_000
                                    + slice_state["request_attempts"][str(level)] * 1000,
                                    sorted(excluded_families),
                                    {"piece_count": min(128, 4 + 2 * level),
                                     "color_mode": ("many", "few", "monochrome")[(level - 1) % 3]},
                                )
                                slice_state["pending_request_id"] = path.parent.name
                    record_solved_problem(
                        solved_problems, episode, start_proximity,
                        success_count, len(trajectories), step,
                    )
                    mastery_event = None
                    mastery_check = None
                    recovery_event = None
                    if (
                        solved_problem is None
                        and difficulty is None
                        and args.curriculum_mode == "mastery"
                        and curriculum_max > 0
                        and step >= args.curriculum_start_step
                    ):
                        if mastery_curriculum.get("target_success", False):
                            mastery_event = record_target_success_group(
                                mastery_curriculum, assistance_key, success_count,
                                len(trajectories), curriculum_max, step,
                            )
                            mastery_ready = False
                            if mastery_event is not None:
                                print(json.dumps({
                                    "event": "curriculum_target_success",
                                    **mastery_event,
                                }), flush=True)
                        else:
                            mastery_ready = record_mastery_group(
                                mastery_curriculum,
                                assistance_key,
                                success_count,
                                args.group_size,
                            )
                        if (recovery_enabled and success_count == 0
                                and not mastery_curriculum.get("target_success", False)):
                            recovery_event = recover_zero_success_group(
                                mastery_curriculum, assistance_key, success_count,
                                curriculum_max, step,
                            )
                            if recovery_event is not None:
                                recovery_event["source"] = str(episode.source)
                                print(json.dumps({
                                    "event": "curriculum_recovery", **recovery_event,
                                }), flush=True)
                        if mastery_ready:
                            mastery_episodes = [
                                item for item in validation
                                if item.task == task
                                and bool(item.metadata.get("obstacles"))
                                == has_obstacles
                            ]
                            if not mastery_episodes:
                                mastery_episodes = [
                                    item for item in training[task]
                                    if bool(
                                        item.metadata.get("obstacles")
                                    ) == has_obstacles
                                ][:10]
                            mastery_successes = sum(
                                rollout(
                                    sim,
                                    controller,
                                    item,
                                    args.seed
                                    + 8_000_000
                                    + step * 10_000
                                    + mastery_index,
                                    args.horizon,
                                    args.action_steps,
                                    bounds[task],
                                    False,
                                    start_proximity,
                                    sampling_noise,
                                ).success
                                for mastery_index, item in enumerate(
                                    mastery_episodes
                                )
                            )
                            mastery_check = {
                                "successes": mastery_successes,
                                "total": len(mastery_episodes),
                                "pass_rate": (
                                    mastery_successes
                                    / len(mastery_episodes)
                                ),
                            }
                            mastery_event = resolve_mastery_check(
                                mastery_curriculum,
                                assistance_key,
                                mastery_successes,
                                len(mastery_episodes),
                                step,
                                args.group_size,
                            )
                    assign_group_rewards(
                        trajectories,
                        args.success_advantage_weight,
                        args.completion_advantage_weight,
                        args.smoothness_advantage_weight,
                        reference_free_completion,
                    )
                    jerk_group_metrics = {}
                    if jerk_enabled:
                        before_jerk = gradient_vector(parameters, component_device)
                        jerk_started = time.monotonic()
                        jerk_losses = group_jerk_backward(
                            controller, trajectories,
                            torch.as_tensor(sim.encode(sim.limits[:, 0]), device=args.device),
                            torch.as_tensor(sim.encode(sim.limits[:, 1]), device=args.device),
                            jerk_weight, jerk_scale, jerk_margin,
                        )
                        phase_seconds["jerk_backward"] += time.monotonic() - jerk_started
                        jerk_gradient_sum.sub_(before_jerk)
                        del before_jerk
                        jerk_gradient_sum.add_(gradient_vector(parameters, component_device))
                        regularized = any(cost is not None for cost in jerk_losses)
                        jerk_groups += int(regularized)
                        jerk_group_metrics = {
                            "jerk_margin": jerk_margin, "jerk_losses": jerk_losses,
                            "jerk_reference_scope": jerk_profile.get("reference_scope", "problem"),
                            "jerk_reference_source": str(episode.source),
                            "jerk_regularizer_enabled": jerk_margin is not None,
                            "jerk_regularizer_applied": any(cost is not None and cost > 0 for cost in jerk_losses),
                            "jerk_group_eligible": success_count > 0,
                            "jerk_reference_count": jerk_profile["reference_count"],
                            "jerk_disabled_reason": jerk_profile["disabled_reason"],
                        }
                        print(json.dumps({
                            "event": "jerk_training_group", "step": step, "task": task,
                            "sampling_bucket": "solved" if solved_problem is not None else "main",
                            "successes": success_count, **jerk_group_metrics,
                        }), flush=True)
                    if boundary_enabled:
                        before_boundary = gradient_vector(parameters, component_device)
                        boundary_losses = group_action_boundary_backward(
                            controller, trajectories,
                            torch.as_tensor(sim.encode(sim.limits[:, 0]), device=args.device),
                            torch.as_tensor(sim.encode(sim.limits[:, 1]), device=args.device),
                            boundary_weight, boundary_scale, POLICY_ARM_CORRECTION_LIMIT,
                        )
                        boundary_gradient_sum.sub_(before_boundary)
                        del before_boundary
                        boundary_gradient_sum.add_(gradient_vector(parameters, component_device))
                        boundary_groups += int(any(cost is not None for cost in boundary_losses))
                        boundary_metrics = {
                            "action_boundary_losses": boundary_losses,
                            "action_boundary_regularizer_applied": any(
                                cost is not None and cost > 0 for cost in boundary_losses),
                            "action_boundary_group_eligible": success_count > 0,
                            "invalid_action_count": sum(
                                item.invalid_action_error is not None for item in trajectories),
                        }
                        jerk_group_metrics.update(boundary_metrics)
                        print(json.dumps({
                            "event": "action_boundary_training_group", "step": step,
                            "command": task, "source": str(episode.source),
                            "successes": success_count, **boundary_metrics,
                        }), flush=True)
                    log["groups"] += 1
                    log["success_count_histogram"][str(success_count)] += 1
                    if success_count == 0:
                        log["zero_success_groups"] += 1
                        log["zero_success_by_task"][task] += 1
                        group_rows.append({
                            "task": task,
                            "sampling_bucket": "solved" if solved_problem is not None else "main",
                            "source": str(episode.source),
                            "target": episode.metadata["target"],
                            "destination": episode.metadata.get("destination"),
                            "assistance_key": assistance_key,
                            "successes": 0,
                            "discarded": True,
                            "start_proximity": start_proximity,
                            "mastery_event": mastery_event,
                            "mastery_check": mastery_check,
                            "curriculum_recovery": recovery_event,
                            **jerk_group_metrics,
                            "trajectories": [
                                serialized_trajectory(item)
                                for item in trajectories
                            ],
                        })
                        del trajectories
                        continue
                    rewards = np.asarray([
                        item.reward for item in trajectories
                    ], dtype=np.float32)
                    advantages, advantage_components = group_advantages(
                        trajectories,
                        args.success_advantage_weight,
                        args.completion_advantage_weight,
                        args.smoothness_advantage_weight,
                        reference_free_completion,
                    )
                    denominator = float(
                        np.abs(advantages).sum() + 1e-6
                    )
                    losses = []
                    task_started = time.monotonic()
                    for trajectory, advantage in zip(
                        trajectories, advantages, strict=True
                    ):
                        losses.append(trajectory_backward(
                            controller,
                            trajectory,
                            float(advantage) / denominator,
                            args.policy_clip,
                        ))
                        trajectory.decisions.clear()
                    phase_seconds["task_backward"] += time.monotonic() - task_started
                    accepted_groups += 1
                    accepted_for_task += 1
                    group_rows.append({
                        "task": task,
                        "sampling_bucket": "solved" if solved_problem is not None else "main",
                        "source": str(episode.source),
                        "target": episode.metadata["target"],
                        "destination": episode.metadata.get("destination"),
                        "assistance_key": assistance_key,
                        "successes": success_count,
                        "discarded": False,
                        "start_proximity": start_proximity,
                        "mastery_event": mastery_event,
                        "mastery_check": mastery_check,
                        "curriculum_recovery": recovery_event,
                        **jerk_group_metrics,
                        "rewards": rewards.tolist(),
                        "advantages": advantages.tolist(),
                        "advantage_components": {
                            name: values.tolist()
                            for name, values
                            in advantage_components.items()
                        },
                        (
                            "trajectory_policy_losses" if sampling_noise is not None
                            else "trajectory_flow_losses"
                        ): losses,
                        "trajectories": [
                            serialized_trajectory(item)
                            for item in trajectories
                        ],
                    })
                    del trajectories, trajectory
            component_metrics = {}
            update_groups = accepted_groups
            if regularizers_enabled:
                active, component_metrics = normalize_gradient_components(
                    parameters, jerk_gradient_sum, accepted_groups, jerk_groups,
                    boundary_sum=boundary_gradient_sum, boundary_groups=boundary_groups,
                )
                update_groups = int(active)
            probes = {}
            replay_implementation = None
            if trainable_scope == "all" and sampling_noise is not None and update_groups:
                replay_implementation = getattr(
                    controller.policy.model, "rl_replay_implementation", "per-transition-prefix-v1",
                )
                if log.get("full_model_gradient_audit", {}).get("implementation") != replay_implementation:
                    probes = full_model_gradient_probes(controller.policy)
            optimizer_started = time.monotonic()
            update_metrics = policy_optimizer_step(optimizer, parameters, update_groups, args.gradient_clip)
            phase_seconds["optimizer"] += time.monotonic() - optimizer_started
            if probes:
                audit = {}
                for group, (name, parameter, index, before) in probes.items():
                    after = float(parameter.detach().flatten()[index])
                    if before == after or not math.isfinite(after):
                        raise ValueError(f"Full-model optimizer did not effectively update {group}")
                    audit[group] = {"parameter": name, "index": index, "before": before, "after": after}
                if "full_model_gradient_audit" in log:
                    log.setdefault("full_model_gradient_audit_history", []).append(log["full_model_gradient_audit"])
                log["full_model_gradient_audit"] = {
                    "step": step, "implementation": replay_implementation, "groups": audit,
                }
            update_metrics.update(component_metrics)
            if regularizers_enabled and not update_groups:
                update_metrics["optimizer_skip_reason"] = (
                    "no_active_task_or_action_regularizer_gradients" if boundary_enabled
                    else "no_active_task_or_jerk_gradients")
            if sampling_noise is not None:
                update_metrics["policy_loss_scale"] = log["policy_optimization"]["policy_loss_scale"]
            if not update_metrics["optimizer_update_applied"]:
                print(json.dumps({
                    "event": "policy_update_skipped",
                    "step": step,
                    **update_metrics,
                }), flush=True)
            elapsed = time.monotonic() - step_start
            phase_seconds["other"] = elapsed - sum(phase_seconds.values())
            step_times.append(elapsed)
            rolling = float(np.mean(step_times[-5:]))
            step_row = {
                "step": step,
                "completed_at": time.time(),
                "step_seconds": elapsed,
                "wall_phase_seconds": phase_seconds,
                "rolling_step_seconds": rolling,
                "eta_seconds": rolling * (args.steps - step),
                "accepted_groups": accepted_groups,
                "discarded_groups": sum(
                    group["discarded"] for group in group_rows
                ),
                "sft_coefficient": 0.,
                "training_objective": training_objective,
                "learning_rate": args.learning_rate,
                "rl_estimator": args.rl_estimator,
                "denoising_noise_scale": sampling_noise,
                "groups": group_rows,
                "solved_bucket_size": len(solved_problems),
                **update_metrics,
            }
            if mastery_curriculum is not None:
                step_row["assistance_curriculum"] = {
                    key: dict(value)
                    for key, value
                    in mastery_curriculum["tasks"].items()
                }
            log["steps"].append(step_row)
            if step % args.validation_interval == 0:
                validation_start = time.monotonic()
                metrics = validation_metrics(
                    sim,
                    controller,
                    validation,
                    bounds,
                    args.seed + 900_000,
                    args.horizon,
                    args.action_steps,
                    sampling_noise,
                )
                metrics["step"] = step
                metrics["validation_seconds"] = (
                    time.monotonic() - validation_start
                )
                log["validation"].append(metrics)
                step_row["validation"] = metrics
                if auxiliary_view is not None:
                    development = validation_metrics(
                        sim, controller, auxiliary_view["development"], bounds,
                        args.seed + 1_900_000, args.horizon, args.action_steps,
                        sampling_noise,
                    )
                    development["step"] = step
                    development["manifest_sha256"] = auxiliary_sha
                    log.setdefault("auxiliary_view_validation", []).append(development)
                if getattr(args, "retain_evaluation_checkpoints", False):
                    save_delta_checkpoint(
                        output / f"delta_step_{step:03}.pt", controller, optimizer, step,
                    )
            save_resume_state(
                output, controller, optimizer, rng, log,
                getattr(args, "checkpoint_generations", 2),
            )
            if jerk_references is not None:
                atomic_json(output / "jerk_reference_margins.json", jerk_references.snapshot())
            write_log(output, log)
            if getattr(args, "record_rollouts", False):
                prune_evidence(output, getattr(args, "rollout_retention_steps", 20))
            print(json.dumps({
                key: value
                for key, value in step_row.items()
                if key != "groups"
            }), flush=True)
            if (output / "STOP").exists():
                print(json.dumps({
                    "event": "stopped_at_checkpoint",
                    "step": step,
                }), flush=True)
                break
    finally:
        if difficulty_requests is not None:
            difficulty_requests.close()
        for rollout_sim in rollout_sims:
            native = getattr(rollout_sim, "_native_simulation", None)
            if native is not None:
                native.close()
            rollout_sim.close()
    if args.save_final_checkpoint:
        checkpoint_output = output / "checkpoint"
        if checkpoint_output.exists():
            base = output / f"checkpoint_step_{args.steps:03}"
            checkpoint_output = base
            suffix = 2
            while checkpoint_output.exists():
                checkpoint_output = Path(f"{base}_{suffix}")
                suffix += 1
        save_entity_checkpoint(
            checkpoint_output,
            controller.policy,
            controller.config,
            "lerobot/smolvla_base",
            {
                "state_mean": controller.state_mean.tolist(),
                "state_std": controller.state_std.tolist(),
                "action_mean": controller.action_mean.tolist(),
                "action_std": controller.action_std.tolist(),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainable-scope", choices=("all", "action-projections"), default="all")
    parser.add_argument("--automatic-difficulty", action="store_true",
                        help="Advance training-only native pile tiers when zero-assistance success exceeds 60%%")
    parser.add_argument("--difficulty-catalog", type=Path)
    parser.add_argument("--native-pool-manifest", type=Path,
                        help="Atomic continuous TRAIN pool; per-tier admissions are frozen in this run")
    parser.add_argument("--difficulty-request-timeout", type=float, default=3600.)
    parser.add_argument("--retain-evaluation-checkpoints", action="store_true",
                        help="Also retain unbounded evaluation optimizer checkpoints; disabled by default")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--refresh-corpus",
        action="store_true",
        help="On resume, refresh training episodes and expert bounds while retaining validation cases.",
    )
    parser.add_argument(
        "--collection",
        type=Path,
        default=Path(
            "data/simulation/entity_commands_clean_projected_1610"
        ),
    )
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--rgb-renderer", choices=("mujoco", "cpu-pathtrace", "gpu-pathtrace"),
                        default="mujoco",
                        help="Policy RGB backend; GPU path tracing requires CUDA/OptiX, with no CPU fallback.")
    parser.add_argument("--pathtrace-samples", type=int, default=16,
                        help="Experimental online budget; historical offline renders used 1024 samples.")
    parser.add_argument("--pathtrace-threads", type=int, default=4)
    parser.add_argument("--batch-rollouts", action=argparse.BooleanOptionalAction, default=False,
                        help="Batch group inference with independent simulator histories and trajectory RNGs.")
    parser.add_argument("--groups-per-task", type=int, default=2)
    parser.add_argument(
        "--max-group-attempts-per-task",
        type=int,
        default=8,
    )
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--action-steps", type=int, default=16)
    parser.add_argument("--record-rollouts", action=argparse.BooleanOptionalAction, default=True,
                        help="Record actions and sparse actual wrist frames for the first rollout of each training group.")
    parser.add_argument("--rollout-retention-steps", type=int, default=20)
    parser.add_argument("--validation-per-task", type=int, default=10)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--initial-validation", action=argparse.BooleanOptionalAction, default=True,
                        help="Evaluate before the first update; disabling retains periodic frozen evaluation.")
    parser.add_argument(
        "--rl-estimator", choices=("flow-mse", "denoising-grpo"),
        default="flow-mse",
        help="Legacy signed flow MSE or recorded-path Gaussian policy gradients.",
    )
    parser.add_argument("--denoising-noise-scale", type=float, default=.1)
    parser.add_argument("--policy-clip", type=float, default=.2)
    parser.add_argument(
        "--success-advantage-weight", type=float, default=1.
    )
    parser.add_argument(
        "--completion-advantage-weight", type=float, default=.5
    )
    parser.add_argument(
        "--smoothness-advantage-weight", type=float, default=1.
    )
    parser.add_argument("--jerk-regularizer-weight", type=float, default=0.,
                        help="Reference-margin command-jerk regularizer; requires smoothness advantage weight 0.")
    parser.add_argument("--jerk-scale", type=float, default=6.)
    parser.add_argument("--action-boundary-weight", type=float, default=0.,
                        help="Raw-arm joint-boundary penalty, including rejected proposals; requires denoising-grpo.")
    parser.add_argument("--action-boundary-scale", type=float, default=6.,
                        help="Joint-boundary penalty scale in degrees; does not change execution limits.")
    parser.add_argument("--learning-rate", type=float, default=3e-7)
    parser.add_argument("--gradient-clip", type=float, default=1.)
    parser.add_argument(
        "--curriculum-start-step", type=int, default=0
    )
    parser.add_argument(
        "--curriculum-mode",
        choices=("mastery", "linear"),
        default="mastery",
    )
    parser.add_argument(
        "--curriculum-end-step", type=int, default=0
    )
    parser.add_argument(
        "--curriculum-decrement", type=float, default=.05
    )
    parser.add_argument(
        "--curriculum-mastery-groups", type=int, default=4
    )
    parser.add_argument("--curriculum-min", type=float, default=0.)
    parser.add_argument("--recover-zero-success-groups", action="store_true",
                        help="Increase gold-prefix assistance by the existing curriculum increment after zero-success groups.")
    parser.add_argument("--target-success-curriculum", action="store_true",
                        help="Adapt assistance using four recent groups toward 40-60% success, instead of perfect-group mastery.")
    parser.add_argument("--reference-jerk-fallback", action="store_true",
                        help="Use a frozen 75th-percentile successful-reference jerk margin when a problem has no usable reference.")
    parser.add_argument("--reference-free-completion", action="store_true",
                        help="Normalize successful-rollout length within each task group instead of using expert completion efficiency.")
    parser.add_argument("--auxiliary-view-previous-sha256",
                        help="Explicit curriculum transition from this admitted manifest hash")
    parser.add_argument("--auxiliary-view-manifest", type=Path,
                        help="Hash-bound harder VIEW train/development references; preserves original fixed validation split.")
    parser.add_argument(
        "--push-curriculum-max", type=float, default=.95
    )
    parser.add_argument(
        "--place-curriculum-max", type=float, default=.95
    )
    parser.add_argument(
        "--grasp-curriculum-max", type=float, default=0.
    )
    parser.add_argument(
        "--view-curriculum-max", type=float, default=0.
    )
    parser.add_argument(
        "--obstacle-curriculum-max", type=float, default=0.
    )
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda"), default="mps"
    )
    parser.add_argument(
        "--save-final-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--seed", type=int, default=51_000)
    parser.add_argument("--checkpoint-generations", type=int, default=2)
    parser.add_argument("--allow-device-rng-migration", action="store_true")
    parser.add_argument("--device-rng-seed", type=int)
    args = parser.parse_args()
    if args.rollout_retention_steps < 1:
        parser.error("--rollout-retention-steps must be positive")
    train(args)


if __name__ == "__main__":
    main()
