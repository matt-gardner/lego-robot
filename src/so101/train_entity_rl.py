from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from so101.control_model.modeling_entity_smolvla import save_entity_checkpoint
from so101.evaluate_entity_sim import EntityPolicy
from so101.robot_sim import BeamSimulation, MODEL_ROOT, OBJECT_NAMES


def apply_start_curriculum(
    sim: BeamSimulation,
    task: str,
    target: str,
    destination: str,
    proximity: float,
) -> None:
    if not 0 <= proximity <= 1:
        raise ValueError("Curriculum proximity must be within [0, 1]")
    if proximity == 0:
        return
    target_position = sim.position(target)
    if task == "push":
        destination_position = sim.position(destination)
        direction = destination_position[:2] - target_position[:2]
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        behind = target_position[:2] - .025 * direction
        approach = sim.push_ik(
            np.r_[behind, .055],
            float(sim.data.qpos[sim.qadr[4]]),
        )
        approach[-1] = sim.limits[-1, 0]
    elif task == "place":
        approach = sim.ik(
            np.r_[target_position[:2], .065],
            sim.yaws[target] + math.pi / 2,
        )
        approach[-1] = (
            sim.limits[-1, 0] + .5 * np.ptp(sim.limits[-1])
        )
    else:
        raise ValueError(f"Unsupported curriculum task: {task}")
    start = sim.data.ctrl[sim.actuators].copy()
    sim.move(
        start + proximity * (approach - start),
        "rl_curriculum_start",
        seconds=.8,
    )


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
    if task == "push":
        gripper_xy = sim.data.site_xpos[sim.site, :2]
        target_xy = sim.position(target)[:2]
        if contacts:
            return "push"
        if np.linalg.norm(gripper_xy - target_xy) < .06:
            return "lower"
        return "approach"
    if task != "place" or destination is None:
        raise ValueError(f"Unsupported RL task: {task}")
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


def transition_metrics(
    sim: BeamSimulation,
    target: str,
    destination: str,
    initial_positions: dict[str, np.ndarray],
) -> dict[str, float | bool | set[str]]:
    target_position = sim.position(target).copy()
    destination_position = sim.position(destination)
    gripper_position = sim.data.site_xpos[sim.site].copy()
    direction = destination_position[:2] - target_position[:2]
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    behind = target_position[:2] - .025 * direction
    wrong_object = False
    for other in OBJECT_NAMES:
        if other == target:
            continue
        lift = float(
            sim.position(other)[2] - initial_positions[other][2]
        )
        if lift >= .03 and sim.contacts(other) == {"fixed", "moving"}:
            wrong_object = True
            break
    return {
        "target_lift": float(
            target_position[2] - initial_positions[target][2]
        ),
        "destination_error": float(np.linalg.norm(
            target_position[:2] - destination_position[:2]
        )),
        "gripper_distance": float(np.linalg.norm(
            gripper_position - target_position
        )),
        "push_approach_distance": float(np.linalg.norm(
            gripper_position - np.r_[behind, .055]
        )),
        "push_lower_distance": float(np.linalg.norm(
            gripper_position - np.r_[behind, .014]
        )),
        "contacts": sim.contacts(target),
        "wrong_object": wrong_object,
    }


def candidate_reward(
    task: str,
    phase: str,
    before: dict[str, float | bool | set[str]],
    after: dict[str, float | bool | set[str]],
    actions: np.ndarray,
) -> tuple[float, dict[str, float]]:
    lift_progress = float(after["target_lift"]) - float(before["target_lift"])
    destination_progress = (
        float(before["destination_error"])
        - float(after["destination_error"])
    )
    gripper_progress = (
        float(before["gripper_distance"])
        - float(after["gripper_distance"])
    )
    retained = after["contacts"] == {"fixed", "moving"}
    task_progress = 0.
    success_bonus = 0.
    if task == "push":
        if phase == "approach":
            task_progress = 4 * (
                float(before["push_approach_distance"])
                - float(after["push_approach_distance"])
            )
        elif phase == "lower":
            task_progress = 4 * (
                float(before["push_lower_distance"])
                - float(after["push_lower_distance"])
            )
        else:
            task_progress = 12 * destination_progress
            task_progress += .5 if after["contacts"] else 0.
        if float(after["destination_error"]) <= .07:
            success_bonus = 3.
    elif phase == "acquire":
        task_progress = (
            3 * gripper_progress
            + 12 * max(lift_progress, 0.)
            + (1.5 if retained else 0.)
        )
    elif phase == "lift":
        task_progress = 12 * lift_progress + (1. if retained else -1.)
    elif phase == "carry":
        task_progress = (
            12 * destination_progress + (1. if retained else -2.)
        )
    elif phase == "lower":
        before_height_error = abs(float(before["target_lift"]) - .02)
        after_height_error = abs(float(after["target_lift"]) - .02)
        task_progress = (
            8 * destination_progress
            + 2 * (before_height_error - after_height_error)
            + (1. if retained else -2.)
        )
    else:
        released = after["contacts"] != {"fixed", "moving"}
        task_progress = (
            8 * destination_progress + (2. if released else -1.)
        )
        if released and float(after["destination_error"]) <= .04:
            success_bonus = 4.
    wrong_object_penalty = -8. if bool(after["wrong_object"]) else 0.
    arm_delta = np.diff(actions[:, :5], axis=0)
    arm_acceleration = np.diff(arm_delta, axis=0)
    arm_jerk = np.diff(arm_acceleration, axis=0)
    rms_jerk = (
        float(np.sqrt(np.mean(arm_jerk ** 2)))
        if arm_jerk.size else 0.
    )
    jerk_penalty = -2e-3 * rms_jerk
    time_penalty = -.002 * len(actions)
    reward = (
        task_progress
        + success_bonus
        + wrong_object_penalty
        + jerk_penalty
        + time_penalty
    )
    return reward, {
        "task_progress": task_progress,
        "success_bonus": success_bonus,
        "wrong_object_penalty": wrong_object_penalty,
        "jerk_penalty": jerk_penalty,
        "time_penalty": time_penalty,
        "rms_arm_jerk": rms_jerk,
    }


def restore_simulation(
    sim: BeamSimulation,
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray,
    time: float,
) -> None:
    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = qvel
    sim.data.ctrl[:] = ctrl
    sim.data.time = time
    mujoco.mj_forward(sim.model, sim.data)


def trainable_parameters(
    policy: torch.nn.Module, expert_layers: int
) -> list[torch.nn.Parameter]:
    for parameter in policy.parameters():
        parameter.requires_grad = False
    layer_count = policy.model.vlm_with_expert.lm_expert.config.num_hidden_layers
    first_layer = max(0, layer_count - expert_layers)
    selected = []
    for name, parameter in policy.named_parameters():
        train = name.startswith((
            "model.action_in_proj.",
            "model.action_out_proj.",
        ))
        if name.startswith("model.vlm_with_expert.lm_expert.layers."):
            layer = int(name.split(".")[4])
            train = layer >= first_layer
        parameter.requires_grad = train
        if train:
            selected.append(parameter)
    if not selected:
        raise RuntimeError("No RL parameters selected")
    return selected


def train(
    checkpoint: Path,
    output: Path,
    benchmark: Path,
    episodes: int,
    decisions: int,
    candidates: int,
    action_steps: int,
    episodes_per_step: int,
    learning_rate: float,
    expert_layers: int,
    device: str,
    model_root: Path,
    seed: int,
    curriculum: bool,
) -> None:
    if min(
        episodes,
        decisions,
        candidates,
        action_steps,
        episodes_per_step,
    ) <= 0:
        raise ValueError("RL counts must be positive")
    if expert_layers < 0:
        raise ValueError("Expert layers must be non-negative")
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    examples = [
        example
        for example in json.loads(benchmark.read_text())["examples"]
        if example["task"] in ("push", "place")
    ]
    if not examples:
        raise ValueError("Benchmark has no push or place examples")
    by_task = {
        task: [example for example in examples if example["task"] == task]
        for task in ("push", "place")
    }
    examples = [
        example
        for index in range(max(map(len, by_task.values())))
        for task in ("push", "place")
        for example in by_task[task][index:index + 1]
    ]
    controller = EntityPolicy(checkpoint, device, True)
    policy = controller.policy
    parameters = trainable_parameters(policy, expert_layers)
    initial_parameters = [
        parameter.detach().cpu().clone() for parameter in parameters
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=1e-6
    )
    sim = BeamSimulation(model_root, render=True)
    logs = []
    episode_metrics = []
    step_metrics = []
    gradient_samples = 0
    accumulated_samples = 0
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)
    try:
        for episode in range(episodes):
            example = examples[episode % len(examples)]
            task = example["task"]
            target = example["target"]
            destination = example["destination"]
            entities = [(target, "object"), (destination, "destination")]
            sim.reset(
                example["seed"],
                varied_layout=True,
                real_scene=True,
                color_targets=True,
            )
            sim.configure_command(task, entities)
            start_proximity = (
                1 - episode / max(episodes - 1, 1)
                if curriculum
                else 0.
            )
            apply_start_curriculum(
                sim,
                task,
                target,
                destination,
                start_proximity,
            )
            controller.reset(sim, seed + episode, zero_noise=False)
            initial_positions = {
                name: sim.position(name).copy() for name in OBJECT_NAMES
            }
            lifted_once = False
            episode_actions = []
            success = False
            wrong_object = False
            for decision in range(decisions):
                phase = control_phase(
                    sim,
                    task,
                    target,
                    destination,
                    float(initial_positions[target][2]),
                    lifted_once,
                )
                inputs = controller.command_inputs(
                    sim, task, entities, phase
                )
                before = transition_metrics(
                    sim, target, destination, initial_positions
                )
                saved_qpos = sim.data.qpos.copy()
                saved_qvel = sim.data.qvel.copy()
                saved_ctrl = sim.data.ctrl.copy()
                saved_time = float(sim.data.time)
                normalized_candidates = []
                action_candidates = []
                rewards = []
                reward_components = []
                policy.eval()
                with torch.no_grad():
                    for _ in range(candidates):
                        prediction = policy.model.sample_actions_entity(
                            *inputs
                        )[0]
                        normalized = prediction[0].detach()
                        actions = (
                            normalized[:, :6].float().cpu().numpy()
                            * controller.action_std
                            + controller.action_mean
                        )
                        executed = []
                        try:
                            for action in actions[:action_steps]:
                                goal, _ = sim.bounded_policy_action(action)
                                sim.step(goal, "rl_candidate")
                                executed.append(sim.encode(goal))
                            after = transition_metrics(
                                sim, target, destination, initial_positions
                            )
                            reward, components = candidate_reward(
                                task,
                                phase,
                                before,
                                after,
                                np.asarray(executed, dtype=np.float32),
                            )
                        except (RuntimeError, ValueError):
                            reward = -20.
                            components = {
                                "task_progress": 0.,
                                "success_bonus": 0.,
                                "wrong_object_penalty": 0.,
                                "jerk_penalty": 0.,
                                "time_penalty": 0.,
                                "rms_arm_jerk": 0.,
                            }
                        finally:
                            restore_simulation(
                                sim,
                                saved_qpos,
                                saved_qvel,
                                saved_ctrl,
                                saved_time,
                            )
                        normalized_candidates.append(normalized)
                        action_candidates.append(actions)
                        rewards.append(reward)
                        reward_components.append(components)
                rewards_array = np.asarray(rewards, dtype=np.float32)
                best_index = int(np.argmax(rewards_array))
                advantage = float(
                    rewards_array[best_index] - rewards_array.mean()
                )
                loss_value = None
                if advantage > 1e-4:
                    target_actions = normalized_candidates[
                        best_index
                    ].unsqueeze(0)
                    policy.train()
                    action_losses = policy.model.forward_entity(
                        *inputs[:5],
                        target_actions,
                        *inputs[5:],
                    )[0]
                    loss = (
                        action_losses[:, :action_steps, :6].mean()
                        * min(advantage, 5.)
                    )
                    loss.backward()
                    loss_value = float(loss.detach())
                    accumulated_samples += 1
                    gradient_samples += 1
                executed_actions = []
                for action in action_candidates[best_index][:action_steps]:
                    goal, _ = sim.bounded_policy_action(action)
                    sim.step(goal, "rl_selected")
                    encoded = sim.encode(goal)
                    controller.observe(sim, encoded)
                    executed_actions.append(encoded)
                episode_actions.extend(executed_actions)
                lifted_once |= (
                    sim.position(target)[2] - initial_positions[target][2]
                ) >= .03
                after = transition_metrics(
                    sim, target, destination, initial_positions
                )
                success = bool(
                    float(after["destination_error"]) <= .07
                    if task == "push"
                    else (
                        lifted_once
                        and float(after["destination_error"]) <= .04
                        and after["contacts"] != {"fixed", "moving"}
                    )
                )
                wrong_object = bool(after["wrong_object"])
                row = {
                    "episode": episode,
                    "decision": decision,
                    "seed": example["seed"],
                    "task": task,
                    "start_proximity": start_proximity,
                    "phase": phase,
                    "rewards": rewards,
                    "best_reward": float(rewards_array[best_index]),
                    "advantage": advantage,
                    "loss": loss_value,
                    "training_step": optimizer_steps + 1,
                    **{
                        f"reward_{key}": value
                        for key, value in reward_components[
                            best_index
                        ].items()
                        if key != "rms_arm_jerk"
                    },
                    "rms_arm_jerk": reward_components[
                        best_index
                    ]["rms_arm_jerk"],
                    "success": success,
                    "completion_seconds": (
                        (decision + 1) * action_steps / sim.fps
                        if success else None
                    ),
                    "completion_efficiency_percent": (
                        100 * {"push": 3, "place": 5}[task]
                        / (decision + 1)
                        if success else None
                    ),
                    "wrong_object": wrong_object,
                    "target_contact": bool(after["contacts"]),
                    "target_retained": (
                        after["contacts"] == {"fixed", "moving"}
                    ),
                    "destination_error_m": float(
                        after["destination_error"]
                    ),
                    "target_lift_m": float(after["target_lift"]),
                }
                logs.append(row)
                print(json.dumps(row), flush=True)
                if success or wrong_object:
                    break
            action_array = np.asarray(
                episode_actions, dtype=np.float32
            )
            arm_jerk = (
                np.diff(action_array[:, :5], n=3, axis=0)
                if len(action_array) > 3
                else np.zeros((0, 5), dtype=np.float32)
            )
            episode_metrics.append({
                "episode": episode,
                "seed": example["seed"],
                "task": task,
                "training_step": optimizer_steps + 1,
                "success": success,
                "completion_seconds": (
                    len(episode_actions) / sim.fps
                    if success else None
                ),
                "completion_efficiency_percent": (
                    100 * {"push": 3, "place": 5}[task]
                    / (decision + 1)
                    if success else None
                ),
                "rms_arm_jerk_degrees_per_frame3": (
                    float(np.sqrt(np.mean(arm_jerk ** 2)))
                    if arm_jerk.size else 0.
                ),
                "wrong_object": wrong_object,
            })
            if (
                accumulated_samples
                and (
                    (episode + 1) % episodes_per_step == 0
                    or episode + 1 == episodes
                )
            ):
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_samples)
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(parameters, 1.)
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                squared_drift = 0.
                parameter_count = 0
                for parameter, initial in zip(
                    parameters, initial_parameters, strict=True
                ):
                    difference = parameter.detach().cpu() - initial
                    squared_drift += float(difference.square().sum())
                    parameter_count += difference.numel()
                step_metrics.append({
                    "training_step": optimizer_steps,
                    "gradient_samples": accumulated_samples,
                    "gradient_norm": gradient_norm,
                    "parameter_rms_drift": (
                        squared_drift / max(parameter_count, 1)
                    ) ** .5,
                    "episodes_completed": episode + 1,
                })
                accumulated_samples = 0
    finally:
        sim.close()
    save_entity_checkpoint(
        output,
        policy,
        controller.config,
        "lerobot/smolvla_base",
        {
            "state_mean": controller.state_mean.tolist(),
            "state_std": controller.state_std.tolist(),
            "action_mean": controller.action_mean.tolist(),
            "action_std": controller.action_std.tolist(),
        },
    )
    (output / "rl_training.json").write_text(json.dumps({
        "algorithm": "oracle reward-weighted flow matching",
        "episodes": episodes,
        "optimizer_steps": optimizer_steps,
        "gradient_samples": gradient_samples,
        "candidates": candidates,
        "action_steps": action_steps,
        "episodes_per_step": episodes_per_step,
        "learning_rate": learning_rate,
        "expert_layers": expert_layers,
        "steps": step_metrics,
        "episode_metrics": episode_metrics,
        "rollouts": logs,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/simulation/command_heldout_40.json"),
    )
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--decisions", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--action-steps", type=int, default=16)
    parser.add_argument("--episodes-per-step", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--expert-layers", type=int, default=2)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda"), default="mps"
    )
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--seed", type=int, default=41_000)
    parser.add_argument(
        "--curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    train(
        args.checkpoint,
        args.output,
        args.benchmark,
        args.episodes,
        args.decisions,
        args.candidates,
        args.action_steps,
        args.episodes_per_step,
        args.learning_rate,
        args.expert_layers,
        args.device,
        args.model_root,
        args.seed,
        args.curriculum,
    )


if __name__ == "__main__":
    main()
