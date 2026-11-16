"""Bounded matched GRASP pilots using overshoot-safe expert-trajectory progress."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import mujoco
import numpy as np
import torch

from so101.calibrate_trajectory_rl import policy_digest
from so101.cloud_checkpoint import atomic_json, validate_state
from so101.control_model.jerk_regularizer import prefix_backward_batch
from so101.control_model.modeling_entity_smolvla import save_entity_checkpoint
from so101.evaluate_entity_sim import EntityPolicy
from so101.pile_task_runtime import body_surface_speed_bound
from so101.rl_renderer import create_simulation
from so101.robot_control import JOINTS
from so101.simulation_corpus import CorpusEpisode
from so101.train_trajectory_rl import (
    normalized_component_advantage, rollout, save_resume_state, serialized_trajectory,
    restore_resume_state, trainable_parameters, trajectory_backward,
)
from so101.trajectory_progress import (
    ProgressState, progress_potential, progress_reward, reward_percentage, short_horizon,
)


def physical_state(model, scratch, qpos, target, contact, speed):
    scratch.qpos[:] = qpos
    scratch.qvel[:] = 0
    mujoco.mj_forward(model, scratch)
    site = model.site("grasp").id
    position = np.asarray(target)
    return ProgressState(
        tuple(scratch.site_xpos[site] - position),
        tuple(scratch.site_xmat[site]),
        float(scratch.qpos[model.jnt_qposadr[model.joint(JOINTS[-1]).id]]), tuple(position),
        bool(contact["bilateral_grip"]), float(speed),
    )


class ProgressObserver:
    def __init__(self, sim):
        self.sim = sim
        self.states = []
        self.scratch = None

    def decision(self, *args):
        pass

    def write(self, event, data):
        if event not in ("applied_action", "rejected_action"):
            return
        native = self.sim._native_simulation
        if self.scratch is None:
            self.scratch = mujoco.MjData(native.model)
        if not self.states:
            pre = data["pre"]
            self.states.append(physical_state(
                native.model, self.scratch, pre["qpos"], pre["target_position"],
                pre["contacts"], 0.,
            ))
        if event == "rejected_action":
            return
        post = data["post"]
        body = native.native.registry[native.active_target]
        speed = body_surface_speed_bound(native.model, native.data, body)
        self.states.append(physical_state(
            native.model, self.scratch, post["qpos"], post["target_position"],
            post["contacts"], speed,
        ))


def load_reference(plan):
    content = Path(plan["trace"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != plan["trace_sha256"]:
        raise ValueError("Expert reference changed")
    trace = json.loads(content)
    if not trace["accepted"] or trace["task"] != "grasp" or trace["split"] != "train":
        raise ValueError("Expected accepted TRAIN GRASP reference")
    entry = trace["entry"]
    for name, expected in entry["files_sha256"].items():
        if hashlib.sha256((Path(entry["root"]) / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed native source: {name}")
    model = mujoco.MjModel.from_binary_path(str(Path(entry["root"]) / "model.mjb"))
    scratch = mujoco.MjData(model)
    rows = trace["rows"]
    reference = tuple(physical_state(
        model, scratch, row["post_qpos"], row["target_position"],
        row["contacts"], row["target_speed_m_s"],
    ) for row in rows)
    scene = json.loads((Path(entry["root"]) / "problem.json").read_text())
    episodes = []
    for level in (.7, .75, .8, .85, .9):
        task = next(task for task in scene["tasks"]
                    if task["assistance"]["value"] == level)
        episodes.append(CorpusEpisode(
            Path(entry["root"]) / "tasks" / task["task_id"],
            {"task": "grasp", "target": task["target_instance_id"],
             "native_entry": entry, "native_task_id": task["task_id"],
             "scene_family_id": entry["scene_family_id"],
             "parent_task_id": task["parent_task_id"],
             "task_assistance": task["assistance"], "reference_available": False},
        ))
    return trace, reference, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bounds", type=Path, required=True)
    parser.add_argument("--deadline", type=int, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=61000)
    parser.add_argument("--arms", nargs="+", choices=("sparse", "progress"),
                        default=("sparse", "progress"))
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--qualify-only", action="store_true")
    args = parser.parse_args()
    if (args.steps < 1 or args.group_size < 2 or args.learning_rate <= 0
            or args.deadline <= time.time() or args.checkpoint_every < 1
            or len(set(args.arms)) != len(args.arms)
            or (args.resume_state is not None and len(args.arms) != 1)):
        raise ValueError("Positive bounded pilot configuration required")
    plan = json.loads(args.plan.read_text())
    trace, reference, episodes = load_reference(plan)
    horizon = short_horizon(len(reference))
    teacher = progress_reward(list(reference), reference)
    stationary = progress_reward([reference[0], reference[0]], reference)
    if teacher["reward"] <= 0 or stationary["reward"] != 0:
        raise ValueError("Expert/negative-control reward qualification failed")
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "reward-qualification.json", {
        "teacher": teacher, "stationary": stationary, "horizon": horizon,
        "reference_steps": len(reference), "parent_task_id": trace["parent_task_id"],
        "levels": [.7, .75, .8, .85, .9],
        "scope": "Saved assistance anchors; progress follows measured recovery path, not a linear assistance clock",
    })
    if args.qualify_only:
        return
    bounds = json.loads(args.bounds.read_text())["expert_bounds"]["grasp"]
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    for arm in args.arms:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        controller = EntityPolicy(args.checkpoint, "cuda", True, full_state_checkpoint=True)
        parameters = trainable_parameters(controller, "all")
        before = policy_digest(controller.policy)
        if args.resume_state is None and before != plan["initial_policy_sha256"]:
            raise ValueError("Pilot initialization differs from the frozen original policy")
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.)
        output = args.output / ("curriculum_rl_241_" + arm)
        output.mkdir(exist_ok=True)
        log_path = output / "trajectory_rl.json"
        if log_path.exists() and json.loads(log_path.read_text())["steps"]:
            raise FileExistsError("Existing pilot steps require reconciliation, not restart")
        log = {
            "status": "training", "rl_estimator": "denoising-grpo", "validation": [],
            "run_config": {**vars(args), "output": str(output), "reward": arm,
                           "horizon": horizon, "action_steps": horizon,
                           "checkpoint": str(args.checkpoint), "plan": str(args.plan),
                           "model_root": str(args.model_root), "bounds": str(args.bounds)},
            "policy_sha256_before": before, "effective_updates": 0, "steps": [],
        }
        log["run_config"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in log["run_config"].items()
        }
        start_step = 0
        if args.resume_state is not None:
            state = torch.load(args.resume_state, map_location="cpu", weights_only=False, mmap=True)
            validate_state(state)
            if (state["log"]["run_config"]["reward"] != arm
                    or state["log"]["policy_sha256_after"] != before
                    or state["log"]["policy_sha256_before"] != plan["initial_policy_sha256"]
                    or state["log"]["run_config"]["horizon"] != horizon
                    or state["log"]["run_config"]["group_size"] != args.group_size):
                raise ValueError("Resume model, objective, reference initialization or rollout protocol differs")
            current_config = log["run_config"]
            log = restore_resume_state(state, controller, optimizer, rng, args.learning_rate)
            start_step = state["step"]
            if start_step >= args.steps or policy_digest(controller.policy) != before:
                raise ValueError("Resume must preserve the exported policy and advance its step count")
            log.update(status="training", run_config=current_config,
                       resume_source=str(args.resume_state), segment_policy_sha256_before=before)
            log.pop("policy_sha256_after", None)
            del state
            gc.collect()
        effective_updates_before = log["effective_updates"]
        atomic_json(log_path, log)
        sim = create_simulation(args.model_root, backend="gpu-pathtrace", samples=16)
        try:
            for step in range(start_step + 1, args.steps + 1):
                if time.time() >= args.deadline:
                    raise TimeoutError("Pilot deadline reached; recover current checkpoint")
                episode = episodes[(step - 1) % len(episodes)]
                trajectories, measurements = [], []
                started = time.monotonic()
                for index in range(args.group_size):
                    observer = ProgressObserver(sim)
                    trajectory = rollout(
                        sim, controller, episode, args.seed + step * 1000 + index,
                        horizon, horizon, bounds, True, 0., .1, causal_trace=observer,
                    )
                    if len(observer.states) >= 2:
                        measurement = progress_reward(
                            observer.states, reference,
                            invalid=trajectory.invalid_action_error is not None,
                            wrong_object=trajectory.wrong_object or trajectory.constraint_violation)
                    elif observer.states and trajectory.invalid_action_error is not None:
                        measurement = {
                            "reward": -.25, "no_applied_action": True,
                            **reward_percentage(-.25, progress_potential(observer.states[0], reference))}
                    else:
                        raise ValueError("No measured starting state for reward normalization")
                    trajectory.reward = (measurement["reward"] if arm == "progress"
                                         else float(trajectory.success))
                    trajectories.append(trajectory)
                    measurements.append({**measurement, "states": [
                        asdict(state) for state in observer.states],
                    })
                rewards = np.asarray([item.reward for item in trajectories], dtype=np.float32)
                advantages = normalized_component_advantage(rewards)
                optimizer.zero_grad(set_to_none=True)
                loss = 0.
                for trajectory, advantage in zip(trajectories, advantages, strict=True):
                    with prefix_backward_batch(controller):
                        loss += trajectory_backward(
                            controller, trajectory, float(advantage) / args.group_size)
                grad_norm = 0.
                if np.any(advantages):
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(
                        parameters, 1., error_if_nonfinite=True))
                    if not np.isfinite(loss) or grad_norm <= 0:
                        raise FloatingPointError("Nonzero advantages need finite effective gradients")
                    optimizer.step()
                    log["effective_updates"] += 1
                row = {"step": step, "assistance": episode.metadata["task_assistance"]["value"],
                       "rewards": rewards.tolist(), "advantages": advantages.tolist(),
                       "reward_percent_of_max": [
                           item["reward_percent_of_max"] if arm == "progress" else 100. * trajectory.reward
                           for item, trajectory in zip(measurements, trajectories, strict=True)],
                       "optimizer_update_applied": bool(np.any(advantages)),
                       "loss": loss, "gradient_norm": grad_norm,
                       "seconds": time.monotonic() - started,
                       "rollouts": [serialized_trajectory(item) for item in trajectories]}
                row["mean_reward_percent_of_max"] = float(np.mean(row["reward_percent_of_max"]))
                log["steps"].append(row)
                if step % len(episodes) == 0:
                    cycle = log["steps"][-len(episodes):]
                    row["balanced_reward_percent_of_max"] = float(np.mean([
                        item["mean_reward_percent_of_max"] for item in cycle]))
                    row["reward_percent_of_max_by_band"] = {
                        str(item["assistance"]): item["mean_reward_percent_of_max"] for item in cycle}
                atomic_json(output / f"step-{step:03}-measurements.json", measurements)
                atomic_json(log_path, log)
                if step % args.checkpoint_every == 0 and step != args.steps:
                    save_resume_state(output, controller, optimizer, rng, log)
                print(json.dumps({"arm": arm, **row}), flush=True)
                del trajectories, measurements
            log["policy_sha256_after"] = policy_digest(controller.policy)
            if (log["effective_updates"] > effective_updates_before
                    and log["policy_sha256_after"] == before):
                raise ValueError("Reported updates did not change policy parameters")
            save_entity_checkpoint(
                output, controller.policy, controller.config, metadata["source_checkpoint"],
                metadata["normalization"], exist_ok=True,
            )
            log["status"] = "completed"
            atomic_json(log_path, log)
            save_resume_state(output, controller, optimizer, rng, log)
        finally:
            native = getattr(sim, "_native_simulation", None)
            if native is not None:
                native.close()
            sim.close()
        del controller, optimizer, parameters, sim
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
