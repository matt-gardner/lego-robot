from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

from lerobot.policies.common.vla_utils import pad_vector
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from so101.control_model.command import (
    ROLE_PROMPTS,
    TASK_PROMPTS,
    camera_point_from_projection,
)
from so101.control_model.modeling_entity_smolvla import (
    EntityModelConfig,
    load_entity_checkpoint,
    prepare_history_images,
)
from so101.robot_sim import BeamSimulation, IMAGE_SHAPE, MODEL_ROOT


def _normalize(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (value.astype(np.float32) - mean) / std


def _box_points(box: np.ndarray, width: int, height: int) -> np.ndarray:
    center_x, center_y, mean_extent, anisotropy_x, anisotropy_y = box
    diagonal = float(np.hypot(width, height))
    half_difference = float(np.hypot(anisotropy_x, anisotropy_y))
    major = mean_extent + half_difference
    minor = max(mean_extent - half_difference, 0)
    angle_degrees = np.rad2deg(np.arctan2(anisotropy_y, anisotropy_x) / 2)
    return cv2.boxPoints((
        (float(center_x * width), float(center_y * height)),
        (float(major * diagonal), float(minor * diagonal)),
        float(angle_degrees),
    ))


def _oriented_iou(first: np.ndarray, second: np.ndarray, width: int, height: int) -> float:
    first_points = _box_points(first, width, height).astype(np.float32)
    second_points = _box_points(second, width, height).astype(np.float32)
    intersection, _ = cv2.intersectConvexConvex(first_points, second_points)
    first_area = abs(float(cv2.contourArea(first_points)))
    second_area = abs(float(cv2.contourArea(second_points)))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0


def _draw_box(
    image: np.ndarray,
    box: np.ndarray,
    color: tuple[int, int, int],
    label: str,
) -> None:
    height, width = image.shape[:2]
    points = _box_points(box, width, height).round().astype(np.int32)
    cv2.polylines(image, [points], True, color, 2)
    label_x = int(np.clip(points[:, 0].min(), 4, width - 1))
    label_y = int(np.clip(points[:, 1].min() - 5, 16, height - 1))
    cv2.putText(
        image,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        .42,
        color,
        1,
        cv2.LINE_AA,
    )


class EntityPolicy:
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        depth_input: bool,
        consistent_noise: bool = False,
        action_samples: int = 1,
        sample_aggregation: str = "mean",
        phase_prompts: bool = False,
        zero_noise: bool = False,
        ensemble_checkpoints: tuple[Path, ...] = (),
        ensemble_action_samples: int | None = None,
        ensemble_tasks: tuple[str, ...] | None = None,
        ensemble_score_penalty: float = 0.,
        loaded_policy: tuple[SmolVLAPolicy, EntityModelConfig, dict] | None = None,
        full_state_checkpoint: bool = False,
    ):
        if action_samples <= 0:
            raise ValueError("action_samples must be positive")
        if sample_aggregation not in ("mean", "smoothest", "goal", "oracle"):
            raise ValueError("Unsupported sample aggregation")
        if loaded_policy is None:
            loaded_policy = (
                load_entity_checkpoint(checkpoint, device, full_state=True)
                if full_state_checkpoint else load_entity_checkpoint(checkpoint, device)
            )
        self.policy, self.config, normalization = loaded_policy
        self.ensemble_policies = []
        for ensemble_checkpoint in ensemble_checkpoints:
            ensemble_policy, ensemble_config, ensemble_normalization = (
                load_entity_checkpoint(ensemble_checkpoint, device)
            )
            if ensemble_config != self.config:
                raise ValueError(
                    "Ensemble checkpoints must use the same entity config"
                )
            for key, value in normalization.items():
                if not np.allclose(value, ensemble_normalization[key]):
                    raise ValueError(
                        "Ensemble checkpoints must use the same normalization"
                    )
            self.ensemble_policies.append(ensemble_policy)
        self.device = device
        self.depth_input = depth_input
        self.consistent_noise = consistent_noise
        self.action_samples = action_samples
        self.ensemble_action_samples = (
            action_samples
            if ensemble_action_samples is None
            else ensemble_action_samples
        )
        if self.ensemble_action_samples <= 0:
            raise ValueError("Ensemble action samples must be positive")
        self.ensemble_tasks = ensemble_tasks
        if ensemble_score_penalty < 0:
            raise ValueError("Ensemble score penalty must be non-negative")
        self.ensemble_score_penalty = ensemble_score_penalty
        self.sample_aggregation = sample_aggregation
        self.phase_prompts = phase_prompts
        self.zero_noise = zero_noise
        if consistent_noise and zero_noise:
            raise ValueError("Consistent random noise and zero noise are exclusive")
        self.state_mean = np.asarray(normalization["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(normalization["state_std"], dtype=np.float32)
        self.action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(normalization["action_std"], dtype=np.float32)
        self.tokenizer = self.policy.model.vlm_with_expert.processor.tokenizer
        self.images: deque[np.ndarray] = deque(maxlen=self.config.history_steps)
        self.states: deque[np.ndarray] = deque(maxlen=self.config.history_steps)
        self.actions: deque[np.ndarray] = deque(maxlen=self.config.history_steps)
        self.action_noise: torch.Tensor | None = None

    def reset(
        self,
        sim: BeamSimulation,
        seed: int,
        zero_noise: bool | None = None,
    ) -> None:
        torch.manual_seed(seed)
        use_zero_noise = self.zero_noise if zero_noise is None else zero_noise
        image = sim.image()
        state = sim.state()
        self.images = deque(
            (image.copy() for _ in range(self.config.history_steps)),
            maxlen=self.config.history_steps,
        )
        self.states = deque(
            (state.copy() for _ in range(self.config.history_steps)),
            maxlen=self.config.history_steps,
        )
        self.actions = deque(
            (state.copy() for _ in range(self.config.history_steps)),
            maxlen=self.config.history_steps,
        )
        self.action_noise = (
            torch.zeros(
                (
                    1,
                    self.policy.config.chunk_size,
                    self.policy.config.max_action_dim,
                ),
                device=self.device,
            )
            if use_zero_noise
            else self.policy.model.sample_noise(
                (
                    1,
                    self.policy.config.chunk_size,
                    self.policy.config.max_action_dim,
                ),
                torch.device(self.device),
            )
            if self.consistent_noise
            else None
        )

    def observe(self, sim: BeamSimulation, executed_action: np.ndarray) -> None:
        self.images.append(sim.image())
        self.states.append(sim.state())
        self.actions.append(executed_action.astype(np.float32))

    def _tokens(
        self, text: str | list[str], length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            [text] if isinstance(text, str) else text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=length,
        )
        return (
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device).bool(),
        )

    def command_inputs(
        self,
        sim: BeamSimulation,
        phrase: str,
        entities: list[tuple[str, str]],
        phase: str | None = None,
    ) -> tuple[torch.Tensor | list[torch.Tensor], ...]:
        if not entities or len(entities) > self.config.max_entities:
            raise ValueError("Invalid command entity count")
        images = torch.from_numpy(np.stack(self.images)).permute(0, 3, 1, 2)
        images = images.float().div(255).unsqueeze(0).to(self.device)
        image_list, image_masks = prepare_history_images(self.policy, images)
        history_state = torch.from_numpy(np.stack([
            _normalize(value, self.state_mean, self.state_std) for value in self.states
        ])).unsqueeze(0).to(self.device)
        history_action = torch.from_numpy(np.stack([
            _normalize(value, self.action_mean, self.action_std) for value in self.actions
        ])).unsqueeze(0).to(self.device)
        state = pad_vector(history_state[:, -1], self.policy.config.max_state_dim)
        task_text = TASK_PROMPTS[phrase]
        if self.phase_prompts:
            if phase is None:
                raise ValueError("Phase-conditioned policy requires a phase")
            task_text = f"{task_text} {phase}"
        task_tokens, task_mask = self._tokens(
            task_text, self.policy.config.tokenizer_max_length
        )
        padded_entities = entities + [
            (entities[0][0], "object")
        ] * (self.config.max_entities - len(entities))
        role_tokens, role_mask = self._tokens(
            [ROLE_PROMPTS[role] for _, role in padded_entities],
            self.config.role_token_length,
        )
        boxes = np.zeros((self.config.max_entities, 5), dtype=np.float32)
        visibility_values = np.zeros(self.config.max_entities, dtype=np.float32)
        depth_values = np.zeros(self.config.max_entities, dtype=np.float32)
        depth_known_values = np.zeros(self.config.max_entities, dtype=np.float32)
        camera_positions = np.zeros(
            (self.config.max_entities, 3), dtype=np.float32
        )
        for index, (entity, _) in enumerate(entities):
            oriented_box = sim.target_oriented_box(entity)
            boxes[index] = oriented_box[:5]
            visibility_values[index] = oriented_box[5]
            entity_depth, valid = sim.entity_camera_depth(entity)
            if self.depth_input and valid:
                depth_values[index] = entity_depth
                depth_known_values[index] = 1
                camera_positions[index] = camera_point_from_projection(
                    float(boxes[index, 0]),
                    float(boxes[index, 1]),
                    float(entity_depth),
                )
        box = torch.from_numpy(boxes).unsqueeze(0).to(self.device)
        visibility = torch.from_numpy(visibility_values).unsqueeze(0).to(self.device)
        depth = torch.from_numpy(depth_values).unsqueeze(0).to(self.device)
        depth_known = torch.from_numpy(depth_known_values).unsqueeze(0).to(self.device)
        camera_position = torch.from_numpy(camera_positions).unsqueeze(0).to(
            self.device
        )
        entity_mask = torch.zeros(
            (1, self.config.max_entities), dtype=torch.bool, device=self.device
        )
        entity_mask[:, :len(entities)] = True
        return (
            image_list,
            image_masks,
            task_tokens,
            task_mask,
            state,
            history_state,
            history_action,
            role_tokens[None],
            role_mask[None],
            box,
            visibility,
            depth,
            depth_known,
            camera_position,
            entity_mask,
        )

    @torch.inference_mode()
    def predict_command(
        self,
        sim: BeamSimulation,
        phrase: str,
        entities: list[tuple[str, str]],
        phase: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        inputs = self.command_inputs(sim, phrase, entities, phase)
        predictions = [
            self.policy.model.sample_actions_entity(
                *inputs,
                noise=self.action_noise,
            )
            for _ in range(self.action_samples)
        ]
        prediction_is_ensemble = [False] * len(predictions)
        if self.ensemble_tasks is None or phrase in self.ensemble_tasks:
            cpu_rng_state = torch.random.get_rng_state()
            device_rng_state = (
                torch.mps.get_rng_state()
                if self.device == "mps"
                else torch.cuda.get_rng_state()
                if self.device == "cuda"
                else None
            )
            ensemble_predictions = [
                policy.model.sample_actions_entity(
                    *inputs,
                    noise=self.action_noise,
                )
                for policy in self.ensemble_policies
                for _ in range(self.ensemble_action_samples)
            ]
            predictions.extend(ensemble_predictions)
            prediction_is_ensemble.extend(
                [True] * len(ensemble_predictions)
            )
            torch.random.set_rng_state(cpu_rng_state)
            if self.device == "mps":
                torch.mps.set_rng_state(device_rng_state)
            elif self.device == "cuda":
                torch.cuda.set_rng_state(device_rng_state)
        if len(predictions) == 1:
            actions, boxes, log_depths, visibility_logits = predictions[0]
        elif self.sample_aggregation == "mean":
            actions, boxes, log_depths, visibility_logits = (
                torch.stack(values).mean(dim=0)
                for values in zip(*predictions, strict=True)
            )
        elif self.sample_aggregation == "smoothest":
            low = sim.encode(sim.limits[:, 0])
            high = sim.encode(sim.limits[:, 1])
            previous = self.actions[-1]
            scores = []
            for candidate_actions, *_ in predictions:
                normalized = (
                    candidate_actions[0, :, :6].float().cpu().numpy()
                )
                candidate = (
                    normalized * self.action_std + self.action_mean
                )
                sequence = np.concatenate((previous[None], candidate))
                delta = np.diff(sequence[:, :5], axis=0)
                acceleration = np.diff(delta, axis=0)
                boundary = np.maximum(
                    low - candidate, 0
                ) + np.maximum(candidate - high, 0)
                scores.append(
                    float(np.mean(delta ** 2))
                    + 2 * float(np.mean(acceleration ** 2))
                    + 100 * float(np.mean(boundary ** 2))
                )
            actions, boxes, log_depths, visibility_logits = predictions[
                int(np.argmin(scores))
            ]
        elif self.sample_aggregation == "goal":
            if phrase not in ("push", "place") or len(entities) < 2:
                actions, boxes, log_depths, visibility_logits = predictions[0]
            else:
                scores = []
                for _, candidate_boxes, _, candidate_visibility in predictions:
                    relative_center = (
                        candidate_boxes[0, :, 0, :2]
                        - candidate_boxes[0, :, 1, :2]
                    )
                    terminal_distance = torch.linalg.vector_norm(
                        relative_center[-8:], dim=-1
                    ).mean()
                    visibility_penalty = (
                        1
                        - candidate_visibility[0, -8:, :2]
                            .sigmoid()
                            .mean()
                    )
                    scores.append(float(
                        terminal_distance + .05 * visibility_penalty
                    ))
                actions, boxes, log_depths, visibility_logits = predictions[
                    int(np.argmin(scores))
                ]
        else:
            if phase is None:
                raise ValueError("Oracle selection requires a control phase")
            target = entities[0][0]
            destination = entities[1][0] if len(entities) > 1 else None
            saved_qpos = sim.data.qpos.copy()
            saved_qvel = sim.data.qvel.copy()
            saved_ctrl = sim.data.ctrl.copy()
            saved_time = float(sim.data.time)
            post_baseline_state = None
            scores = []
            for candidate_index, (candidate_actions, *_) in enumerate(
                predictions
            ):
                if (
                    prediction_is_ensemble[candidate_index]
                    and post_baseline_state is not None
                ):
                    mujoco.mj_setState(
                        sim.model,
                        sim.data,
                        post_baseline_state,
                        mujoco.mjtState.mjSTATE_INTEGRATION,
                    )
                    mujoco.mj_forward(sim.model, sim.data)
                normalized = candidate_actions[0, :, :6].float().cpu().numpy()
                candidate = normalized * self.action_std + self.action_mean
                score = 1e6
                try:
                    for action in candidate[:16]:
                        goal, _ = sim.bounded_policy_action(action)
                        sim.step(goal, "oracle_candidate")
                    target_position = sim.position(target)
                    gripper_position = sim.data.site_xpos[sim.site].copy()
                    contacts = sim.contacts(target)
                    if phrase == "push" and destination is not None:
                        destination_position = sim.position(destination)
                        direction = destination_position[:2] - target_position[:2]
                        direction /= max(float(np.linalg.norm(direction)), 1e-9)
                        behind = target_position[:2] - .025 * direction
                        if phase == "approach":
                            score = float(np.linalg.norm(
                                gripper_position
                                - np.r_[behind, .055]
                            ))
                        elif phase == "lower":
                            score = float(np.linalg.norm(
                                gripper_position
                                - np.r_[behind, .014]
                            ))
                        else:
                            score = float(np.linalg.norm(
                                target_position[:2]
                                - destination_position[:2]
                            ))
                    elif phrase == "place" and destination is not None:
                        destination_error = float(np.linalg.norm(
                            target_position[:2]
                            - sim.position(destination)[:2]
                        ))
                        retained_penalty = (
                            0.
                            if contacts == {"fixed", "moving"}
                            else .25
                        )
                        if phase == "acquire":
                            score = float(np.linalg.norm(
                                gripper_position - target_position
                            )) + retained_penalty
                        elif phase == "lift":
                            score = -float(target_position[2]) + retained_penalty
                        elif phase == "carry":
                            score = destination_error + retained_penalty
                        elif phase == "lower":
                            score = (
                                destination_error
                                + abs(float(target_position[2]) - .025)
                                + retained_penalty
                            )
                        else:
                            score = destination_error + (
                                .25
                                if contacts == {"fixed", "moving"}
                                else 0.
                            )
                    else:
                        score = float(np.linalg.norm(
                            gripper_position - target_position
                        ))
                except (RuntimeError, ValueError):
                    score = 1e6
                finally:
                    sim.data.qpos[:] = saved_qpos
                    sim.data.qvel[:] = saved_qvel
                    sim.data.ctrl[:] = saved_ctrl
                    sim.data.time = saved_time
                    mujoco.mj_forward(sim.model, sim.data)
                    if (
                        candidate_index + 1 == self.action_samples
                        and any(prediction_is_ensemble)
                    ):
                        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
                        post_baseline_state = np.empty(
                            mujoco.mj_stateSize(sim.model, state_spec)
                        )
                        mujoco.mj_getState(
                            sim.model,
                            sim.data,
                            post_baseline_state,
                            state_spec,
                        )
                if prediction_is_ensemble[candidate_index]:
                    score += self.ensemble_score_penalty
                scores.append(score)
            if post_baseline_state is not None:
                mujoco.mj_setState(
                    sim.model,
                    sim.data,
                    post_baseline_state,
                    mujoco.mjtState.mjSTATE_INTEGRATION,
                )
                mujoco.mj_forward(sim.model, sim.data)
            actions, boxes, log_depths, visibility_logits = predictions[
                int(np.argmin(scores))
            ]
        normalized_actions = actions[0, :, :6].float().cpu().numpy()
        denormalized_actions = normalized_actions * self.action_std + self.action_mean
        return (
            denormalized_actions,
            boxes[0].float().cpu().numpy(),
            log_depths[0].float().exp().cpu().numpy(),
            visibility_logits[0].float().sigmoid().cpu().numpy(),
        )

    def predict(
        self, sim: BeamSimulation, target: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        actions, boxes, depths, visibility = self.predict_command(
            sim, "grasp", [(target, "object")]
        )
        return actions, boxes[:, 0], depths[:, 0], visibility[:, 0]


def evaluate(
    checkpoint: Path,
    output: Path,
    episodes: int,
    seed: int,
    horizon: int,
    action_steps: int,
    device: str,
    depth_input: bool,
    model_root: Path,
) -> None:
    if episodes <= 0 or horizon <= 0 or action_steps <= 0:
        raise ValueError("Episodes, horizon, and action steps must be positive")
    output.mkdir(parents=True, exist_ok=False)
    controller = EntityPolicy(checkpoint, device, depth_input)
    sim = BeamSimulation(model_root)
    results = []
    try:
        for episode in range(episodes):
            target = ("studded", "hollow")[episode % 2]
            other = "hollow" if target == "studded" else "studded"
            scene_seed = seed + episode // 2
            sim.reset(scene_seed, varied_layout=True, real_scene=True, color_targets=True)
            sim.active_target = target
            controller.reset(sim, scene_seed)
            episode_root = output / f"episode_{episode:03}"
            episode_root.mkdir()
            writer = cv2.VideoWriter(
                str(episode_root / "rollout.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                sim.fps,
                (640, 240),
            )
            if not writer.isOpened():
                raise RuntimeError("Failed to open rollout video writer")
            success = False
            wrong_object = False
            error = None
            retained = 0
            box_errors = []
            shape_errors = []
            box_ious = []
            depth_errors = []
            executed = 0
            try:
                while executed < horizon and not success and not wrong_object:
                    try:
                        actions, boxes, depths, visibilities = controller.predict(sim, target)
                    except (RuntimeError, ValueError) as exception:
                        error = str(exception)
                        break
                    for index in range(min(action_steps, len(actions), horizon - executed)):
                        try:
                            goal, _ = sim.bounded_policy_action(actions[index])
                        except ValueError as exception:
                            error = str(exception)
                            break
                        sim.step(goal, "entity_policy")
                        controller.observe(sim, actions[index])
                        executed += 1
                        actual_box = np.asarray(
                            sim.target_oriented_box(target), dtype=np.float32
                        )
                        predicted_box = boxes[index]
                        if actual_box[5] > .5:
                            box_errors.append(float(np.linalg.norm(
                                predicted_box[:2] - actual_box[:2]
                            )))
                            shape_errors.append(float(np.linalg.norm(
                                predicted_box[2:5] - actual_box[2:5]
                            )))
                            if visibilities[index] >= .5:
                                box_ious.append(_oriented_iou(
                                    predicted_box,
                                    actual_box[:5],
                                    IMAGE_SHAPE[1],
                                    IMAGE_SHAPE[0],
                                ))
                        actual_depth = sim.target_depth(target)
                        depth_errors.append(abs(float(depths[index]) - actual_depth))
                        wrist = cv2.cvtColor(sim.image(), cv2.COLOR_RGB2BGR)
                        _draw_box(wrist, actual_box[:5], (0, 255, 0), "observed")
                        if visibilities[index] >= .5:
                            _draw_box(
                                wrist,
                                predicted_box,
                                (255, 180, 0),
                                f"pred {depths[index]:.3f}m",
                            )
                        overview = cv2.cvtColor(sim.overview_image(), cv2.COLOR_RGB2BGR)
                        cv2.putText(
                            overview,
                            f"target={target} step={executed}",
                            (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        writer.write(np.concatenate((overview, wrist), axis=1))
                        target_lifted = (
                            sim.position(target)[2] - sim.initial["objects"][target][2] >= .05
                        )
                        other_lifted = (
                            sim.position(other)[2] - sim.initial["objects"][other][2] >= .05
                        )
                        if other_lifted and sim.contacts(other) == {"fixed", "moving"}:
                            wrong_object = True
                            break
                        retained = (
                            retained + 1
                            if target_lifted and sim.contacts(target) == {"fixed", "moving"}
                            else 0
                        )
                        if retained >= sim.fps:
                            success = True
                            break
                    if error is not None:
                        break
            finally:
                writer.release()
            result = {
                "seed": scene_seed,
                "target": target,
                "success": success,
                "wrong_object": wrong_object,
                "steps": executed,
                "error": error,
                "mean_box_center_error": float(np.mean(box_errors)) if box_errors else None,
                "mean_box_shape_error": float(np.mean(shape_errors)) if shape_errors else None,
                "mean_oriented_box_iou": float(np.mean(box_ious)) if box_ious else None,
                "mean_depth_error_m": float(np.mean(depth_errors)) if depth_errors else None,
                "video": str(episode_root / "rollout.mp4"),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            (episode_root / "result.json").write_text(json.dumps(result, indent=2))
            (output / "evaluation.json").write_text(json.dumps({
                "checkpoint": str(checkpoint),
                "depth_input": depth_input,
                "success_rate": sum(row["success"] for row in results) / len(results),
                "wrong_object_rate": sum(row["wrong_object"] for row in results) / len(results),
                "episodes": results,
            }, indent=2))
    finally:
        sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--action-steps", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--depth-input", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        args.output,
        args.episodes,
        args.seed,
        args.horizon,
        args.action_steps,
        args.device,
        args.depth_input,
        args.model_root,
    )


if __name__ == "__main__":
    main()
