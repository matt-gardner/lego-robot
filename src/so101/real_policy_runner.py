"""Run a wrist-only SmolVLA policy continuously on the real SO-101."""
from __future__ import annotations

import argparse
import fcntl
import json
import signal
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from so101.robot_chat import ChatConfig, FollowerArm
from so101.robot_control import JOINTS
from so101.robot_demo import WristStream
from so101.robot_sim import _WristPolicy
from so101.target_perception import target_color


GENERAL_TARGET_CHECKPOINT = Path(
    "outputs/smolvla_target_color_combined_2000/checkpoints/001000/pretrained_model"
)
COLOR_SPECIALIST_CHECKPOINTS = {
    color: Path(
        f"outputs/smolvla_target_color_{color}_dedicated_1000/"
        "checkpoints/000500/pretrained_model"
    )
    for color in ("orange", "navy", "green", "brown", "gray", "teal")
}
COLOR_SPECIALIST_CHECKPOINTS["teal"] = Path(
    "outputs/smolvla_target_color_teal_recovery_1500/"
    "checkpoints/000500/pretrained_model"
)


def pickup_task(target: str) -> str:
    target = target.strip()
    if not target:
        raise ValueError("Target description must not be empty")
    return f"Pick up the {target}."


def resolve_policy_checkpoint(target: str, fallback: bool = False) -> Path:
    try:
        color = target_color(target)
    except ValueError:
        return GENERAL_TARGET_CHECKPOINT
    if fallback:
        return GENERAL_TARGET_CHECKPOINT
    return COLOR_SPECIALIST_CHECKPOINTS[color]


def bounded_action(
    action: np.ndarray,
    current: np.ndarray,
    offset: np.ndarray,
    bounds: np.ndarray,
    max_relative_target: float,
    max_boundary_projection: float,
) -> tuple[np.ndarray, float]:
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (len(JOINTS),) or not np.isfinite(action).all():
        raise ValueError("Policy returned an invalid action")
    desired = action - offset
    violation = float(max(np.max(bounds[:, 0] - desired), np.max(desired - bounds[:, 1]), 0))
    if violation > max_boundary_projection:
        raise ValueError(f"Policy target exceeds the real joint envelope by {violation:.2f} units")
    desired = np.clip(desired, bounds[:, 0], bounds[:, 1])
    desired = np.clip(desired, current - max_relative_target, current + max_relative_target)
    return desired, violation


def apply_contact_phase_guard(
    target: np.ndarray,
    current: np.ndarray,
    bounds: np.ndarray,
    max_relative_target: float,
    precontact_shoulder_target: float | None,
    contact_gripper_threshold: float | None,
    precontact_gripper_target: float | None = None,
    gripper_hold_target: float | None = None,
    postcontact_shoulder_target: float | None = None,
    contact_detected: bool | None = None,
) -> np.ndarray:
    if precontact_shoulder_target is None and contact_gripper_threshold is None:
        return target
    if precontact_shoulder_target is None or contact_gripper_threshold is None:
        raise ValueError("Pre-contact shoulder target and contact gripper threshold must be used together")
    if (
        not np.isfinite(precontact_shoulder_target)
        or not np.isfinite(contact_gripper_threshold)
        or (
            precontact_gripper_target is not None
            and not np.isfinite(precontact_gripper_target)
        )
        or (gripper_hold_target is not None and not np.isfinite(gripper_hold_target))
        or (
            postcontact_shoulder_target is not None
            and not np.isfinite(postcontact_shoulder_target)
        )
    ):
        raise ValueError("Contact phase guard values must be finite")
    guarded = target.copy()
    gripper_index = JOINTS.index("gripper")
    shoulder_index = JOINTS.index("shoulder_lift")
    in_contact = (
        current[gripper_index] <= contact_gripper_threshold
        if contact_detected is None
        else contact_detected
    )
    if not in_contact:
        lower = max(bounds[shoulder_index, 0], current[shoulder_index] - max_relative_target)
        upper = min(bounds[shoulder_index, 1], current[shoulder_index] + max_relative_target)
        guarded[shoulder_index] = np.clip(precontact_shoulder_target, lower, upper)
        if precontact_gripper_target is not None:
            lower = max(bounds[gripper_index, 0], current[gripper_index] - max_relative_target)
            upper = min(bounds[gripper_index, 1], current[gripper_index] + max_relative_target)
            guarded[gripper_index] = np.clip(precontact_gripper_target, lower, upper)
    elif postcontact_shoulder_target is not None:
        lower = max(bounds[shoulder_index, 0], current[shoulder_index] - max_relative_target)
        upper = min(bounds[shoulder_index, 1], current[shoulder_index] + max_relative_target)
        guarded[shoulder_index] = np.clip(postcontact_shoulder_target, lower, upper)
    if gripper_hold_target is not None:
        lower = max(bounds[gripper_index, 0], current[gripper_index] - max_relative_target)
        upper = min(bounds[gripper_index, 1], current[gripper_index] + max_relative_target)
        guarded[gripper_index] = np.clip(gripper_hold_target, lower, upper)
    return guarded


def update_contact_detection(
    gripper_position: float,
    contact_detected: bool,
    consecutive_samples: int,
    contact_gripper_threshold: float | None,
    contact_gripper_minimum: float | None,
    required_samples: int,
) -> tuple[bool, int]:
    if required_samples <= 0:
        raise ValueError("Contact persistence samples must be positive")
    if contact_gripper_threshold is None:
        return contact_detected, consecutive_samples
    if contact_detected:
        fell_below_minimum = contact_gripper_minimum is not None and (
            not np.isfinite(gripper_position)
            or gripper_position < contact_gripper_minimum
        )
        return (False, 0) if fell_below_minimum else (True, consecutive_samples)
    in_contact_range = (
        np.isfinite(gripper_position)
        and gripper_position <= contact_gripper_threshold
        and (
            contact_gripper_minimum is None
            or gripper_position >= contact_gripper_minimum
        )
    )
    consecutive_samples = consecutive_samples + 1 if in_contact_range else 0
    return consecutive_samples >= required_samples, consecutive_samples


class RealObservation:
    def __init__(self, state: np.ndarray, image: np.ndarray):
        self._state = state
        self._image = image

    def state(self) -> np.ndarray:
        return self._state

    def image(self) -> np.ndarray:
        return cv2.resize(self._image, (320, 240))


def orient_wrist_image(image: np.ndarray, flip_horizontal: bool) -> np.ndarray:
    return cv2.flip(image, 1) if flip_horizontal else image


def apply_target_pan_servo(
    target_state: np.ndarray,
    current: np.ndarray,
    bounds: np.ndarray,
    centroid_x: float,
    desired_x: float,
    pixels_per_degree: float,
    max_step: float,
) -> np.ndarray:
    if (
        not np.isfinite(centroid_x)
        or not np.isfinite(desired_x)
        or not np.isfinite(pixels_per_degree)
        or pixels_per_degree == 0
        or not np.isfinite(max_step)
        or max_step <= 0
    ):
        raise ValueError("Invalid target pan servo calibration")
    pan_index = JOINTS.index("shoulder_pan")
    correction = np.clip((desired_x - centroid_x) / pixels_per_degree, -max_step, max_step)
    adjusted = target_state.copy()
    adjusted[pan_index] = np.clip(
        current[pan_index] + correction,
        bounds[pan_index, 0],
        bounds[pan_index, 1],
    )
    return adjusted


def apply_target_flex_servo(
    target_state: np.ndarray,
    current: np.ndarray,
    bounds: np.ndarray,
    centroid_y: float,
    desired_y: float,
    pixels_per_degree: float,
    max_step: float,
) -> np.ndarray:
    if (
        not np.isfinite(centroid_y)
        or not np.isfinite(desired_y)
        or not np.isfinite(pixels_per_degree)
        or pixels_per_degree == 0
        or not np.isfinite(max_step)
        or max_step <= 0
    ):
        raise ValueError("Invalid target flex servo calibration")
    flex_index = JOINTS.index("wrist_flex")
    correction = np.clip((desired_y - centroid_y) / pixels_per_degree, -max_step, max_step)
    adjusted = target_state.copy()
    adjusted[flex_index] = np.clip(
        current[flex_index] + correction,
        bounds[flex_index, 0],
        bounds[flex_index, 1],
    )
    return adjusted


class DiagnosticVideo:
    def __init__(self, path: Path, camera_index: int, fps: int = 30):
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open diagnostic camera {camera_index}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not self.writer.isOpened():
            self.capture.release()
            raise RuntimeError(f"Cannot create diagnostic video {path}")
        self.stopping = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._record, daemon=True)
        self.thread.start()

    def _record(self) -> None:
        try:
            while not self.stopping.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    raise RuntimeError("Diagnostic camera failed during rollout")
                self.writer.write(frame)
        except BaseException as error:
            self.error = error

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError("Diagnostic video recording failed") from self.error

    def close(self) -> None:
        self.stopping.set()
        self.thread.join(timeout=5)
        self.capture.release()
        self.writer.release()
        if self.thread.is_alive():
            raise RuntimeError("Diagnostic video thread did not stop")
        self.check()


def run(
    checkpoint: Path,
    mapping_path: Path,
    bounds_path: Path,
    output: Path,
    target: str,
    device: str,
    fps: int,
    seconds: float,
    max_relative_target: float,
    max_boundary_projection: float,
    overview_camera_index: int,
    action_steps: int,
    precontact_shoulder_target: float | None,
    contact_gripper_threshold: float | None,
    precontact_gripper_target: float | None,
    gripper_hold_target: float | None,
    postcontact_shoulder_target: float | None,
    contact_gripper_minimum: float | None,
    contact_persistence_samples: int,
    cue_target: bool,
    flip_wrist_horizontal: bool = False,
    servo_target_pan: bool = False,
    target_pan_centroid_x: float = 320,
    target_pan_pixels_per_degree: float = -5.2,
    target_pan_max_step: float = 2,
    servo_target_flex: bool = False,
    target_flex_centroid_y: float = 208,
    target_flex_pixels_per_degree: float = -5,
    target_flex_max_step: float = 2,
) -> None:
    if (
        fps <= 0
        or seconds <= 0
        or max_relative_target <= 0
        or max_boundary_projection < 0
        or action_steps <= 0
        or contact_persistence_samples <= 0
    ):
        raise ValueError(
            "fps, seconds, max-relative-target, and action-steps must be positive; "
            "contact-persistence-samples must be positive; "
            "max-boundary-projection must be non-negative"
        )
    mapping = json.loads(mapping_path.read_text())
    bounds_json = json.loads(bounds_path.read_text())
    offset = np.array([mapping["real_to_sim_offsets"][name] for name in JOINTS], dtype=np.float64)
    bounds = np.array([bounds_json[name] for name in JOINTS], dtype=np.float64)
    if bounds.shape != (len(JOINTS), 2) or not np.all(bounds[:, 0] < bounds[:, 1]):
        raise ValueError("Invalid joint bounds")
    apply_contact_phase_guard(
        np.zeros(len(JOINTS)),
        np.zeros(len(JOINTS)),
        bounds,
        max_relative_target,
        precontact_shoulder_target,
        contact_gripper_threshold,
        precontact_gripper_target,
        gripper_hold_target,
        postcontact_shoulder_target,
    )
    if servo_target_pan:
        apply_target_pan_servo(
            np.zeros(len(JOINTS)),
            np.zeros(len(JOINTS)),
            bounds,
            target_pan_centroid_x,
            target_pan_centroid_x,
            target_pan_pixels_per_degree,
            target_pan_max_step,
        )
    if servo_target_flex:
        apply_target_flex_servo(
            np.zeros(len(JOINTS)),
            np.zeros(len(JOINTS)),
            bounds,
            target_flex_centroid_y,
            target_flex_centroid_y,
            target_flex_pixels_per_degree,
            target_flex_max_step,
        )
    if (
        contact_gripper_minimum is not None
        and (
            not np.isfinite(contact_gripper_minimum)
            or contact_gripper_threshold is None
            or contact_gripper_minimum > contact_gripper_threshold
        )
    ):
        raise ValueError(
            "Contact gripper minimum must be finite and no greater than the contact threshold"
        )

    policy = _WristPolicy(checkpoint, device, action_steps)
    task = pickup_task(target)
    policy.reset(0)
    output.mkdir(parents=True, exist_ok=False)
    wrist_frames = output / "wrist"
    wrist_frames.mkdir()
    cued_frames = output / "wrist_cued"
    if cue_target:
        cued_frames.mkdir()
    policy_frames = output / "wrist_policy"
    if flip_wrist_horizontal:
        policy_frames.mkdir()
    log = (output / "actions.jsonl").open("w")
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    lock_path = Path(tempfile.gettempdir()) / "so101-robot-chat.lock"
    arm = camera = video = None
    started = time.perf_counter()
    try:
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            video = DiagnosticVideo(output / "overview.mp4", overview_camera_index)
            arm = FollowerArm(ChatConfig().follower_port)
            camera = WristStream()
            arm.enable()
            deadline = time.perf_counter() + seconds
            step = 0
            contact_detected = False
            contact_samples = 0
            target_seen = False
            while not stop and time.perf_counter() < deadline:
                tick = time.perf_counter()
                video.check()
                positions = arm.positions()
                current = np.array([positions[name] for name in JOINTS], dtype=np.float64)
                contact_detected, contact_samples = update_contact_detection(
                    current[JOINTS.index("gripper")],
                    contact_detected,
                    contact_samples,
                    contact_gripper_threshold,
                    contact_gripper_minimum,
                    contact_persistence_samples,
                )
                image, camera_time = camera.latest()
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(
                    str(wrist_frames / f"{step:06}.jpg"),
                    bgr_image,
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                ):
                    raise RuntimeError("Failed to record wrist observation")
                cue_status = None
                target_observation = None
                if (
                    cue_target
                    or servo_target_pan
                    or servo_target_flex
                ):
                    from so101.target_perception import detect_target

                    try:
                        target_observation, annotated = detect_target(bgr_image, target)
                    except ValueError as perception_error:
                        if not target_seen:
                            log.write(json.dumps({
                                "step": step,
                                "elapsed": time.perf_counter() - started,
                                "camera_age_s": time.perf_counter() - camera_time,
                                "state": current.tolist(),
                                "perception_error": str(perception_error),
                                "command": current.tolist(),
                            }) + "\n")
                            log.flush()
                            policy.reset(step + 1)
                            step += 1
                            remaining = 1 / fps - (time.perf_counter() - tick)
                            if remaining > 0:
                                time.sleep(remaining)
                            continue
                        cue_status = "occluded"
                        if cue_target:
                            annotated = bgr_image.copy()
                            cv2.putText(
                                annotated, "TARGET OCCLUDED", (8, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255),
                                2, cv2.LINE_AA,
                            )
                    else:
                        target_seen = True
                        cue_status = "visible"
                if cue_target:
                    if not cv2.imwrite(
                        str(cued_frames / f"{step:06}.jpg"),
                        annotated,
                        [cv2.IMWRITE_JPEG_QUALITY, 85],
                    ):
                        raise RuntimeError("Failed to record cued wrist observation")
                    image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                image = orient_wrist_image(image, flip_wrist_horizontal)
                if flip_wrist_horizontal and not cv2.imwrite(
                    str(policy_frames / f"{step:06}.jpg"),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                ):
                    raise RuntimeError("Failed to record policy wrist observation")
                observation = RealObservation(current + offset, image)
                raw = policy.action(observation, target, task=task)
                target_state, projection = bounded_action(
                    raw,
                    current,
                    offset,
                    bounds,
                    max_relative_target,
                    max_boundary_projection,
                )
                target_state = apply_contact_phase_guard(
                    target_state,
                    current,
                    bounds,
                    max_relative_target,
                    precontact_shoulder_target,
                    contact_gripper_threshold,
                    precontact_gripper_target,
                    gripper_hold_target,
                    postcontact_shoulder_target,
                    contact_detected,
                )
                if servo_target_pan and not contact_detected:
                    pan_index = JOINTS.index("shoulder_pan")
                    if target_observation is None:
                        target_state[pan_index] = current[pan_index]
                    else:
                        target_state = apply_target_pan_servo(
                            target_state,
                            current,
                            bounds,
                            target_observation.centroid[0],
                            target_pan_centroid_x,
                            target_pan_pixels_per_degree,
                            target_pan_max_step,
                        )
                if servo_target_flex and not contact_detected:
                    flex_index = JOINTS.index("wrist_flex")
                    if target_observation is None:
                        target_state[flex_index] = current[flex_index]
                    else:
                        target_state = apply_target_flex_servo(
                            target_state,
                            current,
                            bounds,
                            target_observation.centroid[1],
                            target_flex_centroid_y,
                            target_flex_pixels_per_degree,
                            target_flex_max_step,
                        )
                command = dict(zip(JOINTS, target_state.tolist(), strict=True))
                arm.command(command)
                row = {
                    "step": step,
                    "elapsed": time.perf_counter() - started,
                    "inference_s": time.perf_counter() - tick,
                    "camera_age_s": time.perf_counter() - camera_time,
                    "state": current.tolist(),
                    "raw_model_action": np.asarray(raw).tolist(),
                    "command": target_state.tolist(),
                    "boundary_projection": projection,
                    "contact_detected": contact_detected,
                    "contact_samples": contact_samples,
                }
                if cue_status is not None:
                    row["target_cue_status"] = cue_status
                if target_observation is not None:
                    row["target_centroid"] = list(target_observation.centroid)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps({k: row[k] for k in ("step", "elapsed", "inference_s", "command")}), flush=True)
                step += 1
                remaining = 1 / fps - (time.perf_counter() - tick)
                if remaining > 0:
                    time.sleep(remaining)
            result = {
                "outcome": "interrupted" if stop else "horizon_complete",
                "steps": step,
                "duration_s": time.perf_counter() - started,
                "fps": fps,
                "horizon_s": seconds,
                "task": task,
                "checkpoint": str(checkpoint),
            }
            (output / "result.json").write_text(json.dumps(result, indent=2))
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        log.close()
        cleanup_error = None
        if arm is not None:
            try:
                if arm.bus.is_connected:
                    arm.bus.disable_torque(num_retry=5)
            except Exception as error:
                cleanup_error = error
            finally:
                arm.close()
        if camera is not None:
            camera.close()
        if video is not None:
            video.close()
        if cleanup_error is not None:
            raise cleanup_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Override automatic color-specialist policy routing.",
    )
    parser.add_argument(
        "--fallback-policy",
        action="store_true",
        help="Use the general target policy instead of the routed color specialist.",
    )
    parser.add_argument(
        "--mapping", type=Path, default=Path("data/real_policy/trial_001/mapping.json")
    )
    parser.add_argument(
        "--bounds",
        type=Path,
        default=Path("data/direct_grasp/alternating_20261024_1316/bounds.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="studded gray beam")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--max-relative-target", type=float, default=12)
    parser.add_argument("--max-boundary-projection", type=float, default=10)
    parser.add_argument("--overview-camera-index", type=int, default=1)
    parser.add_argument("--action-steps", type=int, default=16)
    parser.add_argument("--precontact-shoulder-target", type=float)
    parser.add_argument("--contact-gripper-threshold", type=float)
    parser.add_argument("--precontact-gripper-target", type=float)
    parser.add_argument("--gripper-hold-target", type=float)
    parser.add_argument("--postcontact-shoulder-target", type=float)
    parser.add_argument("--contact-gripper-minimum", type=float)
    parser.add_argument("--contact-persistence-samples", type=int, default=3)
    parser.add_argument("--cue-target", action="store_true")
    parser.add_argument("--flip-wrist-horizontal", action="store_true")
    parser.add_argument("--servo-target-pan", action="store_true")
    parser.add_argument("--target-pan-centroid-x", type=float, default=320)
    parser.add_argument("--target-pan-pixels-per-degree", type=float, default=-5.2)
    parser.add_argument("--target-pan-max-step", type=float, default=2)
    parser.add_argument("--servo-target-flex", action="store_true")
    parser.add_argument("--target-flex-centroid-y", type=float, default=208)
    parser.add_argument("--target-flex-pixels-per-degree", type=float, default=-5)
    parser.add_argument("--target-flex-max-step", type=float, default=2)
    args = parser.parse_args()
    if args.checkpoint is not None and args.fallback_policy:
        parser.error("--checkpoint and --fallback-policy cannot be used together")
    checkpoint = args.checkpoint or resolve_policy_checkpoint(
        args.target, fallback=args.fallback_policy
    )
    run(
        checkpoint,
        args.mapping,
        args.bounds,
        args.output,
        args.target,
        args.device,
        args.fps,
        args.seconds,
        args.max_relative_target,
        args.max_boundary_projection,
        args.overview_camera_index,
        args.action_steps,
        args.precontact_shoulder_target,
        args.contact_gripper_threshold,
        args.precontact_gripper_target,
        args.gripper_hold_target,
        args.postcontact_shoulder_target,
        args.contact_gripper_minimum,
        args.contact_persistence_samples,
        args.cue_target,
        args.flip_wrist_horizontal,
        args.servo_target_pan,
        args.target_pan_centroid_x,
        args.target_pan_pixels_per_degree,
        args.target_pan_max_step,
        args.servo_target_flex,
        args.target_flex_centroid_y,
        args.target_flex_pixels_per_degree,
        args.target_flex_max_step,
    )


if __name__ == "__main__":
    main()
