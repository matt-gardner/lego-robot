"""Persistent local control with continuous wrist-only demonstration recording."""
from __future__ import annotations

import argparse
import fcntl
import json
import signal
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from so101.robot_chat import ChatConfig, FollowerArm
from so101.robot_control import MotionController
from so101.robot_demo import EpisodeRecorder, WristStream


def serve(root: Path, bounds_path: Path, speed: float):
    commands, results = root / "commands", root / "results"
    commands.mkdir(parents=True, exist_ok=True)
    results.mkdir(exist_ok=True)
    if any(results.iterdir()):
        raise ValueError("Use a fresh run directory; command replay is not automatic")
    stopping = False
    recorder = None
    arm = None
    camera = None
    overview_camera = None
    controller = None
    goal = {}
    fps = 10
    alignment_confirmed = False
    closure_pending = False
    last_view = None

    def stop(*_):
        nonlocal stopping
        stopping = True
        if controller is not None:
            controller.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def record_step(state, action):
        nonlocal goal
        goal = dict(action)
        if recorder is not None:
            recorder.sample(state, action)

    def observe(number):
        nonlocal last_view
        for name, stream in (("wrist", camera), ("overview", overview_camera)):
            rgb, _ = stream.latest()
            if not cv2.imwrite(str(root / f"{name}_{number}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Failed to save {name} observation")
            # Preserve originals; stationary low-light diagnostics never enter episodes.
            if recorder is None and rgb.mean() < 65:
                frames = [rgb]
                for _ in range(11):
                    time.sleep(.04)
                    frame, _ = stream.latest()
                    frames.append(frame)
                averaged = np.mean(frames, axis=0).astype(np.uint8)
                gain = min(5, 100 / max(float(averaged.mean()), 1))
                corrected = cv2.convertScaleAbs(averaged, alpha=gain)
                if not cv2.imwrite(
                    str(root / f"{name}_{number}_lowlight.jpg"),
                    cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR),
                ):
                    raise RuntimeError(f"Failed to save corrected {name} observation")
        last_view = {
            "image": f"wrist_{number}.jpg",
            "time": time.perf_counter(),
            "positions": controller.checked_positions(),
        }

    def execute(command):
        nonlocal recorder, stopping, goal, alignment_confirmed, closure_pending
        action = command["action"]
        if action == "start":
            if recorder is not None:
                raise ValueError("An episode is already recording")
            name = command["episode"]
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("Episode must be a single directory name")
            recorder = EpisodeRecorder(
                root / "episodes" / name, camera, fps=fps,
                task=command.get("task", "Pick up a gray Lego piece"),
            )
            recorder.event({"type": "start", "positions": arm.positions()})
            return {"recording": name}
        if action == "move_to":
            state = controller.checked_positions()
            targets = command["targets"]
            closing_gripper = targets.get("gripper", state["gripper"]) < state["gripper"] - 1
            moving_arm = any(
                abs(value - state[name]) > 1 for name, value in targets.items() if name != "gripper"
            )
            if closing_gripper and (not alignment_confirmed or moving_arm):
                raise ValueError("Inspect and confirm alignment before a separate close request")
            if closure_pending and moving_arm:
                raise ValueError("Inspect and confirm the closed gripper before lifting")
            controller.begin_turn()
            if recorder is not None:
                recorder.event({"type": "move_start", "targets": command["targets"]})
            feedback = controller.move_to(command["targets"])
            alignment_confirmed = False
            if closing_gripper:
                closure_pending = True
            elif targets.get("gripper", state["gripper"]) > state["gripper"] + 1:
                closure_pending = False
            if recorder is not None:
                recorder.event({"type": "move_end", "feedback": feedback})
            return feedback
        if action == "policy_chunk":
            if not alignment_confirmed:
                raise ValueError("Inspect and confirm alignment before running a policy chunk")
            targets = command["targets"]
            if not 1 <= len(targets) <= 16:
                raise ValueError("Policy chunks must contain between 1 and 16 actions")
            feedback = []
            for target in targets:
                controller.begin_turn()
                feedback.append(controller.move_to(target))
            alignment_confirmed = False
            closure_pending = True
            return {"outcome": "policy_chunk_complete", "steps": feedback}
        if action == "confirm":
            if not command.get("reason", "").strip():
                raise ValueError("Confirmation requires the visual evidence and reasoning")
            if last_view is None or command.get("image") != last_view["image"]:
                raise ValueError("Confirmation must reference the latest wrist observation")
            if time.perf_counter() - last_view["time"] > 60:
                raise ValueError("Observation is stale; observe again before confirming")
            state = controller.checked_positions()
            if any(abs(state[name] - value) > 1 for name, value in last_view["positions"].items()):
                raise ValueError("Robot moved since observation; observe again before confirming")
            if command["checkpoint"] == "alignment":
                if closure_pending:
                    raise ValueError("Closed gripper needs inspection before more approach motion")
                alignment_confirmed = True
            elif command["checkpoint"] == "closure":
                if not closure_pending:
                    raise ValueError("No closure awaits inspection")
                closure_pending = False
            else:
                raise ValueError("Unknown checkpoint")
            if recorder is not None:
                recorder.event({"type": "visual_confirmation", **command})
            return {"confirmed": command["checkpoint"]}
        if action == "hold":
            duration = float(command["seconds"])
            if not 0 < duration <= 10:
                raise ValueError("Hold must be within (0, 10] seconds")
            deadline = time.perf_counter() + duration
            while time.perf_counter() < deadline and not stopping:
                tick = time.perf_counter()
                record_step(controller.checked_positions(), goal)
                time.sleep(max(0, 1 / fps - (time.perf_counter() - tick)))
            return {"held_seconds": duration}
        if action == "sequence":
            if any(step["action"] in ("confirm", "sequence") for step in command["steps"]):
                raise ValueError("Inspection confirmations must follow actual image review, not a queued sequence")
            return {"segments": [execute(step) for step in command["steps"]]}
        if action == "finish":
            if recorder is None:
                raise ValueError("No episode is recording")
            report = recorder.finish(command.get("outcome", "unverified"), command.get("evidence", ""))
            recorder = None
            return report
        if action == "observe":
            return {"outcome": "observation"}
        if action == "stop":
            stopping = True
            return {"outcome": "stopped"}
        raise ValueError(f"Unknown action: {action}")

    with (Path(tempfile.gettempdir()) / "so101-robot-chat.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            bounds = json.loads(bounds_path.read_text())
            arm = FollowerArm(ChatConfig().follower_port)
            camera = WristStream()
            overview_camera = WristStream(1)
            controller = MotionController(
                arm, motion=True, bounds={name: tuple(pair) for name, pair in bounds.items()},
                allow_partial=True, speed=speed, step_hz=fps, on_step=record_step,
            )
            goal = controller.checked_positions()
            observe("0000")
            (results / "0000.json").write_text(json.dumps(controller.status(), indent=2))
            print("READY", json.dumps(goal), flush=True)
            processed = {"0000"}
            deadline = time.perf_counter() + 1800
            while not stopping and time.perf_counter() < deadline:
                tick = time.perf_counter()
                for path in sorted(commands.glob("*.json")):
                    if path.stem in processed:
                        continue
                    command = json.loads(path.read_text())
                    feedback = execute(command)
                    if not stopping:
                        observe(path.stem)
                    result = {"time": time.time(), "command": command,
                              "feedback": feedback, "status": controller.status()}
                    (results / path.name).write_text(json.dumps(result, indent=2))
                    print(path.stem, json.dumps(feedback), flush=True)
                    processed.add(path.stem)
                    if stopping:
                        break
                if recorder is not None:
                    if recorder.rows:
                        remaining = 1 / fps - (
                            time.perf_counter() - recorder.start - recorder.rows[-1]["elapsed"]
                        )
                        if remaining > 0:
                            time.sleep(remaining)
                    record_step(controller.checked_positions(), goal)
                time.sleep(max(0, 1 / fps - (time.perf_counter() - tick)))
        except BaseException:
            (root / "failure.txt").write_text(traceback.format_exc())
            raise
        finally:
            try:
                if recorder is not None and not recorder.closed:
                    recorder.finish("aborted", "Controller stopped before episode review")
            finally:
                try:
                    if controller is not None:
                        controller.close()
                    elif arm is not None:
                        arm.close()
                finally:
                    if camera is not None:
                        camera.close()
                    if overview_camera is not None:
                        overview_camera.close()
                    (root / "closed.json").write_text(json.dumps({"time": time.time()}))
                    print("CLOSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bounds", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=10)
    args = parser.parse_args()
    serve(args.root, args.bounds, args.speed)
