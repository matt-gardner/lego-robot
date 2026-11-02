import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest
from copilot.tools import ToolInvocation

import so101.robot_chat as robot_chat
import so101.robot_control as robot_control
from so101.robot_chat import make_tools
from so101.robot_control import MotionController, SimulatedArm


def test_nudge_moves_only_requested_joint():
    arm = SimulatedArm()
    controller = MotionController(arm, motion=True, sleep=lambda _: None)
    before = arm.positions()
    result = controller.nudge("wrist_roll", 3)
    assert result["actual_positions"] == {**before, "wrist_roll": 3}


def test_pose_moves_large_joint_targets_together_in_one_request():
    class RecordingArm(SimulatedArm):
        def __init__(self):
            super().__init__()
            self.commands = []

        def command(self, positions):
            self.commands.append(dict(positions))
            super().command(positions)

    arm = RecordingArm()
    bounds = {name: (-40, 40) for name in robot_control.JOINTS}
    bounds["gripper"] = (0, 100)
    sleeps = []
    controller = MotionController(arm, motion=True, bounds=bounds, sleep=sleeps.append)
    before = arm.positions()
    result = controller.move_to({"shoulder_pan": 20, "shoulder_lift": -10})
    assert result["outcome"] == "target_reached"
    assert result["actual_positions"] == {
        **before, "shoulder_pan": 20, "shoulder_lift": -10
    }
    assert result["measured_deltas"] == {"shoulder_pan": 20, "shoulder_lift": -10}
    assert controller.moves_left == 2
    assert len(arm.commands) == 80
    assert sum(sleeps) == pytest.approx(4)
    previous = before
    for command in arm.commands:
        assert command["shoulder_lift"] == -command["shoulder_pan"] / 2
        assert command["shoulder_pan"] - previous["shoulder_pan"] <= 0.25
        assert command["elbow_flex"] == before["elbow_flex"]
        previous = command


@pytest.mark.parametrize("targets", [{}, {"unknown": 2}, {"wrist_roll": float("nan")}])
def test_invalid_pose_never_moves(targets):
    arm = SimulatedArm()
    controller = MotionController(arm, motion=True)
    before = arm.positions()
    with pytest.raises(ValueError, match="known joints"):
        controller.move_to(targets)
    assert arm.positions() == before


def test_pose_preserves_joint_bounds_and_motion_gate():
    arm = SimulatedArm()
    controller = MotionController(arm, motion=False)
    with pytest.raises(RuntimeError, match="disabled"):
        controller.move_to({"wrist_roll": 10})
    controller.motion = True
    with pytest.raises(ValueError, match="startup bounds"):
        controller.move_to({"wrist_roll": 20})
    assert arm.positions()["wrist_roll"] == 0
    assert controller.stopped.is_set()


def test_pose_does_not_hide_a_stationary_joint_behind_another_joints_progress(monkeypatch):
    class OneStationaryJoint(SimulatedArm):
        def command(self, positions):
            before = self.values["elbow_flex"]
            super().command(positions)
            self.values["elbow_flex"] = before

    ticks = iter([0.0, 3.0])
    monkeypatch.setattr(robot_control.time, "monotonic", lambda: next(ticks))
    controller = MotionController(
        OneStationaryJoint(), motion=True, allow_partial=True, sleep=lambda _: None
    )
    with pytest.raises(TimeoutError, match="minimum_progress"):
        controller.move_to({"wrist_roll": 2, "elbow_flex": 2})
    assert controller.stopped.is_set()


def test_pose_reports_individual_errors_when_only_one_joint_is_partial(monkeypatch):
    class UndershootingArm(SimulatedArm):
        def command(self, positions):
            super().command(positions)
            self.values["wrist_roll"] -= 1.1

    ticks = iter([0.0, 3.0])
    monkeypatch.setattr(robot_control.time, "monotonic", lambda: next(ticks))
    controller = MotionController(
        UndershootingArm(), motion=True, allow_partial=True, sleep=lambda _: None
    )
    result = controller.move_to({"elbow_flex": 2, "wrist_roll": 2})
    assert result["outcome"] == "partial"
    assert result["target_errors"] == pytest.approx({"elbow_flex": 0, "wrist_roll": -1.1})
    assert result["measured_deltas"] == pytest.approx({"elbow_flex": 2, "wrist_roll": 0.9})
    assert not controller.stopped.is_set()


def test_pose_stops_on_excessive_tracking_error():
    class LaggingArm(SimulatedArm):
        def command(self, positions):
            super().command(positions)
            self.values["wrist_roll"] -= 9

    controller = MotionController(LaggingArm(), motion=True, sleep=lambda _: None)
    with pytest.raises(RuntimeError, match="tracking error"):
        controller.move_to({"wrist_roll": 10})
    assert controller.stopped.is_set()


def test_pose_detects_uncommanded_motion_during_interpolation():
    class DriftingArm(SimulatedArm):
        def command(self, positions):
            super().command(positions)
            self.values["elbow_flex"] += 3

    controller = MotionController(DriftingArm(), motion=True, sleep=lambda _: None)
    with pytest.raises(RuntimeError, match="uncommanded"):
        controller.move_to({"wrist_roll": 10})
    assert controller.stopped.is_set()


@pytest.mark.parametrize("mode", ["pose", "nudge", "pose_with_held_joint"])
def test_followup_preserves_loaded_joint_and_gripper_holding_goals(monkeypatch, mode):
    class LoadedArm(SimulatedArm):
        def __init__(self):
            super().__init__()
            self.commands = []

        def command(self, positions):
            self.commands.append(dict(positions))
            super().command(positions)
            self.values["shoulder_lift"] = max(0, positions["shoulder_lift"] - 4)
            self.values["gripper"] = max(7, positions["gripper"])

    ticks = iter([0.0, 3.0, 4.0])
    monkeypatch.setattr(robot_control.time, "monotonic", lambda: next(ticks))
    bounds = {name: (-180, 180) for name in robot_control.JOINTS}
    bounds["gripper"] = (0, 100)
    arm = LoadedArm()
    controller = MotionController(
        arm, motion=True, bounds=bounds, allow_partial=True, sleep=lambda _: None
    )
    first = controller.move_to({"shoulder_lift": 10, "gripper": 1})
    assert first["outcome"] == "partial"
    assert first["actual_positions"]["shoulder_lift"] == 6
    assert first["actual_positions"]["gripper"] == 7
    arm.commands.clear()
    if mode == "nudge":
        result = controller.nudge("wrist_roll", 3)
    else:
        targets = {"wrist_roll": 3}
        if mode == "pose_with_held_joint":
            targets.update(shoulder_lift=6, gripper=7)
        result = controller.move_to(targets)
    assert result["outcome"] == "target_reached"
    assert result["actual_positions"]["shoulder_lift"] == 6
    assert result["actual_positions"]["gripper"] == 7
    assert all(command["shoulder_lift"] == 10 for command in arm.commands)
    assert all(command["gripper"] == 1 for command in arm.commands)
    assert not controller.stopped.is_set()


def test_loaded_joint_trajectory_does_not_reset_its_previous_holding_goal(monkeypatch):
    class LoadedArm(SimulatedArm):
        def __init__(self):
            super().__init__()
            self.commands = []

        def command(self, positions):
            self.commands.append(dict(positions))
            super().command(positions)
            self.values["shoulder_lift"] = max(0, positions["shoulder_lift"] - 4)

    ticks = iter([0.0, 3.0, 4.0, 7.0])
    monkeypatch.setattr(robot_control.time, "monotonic", lambda: next(ticks))
    arm = LoadedArm()
    controller = MotionController(
        arm, motion=True, allow_partial=True, sleep=lambda _: None
    )
    assert controller.move_to({"shoulder_lift": 10})["outcome"] == "partial"
    arm.commands.clear()
    result = controller.move_to({"shoulder_lift": 15})
    assert result["outcome"] == "partial"
    assert result["actual_positions"]["shoulder_lift"] == 11
    assert len(arm.commands) == 20
    assert arm.commands[0]["shoulder_lift"] == pytest.approx(10.25)
    assert arm.commands[-1]["shoulder_lift"] == 15
    assert not controller.stopped.is_set()


def test_hold_does_not_reapply_a_stale_goal_after_external_drift():
    arm = SimulatedArm()
    controller = MotionController(arm, motion=True, sleep=lambda _: None)
    controller.move_to({"shoulder_lift": 5})
    arm.values["shoulder_lift"] += 3
    with pytest.raises(RuntimeError, match="between requests"):
        controller.move_to({"wrist_roll": 3})
    assert arm.positions()["wrist_roll"] == 0
    assert controller.stopped.is_set()


@pytest.mark.parametrize("delta", [0, 6, -6, float("nan"), float("inf")])
def test_invalid_delta_never_moves(delta):
    arm = SimulatedArm()
    controller = MotionController(arm, motion=True, sleep=lambda _: None)
    before = arm.positions()
    with pytest.raises(ValueError, match="delta"):
        controller.nudge("wrist_roll", delta)
    assert arm.positions() == before


def test_read_only_and_stop_prevent_motion():
    arm = SimulatedArm()
    controller = MotionController(arm, motion=False)
    with pytest.raises(RuntimeError, match="disabled"):
        controller.nudge("wrist_roll", 1)
    controller.motion = True
    controller.stop()
    with pytest.raises(RuntimeError, match="latched"):
        controller.nudge("wrist_roll", 1)
    assert arm.positions()["wrist_roll"] == 0


def test_turn_budget_and_startup_envelope():
    arm = SimulatedArm()
    controller = MotionController(arm, motion=True, sleep=lambda _: None)
    for _ in range(3):
        controller.nudge("wrist_roll", 5)
    with pytest.raises(ValueError, match="Three moves"):
        controller.nudge("wrist_roll", 1)
    controller.begin_turn()
    with pytest.raises(ValueError, match="startup"):
        controller.nudge("wrist_roll", 1)
    assert arm.positions()["wrist_roll"] == 15


def test_stop_interrupts_interpolation():
    arm = SimulatedArm()
    controller = MotionController(arm, motion=True)
    controller.sleep = lambda _: controller.stop()
    with pytest.raises(RuntimeError, match="Stopped during"):
        controller.nudge("wrist_roll", 5)
    assert 0 < arm.positions()["wrist_roll"] < 5


def test_hardware_failure_latches_stop():
    class BrokenArm(SimulatedArm):
        def command(self, positions):
            raise ConnectionError("serial disconnected")

    controller = MotionController(BrokenArm(), motion=True)
    with pytest.raises(ConnectionError, match="disconnected"):
        controller.nudge("wrist_roll", 1)
    with pytest.raises(RuntimeError, match="latched"):
        controller.nudge("wrist_roll", 1)


@pytest.mark.parametrize("allow_partial", [False, True])
def test_tolerance_does_not_allow_a_stationary_joint_to_report_success(monkeypatch, allow_partial):
    class StationaryArm(SimulatedArm):
        def command(self, positions):
            pass

    ticks = iter([0.0, 3.0])
    monkeypatch.setattr(robot_control.time, "monotonic", lambda: next(ticks))
    controller = MotionController(
        StationaryArm(), motion=True, sleep=lambda _: None, allow_partial=allow_partial
    )
    with pytest.raises(TimeoutError, match="minimum_progress"):
        controller.nudge("wrist_roll", 1)
    assert controller.stopped.is_set()


@pytest.mark.parametrize("allow_partial", [False, True])
def test_partial_feedback_is_explicit_and_opt_in(monkeypatch, allow_partial):
    class UndershootingArm(SimulatedArm):
        def command(self, positions):
            self.values = dict(positions)
            self.values["wrist_roll"] -= 1.1

    ticks = iter([0.0, 3.0])
    monkeypatch.setattr(robot_control.time, "monotonic", lambda: next(ticks))
    controller = MotionController(
        UndershootingArm(), motion=True, sleep=lambda _: None, allow_partial=allow_partial
    )
    if allow_partial:
        result = controller.nudge("wrist_roll", 2)
        assert result["outcome"] == "partial"
        assert result["measured_delta"] == pytest.approx(0.9)
        assert result["target_error"] == pytest.approx(-1.1)
        assert not controller.stopped.is_set()
    else:
        with pytest.raises(TimeoutError, match="Joint did not reach target"):
            controller.nudge("wrist_roll", 2)
        assert controller.stopped.is_set()


def test_gripper_stays_in_range():
    arm = SimulatedArm()
    arm.values["gripper"] = 99
    controller = MotionController(arm, motion=True)
    with pytest.raises(ValueError, match="0-100"):
        controller.nudge("gripper", 2)
    assert arm.positions()["gripper"] == 99


def test_drift_outside_envelope_is_not_commanded():
    arm = SimulatedArm()
    controller = MotionController(arm, motion=True)
    arm.values["elbow_flex"] = 20
    with pytest.raises(ValueError, match="Current pose"):
        controller.nudge("wrist_roll", 2)
    assert arm.positions()["wrist_roll"] == 0
    assert controller.stopped.is_set()


def test_invalid_feedback_latches_stop():
    arm = SimulatedArm()
    controller = MotionController(arm, motion=True)
    arm.values["wrist_roll"] = float("nan")
    with pytest.raises(RuntimeError, match="Invalid position"):
        controller.status()
    arm.values["wrist_roll"] = 0
    with pytest.raises(RuntimeError, match="latched"):
        controller.nudge("wrist_roll", 1)


def test_sdk_tools_return_image_bytes_and_execute_motion():
    arm = SimulatedArm()
    controller = MotionController(arm, motion=True, sleep=lambda _: None)
    frame = np.full((32, 32, 3), 125, dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", frame)
    assert ok
    tools = {
        tool.name: tool
        for tool in make_tools(controller, lambda: jpeg.tobytes(), simulated=True)
    }

    async def exercise():
        result = await tools["observe_scene"].handler(ToolInvocation(arguments={}))
        image = result.binary_results_for_llm[0]
        decoded = cv2.imdecode(
            np.frombuffer(base64.b64decode(image.data), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert np.abs(decoded.astype(float) - frame).mean() < 2
        await tools["nudge_joint"].handler(
            ToolInvocation(arguments={"joint": "wrist_roll", "delta": 2})
        )
        assert arm.positions()["wrist_roll"] == 2
        await tools["stop_robot"].handler(ToolInvocation(arguments={}))
        assert controller.stopped.is_set()

    asyncio.run(exercise())


@pytest.mark.parametrize("fail", [False, True])
def test_one_shot_chat_executes_tools_and_closes_arm(monkeypatch, fail):
    arm = SimulatedArm()
    arm.close = MagicMock()
    monkeypatch.setattr(robot_chat, "SimulatedArm", lambda: arm)
    monkeypatch.setattr(
        robot_chat, "read_prompt", AsyncMock(side_effect=AssertionError("No stdin"))
    )
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)

    async def send(prompt, *, timeout):
        assert prompt == "Move wrist_roll by +1 degree"
        await tools["nudge_joint"].handler(
            ToolInvocation(arguments={"joint": "wrist_roll", "delta": 1})
        )
        if fail:
            raise RuntimeError("Session failed")
        return SimpleNamespace(data=SimpleNamespace(content="Moved"))

    session.send_and_wait = AsyncMock(side_effect=send)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    tools = {}

    async def create_session(**kwargs):
        tools.update({tool.name: tool for tool in kwargs["tools"]})
        return session

    client.create_session = AsyncMock(side_effect=create_session)
    monkeypatch.setattr(robot_chat, "CopilotClient", lambda: client)
    config = robot_chat.ChatConfig(
        simulate=True, motion=True, prompt="Move wrist_roll by +1 degree"
    )
    if fail:
        with pytest.raises(RuntimeError, match="Session failed"):
            asyncio.run(robot_chat.chat(config))
    else:
        asyncio.run(robot_chat.chat(config))
    assert arm.positions()["wrist_roll"] == 1
    session.send_and_wait.assert_awaited_once()
    arm.close.assert_called_once()


@pytest.mark.parametrize("prompt", ["", "  ", "/stop", "/quit"])
def test_invalid_one_shot_prompt_never_connects(monkeypatch, prompt):
    arm = MagicMock()
    monkeypatch.setattr(robot_chat, "FollowerArm", arm)
    with pytest.raises(ValueError, match="--prompt"):
        asyncio.run(robot_chat.chat(robot_chat.ChatConfig(prompt=prompt)))
    arm.assert_not_called()
