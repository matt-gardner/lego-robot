import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import so101.robot_teacher as robot_teacher
from so101.robot_control import JOINTS, MotionController, SimulatedArm
from so101.robot_teacher import Finish, Teacher, TeacherConfig, TeacherStep


def make_teacher(tmp_path, **config):
    controller = MotionController(SimulatedArm(), motion=True, sleep=lambda _: None)
    return Teacher(
        TeacherConfig(workspace=str(tmp_path), **config), controller, lambda: b"camera jpeg"
    )


def test_findings_persist_between_sessions(tmp_path):
    teacher = make_teacher(tmp_path)
    teacher.note("Positive shoulder pan shifted the scene left; recheck next run.")
    another = make_teacher(tmp_path)
    assert another.run != teacher.run
    assert "Positive shoulder pan" in another.findings()
    assert teacher.run_id in another.findings()


def test_move_requires_observation_and_one_move_per_turn(tmp_path):
    teacher = make_teacher(tmp_path)
    step = TeacherStep(joint="wrist_roll", delta=1, hypothesis="Check rotation")

    async def exercise():
        with pytest.raises(ValueError, match="Observe"):
            await teacher.move(step)
        await teacher.observe()
        result = await teacher.move(step)
        assert result.binary_results_for_llm
        assert len(teacher.scenes) == 2
        with pytest.raises(ValueError, match="budget"):
            await teacher.move(step)
        teacher.turn_moves = 0
        teacher.controller.begin_turn()
        await teacher.move(step)

    asyncio.run(exercise())
    assert teacher.controller.arm.positions()["wrist_roll"] == 2
    events = [json.loads(line) for line in teacher.events.read_text().splitlines()]
    assert sum(event["kind"] == "motion_result" for event in events) == 2


def test_total_budget_cannot_be_reset_by_new_turn(tmp_path):
    teacher = make_teacher(tmp_path, max_steps=1)

    async def exercise():
        await teacher.observe()
        await teacher.move(TeacherStep(joint="wrist_roll", delta=1, hypothesis="Check"))
        teacher.turn_moves = 0
        teacher.controller.begin_turn()
        with pytest.raises(ValueError, match="budget"):
            await teacher.move(TeacherStep(joint="wrist_roll", delta=1, hypothesis="Check"))

    asyncio.run(exercise())


def test_camera_failure_after_motion_latches_stop(tmp_path):
    teacher = make_teacher(tmp_path)

    async def exercise():
        await teacher.observe()
        teacher.capture = MagicMock(side_effect=RuntimeError("Camera lost"))
        with pytest.raises(RuntimeError, match="Camera lost"):
            await teacher.move(TeacherStep(joint="wrist_roll", delta=1, hypothesis="Check"))

    asyncio.run(exercise())
    assert teacher.controller.stopped.is_set()
    assert "motion_error" in teacher.events.read_text()


def test_candidate_requires_real_motion_and_recorded_images(tmp_path):
    teacher = make_teacher(tmp_path)
    with pytest.raises(ValueError, match="Evidence"):
        teacher.finish(Finish(outcome="candidate_grasp", reason="Claim", evidence_scene_ids=["fake"]))

    async def exercise():
        await teacher.observe()
        await teacher.move(TeacherStep(joint="wrist_roll", delta=1, hypothesis="Check"))

    asyncio.run(exercise())
    teacher.finish(
        Finish(outcome="candidate_grasp", reason="Review these images", evidence_scene_ids=list(teacher.scenes))
    )
    assert teacher.outcome["independently_verified"] is False
    assert json.loads((teacher.run / "outcome.json").read_text())["outcome"] == "candidate_grasp"


def test_simulation_cannot_claim_candidate_grasp(tmp_path):
    teacher = make_teacher(tmp_path, simulate=True)

    async def exercise():
        await teacher.observe()
        await teacher.move(TeacherStep(joint="wrist_roll", delta=1, hypothesis="Check"))

    asyncio.run(exercise())
    with pytest.raises(ValueError, match="real motion"):
        teacher.finish(
            Finish(outcome="candidate_grasp", reason="Claim", evidence_scene_ids=list(teacher.scenes))
        )


def test_finished_attempt_prevents_further_movement(tmp_path):
    teacher = make_teacher(tmp_path)
    teacher.finish(Finish(outcome="blocked", reason="No target visible"))
    with pytest.raises(RuntimeError, match="finished"):
        asyncio.run(teacher.move(TeacherStep(joint="wrist_roll", delta=1, hypothesis="Check")))


def test_explicit_bounds_preserve_gripper_range_and_arm_limits():
    arm = SimulatedArm()
    bounds = {name: (-20.0, 20.0) for name in JOINTS}
    bounds["gripper"] = (0.0, 60.0)
    controller = MotionController(arm, motion=True, bounds=bounds, sleep=lambda _: None)
    for _ in range(4):
        controller.begin_turn()
        controller.nudge("shoulder_pan", 5)
    with pytest.raises(ValueError, match="bounds"):
        controller.nudge("shoulder_pan", 1)
    assert arm.positions()["shoulder_pan"] == 20
    assert controller.stopped.is_set()


@pytest.mark.parametrize("tool_only", [False, True])
def test_teacher_session_uses_workspace_and_restricted_tools(tmp_path, monkeypatch, tool_only):
    arm = SimulatedArm()
    arm.close = MagicMock()
    monkeypatch.setattr(robot_teacher, "SimulatedArm", lambda: arm)
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    tools = {}

    async def send(prompt, *, attachments, timeout):
        assert "Prior findings" in prompt
        assert attachments[0]["type"] == "blob"
        await tools["record_finding"].handler(
            SimpleNamespace(arguments={"text": "Need a real camera to evaluate grasp"})
        )
        await tools["finish_attempt"].handler(
            SimpleNamespace(arguments={"outcome": "blocked", "reason": "Simulation"})
        )
        return None if tool_only else SimpleNamespace(data=SimpleNamespace(content="Stopped in simulation"))

    session.send_and_wait = AsyncMock(side_effect=send)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)

    async def create(**kwargs):
        assert kwargs["working_directory"] == str(tmp_path)
        assert kwargs["available_tools"] == [tool.name for tool in kwargs["tools"]]
        assert kwargs["enable_config_discovery"] is False
        tools.update({tool.name: tool for tool in kwargs["tools"]})
        return session

    client.create_session = AsyncMock(side_effect=create)
    monkeypatch.setattr(robot_teacher, "CopilotClient", lambda: client)
    run = asyncio.run(
        robot_teacher.teach(TeacherConfig(workspace=str(tmp_path), simulate=True))
    )
    assert (run / "outcome.json").exists()
    assert "Need a real camera" in (tmp_path / "findings.jsonl").read_text()
    arm.close.assert_called_once()
