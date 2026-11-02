import json
import time

import numpy as np
import pytest

from so101.robot_control import JOINTS, MotionController, SimulatedArm
from so101.robot_demo import EpisodeRecorder, export_episodes


class FakeCamera:
    def latest(self):
        return np.full((480, 640, 3), [250, 20, 5], dtype=np.uint8), time.perf_counter()


def test_continuous_wrist_only_recording_preserves_state_actions_and_timing(tmp_path):
    camera = FakeCamera()
    recorder = EpisodeRecorder(tmp_path / "episode", camera, task="Grasp gray")
    state = dict.fromkeys(JOINTS, 0.0)
    action = {**state, "shoulder_pan": 2}
    recorder.sample(state, action)
    time.sleep(.1)
    recorder.sample(action, action)
    report = recorder.finish("unverified", "needs visual inspection")
    rows = [json.loads(line) for line in (recorder.root / "frames.jsonl").read_text().splitlines()]
    assert report["camera_features"] == ["observation.images.wrist"]
    assert rows[0]["action"][0] == 2
    assert rows[0]["state"][0] == 0
    assert rows[1]["elapsed"] > rows[0]["elapsed"]
    assert all((recorder.root / row["image"]).exists() for row in rows)
    with pytest.raises(ValueError, match="verified"):
        export_episodes([recorder.root], tmp_path / "dataset", "local/test")


def test_faster_coordinated_control_records_each_command():
    arm = SimulatedArm()
    observed = []
    sleeps = []
    controller = MotionController(
        arm, motion=True, speed=10, step_hz=10,
        on_step=lambda state, action: observed.append((dict(state), dict(action))),
        sleep=sleeps.append,
    )
    controller.move_to({"shoulder_pan": 10, "elbow_flex": -5})
    assert len(observed) == 11
    assert observed[0][0]["shoulder_pan"] == 0
    assert observed[0][1]["shoulder_pan"] == 1
    assert observed[-1][0]["shoulder_pan"] == 10
    assert observed[-1][1]["elbow_flex"] == -5
    assert sum(sleeps) <= 1


def test_camera_failure_stops_motion():
    controller = MotionController(
        SimulatedArm(), motion=True,
        on_step=lambda *_: (_ for _ in ()).throw(RuntimeError("camera stale")),
    )
    with pytest.raises(RuntimeError, match="camera stale"):
        controller.move_to({"shoulder_pan": 5})
    assert controller.stopped.is_set()
    assert controller.arm.positions()["shoulder_pan"] == 0


def test_absolute_pose_holds_already_aligned_joints():
    arm = SimulatedArm()
    arm.values["wrist_roll"] = 0.3
    controller = MotionController(arm, motion=True, sleep=lambda _: None)
    result = controller.move_to({"wrist_roll": 0.5, "shoulder_pan": 5})
    assert result["target_positions"]["wrist_roll"] == 0.3
    assert result["measured_deltas"] == {"shoulder_pan": 5}


def test_large_timing_gaps_fail_qualification(tmp_path):
    recorder = EpisodeRecorder(tmp_path / "episode", FakeCamera(), task="Grasp gray")
    state = dict.fromkeys(JOINTS, 0.0)
    recorder.sample(state, state)
    recorder.start -= 1
    recorder.sample(state, state)
    assert not recorder.finish("success", "test")["timing_acceptable"]


def test_already_reached_absolute_pose_does_not_stop_or_command():
    arm = SimulatedArm()
    observed = []
    controller = MotionController(
        arm, motion=True, on_step=lambda *_: observed.append(True), sleep=lambda _: None,
    )
    result = controller.move_to({"shoulder_pan": 0.5})
    assert result["outcome"] == "target_reached"
    assert result["actual_positions"]["shoulder_pan"] == 0
    assert observed == []
    assert not controller.stopped.is_set()


@pytest.mark.parametrize("simulation", [False, True])
def test_successful_export_round_trips_wrist_only_and_actual_timestamps(tmp_path, simulation):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    class Camera(FakeCamera):
        def latest(self):
            image, timestamp = super().latest()
            return (image[:240, :320] if simulation else image), timestamp

    recorder = EpisodeRecorder(tmp_path / "episode", Camera(), task="Synthetic test")
    state = dict.fromkeys(JOINTS, 0.0)
    for _ in range(4):
        recorder.sample(state, {**state, "shoulder_pan": 2})
        time.sleep(.1)
    report = recorder.finish("success", "Synthetic fixture, not a physical success")
    assert report["timing_acceptable"]
    if simulation:
        report["simulation"] = True
        (recorder.root / "episode.json").write_text(json.dumps(report))
        path = recorder.root / "frames.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for index, row in enumerate(rows):
            row["elapsed"] = row["camera_elapsed"] = index / report["fps"] - (1e-13 if index else 0)
            row["target_box"] = [.4, .5, .2, .1, 1]
            row["target_depth_m"] = .18
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    destination = tmp_path / "dataset"
    export_episodes([recorder.root], destination, "local/so101-test")
    dataset = LeRobotDataset("local/so101-test", root=destination, download_videos=False)
    assert len(dataset) >= 3
    if simulation:
        assert len(dataset) == 4
    assert dataset.meta.camera_keys == ["observation.images.wrist"]
    assert dataset.meta.features["observation.state"]["names"] == list(JOINTS)
    assert dataset[0]["observation.images.wrist"].shape == ((3, 240, 320) if simulation else (3, 480, 640))
    assert dataset.meta.info.robot_type == ("so101_sim" if simulation else "so101_follower")
    assert dataset[0]["action"][0].item() == 2
    assert dataset[0]["observation.state"][0].item() == 0
    assert dataset[0]["observation.state"].shape == (6,)
    assert dataset[1]["recording.elapsed"].item() > dataset[0]["recording.elapsed"].item()
    if simulation:
        assert dataset.meta.features["observation.target_box"]["names"] == [
            "center_x", "center_y", "width", "height", "visible",
        ]
        np.testing.assert_allclose(
            dataset[0]["observation.target_box"].numpy(),
            [.4, .5, .2, .1, 1],
        )
        assert dataset[0]["observation.target_depth_m"].item() == pytest.approx(.18)
    else:
        assert "observation.target_box" not in dataset.meta.features
        assert "observation.target_depth_m" not in dataset.meta.features
    assert (destination / "recording_sources.json").exists()


def test_export_rejects_mixed_uncalibrated_real_and_simulated_sources(tmp_path):
    sources = []
    for simulation in (False, True):
        source = tmp_path / str(simulation)
        source.mkdir()
        (source / "episode.json").write_text(json.dumps({
            "simulation": simulation, "outcome": "success",
            "timing_acceptable": True, "fps": 10, "frames": 1,
        }))
        (source / "frames.jsonl").write_text(json.dumps({"image": "not-read.jpg"}) + "\n")
        sources.append(source)
    with pytest.raises(ValueError, match="simulated and real"):
        export_episodes(sources, tmp_path / "dataset", "local/test")
