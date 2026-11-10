from types import SimpleNamespace

import numpy as np
import pytest

from so101 import collect_harder_view as collector


def test_context_switch_restores_gl_context_then_offscreen_buffer(monkeypatch):
    events = []
    context = object()
    sim = SimpleNamespace(renderer=SimpleNamespace(
        _gl_context=SimpleNamespace(make_current=lambda: events.append("current")),
        _mjr_context=context,
    ))
    monkeypatch.setattr(collector.mujoco, "mjr_setBuffer",
                        lambda buffer, owner: events.append((buffer, owner)))
    collector.activate_render_context(sim)
    assert events == ["current", (collector.mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)]
    sim.renderer = None
    with pytest.raises(ValueError, match="live renderer"):
        collector.activate_render_context(sim)


def test_numeric_recorder_never_enables_legacy_image_writer(monkeypatch):
    sim = object.__new__(collector.ReferenceSimulation)
    sim.episode_root = None
    sim.rows = []
    sim.frame_start = 1.
    sim.data = SimpleNamespace(
        time=1., qpos=np.zeros(2), qvel=np.zeros(2), ctrl=np.zeros(6), act=np.zeros(0),
    )
    sim.state = lambda: np.zeros(6)
    sim.encode = lambda value: value
    sim.decode = lambda value: value

    def legacy_step(self, target, phase):
        assert self.episode_root is None
        self.data.qpos[:] = 1.

    monkeypatch.setattr(collector.BeamSimulation, "step", legacy_step)
    sim.step(np.zeros(6), "view")
    assert sim.rows[0]["qpos"] == [0., 0.]
    assert sim.rows[0]["action"] == [0.] * 6
    assert "image" not in sim.rows[0]
    sim.episode_root = "forbidden"
    with pytest.raises(RuntimeError, match="image writer"):
        sim.step(np.zeros(6), "view")


def test_reference_executes_serialized_float32_control_path(monkeypatch):
    sim = object.__new__(collector.ReferenceSimulation)
    sim.episode_root = None
    sim.rows = []
    sim.frame_start = 0.
    sim.data = SimpleNamespace(
        time=0., qpos=np.zeros(2), qvel=np.zeros(2), ctrl=np.zeros(6), act=np.zeros(0),
    )
    sim.state = lambda: np.zeros(6)
    sim.limits = np.array([[-3., 3.]] * 6)
    executed = []
    monkeypatch.setattr(collector.BeamSimulation, "step",
                        lambda self, target, phase: executed.append(target.copy()))
    original = np.array([.123456789] * 6)
    sim.step(original, "view")
    restored = sim.decode(np.asarray(sim.rows[0]["action"], dtype=float))
    np.testing.assert_array_equal(executed[0], restored)
    assert not np.array_equal(executed[0], original)


def test_storage_floor_blocks_before_reading_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(collector.shutil, "disk_usage",
                        lambda _: SimpleNamespace(free=collector.ADMISSION_BYTES - 1))
    with pytest.raises(RuntimeError, match="10GiB"):
        collector.collect(tmp_path / "missing", tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_legacy_goal_does_not_confuse_visible_fragment_with_centered_target():
    assert not collector.meets_view({"visible": True, "center_error": .31}, 40)
    assert not collector.meets_view({"visible": False, "center_error": .1}, 40)
    assert not collector.meets_view({"visible": True, "center_error": .1}, 29)
    assert collector.meets_view({"visible": True, "center_error": .1}, 30)


def test_replay_requires_matching_preaction_state_and_consecutive_success(monkeypatch):
    monkeypatch.setattr(collector, "activate_render_context", lambda _: None)
    monkeypatch.setattr(collector, "restore_episode", lambda *_: ("target", None, []))
    monkeypatch.setattr(collector, "view_metrics",
                        lambda *_: {"visible": True, "center_error": .1})

    class Simulator:
        fps = 2
        data = SimpleNamespace(qpos=np.zeros(2))

        def decode(self, action):
            return action

        def step(self, action, phase):
            pass

    rows = [{"qpos": [0., 0.], "action": [0.] * 6, "phase": "view"} for _ in range(31)]
    assert collector.replay_reference(Simulator(), {}, rows)["success"]
    assert not collector.replay_reference(Simulator(), {}, rows[:30])["success"]
    rows[0]["qpos"] = [1., 0.]
    with pytest.raises(ValueError, match="state mismatch"):
        collector.replay_reference(Simulator(), {}, rows)
