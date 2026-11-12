import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from so101.causal_trace import CausalTrace
from so101.simulation_corpus import CorpusEpisode
from so101 import train_trajectory_rl as trainer


def test_trace_limit_is_explicit_and_never_exceeded(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = CausalTrace(path, {"seed": 1}, 1024)
    trace.write("large", {"values": "x" * 2048})
    trace.write("later", {})
    trace.close(True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert path.stat().st_size <= 1024
    assert rows[-1] == {"event": "truncated", "reason": "byte_limit", "complete": False}
    with pytest.raises(FileExistsError):
        CausalTrace(path, {}, 1024)


def test_image_payload_is_hashed_not_stored_and_failure_is_explicit(tmp_path):
    trace = CausalTrace(tmp_path / "trace.jsonl", {}, 20000)
    trace.decision(([torch.ones(1, 3, 8, 8)], torch.ones(1, 6)),
                   SimpleNamespace(device="cpu"), 0, 0, "acquire")
    trace.close(False)
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    image = rows[1]["observation"][0][0]
    assert "sha256" in image and "values" not in image
    assert rows[-1] == {"event": "end", "complete": False}


@pytest.mark.parametrize("reject", [False, True])
def test_trace_distinguishes_prediction_from_applied_command_and_contacts(tmp_path, monkeypatch, reject):
    class Native:
        fps = 1
        obstacle_contact_events = []

        def __init__(self):
            self.value = 0

        def position(self, name):
            return np.array([.2, 0., 0.])

        def bounded_policy_action(self, action):
            if reject:
                raise ValueError("outside boundary")
            return action / 2, 1.

        def encode(self, action):
            return action

        def step(self, action, phase):
            self.value += 1

        def native_success(self, executed, stable):
            return True, 1

    sim = Native()
    monkeypatch.setattr(trainer, "NativePileSimulation", Native)
    monkeypatch.setattr(trainer, "native_snapshot", lambda sim, target: {
        "state": sim.value, "contacts": {"bilateral_grip": bool(sim.value)},
    })
    monkeypatch.setattr(trainer, "reset_from_episode", lambda *args: ("short", None, []))
    monkeypatch.setattr(trainer, "replay_gold_prefix", lambda *args: (0, 0.))
    monkeypatch.setattr(trainer, "control_phase", lambda *args: "acquire")
    monkeypatch.setattr(trainer, "wrong_object_lift", lambda *args: False)
    controller = SimpleNamespace(
        reset=lambda *a, **k: None, command_inputs=lambda *a: ([], torch.ones(1, 2)),
        policy=SimpleNamespace(eval=lambda: None, model=SimpleNamespace(
            sample_actions_entity=lambda *args: (torch.ones(1, 1, 6),))),
        action_std=np.ones(6), action_mean=np.zeros(6), observe=lambda *a: None,
        config=SimpleNamespace(history_steps=4), device="cpu",
    )
    trace = CausalTrace(tmp_path / "trace.jsonl", {"seed": 7}, 16384)
    rng = torch.get_rng_state().clone()
    result = trainer.rollout(sim, controller, CorpusEpisode(tmp_path, {"task": "grasp"}),
                             7, 1, 1, {"optimal_steps": 1, "jerk_p10": .1}, False,
                             causal_trace=trace)
    trace.close(True)
    assert torch.equal(rng, torch.get_rng_state())
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    decision = next(row for row in rows if row["event"] == "decision")
    assert decision["rng_state_hex"] and decision["observation"][1]["values"] == [[1., 1.]]
    chunk = next(row for row in rows if row["event"] == "sampled_chunk")
    assert chunk["model_commands"] == [[1.] * 6]
    row = next(row for row in rows if row["event"].endswith("_action"))
    assert row["model_command"] == [1.] * 6
    if reject:
        assert row["event"] == "rejected_action" and row["pre"]["state"] == 0
        assert result.steps == 0 and result.invalid_action_error
    else:
        assert row["applied_command"] == [.5] * 6
        assert row["pre"]["state"] == 0 and row["post"]["state"] == 1
        assert row["post"]["contacts"]["bilateral_grip"]
        assert result.success and result.steps == 1
