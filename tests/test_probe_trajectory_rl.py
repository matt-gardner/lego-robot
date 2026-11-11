import json
from types import SimpleNamespace

import pytest

from so101.probe_trajectory_rl import binary_assistance_step, probe, success_interval
import so101.probe_trajectory_rl as module


@pytest.mark.parametrize("plan,calibrate,message", [
    ([], False, "contain problems"),
    ([{"task": "view", "source": "a", "assistance": 0.}] * 2, False, "duplicates"),
    ([{"task": "view", "source": "a", "assistance": .5}], True, "without assistance"),
])
def test_invalid_measurement_population_fails_before_loading_policy(tmp_path, plan, calibrate, message):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    args = SimpleNamespace(
        plan=path, native_pool=None, native_references=None,
        calibrate_assistance=calibrate, native_only=True, initial_only=True,
    )
    with pytest.raises(ValueError, match=message):
        probe(args)


def test_probe_intervals_preserve_uncertainty_at_extremes():
    low, high = success_interval(0, 16)
    assert low == pytest.approx(0)
    assert .19 < high < .20
    low, high = success_interval(16, 16)
    assert .80 < low < .81
    assert high == pytest.approx(1)
    low, high = success_interval(8, 16)
    assert low < .5 < high
    wider = high - low
    low, high = success_interval(16, 32)
    assert high - low < wider


def test_binary_search_bisects_measured_bounds_and_stops_at_mixed_outcomes():
    state = {"lower": 0., "upper": .95, "rounds": 0}
    level, status = binary_assistance_step(state, .95, 16, 16, 6)
    assert level == pytest.approx(.475) and status is None
    level, status = binary_assistance_step(state, level, 0, 16, 6)
    assert level == pytest.approx(.7125) and status is None
    assert binary_assistance_step(state, level, 8, 16, 6) == (None, "measured-mixed-outcomes")
    state = {"lower": 0., "upper": .95, "rounds": 0}
    assert binary_assistance_step(state, .95, 0, 16, 6) == (
        None, "no-mixed-level-at-maximum-assistance",
    )


def test_short_horizon_is_used_and_explicitly_labeled(tmp_path, monkeypatch):
    source = str(tmp_path / "episode")
    plan, split, output = (tmp_path / name for name in ("plan.json", "split.json", "results.json"))
    plan.write_text(json.dumps([{"task": "grasp", "source": source, "assistance": .95}]))
    split.write_text(json.dumps({"training": {"grasp": [source]}, "validation": [],
                                 "expert_bounds": {"grasp": {}}}))
    episode = SimpleNamespace(source=source, task="grasp",
                              metadata={"native_entry": {}, "native_reference": {}})
    policy = SimpleNamespace(eval=lambda: None)
    monkeypatch.setattr(module, "load_corpus", lambda _: {"grasp": [episode]})
    monkeypatch.setattr(module, "EntityPolicy", lambda *_: SimpleNamespace(policy=policy))
    monkeypatch.setattr(module, "trainable_parameters", lambda *_: None)
    monkeypatch.setattr(module, "policy_digest", lambda _: "unchanged")
    monkeypatch.setattr(module, "create_simulation", lambda *_, **__: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(module, "load_reference_rows", lambda _: [None] * 80)
    horizons = []

    def sample(*args):
        assert output.exists()
        horizons.append(args[4])
        return None

    monkeypatch.setattr(module, "rollout", sample)
    monkeypatch.setattr(module, "serialized_trajectory", lambda _: {"success": True})
    args = SimpleNamespace(
        plan=plan, split=split, output=output, collection=tmp_path, checkpoint=tmp_path,
        training_output=tmp_path, model_root=tmp_path, native_pool=None, native_references=None,
        calibrate_assistance=False, native_only=True, initial_only=True,
        repeats=16, batch_rollouts=1, seed=100, horizon=50,
        binary_search=False, search_rounds=2, resume=False,
    )
    probe(args)
    result = json.loads(output.read_text())
    assert horizons == [50] * 16
    assert result["horizon"] == 50
    assert result["measurement_scope"] == "short-horizon-screening"
    assert result["policy_sha256_before"] == result["policy_sha256_after"] == "unchanged"
    result["complete"] = False
    output.write_text(json.dumps(result))
    args.binary_search, args.resume = True, True
    probe(args)
    resumed = json.loads(output.read_text())
    assert len(horizons) == 32
    assert [(group["assistance"], group["total"]) for group in resumed["groups"]] == [(.95, 16), (.475, 16)]
    assert resumed["search_strategy"] == "binary-search"
    resumed["complete"] = False
    output.write_text(json.dumps(resumed))
    args.horizon = 51
    with pytest.raises(ValueError, match="horizon"):
        probe(args)
