import json
from types import SimpleNamespace

import pytest

from so101.probe_trajectory_rl import probe, success_interval
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


def test_short_horizon_is_used_and_explicitly_labeled(tmp_path, monkeypatch):
    source = str(tmp_path / "episode")
    plan, split, output = (tmp_path / name for name in ("plan.json", "split.json", "results.json"))
    plan.write_text(json.dumps([{"task": "grasp", "source": source, "assistance": .95}]))
    split.write_text(json.dumps({"training": {"grasp": [source]}, "validation": [],
                                 "expert_bounds": {"grasp": {}}}))
    episode = SimpleNamespace(source=source, task="grasp", metadata={})
    policy = SimpleNamespace(eval=lambda: None)
    monkeypatch.setattr(module, "load_corpus", lambda _: {"grasp": [episode]})
    monkeypatch.setattr(module, "EntityPolicy", lambda *_: SimpleNamespace(policy=policy))
    monkeypatch.setattr(module, "trainable_parameters", lambda *_: None)
    monkeypatch.setattr(module, "policy_digest", lambda _: "unchanged")
    monkeypatch.setattr(module, "create_simulation", lambda *_, **__: SimpleNamespace(close=lambda: None))
    horizons = []

    def sample(*args):
        assert output.exists()
        horizons.append(args[4])
        return None

    monkeypatch.setattr(module, "rollout", sample)
    monkeypatch.setattr(module, "serialized_trajectory", lambda _: {"success": True})
    probe(SimpleNamespace(
        plan=plan, split=split, output=output, collection=tmp_path, checkpoint=tmp_path,
        training_output=tmp_path, model_root=tmp_path, native_pool=None, native_references=None,
        calibrate_assistance=False, native_only=False, initial_only=True,
        repeats=16, batch_rollouts=1, seed=100, horizon=50,
    ))
    result = json.loads(output.read_text())
    assert horizons == [50] * 16
    assert result["horizon"] == 50
    assert result["measurement_scope"] == "short-horizon-screening"
    assert result["policy_sha256_before"] == result["policy_sha256_after"] == "unchanged"
