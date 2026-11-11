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


@pytest.fixture
def probe_setup(tmp_path, monkeypatch):
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
        repeats=4, batch_rollouts=1, seed=100, horizon=50,
        binary_search=False, search_rounds=2, resume=False,
    )
    return args, horizons


def test_short_horizon_is_used_and_explicitly_labeled(probe_setup):
    args, horizons = probe_setup
    output = args.output
    probe(args)
    result = json.loads(output.read_text())
    assert horizons == [50] * 4
    assert result["horizon"] == 50
    assert result["measurement_scope"] == "short-horizon-screening"
    assert result["policy_sha256_before"] == result["policy_sha256_after"] == "unchanged"
    result["complete"] = False
    output.write_text(json.dumps(result))
    args.binary_search, args.resume = True, True
    probe(args)
    resumed = json.loads(output.read_text())
    assert len(horizons) == 8
    assert [(group["assistance"], group["total"]) for group in resumed["groups"]] == [(.95, 4), (.475, 4)]
    assert resumed["search_strategy"] == "binary-search"
    resumed["complete"] = False
    output.write_text(json.dumps(resumed))
    args.horizon = 51
    with pytest.raises(ValueError, match="horizon"):
        probe(args)


@pytest.mark.parametrize("fail_at_ceiling", [False, True])
def test_binary_screen_switches_after_four_and_only_confirms_mixed_levels(
        probe_setup, monkeypatch, fail_at_ceiling):
    args, _ = probe_setup
    args.binary_search, args.batch_rollouts, args.search_rounds = True, 4, 6
    calls = []

    def sample(sims, controller, episode, seeds, horizon, action_steps, bounds,
               assistance, *rest, **kwargs):
        calls.append((assistance, list(seeds)))
        assert len(seeds) == 4 and horizon == 50
        if assistance == .95:
            return [not fail_at_ceiling] * 4
        if assistance == .475:
            return [False] * 4
        return [True, False, False, False]

    monkeypatch.setattr(module, "rollout_group", sample)
    monkeypatch.setattr(module, "serialized_trajectory", lambda success: {"success": success})
    probe(args)
    result = json.loads(args.output.read_text())
    if fail_at_ceiling:
        assert calls == [(.95, [100, 101, 102, 103])]
        assert result["groups"][0]["total"] == 4
    else:
        assert [level for level, _ in calls] == pytest.approx([.95, .475, .7125, .7125])
        assert [seeds for _, seeds in calls] == [[100, 101, 102, 103]] * 3 + [[104, 105, 106, 107]]
        assert [group["total"] for group in result["groups"]] == [4, 4, 8]
        assert [group["planned_attempts"] for group in result["groups"]] == [4, 4, 8]
        decision = next(iter(result["assistance_decisions"].values()))
        assert (decision["successes"], decision["total"], decision["status"]) == (
            2, 8, "measured-mixed-outcomes",
        )


@pytest.mark.parametrize("count", [1, 3, 5])
def test_bucket_screen_requires_four_distinct_problems(count):
    rows = [{"bucket_id": "20:grasp", "source": str(index)} for index in range(count)]
    with pytest.raises(ValueError, match="four distinct"):
        module.bucket_population(rows)
    with pytest.raises(ValueError, match="four distinct"):
        module.bucket_population([{"bucket_id": "20:grasp", "source": "same"}] * 4)


@pytest.mark.parametrize("all_fail", [True, False])
def test_bucket_screen_samples_each_problem_once_and_confirms_only_mixed(
        probe_setup, monkeypatch, all_fail):
    args, _ = probe_setup
    args.bucket_screening, args.binary_search = True, True
    args.search_rounds = 6
    rows = [{"task": "grasp", "source": f"problem-{index}", "assistance": .95,
             "bucket_id": "20:grasp"} for index in range(4)]
    args.plan.write_text(json.dumps(rows))
    episodes = [SimpleNamespace(source=row["source"], task="grasp",
                                metadata={"native_entry": {}, "native_reference": {}})
                for row in rows]
    monkeypatch.setattr(module, "load_corpus", lambda _: {"grasp": episodes})
    args.split.write_text(json.dumps({
        "training": {"grasp": [row["source"] for row in rows]},
        "validation": [], "expert_bounds": {"grasp": {}},
    }))
    calls = []

    def sample(sim, controller, episode, seed, horizon, action_steps, bounds,
               store_decisions, assistance, noise):
        calls.append((assistance, episode.source, seed))
        return (not all_fail) if assistance == .95 else episode.source == "problem-0"

    monkeypatch.setattr(module, "rollout", sample)
    monkeypatch.setattr(module, "serialized_trajectory", lambda success: {"success": success})
    probe(args)
    result = json.loads(args.output.read_text())
    assert calls[:4] == [(.95, f"problem-{index}", 100 + index) for index in range(4)]
    assert len(calls) == (4 if all_fail else 12)
    decision = result["assistance_decisions"]["20:grasp"]
    if all_fail:
        assert decision["total"] == 4 and decision["successes"] == 0
    else:
        assert calls[4:8] == [(.475, f"problem-{index}", 100 + index) for index in range(4)]
        assert calls[8:] == [(.475, f"problem-{index}", 104 + index) for index in range(4)]
        assert decision["total"] == 8 and decision["successes"] == 2
        assert [group["planned_attempts"] for group in result["groups"]] == [1] * 4 + [2] * 4
    result["complete"] = False
    args.output.write_text(json.dumps(result))
    args.resume = True
    probe(args)
    assert len(calls) == (4 if all_fail else 12)
