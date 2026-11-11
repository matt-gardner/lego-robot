import json
import os
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

import so101.rl_dashboard as rl_dashboard
from so101.rl_dashboard import RunStore, active_run_outputs, make_server, normalize_log


def test_nested_recognition_log_is_not_action_sft_or_rl(tmp_path):
    directory = tmp_path / "piece-recognition-v1/frozen/run"
    directory.mkdir(parents=True)
    (directory / "sft_training.json").write_text(json.dumps({
        "kind": "piece_recognition_sft",
        "run_config": {"objective": "recognition-only"},
        "steps": [{"step": 1, "loss": 6.3, "samples_seen": 4, "learning_rate": 1e-5}],
        "validation": [],
    }))
    store = RunStore(tmp_path)
    identity, path = next(iter(store.discover().items()))
    result = store.load(identity, path)
    assert result["kind"] == "recognition"
    assert result["latest_train"]["loss"] == 6.3
    assert result["latest_train"]["samples_seen"] == 4
    assert result["latest_eval"] is None


def test_recognition_evaluations_preserve_split_metrics_and_sidecar_updates(tmp_path):
    log = tmp_path / "sft_training.json"
    metrics = {
        "step": 100, "n": 20, "families": 3, "top1": .5, "top3": .8, "top5": 1,
        "macro_top1": .3, "scope": "Box-view observations, not independent pieces.",
        "seen_unseen": {"seen_id": {"n": 12, "top1": .75}, "unseen_id": {"n": 8, "top1": .125}},
        "category_accuracy": {"family": {"n": 20, "accuracy": .6}, "kind": {"n": 20, "accuracy": .7}},
        "training_majority_baseline": {"top1": .4},
    }
    log.write_text(json.dumps({
        "kind": "piece_recognition_sft",
        "steps": [{"step": 99, "loss": 3, "samples_seen": 396},
                  {"step": 100, "loss": 2, "samples_seen": 400,
                   "train_top1": .25, "train_top3": .5, "train_top5": .75, "train_accuracy_n": 16}],
        "validation": [metrics], "training_evaluation": [{**metrics, "n": 40}],
    }))
    store = RunStore(tmp_path)
    identity, path = next(iter(store.discover().items()))
    first = store.load(identity, path)
    assert first["latest_eval"]["top1"] == .5
    assert first["latest_eval"]["samples_seen"] == 400
    assert first["latest_eval"]["category_family_accuracy"] == .6
    assert first["latest_eval"]["unseen_id_n"] == 8
    assert "success_rate" not in first["latest_eval"]
    assert first["latest_train"]["train_top1"] == .25
    assert first["series"][0]["points"][0]["train_top1"] is None
    assert first["training_evaluation"][0]["n"] == 40
    assert "final_test" not in first
    sidecar = log.with_name("test_results.json")
    sidecar.write_text(json.dumps({
        **metrics, "top1": .2, "selected_step": 100, "selection": "development macro_top1",
        "images_sha256": "a" * 64, "checkpoint_files_sha256": {"config": "b" * 64},
    }))
    second = store.load(identity, path)
    assert second["revision"] != first["revision"]
    assert second["latest_eval"]["top1"] == .5
    assert second["final_test"]["top1"] == .2
    assert second["final_test"]["selected_step"] == 100
    sidecar.write_text("{")
    assert store.load(identity, path)["error"]


def training_log(step=1):
    return {
        "algorithm": "trajectory group-relative flow matching",
        "steps": [{
            "step": step,
            "learning_rate": 3e-6,
            "gradient_norm": 0.4,
            "groups": [{
                "task": "push",
                "discarded": False,
                "start_proximity": 0.9,
                "trajectories": [
                    {"success": True, "wrong_object": False,
                     "completion_efficiency": 0.8, "rms_jerk": 2, "reward": 1.4},
                    {"success": False, "wrong_object": True,
                     "completion_efficiency": 0, "rms_jerk": 4, "reward": 0},
                ],
            }, {
                "task": "place",
                "discarded": True,
                "start_proximity": 0.5,
                "trajectories": [
                    {"success": False, "wrong_object": False,
                     "completion_efficiency": 0, "rms_jerk": 6, "reward": 0},
                ],
            }],
        }],
        "validation": [{
            "step": 0, "success_rate": 0.25, "episodes": 40,
            "completion_efficiency": 0.6, "rms_jerk": 5,
            "wrong_object_rate": 0,
            "by_task": {"push": {"success_rate": 0.1}},
        }],
    }


def sft_log():
    return {
        "kind": "sft",
        "run_config": {"steps": 5000, "schedule": "triangular"},
        "dataset": {"arbitrary_new_metadata": True},
        "steps": [{
            "step": 1, "loss": 0.4, "action_loss": 0.3, "box_loss": 0.1,
            "contrast_loss": float("nan"), "learning_rate": 1e-5,
            "gradient_norm": 0.8, "step_seconds": 2, "completed_at": 1000,
            "eta_seconds": 9998, "model": {"parameters": [1, 2]},
        }],
        "validation": [{
            "step": 1, "sampler": "sde", "denoising_noise_scale": 0.1, "episodes": 40,
            "success_rate": 0.5, "completion_efficiency": 0.7, "rms_jerk": 3,
            "wrong_object_rate": 0.1,
            "by_task": {
                task: {"success_rate": 0 if task == "place" else 0.5,
                       "completion_efficiency": 0.7, "rms_jerk": 3}
                for task in ("grasp", "view", "push", "place")
            },
            "rollouts": [{"source": "example", "seed": 1, "task": "push",
                          "success": True, "invalid_action_error": None,
                          "frames": ["large-frame-payload"]}],
        }],
    }


def test_sft_exposure_preserves_logged_train_and_evaluation_counts():
    data = sft_log()
    data["steps"][0].update(samples_seen=128)
    data["validation"][0].update(samples_seen=0, supervised_loss=2.5, evaluation_seconds=198,
                                 by_task={"grasp": {"success_rate": 0.5}})
    normalized = normalize_log(data)
    assert normalized["latest_train"]["samples_seen"] == 128
    assert normalized["latest_eval"]["samples_seen"] == 0
    assert normalized["latest_eval"]["supervised_loss"] == 2.5
    assert normalized["latest_eval"]["evaluation_seconds"] == 198
    task_eval = next(row for row in normalized["series"] if row["phase"] == "eval" and row["task"] == "grasp")
    assert task_eval["points"][0]["samples_seen"] == 0
    del data["steps"][0]["samples_seen"]
    assert normalize_log(data)["latest_train"]["samples_seen"] is None


def test_sft_has_loss_only_training_and_same_rollout_evaluation_metrics():
    result = normalize_log(sft_log())
    assert result["kind"] == "sft"
    assert result["axis"] == "Optimizer step"
    train = result["latest_train"]
    assert train["loss"] == 0.4
    assert train["action_loss"] == 0.3
    assert train["box_loss"] == 0.1
    assert "contrast_loss" not in train
    assert train["learning_rate"] == 1e-5
    assert train["eta_seconds"] == 9998
    for key in ("success_rate", "completion_efficiency", "rms_jerk", "wrong_object_rate",
                "samples", "jerk_margin_gap", "model"):
        assert key not in train
    series = {(row["phase"], row["task"]): row["points"] for row in result["series"]}
    assert len(series) == 6
    assert result["latest_eval"]["success_rate"] == 0.5
    assert result["latest_eval"]["completion_efficiency"] == 0.7
    assert result["latest_eval"]["rms_jerk"] == 3
    assert result["latest_eval"]["wrong_object_rate"] == 0.1
    assert series["eval", "place"][0]["completion_efficiency"] is None
    assert series["eval", "push"][0]["wrong_object_rate"] is None
    assert result["best_eval"] == result["latest_eval"]
    assert result["config"]["run_config"] == sft_log()["run_config"]
    payload = json.dumps(result, allow_nan=False)
    assert "large-frame-payload" not in payload
    assert "rollouts" not in payload


def test_sft_discovery_before_validation_skips_rl_reference_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(rl_dashboard, "active_run_outputs", lambda: set())
    directory = tmp_path / "sft"
    directory.mkdir()
    data = sft_log()
    del data["validation"]
    path = directory / "sft_training.json"
    path.write_text(json.dumps(data))
    # SFT split metadata is not a trajectory-RL jerk-reference manifest.
    (directory / "data_split.json").write_text('{"arbitrary_split": true}')
    (directory / "model.json").write_text('{"parameters": [1, 2]}')
    (directory / "frames.json").write_text('{"frames": [1, 2]}')
    store = RunStore(tmp_path)
    catalog = store.catalog()
    assert len(catalog) == 1
    run = catalog[0]
    assert run["kind"] == "sft"
    assert run["archived"] is False
    assert run["latest_train"]["loss"] == 0.4
    assert run["latest_eval"] is None
    assert run["last_step_at"] == 1000
    assert run["last_step_time_estimated"] is False
    assert "series" not in run
    data = sft_log()
    path.write_text(json.dumps(data))
    assert store.get(run["id"])["latest_eval"]["success_rate"] == 0.5
    path.write_text("{")
    assert store.get(run["id"])["kind"] == "sft"
    assert store.get(run["id"])["error"]
    other = tmp_path / "rl"
    other.mkdir()
    (other / "trajectory_rl.json").write_text(json.dumps(training_log()))
    assert {row["kind"] for row in store.catalog()} == {"rl", "sft"}


def test_trajectory_metrics_include_discarded_groups_and_separate_validation():
    result = normalize_log(training_log())
    assert result["kind"] == "rl"
    series = {(row["phase"], row["task"]): row["points"] for row in result["series"]}
    train = series["train", "all"][0]
    assert train["success_rate"] == pytest.approx(1 / 3)
    assert train["completion_efficiency"] == 0.8
    assert train["rms_jerk"] == 4
    assert train["discard_rate"] == 0.5
    assert train["learning_rate"] == 3e-6
    assert series["train", "place"][0]["completion_efficiency"] is None
    assert series["train", "push"][0]["assistance"] == 0.9
    assert series["eval", "all"][0]["success_rate"] == 0.25
    assert series["eval", "push"][0]["wrong_object_rate"] is None


def test_legacy_episode_metrics_use_successful_completion_and_step_axis():
    result = normalize_log({
        "steps": [{"training_step": 2, "gradient_norm": 0.7}],
        "episode_metrics": [
            {"episode": 0, "training_step": 2, "task": "push", "success": True,
             "completion_efficiency_percent": 70, "rms_arm_jerk_degrees_per_frame3": 3},
            {"episode": 1, "training_step": 2, "task": "push", "success": False,
             "completion_efficiency_percent": None, "rms_arm_jerk_degrees_per_frame3": 5},
        ],
    })
    assert result["kind"] == "rl"
    assert result["axis"] == "Optimizer step"
    assert result["latest_train"]["step"] == 2
    assert result["latest_train"]["completion_efficiency"] == 0.7
    assert result["latest_train"]["gradient_norm"] == 0.7
    assert result["latest_train"]["rms_jerk"] == 4


def test_jerk_gap_uses_each_group_margin_and_weights_eligible_rollouts():
    data = training_log()
    groups = data["steps"][0]["groups"]
    groups[0]["jerk_margin"] = 3
    groups[1]["jerk_margin"] = 5
    groups[0]["jerk_reference_scope"] = groups[1]["jerk_reference_scope"] = "problem"
    groups[1]["sampling_bucket"] = "solved"
    for group in groups:
        for item in group["trajectories"]:
            item["steps"] = 8
    groups.append({"task": "push", "jerk_margin": None,
                   "trajectories": [{"steps": 8, "rms_jerk": 100}]})
    result = normalize_log(data)
    series = {(row["phase"], row["task"]): row["points"] for row in result["series"]}
    assert series["train", "all"][0]["jerk_margin_gap"] == pytest.approx(1 / 3)
    assert series["train", "all"][0]["jerk_margin_samples"] == 3
    assert series["train", "push"][0]["jerk_margin_gap"] == 0
    assert series["train", "place"][0]["jerk_margin_gap"] == 1
    assert data["steps"][0]["groups"][0]["jerk_margin"] == 3


def test_jerk_gap_preserves_negative_and_zero_margins_without_inventing_missing_data():
    data = training_log()
    assert normalize_log(data)["latest_train"]["jerk_margin_gap"] is None
    groups = data["steps"][0]["groups"]
    groups[0]["jerk_margin"] = 0
    groups[0]["jerk_reference_scope"] = "problem"
    groups[0]["trajectories"] = [{"steps": 4, "rms_jerk": 2},
                                {"steps": 3, "rms_jerk": 100}]
    groups[1]["jerk_margin"] = 6
    groups[1]["jerk_reference_scope"] = "problem"
    groups[1]["trajectories"] = [{"steps": 4, "rms_jerk": 4},
                                {"steps": 4, "rms_jerk": float("nan")}]
    result = normalize_log(data)
    assert result["latest_train"]["jerk_margin_gap"] == 0
    assert result["latest_train"]["jerk_margin_samples"] == 2
    place = next(row for row in result["series"] if row["phase"] == "train" and row["task"] == "place")
    assert place["points"][0]["jerk_margin_gap"] == -2


def test_historical_jerk_plot_recomputes_each_problem_reference_not_old_pooled_margin():
    data = training_log()
    groups = data["steps"][0]["groups"]
    groups[1]["task"] = "push"
    for source, group in zip(("problem-a", "problem-b"), groups, strict=True):
        group["source"] = source
        group["jerk_margin"] = 100  # Old command-level P90 must not be used.
        for item in group["trajectories"]:
            item["steps"] = 8
    references = SimpleNamespace(summary=lambda source, assistance: {
        "margin": {"problem-a": 1, "problem-b": 5}[source],
    })
    assert normalize_log(data)["latest_train"]["jerk_margin_gap"] is None
    result = normalize_log(data, references)
    assert result["latest_train"]["jerk_margin_gap"] == pytest.approx(5 / 3)
    assert result["latest_train"]["jerk_margin_samples"] == 3


def test_optimization_phase_boundary_preserves_raw_gradient_history():
    data = training_log(96)
    data["steps"][0]["gradient_norm"] = 4600
    data["steps"].append({
        **data["steps"][0], "step": 97, "gradient_norm": 0.5,
    })
    data["policy_optimization_changes"] = [
        {"start_step": 97, "version": "normalized-raoblackwell-fp32-kl-v2"},
    ]
    result = normalize_log(data)
    assert result["config"]["policy_optimization_changes"] == data["policy_optimization_changes"]
    training = next(
        row["points"] for row in result["series"]
        if row["phase"] == "train" and row["task"] == "all"
    )
    assert [row["gradient_norm"] for row in training] == [4600, 0.5]
    assert [row["rms_jerk"] for row in training] == [4, 4]
    assert data["steps"][0]["gradient_norm"] == 4600


def test_old_decision_logs_are_not_mislabeled_as_optimizer_steps():
    result = normalize_log({
        "rollouts": [{"episode": 4, "task": "push", "success": False}],
    })
    assert "decision-level" in result["axis"]
    assert result["latest_train"]["step"] == 4
    assert result["latest_train"]["rms_jerk"] is None


def test_step_timing_uses_completion_timestamp_not_log_modification(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    path = directory / "trajectory_rl.json"
    data = training_log()
    data["steps"][0].update(step_seconds=324.1, completed_at=1000)
    path.write_text(json.dumps(data))
    os.utime(path, (2000, 2000))
    store = RunStore(tmp_path)
    run = store.catalog()[0]
    assert run["latest_train"]["step_seconds"] == 324.1
    assert run["last_step_at"] == 1000
    assert run["last_step_time_estimated"] is False


def test_legacy_step_clock_only_resets_when_step_advances(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    path = directory / "trajectory_rl.json"
    path.write_text(json.dumps(training_log()))
    os.utime(path, (1000, 1000))
    store = RunStore(tmp_path)
    run = store.catalog()[0]
    assert run["last_step_at"] == 1000
    assert run["last_step_time_estimated"] is True
    os.utime(path, (2000, 2000))
    assert store.get(run["id"])["last_step_at"] == 1000
    path.write_text(json.dumps(training_log(2)))
    os.utime(path, (3000, 3000))
    assert store.get(run["id"])["last_step_at"] == 3000


def test_no_completed_steps_has_no_step_clock(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    data = training_log()
    data["steps"] = []
    (directory / "trajectory_rl.json").write_text(json.dumps(data))
    run = RunStore(tmp_path).catalog()[0]
    assert run["last_step_at"] is None
    assert run["last_step_time_estimated"] is False


def test_process_activity_matches_trainer_outputs_not_command_mentions(tmp_path, monkeypatch):
    trajectory = tmp_path / "trajectory"
    chunk = tmp_path / "chunk run"
    packaged = tmp_path / "packaged"
    commands = "\n".join([
        f"/usr/bin/python3 -u train_trajectory_rl.py --output {trajectory}",
        f"/usr/bin/python3 train_entity_rl.py --output='{chunk}'",
        f"/usr/bin/python3 -c train_trajectory_rl.py --output {tmp_path / 'not-a-trainer'}",
        f"bash train_trajectory_rl.py --output {tmp_path / 'shell'}",
        f"/usr/bin/python3 -u -m so101.train_trajectory_rl --output {packaged}",
        f"/usr/bin/python3 -c so101.train_trajectory_rl --output {tmp_path / 'module-mention'}",
        f"/usr/bin/python3 -m unrelated so101.train_trajectory_rl --output {tmp_path / 'unrelated'}",
    ])
    monkeypatch.setattr(
        rl_dashboard.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=commands),
    )
    assert active_run_outputs() == {trajectory, chunk, packaged}


def test_sft_process_activity_matches_exact_python_execution_target(tmp_path, monkeypatch):
    module = tmp_path / "module run"
    script = tmp_path / "script"
    commands = "\n".join([
        f"/usr/bin/python3 -u -m so101.train_entity_sim --output '{module}' --steps 5000",
        f"/usr/bin/python3 -W ignore /code/train_entity_sim.py --output={script}",
        f"bash -c 'python -m so101.train_entity_sim --output {tmp_path / 'shell'}'",
        f"/usr/bin/python3 wrapper.py so101.train_entity_sim --output {tmp_path / 'wrapper'}",
        f"/usr/bin/python3 -c 'so101.train_entity_sim' --output {tmp_path / 'mention'}",
        f"/usr/bin/python3 -m wrapper so101.train_entity_sim --output {tmp_path / 'module-wrapper'}",
        f"/usr/bin/python3 -m so101.train_entity_sim_extra --output {tmp_path / 'suffix'}",
        f"/usr/bin/python3 /code/other_train_entity_sim.py --output {tmp_path / 'script-suffix'}",
    ])
    monkeypatch.setattr(
        rl_dashboard.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=commands),
    )
    assert active_run_outputs() == {module, script}


def test_activity_changes_without_new_log_and_never_activates_archives(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "trajectory_rl.json").write_text(json.dumps(training_log()))
    (directory / "attempt_step001.json").write_text(json.dumps(training_log()))
    active = {directory}
    monkeypatch.setattr(rl_dashboard, "active_run_outputs", lambda: active)
    store = RunStore(tmp_path)
    catalog = store.catalog()
    live = next(row for row in catalog if not row["archived"])
    archived = next(row for row in catalog if row["archived"])
    assert live["active"] is True
    assert archived["active"] is False
    first = store.get(live["id"])
    revision = first["revision"]
    active.clear()
    stopped = store.get(live["id"])
    assert stopped["revision"] == revision
    assert stopped["active"] is False
    assert all(not row["active"] for row in store.catalog())
    active.add(directory)
    assert store.get(live["id"])["active"] is True


def test_activity_check_failure_is_explicit_and_not_reported_as_active(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "trajectory_rl.json").write_text(json.dumps(training_log()))

    def failed_check():
        raise OSError("process lookup unavailable")

    monkeypatch.setattr(rl_dashboard, "active_run_outputs", failed_check)
    run = RunStore(tmp_path).catalog()[0]
    assert run["active"] is None
    assert "process lookup unavailable" in run["activity_error"]
    assert run["latest_train"]["step"] == 1


def test_remote_liveness_expires_even_without_new_log(tmp_path, monkeypatch):
    directory = tmp_path / "cloud/run"
    directory.mkdir(parents=True)
    (directory / "trajectory_rl.json").write_text(json.dumps(training_log()))
    status = directory / "remote_status.json"
    status.write_text(json.dumps({"state": "running", "observed_at": 1000, "heartbeat": 1000}))
    monkeypatch.setattr(rl_dashboard, "active_run_outputs", lambda: set())
    monkeypatch.setattr(rl_dashboard.time, "time", lambda: 1020)
    store = RunStore(tmp_path)
    run = store.catalog()[0]
    assert run["active"] is True
    assert run["remote"]["contact_age_seconds"] == 20
    monkeypatch.setattr(rl_dashboard.time, "time", lambda: 1100)
    assert store.get(run["id"])["active"] is False
    assert store.get(run["id"])["remote"]["state"] == "stale"
    for state in ("paused", "error", "interrupted", "waiting_for_capacity"):
        status.write_text(json.dumps({"state": state, "observed_at": 1100, "heartbeat": 1100}))
        assert store.get(run["id"])["active"] is False
    status.write_text(json.dumps({"state": "waiting_for_capacity", "observed_at": 1100}))
    assert store.get(run["id"])["remote"]["state"] == "waiting_for_capacity"


def test_corrupt_remote_status_never_falls_back_to_local_activity(tmp_path, monkeypatch):
    directory = tmp_path / "cloud/run"
    directory.mkdir(parents=True)
    (directory / "trajectory_rl.json").write_text(json.dumps(training_log()))
    (directory / "remote_status.json").write_text("{")
    monkeypatch.setattr(rl_dashboard, "active_run_outputs", lambda: {directory})
    run = RunStore(tmp_path).catalog()[0]
    assert run["active"] is False
    assert "Cannot read remote status" in run["activity_error"]


def test_nonfinite_values_are_missing_not_invalid_json():
    data = training_log()
    data["validation"][0]["rms_jerk"] = float("nan")
    result = normalize_log(data)
    assert result["latest_eval"]["rms_jerk"] is None
    json.dumps(result, allow_nan=False)


def test_validation_without_success_has_no_conditional_completion():
    data = training_log()
    data["validation"][0].update(success_rate=0, completion_efficiency=0)
    result = normalize_log(data)
    assert result["latest_eval"]["success_rate"] == 0
    assert result["latest_eval"]["completion_efficiency"] is None


def test_discovery_caching_partial_writes_and_recovery(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    path = directory / "trajectory_rl.json"
    path.write_text(json.dumps(training_log()))
    (directory / "attempt_step001.json").write_text(json.dumps(training_log()))
    store = RunStore(tmp_path)
    catalog = store.catalog()
    assert len(catalog) == 2
    assert sum(row["archived"] for row in catalog) == 1
    summary = next(row for row in catalog if not row["archived"])
    assert "series" not in summary
    run_id = summary["id"]
    first = store.get(run_id)
    assert store.get(run_id) is first
    path.write_text('{"steps": [')
    stale = store.get(run_id)
    assert stale["error"]
    assert stale["latest_train"]["step"] == 1
    path.write_text(json.dumps(training_log(2)))
    fresh = store.get(run_id)
    assert fresh["error"] is None
    assert fresh["latest_train"]["step"] == 2
    assert fresh["revision"] != first["revision"]


def test_discovery_rejects_symlink_outside_run_root(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(training_log()))
    root = tmp_path / "outputs"
    root.mkdir()
    (root / "trajectory_rl.json").symlink_to(outside)
    assert RunStore(root).catalog() == []


def test_media_run_lookup_reuses_discovery_and_rechecks_path_boundary(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    root.mkdir()
    path = root / "trajectory_rl.json"
    path.write_text(json.dumps(training_log()))
    store = RunStore(root)
    run_id = next(iter(store.discover()))
    monkeypatch.setattr(store, "discover", lambda: pytest.fail("Media lookup rescanned the output tree"))
    assert store.path_for(run_id) == path
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    path.unlink()
    path.symlink_to(outside)
    assert store.path_for(run_id) is None


@pytest.mark.parametrize("base_path", ["", "/so101", "/lego-robot"])
def test_feedback_sign_in_stays_at_origin_root_on_both_browser_pages(tmp_path, base_path):
    server = make_server(tmp_path, "127.0.0.1", 0, base_path)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        for page in ("lego", "examples"):
            with urlopen(f"http://127.0.0.1:{server.server_port}{base_path}/{page}") as response:
                body = response.read().decode()
            assert 'href="/family"' in body
            assert f'href="{base_path}/lego"' in body
            assert f'src="{base_path}/lego.js"' in body
            if base_path:
                assert f'href="{base_path}/family"' not in body
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("base_path", ["", "/so101", "/lego-robot"])
def test_http_serves_assets_and_observes_new_runs_without_restart(tmp_path, base_path):
    server = make_server(tmp_path, "127.0.0.1", 0, base_path)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}{base_path}"
    try:
        with urlopen(f"{base}/api/runs") as response:
            assert json.load(response)["runs"] == []
        with urlopen(base) as response:
            assert b"RL Observatory" in response.read()
            assert response.headers["Cache-Control"] == "no-store"
        directory = tmp_path / "live"
        directory.mkdir()
        path = directory / "trajectory_rl.json"
        path.write_text(json.dumps(training_log()))
        with urlopen(f"{base}/api/runs") as response:
            run = json.load(response)["runs"][0]
        with urlopen(f"{base}/api/runs/{run['id']}") as response:
            assert json.load(response)["latest_train"]["step"] == 1
        path.write_text(json.dumps(training_log(3)))
        with urlopen(f"{base}/api/runs/{run['id']}") as response:
            assert json.load(response)["latest_train"]["step"] == 3
        for route in ("/api/runs/unknown", "/../pyproject.toml"):
            with pytest.raises(HTTPError) as caught:
                urlopen(base + route)
            assert caught.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_prefixed_pages_assets_data_boundaries_and_migration_notice(tmp_path):
    server = make_server(tmp_path, "127.0.0.1", 0, "/so101/",
                         "Awaiting verified migration; no results loaded <test>")
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        for route in ("/so101/", "/so101/data", "/so101/lego", "/so101/examples"):
            with urlopen(origin + route) as response:
                body = response.read().decode()
                assert 'data-base-path="/so101"' in body
                assert 'href="/so101/dashboard.css"' in body
                assert 'src="/so101/' in body
                assert "Awaiting verified migration; no results loaded &lt;test&gt;" in body
                assert 'href="/api' not in body
                assert '<header class="dashboard-header">' in body
                assert 'aria-label="Dashboard navigation"' in body
                assert "<!-- dashboard-header -->" not in body
                assert body.count('aria-current="page"') == 1
                assert f'href="{route}" aria-current="page"' in body
                for target in ("/", "/data", "/lego", "/examples"):
                    assert f'href="/so101{target}"' in body
                if route == "/so101/":
                    assert 'id="observatory-title"' in body and 'id="connection"' in body
        with urlopen(origin + "/so101") as response:
            assert response.url == origin + "/so101/"
        for asset in ("dashboard.js", "dashboard.css", "data.js", "data.css"):
            with urlopen(origin + "/so101/" + asset) as response:
                assert response.status == 200
        for route in ("/api/runs", "/", "/so101-other/api/runs", "/so101/api/data/unknown"):
            with pytest.raises(HTTPError) as caught:
                urlopen(origin + route)
            assert caught.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("base_path", ["relative", "//evil", "/../admin", "/so101?x", '/"><script>'])
def test_invalid_base_paths_are_rejected(tmp_path, base_path):
    with pytest.raises(ValueError, match="Base path"):
        make_server(tmp_path, "127.0.0.1", 0, base_path)
