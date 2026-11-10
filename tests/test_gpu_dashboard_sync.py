import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "gpu_dashboard_sync", Path(__file__).parents[1] / "scripts/sync_gpu_dashboard.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("name,container", [
    ("entity_smolvla_sft_b64_lr1e-04", "b64-lr1e-04"),
    ("entity_smolvla_sft_b64_lr3e-05", "b64-lr3e5"),
    ("entity_smolvla_sft_b64_lr3e-05", "b64-lr3e-05"),
])
def test_dynamic_and_original_container_names_remain_active(tmp_path, name, container):
    module.sync({
        "observed_at": time.time(), "running": [f"lego-sweep-job-{container}"],
        "logs": {name: {"status": "training", "steps": [{"step": 1}]}},
    }, tmp_path, "job")
    status = json.loads((tmp_path / name / "remote_status.json").read_text())
    assert status["state"] == "running"
    assert status["phase"] == "training"
    assert status["error"] is None


@pytest.mark.parametrize("running,state", [(["long-container"], "running"),
                                          (["long-container-other"], "failed")])
def test_long_run_uses_new_instance_and_exact_container_identity(tmp_path, running, state):
    name = "entity_smolvla_sft_long_b64_lr3e-05"
    module.sync({
        "observed_at": time.time(), "running": running,
        "logs": {name: {"status": "training", "steps": [{"step": 385}]}},
    }, tmp_path, "new-job", instance_id="new-instance", container_name="long-container")
    status = json.loads((tmp_path / name / "remote_status.json").read_text())
    assert status["state"] == state
    assert status["instance_id"] == "new-instance"
    assert status["job_id"] == "new-job"


@pytest.mark.parametrize("running,step,state", [
    (["rl-container"], 1, "running"), ([], 1, "failed"), ([], 10, "completed"),
])
@pytest.mark.parametrize("name", ["curriculum_rl_7656", "full_model_rl"])
def test_trajectory_mirror_retains_unassisted_validation(tmp_path, running, step, state, name):
    log = {"steps": [{"step": step}], "validation": [{"step": 0}],
           "run_config": {"steps": 10}}
    if name == "full_model_rl":
        log["rl_estimator"] = "denoising-grpo"
    module.sync({
        "observed_at": time.time(), "running": running,
        "logs": {name: log},
    }, tmp_path, "job", container_name="rl-container")
    destination = tmp_path / name
    assert json.loads((destination / "trajectory_rl.json").read_text()) == log
    status = json.loads((destination / "remote_status.json").read_text())
    assert status["state"] == state
    assert not (destination / "sft_training.json").exists()


def test_trajectory_mirror_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError, match="Invalid GPU trial"):
        module.sync({
            "observed_at": time.time(), "running": [],
            "logs": {"../escaped": {"rl_estimator": "denoising-grpo"}},
        }, tmp_path, "job", container_name="rl-container")


@pytest.mark.parametrize("existing", ["absent", "absent-lowercase", "unreachable", "running", "stopped", "different"])
def test_prepared_launch_is_idempotent_and_preserves_existing_trainers(tmp_path, monkeypatch, existing):
    command = ["docker", "run", "--detach", "--name", "trainer", "image",
               "python", "-m", "so101.train_trajectory_rl"]
    path = tmp_path / "command.json"
    path.write_text(json.dumps(command))
    monkeypatch.setattr(module.sys, "argv", ["launch", str(path), "trainer"])
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv == command:
            return SimpleNamespace(returncode=0)
        assert argv == ["docker", "inspect", "trainer"]
        if existing.startswith("absent") and len(calls) == 1:
            message = "error: no such object: trainer" if existing == "absent-lowercase" else "No such object: trainer"
            return SimpleNamespace(returncode=1, stderr=message)
        if existing == "unreachable":
            return SimpleNamespace(returncode=1, stderr="Cannot connect to the Docker daemon")
        return SimpleNamespace(returncode=0, stdout=json.dumps([{
            "Config": {"Image": "other" if existing == "different" else "image",
                       "Cmd": command[6:]},
            "State": {"Running": existing != "stopped"},
        }]))

    monkeypatch.setattr(module.subprocess, "run", run)
    if existing in ("stopped", "different", "unreachable"):
        with pytest.raises((ValueError, RuntimeError)):
            exec(module.REMOTE_LAUNCH, {})
    else:
        exec(module.REMOTE_LAUNCH, {})
    assert sum(argv == command for argv in calls) == int(existing.startswith("absent"))


def test_remote_request_surfaces_captured_error(monkeypatch):
    def run(*args, **kwargs):
        raise module.subprocess.CalledProcessError(1, ["ssh"], stderr="GPU mount does not exist")

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="GPU mount does not exist"):
        module.remote_request(SimpleNamespace(known_hosts="/known", host="gpu"), "", [])


def test_launch_automatically_starts_mirroring_without_watch_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(module.sys, "argv", [
        "sync", "--output", str(tmp_path), "--job-id", "job",
        "--container-name", "trainer", "--launch-command", "/remote/command.json",
    ])
    calls = []
    monkeypatch.setattr(module, "remote_request", lambda *args: calls.append("launch") or {})
    monkeypatch.setattr(module, "watch", lambda args: calls.append(("watch", args.watch)))
    module.main()
    assert calls == ["launch", ("watch", True)]


@pytest.mark.parametrize("step,completed", [(1, False), (10, True)])
def test_mirror_finishes_with_trainer_and_surfaces_failure(tmp_path, monkeypatch, step, completed):
    monkeypatch.setattr(module, "remote_request", lambda *args: {
        "observed_at": time.time(), "running": [],
        "logs": {"full_model_rl": {"steps": [{"step": step}], "validation": [],
                                 "rl_estimator": "denoising-grpo", "run_config": {"steps": 10}}},
    })
    args = SimpleNamespace(remote_root="/remote", output=tmp_path, job_id="job",
                           instance_id="instance", container_name="trainer")
    if completed:
        assert module.refresh(args) is True
    else:
        with pytest.raises(RuntimeError, match="Trainer exited"):
            module.refresh(args)
