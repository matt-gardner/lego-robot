import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from so101.aws_worker_run import checkpoint_stop, launch, relative_output
from so101.control_broker import canonical


def arguments():
    return ["--output", "outputs/trial", "--training-state",
            "outputs/trial/training_state.pt", "--device", "cuda"]


@pytest.mark.parametrize("argv", [
    ["--output", "/original", "--device", "cuda"],
    ["--output", "outputs/../original", "--training-state", "original/training_state.pt",
     "--device", "cuda"],
    arguments() + ["--output", "outputs/other"],
    ["--output", "outputs/trial", "--training-state", "outputs/trial/training_state.pt",
     "--device", "mps"],
])
def test_unsafe_or_non_cuda_launch_arguments_fail(argv):
    with pytest.raises(ValueError):
        relative_output(argv)


def setup_launch(tmp_path, monkeypatch):
    import so101.aws_worker_run as module
    staged, attempts = tmp_path / "staged", tmp_path / "attempts"
    staged.mkdir()
    attempts.mkdir()
    (staged / "training_state.pt").write_bytes(b"opaque verified checkpoint")
    (staged / ".staged.json").write_text("{}")
    (staged / "data").mkdir()
    job, digest = "1" * 32, "a" * 64
    plan = {"schema": 1, "spec_sha256": digest, "argv": arguments(),
            "readonly_mounts": {"/opt/so101/data/collection": "data"}}
    binding = {"phase": "staged-idle", "job_id": job, "spec_sha256": digest,
               "manifest_sha256": "b" * 64, "deadline": 100,
               "staging_receipt_sha256": hashlib.sha256(b"{}").hexdigest()}
    manifest = {"spec_sha256": digest,
                "spec": {"recipe": {"argv": arguments()}, "checkpoint": {
                    "sha256": hashlib.sha256((staged / "training_state.pt").read_bytes()).hexdigest()}},
                "runtime": {"docker_cli": "/usr/bin/docker", "image": "worker@sha256:" + "c" * 64}}
    monkeypatch.setattr(module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(module.os, "chown", lambda *args: None)
    monkeypatch.setattr(module, "trusted_path", lambda path: path.resolve(strict=True))
    monkeypatch.setattr(module, "trusted_json",
                        lambda path: binding if path.name == ".bootstrap.json" else plan)
    monkeypatch.setattr(module, "load_manifest", lambda *args: manifest)
    monkeypatch.setattr(module, "require_guard", lambda run: None)
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="d" * 64 + "\n")
    kwargs = dict(manifest_path=tmp_path / "manifest", manifest_sha256="b" * 64,
                  plan_path=tmp_path / "plan",
                  plan_sha256=hashlib.sha256(canonical(plan)).hexdigest(),
                  staged=staged, attempts=attempts, job_id=job, spec_sha256=digest,
                  run=run, wall=lambda: 1)
    return kwargs, calls, staged, attempts


def test_launch_copies_state_to_new_output_and_disallows_replay(tmp_path, monkeypatch):
    kwargs, calls, staged, attempts = setup_launch(tmp_path, monkeypatch)
    original = (staged / "training_state.pt").read_bytes()
    assert launch(**kwargs) == "d" * 64
    args = calls[0]
    assert "--network=none" in args and "--read-only" in args and "--gpus=device=0" in args
    assert args[-len(arguments()):] == arguments()
    assert (attempts / kwargs["job_id"] / "output/training_state.pt").read_bytes() == original
    assert (staged / "training_state.pt").read_bytes() == original
    with pytest.raises(FileExistsError):
        launch(**kwargs)
    assert len(calls) == 1


def test_expired_lease_never_calls_docker(tmp_path, monkeypatch):
    kwargs, calls, staged, attempts = setup_launch(tmp_path, monkeypatch)
    kwargs["wall"] = lambda: 101
    with pytest.raises(ValueError, match="expired"):
        launch(**kwargs)
    assert not calls and not list(attempts.iterdir())


def test_changed_checkpoint_never_calls_docker(tmp_path, monkeypatch):
    kwargs, calls, staged, attempts = setup_launch(tmp_path, monkeypatch)
    (staged / "training_state.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        launch(**kwargs)
    assert not calls


def test_ssm_documents_use_only_validated_environment_parameters():
    for action in ("start", "checkpoint-stop"):
        document = json.loads(Path(f"scripts/so101-worker-{action}.ssm.json").read_text())
        assert document["schemaVersion"] == "2.2"
        assert set(document["parameters"]) == {"JobId", "SpecSHA256"}
        for parameter in document["parameters"].values():
            assert parameter["interpolationType"] == "ENV_VAR"
            assert parameter["allowedPattern"].startswith("^[0-9a-f]")
        commands = document["mainSteps"][0]["inputs"]["runCommand"]
        assert commands[0] == "set -eu"
        assert f"so101.aws_worker_run {action}" in commands[1]
        assert "{{" not in commands[1]


@pytest.mark.parametrize("exit_status", ["0", "1"])
def test_stop_only_new_output_and_distinguishes_failed_exit(tmp_path, monkeypatch, exit_status):
    import so101.aws_worker_run as module
    kwargs, calls, staged, attempts = setup_launch(tmp_path, monkeypatch)
    (staged / "STOP").write_bytes(b"original STOP unchanged")
    launch(**kwargs)
    monkeypatch.setattr(module, "trusted_json", lambda path: json.loads(path.read_text()))
    cli = tmp_path / "docker"
    cli.write_text("fake CLI")
    def wait(args, **options):
        assert args[-2:] == ["wait", "d" * 64]
        assert options["timeout"] == 120
        return subprocess.CompletedProcess(args, 0, stdout=exit_status)
    options = dict(attempts=attempts, job_id=kwargs["job_id"],
                   spec_sha256=kwargs["spec_sha256"], docker_cli=str(cli),
                   grace_seconds=120, run=wait)
    if exit_status == "0":
        checkpoint_stop(**options)
        receipt = json.loads((attempts / kwargs["job_id"] / "checkpoint-stop.json").read_text())
        assert receipt["phase"] == "exited-awaiting-validation"
    else:
        with pytest.raises(ValueError, match="clean"):
            checkpoint_stop(**options)
        assert not (attempts / kwargs["job_id"] / "checkpoint-stop.json").exists()
    assert (staged / "STOP").read_bytes() == b"original STOP unchanged"
    assert (attempts / kwargs["job_id"] / "output/STOP").is_file()
