import hashlib
import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

from so101.aws_worker_run import checkpoint_stop, launch, publish_stopped, relative_output
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
        expected = {"JobId", "SpecSHA256"}
        if action == "checkpoint-stop":
            expected.add("StopDeadline")
        assert set(document["parameters"]) == expected
        for name, parameter in document["parameters"].items():
            assert parameter["interpolationType"] == "ENV_VAR"
            assert parameter["allowedPattern"].startswith(
                "^[1-9]" if name == "StopDeadline" else "^[0-9a-f]")
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


def test_stopped_checkpoint_validates_snapshot_and_confirms_versions_before_success(tmp_path, monkeypatch):
    import so101.aws_worker_run as module
    kwargs, calls, staged, attempts = setup_launch(tmp_path, monkeypatch)
    launch(**kwargs)
    monkeypatch.setattr(module.os, "geteuid", lambda: (attempts.stat().st_uid))
    job, digest = kwargs["job_id"], kwargs["spec_sha256"]
    manifest = module.load_manifest(None, None)
    manifest.update(account_id="133961745003", aws_cli="/opt/aws/bin/aws", max_bytes=1000)
    manifest["spec"]["output_uri"] = "s3://approved/runs/lego-robot/"
    stage_root = tmp_path / "stage-root"
    (stage_root / job).mkdir(parents=True)
    (stage_root / job / "training_state.pt").write_bytes((staged / "training_state.pt").read_bytes())
    (attempts / job / "checkpoint-stop.json").write_text(json.dumps({
        "phase": "exited-awaiting-validation", "job_id": job, "spec_sha256": digest,
    }))
    monkeypatch.setattr(module, "trusted_json", lambda path: json.loads(path.read_text()))
    cli = tmp_path / "docker"
    cli.write_text("fake CLI")
    manifest["runtime"]["docker_cli"] = str(cli)
    objects, operations = {}, []
    def run(args, **options):
        if args[0] == str(cli):
            operations.append("validate")
            path = next(item.split(",")[1].removeprefix("src=") for item in args
                        if item.startswith("type=bind,src="))
            data = Path(path).read_bytes()
            result = {"step": 1, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        else:
            operation = args[2]
            operations.append(operation)
            assert options["timeout"] == 120
            key = next(item.removeprefix("--key=") for item in args if item.startswith("--key="))
            if operation == "put-object":
                assert "--if-none-match" in args and key not in objects
                data = Path(args[args.index("--body") + 1]).read_bytes()
                objects[key] = {"VersionId": f"version-{len(objects)}", "ContentLength": len(data),
                                "ChecksumSHA256": base64.b64encode(hashlib.sha256(data).digest()).decode()}
            result = objects[key]
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(result))
    config = {"manifest": "rootmanifest", "manifest_sha256": "b" * 64,
              "attempts_root": str(attempts), "staging_root": str(stage_root)}
    result = publish_stopped(config, job_id=job, spec_sha256=digest, run=run)
    assert operations == ["validate", "validate", "put-object", "head-object", "put-object", "head-object"]
    assert result["phase"] == "versioned-checkpoint-published" and len(objects) == 2
    assert json.loads((attempts / job / "recovery.json").read_text()) == result


def test_cli_caps_stop_work_by_controller_deadline_including_dispatch_delay(monkeypatch):
    import so101.aws_worker_run as module
    config = {key: "unused" for key in (
        "manifest", "manifest_sha256", "plan", "plan_sha256", "staging_root",
        "attempts_root", "docker_cli", "qualification")}
    config["grace_seconds"] = 300
    monkeypatch.setattr(module, "trusted_json", lambda path: config)
    monkeypatch.setattr(module.time, "time", lambda: 1000)
    monkeypatch.setattr(module.time, "monotonic", lambda: 1)
    monkeypatch.setattr(sys, "argv", ["worker", "checkpoint-stop", "--job-id", "1" * 32,
                                    "--spec-sha256", "a" * 64, "--stop-deadline", "1010"])
    observed = []
    def run(command, **options):
        observed.append(options["timeout"])
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "checkpoint_stop",
                        lambda *args, **options: options["run"](["wait"], timeout=300))
    monkeypatch.setattr(module, "publish_stopped", lambda *args, **options: {"phase": "published"})
    module.main()
    assert observed == [10]
