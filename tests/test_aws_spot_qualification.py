import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from so101 import aws_spot_qualification as module
from so101.aws_artifacts import Artifact
from so101.aws_spot_workload import header
from so101.aws_spot_workload import model_optimizer, validate


def setup(tmp_path, monkeypatch):
    job, spec = "a" * 32, "b" * 64
    staged = tmp_path / "staged" / job
    staged.mkdir(parents=True)
    (staged / "training_state.pt").write_bytes(b"original untouched")
    (staged / ".staged.json").write_bytes(b"{}")
    original_sha = hashlib.sha256(b"original untouched").hexdigest()
    binding = {"phase": "staged-idle", "job_id": job, "spec_sha256": spec,
               "manifest_sha256": "c" * 64, "deadline": 2000,
               "staging_receipt_sha256": hashlib.sha256(b"{}").hexdigest()}
    (staged / ".bootstrap.json").write_text(json.dumps(binding))
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    manifest = {
        "spec_sha256": spec, "account_id": "123456789012", "aws_cli": "/usr/bin/true",
        "spec": {"checkpoint": {"sha256": original_sha}, "log_group": "/so101/worker",
                 "output_uri": "s3://approved/project/runs/"},
        "runtime": {"image": "private/worker@sha256:" + "d" * 64,
                    "commit": "e" * 40, "source_sha256": "f" * 64,
                    "docker_cli": "/usr/bin/true"},
    }
    config = {"manifest": "manifest", "manifest_sha256": "c" * 64,
              "staging_root": str(tmp_path / "staged"), "attempts_root": str(attempts)}
    monkeypatch.setattr(module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(module.os, "chown", lambda *args: None)
    monkeypatch.setattr(module, "trusted_path", lambda path: path.resolve(strict=True))
    monkeypatch.setattr(module, "trusted_json", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(module, "load_manifest", lambda *args: manifest)
    monkeypatch.setattr(module, "require_guard", lambda run: None)
    monkeypatch.setattr(module.RuntimeImage, "verify", lambda *args, **kwargs: {})
    return config, job, spec, manifest, staged


def test_disposable_start_isolated_from_original_and_training(tmp_path, monkeypatch):
    config, job, spec, manifest, staged = setup(tmp_path, monkeypatch)
    calls = []
    result = module.start(config, job_id=job, spec_sha256=spec, wall=lambda: 1000,
                          run=lambda command, **options:
                          calls.append(command) or SimpleNamespace(stdout="1" * 64))
    assert result["original_trial_advanced"] is False
    assert not (tmp_path / "attempts" / job).exists()
    assert (tmp_path / "attempts/spot-qualification" / job / "intent.json").is_file()
    assert (staged / "training_state.pt").read_bytes() == b"original untouched"
    command, = calls
    assert "--network=none" in command and "--gpus=device=0" in command
    assert "so101.aws_spot_workload" in command and "so101.train_entity_sim" not in command
    assert str(staged / "training_state.pt") not in " ".join(command)
    assert command[command.index("--padding-bytes") + 1] == str(len(b"original untouched"))


def test_disposable_cannot_overlap_original_training(tmp_path, monkeypatch):
    config, job, spec, _, _ = setup(tmp_path, monkeypatch)
    (tmp_path / "attempts" / job).mkdir()
    with pytest.raises(ValueError, match="overlap"):
        module.start(config, job_id=job, spec_sha256=spec, wall=lambda: 1000,
                     run=lambda *args, **kwargs: pytest.fail("launch"))


def test_publication_state_and_event_namespaces_cannot_be_training_resume(tmp_path, monkeypatch):
    config, job, spec, manifest, staged = setup(tmp_path, monkeypatch)
    module.start(config, job_id=job, spec_sha256=spec, wall=lambda: 1000,
                 run=lambda *args, **kwargs: SimpleNamespace(stdout="1" * 64))
    attempt = tmp_path / "attempts/spot-qualification" / job
    (attempt / "output/training_state.pt").write_bytes(b"x" * 100)
    (tmp_path / "attempts" / (job + "-spot.json")).write_text(json.dumps({
        "job_id": job, "spec_sha256": spec, "deadline": 1100,
        "notice": {"action": "terminate", "time": "1970-01-01T00:18:20+00:00"},
    }))
    prefixes = []

    class Publisher:
        def __init__(self, **kwargs):
            prefixes.append(kwargs["output_prefix"])
            self.prefix = kwargs["output_prefix"]

        def publish(self, source, directory, *, validate_snapshot=None, **kwargs):
            if validate_snapshot is not None:
                copy = directory / "snapshot.pt"
                copy.write_bytes(source.read_bytes())
                copy.chmod(0o600)
                validate_snapshot(copy)
            return Artifact(uri=self.prefix + job + "/manifest.json", version="v1",
                            sha256="a" * 64, size=100)

    def validate(command, **options):
        assert "--user=0:0" in command
        assert "--read-only" in command and "--network=none" in command
        source = Path(next(item for item in command if item.startswith("type=bind,src="))
                      .split("src=", 1)[1].split(",dst=", 1)[0])
        return SimpleNamespace(stdout=json.dumps({
            "step": 1, "size": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "padding_bytes": len(b"original untouched"), "cuda_next_update_exact": False,
        }))

    monkeypatch.setattr(module, "CheckpointPublisher", Publisher)
    isolated = dict(config, attempts_root=str(tmp_path / "attempts/spot-qualification"))
    result = module.publish(isolated, job_id=job, spec_sha256=spec, run=validate)
    assert prefixes == [manifest["spec"]["output_uri"] + "qualification/",
                        manifest["spec"]["output_uri"] + "qualification-events/"]
    assert result["phase"] == "spot-qualification-published"
    assert (staged / "training_state.pt").read_bytes() == b"original untouched"


@pytest.mark.parametrize("change", [{"format": "SFT"}, {"seed": 0}, {"step": 0},
                                   {"padding_bytes": 0}, {"cuda": "12.0"}, {"bf16": False}])
def test_wrong_workload_header_rejected(change):
    state = dict(format="so101-spot-qualification-v1", seed=1061000, step=1,
                 padding_bytes=100, cuda="13.0", bf16=True)
    header(state, 100)
    with pytest.raises(ValueError):
        header(state | change, 100)


def cpu_state(tmp_path):
    import torch
    model, optimizer = model_optimizer(torch, "cpu")
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.randn(16, 64)).square().mean().backward()
        optimizer.step()
    # Synthetic metadata exercises CPU validation only, never claims real CUDA qualification.
    state = dict(format="so101-spot-qualification-v1", seed=1061000, step=2,
                 padding_bytes=100, padding=torch.zeros(100, dtype=torch.uint8),
                 cuda="13.0", bf16=True, model=model.state_dict(),
                 optimizer=optimizer.state_dict(), cpu_rng=torch.get_rng_state(),
                 cuda_rng=torch.zeros(16, dtype=torch.uint8), next_update_sha256="f" * 64)
    path = tmp_path / "disposable.pt"
    torch.save(state, path)
    return torch, path, state


def test_cpu_full_adam_body_validation_is_not_cuda_pass(tmp_path):
    torch, path, state = cpu_state(tmp_path)
    result = validate(torch, path, 100)
    assert result["step"] == 2
    assert result["cuda_next_update_exact"] is False
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("change", ["model-nan", "adam-nan", "adam-step", "rng", "padding"])
def test_corrupt_full_state_body_rejected(tmp_path, change):
    torch, path, state = cpu_state(tmp_path)
    if change == "model-nan":
        state["model"]["weight"][0, 0] = float("nan")
    elif change == "adam-nan":
        state["optimizer"]["state"][0]["exp_avg"][0, 0] = float("nan")
    elif change == "adam-step":
        state["optimizer"]["state"][0]["step"] = torch.tensor(99.)
    elif change == "rng":
        state["cuda_rng"] = torch.zeros(0, dtype=torch.uint8)
    else:
        state["padding"] = torch.zeros(99, dtype=torch.uint8)
    torch.save(state, path)
    with pytest.raises(ValueError):
        validate(torch, path, 100)


def test_cold_gpu_check_binds_exact_staged_artifact_without_training(tmp_path, monkeypatch):
    config, job, spec, manifest, staged = setup(tmp_path, monkeypatch)
    recovered = staged / "spot_recovered_state.pt"
    recovered.write_bytes(b"recovered" * 100)
    artifact = dict(uri="s3://approved/project/runs/qualification/old/state.pt",
                    version="v1", sha256=hashlib.sha256(recovered.read_bytes()).hexdigest(),
                    size=recovered.stat().st_size)
    manifest["artifacts"] = {"spot_recovered_state.pt": artifact}
    calls = []

    def run(command, **options):
        calls.append(command)
        assert "--gpus=device=0" in command and "--user=0:0" in command
        assert "--network=none" in command and "check-cuda" in command
        return SimpleNamespace(stdout=json.dumps({
            "step": 2, "size": artifact["size"], "sha256": artifact["sha256"],
            "padding_bytes": len(b"original untouched"), "cuda_next_update_exact": True,
        }))

    result = module.check_recovered(config, job_id=job, spec_sha256=spec, run=run,
                                    wall=lambda: 1000)
    assert result["artifact"] == artifact
    assert result["status"] == "SPOT_CUDA_RECOVERY_PASS"
    assert result["aggregate_pass"] is False
    assert result["original_trial_advanced"] is False
    assert len(calls) == 1 and not (tmp_path / "attempts" / job).exists()
    assert (staged / "training_state.pt").read_bytes() == b"original untouched"


def test_cold_check_rejects_changed_artifact_before_gpu_call(tmp_path, monkeypatch):
    config, job, spec, manifest, staged = setup(tmp_path, monkeypatch)
    recovered = staged / "spot_recovered_state.pt"
    recovered.write_bytes(b"changed")
    manifest["artifacts"] = {"spot_recovered_state.pt": {"sha256": "e" * 64, "size": 7}}
    with pytest.raises(ValueError, match="verified new input"):
        module.check_recovered(config, job_id=job, spec_sha256=spec, wall=lambda: 1000,
                               run=lambda *args, **kwargs: pytest.fail("GPU call"))
