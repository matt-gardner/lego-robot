import hashlib
import json

import pytest

from so101 import aws_qualification_recovery as module


def setup(tmp_path, monkeypatch, *, change=None, logs=True):
    job, spec = "a" * 32, "b" * 64
    state_manifest = {"uri": "s3://approved/runs/qualification/" + job + "/manifest.json",
                      "version": "v1", "sha256": "c" * 64, "size": 100}
    body = dict(job_id=job, spec_sha256=spec, workload="spot-qualification",
                original_trial_advanced=False, original_sha256="d" * 64,
                padding_bytes=100, state_bytes=100, state_manifest=state_manifest,
                log_stream=job + "-spot-qualification", deadline=1100,
                notice={"action": "terminate", "time": "1970-01-01T00:18:20+00:00"})
    body.update(change or {})
    raw = json.dumps(body).encode()
    artifact = {"uri": "s3://approved/runs/qualification-events/" + job + "/evidence.pt",
                "version": "v1", "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
    prefixes = []

    def recover(request, root, **options):
        prefixes.append(options["prefix"])
        (root / job).mkdir(parents=True, exist_ok=True)
        if options["prefix"].endswith("qualification/"):
            return {"manifest": state_manifest, "checkpoint": {"size": 100}}
        return {"checkpoint": artifact}

    def request(*args):
        if "get-object" in args:
            from pathlib import Path
            Path(args[-1]).write_bytes(raw)
            return {"VersionId": "v1"}
        if "head-object" in args:
            return {"VersionId": "original-v1", "ContentLength": 100}
        if "get-log-events" in args:
            return {"events": [{"message": json.dumps({
                "workload": "so101-spot-qualification-v1", "checkpoint_stop": True,
                "original_trial_advanced": False,
            })}]} if logs else {"events": []}
        pytest.fail("unexpected request")

    monkeypatch.setattr(module, "recover", recover)
    kwargs = dict(request=request, root=tmp_path, owner="123456789012",
                  prefix="s3://approved/runs/", job_id=job, spec_sha256=spec,
                  original_checkpoint={"uri": "s3://approved/original.pt",
                                       "version": "original-v1", "sha256": "d" * 64},
                  log_group="/so101/worker", termination_observed_at=1050)
    return kwargs, prefixes


def test_evidence_is_namespace_isolated_and_never_aggregate_pass_or_local_export(tmp_path, monkeypatch):
    kwargs, prefixes = setup(tmp_path, monkeypatch)
    result = module.evidence(**kwargs)
    assert prefixes == ["s3://approved/runs/qualification/",
                        "s3://approved/runs/qualification-events/"]
    assert result["status"] == "QUALIFICATION_EVIDENCE"
    assert result["termination_observed_by_deadline"] is True
    assert result["checkpoint_stop_log_verified"] is True
    assert result["aggregate_pass"] is False
    assert result["local_full_checkpoint_export_verified"] is False


@pytest.mark.parametrize("change", [
    {"original_trial_advanced": True}, {"original_sha256": "e" * 64},
    {"padding_bytes": 99}, {"state_bytes": 99}, {"state_manifest": {}},
    {"deadline": 1200}, {"job_id": "e" * 32},
])
def test_unbound_or_unrepresentative_evidence_rejected(tmp_path, monkeypatch, change):
    kwargs, _ = setup(tmp_path, monkeypatch, change=change)
    with pytest.raises(ValueError):
        module.evidence(**kwargs)


def test_late_termination_and_missing_logs_are_explicit_unproven_flags(tmp_path, monkeypatch):
    kwargs, _ = setup(tmp_path, monkeypatch, logs=False)
    result = module.evidence(**(kwargs | {"termination_observed_at": 1200}))
    assert result["termination_observed_by_deadline"] is False
    assert result["checkpoint_stop_log_verified"] is False
    assert result["aggregate_pass"] is False
