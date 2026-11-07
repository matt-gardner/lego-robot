from datetime import datetime, timezone
import json
import subprocess
from urllib.error import HTTPError

import pytest

from so101.aws_spot import notice_deadline, tick


def setup(tmp_path, monkeypatch, notice):
    import so101.aws_spot as module
    job, digest = "1" * 32, "a" * 64
    manifest = {"account_id": "133961745003", "ami": "ami-123abc",
                "instance_type": "g6.2xlarge", "spec_sha256": digest, "broker_id": "lego-robot"}
    monkeypatch.setattr(module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(module, "load_manifest", lambda *args: manifest)
    monkeypatch.setattr(module, "trusted_path", lambda path: path.resolve(strict=True))
    monkeypatch.setattr(module, "trusted_json", lambda path: json.loads(path.read_text()))
    class Metadata:
        def token(self):
            return "token"
        def deadline(self):
            return 1300
        def read(self, path, token):
            assert token == "token"
            if path == "dynamic/instance-identity/document":
                return json.dumps({"accountId": manifest["account_id"], "region": "us-west-2",
                                   "imageId": manifest["ami"], "instanceType": "g6.2xlarge"})
            if path.endswith("so101-job"):
                return job
            if path.endswith("so101-spec"):
                return digest
            if path.endswith("so101-broker"):
                return "lego-robot"
            if isinstance(notice, Exception):
                raise notice
            return json.dumps(notice)
    config = {"manifest": "rootmanifest", "manifest_sha256": "b" * 64,
              "attempts_root": str(tmp_path)}
    events = []
    return config, Metadata(), events, job


def notice(time=1100):
    return {"action": "terminate", "time": datetime.fromtimestamp(time, timezone.utc).isoformat()}


def test_absent_spot_notice_is_idle_not_error_or_shutdown(tmp_path, monkeypatch):
    config, metadata, events, job = setup(
        tmp_path, monkeypatch, HTTPError("http://imds", 404, "no notice", {}, None))
    assert not tick(config, metadata=metadata, wall=lambda: 1000,
                    shutdown=lambda: events.append("shutdown"))
    assert not events and not list(tmp_path.iterdir())


def test_non404_metadata_failure_propagates_without_shutting_down_cpu(tmp_path, monkeypatch):
    config, metadata, events, job = setup(
        tmp_path, monkeypatch, HTTPError("http://imds", 401, "unauthorized", {}, None))
    with pytest.raises(HTTPError):
        tick(config, metadata=metadata, shutdown=lambda: events.append("shutdown"))
    assert not events


def test_spot_on_running_attempt_publishes_before_shutdown_without_replay(tmp_path, monkeypatch):
    config, metadata, events, job = setup(tmp_path, monkeypatch, notice())
    (tmp_path / job).mkdir()
    (tmp_path / job / "launched.json").write_text("{}")
    def stop(config, **options):
        assert options["stop_deadline"] == 1100
        events.append("published")
    kwargs = dict(metadata=metadata, wall=lambda: 1000, stop=stop,
                  shutdown=lambda: events.append("shutdown"))
    assert tick(config, **kwargs)
    assert events == ["published", "shutdown"]
    assert tick(config, **kwargs)
    assert events == ["published", "shutdown", "shutdown"]


def test_failed_checkpoint_requests_shutdown_and_keeps_error(tmp_path, monkeypatch):
    config, metadata, events, job = setup(tmp_path, monkeypatch, notice())
    (tmp_path / job).mkdir()
    (tmp_path / job / "launched.json").write_text("{}")
    def failed(*args, **options):
        raise subprocess.TimeoutExpired(["checkpoint"], 120)
    with pytest.raises(subprocess.TimeoutExpired):
        tick(config, metadata=metadata, wall=lambda: 1000, stop=failed,
             shutdown=lambda: events.append("shutdown"))
    assert events == ["shutdown"]


def test_notice_is_capped_by_two_minutes_and_original_lease():
    assert notice_deadline(notice(1400), 1000, 1300) == 1120
    assert notice_deadline(notice(1100), 1000, 1050) == 1050
    with pytest.raises(ValueError, match="UTC"):
        notice_deadline({"action": "terminate", "time": "2026-11-03T00:00:00"}, 1000, 1300)


def test_real_notice_routes_disposable_attempt_and_publishes_before_shutdown(tmp_path, monkeypatch):
    import so101.aws_spot as module
    config, metadata, events, job = setup(tmp_path, monkeypatch, notice())
    attempt = tmp_path / "spot-qualification" / job
    attempt.mkdir(parents=True)
    (attempt / "launched.json").write_text("{}")

    def qualification(config, **options):
        assert options["stop_deadline"] == 1100
        assert (attempt / "stop-deadline.json").is_file()
        events.append("qualified-publication")

    monkeypatch.setattr(module, "qualification_stop", qualification)
    assert tick(config, metadata=metadata, wall=lambda: 1000,
                stop=lambda *args, **kwargs: pytest.fail("training stop"),
                shutdown=lambda: events.append("shutdown"))
    assert events == ["qualified-publication", "shutdown"]


def test_ambiguous_training_and_disposable_attempts_fail_closed(tmp_path, monkeypatch):
    config, metadata, events, job = setup(tmp_path, monkeypatch, notice())
    (tmp_path / job).mkdir()
    (tmp_path / "spot-qualification" / job).mkdir(parents=True)
    with pytest.raises(ValueError, match="must not overlap"):
        tick(config, metadata=metadata, wall=lambda: 1000,
             shutdown=lambda: events.append("shutdown"))
    assert events == []
