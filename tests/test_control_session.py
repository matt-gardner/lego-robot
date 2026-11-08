import json
import subprocess
from pathlib import Path

import pytest

import so101.control_session as sessions


SESSION = "4a6bf0aa-50d4-403b-b9fd-0c6de2003217"
CONTAINER = "a" * 64


@pytest.fixture
def setup(tmp_path, monkeypatch):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "dashboard_service": "so101-web.service",
        "dashboard_health_url": "http://127.0.0.1:8765/lego-robot/api/runs",
        "gpu_enabled": False,
        "clients": [{"id": "host", "token_sha256": "b" * 64, "actions": ["dashboard.status"]}],
    }))
    presets = tmp_path / "presets.json"
    presets.write_text(json.dumps({"lego-approved-sweep": {
        "repository": "matt-gardner/lego-robot",
        "actions": ["gpu.submit", "gpu.status", "gpu.start", "gpu.stop"],
        "spec_sha256": ["c" * 64],
    }}))
    monkeypatch.setattr(sessions.os, "geteuid", lambda: 0)
    monkeypatch.setattr(sessions, "trusted_path", lambda path: path.resolve(strict=True))
    monkeypatch.setattr(sessions, "trusted_json", lambda path: json.loads(path.read_text()))
    calls = []
    options = {"policy_path": policy, "presets_path": presets,
               "lock_path": tmp_path / "registration.lock",
               "run": lambda args, **kwargs: calls.append(args)}
    return options, policy, calls


def test_registration_is_private_idempotent_and_refuses_rebinding(setup):
    options, policy, calls = setup
    first = sessions.register(SESSION, CONTAINER, "d" * 64, "lego-approved-sweep", **options)
    assert first["registered"] and first["reloaded"]
    assert sessions.register(SESSION, CONTAINER, "d" * 64, "lego-approved-sweep", **options) == first
    clients = json.loads(policy.read_text())["clients"]
    assert len(clients) == 2 and clients[-1]["spec_sha256"] == ["c" * 64]
    assert policy.stat().st_mode & 0o777 == 0o600
    assert (policy.parent / "sessions.json").stat().st_mode & 0o777 == 0o600
    for container, digest in [(CONTAINER, "e" * 64), ("f" * 64, "d" * 64)]:
        with pytest.raises(ValueError, match="rebind"):
            sessions.register(SESSION, container, digest, "lego-approved-sweep", **options)
    assert len(calls) == 4


def test_reload_failure_is_explicit_and_same_binding_retry_recovers(setup):
    options, policy, calls = setup
    def fail(args, **kwargs):
        raise subprocess.CalledProcessError(1, args)
    with pytest.raises(subprocess.CalledProcessError):
        sessions.register(SESSION, CONTAINER, "d" * 64, "lego-approved-sweep",
                          **{**options, "run": fail})
    sessions.register(SESSION, CONTAINER, "d" * 64, "lego-approved-sweep", **options)
    assert len(json.loads(policy.read_text())["clients"]) == 2


def test_revocation_is_bound_to_container_and_retry_is_idempotent(setup):
    options, policy, calls = setup
    sessions.register(SESSION, CONTAINER, "d" * 64, "lego-approved-sweep", **options)
    revoke_options = {key: value for key, value in options.items() if key != "presets_path"}
    with pytest.raises(ValueError, match="another container"):
        sessions.revoke(SESSION, "f" * 64, **revoke_options)
    sessions.revoke(SESSION, CONTAINER, **revoke_options)
    sessions.revoke(SESSION, CONTAINER, **revoke_options)
    assert [client["id"] for client in json.loads(policy.read_text())["clients"]] == ["host"]
    assert json.loads((policy.parent / "sessions.json").read_text()) == {}
