import hashlib
import json
import sqlite3

import pytest

import so101.aws_budget_reconcile as accounting


def job():
    return {"id": "a" * 32, "owner": "owner", "digest": "b" * 64,
            "status": "stopped", "instance": "i-123", "deadline": 7200,
            "max_hours": 2, "max_usd_per_hour": .9776, "reserved": 1.9552}


def receipt():
    return {"total_compute_usd": 20, "jobs": [job()], "evidence": [
        {"job_id": job()["id"], "instance": "i-123", "state": "terminated",
         "observed_at": 418},
    ]}


def test_paid_accounting_rounds_up_and_increases_an_underreserved_overrun():
    changes = accounting.proposals(receipt())
    assert changes[0]["after"] == "0.113511"
    payload = receipt()
    payload["jobs"][0].update(deadline=900, max_hours=.25, reserved=.2444)
    payload["evidence"][0]["observed_at"] = 1300
    assert accounting.proposals(payload)[0]["after"] == "0.353023"


def test_unallocated_requires_exact_negative_ack_and_empty_inventory():
    payload = receipt()
    payload["jobs"][0].update(instance=None, status="failed")
    payload["evidence"] = [{
        "failures": [{"client_token": job()["id"], "errorCode": "Server.InsufficientInstanceCapacity",
                      "eventID": "event"}],
        "instances": {job()["id"]: []},
    }]
    assert accounting.proposals(payload)[0]["after"] == "0"
    payload["evidence"][0]["instances"][job()["id"]] = ["allocated"]
    assert accounting.proposals(payload) == []
    payload["evidence"][0]["instances"][job()["id"]] = []
    payload["jobs"][0]["reserved"] = 0
    assert accounting.proposals(payload) == []


def test_live_paid_identity_mismatch_rejects_accounting():
    item = accounting.proposals(receipt())[0]
    instance = {"InstanceId": "i-123", "State": {"Name": "terminated"},
                "ClientToken": "another", "LaunchTime": "1970-01-01T00:00:01Z",
                "Tags": [{"Key": "so101-job", "Value": job()["id"]},
                         {"Key": "so101-spec", "Value": job()["digest"]}]}
    with pytest.raises(ValueError, match="identity"):
        accounting.verify_live(item, job(), lambda *args: {"Reservations": [{"Instances": [instance]}]})


def test_accounting_preserves_rows_and_audits_once_with_stale_row_guard(tmp_path, monkeypatch):
    payload = receipt()
    evidence = tmp_path / "receipt.json"
    evidence.write_text(json.dumps(payload))
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    policy = tmp_path / "policy.json"
    policy.write_text('{"total_compute_usd":20}')
    database = tmp_path / "jobs.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, owner TEXT, digest TEXT, status TEXT, "
                   "instance TEXT, deadline REAL, reserved REAL)")
        db.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?)",
                   tuple(job()[name] for name in
                         ("id", "owner", "digest", "status", "instance", "deadline", "reserved")))
    monkeypatch.setattr(accounting.os, "geteuid", lambda: 0)
    monkeypatch.setattr(accounting, "trusted_path", lambda path: path)
    monkeypatch.setattr(accounting, "trusted_json", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(accounting, "require_no_gpu", lambda request: None)
    monkeypatch.setattr(accounting, "verify_live", lambda *args: None)
    result = accounting.reconcile(database, evidence, digest, policy, apply=True)
    assert result["population"] == 1 and result["after"] == "0.113511"
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        assert db.execute("SELECT reserved FROM jobs").fetchone()[0] == .113511
        assert db.execute("SELECT COUNT(*) FROM reservation_accounting").fetchone()[0] == 1
    with pytest.raises(ValueError, match="row changed"):
        accounting.reconcile(database, evidence, digest, policy, apply=True)
