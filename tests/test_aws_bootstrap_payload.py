import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import time

import pytest

from so101.aws_bootstrap_payload import userdata
from so101.control_broker import canonical, validate_spec


IDENTITY = dict(account_id="123456789012", region="us-west-2", ami="ami-123abc",
                instance_type="g6.2xlarge", broker_id="broker", aws_cli="/usr/bin/true")


def sha(body):
    return hashlib.sha256(body).hexdigest()


def config():
    return dict(IDENTITY, source_sha256="c" * 64, launcher_sha256="d" * 64,
                early_sha256=sha(Path("scripts/so101-worker-early.py").read_bytes()),
                guard_sha256=sha(Path("src/so101/aws_worker_guard.py").read_bytes()),
                guard_unit_sha256=sha(Path("scripts/lego-robot-worker-guard.service").read_bytes()))


class Metadata:
    def __init__(self, body, spec, instance_type="g6.2xlarge"):
        self.body, self.spec, self.instance_type = body, spec, instance_type

    def read(self, path):
        if path == "user-data":
            return self.body
        if path.endswith("document"):
            return canonical(dict(accountId=IDENTITY["account_id"], region="us-west-2",
                                  imageId=IDENTITY["ami"], instanceType=self.instance_type,
                                  pendingTime=datetime.fromtimestamp(time.time() - 10,
                                                                    timezone.utc).isoformat()))
        return {"so101-job": "a" * 32, "so101-spec": self.spec, "so101-broker": "broker",
                "so101-deadline": str(int(time.time()) + 300)}[path.rsplit("/", 1)[-1]].encode()


def test_full_template_spec_config_construction_is_noncircular(tmp_path, monkeypatch):
    body = userdata(Path("."), IDENTITY, source_sha256="c" * 64, launcher_sha256="d" * 64)
    assert len(body) <= 16384
    template = {"UserData": base64.b64encode(body).decode(), "Version": "7"}
    spec = dict(
        commit="b" * 40, image="private/worker@sha256:" + "e" * 64,
        dataset_sha256="f" * 64, split_sha256="f" * 64,
        checkpoint=dict(uri="s3://approved/project/state.pt", version="v1", sha256="f" * 64),
        recipe={"argv": ["approved"]}, output_uri="s3://approved/project/runs/",
        launch_template_id="lt-123abc", launch_template_version=template["Version"],
        bootstrap_sha256=sha(base64.b64decode(template["UserData"])), ami=IDENTITY["ami"],
        instance_type=IDENTITY["instance_type"], max_hours=1, max_usd_per_hour=.5,
        stop_document="Stop", stop_document_version="1", start_document="Start",
        start_document_version="1", log_group="/so101/worker",
    )
    validate_spec(spec, dict(artifact_prefix="s3://approved/project/",
                            instance_types=["g6.2xlarge"], amis=["ami-123abc"],
                            stop_documents=["Stop"], start_documents=["Start"],
                            log_prefix="/so101/", max_hours=12, max_usd_per_hour=1))
    spec_sha = sha(canonical(spec))
    manifest = dict(spec=spec, spec_sha256=spec_sha, broker_id="broker",
                    account_id=IDENTITY["account_id"], ami=IDENTITY["ami"],
                    instance_type=IDENTITY["instance_type"], aws_cli=IDENTITY["aws_cli"],
                    runtime={"source_sha256": "c" * 64},
                    artifacts={"source.tar": dict(uri="s3://approved/project/source.tar",
                                                 version="v1", size=1024, sha256="c" * 64)},
                    prefixes=["s3://approved/project/"])
    plan = {"schema": 1, "spec_sha256": spec_sha, "argv": ["approved"],
            "readonly_mounts": {}, "archives": {}}
    execution = {"manifest_sha256": sha(canonical(manifest)),
                 "plan_sha256": sha(canonical(plan)), "manifest": "/etc/so101-worker/manifest.json",
                 "plan": "/etc/so101-worker/plan.json", "staging_root": "/var/lib/so101-worker/staged",
                 "attempts_root": "/var/lib/so101-worker/attempts",
                 "qualification": "/etc/so101-worker/qualification.json",
                 "docker_cli": "/usr/bin/true", "grace_seconds": 120}
    payload = base64.b64encode(canonical(dict(manifest=manifest, plan=plan,
                                            execution=execution))).decode()
    # Changing per-job configuration cannot change already-frozen user data.
    assert userdata(Path("."), IDENTITY, source_sha256="c" * 64,
                    launcher_sha256="d" * 64) == body
    namespace = runpy.run_path("scripts/so101-worker-early.py")
    provision = namespace["provision"]
    globals_ = provision.__globals__
    root, units = tmp_path / "early", tmp_path / "units"
    root.mkdir()
    units.mkdir()
    for name, source in (("early.py", "scripts/so101-worker-early.py"),
                         ("aws_worker_guard.py", "src/so101/aws_worker_guard.py")):
        (root / name).write_bytes(Path(source).read_bytes())
    (root / "early.json").write_bytes(canonical(config()))
    (root / "early-receipt.json").write_bytes(canonical({
        "schema": 1, "job_id": "a" * 32, "spec_sha256": spec_sha,
        "config_sha256": sha(canonical(config())),
    }))
    (units / "lego-robot-worker-guard.service").write_bytes(
        Path("scripts/lego-robot-worker-guard.service").read_bytes())
    monkeypatch.setitem(globals_, "ROOT", root)
    monkeypatch.setitem(globals_, "UNITS", units)
    monkeypatch.setitem(globals_, "PROVISION", tmp_path / "provision")
    monkeypatch.setitem(globals_, "CONFIG", tmp_path / "config")
    monkeypatch.setitem(globals_, "trusted", lambda path: path.resolve(strict=True))
    monkeypatch.setattr(namespace["os"], "geteuid", lambda: 0)
    (tmp_path / "config").mkdir()
    calls = []
    # Existing configuration fails closed only AFTER complete hash/tag validation.
    with pytest.raises(FileExistsError):
        provision("a" * 32, spec_sha, payload, metadata=Metadata(body, spec_sha),
                  run=lambda *args, **kwargs: calls.append(args))
    assert len(calls) == 1  # Active guard check, never source download or launch.
    with pytest.raises(ValueError, match="job/spec"):
        provision("0" * 32, spec_sha, payload, metadata=Metadata(body, spec_sha),
                  run=lambda *args, **kwargs: pytest.fail("runtime call"))


def test_wrong_cpu_never_installs_guard(tmp_path, monkeypatch):
    namespace = runpy.run_path("scripts/so101-worker-early.py")
    bootstrap = namespace["bootstrap"]
    monkeypatch.setattr(namespace["os"], "geteuid", lambda: 0)
    with pytest.raises(ValueError, match="approved GPU"):
        bootstrap(config(), b"", b"", b"", metadata=Metadata(b"", "b" * 64, "m5.large"),
                  run=lambda *args, **kwargs: pytest.fail("runtime call"))


def test_userdata_rejects_per_job_hash_inputs():
    with pytest.raises(ValueError, match="must not contain"):
        userdata(Path("."), dict(IDENTITY, spec_sha256="a" * 64),
                 source_sha256="c" * 64, launcher_sha256="d" * 64)
