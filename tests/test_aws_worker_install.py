import base64
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import venv

import pytest

from so101 import aws_worker_install as module
from so101.aws_worker_install import pull_image
from so101.control_broker import canonical


class Metadata:
    def __init__(self, instance_type="g5.2xlarge"):
        self.instance_type = instance_type

    def token(self):
        return "token"

    def read(self, path, token):
        if path.endswith("document"):
            return json.dumps({"accountId": "123456789012", "region": "us-west-2",
                               "instanceType": self.instance_type, "imageId": "ami-123"})
        return {"so101-job": "a" * 32, "so101-spec": "b" * 64,
                "so101-broker": "broker"}[path.rsplit("/", 1)[-1]]

    def deadline(self):
        return 2000


def setup(tmp_path, monkeypatch):
    for name in ("WORKER", "GUARD", "STATE", "CONFIG", "UNITS"):
        monkeypatch.setattr(module, name, tmp_path / name.lower())
    module.CONFIG.mkdir()
    module.UNITS.mkdir()
    source = module.WORKER / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "src/so101").mkdir(parents=True)
    for name in ("lego-robot-worker-guard.service", "so101-worker-spot.service",
                 "so101-worker-spot.timer"):
        (source / "scripts" / name).write_bytes((Path("scripts") / name).read_bytes())
    (source / "src/so101/aws_worker_guard.py").write_text("# pinned fixture\n")
    manifest = {
        "account_id": "123456789012", "instance_type": "g5.2xlarge", "ami": "ami-123",
        "spec_sha256": "b" * 64, "broker_id": "broker", "aws_cli": "/usr/bin/true",
        "runtime": {"docker_cli": "/usr/bin/true", "source_sha256": "c" * 64,
                    "image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/"
                             "lego-robot-runtime@sha256:" + "d" * 64},
        "spec": {"recipe": {"argv": ["approved"]}},
    }
    plan = {"schema": 1, "spec_sha256": "b" * 64, "argv": ["approved"],
            "archives": {}, "readonly_mounts": {}}
    config = {
        "manifest": str(module.CONFIG / "manifest.json"), "manifest_sha256": "e" * 64,
        "plan": str(module.CONFIG / "plan.json"),
        "plan_sha256": hashlib.sha256(canonical(plan)).hexdigest(),
        "staging_root": str(module.STATE / "staged"),
        "attempts_root": str(module.STATE / "attempts"), "docker_cli": "/usr/bin/true",
        "grace_seconds": 120, "qualification": str(module.CONFIG / "qualification.json"),
    }
    monkeypatch.setattr(module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(module, "trusted_path", lambda path: path.resolve(strict=True))
    monkeypatch.setattr(module, "trusted_json",
                        lambda path: config if path.name == "execution.json" else plan)
    monkeypatch.setattr(module, "load_manifest", lambda *args: manifest)
    events = []

    class Venv:
        def __init__(self, **options):
            assert options == {"with_pip": False, "symlinks": True}

        def create(self, target):
            events.append("venv")
            (target / "lib/python3.12/site-packages").mkdir(parents=True)

    monkeypatch.setattr(module.venv, "EnvBuilder", Venv)

    def run(command, **options):
        events.append(command)
        assert options["timeout"] <= 1000
        if "info" in command:
            return SimpleNamespace(stdout='{"nvidia": {}}')
        return SimpleNamespace(stdout="")

    def stage(*args, **kwargs):
        events.append("stage")
        return module.STATE / "staged" / ("a" * 32)

    monkeypatch.setattr(module, "stage_idle", stage)
    monkeypatch.setattr(module, "prepare", lambda *args: events.append("prepare"))
    monkeypatch.setattr(module, "pull_image", lambda *args, **kwargs: events.append("pull"))
    return manifest, config, events, run


def test_install_guard_before_long_operations_and_leave_idle(tmp_path, monkeypatch):
    _, config, events, run = setup(tmp_path, monkeypatch)
    receipt = module.install(hashlib.sha256(canonical(config)).hexdigest(), "c" * 64,
                             metadata=Metadata(), run=run, wall=lambda: 1000)
    enable = ["/usr/bin/systemctl", "enable", "--now", "lego-robot-worker-guard.service"]
    assert events.index(enable) < events.index("venv") < events.index("pull")
    assert events.index("pull") < events.index("stage") < events.index("prepare")
    assert receipt["phase"] == "prepared-idle"
    assert (module.STATE / "installed.json").is_file()
    assert not (module.CONFIG / "qualification.json").exists()
    assert not any(isinstance(event, list) and "run" in event for event in events)
    assert (module.GUARD / "aws_worker_guard.py").is_file()
    assert (module.WORKER / "lib/python3.12/site-packages/so101-worker.pth").read_text() == (
        str(module.WORKER / "source/src") + "\n")


def test_cpu_identity_refuses_services_or_state(tmp_path, monkeypatch):
    _, config, events, run = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="approved GPU"):
        module.install(hashlib.sha256(canonical(config)).hexdigest(), "c" * 64,
                       metadata=Metadata("m5.large"), run=run, wall=lambda: 1000)
    assert events == []
    assert not module.STATE.exists()
    assert not module.GUARD.exists()


@pytest.mark.parametrize("change", ["digest", "paths", "source", "lease"])
def test_bad_initial_bindings_never_activate_guard(tmp_path, monkeypatch, change):
    _, config, events, run = setup(tmp_path, monkeypatch)
    if change == "paths":
        config["attempts_root"] = "/original/output"
    digest = hashlib.sha256(canonical(config)).hexdigest()
    with pytest.raises(ValueError):
        module.install("0" * 64 if change == "digest" else digest,
                       "0" * 64 if change == "source" else "c" * 64,
                       metadata=Metadata(), run=run,
                       wall=lambda: 3000 if change == "lease" else 1000)
    assert events == []
    assert not module.STATE.exists()


def test_no_overwrite_of_existing_worker_state(tmp_path, monkeypatch):
    _, config, events, run = setup(tmp_path, monkeypatch)
    module.STATE.mkdir()
    (module.STATE / "original").write_text("preserved")
    with pytest.raises(ValueError, match="overwrite"):
        module.install(hashlib.sha256(canonical(config)).hexdigest(), "c" * 64,
                       metadata=Metadata(), run=run, wall=lambda: 1000)
    assert events == []
    assert (module.STATE / "original").read_text() == "preserved"


def test_ecr_pull_uses_instance_role_private_config_and_no_retained_auth(tmp_path, monkeypatch):
    manifest, _, _, _ = setup(tmp_path, monkeypatch)
    calls = []

    def run(command, **options):
        calls.append((command, options))
        if "get-authorization-token" in command:
            assert options["env"]["AWS_SHARED_CREDENTIALS_FILE"] == "/dev/null"
            return SimpleNamespace(stdout=json.dumps({"authorizationData": [{
                "proxyEndpoint": "https://123456789012.dkr.ecr.us-west-2.amazonaws.com",
                "authorizationToken": base64.b64encode(b"AWS:temporary").decode(),
            }]}))
        auth = Path(options["env"]["DOCKER_CONFIG"])
        assert auth.is_dir() and auth.stat().st_mode & 0o077 == 0
        (auth / "config.json").write_text("sensitive temporary authorization")
        if "login" in command:
            assert options["input"] == "temporary"
        return SimpleNamespace(stdout="")

    pull_image(manifest, tmp_path, run=run)
    assert len(calls) == 3
    assert not Path(calls[-1][1]["env"]["DOCKER_CONFIG"]).exists()
    assert calls[-1][0][-1] == manifest["runtime"]["image"]


def test_no_public_or_other_registry_pull(tmp_path, monkeypatch):
    manifest, _, _, _ = setup(tmp_path, monkeypatch)
    manifest["runtime"]["image"] = "public/other@sha256:" + "d" * 64
    with pytest.raises(ValueError, match="private Oregon"):
        pull_image(manifest, tmp_path, run=lambda *args, **kwargs: pytest.fail("cloud call"))


def test_isolated_stdlib_interpreter_imports_worker_without_training_dependencies(tmp_path):
    root = tmp_path / "interpreter"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
    sites, = (root / "lib").glob("python*/site-packages")
    (sites / "so101-worker.pth").write_text(str(Path("src").resolve()) + "\n")
    result = subprocess.run([
        str(root / "bin/python"), "-I", "-c",
        "import sys; import so101.aws_worker_install; import so101.aws_spot; "
        "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; "
        "print('stdlib worker imports verified')",
    ], check=True, capture_output=True, text=True, timeout=30,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "/invalid/untrusted"})
    assert result.stdout.strip() == "stdlib worker imports verified"


def early_guard(tmp_path, monkeypatch):
    manifest, config, events, run = setup(tmp_path, monkeypatch)
    source = module.WORKER / "source"
    for name in ("so101-worker-early.py", "so101-worker-install.py"):
        (source / "scripts" / name).write_bytes((Path("scripts") / name).read_bytes())
    module.GUARD.mkdir()
    early = dict(account_id=manifest["account_id"], region="us-west-2",
                 ami=manifest["ami"], instance_type=manifest["instance_type"],
                 broker_id=manifest["broker_id"], aws_cli=manifest["aws_cli"],
                 source_sha256="c" * 64)
    for target, original, key in (
        (module.GUARD / "early.py", source / "scripts/so101-worker-early.py", "early_sha256"),
        (module.GUARD / "aws_worker_guard.py", source / "src/so101/aws_worker_guard.py",
         "guard_sha256"),
        (module.UNITS / "lego-robot-worker-guard.service",
         source / "scripts/lego-robot-worker-guard.service", "guard_unit_sha256"),
    ):
        target.write_bytes(original.read_bytes())
        early[key] = hashlib.sha256(original.read_bytes()).hexdigest()
    early["launcher_sha256"] = hashlib.sha256(
        (source / "scripts/so101-worker-install.py").read_bytes()).hexdigest()
    (module.GUARD / "early.json").write_bytes(canonical(early))
    receipt = {"schema": 1, "job_id": "a" * 32, "spec_sha256": "b" * 64,
               "config_sha256": hashlib.sha256(canonical(early)).hexdigest()}
    (module.GUARD / "early-receipt.json").write_bytes(canonical(receipt))
    prior = module.trusted_json
    monkeypatch.setattr(module, "trusted_json",
                        lambda path: json.loads(path.read_bytes())
                        if path.parent == module.GUARD else prior(path))
    return config, events, run


def test_only_exact_early_guard_adopted_without_overwrite(tmp_path, monkeypatch):
    config, events, run = early_guard(tmp_path, monkeypatch)
    original = (module.GUARD / "aws_worker_guard.py").read_bytes()
    result = module.install(hashlib.sha256(canonical(config)).hexdigest(), "c" * 64,
                            metadata=Metadata(), run=run, wall=lambda: 1000)
    assert result["phase"] == "prepared-idle"
    assert (module.GUARD / "aws_worker_guard.py").read_bytes() == original
    assert ["/usr/bin/systemctl", "enable", "--now", "lego-robot-worker-guard.service"] not in events


@pytest.mark.parametrize("change", ["guard", "unit", "early", "extra"])
def test_altered_or_ambiguous_early_guard_rejected(tmp_path, monkeypatch, change):
    config, events, run = early_guard(tmp_path, monkeypatch)
    path = {"guard": module.GUARD / "aws_worker_guard.py",
            "unit": module.UNITS / "lego-robot-worker-guard.service",
            "early": module.GUARD / "early.py", "extra": module.GUARD / "unknown"}[change]
    path.write_text("altered")
    with pytest.raises(ValueError, match="ambiguous|differs"):
        module.install(hashlib.sha256(canonical(config)).hexdigest(), "c" * 64,
                       metadata=Metadata(), run=run, wall=lambda: 1000)
    assert events == []
    assert not module.STATE.exists()
