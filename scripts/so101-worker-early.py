"""Standalone two-phase bootstrap: early lease guard, then trusted SSM provisioning."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import tarfile
import time
from urllib.request import ProxyHandler, Request, build_opener


ROOT = Path("/opt/lego-robot-worker")
UNITS = Path("/etc/systemd/system")
CONFIG = Path("/etc/so101-worker")
PROVISION = Path("/var/lib/so101-provision")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def trusted(path):
    resolved = path.resolve(strict=True)
    for candidate in (path, *path.parents, resolved, *resolved.parents):
        info = candidate.stat()
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError("Bootstrap path must be root-owned and not writable by others")
    return resolved


def write(path, body):
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Metadata:
    def __init__(self):
        self.opener = build_opener(ProxyHandler({}))
        request = Request("http://169.254.169.254/latest/api/token", method="PUT",
                          headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        with self.opener.open(request, timeout=2) as response:
            self.token = response.read(4097).decode()
        if not self.token or len(self.token) > 4096:
            raise ValueError("Invalid IMDS token")

    def read(self, path):
        request = Request("http://169.254.169.254/latest/" + path,
                          headers={"X-aws-ec2-metadata-token": self.token})
        with self.opener.open(request, timeout=2) as response:
            body = response.read(65537)
        if len(body) > 65536:
            raise ValueError("Oversized IMDS response")
        return body


def binding(config, metadata, now):
    if (config.get("region") != "us-west-2"
            or config.get("instance_type") not in {"g5.2xlarge", "g6.2xlarge"}
            or not re.fullmatch(r"[0-9]{12}", config.get("account_id", ""))
            or not re.fullmatch(r"ami-[0-9a-f]+", config.get("ami", ""))
            or not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", config.get("broker_id", ""))):
        raise ValueError("Early bootstrap requires exact approved GPU identity")
    identity = json.loads(metadata.read("dynamic/instance-identity/document"))
    expected = {"accountId": config["account_id"], "region": config["region"],
                "imageId": config["ami"], "instanceType": config["instance_type"]}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Early bootstrap is not on the approved GPU")
    tags = {name: metadata.read("meta-data/tags/instance/so101-" + name).decode()
            for name in ("job", "spec", "broker", "deadline")}
    if (not re.fullmatch(r"[0-9a-f]{32}", tags["job"])
            or not re.fullmatch(r"[0-9a-f]{64}", tags["spec"])
            or tags["broker"] != config["broker_id"]
            or not re.fullmatch(r"[1-9][0-9]{0,11}", tags["deadline"])):
        raise ValueError("Early bootstrap job/spec/broker/deadline tags are invalid")
    launched = datetime.fromisoformat(identity["pendingTime"].replace("Z", "+00:00"))
    if launched.utcoffset() != timezone.utc.utcoffset(launched):
        raise ValueError("GPU launch time must be UTC")
    deadline = min(int(tags["deadline"]), launched.timestamp() + 43200)
    if not math.isfinite(deadline) or deadline <= now:
        raise ValueError("Early bootstrap lease has expired")
    return tags["job"], tags["spec"], deadline


def bootstrap(config, guard, unit, code, *, metadata=None, run=subprocess.run):
    if os.geteuid() != 0:
        raise PermissionError("Early bootstrap requires root")
    job, spec, deadline = binding(config, metadata or Metadata(), time.time())
    if (digest(guard) != config["guard_sha256"]
            or digest(unit) != config["guard_unit_sha256"]
            or digest(code) != config["early_sha256"]
            or any(not re.fullmatch(r"[0-9a-f]{64}", config.get(key, ""))
                   for key in ("source_sha256", "launcher_sha256"))):
        raise ValueError("Early bootstrap code differs from pinned bytes")
    trusted(ROOT.parent)
    trusted(UNITS)
    if ROOT.exists() or (UNITS / "lego-robot-worker-guard.service").exists():
        raise ValueError("Early bootstrap cannot replace existing worker assets")
    ROOT.mkdir(mode=0o755)
    write(ROOT / "early.json", canonical(config))
    write(ROOT / "early.py", code)
    write(ROOT / "aws_worker_guard.py", guard)
    write(UNITS / "lego-robot-worker-guard.service", unit)
    env = {"PATH": "/usr/bin:/bin"}
    run(["/usr/bin/systemctl", "daemon-reload"], check=True, timeout=10, env=env)
    run(["/usr/bin/systemctl", "enable", "--now", "lego-robot-worker-guard.service"],
        check=True, timeout=10, env=env)
    run(["/usr/bin/systemctl", "is-active", "--quiet", "lego-robot-worker-guard.service"],
        check=True, timeout=10, env=env)
    write(ROOT / "early-receipt.json", canonical({
        "schema": 1, "config_sha256": digest(canonical(config)), "job_id": job,
        "spec_sha256": spec, "deadline": deadline,
    }))


def provision(job, spec, encoded, *, metadata=None, run=subprocess.run):
    if os.geteuid() != 0:
        raise PermissionError("Postlaunch provisioning requires trusted root SSM")
    config = json.loads(trusted(ROOT / "early.json").read_bytes())
    for path, expected in (
        (ROOT / "early.py", config["early_sha256"]),
        (ROOT / "aws_worker_guard.py", config["guard_sha256"]),
        (UNITS / "lego-robot-worker-guard.service", config["guard_unit_sha256"]),
    ):
        if path.is_symlink() or digest(trusted(path).read_bytes()) != expected:
            raise ValueError("Early bootstrap assets changed before provisioning")
    metadata = metadata or Metadata()
    actual_job, actual_spec, deadline = binding(config, metadata, time.time())
    if job != actual_job or spec != actual_spec:
        raise ValueError("SSM provisioning does not match instance job/spec tags")
    receipt = json.loads(trusted(ROOT / "early-receipt.json").read_bytes())
    if (receipt.get("schema") != 1 or receipt.get("job_id") != job
            or receipt.get("spec_sha256") != spec
            or receipt.get("config_sha256") != digest(canonical(config))):
        raise ValueError("Early guard receipt does not bind this provisioning")
    if len(encoded) > 32768:
        raise ValueError("Provision payload exceeds its bounded size")
    payload = json.loads(base64.b64decode(encoded, validate=True))
    if set(payload) != {"manifest", "plan", "execution"}:
        raise ValueError("Provision only exact manifest/plan/execution objects")
    manifest, plan, execution = (payload[key] for key in ("manifest", "plan", "execution"))
    fields = {"manifest", "manifest_sha256", "plan", "plan_sha256", "staging_root",
              "attempts_root", "docker_cli", "grace_seconds", "qualification"}
    if set(execution) != fields or set(plan) != {
        "schema", "spec_sha256", "argv", "readonly_mounts", "archives"
    }:
        raise ValueError("Provision only the exact execution and plan schemas")
    paths = {"manifest": "/etc/so101-worker/manifest.json", "plan": "/etc/so101-worker/plan.json",
             "staging_root": "/var/lib/so101-worker/staged",
             "attempts_root": "/var/lib/so101-worker/attempts",
             "qualification": "/etc/so101-worker/qualification.json"}
    if any(execution[key] != value for key, value in paths.items()):
        raise ValueError("Provision execution paths must be fixed worker paths")
    registered = manifest["spec"]
    if (digest(canonical(registered)) != spec or manifest["spec_sha256"] != spec
            or manifest["broker_id"] != config["broker_id"]
            or manifest["account_id"] != config["account_id"]
            or manifest["ami"] != config["ami"]
            or manifest["instance_type"] != config["instance_type"]
            or manifest["runtime"]["source_sha256"] != config["source_sha256"]
            or manifest["aws_cli"] != config["aws_cli"]
            or registered["bootstrap_sha256"] != digest(metadata.read("user-data"))
            or plan["spec_sha256"] != spec or plan["argv"] != registered["recipe"]["argv"]
            or execution["manifest_sha256"] != digest(canonical(manifest))
            or execution["plan_sha256"] != digest(canonical(plan))):
        raise ValueError("Postlaunch payload does not bind frozen template/spec/configuration")
    artifact = manifest["artifacts"]["source.tar"]
    if (artifact["sha256"] != config["source_sha256"]
            or type(artifact["size"]) is not int or not 0 < artifact["size"] <= 64 * 1024**2
            or not re.fullmatch(r"[A-Za-z0-9._+/=-]{1,1024}", artifact["version"])
            or artifact["version"] == "null"
            or not re.fullmatch(r"s3://[a-z0-9.-]+/[A-Za-z0-9._/=-]+", artifact["uri"])):
        raise ValueError("Provision source requires exact bounded S3 version and raw digest")
    if not any(artifact["uri"].startswith(prefix) for prefix in manifest["prefixes"]):
        raise ValueError("Provision source is outside approved artifact prefixes")
    trusted(Path(config["aws_cli"]))
    trusted(CONFIG.parent)
    trusted(PROVISION.parent)

    def bounded(command, **options):
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("Lease expired during postlaunch provisioning")
        options["timeout"] = min(options["timeout"], remaining)
        return run(command, **options)

    bounded(["/usr/bin/systemctl", "is-active", "--quiet", "lego-robot-worker-guard.service"],
            check=True, timeout=10, env={"PATH": "/usr/bin:/bin"})
    PROVISION.mkdir(mode=0o700, exist_ok=True)
    trusted(PROVISION)
    attempt = PROVISION / job
    attempt.mkdir(mode=0o700)
    write(attempt / "intent.json", canonical({"job_id": job, "spec_sha256": spec,
                                            "execution_sha256": digest(canonical(execution))}))
    CONFIG.mkdir(mode=0o700)
    for name, value in payload.items():
        write(CONFIG / (name + ".json"), canonical(value))
    bucket, key = artifact["uri"][5:].split("/", 1)
    archive = attempt / "source.tar"
    result = bounded([
        config["aws_cli"], "s3api", "get-object", "--bucket", bucket, "--key=" + key,
        "--version-id=" + artifact["version"], "--expected-bucket-owner", config["account_id"],
        "--region", "us-west-2", "--output", "json", "--no-cli-pager", str(archive),
    ], check=True, capture_output=True, text=True, timeout=300,
        env={"PATH": "/usr/bin:/bin", "HOME": str(attempt), "AWS_CONFIG_FILE": "/dev/null",
             "AWS_SHARED_CREDENTIALS_FILE": "/dev/null", "AWS_EC2_METADATA_DISABLED": "false",
             "AWS_MAX_ATTEMPTS": "1"})
    if (json.loads(result.stdout).get("VersionId") != artifact["version"]
            or archive.is_symlink() or archive.stat().st_size != artifact["size"]
            or digest(archive.read_bytes()) != config["source_sha256"]):
        raise ValueError("Postlaunch source version/size/body verification failed")
    with tarfile.open(archive, "r:") as bundle:
        member = bundle.getmember("scripts/so101-worker-install.py")
        if not member.isfile() or not 0 < member.size <= 65536:
            raise ValueError("Pinned launcher must be a bounded regular source member")
        body = bundle.extractfile(member).read()
    if digest(body) != config["launcher_sha256"]:
        raise ValueError("Postlaunch launcher differs from frozen raw SHA256")
    launcher = attempt / "launcher.py"
    write(launcher, body)
    bounded([
        "/usr/bin/python3", "-I", str(launcher), "--source-archive", str(archive),
        "--source-sha256", config["source_sha256"],
        "--execution-sha256", digest(canonical(execution)),
    ], check=True, capture_output=True, text=True, timeout=3600, env={"PATH": "/usr/bin:/bin"})
    write(attempt / "prepared.json", canonical({"job_id": job, "spec_sha256": spec,
                                              "phase": "prepared-idle"}))


if __name__ == "__main__":
    provision(os.environ["SSM_JobId"], os.environ["SSM_SpecSHA256"],
              os.environ["SSM_ProvisionBase64"])
