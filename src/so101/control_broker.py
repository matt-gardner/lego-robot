"""Host-only capability broker. Clients never receive host or AWS credentials."""
from __future__ import annotations

import argparse
import base64
import configparser
from datetime import datetime, timezone
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import re
import secrets
import sqlite3
import ssl
import subprocess
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid


LOG = logging.getLogger(__name__)
IDENTIFIER = re.compile(r"[a-z][a-z0-9-]{0,62}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
TERMINAL = {"stopped", "failed"}


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def trusted_text(path: Path) -> str:
    resolved = path.resolve(strict=True)
    for candidate in (path, *path.parents, resolved, *resolved.parents):
        info = candidate.stat()
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError(f"Host file and parents must be root-owned and not writable: {candidate}")
    return resolved.read_text()


def trusted_json(path: Path) -> dict:
    value = json.loads(trusted_text(path))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_policy(policy: dict) -> None:
    if policy.get("dashboard_service") != "so101-web.service":
        raise ValueError("Only the SO101 dashboard unit may be managed")
    if policy.get("dashboard_health_url") != "http://127.0.0.1:8765/lego-robot/api/runs":
        raise ValueError("Dashboard health URL must be the fixed loopback catalog")
    clients = policy.get("clients")
    if not isinstance(clients, list) or not clients:
        raise ValueError("At least one scoped client is required")
    allowed = {
        "dashboard.status", "dashboard.restart", "gpu.submit", "gpu.resume",
        "gpu.start", "gpu.status", "gpu.logs", "gpu.stop",
    }
    identifiers = set()
    hashes = set()
    for client in clients:
        if (set(client) != {"id", "token_sha256", "actions"}
                or not IDENTIFIER.fullmatch(client["id"])
                or not DIGEST.fullmatch(client["token_sha256"])
                or not set(client["actions"]) <= allowed):
            raise ValueError("Invalid client capability")
        if client["id"] in identifiers or client["token_sha256"] in hashes:
            raise ValueError("Client IDs and capability hashes must be unique")
        identifiers.add(client["id"])
        hashes.add(client["token_sha256"])
    if not isinstance(policy.get("gpu_enabled"), bool):
        raise ValueError("gpu_enabled must explicitly be true or false")
    if policy["gpu_enabled"]:
        validate_controller_settings(policy)
        for key in ("max_hours", "max_usd_per_hour", "total_compute_usd", "max_disk_gib"):
            finite_positive(policy[key], key)
        if not 1 <= finite_positive(policy["checkpoint_grace_seconds"], "checkpoint grace") <= 600:
            raise ValueError("Checkpoint grace must be 1..600 seconds")
        if policy["region"] != "us-west-2":
            raise ValueError("Only the qualified Oregon region is supported")
        if not IDENTIFIER.fullmatch(policy["broker_id"]):
            raise ValueError("Invalid broker ID")
        for digest in policy.get("training_authorized_specs", []):
            if not DIGEST.fullmatch(digest):
                raise ValueError("Training authorization must identify an exact spec digest")


def validate_controller_settings(policy: dict) -> None:
    if not isinstance(policy.get("aws_profile"), str) or not IDENTIFIER.fullmatch(policy["aws_profile"]):
        raise ValueError("An explicit controller AWS profile is required")
    if not re.fullmatch(r"arn:aws:iam::[0-9]{12}:role/so101-controller",
                        str(policy.get("controller_role_arn", ""))):
        raise ValueError("The exact so101-controller role ARN is required")
    for key in ("aws_cli", "aws_config"):
        if not isinstance(policy.get(key), str) or not Path(policy[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute host path")
    if policy.get("region") != "us-west-2":
        raise ValueError("Only the qualified Oregon region is supported")


def validate_controller_profile(policy: dict) -> None:
    config = configparser.ConfigParser(interpolation=None)
    config.read_string(trusted_text(Path(policy["aws_config"])))
    section = "profile " + policy["aws_profile"]
    expected = {
        "role_arn": policy["controller_role_arn"],
        "credential_source": "Ec2InstanceMetadata",
        "region": policy["region"],
    }
    if config.defaults() or config.sections() != [section] or dict(config[section]) != expected:
        raise ValueError("Controller config must contain only the pinned role and EC2 credential source")


def controller_request(policy: dict, run, *args: str) -> dict:
    validate_controller_settings(policy)
    validate_controller_profile(policy)
    # Do not inherit host access keys, alternate endpoints, profiles, or credential processes.
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/var/lib/lego-robot-control",
        "AWS_CONFIG_FILE": policy["aws_config"],
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
        "AWS_EC2_METADATA_DISABLED": "false",
        "AWS_METADATA_SERVICE_TIMEOUT": "3",
        "AWS_METADATA_SERVICE_NUM_ATTEMPTS": "1",
        "AWS_MAX_ATTEMPTS": "1",
    }
    result = run([
        policy["aws_cli"], *args, "--profile", policy["aws_profile"],
        "--region", policy["region"], "--output", "json", "--no-cli-pager",
    ], text=True, capture_output=True, check=True, timeout=60, env=environment)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def verify_controller(policy: dict, run=subprocess.run) -> dict:
    identity = controller_request(policy, run, "sts", "get-caller-identity")
    account = policy["controller_role_arn"].split(":")[4]
    prefix = f"arn:aws:sts::{account}:assumed-role/so101-controller/"
    arn = identity.get("Arn")
    if (identity.get("Account") != account or not isinstance(arn, str)
            or not arn.startswith(prefix) or not arn.removeprefix(prefix)):
        raise PermissionError("AWS caller is not the pinned so101-controller role")
    return {"Account": account, "Arn": arn, "region": policy["region"]}


def finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def wait_for_dashboard(url: str, timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Dashboard did not become ready before the readiness deadline")
        try:
            with urlopen(url, timeout=min(2, remaining)) as response:
                if response.status != 200:
                    raise RuntimeError("Dashboard health check did not return HTTP 200")
                return
        except (URLError, TimeoutError, ConnectionError) as error:
            if isinstance(error, HTTPError) and error.code < 500:
                raise
            LOG.warning("Dashboard not ready yet: %s", error)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Dashboard did not become ready before the readiness deadline") from error
            threading.Event().wait(min(.25, remaining))


def validate_spec(spec: dict, policy: dict) -> None:
    required = {
        "commit", "image", "dataset_sha256", "split_sha256", "checkpoint",
        "recipe", "output_uri", "launch_template_id", "launch_template_version",
        "bootstrap_sha256", "ami", "instance_type", "max_hours", "max_usd_per_hour",
        "stop_document", "stop_document_version", "log_group",
        "start_document", "start_document_version",
    }
    if set(spec) != required:
        raise ValueError(f"Job fields differ from schema: {sorted(set(spec) ^ required)}")
    if not COMMIT.fullmatch(spec["commit"]):
        raise ValueError("A full immutable Git commit is required")
    if not re.fullmatch(r"[a-zA-Z0-9./:_-]+@sha256:[0-9a-f]{64}", spec["image"]):
        raise ValueError("Runtime image must be pinned by SHA256 digest")
    for key in ("dataset_sha256", "split_sha256", "bootstrap_sha256"):
        if not DIGEST.fullmatch(spec[key]):
            raise ValueError(f"Invalid {key}")
    checkpoint = spec["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"uri", "version", "sha256"}:
        raise ValueError("Checkpoint requires uri, immutable version, and sha256")
    if not DIGEST.fullmatch(checkpoint["sha256"]) or not checkpoint["version"]:
        raise ValueError("Checkpoint must be versioned and checksummed")
    if not isinstance(spec["recipe"], dict) or not spec["recipe"]:
        raise ValueError("A frozen recipe is required")
    prefix = policy["artifact_prefix"]
    for uri in (checkpoint["uri"], spec["output_uri"]):
        if not isinstance(uri, str) or not uri.startswith(prefix) or ".." in uri.split("/"):
            raise ValueError("Artifacts must stay in the operator-approved S3 prefix")
    if not prefix.startswith("s3://") or not prefix.endswith("/"):
        raise ValueError("Artifact prefix must be a scoped S3 prefix ending in /")
    if spec["instance_type"] not in policy["instance_types"]:
        raise ValueError("Instance type is not approved")
    if spec["ami"] not in policy["amis"] or not re.fullmatch(r"ami-[0-9a-f]+", spec["ami"]):
        raise ValueError("AMI is not approved")
    if not re.fullmatch(r"lt-[0-9a-f]+", spec["launch_template_id"]):
        raise ValueError("Invalid launch template")
    for key in ("launch_template_version", "stop_document_version", "start_document_version"):
        if not isinstance(spec[key], str) or not re.fullmatch(r"[1-9][0-9]*", spec[key]):
            raise ValueError(f"{key} must be an immutable numeric version")
    if spec["stop_document"] not in policy["stop_documents"]:
        raise ValueError("Checkpoint-stop document is not approved")
    if spec["start_document"] not in policy["start_documents"]:
        raise ValueError("Training-start document is not approved")
    if not isinstance(spec["log_group"], str) or not spec["log_group"].startswith(policy["log_prefix"]):
        raise ValueError("Log group is outside the approved prefix")
    hours = finite_positive(spec["max_hours"], "max_hours")
    rate = finite_positive(spec["max_usd_per_hour"], "max_usd_per_hour")
    if hours > finite_positive(policy["max_hours"], "policy max_hours"):
        raise ValueError("Job exceeds maximum lease duration")
    if rate > finite_positive(policy["max_usd_per_hour"], "policy max_usd_per_hour"):
        raise ValueError("Job exceeds approved hourly rate")


class Broker:
    def __init__(self, policy: dict, state: Path, specs: Path, run=subprocess.run):
        validate_policy(policy)
        self.policy, self.specs, self.run = policy, specs, run
        if policy["gpu_enabled"]:
            verify_controller(policy, run)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(state, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, owner TEXT NOT NULL, "
            "spec_id TEXT NOT NULL, digest TEXT NOT NULL, spec TEXT NOT NULL, "
            "status TEXT NOT NULL, instance TEXT, deadline REAL NOT NULL, "
            "reserved REAL NOT NULL, parent TEXT, error TEXT, "
            "stop_command TEXT, stop_deadline REAL)"
        )
        self.db.commit()

    def aws(self, *args: str) -> dict:
        return controller_request(self.policy, self.run, *args)

    def identity(self, token: str, action: str) -> str:
        digest = hashlib.sha256(token.encode()).hexdigest()
        for entry in self.policy["clients"]:
            if hmac.compare_digest(digest, entry["token_sha256"]):
                if action not in entry["actions"]:
                    raise PermissionError("Capability does not permit this action")
                return entry["id"]
        raise PermissionError("Invalid capability")

    def job(self, job_id: str, owner: str) -> dict:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("Invalid job ID")
        row = self.db.execute("SELECT * FROM jobs WHERE id=? AND owner=?", (job_id, owner)).fetchone()
        if row is None:
            raise PermissionError("Job is not owned by this capability")
        return dict(row)

    def update(self, job_id: str, **fields: object) -> None:
        if not set(fields) <= {"status", "instance", "error", "stop_command", "stop_deadline"}:
            raise ValueError("Invalid state update")
        with self.db:
            self.db.execute(
                f"UPDATE jobs SET {','.join(key + '=?' for key in fields)} WHERE id=?",
                (*fields.values(), job_id),
            )

    def instances(self, job_id: str) -> list[dict]:
        data = self.aws(
            "ec2", "describe-instances", "--filters",
            "Name=tag:so101-broker,Values=" + self.policy["broker_id"],
            "Name=tag:so101-job,Values=" + job_id,
        )
        return [item for group in data.get("Reservations", []) for item in group["Instances"]]

    def owned_instance(self, job: dict) -> dict:
        items = [item for item in self.instances(job["id"]) if item["InstanceId"] == job["instance"]]
        if len(items) != 1:
            raise RuntimeError("Cannot verify instance ownership; operator recovery required")
        return items[0]

    def reserve(self, owner: str, spec_id: str, digest: str, parent: str | None) -> dict:
        if self.policy.get("gpu_enabled") is not True:
            raise ValueError("GPU provisioning is disabled pending host qualification")
        if not IDENTIFIER.fullmatch(spec_id):
            raise ValueError("Invalid approved job specification ID")
        spec = trusted_json(self.specs / (spec_id + ".json"))
        validate_spec(spec, self.policy)
        actual = hashlib.sha256(canonical(spec)).hexdigest()
        if not hmac.compare_digest(actual, digest):
            raise ValueError("Approved job specification digest changed")
        if parent:
            previous = self.job(parent, owner)
            if previous["status"] not in TERMINAL:
                raise ValueError("Stop the previous attempt before resuming")
            old = json.loads(previous["spec"])
            if spec["checkpoint"] == old["checkpoint"]:
                raise ValueError("Resume requires an independently verified new checkpoint specification")
            if any(spec[key] != old[key] for key in (
                "commit", "image", "dataset_sha256", "split_sha256", "recipe",
            )):
                raise ValueError("Resume must preserve code, runtime, dataset, split, and recipe")
        cost = spec["max_hours"] * spec["max_usd_per_hour"]
        now = time.time()
        with self.db:
            active = self.db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status NOT IN ('stopped','failed')"
            ).fetchone()[0]
            if active:
                raise ValueError("Only one nonterminal GPU attempt is allowed")
            reserved = self.db.execute("SELECT COALESCE(SUM(reserved),0) FROM jobs").fetchone()[0]
            if reserved + cost > finite_positive(self.policy["total_compute_usd"], "total_compute_usd"):
                raise ValueError("Lifetime compute reservation budget exhausted")
            job_id = uuid.uuid4().hex
            self.db.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, owner, spec_id, actual, canonical(spec).decode(), "provisioning",
                 None, now + spec["max_hours"] * 3600, cost, parent, None, None, None),
            )
        return self.job(job_id, owner)

    def schedule(self, job: dict) -> None:
        when = datetime.fromtimestamp(job["deadline"], timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        self.aws(
            "scheduler", "create-schedule", "--name", "so101-" + job["id"],
            "--client-token", job["id"],
            "--group-name", self.policy["schedule_group"],
            "--schedule-expression", f"at({when})", "--schedule-expression-timezone", "UTC",
            "--flexible-time-window", '{"Mode":"OFF"}', "--action-after-completion", "DELETE",
            "--target", json.dumps({
                "Arn": "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances",
                "RoleArn": self.policy["shutdown_role_arn"],
                "Input": json.dumps({"InstanceIds": [job["instance"]]}),
                "RetryPolicy": {"MaximumEventAgeInSeconds": 3600, "MaximumRetryAttempts": 10},
            }),
        )

    def submit(self, owner: str, spec_id: str, digest: str, parent: str | None = None) -> dict:
        with self.lock:
            job = self.reserve(owner, spec_id, digest, parent)
            spec = json.loads(job["spec"])
            try:
                template = self.aws(
                    "ec2", "describe-launch-template-versions", "--launch-template-id",
                    spec["launch_template_id"], "--versions", spec["launch_template_version"],
                )["LaunchTemplateVersions"][0]["LaunchTemplateData"]
                bootstrap = base64.b64decode(template["UserData"], validate=True)
                if (template.get("ImageId") != spec["ami"]
                        or hashlib.sha256(bootstrap).hexdigest() != spec["bootstrap_sha256"]):
                    raise ValueError("Launch template does not match the qualified worker bootstrap/AMI")
                if template.get("InstanceType") != spec["instance_type"]:
                    raise ValueError("Launch template instance type differs from job specification")
                if template.get("InstanceMarketOptions", {}).get("MarketType") != "spot":
                    raise ValueError("Only qualified Spot worker templates are permitted")
                spot = template["InstanceMarketOptions"]["SpotOptions"]
                if (spot.get("SpotInstanceType", "one-time") != "one-time"
                        or spot.get("InstanceInterruptionBehavior", "terminate") != "terminate"
                        or finite_positive(float(spot["MaxPrice"]), "Spot MaxPrice")
                        > spec["max_usd_per_hour"]):
                    raise ValueError("Spot template must have a bounded price and no persistent replacement")
                if template.get("MetadataOptions", {}).get("HttpTokens") != "required":
                    raise ValueError("Worker must require IMDSv2")
                if template.get("MetadataOptions", {}).get("InstanceMetadataTags") != "enabled":
                    raise ValueError("Worker must expose lease tags through IMDSv2")
                if template.get("InstanceInitiatedShutdownBehavior") != "terminate":
                    raise ValueError("Worker OS shutdown must terminate the instance")
                disks = template.get("BlockDeviceMappings", [])
                if not disks or any(
                    not item.get("Ebs", {}).get("Encrypted")
                    or item["Ebs"].get("DeleteOnTermination") is not True
                    for item in disks
                ):
                    raise ValueError("Qualified worker disks must be encrypted and ephemeral")
                if sum(item["Ebs"].get("VolumeSize", 0) for item in disks) > self.policy["max_disk_gib"]:
                    raise ValueError("Worker disk allocation exceeds policy")
                tags = [
                    {"Key": "so101-broker", "Value": self.policy["broker_id"]},
                    {"Key": "so101-job", "Value": job["id"]},
                    {"Key": "so101-spec", "Value": job["digest"]},
                    {"Key": "so101-deadline", "Value": str(int(job["deadline"]))},
                ]
                result = self.aws(
                    "ec2", "run-instances", "--launch-template", json.dumps({
                        "LaunchTemplateId": spec["launch_template_id"],
                        "Version": spec["launch_template_version"],
                    }), "--min-count", "1", "--max-count", "1", "--client-token", job["id"],
                    "--tag-specifications", json.dumps([
                        {"ResourceType": kind, "Tags": tags} for kind in ("instance", "volume")
                    ]),
                )
                instance = result["Instances"][0]["InstanceId"]
                self.update(job["id"], instance=instance)
                job["instance"] = instance
                self.schedule(job)
                self.update(job["id"], status="idle")
            except (subprocess.SubprocessError, OSError, ValueError, KeyError, IndexError) as error:
                # An ambiguous launch must retain its reservation until reconciliation.
                self.update(job["id"], status="recovery-required", error=str(error))
                LOG.exception("Provisioning failed for job %s; reconciling paid resources", job["id"])
                self.reconcile(job["id"])
                raise
            return self.job(job["id"], owner)

    def reconcile(self, job_id: str) -> None:
        for item in self.instances(job_id):
            if item["State"]["Name"] != "terminated":
                self.aws("ec2", "terminate-instances", "--instance-ids", item["InstanceId"])
        # Keep recovery-required until AWS confirms termination on a later check.

    def status(self, owner: str, job_id: str) -> dict:
        with self.lock:
            job = self.job(job_id, owner)
            if job["status"] in TERMINAL:
                return job
            items = self.instances(job_id)
            if items and all(item["State"]["Name"] == "terminated" for item in items):
                self.update(job_id, status="stopped")
            elif time.time() >= job["deadline"] or job["status"] in {"provisioning", "recovery-required"}:
                self.reconcile(job_id)
            elif job["status"] == "stopping":
                terminate = time.time() >= job["stop_deadline"]
                if not terminate:
                    invocation = self.aws(
                        "ssm", "get-command-invocation", "--command-id", job["stop_command"],
                        "--instance-id", job["instance"],
                    )
                    state = invocation["Status"]
                    terminate = state in {"Success", "Cancelled", "TimedOut", "Failed"}
                    if terminate and state != "Success":
                        self.update(job_id, error="Checkpoint-stop command ended with " + state)
                if terminate:
                    if time.time() >= job["stop_deadline"]:
                        self.update(job_id, error="Checkpoint grace expired; termination requested")
                    self.reconcile(job_id)
            return self.job(job_id, owner)

    def start(self, owner: str, job_id: str) -> dict:
        with self.lock:
            job = self.status(owner, job_id)
            if job["status"] != "idle" or time.time() >= job["deadline"]:
                raise ValueError("Only an idle, unexpired attempt may start training")
            if job["digest"] not in self.policy.get("training_authorized_specs", []):
                raise PermissionError("This job has no separate operator training authorization")
            instance = self.owned_instance(job)
            if instance["State"]["Name"] != "running":
                raise ValueError("Worker is not running")
            spec = json.loads(job["spec"])
            # Persist intent before dispatch; an ambiguous command must never be repeated.
            self.update(job_id, status="start-requested")
            self.aws(
                "ssm", "send-command", "--document-name", spec["start_document"],
                "--document-version", spec["start_document_version"],
                "--instance-ids", job["instance"], "--parameters",
                json.dumps({"JobId": [job_id], "SpecSHA256": [job["digest"]]}),
            )
            return self.job(job_id, owner)

    def stop(self, owner: str, job_id: str) -> dict:
        with self.lock:
            job = self.job(job_id, owner)
            if job["status"] in TERMINAL:
                return job
            if job["status"] == "stopping":
                return self.status(owner, job_id)
            if not job["instance"]:
                self.reconcile(job_id)
                return self.job(job_id, owner)
            self.owned_instance(job)
            spec = json.loads(job["spec"])
            try:
                command = self.aws(
                    "ssm", "send-command", "--document-name", spec["stop_document"],
                    "--document-version", spec["stop_document_version"],
                    "--instance-ids", job["instance"], "--parameters",
                    json.dumps({"JobId": [job_id], "SpecSHA256": [job["digest"]]}),
                )["Command"]["CommandId"]
            except (subprocess.SubprocessError, OSError, KeyError):
                LOG.exception("Checkpoint request failed for %s; requesting termination", job_id)
                self.reconcile(job_id)
                raise
            self.update(
                job_id, status="stopping", stop_command=command,
                stop_deadline=min(job["deadline"], time.time() + self.policy["checkpoint_grace_seconds"]),
            )
            return self.job(job_id, owner)

    def guard(self) -> None:
        with self.lock:
            jobs = self.db.execute(
                "SELECT id,owner FROM jobs WHERE status NOT IN ('stopped','failed')"
            ).fetchall()
            for job in jobs:
                try:
                    self.status(job["owner"], job["id"])
                except (OSError, subprocess.SubprocessError, RuntimeError, configparser.Error,
                        KeyError, ValueError):
                    LOG.exception("GPU lease/checkpoint guard failed for %s", job["id"])

    def watch(self, stopped: threading.Event) -> None:
        while not stopped.is_set():
            self.guard()
            stopped.wait(10)

    def logs(self, owner: str, job_id: str) -> dict:
        job = self.job(job_id, owner)
        spec = json.loads(job["spec"])
        return self.aws(
            "logs", "get-log-events", "--log-group-name", spec["log_group"],
            "--log-stream-name", job_id, "--limit", "100",
        )

    def dashboard(self, restart: bool) -> dict:
        service = self.policy["dashboard_service"]
        if service != "so101-web.service":
            raise ValueError("Only so101-web.service can be managed")
        if restart:
            self.run(["/usr/bin/systemctl", "restart", service], check=True, timeout=30,
                     text=True, capture_output=True)
        result = self.run(
            ["/usr/bin/systemctl", "is-active", service], check=True, timeout=10,
            text=True, capture_output=True,
        )
        wait_for_dashboard(self.policy["dashboard_health_url"])
        return {"service": service, "state": result.stdout.strip(), "healthy": True}

    def dispatch(self, token: str, action: str, body: dict) -> dict:
        owner = self.identity(token, action)
        fields = {
            "dashboard.status": set(), "dashboard.restart": set(),
            "gpu.submit": {"spec_id", "spec_sha256"},
            "gpu.resume": {"spec_id", "spec_sha256", "parent"},
            "gpu.status": {"job_id"}, "gpu.logs": {"job_id"}, "gpu.stop": {"job_id"},
            "gpu.start": {"job_id"},
        }
        if action not in fields or set(body) != fields[action]:
            raise ValueError("Unknown action or unexpected request fields")
        if action.startswith("dashboard."):
            return self.dashboard(action.endswith(".restart"))
        if action in {"gpu.submit", "gpu.resume"}:
            return self.submit(owner, body["spec_id"], body["spec_sha256"], body.get("parent"))
        return getattr(self, action.split(".")[1])(owner, body["job_id"])


def make_server(broker: Broker, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Request body must be 1..8192 bytes")
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Transfer encoding is not supported")
                auth = self.headers.get("Authorization", "")
                if not auth.startswith("Bearer "):
                    raise PermissionError("Bearer capability required")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                result = broker.dispatch(auth[7:], self.path.removeprefix("/"), body)
                self.reply(200, result)
            except PermissionError as error:
                self.reply(403, {"error": str(error)})
            except (ValueError, TypeError) as error:
                self.reply(400, {"error": str(error)})
            except (OSError, subprocess.SubprocessError, RuntimeError, configparser.Error, KeyError, IndexError):
                LOG.exception("Broker operation failed: %s", self.path)
                self.reply(503, {"error": "Host operation failed; inspect broker journal"})

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(15)

        def reply(self, status: int, value: dict) -> None:
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args: object) -> None:
            LOG.info("%s %s", self.client_address[0], fmt % args)

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--policy", type=Path, required=True)
    serve.add_argument("--state", type=Path, required=True)
    serve.add_argument("--specs", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--cert", type=Path, required=True)
    serve.add_argument("--key", type=Path, required=True)
    sub.add_parser("token")
    preflight = sub.add_parser("preflight", help="Read-only verification of the pinned AWS controller")
    preflight.add_argument("--policy", type=Path, required=True)
    call = sub.add_parser("call")
    call.add_argument("action", choices=[
        "dashboard.status", "dashboard.restart", "gpu.submit", "gpu.resume",
        "gpu.status", "gpu.logs", "gpu.stop", "gpu.start",
    ])
    call.add_argument("--url", required=True)
    call.add_argument("--token-file", type=Path, required=True)
    call.add_argument("--ca-file", type=Path, required=True)
    call.add_argument("--body", default="{}")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.command == "token":
        token = secrets.token_urlsafe(48)
        print(json.dumps({"token": token, "token_sha256": hashlib.sha256(token.encode()).hexdigest()}))
    elif args.command == "preflight":
        print(json.dumps(verify_controller(trusted_json(args.policy))))
    elif args.command == "call":
        if urlsplit(args.url).scheme != "https":
            parser.error("Capabilities must be sent over verified HTTPS")
        payload = json.dumps(json.loads(args.body)).encode()
        request = Request(args.url.rstrip("/") + "/" + args.action, data=payload, headers={
            "Authorization": "Bearer " + args.token_file.read_text().strip(),
            "Content-Type": "application/json",
        })
        try:
            with urlopen(request, timeout=660, context=ssl.create_default_context(cafile=args.ca_file)) as response:
                print(response.read().decode())
        except HTTPError as error:
            parser.exit(1, error.read().decode() + "\n")
    else:
        policy = trusted_json(args.policy)
        broker = Broker(policy, args.state, args.specs)
        stopped = threading.Event()
        watcher = threading.Thread(target=broker.watch, args=(stopped,), daemon=True)
        watcher.start()
        server = make_server(broker, args.host, args.port)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(args.cert, args.key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        try:
            server.serve_forever()
        finally:
            stopped.set()
            server.server_close()
            watcher.join(timeout=90)


if __name__ == "__main__":
    main()
