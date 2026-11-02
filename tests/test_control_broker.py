import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request, urlopen

from so101.control_broker import Broker, canonical, make_server, trusted_json, validate_spec, wait_for_dashboard


class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.policy = {
            "clients": [{"id": "session", "token_sha256": hashlib.sha256(b"secret").hexdigest(),
                         "actions": ["dashboard.status", "gpu.submit", "gpu.status"]}],
            "dashboard_service": "so101-web.service",
            "dashboard_health_url": "http://127.0.0.1:8765/lego-robot/api/runs",
            "gpu_enabled": True, "broker_id": "lego-robot", "aws_cli": "/usr/bin/aws",
            "region": "us-west-2", "artifact_prefix": "s3://approved/lego-robot/",
            "instance_types": ["g6.2xlarge"], "amis": ["ami-123abc"],
            "max_hours": 2, "max_usd_per_hour": 1, "total_compute_usd": 4,
            "stop_documents": ["CheckpointStop"], "start_documents": ["StartApprovedJob"],
            "log_prefix": "/lego-robot/", "schedule_group": "lego-robot",
            "shutdown_role_arn": "arn:aws:iam::123456789012:role/shutdown",
            "checkpoint_grace_seconds": 300, "max_disk_gib": 100,
        }
        self.spec = {
            "commit": "a" * 40, "image": "example/worker@sha256:" + "b" * 64,
            "dataset_sha256": "c" * 64, "split_sha256": "d" * 64,
            "checkpoint": {"uri": "s3://approved/lego-robot/state.pt", "version": "version1",
                           "sha256": "e" * 64},
            "recipe": {"batch": 64}, "output_uri": "s3://approved/lego-robot/results/",
            "launch_template_id": "lt-123abc", "launch_template_version": "1",
            "bootstrap_sha256": hashlib.sha256(b"qualified idle bootstrap").hexdigest(),
            "ami": "ami-123abc", "instance_type": "g6.2xlarge",
            "max_hours": 1, "max_usd_per_hour": .5,
            "stop_document": "CheckpointStop", "stop_document_version": "1",
            "start_document": "StartApprovedJob", "start_document_version": "1",
            "log_group": "/lego-robot/jobs",
        }
        self.calls = []
        self.broker = Broker(self.policy, self.root / "state.sqlite", self.root, self.command)
        self.trust = patch("so101.control_broker.trusted_json", return_value=self.spec)
        self.trust.start()

    def tearDown(self):
        self.trust.stop()
        self.broker.db.close()
        self.temporary.cleanup()

    def digest(self):
        return hashlib.sha256(canonical(self.spec)).hexdigest()

    def command(self, args, **kwargs):
        self.calls.append(args)
        if "describe-launch-template-versions" in args:
            data = {"LaunchTemplateVersions": [{"LaunchTemplateData": {
                "ImageId": "ami-123abc", "InstanceType": "g6.2xlarge",
                "UserData": base64.b64encode(b"qualified idle bootstrap").decode(),
                "InstanceMarketOptions": {"MarketType": "spot", "SpotOptions": {"MaxPrice": "0.5"}},
                "MetadataOptions": {"HttpTokens": "required"},
                "BlockDeviceMappings": [{"Ebs": {
                    "Encrypted": True, "DeleteOnTermination": True, "VolumeSize": 80,
                }}],
            }}]}
        elif "run-instances" in args:
            data = {"Instances": [{"InstanceId": "i-123abc"}]}
        elif "describe-instances" in args:
            data = {"Reservations": [{"Instances": [{
                "InstanceId": "i-123abc", "State": {"Name": "running"},
            }]}]}
        elif "send-command" in args:
            data = {"Command": {"CommandId": "command-1"}}
        elif "get-command-invocation" in args:
            data = {"Status": "Success"}
        else:
            data = {}
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(data))

    def reserve(self):
        return self.broker.reserve("session", "approved", self.digest(), None)

    def test_strict_spec(self):
        validate_spec(self.spec, self.policy)
        for key, value in (
            ("commit", "main"), ("image", "example/worker:latest"),
            ("launch_template_version", "$Latest"), ("max_hours", True),
            ("max_hours", float("nan")), ("max_usd_per_hour", 2),
            ("output_uri", "s3://another/private"), ("instance_type", "p5.48xlarge"),
        ):
            with self.subTest(key=key, value=value), self.assertRaises((ValueError, TypeError)):
                validate_spec({**self.spec, key: value}, self.policy)

    def test_capability_scope_and_cross_owner(self):
        self.assertEqual(self.broker.identity("secret", "gpu.submit"), "session")
        for token, action in (("bad", "gpu.submit"), ("secret", "gpu.stop")):
            with self.assertRaises(PermissionError):
                self.broker.identity(token, action)
        job = self.reserve()
        with self.assertRaises(PermissionError):
            self.broker.job(job["id"], "someone-else")

    def test_no_arbitrary_fields(self):
        with self.assertRaises(ValueError):
            self.broker.dispatch("secret", "gpu.submit", {"shell": "id"})
        with self.assertRaises(ValueError):
            self.broker.dispatch("secret", "dashboard.status", {"service": "other"})

    def test_submit_disabled_and_digest_required(self):
        self.policy["gpu_enabled"] = False
        with self.assertRaisesRegex(ValueError, "disabled"):
            self.reserve()
        self.policy["gpu_enabled"] = True
        with self.assertRaisesRegex(ValueError, "digest"):
            self.broker.reserve("session", "approved", "0" * 64, None)

    def test_one_active_and_durable_budget(self):
        first = self.reserve()
        with self.assertRaisesRegex(ValueError, "one nonterminal"):
            self.reserve()
        self.broker.update(first["id"], status="stopped")
        self.policy["total_compute_usd"] = .5
        with self.assertRaisesRegex(ValueError, "budget"):
            self.reserve()

    def test_submit_pins_template_creates_backstop_and_never_starts_training(self):
        job = self.broker.submit("session", "approved", self.digest())
        self.assertEqual(job["status"], "idle")
        launch = next(args for args in self.calls if "run-instances" in args)
        self.assertEqual(launch[launch.index("--client-token") + 1], job["id"])
        self.assertTrue(any("create-schedule" in args for args in self.calls))
        self.assertFalse(any("send-command" in args for args in self.calls))
        with self.assertRaisesRegex(PermissionError, "training authorization"):
            self.broker.start("session", job["id"])

    def test_failed_backstop_reconciles_paid_resources_and_keeps_reservation(self):
        original = self.broker.run
        def run(args, **kwargs):
            if "create-schedule" in args:
                raise subprocess.CalledProcessError(1, args, stderr="denied")
            return original(args, **kwargs)
        self.broker.run = run
        with self.assertRaises(subprocess.CalledProcessError):
            self.broker.submit("session", "approved", self.digest())
        self.assertTrue(any("terminate-instances" in args for args in self.calls))
        row = dict(self.broker.db.execute("SELECT * FROM jobs").fetchone())
        self.assertEqual(row["status"], "recovery-required")
        self.assertEqual(row["reserved"], .5)

    def test_ambiguous_launch_reconciles_by_durable_job_tags(self):
        original = self.broker.run
        def run(args, **kwargs):
            if "run-instances" in args:
                raise subprocess.TimeoutExpired(args, 60)
            return original(args, **kwargs)
        self.broker.run = run
        with self.assertRaises(subprocess.TimeoutExpired):
            self.broker.submit("session", "approved", self.digest())
        self.assertTrue(any("describe-instances" in args for args in self.calls))
        self.assertTrue(any("terminate-instances" in args for args in self.calls))

    def test_training_start_requires_separate_digest_grant_and_is_not_replayed(self):
        job = self.broker.submit("session", "approved", self.digest())
        self.policy["training_authorized_specs"] = [job["digest"]]
        result = self.broker.start("session", job["id"])
        self.assertEqual(result["status"], "start-requested")
        with self.assertRaises(ValueError):
            self.broker.start("session", job["id"])

    def test_checkpoint_stop_is_asynchronous_then_terminates(self):
        job = self.broker.submit("session", "approved", self.digest())
        result = self.broker.stop("session", job["id"])
        self.assertEqual(result["status"], "stopping")
        self.assertFalse(any("terminate-instances" in args for args in self.calls))
        self.broker.status("session", job["id"])
        self.assertTrue(any("terminate-instances" in args for args in self.calls))

    def test_expired_stop_terminates_even_if_ssm_unavailable(self):
        job = self.broker.submit("session", "approved", self.digest())
        self.broker.stop("session", job["id"])
        self.broker.update(job["id"], stop_deadline=0)
        self.broker.status("session", job["id"])
        self.assertTrue(any("terminate-instances" in args for args in self.calls))

    def test_resume_requires_new_verified_state_and_frozen_recipe(self):
        job = self.reserve()
        self.broker.update(job["id"], status="stopped")
        with self.assertRaisesRegex(ValueError, "new checkpoint"):
            self.broker.reserve("session", "approved", self.digest(), job["id"])
        self.spec["checkpoint"] = {**self.spec["checkpoint"], "version": "version2"}
        self.spec["recipe"] = {"batch": 128}
        with self.assertRaisesRegex(ValueError, "preserve"):
            self.broker.reserve("session", "approved", self.digest(), job["id"])

    def test_restart_is_exact_allowlisted_service(self):
        with patch("so101.control_broker.urlopen") as response:
            response.return_value.__enter__.return_value.status = 200
            result = self.broker.dashboard(True)
        self.assertTrue(result["healthy"])
        self.assertEqual(self.calls[0], ["/usr/bin/systemctl", "restart", "so101-web.service"])
        self.policy["dashboard_service"] = "ssh.service"
        with self.assertRaises(ValueError):
            self.broker.dashboard(True)

    def test_restart_waits_for_readiness_without_restarting_twice(self):
        with patch("so101.control_broker.urlopen") as response:
            ready = response.return_value
            ready.__enter__.return_value.status = 200
            response.side_effect = [URLError("connection refused"), ready]
            self.assertTrue(self.broker.dashboard(True)["healthy"])
        self.assertEqual(sum("restart" in args for args in self.calls), 1)

    def test_readiness_failure_is_bounded_and_visible(self):
        with patch("so101.control_broker.urlopen", side_effect=URLError("still down")):
            with self.assertRaisesRegex(RuntimeError, "readiness deadline"):
                wait_for_dashboard(self.policy["dashboard_health_url"], timeout=.01)

    def test_health_authorization_error_is_not_hidden_by_retries(self):
        with patch("so101.control_broker.urlopen", side_effect=HTTPError("url", 403, "denied", {}, None)) as call:
            with self.assertRaises(HTTPError):
                wait_for_dashboard(self.policy["dashboard_health_url"])
            self.assertEqual(call.call_count, 1)

    def test_http_denies_unauthenticated_and_wrong_capability_requests(self):
        server = make_server(self.broker, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            for token in ("bad", "secret"):
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/gpu.stop", data=b'{"job_id":"x"}',
                    headers={"Authorization": "Bearer " + token},
                )
                with self.assertRaises(HTTPError) as error:
                    urlopen(request)
                self.assertEqual(error.exception.code, 403)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_operator_policy_cannot_be_container_writable(self):
        path = self.root / "untrusted.json"
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "root-owned"):
            trusted_json(path)


if __name__ == "__main__":
    unittest.main()
