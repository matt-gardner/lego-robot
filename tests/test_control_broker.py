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

from so101.control_broker import (
    Broker, canonical, make_server, trusted_json,
    validate_controller_profile, validate_controller_settings, validate_spec,
    verify_controller, wait_for_dashboard,
)


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
            "aws_profile": "so101-controller", "aws_config": "/etc/lego-robot-control/aws-config",
            "controller_role_arn": "arn:aws:iam::123456789012:role/so101-controller",
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
        self.recovery_objects = {}
        self.profile = patch("so101.control_broker.validate_controller_profile")
        self.profile.start()
        self.broker = Broker(self.policy, self.root / "state.sqlite", self.root, self.command)
        self.calls.clear()
        self.trust = patch("so101.control_broker.trusted_json", return_value=self.spec)
        self.trust.start()

    def tearDown(self):
        self.trust.stop()
        self.profile.stop()
        self.broker.db.close()
        self.temporary.cleanup()

    def digest(self):
        return hashlib.sha256(canonical(self.spec)).hexdigest()

    def test_reservation_is_atomic_across_two_connections(self):
        second = Broker(self.policy, self.root / "state.sqlite", self.root, self.command)
        first_read, release = threading.Event(), threading.Event()
        second_begin, second_read = threading.Event(), threading.Event()
        results = []

        def first_trace(statement):
            if statement.startswith("SELECT COALESCE(SUM"):
                first_read.set()
                if not release.wait(5):
                    raise RuntimeError("Reservation test timed out")

        def second_trace(statement):
            if statement == "BEGIN IMMEDIATE":
                second_begin.set()
            if statement.startswith("SELECT COUNT"):
                second_read.set()

        def reserve(broker):
            try:
                results.append(broker.reserve("session", "spec", self.digest(), None))
            except ValueError as error:
                results.append(error)

        self.broker.db.set_trace_callback(first_trace)
        second.db.set_trace_callback(second_trace)
        first = threading.Thread(target=reserve, args=(self.broker,))
        other = threading.Thread(target=reserve, args=(second,))
        try:
            first.start()
            self.assertTrue(first_read.wait(3))
            other.start()
            self.assertTrue(second_begin.wait(3))
            self.assertFalse(second_read.is_set())
        finally:
            release.set()
            first.join(5)
            if other.ident is not None:
                other.join(5)
            self.broker.db.set_trace_callback(None)
            second.db.close()
        self.assertFalse(first.is_alive())
        self.assertFalse(other.is_alive())
        self.assertEqual(sum(isinstance(result, dict) for result in results), 1)
        self.assertEqual(sum(isinstance(result, ValueError) for result in results), 1)
        self.assertEqual(self.broker.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 1)
        self.assertFalse(self.broker.db.in_transaction)

    def command(self, args, **kwargs):
        self.calls.append(args)
        if "get-caller-identity" in args:
            data = {"Account": "123456789012",
                    "Arn": "arn:aws:sts::123456789012:assumed-role/so101-controller/test"}
        elif "describe-regions" in args:
            data = {"Regions": [{"RegionName": "us-west-2"}]}
        elif "describe-instance-types" in args:
            data = {"InstanceTypes": [{"InstanceType": "t3.small"}]}
        elif "describe-launch-template-versions" in args:
            data = {"LaunchTemplateVersions": [{"LaunchTemplateData": {
                "ImageId": "ami-123abc", "InstanceType": "g6.2xlarge",
                "UserData": base64.b64encode(b"qualified idle bootstrap").decode(),
                "InstanceMarketOptions": {"MarketType": "spot", "SpotOptions": {"MaxPrice": "0.5"}},
                "MetadataOptions": {"HttpTokens": "required", "InstanceMetadataTags": "enabled"},
                "InstanceInitiatedShutdownBehavior": "terminate",
                "BlockDeviceMappings": [{"Ebs": {
                    "Encrypted": True, "DeleteOnTermination": True, "VolumeSize": 80,
                }}],
            }}]}
        elif "run-instances" in args:
            data = {"Instances": [{"InstanceId": "i-123abc"}]}
        elif "describe-instances" in args:
            if "Name=instance-state-name,Values=pending,running,stopping,shutting-down" in args:
                data = {"Reservations": [{"Instances": [{"InstanceType": "t3.small"}]}]}
            else:
                data = {"Reservations": [{"Instances": [{
                    "InstanceId": "i-123abc", "State": {"Name": "running"},
                }]}]}
        elif "list-objects-v2" in args:
            prefix = args[args.index("--prefix") + 1]
            job_id = prefix.rstrip("/").split("/")[-1]
            job = self.broker.job(job_id, "session")
            checkpoint = b"new full state"
            checksum = hashlib.sha256(checkpoint).hexdigest()
            checkpoint_key = prefix + "step_000000304/" + checksum + ".pt"
            body = canonical({
                "schema": 1, "job_id": job_id, "spec_sha256": job["digest"], "step": 304,
                "checkpoint": {"uri": "s3://approved/" + checkpoint_key, "version": "checkpoint-v1",
                               "sha256": checksum, "size": len(checkpoint)},
            })
            manifest_key = prefix + "step_000000304/" + hashlib.sha256(body).hexdigest() + ".json"
            self.recovery_objects = {
                manifest_key: (body, "manifest-v1"), checkpoint_key: (checkpoint, "checkpoint-v1"),
            }
            data = {"Contents": [{"Key": key} for key in self.recovery_objects], "IsTruncated": False}
        elif "head-object" in args or "get-object" in args:
            key = next(item.removeprefix("--key=") for item in args if item.startswith("--key="))
            body, version = self.recovery_objects[key]
            if "get-object" in args:
                Path(next(item for item in args if item.endswith(".part"))).write_bytes(body)
            data = {"VersionId": version, "ContentLength": len(body),
                    "ChecksumSHA256": base64.b64encode(hashlib.sha256(body).digest()).decode()}
        elif "send-command" in args:
            data = {"Command": {"CommandId": "command-1"}}
        elif "get-command-invocation" in args:
            data = {"Status": "Success"}
        else:
            data = {}
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(data))

    def reserve(self):
        return self.broker.reserve("session", "approved", self.digest(), None)

    def test_account_gpu_inventory_blocks_submit_before_reserving_or_launching(self):
        def command(args, **kwargs):
            if "describe-instance-types" in args:
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps({
                    "InstanceTypes": [{"InstanceType": "t3.small", "GpuInfo": {}}],
                }))
            return self.command(args, **kwargs)
        self.broker.run = command
        with self.assertRaisesRegex(ValueError, "Existing GPU"):
            self.broker.submit("session", "approved", self.digest())
        self.assertEqual(self.broker.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0)
        self.assertFalse(any("run-instances" in call for call in self.calls))

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

    def test_controller_profile_is_explicit_and_credentials_are_not_inherited(self):
        captured = {}
        def run(args, **kwargs):
            captured.update(kwargs)
            return self.command(args, **kwargs)
        with patch.dict("os.environ", {
            "AWS_ACCESS_KEY_ID": "unrelated-host-key",
            "AWS_ENDPOINT_URL": "https://untrusted.invalid",
            "AWS_PROFILE": "administrator",
        }):
            identity = verify_controller(self.policy, run)
        self.assertIn("assumed-role/so101-controller/", identity["Arn"])
        args = self.calls[-1]
        self.assertEqual(args[args.index("--profile") + 1], "so101-controller")
        env = captured["env"]
        self.assertNotIn("AWS_ACCESS_KEY_ID", env)
        self.assertNotIn("AWS_ENDPOINT_URL", env)
        self.assertNotIn("AWS_PROFILE", env)
        self.assertEqual(env["AWS_CONFIG_FILE"], self.policy["aws_config"])
        self.assertEqual(env["HOME"], "/var/lib/lego-robot-control")
        self.assertEqual(env["AWS_SHARED_CREDENTIALS_FILE"], "/dev/null")
        self.assertEqual(env["AWS_MAX_ATTEMPTS"], "1")

    def test_controller_rejects_wrong_role_or_account_before_creating_ledger(self):
        for account, role in (("123456789012", "dockyard-host"),
                              ("999999999999", "so101-controller")):
            with self.subTest(account=account, role=role):
                def run(args, **kwargs):
                    return subprocess.CompletedProcess(args, 0, stdout=json.dumps({
                        "Account": account, "Arn": f"arn:aws:sts::{account}:assumed-role/{role}/test",
                    }))
                state = self.root / "must-not-exist.sqlite"
                with self.assertRaisesRegex(PermissionError, "pinned"):
                    Broker(self.policy, state, self.root, run)
                self.assertFalse(state.exists())

    def test_controller_failure_is_not_replayed_or_replaced_with_host_credentials(self):
        calls = []
        def denied(args, **kwargs):
            calls.append(args)
            raise subprocess.CalledProcessError(1, args, stderr="AssumeRole denied")
        with self.assertRaises(subprocess.CalledProcessError):
            verify_controller(self.policy, denied)
        self.assertEqual(len(calls), 1)

    def test_controller_settings_do_not_default_to_host_chain(self):
        for key in ("aws_profile", "controller_role_arn", "aws_config", "aws_cli"):
            policy = dict(self.policy)
            policy.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_controller_settings(policy)

    def test_profile_requires_pinned_role_and_only_ec2_source(self):
        profile = (
            "[profile so101-controller]\n"
            "role_arn = arn:aws:iam::123456789012:role/so101-controller\n"
            "credential_source = Ec2InstanceMetadata\nregion = us-west-2\n"
        )
        self.profile.stop()
        with patch("so101.control_broker.trusted_text", return_value=profile):
            validate_controller_profile(self.policy)
        for altered in (
            profile.replace("so101-controller\n", "administrator\n"),
            profile.replace("credential_source = Ec2InstanceMetadata", "source_profile = default"),
            profile + "credential_process = unsafe\n",
            profile + "endpoint_url = https://untrusted.invalid\n",
            "[default]\nregion = us-west-2\n" + profile,
            "[DEFAULT]\nendpoint_url = https://untrusted.invalid\n" + profile,
        ):
            with self.subTest(config=altered), patch(
                "so101.control_broker.trusted_text", return_value=altered,
            ), self.assertRaisesRegex(ValueError, "pinned role"):
                validate_controller_profile(self.policy)
        self.profile.start()

    def test_dashboard_only_startup_does_not_require_aws(self):
        policy = {**self.policy, "gpu_enabled": False}
        for key in ("aws_profile", "controller_role_arn", "aws_config", "aws_cli"):
            policy.pop(key)
        def unexpected(*args, **kwargs):
            self.fail("Dashboard-only startup must not call AWS")
        broker = Broker(policy, self.root / "dashboard.sqlite", self.root, unexpected)
        broker.db.close()

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

    def test_unsafe_shutdown_or_invisible_deadline_rejected_before_launch(self):
        original = self.broker.run
        for key, value in (
            ("InstanceInitiatedShutdownBehavior", None),
            ("InstanceInitiatedShutdownBehavior", "stop"),
            ("MetadataOptions", {"HttpTokens": "required"}),
            ("MetadataOptions", {"HttpTokens": "required", "InstanceMetadataTags": "disabled"}),
        ):
            with self.subTest(key=key, value=value):
                def run(args, **kwargs):
                    result = original(args, **kwargs)
                    if "describe-launch-template-versions" in args:
                        data = json.loads(result.stdout)
                        data["LaunchTemplateVersions"][0]["LaunchTemplateData"][key] = value
                        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(data))
                    return result
                self.broker.run = run
                self.calls.clear()
                with self.assertRaises(ValueError):
                    self.broker.submit("session", "approved", self.digest())
                self.assertFalse(any("run-instances" in args for args in self.calls))
                self.broker.db.execute("DELETE FROM jobs")
                self.broker.db.commit()

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
        command = next(args for args in self.calls if "send-command" in args)
        self.assertEqual(json.loads(command[command.index("--cloud-watch-output-config") + 1]), {
            "CloudWatchOutputEnabled": True, "CloudWatchLogGroupName": self.spec["log_group"],
        })
        with self.assertRaises(ValueError):
            self.broker.start("session", job["id"])

    def test_checkpoint_stop_is_asynchronous_then_terminates(self):
        job = self.broker.submit("session", "approved", self.digest())
        result = self.broker.stop("session", job["id"])
        self.assertEqual(result["status"], "stopping")
        command = next(args for args in self.calls if "send-command" in args)
        self.assertIn("--cloud-watch-output-config", command)
        self.assertEqual(json.loads(command[command.index("--parameters") + 1])["StopDeadline"],
                         [str(int(result["stop_deadline"]))])
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

    def test_resume_only_accepts_the_independently_verified_published_checkpoint(self):
        previous = self.reserve()
        self.broker.update(previous["id"], status="stopped")
        recovery = self.broker.recover("session", previous["id"])
        self.spec["checkpoint"] = {key: recovery["checkpoint"][key]
                                   for key in ("uri", "version", "sha256")}
        wrong = {**self.spec["checkpoint"], "version": "unverified-version"}
        self.spec["checkpoint"] = wrong
        with self.assertRaisesRegex(ValueError, "independently verified"):
            self.broker.reserve("session", "approved", self.digest(), previous["id"])
        self.spec["checkpoint"] = {key: recovery["checkpoint"][key]
                                   for key in ("uri", "version", "sha256")}
        resumed = self.broker.reserve("session", "approved", self.digest(), previous["id"])
        self.assertEqual(resumed["parent"], previous["id"])

    def test_recovery_cannot_read_another_capability_owners_job(self):
        job = self.reserve()
        count = len(self.calls)
        with self.assertRaises(PermissionError):
            self.broker.recover("another", job["id"])
        self.assertEqual(len(self.calls), count)

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
