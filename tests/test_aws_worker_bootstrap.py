import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from so101.aws_worker_bootstrap import load_manifest, stage_idle
from so101.control_broker import canonical, trusted_path


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.body = b"checkpoint bytes; no deserialization"
        self.cli = self.root / "aws"
        self.cli.write_text("test-only executable placeholder")
        self.docker = self.root / "docker"
        self.docker.write_text("test-only executable placeholder")
        self.manifest = {
            "schema": 2, "broker_id": "lego-robot", "spec_sha256": "b" * 64,
            "account_id": "133961745003", "instance_type": "g6.2xlarge", "ami": "ami-123abc",
            "aws_cli": str(self.cli), "prefixes": ["s3://approved/runs/lego-robot/"],
            "max_bytes": 2 * len(self.body), "artifacts": {
                "training_state.pt": {
                    "uri": "s3://approved/runs/lego-robot/state.pt", "version": "version-1",
                    "sha256": hashlib.sha256(self.body).hexdigest(), "size": len(self.body),
                },
            },
        }
        self.manifest["artifacts"]["source.tar"] = copy.deepcopy(
            self.manifest["artifacts"]["training_state.pt"])
        self.manifest["runtime"] = {
            "docker_cli": str(self.docker), "image": "example/worker@sha256:" + "d" * 64,
            "commit": "e" * 40, "source_sha256": hashlib.sha256(self.body).hexdigest(),
        }
        self.manifest["spec"] = {
            "image": self.manifest["runtime"]["image"], "commit": self.manifest["runtime"]["commit"],
            "ami": self.manifest["ami"], "instance_type": self.manifest["instance_type"],
            "checkpoint": {key: value for key, value in
                           self.manifest["artifacts"]["training_state.pt"].items() if key != "size"},
        }
        self.manifest["spec_sha256"] = hashlib.sha256(canonical(self.manifest["spec"])).hexdigest()
        self.identity = {
            "accountId": "133961745003", "region": "us-west-2",
            "instanceType": "g6.2xlarge", "imageId": "ami-123abc",
        }
        self.tags = {"job": "a" * 32, "spec": self.manifest["spec_sha256"], "broker": "lego-robot"}
        self.metadata = Mock()
        self.metadata.token.return_value = "token"
        self.metadata.deadline.return_value = 1100
        self.metadata.read.side_effect = self.read_metadata
        self.calls = []
        self.trust_json = patch("so101.aws_worker_bootstrap.trusted_json",
                                side_effect=lambda path: copy.deepcopy(self.manifest))
        self.trust_path = patch("so101.aws_worker_bootstrap.trusted_path", side_effect=lambda path: path)
        self.trust_json.start()
        self.trust_path.start()
        self.addCleanup(self.trust_json.stop)
        self.addCleanup(self.trust_path.stop)

    def read_metadata(self, path, token):
        self.assertEqual(token, "token")
        if path == "dynamic/instance-identity/document":
            return json.dumps(self.identity)
        return self.tags[path.removeprefix("meta-data/tags/instance/so101-")]

    def command(self, args, **kwargs):
        self.calls.append(args)
        if args[0] == "/usr/bin/systemctl":
            return subprocess.CompletedProcess(args, 0)
        if args[0] == str(self.docker):
            runtime = self.manifest["runtime"]
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps([{
                "RepoDigests": [runtime["image"]], "Os": "linux", "Architecture": "amd64",
                "Config": {"Labels": {
                    "org.opencontainers.image.revision": runtime["commit"],
                    "io.so101.source-sha256": runtime["source_sha256"],
                }},
            }]))
        if args[2] == "get-object":
            Path(args[-1]).write_bytes(self.body)
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps({
            "VersionId": "version-1", "ContentLength": len(self.body),
        }))

    def digest(self):
        return hashlib.sha256(canonical(self.manifest)).hexdigest()

    def stage(self, **kwargs):
        return stage_idle(
            self.root / "manifest.json", self.digest(), self.root,
            metadata=self.metadata, run=self.command, wall=kwargs.get("wall", lambda: 1000),
            monotonic=kwargs.get("monotonic", lambda: 0),
        )

    def test_idle_receipt_binds_root_manifest_spec_and_staged_files(self):
        destination = self.stage()
        receipt = json.loads((destination / ".bootstrap.json").read_text())
        self.assertEqual(receipt["phase"], "staged-idle")
        self.assertEqual(receipt["schema"], 2)
        self.assertEqual(receipt["manifest_sha256"], self.digest())
        self.assertEqual(receipt["spec_sha256"], self.tags["spec"])
        self.assertEqual(receipt["job_id"], self.tags["job"])
        self.assertEqual(receipt["runtime"]["image"], self.manifest["runtime"]["image"])
        self.assertEqual(receipt["staging_receipt_sha256"],
                         hashlib.sha256((destination / ".staged.json").read_bytes()).hexdigest())
        system = [args for args in self.calls if args[0] == "/usr/bin/systemctl"]
        self.assertEqual(len(system), 2)
        self.assertTrue(all(args[1:] == [
            "is-active", "--quiet", "lego-robot-worker-guard.service",
        ] for args in system))
        self.assertFalse(any("start" in args or "poweroff" in args for args in self.calls))
        self.assertFalse((destination / "STOP").exists())

    def test_manifest_digest_mismatch_precedes_metadata_and_commands(self):
        with self.assertRaisesRegex(ValueError, "pinned canonical"):
            stage_idle(self.root / "manifest.json", "0" * 64, self.root,
                       metadata=self.metadata, run=self.command)
        self.metadata.token.assert_not_called()
        self.assertEqual(self.calls, [])

    def test_manifest_schema_is_exact_and_pins_approved_worker(self):
        for changes in (
            {"schema": True}, {"schema": 1}, {"extra": "not allowed"}, {"broker_id": "../bad"},
            {"spec_sha256": "main"}, {"instance_type": "p5.48xlarge"},
            {"ami": "latest"}, {"prefixes": []}, {"max_bytes": True}, {"artifacts": {}},
        ):
            original = self.manifest
            self.manifest = {**original, **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                load_manifest(self.root / "manifest.json", self.digest())
            self.manifest = original

    def test_region_account_instance_and_ami_must_match(self):
        for key, value in (
            ("region", "us-east-1"), ("accountId", "111111111111"),
            ("instanceType", "r6a.large"), ("imageId", "ami-fff"),
        ):
            original = self.identity[key]
            self.identity[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "instance identity"):
                self.stage()
            self.identity[key] = original
        self.assertFalse(any("s3api" in args for args in self.calls))

    def test_broker_spec_and_job_tag_mismatch_prevents_transfers(self):
        for key, value in (("job", "../escape"), ("spec", "c" * 64), ("broker", "other")):
            original = self.tags[key]
            self.tags[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "tags differ"):
                self.stage()
            self.tags[key] = original
        self.assertFalse(any("s3api" in args for args in self.calls))

    def test_inactive_guard_aborts_before_metadata(self):
        def inactive(args, **kwargs):
            raise subprocess.CalledProcessError(3, args)
        with self.assertRaises(subprocess.CalledProcessError):
            stage_idle(self.root / "manifest.json", self.digest(), self.root,
                       metadata=self.metadata, run=inactive)
        self.metadata.token.assert_not_called()

    def test_expired_lease_prevents_transfers(self):
        for deadline in (999, float("nan"), float("inf")):
            self.metadata.deadline.return_value = deadline
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                self.stage()
        self.assertFalse(any("s3api" in args for args in self.calls))

    def test_expiration_during_staging_does_not_commit_idle_binding(self):
        wall = Mock(side_effect=[1000, 1101])
        with self.assertRaisesRegex(ValueError, "expired during staging"):
            self.stage(wall=wall)
        destination = self.root / self.tags["job"]
        self.assertTrue((destination / ".staged.json").exists())
        self.assertFalse((destination / ".bootstrap.json").exists())

    def test_wall_rollback_does_not_renew_idle_staging_deadline(self):
        with self.assertRaisesRegex(ValueError, "expired during staging"):
            self.stage(wall=Mock(side_effect=[1000, 1]),
                       monotonic=Mock(side_effect=[0, 101]))
        self.assertFalse((self.root / self.tags["job"] / ".bootstrap.json").exists())

    def test_no_binding_if_guard_fails_after_download(self):
        original = self.command
        count = 0
        def command(args, **kwargs):
            nonlocal count
            if args[0] == "/usr/bin/systemctl":
                count += 1
                if count == 2:
                    raise subprocess.CalledProcessError(3, args)
            return original(args, **kwargs)
        with self.assertRaises(subprocess.CalledProcessError):
            stage_idle(self.root / "manifest.json", self.digest(), self.root,
                       metadata=self.metadata, run=command, wall=lambda: 1000)
        self.assertFalse((self.root / self.tags["job"] / ".bootstrap.json").exists())

    def test_staging_root_and_cli_require_root_trust(self):
        with patch("so101.aws_worker_bootstrap.trusted_path", side_effect=ValueError("not trusted")):
            with self.assertRaisesRegex(ValueError, "not trusted"):
                self.stage()
        self.assertEqual(self.calls, [])
        self.metadata.token.assert_not_called()
        with self.assertRaisesRegex(ValueError, "root-owned"):
            trusted_path(self.root)

    def test_runtime_mismatch_aborts_before_artifact_transfer(self):
        original = self.command
        def command(args, **kwargs):
            if args[0] == str(self.docker):
                raise subprocess.CalledProcessError(1, args, stderr="No such image")
            return original(args, **kwargs)
        with self.assertRaises(subprocess.CalledProcessError):
            stage_idle(self.root / "manifest.json", self.digest(), self.root,
                       metadata=self.metadata, run=command)
        self.assertFalse(any("s3api" in args for args in self.calls))
        self.assertFalse((self.root / self.tags["job"]).exists())

    def test_runtime_source_archive_must_match_manifest(self):
        self.manifest["runtime"]["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source.tar"):
            self.stage()
        self.assertEqual(self.calls, [])

    def test_registered_spec_contents_must_hash_to_expected_digest(self):
        self.manifest["spec"]["commit"] = "f" * 40
        with self.assertRaisesRegex(ValueError, "registered specification"):
            self.stage()
        self.assertEqual(self.calls, [])

    def test_runtime_and_checkpoint_cannot_differ_from_registered_spec(self):
        original = copy.deepcopy(self.manifest)
        for field in ("runtime", "checkpoint", "ami", "instance_type"):
            self.manifest = copy.deepcopy(original)
            if field == "runtime":
                self.manifest["runtime"]["commit"] = "f" * 40
            elif field == "checkpoint":
                self.manifest["artifacts"]["training_state.pt"]["version"] = "other-version"
            elif field == "ami":
                self.manifest["ami"] = "ami-fff"
            else:
                self.manifest["instance_type"] = "g5.2xlarge"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.stage()
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
