from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from so101.aws_artifacts import Artifact, ArtifactStore


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.job = "a" * 32
        self.body = b"opaque checkpoint; never deserialize during staging"
        self.artifact = Artifact(
            "s3://approved/runs/lego-robot/checkpoint.pt", "version-1",
            hashlib.sha256(self.body).hexdigest(), len(self.body),
        )
        self.calls = []
        self.store = ArtifactStore(
            owner="133961745003",
            prefixes=("s3://approved/inputs/lego-robot/", "s3://approved/runs/lego-robot/"),
            aws_cli="/opt/aws/bin/aws", run=self.command,
        )

    def command(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[2] == "get-object":
            Path(args[-1]).write_bytes(self.body)
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps({
            "VersionId": self.artifact.version, "ContentLength": len(self.body),
        }))

    def stage(self, **kwargs):
        return self.store.stage(
            self.root, self.job, {"training_state.pt": self.artifact},
            max_bytes=kwargs.get("max_bytes", len(self.body)),
        )

    def test_exact_version_owner_checksum_and_commit_receipt(self):
        destination = self.stage()
        self.assertEqual((destination / "training_state.pt").read_bytes(), self.body)
        receipt = json.loads((destination / ".staged.json").read_text())
        self.assertEqual(receipt["job_id"], self.job)
        self.assertEqual(receipt["artifacts"]["training_state.pt"]["version"], "version-1")
        self.assertEqual(destination.stat().st_mode & 0o777, 0o700)
        self.assertEqual([args[2] for args, _ in self.calls], ["head-object", "get-object"])
        for args, kwargs in self.calls:
            self.assertEqual(args[args.index("--expected-bucket-owner") + 1], "133961745003")
            self.assertIn("--version-id=version-1", args)
            self.assertEqual(args[args.index("--region") + 1], "us-west-2")
            self.assertTrue(kwargs["check"])
            self.assertEqual(kwargs["timeout"], 3600)

    def test_worker_credentials_do_not_inherit_keys_profiles_or_endpoints(self):
        with patch.dict("os.environ", {
            "AWS_ACCESS_KEY_ID": "unrelated", "AWS_PROFILE": "controller",
            "AWS_ENDPOINT_URL": "https://untrusted.invalid",
        }):
            self.stage()
        env = self.calls[0][1]["env"]
        self.assertNotIn("AWS_ACCESS_KEY_ID", env)
        self.assertNotIn("AWS_PROFILE", env)
        self.assertNotIn("AWS_ENDPOINT_URL", env)
        self.assertEqual(env["AWS_SHARED_CREDENTIALS_FILE"], "/dev/null")
        self.assertEqual(env["AWS_CONFIG_FILE"], "/dev/null")
        self.assertEqual(env["AWS_EC2_METADATA_DISABLED"], "false")

    def test_existing_attempt_and_stop_are_never_modified(self):
        destination = self.root / self.job
        destination.mkdir()
        (destination / "STOP").write_text("original stop")
        (destination / "training_state.pt").write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            self.stage()
        self.assertEqual(self.calls, [])
        self.assertEqual((destination / "STOP").read_text(), "original stop")
        self.assertEqual((destination / "training_state.pt").read_bytes(), b"original")

    def test_staging_budget_rejects_before_network_or_filesystem_writes(self):
        for budget in (True, 0, -1, 1.5, len(self.body) - 1):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.stage(max_bytes=budget)
        self.assertEqual(self.calls, [])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_references_never_stage(self):
        invalid = (
            {"version": ""}, {"version": "null"}, {"version": "version with spaces"},
            {"sha256": "bad"}, {"size": True}, {"size": 0},
            {"uri": "s3://approved/runs/other/file"},
            {"uri": "s3://approved/runs/lego-robot2/file"},
            {"uri": "s3://approved/runs/lego-robot/../file"},
            {"uri": "s3://approved/runs/lego-robot//file"},
            {"uri": "s3://approved/runs/lego-robot/file?versionId=x"},
            {"uri": "s3://approved/runs/lego-robot/fi\nle"},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.store.stage(self.root, self.job, {"file": replace(self.artifact, **changes)},
                                 max_bytes=1000)
        self.assertEqual(self.calls, [])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_unsafe_local_names_and_job_ids_rejected(self):
        for name in ("../escape", "/absolute", "sub/file", ".staged.json", ".", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.store.stage(self.root, self.job, {name: self.artifact}, max_bytes=1000)
        for job in ("../escape", "not-a-job", "", "a" * 33):
            with self.subTest(job=job), self.assertRaises(ValueError):
                self.store.stage(self.root, job, {"file": self.artifact}, max_bytes=1000)
        self.assertEqual(self.calls, [])

    def test_wrong_head_version_or_size_prevents_download(self):
        for response in (
            {"VersionId": "wrong", "ContentLength": len(self.body)},
            {"VersionId": "version-1", "ContentLength": len(self.body) + 1},
            {"VersionId": "version-1", "ContentLength": True}, {},
        ):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as root:
                calls = []
                def run(args, **kwargs):
                    calls.append(args)
                    return subprocess.CompletedProcess(args, 0, stdout=json.dumps(response))
                self.store.run = run
                with self.assertRaises(ValueError):
                    self.store.stage(Path(root), self.job, {"file": self.artifact}, max_bytes=1000)
                self.assertEqual([args[2] for args in calls], ["head-object"])
                self.assertFalse((Path(root) / self.job / ".staged.json").exists())

    def test_corrupted_download_has_no_committed_receipt(self):
        self.artifact = replace(self.artifact, sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.stage()
        destination = self.root / self.job
        self.assertFalse((destination / ".staged.json").exists())
        self.assertFalse((destination / "training_state.pt").exists())

    def test_cli_failure_is_visible_not_retried_and_has_no_receipt(self):
        calls = []
        def denied(args, **kwargs):
            calls.append(args)
            raise subprocess.CalledProcessError(1, args, stderr="access denied")
        self.store.run = denied
        with self.assertRaises(subprocess.CalledProcessError):
            self.stage()
        self.assertEqual(len(calls), 1)
        self.assertFalse((self.root / self.job / ".staged.json").exists())

    def test_get_response_version_must_match_too(self):
        original = self.store.run
        def run(args, **kwargs):
            result = original(args, **kwargs)
            if args[2] == "get-object":
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps({
                    "VersionId": "unexpected", "ContentLength": len(self.body),
                }))
            return result
        self.store.run = run
        with self.assertRaisesRegex(ValueError, "version/size"):
            self.stage()
        self.assertFalse((self.root / self.job / ".staged.json").exists())

    def test_partial_multi_object_attempt_is_not_committed(self):
        original = self.store.run
        def run(args, **kwargs):
            if "--key=runs/lego-robot/missing" in args:
                raise subprocess.CalledProcessError(1, args, stderr="NoSuchVersion")
            return original(args, **kwargs)
        self.store.run = run
        with self.assertRaises(subprocess.CalledProcessError):
            self.store.stage(self.root, self.job, {
                "first.pt": self.artifact,
                "second.pt": replace(self.artifact, uri="s3://approved/runs/lego-robot/missing"),
            }, max_bytes=2 * len(self.body))
        destination = self.root / self.job
        self.assertTrue((destination / "first.pt").exists())
        self.assertFalse((destination / ".staged.json").exists())
        with self.assertRaises(FileExistsError):
            self.stage(max_bytes=1000)

    def test_downloads_overlap_and_all_verify_before_receipt(self):
        barrier = threading.Barrier(2)
        original = self.store.run
        def run(args, **kwargs):
            if args[2] == "get-object":
                barrier.wait(timeout=5)
            return original(args, **kwargs)
        self.store.run = run
        destination = self.store.stage(self.root, self.job, {
            "dataset.sqlite": self.artifact, "training_state.pt": self.artifact,
        }, max_bytes=2 * len(self.body))
        self.assertEqual((destination / "dataset.sqlite").read_bytes(), self.body)
        self.assertEqual((destination / "training_state.pt").read_bytes(), self.body)
        receipt = json.loads((destination / ".staged.json").read_text())
        self.assertEqual(set(receipt["artifacts"]), {"dataset.sqlite", "training_state.pt"})

    def test_parallel_corruption_does_not_commit_population(self):
        destination = self.root / self.job
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.store.stage(self.root, self.job, {
                "valid.pt": self.artifact,
                "corrupt.pt": replace(self.artifact, sha256="0" * 64),
            }, max_bytes=2 * len(self.body))
        self.assertFalse((destination / ".staged.json").exists())
        self.assertFalse((destination / "corrupt.pt").exists())

    def test_empty_manifest_is_not_success(self):
        with self.assertRaises(ValueError):
            self.store.stage(self.root, self.job, {}, max_bytes=1000)
        self.assertEqual(self.calls, [])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_bad_store_configuration_rejected(self):
        for changes in (
            {"owner": "wrong"}, {"aws_cli": "aws"}, {"prefixes": ()},
            {"prefixes": ("s3://approved/",)},
            {"prefixes": ("s3://approved/runs/lego-robot",)},
        ):
            config = {
                "owner": "133961745003", "aws_cli": "/opt/aws/bin/aws",
                "prefixes": ("s3://approved/runs/lego-robot/",),
            }
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ArtifactStore(**{**config, **changes})


if __name__ == "__main__":
    unittest.main()
