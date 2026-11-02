import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from so101.aws_checkpoint import CheckpointPublisher
from so101.aws_artifacts import Artifact, ArtifactStore


class CheckpointPublisherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "original.pt"
        self.body = b"opaque complete checkpoint bytes"
        self.source.write_bytes(self.body)
        self.objects = {}
        self.calls = []
        self.publisher = CheckpointPublisher(
            owner="133961745003", aws_cli="/opt/aws/bin/aws",
            output_prefix="s3://approved/runs/lego-robot/", run=self.command,
        )
        self.job = "a" * 32

    def command(self, args, **kwargs):
        self.calls.append(args)
        key = next(value.removeprefix("--key=") for value in args if value.startswith("--key="))
        if args[2] == "put-object":
            self.assertEqual(args[args.index("--if-none-match") + 1], "*")
            self.assertEqual(args[args.index("--server-side-encryption") + 1], "AES256")
            self.assertEqual(args[args.index("--checksum-algorithm") + 1], "SHA256")
            if key in self.objects:
                raise subprocess.CalledProcessError(1, args, stderr="PreconditionFailed")
            body = Path(args[args.index("--body") + 1]).read_bytes()
            checksum = base64.b64encode(hashlib.sha256(body).digest()).decode()
            self.assertEqual(args[args.index("--checksum-sha256") + 1], checksum)
            self.objects[key] = (body, f"version-{len(self.objects) + 1}", checksum)
        body, version, checksum = self.objects[key]
        if args[2] in {"head-object", "get-object"}:
            self.assertIn("--version-id=" + version, args)
        if args[2] == "get-object":
            Path(args[-1]).write_bytes(body)
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps({
            "VersionId": version, "ContentLength": len(body), "ChecksumSHA256": checksum,
        }))

    def publish(self, **kwargs):
        return self.publisher.publish(
            self.source, self.root, job_id=self.job, step=304,
            spec_sha256="b" * 64, max_bytes=kwargs.get("max_bytes", 1000),
        )

    def directory(self):
        return self.root / self.job / "step_000000304"

    def test_checkpoint_version_then_manifest_version_commit_without_original_changes(self):
        result = self.publish()
        self.assertEqual([args[2] for args in self.calls],
                         ["put-object", "head-object", "put-object", "head-object"])
        self.assertEqual(len(self.objects), 2)
        self.assertEqual(result.version, "version-2")
        published = json.loads((self.directory() / "published.json").read_text())
        self.assertEqual(published["manifest"]["uri"], result.uri)
        self.assertEqual(published["checkpoint"]["version"], "version-1")
        body = next(body for key, (body, _, _) in self.objects.items() if key.endswith(".json"))
        manifest = json.loads(body)
        self.assertEqual(manifest["step"], 304)
        self.assertEqual(manifest["spec_sha256"], "b" * 64)
        self.assertEqual(manifest["checkpoint"]["sha256"], hashlib.sha256(self.body).hexdigest())
        self.assertEqual(self.source.read_bytes(), self.body)
        self.assertEqual(self.directory().stat().st_mode & 0o777, 0o700)
        for args in self.calls:
            self.assertEqual(args[args.index("--expected-bucket-owner") + 1], "133961745003")

    def test_byte_ceiling_and_empty_source_fail_before_aws(self):
        for body in (b"", b"x" * 1001):
            with self.subTest(size=len(body)), tempfile.TemporaryDirectory() as root:
                self.source.write_bytes(body)
                with self.assertRaises(ValueError):
                    self.publisher.publish(self.source, Path(root), job_id=self.job, step=0,
                                           spec_sha256="b" * 64, max_bytes=1000)
        self.assertEqual(self.calls, [])

    def test_changed_source_is_rejected_before_publication(self):
        original = os.fstat
        count = 0
        def changed(descriptor):
            nonlocal count
            value = original(descriptor)
            count += 1
            return SimpleNamespace(
                st_mode=value.st_mode, st_size=value.st_size, st_ctime_ns=value.st_ctime_ns,
                st_mtime_ns=value.st_mtime_ns + (1 if count > 1 else 0),
            )
        with patch("so101.aws_checkpoint.os.fstat", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "changed in place"):
                self.publish()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.directory() / "intent.json").exists())

    def test_symlink_and_fifo_sources_are_not_read(self):
        for kind in ("symlink", "fifo"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                source = Path(root) / "unsafe"
                if kind == "symlink":
                    source.symlink_to(self.source)
                else:
                    os.mkfifo(source)
                with self.assertRaises((OSError, ValueError)):
                    self.publisher.publish(source, Path(root), job_id=self.job, step=0,
                                           spec_sha256="b" * 64, max_bytes=1000)
        self.assertEqual(self.calls, [])

    def test_ambiguous_put_retains_intent_and_is_never_replayed(self):
        def timeout(args, **kwargs):
            self.calls.append(args)
            raise subprocess.TimeoutExpired(args, 3600)
        self.publisher.run = timeout
        with self.assertRaises(subprocess.TimeoutExpired):
            self.publish()
        self.assertTrue((self.directory() / "intent.json").exists())
        self.assertFalse((self.directory() / "published.json").exists())
        with self.assertRaises(FileExistsError):
            self.publish()
        self.assertEqual(len(self.calls), 1)

    def test_manifest_failure_keeps_checkpoint_version_but_no_committed_publication(self):
        original = self.publisher.run
        def fail_manifest(args, **kwargs):
            key = next(value for value in args if value.startswith("--key="))
            if args[2] == "put-object" and key.endswith(".json"):
                raise subprocess.CalledProcessError(1, args, stderr="denied")
            return original(args, **kwargs)
        self.publisher.run = fail_manifest
        with self.assertRaises(subprocess.CalledProcessError):
            self.publish()
        self.assertEqual(len(self.objects), 1)
        self.assertFalse((self.directory() / "published.json").exists())

    def test_wrong_version_or_checksum_never_commits_manifest(self):
        original = self.publisher.run
        for changes in ({"VersionId": "null"}, {"ChecksumSHA256": "wrong"}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as root:
                self.objects.clear()
                def corrupt(args, **kwargs):
                    result = original(args, **kwargs)
                    return subprocess.CompletedProcess(args, 0,
                                                       stdout=json.dumps({**json.loads(result.stdout), **changes}))
                self.publisher.run = corrupt
                with self.assertRaises(ValueError):
                    self.publisher.publish(self.source, Path(root), job_id=self.job, step=304,
                                           spec_sha256="b" * 64, max_bytes=1000)
                self.assertEqual(len(self.objects), 1)
                self.assertFalse((Path(root) / self.job / "step_000000304" / "published.json").exists())

    def test_publication_job_directory_must_be_private_and_not_symlinked(self):
        job = self.root / self.job
        job.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.publish()
        job.unlink()
        job.mkdir(mode=0o755)
        job.chmod(0o755)
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(self.calls, [])

    def test_input_namespace_cannot_be_used_for_checkpoint_writes(self):
        with self.assertRaises(ValueError):
            CheckpointPublisher(owner="133961745003", aws_cli="/opt/aws/bin/aws",
                                output_prefix="s3://approved/inputs/lego-robot/")

    def test_published_manifest_and_checkpoint_recover_by_exact_version_and_checksum(self):
        manifest_ref = self.publish()
        reader = ArtifactStore(
            owner="133961745003", aws_cli="/opt/aws/bin/aws",
            prefixes=("s3://approved/runs/lego-robot/",), run=self.command,
        )
        manifest_directory = reader.stage(
            self.root, "c" * 32, {"manifest.json": manifest_ref}, max_bytes=manifest_ref.size,
        )
        manifest = json.loads((manifest_directory / "manifest.json").read_text())
        checkpoint_ref = Artifact(**manifest["checkpoint"])
        checkpoint_directory = reader.stage(
            self.root, "d" * 32, {"training_state.pt": checkpoint_ref}, max_bytes=checkpoint_ref.size,
        )
        self.assertEqual((checkpoint_directory / "training_state.pt").read_bytes(), self.body)
        self.assertEqual(manifest["job_id"], self.job)
        self.assertEqual(manifest["step"], 304)
        self.assertEqual(manifest["spec_sha256"], "b" * 64)

    def test_bad_exact_version_head_aborts_before_manifest_upload(self):
        original = self.publisher.run
        def corrupt_head(args, **kwargs):
            result = original(args, **kwargs)
            if args[2] == "head-object":
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps({
                    **json.loads(result.stdout), "VersionId": "unexpected",
                }))
            return result
        self.publisher.run = corrupt_head
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(len(self.objects), 1)
        self.assertFalse((self.directory() / "published.json").exists())


if __name__ == "__main__":
    unittest.main()
