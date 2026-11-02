import copy
import json
import subprocess
import unittest
from unittest.mock import Mock, patch

from so101.aws_runtime import RuntimeImage


class RuntimeImageTests(unittest.TestCase):
    def setUp(self):
        self.runtime = RuntimeImage("example/worker@sha256:" + "a" * 64, "b" * 40, "c" * 64)
        self.local = {
            "RepoDigests": [self.runtime.image], "Os": "linux", "Architecture": "amd64",
            "Config": {"Labels": {
                "org.opencontainers.image.revision": self.runtime.commit,
                "io.so101.source-sha256": self.runtime.source_sha256,
            }},
        }

    def verify(self, local=None):
        run = Mock(return_value=subprocess.CompletedProcess(
            [], 0, stdout=json.dumps([self.local if local is None else local])))
        return self.runtime.verify("/usr/bin/docker", run=run), run

    def test_only_inspects_fixed_local_daemon_with_isolated_environment(self):
        with patch.dict("os.environ", {"DOCKER_HOST": "tcp://untrusted:2375",
                                       "DOCKER_CONTEXT": "untrusted", "DOCKER_CONFIG": "/tmp"}):
            identity, run = self.verify()
        self.assertEqual(identity["image"], self.runtime.image)
        self.assertEqual(run.call_args.args[0], [
            "/usr/bin/docker", "--host", "unix:///var/run/docker.sock",
            "image", "inspect", self.runtime.image,
        ])
        self.assertEqual(run.call_args.kwargs["timeout"], 30)
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertEqual(set(run.call_args.kwargs["env"]), {"PATH", "HOME"})

    def test_mutable_image_or_source_pins_rejected_before_command(self):
        for image, commit, source in (
            ("example/worker:latest", "b" * 40, "c" * 64),
            (self.runtime.image, "main", "c" * 64),
            (self.runtime.image, "b" * 40, "bad"),
        ):
            run = Mock()
            with self.subTest(image=image, commit=commit, source=source), self.assertRaises(ValueError):
                RuntimeImage(image, commit, source).verify("/usr/bin/docker", run=run)
            run.assert_not_called()

    def test_repository_digest_and_platform_must_match(self):
        for changes in (
            {"RepoDigests": []}, {"RepoDigests": ["example/other@sha256:" + "a" * 64]},
            {"Os": "windows"}, {"Architecture": "arm64"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.verify({**self.local, **changes})

    def test_labels_require_both_commit_and_source_archive_digest(self):
        for changes in (
            {"Config": None}, {"Config": {}}, {"Config": {"Labels": None}},
            {"Config": {"Labels": {"org.opencontainers.image.revision": self.runtime.commit}}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.verify({**self.local, **changes})
        local = copy.deepcopy(self.local)
        local["Config"]["Labels"]["io.so101.source-sha256"] = "d" * 64
        with self.assertRaises(ValueError):
            self.verify(local)

    def test_missing_image_errors_propagate_without_pull_or_retry(self):
        run = Mock(side_effect=subprocess.CalledProcessError(1, ["docker"], stderr="No such image"))
        with self.assertRaises(subprocess.CalledProcessError):
            self.runtime.verify("/usr/bin/docker", run=run)
        self.assertEqual(run.call_count, 1)

    def test_ambiguous_inspect_output_rejected(self):
        for output in (None, {}, [], [self.local, self.local], [None]):
            run = Mock(return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps(output)))
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.runtime.verify("/usr/bin/docker", run=run)

    def test_relative_docker_binary_rejected(self):
        run = Mock()
        with self.assertRaises(ValueError):
            self.runtime.verify("docker", run=run)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
