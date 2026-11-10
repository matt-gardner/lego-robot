import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

source = Path(__file__).with_name("lego_dashboard_deploy.py")
if not source.exists():
    source = Path(__file__).resolve().parents[1] / "scripts" / "lego_dashboard_deploy.py"
spec = importlib.util.spec_from_file_location("dashboard_deploy", source)
assert spec is not None and spec.loader is not None
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class DeploymentTests(unittest.TestCase):
    def test_removed_status_argument_is_rejected_without_deployment(self):
        with patch.object(deploy.sys, "argv", ["helper", "--status"]), \
                patch.object(deploy.os, "geteuid", return_value=0), \
                patch.object(deploy, "deploy") as publish, \
                patch("builtins.print") as output:
            self.assertEqual(deploy.main(), 2)
        publish.assert_not_called()
        self.assertEqual(output.call_args.kwargs, {"file": deploy.sys.stderr})

    def test_only_published_scoped_source_overrides_frozen_baseline(self):
        with tempfile.TemporaryDirectory() as root:
            baseline = Path(root) / "baseline"
            (baseline / "so101").mkdir(parents=True)
            (baseline / "so101" / "producer.py").write_text("frozen producer")
            release = Path(root) / "release"
            release.mkdir()
            with patch.object(deploy, "git", side_effect=[b"100644 blob x file", b"published"]):
                hashes = deploy.stage_release(
                    baseline, release, Path(root), "a" * 40, ["src/so101/lego_data_browser.py"]
                )
            self.assertEqual((release / "src/so101/producer.py").read_text(), "frozen producer")
            self.assertEqual((release / "src/so101/lego_data_browser.py").read_text(), "published")
            self.assertIn("src/so101/lego_data_browser.py", hashes)

    def test_rejects_published_symlink(self):
        with tempfile.TemporaryDirectory() as root:
            baseline = Path(root) / "baseline"
            baseline.mkdir()
            destination = Path(root) / "release"
            destination.mkdir()
            with patch.object(deploy, "git", return_value=b"120000 blob x file"):
                with self.assertRaisesRegex(ValueError, "regular file"):
                    deploy.stage_release(baseline, destination, Path(root), "a" * 40,
                                         ["src/so101/lego_data_browser.py"])

    def test_frozen_baseline_hash_changes_with_content(self):
        with tempfile.TemporaryDirectory() as root:
            baseline = Path(root)
            (baseline / "dependency.py").write_text("original")
            original = deploy.tree_hash(baseline)
            (baseline / "dependency.py").write_text("changed")
            self.assertNotEqual(original, deploy.tree_hash(baseline))

    def test_failed_health_restores_previous_override_and_restarts_it(self):
        with tempfile.TemporaryDirectory() as root:
            override = Path(root) / "override.conf"
            override.write_bytes(b"previous release\n")
            with patch.object(deploy, "run", return_value=b"active") as run:
                with patch.object(deploy, "health", side_effect=RuntimeError("bad archive")):
                    with self.assertRaisesRegex(RuntimeError, "previous release restored"):
                        deploy.activate(override, Path(root) / "release", "http://127.0.0.1:8765")
            self.assertEqual(override.read_bytes(), b"previous release\n")
            self.assertEqual(sum(call.args[1] == "restart" for call in run.call_args_list), 2)


if __name__ == "__main__":
    unittest.main()
