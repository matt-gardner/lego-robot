import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

source = Path(__file__).with_name("lego_dashboard_deploy.py")
if not source.exists():
    source = Path(__file__).resolve().parents[1] / "scripts" / "lego_dashboard_deploy.py"
spec = importlib.util.spec_from_file_location("dashboard_deploy", source)
assert spec is not None and spec.loader is not None
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class DeploymentTests(unittest.TestCase):
    def test_live_timeout_names_endpoint_and_elapsed_without_changing_budget(self):
        url = "http://127.0.0.1:8765/lego-robot/api/data/all?limit=1"
        opener = MagicMock()
        opener.open.side_effect = TimeoutError("timed out")
        with patch.object(deploy, "build_opener", return_value=opener), \
                patch.object(deploy.time, "monotonic", side_effect=[10, 14]):
            with self.assertRaisesRegex(RuntimeError, r"api/data/all\?limit=1 after 4\.000s: timed out"):
                deploy.live_bytes(url)
        opener.open.assert_called_once_with(url, timeout=4)

    def status_responses(self) -> list[dict]:
        release = {"id": "a" * 64, "kind": "parts", "version": deploy.EVIDENCE_VERSION,
                   "catalog_sha256": "b" * 64, "artifact_sha256": "b" * 64,
                   "validated_parts": 685, "qualification_notices": [],
                   "publication_mtime_ns": 1}
        return [{"releases": [release]}] + [
            {"release": release, "total": count, "limit": 1}
            for count in (685, 70, 24230, 0)
        ]

    def invoke_status(self, responses):
        with patch.object(deploy, "run", side_effect=[b"active", b"123"]):
            with patch.object(Path, "read_bytes", return_value=b"PYTHONPATH=/opt/lego-dashboard/releases/test/src"):
                with patch.object(Path, "read_text", return_value=json.dumps({"sha256": {}})):
                    with patch.object(deploy, "trusted"):
                        with patch.object(deploy, "catalog_api", side_effect=responses) as api, \
                                patch.object(deploy, "data_evidence", return_value={}):
                            result = deploy.status()
        return result, api

    def test_data_evidence_checks_live_features_task_filter_and_asset_bytes(self):
        html = (b'aria-label="Dashboard navigation" id="object_count" id="expert_outcome" '
                b'id="expert_acceptance" <dialog id="detail"><select id="data-run"></dialog>')
        listing = {"total": 2, "limit": 1, "unavailable_source_count": 1,
                   "items": [{"id": "example", "task": "grasp"}]}
        publication = {"sha256": {f"src/so101/dashboard/{name}": hashlib.sha256(name.encode()).hexdigest()
                                  for name in ("data.js", "data.css")}}
        replies = [json.dumps(listing).encode(), json.dumps({**listing, "total": 1}).encode(),
                   html, b"data.js", b"data.css", b'{"runs": []}']
        with patch.object(deploy, "live_bytes", side_effect=replies):
            result = deploy.data_evidence(publication)
        self.assertEqual(result["listings"]["all"]["total"], 2)
        self.assertEqual(result["listings"]["grasp"]["total"], 1)
        self.assertTrue(result["page"]["training_context_in_details"])
        for bad in (html + b'id="usage"', html.replace(b'<dialog id="detail">', b"")):
            with patch.object(deploy, "live_bytes", side_effect=[replies[0], replies[1], bad]):
                with self.assertRaises(ValueError):
                    deploy.data_evidence(publication)
        with patch.object(deploy, "live_bytes", side_effect=replies[:3] + [b"stale javascript"]):
            with self.assertRaisesRegex(ValueError, "differs from deployed"):
                deploy.data_evidence(publication)

    def test_status_returns_live_filtered_counts_and_hashes_without_restart(self):
        result, api = self.invoke_status(self.status_responses())
        evidence = result["catalog_evidence"]
        self.assertEqual(evidence["stored_status_counts"],
                         {"verified": 685, "unavailable": 70, "imported": 24230, "prepared": 0})
        self.assertEqual(evidence["verification_attempts"], 755)
        self.assertEqual(evidence["total_parts"], 24985)
        self.assertEqual(evidence["release"]["catalog_sha256"], "b" * 64)
        self.assertEqual(api.call_args_list[2].args[0],
                         {"release": "a" * 64, "status": "unavailable", "limit": "1"})

    def test_status_rejects_changed_publication_or_invalid_counts(self):
        for change in ({"release": {"id": "changed"}}, {"total": -1}, {"total": True}):
            responses = self.status_responses()
            responses[1] = {**responses[1], **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.invoke_status(responses)
        with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
            self.invoke_status([{"releases": [{**self.status_responses()[0]["releases"][0],
                                             "version": "other"}]}])

    def test_status_bounds_history_to_native_receipt_and_keeps_required_v17(self):
        responses = self.status_responses()
        original = responses[0]["releases"][0]
        responses[0]["releases"] += [
            {**original, "id": f"{index:064x}", "version": f"other-{index}",
             "publication_mtime_ns": index}
            for index in range(2, 40)
        ]
        result, _ = self.invoke_status(responses)
        self.assertEqual(result["parts_release_count"], 39)
        self.assertEqual(len(result["releases"]), 9)
        self.assertEqual(result["omitted_parts_releases"], 30)
        self.assertIn(original, result["releases"])
        self.assertLessEqual(len(json.dumps(result).encode()) + 1, 16384)
        responses = self.status_responses()
        responses[0]["releases"][0]["qualification_notices"] = ["x" * 16384]
        with self.assertRaisesRegex(ValueError, "native 16 KiB"):
            self.invoke_status(responses)

    def test_catalog_api_rejects_oversized_error_and_redirect_responses(self):
        for body in (b"x" * (deploy.API_LIMIT + 1), b'{"error":"broken"}', b"[]"):
            response = MagicMock()
            response.status = 200
            response.read.return_value = body
            opener = MagicMock()
            opener.open.return_value.__enter__.return_value = response
            with patch.object(deploy, "build_opener", return_value=opener):
                with self.assertRaises(ValueError):
                    deploy.catalog_api()
            response.read.assert_called_once_with(deploy.API_LIMIT + 1)
        with self.assertRaisesRegex(ValueError, "redirect refused"):
            deploy.NoRedirect().redirect_request(None, None, 302, None, None,
                                                 "https://elsewhere.invalid")

    def test_status_cli_failure_is_nonzero_without_success_stdout(self):
        with patch.object(deploy.sys, "argv", ["helper", "--status"]):
            with patch.object(deploy.os, "geteuid", return_value=0):
                with patch.object(deploy, "status", side_effect=RuntimeError("API unavailable")):
                    with patch("builtins.print") as output:
                        self.assertEqual(deploy.main(), 1)
        self.assertEqual(output.call_args.kwargs, {"file": deploy.sys.stderr})
        self.assertIn("API unavailable", output.call_args.args[0])

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
