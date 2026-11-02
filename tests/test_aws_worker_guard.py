from datetime import datetime, timezone
import json
import subprocess
import unittest
from unittest.mock import MagicMock, Mock, patch
from urllib.error import URLError

from so101.aws_worker_guard import MAX_LEASE_SECONDS, Metadata, guard, poweroff


class WorkerGuardTests(unittest.TestCase):
    def test_metadata_uses_token_and_caps_at_launch_plus_twelve_hours(self):
        metadata = Metadata()
        launched = datetime(2026, 11, 2, tzinfo=timezone.utc).timestamp()
        metadata.read = Mock(side_effect=[
            "token", str(int(launched + 2 * MAX_LEASE_SECONDS)),
            json.dumps({"pendingTime": "2026-11-02T00:00:00Z"}),
        ])
        self.assertEqual(metadata.deadline(), launched + MAX_LEASE_SECONDS)
        self.assertEqual(metadata.read.call_args_list[1].args,
                         ("meta-data/tags/instance/so101-deadline", "token"))
        self.assertEqual(metadata.read.call_args_list[2].args,
                         ("dynamic/instance-identity/document", "token"))

    def test_invalid_or_unavailable_metadata_fails_closed(self):
        for error in (URLError("unavailable"), TimeoutError(), ValueError("invalid"),
                      KeyError("pendingTime"), TypeError("invalid identity")):
            with self.subTest(error=error):
                metadata = Mock()
                metadata.deadline.side_effect = error
                shutdown = Mock()
                guard(metadata, shutdown=shutdown)
                shutdown.assert_called_once_with()

    def test_invalid_deadline_tags_are_rejected(self):
        for raw in ("", "nan", "inf", "-1", "0", "1.5", " 123", "1" * 13):
            with self.subTest(raw=raw):
                metadata = Metadata()
                metadata.read = Mock(side_effect=["token", raw])
                with self.assertRaises(ValueError):
                    metadata.deadline()

    def test_missing_or_naive_launch_time_is_rejected(self):
        for identity in ({}, [], None, {"pendingTime": 1},
                         {"pendingTime": "2026-11-02T00:00:00"},
                         {"pendingTime": "2026-11-02T00:00:00+01:00"}):
            with self.subTest(identity=identity):
                metadata = Metadata()
                metadata.read = Mock(side_effect=["token", "1800000000", json.dumps(identity)])
                with self.assertRaises((KeyError, ValueError)):
                    metadata.deadline()

    def run_clock(self, deadline, wall_rate=1):
        clock = {"wall": 1000.0, "mono": 0.0}
        waits = []
        def wait(seconds):
            waits.append(seconds)
            clock["mono"] += seconds
            clock["wall"] += seconds * wall_rate
        metadata = Mock()
        metadata.deadline.return_value = deadline
        shutdown = Mock()
        guard(metadata, wall=lambda: clock["wall"], monotonic=lambda: clock["mono"],
              wait=wait, shutdown=shutdown)
        shutdown.assert_called_once_with()
        metadata.deadline.assert_called_once_with()
        return clock, waits

    def test_exact_expiration_and_expired_boot(self):
        clock, waits = self.run_clock(1012)
        self.assertEqual(waits, [5, 5, 2])
        self.assertEqual(clock["mono"], 12)
        self.assertEqual(self.run_clock(999)[1], [])

    def test_clock_rollback_never_extends_lease(self):
        clock, _ = self.run_clock(1012, wall_rate=-100)
        self.assertEqual(clock["mono"], 12)

    def test_clock_jump_forward_shortens_lease(self):
        clock, _ = self.run_clock(1100, wall_rate=100)
        self.assertEqual(clock["mono"], 5)

    def test_guard_caps_even_a_bad_metadata_adapter(self):
        clock, _ = self.run_clock(1000 + MAX_LEASE_SECONDS * 2)
        self.assertEqual(clock["mono"], MAX_LEASE_SECONDS)
        for value in (float("nan"), float("inf")):
            self.assertEqual(self.run_clock(value)[1], [])

    def test_shutdown_command_is_fixed_and_failure_propagates(self):
        with patch("so101.aws_worker_guard.subprocess.run") as run:
            poweroff()
            self.assertEqual(run.call_args.args[0], ["/usr/bin/systemctl", "poweroff"])
            self.assertTrue(run.call_args.kwargs["check"])
            self.assertEqual(run.call_args.kwargs["timeout"], 10)
        with self.assertRaises(subprocess.CalledProcessError):
            guard(Mock(deadline=Mock(return_value=1)), wall=lambda: 2,
                  shutdown=Mock(side_effect=subprocess.CalledProcessError(1, "poweroff")))

    def test_imds_transport_has_no_proxy_and_uses_bounded_token_request(self):
        with patch("so101.aws_worker_guard.build_opener") as build:
            build.return_value.open.return_value.__enter__.return_value.read.return_value = b"token"
            metadata = Metadata()
            self.assertEqual(metadata.read("api/token"), "token")
            self.assertEqual(build.call_args.args[0].proxies, {})
            request = build.return_value.open.call_args.args[0]
            self.assertEqual(request.full_url, "http://169.254.169.254/latest/api/token")
            self.assertEqual(request.method, "PUT")
            self.assertEqual(build.return_value.open.call_args.kwargs["timeout"], 2)

    def test_metadata_transport_rejects_oversized_response(self):
        metadata = Metadata()
        metadata.opener = MagicMock()
        metadata.opener.open.return_value.__enter__.return_value.read.return_value = b"x" * 16385
        with self.assertRaisesRegex(ValueError, "size limit"):
            metadata.read("api/token")


if __name__ == "__main__":
    unittest.main()
