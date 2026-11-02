"""Standalone, stdlib-only AWS worker hard lease backstop (not a trainer)."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
import os
import re
import subprocess
import time
from urllib.request import ProxyHandler, Request, build_opener


LOG = logging.getLogger(__name__)
IMDS = "http://169.254.169.254/latest/"
MAX_LEASE_SECONDS = 12 * 3600
POLL_SECONDS = 5


class Metadata:
    def __init__(self):
        self.opener = build_opener(ProxyHandler({}))

    def read(self, path: str, token: str | None = None) -> str:
        headers = ({"X-aws-ec2-metadata-token": token} if token is not None
                   else {"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        request = Request(IMDS + path, headers=headers,
                          method="GET" if token is not None else "PUT")
        with self.opener.open(request, timeout=2) as response:
            value = response.read(16385)
        if len(value) > 16384:
            raise ValueError("IMDS response exceeds size limit")
        return value.decode("utf-8")

    def deadline(self) -> float:
        token = self.read("api/token")
        if not token or len(token) > 4096:
            raise ValueError("Invalid IMDSv2 token")
        raw = self.read("meta-data/tags/instance/so101-deadline", token)
        if not re.fullmatch(r"[1-9][0-9]{0,11}", raw):
            raise ValueError("Invalid worker deadline tag")
        identity = json.loads(self.read("dynamic/instance-identity/document", token))
        if not isinstance(identity, dict) or not isinstance(identity.get("pendingTime"), str):
            raise ValueError("Instance identity requires a launch timestamp")
        launched = datetime.fromisoformat(identity["pendingTime"].replace("Z", "+00:00"))
        if launched.utcoffset() != timezone.utc.utcoffset(launched):
            raise ValueError("Instance launch time must be UTC")
        # Bound by launch time, not guard start time: restarting cannot renew a lease.
        return min(float(raw), launched.timestamp() + MAX_LEASE_SECONDS)


def poweroff() -> None:
    subprocess.run(
        ["/usr/bin/systemctl", "poweroff"],
        check=True, timeout=10,
        env={"PATH": "/usr/bin:/bin"},
    )


def guard(metadata=None, *, wall=time.time, monotonic=time.monotonic,
          wait=time.sleep, shutdown=poweroff) -> None:
    """Fail closed on unavailable metadata; never extend a captured lease."""
    try:
        started = monotonic()
        deadline = (metadata if metadata is not None else Metadata()).deadline()
        remaining = deadline - wall()
        if not math.isfinite(deadline) or not math.isfinite(remaining):
            raise ValueError("Worker deadline must be finite")
        end = started + min(MAX_LEASE_SECONDS, max(0, remaining))
        while True:
            remaining = min(deadline - wall(), end - monotonic())
            if remaining <= 0:
                break
            wait(min(POLL_SECONDS, remaining))
        LOG.warning("Worker hard lease expired; requesting OS poweroff")
    except (OSError, ValueError, KeyError, TypeError) as error:
        LOG.error("Worker lease unavailable or invalid; requesting OS poweroff: %s", error)
    # Do not suppress shutdown failures: systemd must restart the guard and retry.
    shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if os.geteuid() != 0:
        raise SystemExit("AWS worker guard must run as root")
    guard()
