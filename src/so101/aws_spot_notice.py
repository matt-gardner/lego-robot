"""Shared strict Spot notice/deadline parsing without worker/controller imports."""
from datetime import datetime, timezone


def notice_deadline(notice: dict, now: float, lease: float) -> int:
    if not isinstance(notice, dict) or set(notice) != {"action", "time"}:
        raise ValueError("Malformed Spot interruption notice")
    if notice["action"] not in {"terminate", "stop", "hibernate"}:
        raise ValueError("Unknown Spot interruption action")
    deadline = datetime.fromisoformat(notice["time"].replace("Z", "+00:00"))
    if deadline.utcoffset() != timezone.utc.utcoffset(deadline):
        raise ValueError("Spot interruption time must be UTC")
    return int(min(deadline.timestamp(), now + 120, lease))
