"""Ordering by recorded generation completion, never filesystem publication."""

from datetime import datetime, timezone
import re


def generation_order(row: dict) -> tuple:
    value = row.get("generated_at")
    if value is None:
        return (True, 0, row["id"], row.get("source", ""))
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})", value
    ):
        raise ValueError(f"Invalid generated_at for {row['id']}: expected RFC3339 timestamp")
    try:
        recorded = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid generated_at for {row['id']}") from error
    elapsed = recorded.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    microseconds = (elapsed.days * 86400 + elapsed.seconds) * 1_000_000 + elapsed.microseconds
    return (False, -microseconds, row["id"], row.get("source", ""))
