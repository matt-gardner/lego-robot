"""Conservative local reservations, not AWS billing or retention enforcement."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3


LIMIT_MICROUSD = 5_000_000
GIB = 1024 ** 3


def estimate(*, bytes_retained: int, retention_hours: int, upload_bytes: int,
             download_bytes: int, requests: int, storage_micro_per_gib_hour: int,
             transfer_micro_per_gib: int, request_micro: int,
             fixed_micro: int) -> int:
    values = locals()
    if any(type(value) is not int or value < 0 for value in values.values()):
        raise ValueError("Cost ceilings and rates must be nonnegative integers")
    if (retention_hours == 0 or storage_micro_per_gib_hour == 0
            or transfer_micro_per_gib == 0 or request_micro == 0):
        raise ValueError("An explicit retention horizon and conservative positive rates are required")
    storage = bytes_retained * retention_hours * storage_micro_per_gib_hour
    transfer = (upload_bytes + download_bytes) * transfer_micro_per_gib
    return ((storage + GIB - 1) // GIB + (transfer + GIB - 1) // GIB
            + requests * request_micro + fixed_micro)


class Reservations:
    """Single operator-owned ledger; ambiguous operations keep all reservations."""

    def __init__(self, path: Path):
        self.db = sqlite3.connect(path, isolation_level=None)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS reservations "
            "(id TEXT PRIMARY KEY, digest TEXT NOT NULL, micro INTEGER NOT NULL)"
        )

    def close(self) -> None:
        self.db.close()

    def reserve(self, operation_id: str, contract: dict) -> int:
        if not isinstance(operation_id, str) or not re.fullmatch(r"[0-9a-f]{32}", operation_id):
            raise ValueError("A unique immutable operation ID is required")
        if not isinstance(contract, dict) or set(contract) != {
            "schema", "spec_sha256", "temporary_copy_policy_sha256", "estimate",
        }:
            raise ValueError("Reservation requires an exact bounded-copy accounting contract")
        if type(contract["schema"]) is not int or contract["schema"] != 1:
            raise ValueError("Unsupported reservation contract schema")
        for key in ("spec_sha256", "temporary_copy_policy_sha256"):
            if not isinstance(contract[key], str) or not re.fullmatch(r"[0-9a-f]{64}", contract[key]):
                raise ValueError("Reservation must bind specification and approved accounting policy")
        fields = {
            "bytes_retained", "retention_hours", "upload_bytes", "download_bytes",
            "requests", "storage_micro_per_gib_hour", "transfer_micro_per_gib",
            "request_micro", "fixed_micro",
        }
        if not isinstance(contract["estimate"], dict) or set(contract["estimate"]) != fields:
            raise ValueError("All cost exposure bounds must be declared")
        cost = estimate(**contract["estimate"])
        if cost <= 0:
            raise ValueError("An operation must reserve positive cost exposure")
        digest = hashlib.sha256(json.dumps(
            contract, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = self.db.execute(
                "SELECT digest,micro FROM reservations WHERE id=?", (operation_id,),
            ).fetchone()
            if existing:
                if existing[0] != digest:
                    raise ValueError("Operation ID already binds a different accounting contract")
                self.db.execute("COMMIT")
                return existing[1]
            total = self.db.execute("SELECT COALESCE(SUM(micro),0) FROM reservations").fetchone()[0]
            if cost > LIMIT_MICROUSD - total:
                raise ValueError("Cumulative noncompute reservation allowance exhausted")
            self.db.execute("INSERT INTO reservations VALUES (?,?,?)", (operation_id, digest, cost))
            self.db.execute("COMMIT")
            return cost
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
