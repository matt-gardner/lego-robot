"""Transactional curriculum family membership and per-category visual audits."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from so101.lego_curriculum import visual_audit_due


class CurriculumLedger:
    def __init__(self, path: Path, *, audit_seed: int):
        if type(audit_seed) is not int:
            raise ValueError("Audit seed must be an integer")
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS settings(audit_seed INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS families(
                id TEXT PRIMARY KEY, split TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS examples(
                id TEXT PRIMARY KEY, family TEXT NOT NULL REFERENCES families(id),
                category TEXT NOT NULL, ordinal INTEGER NOT NULL,
                source TEXT NOT NULL, audit_status TEXT NOT NULL,
                findings TEXT,
                UNIQUE(category,ordinal)
            );
        """)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            row = self.connection.execute("SELECT audit_seed FROM settings").fetchone()
            if row is None:
                self.connection.execute("INSERT INTO settings VALUES (?)", (audit_seed,))
            elif row[0] != audit_seed:
                raise ValueError("Audit seed changed; existing selections must be preserved")
        self.audit_seed = audit_seed

    def reserve_family(self, family: str, split: str) -> str:
        """Reserve membership before simulation; descendants reuse the same family."""
        if not isinstance(family, str) or not family:
            raise ValueError("Family identity is required")
        if split not in ("train", "development", "sealed-test"):
            raise ValueError("Unknown curriculum split")
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                "INSERT OR IGNORE INTO families VALUES (?,?)", (family, split)
            )
            existing = self.connection.execute(
                "SELECT split FROM families WHERE id=?", (family,)
            ).fetchone()[0]
            if existing != split:
                raise ValueError("Scene family already belongs to a different split")
        return existing

    def accept_pair(self, report: dict) -> list[dict]:
        """Idempotently register a qualified pair, never a single successful prefix."""
        sources = report.get("successful_episodes", [])
        attempts = report.get("attempts", [])
        if (report.get("status") != "complete" or len(sources) != 2
                or len(set(sources)) != 2 or len(attempts) != 2
                or any(row.get("outcome") != "success" for row in attempts)
                or {row.get("source") for row in attempts} != set(sources)):
            raise ValueError("Audit ledger requires a complete independently qualified pair")
        family = report["scene_family_id"]
        category = report["tags"]["category"]
        if not isinstance(category, str) or not category:
            raise ValueError("Category identity is required")
        ids = [row["example_id"] for row in attempts]
        if len(set(ids)) != 2:
            raise ValueError("Pair example identities must differ")
        results = []
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            membership = self.connection.execute(
                "SELECT split FROM families WHERE id=?", (family,)
            ).fetchone()
            if membership is None or membership[0] != report["split"]:
                raise ValueError("Scene family must be reserved before generation")
            for attempt in attempts:
                existing = self.connection.execute(
                    "SELECT family,category,source,ordinal,audit_status FROM examples WHERE id=?",
                    (attempt["example_id"],),
                ).fetchone()
                if existing is not None:
                    if existing[:3] != (family, category, attempt["source"]):
                        raise ValueError("Accepted example identity changed")
                    ordinal, status = existing[3:]
                else:
                    ordinal = self.connection.execute(
                        "SELECT COALESCE(MAX(ordinal),0)+1 FROM examples WHERE category=?",
                        (category,),
                    ).fetchone()[0]
                    status = "pending" if visual_audit_due(
                        category=category, accepted_count=ordinal, seed=self.audit_seed
                    ) else "not-selected"
                    self.connection.execute(
                        "INSERT INTO examples VALUES (?,?,?,?,?,?,NULL)",
                        (attempt["example_id"], family, category, ordinal,
                         attempt["source"], status),
                    )
                results.append({"example_id": attempt["example_id"],
                                "ordinal": ordinal, "audit_status": status})
        return results

    def pending_audits(self) -> list[dict]:
        return [
            dict(zip(("example_id", "family", "category", "source"), row))
            for row in self.connection.execute(
                "SELECT id,family,category,source FROM examples "
                "WHERE audit_status='pending' ORDER BY category,ordinal"
            )
        ]

    def record_audit(self, example_id: str, *, passed: bool, findings: str) -> None:
        if type(passed) is not bool or not isinstance(findings, str) or not findings.strip():
            raise ValueError("Visual audit needs an explicit judgment and findings")
        with self.connection:
            changed = self.connection.execute(
                "UPDATE examples SET audit_status=?,findings=? "
                "WHERE id=? AND audit_status='pending'",
                ("passed" if passed else "failed", json.dumps(findings), example_id),
            ).rowcount
            if changed != 1:
                raise ValueError("Example has no pending visual audit")

    def close(self) -> None:
        self.connection.close()
