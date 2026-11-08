"""Evidence-backed historical reservation accounting; never increase the spending limit."""
from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import time

from so101.aws_inventory import require_no_gpu
from so101.control_broker import controller_request, trusted_json, trusted_path


def paid_bound(job: dict, observed_at: float) -> Decimal:
    start = job["deadline"] - job["max_hours"] * 3600
    if not all(math.isfinite(value) for value in (
        start, observed_at, job["max_hours"], job["max_usd_per_hour"],
    )) or observed_at < start or job["max_hours"] <= 0 or job["max_usd_per_hour"] <= 0:
        raise ValueError("Invalid historical paid interval")
    seconds = math.ceil(max(60, observed_at - start))
    return (Decimal(seconds) * Decimal(str(job["max_usd_per_hour"])) / 3600).quantize(
        Decimal(".000001"), rounding=ROUND_CEILING,
    )


def proposals(receipt: dict) -> list[dict]:
    jobs = {job["id"]: job for job in receipt["jobs"]}
    if len(jobs) != len(receipt["jobs"]):
        raise ValueError("Duplicate historical job identity")
    result = {}
    for evidence in receipt["evidence"]:
        job = jobs.get(evidence.get("job_id"))
        if evidence.get("state") == "terminated" and job is not None:
            if job["instance"] != evidence.get("instance") or job["status"] not in {"stopped", "failed"}:
                raise ValueError("Termination evidence differs from the historical job")
            bound = paid_bound(job, evidence["observed_at"])
            result[job["id"]] = {
                "job_id": job["id"], "before": str(job["reserved"]), "after": str(bound),
                "kind": "paid", "instance": job["instance"],
                "observed_at": evidence["observed_at"],
            }
        for failure in evidence.get("failures", []):
            job = jobs.get(failure.get("client_token"))
            if (job is not None and job["instance"] is None and job["status"] == "failed"
                    and evidence.get("instances", {}).get(job["id"]) == []
                    and failure.get("errorCode") == "Server.InsufficientInstanceCapacity"
                    and failure.get("eventID") and Decimal(str(job["reserved"])) > 0):
                result[job["id"]] = {
                    "job_id": job["id"], "before": str(job["reserved"]), "after": "0",
                    "kind": "unallocated", "event_id": failure["eventID"],
                }
    return [item for item in result.values()
            if Decimal(item["before"]) != Decimal(item["after"])]


def verify_live(item: dict, job: dict, request) -> None:
    if item["kind"] == "paid":
        response = request("ec2", "describe-instances", "--instance-ids", item["instance"])
        instances = [instance for reservation in response["Reservations"]
                     for instance in reservation["Instances"]]
        if len(instances) != 1:
            raise ValueError("Paid job requires exactly one authoritative EC2 instance")
        instance = instances[0]
        launched = datetime.fromisoformat(instance["LaunchTime"].replace("Z", "+00:00")).timestamp()
        tags = {tag["Key"]: tag["Value"] for tag in instance.get("Tags", [])}
        if (instance["InstanceId"] != item["instance"] or instance["State"]["Name"] != "terminated"
                or instance.get("ClientToken") != item["job_id"]
                or tags.get("so101-job") != item["job_id"]
                or tags.get("so101-spec") != job["digest"]
                or not job["deadline"] - job["max_hours"] * 3600 <= launched <= item["observed_at"]):
            raise ValueError("Live EC2 termination/identity does not support the paid bound")
    else:
        response = request("cloudtrail", "lookup-events", "--lookup-attributes",
                           "AttributeKey=EventId,AttributeValue=" + item["event_id"])
        events = [json.loads(event["CloudTrailEvent"]) for event in response["Events"]
                  if event["EventId"] == item["event_id"]]
        if (len(events) != 1 or events[0].get("eventName") != "RunInstances"
                or events[0].get("errorCode") != "Server.InsufficientInstanceCapacity"
                or events[0].get("requestParameters", {}).get("clientToken") != item["job_id"]):
            raise ValueError("CloudTrail does not confirm this exact unallocated client token")
        response = request("ec2", "describe-instances", "--filters",
                           "Name=client-token,Values=" + item["job_id"])
        if any(reservation["Instances"] for reservation in response["Reservations"]):
            raise ValueError("Rejected client token has an allocated instance")


def reconcile(database: Path, receipt_path: Path, receipt_sha256: str, policy_path: Path,
              *, apply: bool = False, run=subprocess.run) -> dict:
    if os.geteuid() != 0:
        raise PermissionError("Historical accounting requires the installed trusted host entrypoint")
    receipt_path = trusted_path(receipt_path)
    if hashlib.sha256(receipt_path.read_bytes()).hexdigest() != receipt_sha256:
        raise ValueError("Accounting evidence differs from the reviewed receipt pin")
    receipt = trusted_json(receipt_path)
    policy = trusted_json(policy_path)
    if receipt["total_compute_usd"] != policy["total_compute_usd"]:
        raise ValueError("Accounting cannot change the authorized spending limit")
    request = lambda *args: controller_request(policy, run, *args)
    require_no_gpu(lambda region, *args: request(*args))
    changes = proposals(receipt)
    jobs = {job["id"]: job for job in receipt["jobs"]}
    for item in changes:
        verify_live(item, jobs[item["job_id"]], request)
    database = trusted_path(database)
    with sqlite3.connect(database) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN IMMEDIATE")
        current = {row["id"]: dict(row) for row in db.execute("SELECT * FROM jobs")}
        if set(current) != set(jobs):
            raise ValueError("Historical population changed; obtain a new accounting receipt")
        for job_id, job in jobs.items():
            if any(current[job_id][key] != job[key] for key in (
                "owner", "digest", "status", "instance", "deadline", "reserved",
            )):
                raise ValueError("Historical ledger row changed; refusing a stale adjustment")
        before = sum(Decimal(str(job["reserved"])) for job in current.values())
        after = before + sum(Decimal(item["after"]) - Decimal(item["before"]) for item in changes)
        if after < 0 or after > Decimal(str(policy["total_compute_usd"])):
            raise ValueError("Historical accounting does not fit the existing spending ceiling")
        if apply:
            db.execute(
                "CREATE TABLE IF NOT EXISTS reservation_accounting "
                "(receipt_sha256 TEXT PRIMARY KEY, applied_at REAL NOT NULL, changes TEXT NOT NULL)"
            )
            db.execute("INSERT INTO reservation_accounting VALUES (?, ?, ?)",
                       (receipt_sha256, time.time(), json.dumps(changes, allow_nan=False)))
            for item in changes:
                db.execute("UPDATE jobs SET reserved=? WHERE id=?",
                           (float(Decimal(item["after"])), item["job_id"]))
        return {"applied": apply, "population": len(jobs), "before": str(before),
                "after": str(after), "limit": policy["total_compute_usd"], "changes": changes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(reconcile(args.database, args.receipt, args.receipt_sha256,
                               args.policy, apply=args.apply)))


if __name__ == "__main__":
    main()
