"""Broker-owned, immutable SSM preparation; callers supply only an owned job and phase."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import uuid

from so101.control_broker import trusted_json


WORKFLOWS = Path("/etc/lego-robot-control/workflows.json")


def catalog(broker, owner: str) -> dict:
    client = next(item for item in broker.policy["clients"] if item["id"] == owner)
    allowed = client.get("spec_sha256", [])
    workflows = trusted_json(WORKFLOWS)
    result = []
    for digest, workflow in workflows.items():
        if digest in allowed:
            result.append({"spec_id": workflow["spec_id"], "spec_sha256": digest,
                           "family": workflow.get("family")})
    return {"specs": result}


def prepare(broker, owner: str, job_id: str, phase: str) -> dict:
    if phase not in {"provision", "qualify", "approve"}:
        raise ValueError("Unknown GPU preparation phase")
    with broker.lock:
        job = broker.job(job_id, owner)
        if job["status"] in {"stopped", "terminated", "failed"} or time.time() >= job["deadline"]:
            raise ValueError("GPU preparation requires an owned, unexpired live job")
        workflow = trusted_json(WORKFLOWS)[job["digest"]]
        step = workflow["steps"][phase]
        instance = broker.owned_instance(job)
        if instance["State"]["Name"] == "pending":
            return {"phase": phase, "status": "WaitingForEC2",
                    "reason": "The owned EC2 instance is still pending"}
        if instance["State"]["Name"] != "running":
            raise ValueError("GPU worker is not yet running")
        broker.db.execute(
            "CREATE TABLE IF NOT EXISTS workflow_commands "
            "(job TEXT, phase TEXT, intent TEXT, created REAL, command TEXT, "
            "PRIMARY KEY(job,phase))"
        )
        broker.db.commit()
        row = broker.db.execute(
            "SELECT * FROM workflow_commands WHERE job=? AND phase=?", (job_id, phase),
        ).fetchone()
        parameters = deepcopy(step["parameters"])
        parameters["JobId"] = [job_id]
        if phase != "provision":
            parameters["SpecSHA256"] = [job["digest"]]
        if phase == "approve":
            import base64
            report = deepcopy(workflow["qualification_report"])
            report.update(job_id=job_id, spec_sha256=job["digest"])
            parameters["QualificationBase64"] = [
                base64.b64encode(json.dumps(report).encode()).decode(),
            ]
        if row is None:
            intent = "so101-" + uuid.uuid4().hex
            created = time.time()
            broker.db.execute(
                "INSERT INTO workflow_commands VALUES (?,?,?,?,NULL)",
                (job_id, phase, intent, created),
            )
            broker.db.commit()
            try:
                response = broker.aws(
                    "ssm", "send-command", "--document-name", step["document"],
                    "--document-version", step["version"], "--instance-ids", job["instance"],
                    "--comment", intent, "--parameters", json.dumps(parameters),
                    "--cloud-watch-output-config", json.dumps({
                        "CloudWatchOutputEnabled": True,
                        "CloudWatchLogGroupName": json.loads(job["spec"])["log_group"],
                    }),
                )
            except subprocess.CalledProcessError as error:
                if error.returncode == 254 and "(InvalidInstanceId)" in (error.stderr or ""):
                    # AWS explicitly rejected the request before command acceptance.
                    broker.db.execute(
                        "DELETE FROM workflow_commands WHERE job=? AND phase=?", (job_id, phase),
                    )
                    broker.db.commit()
                    return {"phase": phase, "status": "WaitingForSSM",
                            "reason": "AWS rejected dispatch before SSM registration"}
                raise
            command = response["Command"]["CommandId"]
            broker.db.execute(
                "UPDATE workflow_commands SET command=? WHERE job=? AND phase=?",
                (command, job_id, phase),
            )
            broker.db.commit()
        elif row["command"] is None:
            # Recover accepted commands after response loss; never resubmit an
            # ambiguous intent merely because the HTTP caller retries.
            response = broker.aws(
                "ssm", "list-commands", "--instance-id", job["instance"], "--filters",
                "key=InvokedAfter,value=" + datetime.fromtimestamp(
                    row["created"] - 60, timezone.utc).isoformat(),
            )
            if response.get("NextToken"):
                raise RuntimeError("Ambiguous SSM dispatch requires complete command history")
            matches = [
                item for item in response["Commands"]
                if item.get("Comment") == row["intent"]
                and item.get("DocumentName") == step["document"]
                and item.get("DocumentVersion") == step["version"]
                and item.get("InstanceIds") == [job["instance"]]
            ]
            if not matches:
                return {"phase": phase, "status": "DispatchUncertain",
                        "reason": "No accepted command is yet visible; dispatch was not replayed"}
            if len(matches) != 1:
                raise RuntimeError("Multiple commands match a single GPU preparation intent")
            command = matches[0]["CommandId"]
            broker.db.execute(
                "UPDATE workflow_commands SET command=? WHERE job=? AND phase=?",
                (command, job_id, phase),
            )
            broker.db.commit()
        else:
            command = row["command"]
        try:
            invocation = broker.aws(
                "ssm", "get-command-invocation", "--command-id", command,
                "--instance-id", job["instance"],
            )
        except subprocess.CalledProcessError as error:
            if error.returncode == 254 and "(InvocationDoesNotExist)" in (error.stderr or ""):
                return {"phase": phase, "status": "Pending", "command_id": command,
                        "reason": "Accepted SSM command invocation is not yet visible"}
            raise
        return {"phase": phase, "status": invocation["Status"], "command_id": command,
                "stdout": invocation.get("StandardOutputContent", ""),
                "stderr": invocation.get("StandardErrorContent", "")}
