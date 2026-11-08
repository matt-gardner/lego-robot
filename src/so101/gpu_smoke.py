"""Exercise an independently managed GPU: submit, stage, execute CUDA checks, terminate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import ssl
import time
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from so101.cloud_checkpoint import atomic_json


class BrokerFailure(RuntimeError):
    def __init__(self, status: int, body: dict):
        super().__init__(f"GPU broker HTTP {status}: {body.get('error', 'request failed')}")
        self.body = body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", type=Path,
                        default=Path.home() / ".config/lego-robot/control.json")
    parser.add_argument("--spec-id", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    connection = json.loads(args.connection.read_bytes())
    context = ssl.create_default_context(cafile=connection["ca_file"])
    token = Path(connection["token_file"]).read_text().strip()
    def call(action: str, body: dict) -> dict:
        request = Request(connection["url"].rstrip("/") + "/" + action,
                          data=json.dumps(body).encode(),
                          headers={"Authorization": "Bearer " + token,
                                   "Content-Type": "application/json"})
        try:
            with urlopen(request, context=context, timeout=660) as response:
                return json.load(response)
        except HTTPError as error:
            body = json.load(error)
            raise BrokerFailure(error.code, body) from error
    args.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    saved = args.state_dir / "job.json"
    if saved.exists():
        job = json.loads(saved.read_bytes())
    else:
        intent = args.state_dir / "submit-intent.json"
        existing = None
        if intent.exists():
            existing = json.loads(intent.read_bytes())
            if "request_id" not in existing:
                raise RuntimeError("Legacy submission intent has no idempotency key; do not replay allocation")
        specs = call("gpu.catalog", {})["specs"]
        matches = [spec for spec in specs if spec["spec_id"] == args.spec_id]
        if len(matches) != 1:
            raise ValueError("Requested specification is not in this session's installed catalog")
        primary = matches[0]
        candidates = [primary] + [
            item for item in specs if item["spec_id"] != primary["spec_id"]
            and primary.get("family") is not None and item.get("family") == primary["family"]
        ]
        if existing is not None:
            indexes = [index for index, item in enumerate(candidates)
                       if item["spec_id"] == existing["spec_id"]]
            if len(indexes) != 1:
                raise ValueError("Recorded allocation no longer belongs to the installed workflow family")
            candidates = candidates[indexes[0]:]
        for candidate in candidates:
            body = existing or {
                "spec_id": candidate["spec_id"], "spec_sha256": candidate["spec_sha256"],
                "request_id": str(uuid.uuid4()),
            }
            request_id = body["request_id"]
            existing = None
            atomic_json(intent, body)
            try:
                job = call("gpu.submit", body)
            except BrokerFailure as error:
                job = error.body.get("job")
                atomic_json(args.state_dir / ("allocation-failure-" + request_id + ".json"), error.body)
                if (job is not None and job["status"] == "failed" and not job["instance"]
                        and error.body.get("aws_error_code") == "InsufficientInstanceCapacity"):
                    print(json.dumps({"spec_id": candidate["spec_id"],
                                      "status": "AWSInsufficientCapacity", "job_id": job["id"]}), flush=True)
                    continue
                if job is not None:
                    if job["status"] == "failed":
                        if not job["instance"] and job.get("error") == "InsufficientInstanceCapacity":
                            continue
                        raise RuntimeError("Recorded GPU allocation failed: " + str(job.get("error")))
                    atomic_json(saved, job)
                    call("gpu.stop", {"job_id": job["id"]})
                raise
            atomic_json(saved, job)
            break
        else:
            raise RuntimeError("AWS reports insufficient capacity in every installed equivalent GPU zone")
    try:
        for phase in ("provision", "qualify"):
            while time.time() < min(job["deadline"] - 180, job["deadline"]):
                receipt = call("gpu.prepare", {"job_id": job["id"], "phase": phase})
                atomic_json(args.state_dir / (phase + ".json"), receipt)
                print(json.dumps({"job_id": job["id"], "phase": phase,
                                  "status": receipt["status"]}), flush=True)
                if receipt["status"] == "Success":
                    if phase == "qualify":
                        qualification = json.loads(receipt["stdout"])
                        if (qualification.get("status") != "RUNTIME_PASS"
                                or qualification.get("job_id") != job["id"]
                                or qualification.get("spec_sha256") != job["digest"]):
                            raise ValueError("GPU qualification receipt does not match this job")
                    break
                if receipt["status"] not in (
                    "Pending", "InProgress", "Delayed", "WaitingForEC2", "WaitingForSSM", "DispatchUncertain",
                ):
                    raise RuntimeError(json.dumps(receipt))
                time.sleep(10)
            else:
                raise TimeoutError("GPU preparation exceeded its bounded lease")
    finally:
        call("gpu.stop", {"job_id": job["id"]})
        while time.time() < job["deadline"] + 300:
            stopped = call("gpu.status", {"job_id": job["id"]})
            atomic_json(args.state_dir / "termination.json", stopped)
            if stopped["status"] == "stopped":
                print(json.dumps({"job_id": job["id"], "status": "stopped"}), flush=True)
                break
            time.sleep(10)
        else:
            raise TimeoutError("GPU termination was not confirmed")


if __name__ == "__main__":
    main()
