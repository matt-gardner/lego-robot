"""Exercise an independently managed GPU: submit, stage, execute CUDA checks, terminate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import ssl
import time
from urllib.request import Request, urlopen

from so101.cloud_checkpoint import atomic_json


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
        with urlopen(request, context=context, timeout=660) as response:
            return json.load(response)
    args.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    saved = args.state_dir / "job.json"
    if saved.exists():
        job = json.loads(saved.read_bytes())
    else:
        intent = args.state_dir / "submit-intent.json"
        if intent.exists():
            raise RuntimeError("Submission response is uncertain; do not replay the allocation")
        specs = call("gpu.catalog", {})["specs"]
        matches = [spec for spec in specs if spec["spec_id"] == args.spec_id]
        if len(matches) != 1:
            raise ValueError("Requested specification is not in this session's installed catalog")
        atomic_json(intent, matches[0])
        job = call("gpu.submit", matches[0])
        atomic_json(saved, job)
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
                    "Pending", "InProgress", "Delayed", "WaitingForSSM", "DispatchUncertain",
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
