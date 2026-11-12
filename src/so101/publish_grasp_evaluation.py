"""Publish mirrored GRASP inference results without representing them as training."""

import json
import time
from pathlib import Path


def publish_evaluation(results: Path, destination: Path, *, state: str) -> None:
    if state not in ("running", "completed", "failed"):
        raise ValueError(f"Unsupported evaluation state: {state}")
    data = json.loads(results.read_text())
    completed, expected = len(data["groups"]), data["expected_rollouts"]
    measured = sum(
        cell["total"]
        for cells in data["cells_by_recovery_partition"].values()
        for cell in cells if cell["piece_count"] is None
    )
    if data["optimizer"] is not None or expected <= 0 or not 0 <= completed <= expected or measured != completed:
        raise ValueError("Inconsistent inference-only GRASP population")
    if (data["complete"] and completed != expected) or (state == "completed" and not data["complete"]):
        raise ValueError("GRASP completion state does not match the measured population")
    now = time.time()
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("grasp_evaluation.json", {**data, "kind": "grasp_assistance_evaluation"}),
        ("remote_status.json", {"state": state, "observed_at": now, "heartbeat": now}),
    ):
        temporary = destination / (name + ".tmp")
        temporary.write_text(json.dumps(payload))
        temporary.chmod(0o644)
        temporary.replace(destination / name)
