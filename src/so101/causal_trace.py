"""Bounded opt-in native rollout diagnostics, separate from policy inputs."""
import hashlib
import json
from pathlib import Path

import torch

from so101.pile_task_runtime import target_contact_state


def observation_summary(value, *, pixels=False):
    if isinstance(value, (tuple, list)):
        return [observation_summary(item, pixels=pixels) for item in value]
    tensor = value.detach().cpu().contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    result = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
              "sha256": hashlib.sha256(raw).hexdigest()}
    if not pixels:
        result["values"] = tensor.tolist()
    return result


def native_snapshot(sim, target):
    return {"qpos": sim.data.qpos.tolist(), "ctrl": sim.data.ctrl.tolist(),
            "joint_state": sim.state().tolist(), "sim_time": float(sim.data.time),
            "target_position": sim.position(target).tolist(),
            "contacts": target_contact_state(sim.native, target)}


class CausalTrace:
    def __init__(self, path: Path, metadata: dict, max_bytes: int):
        if max_bytes < 1024:
            raise ValueError("Causal trace byte limit must be at least 1024")
        self.path = path
        self.max_bytes = max_bytes
        self.written = 0
        self.truncated = False
        self.stream = path.open("xb")
        self.write("header", {**metadata, "image_storage": "hashes only, no pixels",
                              "byte_limit": max_bytes})

    def write(self, event, data):
        if self.truncated:
            return
        encoded = (json.dumps({"event": event, **data}, allow_nan=False) + "\n").encode()
        # Reserve enough room to make truncation explicit rather than silently dropping data.
        if self.written + len(encoded) + 128 > self.max_bytes:
            encoded = b'{"event":"truncated","reason":"byte_limit","complete":false}\n'
            self.truncated = True
        self.stream.write(encoded)
        self.stream.flush()
        self.written += len(encoded)

    def decision(self, inputs, controller, chunk, executed, phase):
        summaries = [observation_summary(value, pixels=index == 0)
                     for index, value in enumerate(inputs)]
        rng = torch.cuda.get_rng_state() if controller.device == "cuda" else torch.get_rng_state()
        self.write("decision", {"chunk": chunk, "executed": executed, "phase": phase,
                                "observation": summaries, "rng_backend": controller.device,
                                "rng_state_hex": rng.cpu().numpy().tobytes().hex()})

    def close(self, complete):
        if not self.stream.closed:
            try:
                self.write("end", {"complete": complete})
            finally:
                self.stream.close()
