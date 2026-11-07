"""Disposable CUDA/Adam workload; never loads or advances the original SFT trial."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import time


FORMAT = "so101-spot-qualification-v1"
SEED = 1061000


def header(state: dict, padding_bytes: int) -> None:
    if (state.get("format") != FORMAT or state.get("seed") != SEED
            or state.get("cuda") != "13.0" or state.get("bf16") is not True
            or type(state.get("step")) is not int or state["step"] < 1
            or state.get("padding_bytes") != padding_bytes):
        raise ValueError("Invalid disposable qualification checkpoint identity")


def model_optimizer(torch, device):
    model = torch.nn.Linear(64, 64).to(device)
    return model, torch.optim.Adam(model.parameters(), lr=1e-4)


def update(torch, model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(torch.randn(16, 64, device="cuda")).square().mean()
    if not torch.isfinite(loss):
        raise ValueError("Disposable CUDA workload produced a nonfinite loss")
    loss.backward()
    optimizer.step()


def tensors(model, optimizer):
    return [value.detach().cpu().clone() for value in model.parameters()] + [
        values[key].detach().cpu().clone() for values in optimizer.state.values()
        for key in ("step", "exp_avg", "exp_avg_sq")]


def tensor_digest(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def forecast(torch, state):
    model, optimizer = model_optimizer(torch, "cuda")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"], 0)
    update(torch, model, optimizer)
    result = tensor_digest(tensors(model, optimizer))
    # Forecasting a clone must not consume the running workload's saved RNG stream.
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"], 0)
    return result


def validate(torch, path, padding_bytes, *, cuda=False):
    state = torch.load(path, map_location="cpu", weights_only=True)
    header(state, padding_bytes)
    if not re.fullmatch(r"[0-9a-f]{64}", state.get("next_update_sha256", "")):
        raise ValueError("Disposable checkpoint lacks its pre-interruption next-update reference")
    padding = state["padding"]
    if padding.dtype != torch.uint8 or padding.numel() != padding_bytes:
        raise ValueError("Disposable checkpoint padding has the wrong byte population")
    if (state["cpu_rng"].dtype != torch.uint8 or state["cpu_rng"].numel() == 0
            or state["cuda_rng"].dtype != torch.uint8 or state["cuda_rng"].numel() == 0):
        raise ValueError("Disposable checkpoint lacks full CPU/CUDA RNG state")
    model, optimizer = model_optimizer(torch, "cpu")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if len(optimizer.state) != len(list(model.parameters())):
        raise ValueError("Disposable checkpoint lacks complete Adam state")
    for parameter, values in optimizer.state.items():
        if (not torch.isfinite(parameter).all()
                or any(key not in values for key in ("step", "exp_avg", "exp_avg_sq"))
                or values["step"].item() != state["step"]):
            raise ValueError("Disposable checkpoint Adam update identity is invalid")
        for key in ("exp_avg", "exp_avg_sq"):
            if values[key].shape != parameter.shape or not torch.isfinite(values[key]).all():
                raise ValueError("Disposable checkpoint Adam body is invalid")
    if cuda:
        if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
                or torch.version.cuda != "13.0"):
            raise ValueError("Recovered next-update check requires one actual CUDA13 GPU")
        torch.use_deterministic_algorithms(True)
        results = []
        for _ in range(2):
            restored, adam = model_optimizer(torch, "cuda")
            restored.load_state_dict(state["model"], strict=True)
            adam.load_state_dict(copy.deepcopy(state["optimizer"]))
            torch.set_rng_state(state["cpu_rng"])
            torch.cuda.set_rng_state(state["cuda_rng"], 0)
            update(torch, restored, adam)
            results.append(tensors(restored, adam))
        if (not all(torch.equal(left, right) for left, right in zip(*results, strict=True))
                or any(tensor_digest(values) != state["next_update_sha256"] for values in results)):
            raise ValueError("Recovered disposable next model/Adam update is not exact")
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"step": state["step"], "size": path.stat().st_size, "sha256": checksum,
            "padding_bytes": padding_bytes, "cuda_next_update_exact": cuda}


def run(torch, output, padding_bytes, duration):
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.version.cuda != "13.0" or not torch.cuda.is_bf16_supported()):
        raise ValueError("Disposable workload requires exactly one CUDA13/BF16 GPU")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    model, optimizer = model_optimizer(torch, "cuda")
    padding = torch.zeros(padding_bytes, dtype=torch.uint8)
    started, step = time.monotonic(), 0
    while True:
        update(torch, model, optimizer)
        step += 1
        state = {
            "format": FORMAT, "seed": SEED, "cuda": torch.version.cuda, "bf16": True,
            "step": step, "padding_bytes": padding_bytes, "padding": padding,
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(0),
        }
        state["next_update_sha256"] = forecast(torch, state)
        temporary = output / ".state.part"
        torch.save(state, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(output / "training_state.pt")
        descriptor = os.open(output, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        stopped = (output / "STOP").exists()
        print(json.dumps({"workload": FORMAT, "step": step, "checkpoint_bytes":
                          (output / "training_state.pt").stat().st_size,
                          "checkpoint_stop": stopped, "original_trial_advanced": False}),
              flush=True)
        if stopped:
            return
        if time.monotonic() - started >= duration:
            raise TimeoutError("Disposable workload expired without a qualified Spot stop")
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "check", "check-cuda"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--padding-bytes", type=int, required=True)
    parser.add_argument("--duration", type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.padding_bytes <= 4 * 1024**3 or not 1 <= args.duration <= 1800:
        raise ValueError("Disposable workload requires bounded padding and duration")
    # Torch is an optional heavy dependency present only in the pinned runtime image.
    import torch
    if args.action == "run":
        if args.output is None or not args.output.is_dir():
            raise ValueError("Disposable workload requires its new writable output")
        run(torch, args.output, args.padding_bytes, args.duration)
    else:
        if args.state is None:
            raise ValueError("Checkpoint validation requires an exact state path")
        print(json.dumps(validate(torch, args.state, args.padding_bytes,
                                  cuda=args.action == "check-cuda")))


if __name__ == "__main__":
    main()
