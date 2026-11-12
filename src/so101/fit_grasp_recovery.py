"""Bounded policy SFT on verified native recovery demonstrations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from so101.calibrate_trajectory_rl import policy_digest
from so101.cloud_checkpoint import atomic_json
from so101.control_model.modeling_entity_smolvla import save_entity_checkpoint
from so101.evaluate_entity_sim import EntityPolicy
from so101.train_entity_sim import save_training_state
from so101.train_trajectory_rl import trainable_parameters


def _batch(values, device):
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values).to(device)
    if isinstance(first, (tuple, list)):
        return type(first)(_batch(list(items), device) for items in zip(*values, strict=True))
    raise TypeError(f"Unsupported recorded policy input: {type(first)}")


def recovery_batch(samples, action_dim, device):
    if not samples or len(samples) % 2:
        raise ValueError("Recovery microbatch must be positive and even")
    if action_dim < 6 or any(len(sample["inputs"]) != 15 for sample in samples):
        raise ValueError("Expected the maintained 15-input policy contract and six robot actions")
    actions = torch.cat([sample["actions"] for sample in samples]).to(device)
    if actions.ndim != 3 or actions.shape[-1] != 6 or not torch.isfinite(actions).all():
        raise ValueError("Expected finite normalized six-dimensional future actions")
    valid = torch.tensor([sample["valid_steps"] for sample in samples], device=device)
    if (valid < 1).any() or (valid > actions.shape[1]).any():
        raise ValueError("Invalid recorded future-action length")
    inputs = _batch([sample["inputs"] for sample in samples], device)
    actions = F.pad(actions, (0, action_dim - 6))
    return (*inputs[:5], actions, *inputs[5:]), (
        torch.arange(actions.shape[1], device=device)[None] < valid[:, None]
    )


def recovery_loss(action_losses, valid):
    return (action_losses[..., :6] * valid[..., None]).sum() / (valid.sum() * 6)


def recovery_parameters(policy: torch.nn.Module, scope: str):
    modules = {
        "action-output": ("model.action_out_proj",),
        "action-expert": ("model.vlm_with_expert.lm_expert",
                          "model.action_in_proj", "model.action_out_proj"),
    }
    if scope not in modules:
        raise ValueError(f"Unknown recovery trainable scope: {scope}")
    selected = [policy.get_submodule(name) for name in modules[scope]]
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    parameters = [parameter for module in selected for parameter in module.parameters()]
    if not parameters:
        raise ValueError("No maintained recovery parameters found")
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def fit(args):
    if args.steps < 1 or args.batch_size < 2 or args.batch_size % 2 or args.learning_rate <= 0:
        raise ValueError("Expected positive update count, positive LR, and positive even batch")
    if (args.output / "sft_training.json").exists():
        raise FileExistsError("Inspect the existing run before retrying; automatic restart is disabled")
    report = json.loads((args.data / "results.json").read_text())
    if not report["complete"] or not report["reports"] or not all(
        row["replay_success"] for row in report["reports"]
    ):
        raise ValueError("Only successful verified native replays may supply recovery supervision")
    samples_path = args.data / "samples.pt"
    samples = torch.load(samples_path, map_location="cpu", weights_only=True)
    parents = {row["parent_task_id"] for row in report["reports"]}
    if len(samples) != report["samples"] or any(
        sample["parent_task_id"] not in parents for sample in samples
    ):
        raise ValueError("Recorded training samples do not match demonstration provenance")
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    controller = EntityPolicy(args.checkpoint, args.device, True)
    trainable_parameters(controller, "all")
    before = policy_digest(controller.policy)
    if before != report["policy_sha256"]:
        raise ValueError("Demonstration source policy differs from the requested initial policy")
    parameters = recovery_parameters(controller.policy, args.trainable_scope)
    controller.policy.eval()
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    log = {
        "kind": "grasp_recovery_sft", "status": "training", "steps": [],
        "run_config": {
            "output": str(args.output), "steps": args.steps, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "seed": args.seed, "device": args.device,
            "trainable_scope": args.trainable_scope, "action_only": True,
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        },
        "dataset": report,
        "samples_sha256": hashlib.sha256(samples_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "initial_policy_sha256": before,
        "evaluation": "None: these are training losses, not autonomous success measurements",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    Path(args.output / "fit_grasp_recovery.py").write_text(Path(__file__).read_text())
    atomic_json(args.output / "sft_training.json", log)
    for step in range(1, args.steps + 1):
        selected = rng.choice(len(samples), args.batch_size, replace=len(samples) < args.batch_size)
        inputs, valid = recovery_batch(
            [samples[int(index)] for index in selected],
            controller.policy.config.max_action_dim, args.device,
        )
        optimizer.zero_grad(set_to_none=True)
        outputs = controller.policy.model.forward_entity(*inputs)
        loss = recovery_loss(outputs[0], valid)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite recovery action loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        row = {"step": step, "loss": float(loss.detach()),
               "action_loss": float(loss.detach()), "gradient_norm": float(gradient_norm),
               "elapsed_seconds": time.monotonic() - started}
        log["steps"].append(row)
        atomic_json(args.output / "sft_training.json", log)
        print(json.dumps(row), flush=True)
    after = policy_digest(controller.policy)
    if before == after:
        raise ValueError("Recovery fitting did not change the policy")
    save_entity_checkpoint(
        args.output / "checkpoint", controller.policy, controller.config,
        metadata["source_checkpoint"], metadata["normalization"],
    )
    log["final_policy_sha256"] = after
    save_training_state(args.output / "training_state.pt", controller.policy, optimizer,
                        args.steps, rng, log)
    log["status"] = "completed"
    atomic_json(args.output / "sft_training.json", log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17553001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trainable-scope", choices=("action-output", "action-expert"),
                        default="action-output")
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
