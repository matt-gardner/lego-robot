"""Bounded policy SFT on verified native recovery demonstrations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def current_grasp_loss(logits, targets, known):
    if logits.shape != targets.shape or logits.shape != known.shape or known.dtype != torch.bool:
        raise ValueError("Current grasp logits, labels and boolean known-mask must align")
    if not known.any():
        raise ValueError("Current grasp batch has no known labels")
    selected, labels = logits[known], targets[known]
    if (not torch.isfinite(selected).all() or not torch.isfinite(labels).all()
            or ((labels != 0) & (labels != 1)).any()):
        raise ValueError("Current grasp requires finite logits and binary known labels")
    return F.binary_cross_entropy_with_logits(selected, labels)


def load_current_grasp_labels(samples, data, path, samples_sha256):
    if not samples or any(len(sample["inputs"]) != 15 for sample in samples):
        raise ValueError("Current grasp requires nonempty recorded observation-only samples")
    manifest_bytes = (data / "results.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    labels = json.loads(path.read_bytes())
    if (labels.get("kind") != "exact-native-cpu-current-state-labels"
            or labels.get("schema_version") != 1
            or labels["samples_sha256"] != samples_sha256
            or labels["dataset_manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()
            or len(labels["samples"]) != len(samples)
            or len(labels["qualifications"]) != len(manifest["reports"])
            or sum(report["samples"] for report in manifest["reports"]) != len(samples)):
        raise ValueError("Current grasp labels do not bind this verified dataset")
    entities = samples[0]["inputs"][-1].shape[-1]
    targets = torch.zeros(len(samples), entities, 2)
    known = torch.zeros_like(targets, dtype=torch.bool)
    offset = 0
    for report_index, (report, qualification) in enumerate(zip(
        manifest["reports"], labels["qualifications"], strict=True,
    )):
        errors = (qualification["maximum_qpos_error"],
                  qualification["maximum_target_position_error"])
        if (qualification["trace_sha256"] != report["trace_sha256"]
                or qualification["source"] != report["source"]
                or not all(math.isfinite(error) and 0 <= error <= 1e-8 for error in errors)
                or qualification["qualified_states"] != report["steps"] + 1
                or qualification["recorded_steps"] != report["steps"]
                or qualification["samples"] != report["samples"]
                or qualification["model_sha256"] != qualification["scene_files_sha256"]["model.mjb"]):
            raise ValueError("Current grasp labels lack exact-replay qualification")
        scene_path = Path(report["source"]).parents[1] / "problem.json"
        scene_bytes = scene_path.read_bytes()
        if hashlib.sha256(scene_bytes).hexdigest() != qualification["scene_files_sha256"]["problem.json"]:
            raise ValueError("Current grasp task identity changed")
        task = next(task for task in json.loads(scene_bytes)["tasks"]
                    if task["task_id"] == Path(report["source"]).name)
        if task["parent_task_id"] != report["parent_task_id"]:
            raise ValueError("Current grasp task parent differs from the capture")
        target = task["target_instance_id"]
        for index in range(offset, offset + report["samples"]):
            sample, row = samples[index], labels["samples"][index]
            active = sample["inputs"][-1]
            if active.shape != (1, entities) or active.dtype != torch.bool:
                raise ValueError("Current grasp requires the recorded entity mask")
            identities = sample.get("entity_instance_ids")
            if identities is None:
                if active.tolist() != [[True] + [False] * (entities - 1)]:
                    raise ValueError("Multi-entity recovery samples must capture explicit instance IDs")
                identities = [target] + [None] * (entities - 1)
            if (row["sample_index"] != index or row["report_index"] != report_index
                    or row["input_step"] != sample["step"]
                    or sample["parent_task_id"] != report["parent_task_id"]
                    or row["parent_task_id"] != report["parent_task_id"]
                    or row["target_instance_id"] != target
                    or row["entity_instance_ids"] != identities or len(identities) != entities
                    or len({piece for piece in identities if piece is not None})
                    != sum(piece is not None for piece in identities)):
                raise ValueError("Current grasp labels have wrong input time or entity identity")
            for slot, piece in enumerate(identities):
                if bool(active[0, slot]) != (piece is not None):
                    raise ValueError("Padded entities must not receive physical labels")
                if piece is None:
                    continue
                contact = row["pieces"][piece]["bilateral_grip"]
                if type(contact) is not bool:
                    raise ValueError("Bilateral contact label must be boolean")
                targets[index, slot, 0], known[index, slot, 0] = float(contact), True
                if piece == target:
                    grasp = row["target_native_current_predicate"]
                    if type(grasp) is not bool or (grasp and not contact):
                        raise ValueError("Native current grasp requires bilateral contact")
                    targets[index, slot, 1], known[index, slot, 1] = float(grasp), True
            if not known[index, :, 1].any():
                raise ValueError("Native target is not a represented input entity")
        offset += report["samples"]
    if offset != len(samples):
        raise ValueError("Current grasp capture reports do not cover the sample population")
    return targets, known


def recovery_parameters(policy: torch.nn.Module, scope: str):
    modules = {
        "action-output": ("model.action_out_proj",),
        "action-expert": ("model.vlm_with_expert.lm_expert",
                          "model.action_in_proj", "model.action_out_proj"),
        "action-expert-grasp": (
            "model.vlm_with_expert.lm_expert", "model.action_in_proj", "model.action_out_proj",
            "model.entity_numeric_proj", "model.entity_norm", "model.history_motion_proj",
            "model.history_time_embedding", "model.current_grasp_head",
        ),
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
    auxiliary = args.grasp_labels is not None
    if (not math.isfinite(args.grasp_loss_weight)
            or (auxiliary and (args.grasp_loss_weight <= 0
                               or args.trainable_scope != "action-expert-grasp"))
            or (not auxiliary and (args.grasp_loss_weight != 0
                                   or args.trainable_scope == "action-expert-grasp"))):
        raise ValueError("Current grasp requires labels, positive loss weight and action-expert-grasp scope together")
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
    samples_sha256 = hashlib.sha256(samples_path.read_bytes()).hexdigest()
    if auxiliary:
        grasp_targets, grasp_known = load_current_grasp_labels(
            samples, args.data, args.grasp_labels, samples_sha256,
        )
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    controller = EntityPolicy(args.checkpoint, args.device, True)
    trainable_parameters(controller, "all")
    source_digest = policy_digest(controller.policy)
    if source_digest != report["policy_sha256"]:
        raise ValueError("Demonstration source policy differs from the requested initial policy")
    if auxiliary:
        torch.manual_seed(args.seed)
        controller.config = controller.policy.model.enable_current_grasp_prediction()
    before = policy_digest(controller.policy)
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
            "trainable_scope": args.trainable_scope, "action_only": not auxiliary,
            "grasp_loss_weight": args.grasp_loss_weight,
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        },
        "dataset": report,
        "samples_sha256": samples_sha256,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "initial_policy_sha256": before,
        "source_policy_sha256": source_digest,
        "evaluation": "None: these are training losses, not autonomous success measurements",
    }
    if auxiliary:
        log["current_grasp_supervision"] = {
            "path": str(args.grasp_labels),
            "sha256": hashlib.sha256(args.grasp_labels.read_bytes()).hexdigest(),
            "signals": ["bilateral_contact", "native_current_grasp_predicate"],
            "scope": "Current TRAIN inputs only; native predicate is not dwell success",
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
        action_loss = recovery_loss(outputs[0], valid)
        loss = action_loss
        if auxiliary:
            observation = (*inputs[:5], *inputs[6:])
            logits = controller.policy.model.current_grasp_logits(observation)
            grasp_loss = current_grasp_loss(
                logits, grasp_targets[selected].to(args.device), grasp_known[selected].to(args.device),
            )
            loss = loss + args.grasp_loss_weight * grasp_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite recovery action loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        row = {"step": step, "loss": float(loss.detach()),
               "action_loss": float(action_loss.detach()), "gradient_norm": float(gradient_norm),
               "elapsed_seconds": time.monotonic() - started}
        if auxiliary:
            row["current_grasp_loss"] = float(grasp_loss.detach())
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
    parser.add_argument("--trainable-scope", choices=("action-output", "action-expert", "action-expert-grasp"),
                        default="action-output")
    parser.add_argument("--grasp-labels", type=Path)
    parser.add_argument("--grasp-loss-weight", type=float, default=0.)
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
