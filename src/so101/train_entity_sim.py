from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate
from torch.utils.checkpoint import checkpoint

from lerobot.policies.common.vla_utils import pad_vector
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from so101.control_model.modeling_entity_smolvla import (
    EntityModelConfig,
    attach_entity_modules,
    load_entity_checkpoint,
    prepare_history_images,
    save_entity_checkpoint,
)
from so101.control_model.simulation_dataset import EntityTrajectoryDataset
from so101.cloud_checkpoint import atomic_json, validate_state
from so101.evaluate_entity_sim import EntityPolicy
from so101.robot_sim import BeamSimulation, MODEL_ROOT
from so101.resume_rng import DeviceRNGPlan, plan_device_rng, restore_device_rng
from so101.train_trajectory_rl import CorpusEpisode, validation_metrics


def tokenize(tokenizer, texts: list[str], length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=length,
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device).bool()


def linear_probability(
    step: int,
    start: float,
    end: float,
    decay_steps: int,
) -> float:
    if decay_steps <= 1:
        return end
    progress = min(max((step - 1) / (decay_steps - 1), 0), 1)
    return start + progress * (end - start)


def scheduled_learning_rate(step: int, steps: int, peak: float, schedule: str,
                            warmup_fraction: float = .1) -> float:
    if steps <= 0 or not 0 <= step <= steps or not math.isfinite(peak) or peak <= 0:
        raise ValueError("Learning-rate schedule requires valid steps and a finite positive peak")
    if schedule == "constant":
        return peak
    if schedule not in {"triangular", "warmup-constant"}:
        raise ValueError(f"Unknown learning-rate schedule: {schedule}")
    if steps < 2 or not math.isfinite(warmup_fraction) or not 0 < warmup_fraction < 1:
        raise ValueError("Warmup learning rate requires at least two steps and warmup fraction in (0,1)")
    peak_step = min(steps - 1, max(1, round(steps * warmup_fraction)))
    if schedule == "warmup-constant":
        return peak * min(step / peak_step, 1)
    return peak * (step / peak_step if step <= peak_step else (steps - step) / (steps - peak_step))


@contextmanager
def isolated_evaluation(policy, device):
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.random.get_rng_state()
    device_rng = (torch.mps.get_rng_state() if device == "mps"
                  else torch.cuda.get_rng_state() if device == "cuda" else None)
    was_training = policy.training
    try:
        policy.eval()
        yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.random.set_rng_state(cpu_rng)
        if device == "mps":
            torch.mps.set_rng_state(device_rng)
        elif device == "cuda":
            torch.cuda.set_rng_state(device_rng)
        policy.train(was_training)


def evaluate_without_training_rng(controller, sim, episodes, bounds, seed):
    with isolated_evaluation(controller.policy, controller.device):
        return validation_metrics(sim, controller, episodes, bounds, seed, 300, 16, .1)


def forward_supervised_batch(
    model, inputs: tuple, micro_batch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if micro_batch_size <= 0 or micro_batch_size % 2:
        raise ValueError("Supervised microbatch size must be positive and even")
    batch_size = inputs[4].shape[0]
    if batch_size <= micro_batch_size:
        return model.forward_entity(*inputs)
    outputs = []
    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        chunk = tuple(
            [tensor[start:end] for tensor in value] if isinstance(value, list)
            else value[start:end] if isinstance(value, torch.Tensor) else value
            for value in inputs
        )
        outputs.append(checkpoint(model.forward_entity, *chunk, use_reentrant=False))
    # Reduce the concatenated outputs once: padding/entity weights stay identical
    # to a full batch, and clipping/Adam still happen once per optimizer update.
    return (
        torch.cat([output[0] for output in outputs], dim=0),
        torch.cat([output[1] for output in outputs], dim=0),
        torch.cat([output[2] for output in outputs], dim=0),
        torch.cat([output[3] for output in outputs], dim=0),
        torch.cat([output[4] for output in outputs], dim=0),
    )


def resume_training_log(saved: dict, requested_config: dict, step: int,
                        allow_device_migration: bool = False) -> dict:
    previous_interval = saved["run_config"]["validation_interval"]
    previous_config = dict(saved["run_config"])
    if previous_config.pop("contrast_coefficient", 0) != 0:
        raise ValueError("Cannot resume a checkpoint with the removed contrast objective enabled")
    for name in ("supervised_validation_probe", "retain_validation_checkpoints"):
        if name in requested_config:
            previous_config.setdefault(name, False)
    comparable = {**previous_config, "validation_interval": requested_config["validation_interval"]}
    device_changed = previous_config.get("device") != requested_config.get("device")
    if device_changed and allow_device_migration:
        comparable["device"] = requested_config["device"]
    if comparable != requested_config:
        raise ValueError("Resumed SFT configuration must match its recorded schedule and split")
    if device_changed:
        saved.setdefault("configuration_changes", []).append({
            "step": step, "setting": "device", "from": previous_config["device"],
            "to": requested_config["device"],
        })
        saved["run_config"]["device"] = requested_config["device"]
    if previous_interval != requested_config["validation_interval"]:
        saved.setdefault("configuration_changes", []).append({
            "step": step, "setting": "validation_interval",
            "from": previous_interval, "to": requested_config["validation_interval"],
        })
        saved["run_config"]["validation_interval"] = requested_config["validation_interval"]
    saved["run_config"].pop("contrast_coefficient", None)
    for name in ("supervised_validation_probe", "retain_validation_checkpoints"):
        if name in requested_config:
            saved["run_config"][name] = requested_config[name]
    return saved


def save_training_state(
    path: Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    rng: np.random.Generator | None = None,
    log: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {
        "step": step,
        "model": {
            name: parameter.detach().cpu()
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        },
        "optimizer": optimizer.state_dict(),
    }
    if rng is not None:
        device = str(next(policy.parameters()).device).split(":")[0]
        state["rng"] = {
            "backend": device,
            "sampling": rng.bit_generator.state, "python": random.getstate(),
            "numpy": np.random.get_state(), "torch": torch.random.get_rng_state(),
            "device": (torch.mps.get_rng_state() if device == "mps"
                       else torch.cuda.get_rng_state() if device == "cuda" else None),
        }
    if log is not None:
        state["log"] = log
    torch.save(state, temporary)
    temporary.replace(path)


def restore_training_state(
    state: dict,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    requested_config: dict,
    device_rng_plan: DeviceRNGPlan,
) -> dict:
    log = resume_training_log(
        state["log"], requested_config, int(state["step"]), device_rng_plan.migration,
    )
    named_parameters = dict(policy.named_parameters())
    with torch.no_grad():
        for name, value in state["model"].items():
            named_parameters[name].copy_(value.to(named_parameters[name].device))
    optimizer.load_state_dict(state["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = requested_config["learning_rate"]
    rng.bit_generator.state = state["rng"]["sampling"]
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.random.set_rng_state(state["rng"]["torch"].cpu())
    restore_device_rng(device_rng_plan, state, torch)
    return log


def supervised_batch_losses(
    policy, batch, entity_config, device, depth_mode, matched_step,
    obstacle_input_probability, action_coefficient, box_coefficient,
    offscreen_center_coefficient,
):
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    batch_size = len(batch["task"])
    images = batch["images"].to(device)
    image_list, image_masks = prepare_history_images(policy, images)
    history_state = batch["history_state"].to(device)
    history_action = batch["history_action"].to(device)
    state = pad_vector(history_state[:, -1], policy.config.max_state_dim)
    actions = pad_vector(batch["actions"].to(device), policy.config.max_action_dim)
    task_tokens, task_mask = tokenize(
        tokenizer, list(batch["task"]), policy.config.tokenizer_max_length, device
    )
    role_columns = batch["roles"]
    role_texts = [
        role_columns[entity][sample]
        for sample in range(batch_size)
        for entity in range(entity_config.max_entities)
    ]
    role_tokens, role_mask = tokenize(
        tokenizer, role_texts, entity_config.role_token_length, device
    )
    role_tokens = role_tokens.reshape(batch_size, entity_config.max_entities, -1)
    role_mask = role_mask.reshape(batch_size, entity_config.max_entities, -1)
    obstacle_slots = torch.tensor([
        [role_columns[entity][sample] == "obstacle"
         for entity in range(entity_config.max_entities)]
        for sample in range(batch_size)
    ], device=device)
    obstacle_available = (
        torch.rand(batch_size, entity_config.max_entities, device=device)
        < obstacle_input_probability
    )
    input_entity_mask = (
        batch["entity_mask"].to(device) & (~obstacle_slots | obstacle_available)
    )
    input_mask = input_entity_mask.unsqueeze(-1)
    entity_box = batch["entity_box"].to(device) * input_mask
    entity_visibility = batch["entity_visibility"].to(device) * input_entity_mask
    entity_depth = batch["entity_depth_m"].to(device) * input_entity_mask
    entity_depth_valid = batch["entity_depth_valid"].to(device) * input_entity_mask
    entity_camera_position = batch["entity_camera_position"].to(device) * input_mask
    if depth_mode == "blind":
        depth_known = torch.zeros_like(entity_depth)
        depth_input = torch.zeros_like(entity_depth)
    elif depth_mode == "mixed":
        depth_known = (torch.rand_like(entity_depth) >= .5).float() * entity_depth_valid
        depth_input = entity_depth * depth_known
    else:
        depth_known = entity_depth_valid
        depth_input = entity_depth * depth_known
    if matched_step:
        half = batch_size // 2
        noise = policy.model.sample_noise(
            (half, *actions.shape[1:]), actions.device
        ).repeat_interleave(2, dim=0)
        flow_time = policy.model.sample_time(half, actions.device).repeat_interleave(2)
    else:
        noise = policy.model.sample_noise(actions.shape, actions.device)
        flow_time = policy.model.sample_time(batch_size, actions.device)
    action_losses, predicted_box, predicted_log_depth, visibility_logits, _ = (
        forward_supervised_batch(policy.model, (
            image_list, image_masks, task_tokens, task_mask, state, actions,
            history_state, history_action, role_tokens, role_mask, entity_box,
            entity_visibility, depth_input, depth_known,
            entity_camera_position * depth_known.unsqueeze(-1),
            input_entity_mask, noise, flow_time,
        ))
    )
    valid_steps = ~batch["action_is_pad"].to(device)
    action_mask = valid_steps[:, :, None]
    action_loss = (action_losses[:, :, :6] * action_mask).sum() / (
        action_mask.sum() * 6
    ).clamp_min(1)
    future_visibility = batch["future_visibility"].to(device)
    tracking_mask = valid_steps[:, :, None] & input_entity_mask[:, None, :]
    visible_mask = tracking_mask & (future_visibility > .5)
    center_valid_mask = tracking_mask & (batch["future_center_valid"].to(device) > .5)
    visible_center_mask = center_valid_mask & (future_visibility > .5)
    offscreen_center_mask = center_valid_mask & (future_visibility <= .5)
    future_box = batch["future_box"].to(device)
    step_weights = torch.linspace(1, 2, predicted_box.shape[1], device=device)
    step_weights = step_weights / step_weights.mean()
    center_errors = F.smooth_l1_loss(
        predicted_box[..., :2], future_box[..., :2], beta=.02, reduction="none"
    ).mean(dim=-1)

    def weighted_center_loss(mask):
        if not mask.any():
            return predicted_box[..., :2].sum() * 0
        weights = step_weights[None, :, None] * mask
        return (center_errors * weights).sum() / weights.sum().clamp_min(1)

    visible_center_loss = weighted_center_loss(visible_center_mask)
    offscreen_center_loss = weighted_center_loss(offscreen_center_mask)
    if visible_mask.any():
        shape_errors = F.smooth_l1_loss(
            predicted_box[..., 2:5], future_box[..., 2:5], beta=.02, reduction="none"
        ).mean(dim=-1)
        weights = step_weights[None, :, None] * visible_mask
        shape_loss = (shape_errors * weights).sum() / weights.sum().clamp_min(1)
    else:
        shape_loss = predicted_box[..., 2:5].sum() * 0
    box_loss = (
        2 * visible_center_loss
        + 2 * offscreen_center_coefficient * offscreen_center_loss + shape_loss
    )
    future_depth = batch["future_depth_m"].to(device).clamp_min(1e-4)
    depth_mask = tracking_mask & (batch["future_depth_valid"].to(device) > .5)
    depth_loss = (
        F.smooth_l1_loss(predicted_log_depth[depth_mask], future_depth.log()[depth_mask])
        if depth_mask.any() else predicted_log_depth.sum() * 0
    )
    visibility_loss = F.binary_cross_entropy_with_logits(
        visibility_logits[tracking_mask], future_visibility[tracking_mask]
    )
    loss = (action_coefficient * action_loss + box_coefficient * box_loss
            + depth_loss + .1 * visibility_loss)
    return {
        "loss": loss, "action_loss": action_loss, "box_loss": box_loss,
        "depth_loss": depth_loss, "visibility_loss": visibility_loss,
    }, input_entity_mask, obstacle_slots


def supervised_validation_probe(policy, dataset, indices, entity_config, device,
                                depth_mode, seed, coefficients):
    with isolated_evaluation(policy, device), torch.no_grad():
        torch.manual_seed(seed)
        if device == "mps":
            torch.mps.manual_seed(seed)
        elif device == "cuda":
            torch.cuda.manual_seed(seed)
        batch = default_collate([dataset[index] for index in indices])
        losses, _, _ = supervised_batch_losses(
            policy, batch, entity_config, device, depth_mode, False, 1., *coefficients
        )
        result = {f"supervised_{name}": float(value) for name, value in losses.items()}
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError("Non-finite held-out supervised loss")
        return result


def train(
    collection: Path,
    output: Path,
    pretrained: str,
    steps: int,
    batch_size: int,
    history_steps: int,
    depth_mode: str,
    device: str,
    learning_rate: float,
    seed: int,
    paired_batches: bool,
    resume: Path | None,
    tasks: tuple[str, ...] | None,
    action_coefficient: float,
    manipulation_focus_probability: float,
    box_coefficient: float,
    offscreen_center_coefficient: float,
    phase_prompts: bool,
    obstacle_input_start_probability: float,
    obstacle_input_end_probability: float,
    obstacle_input_decay_steps: int,
    obstacle_sample_probability: float,
    training_state: Path | None,
    checkpoint_interval: int,
    trainable_scope: str,
    learning_rate_schedule: str = "constant",
    validation_split: Path | None = None,
    validation_interval: int = 250,
    validation_seed: int = 961000,
    warmup_fraction: float = .1,
    supervised_validation_probe_enabled: bool = False,
    retain_validation_checkpoints: bool = False,
    allow_device_rng_migration: bool = False,
    device_rng_seed: int | None = None,
    episode_storage: Path | None = None,
    initialize_only: bool = False,
) -> None:
    if initialize_only and (
        resume is not None or output.exists()
        or (training_state is not None and training_state.exists())
    ):
        raise ValueError("Initialization requires a fresh output and no existing model or training state")
    scheduled_learning_rate(0, steps, learning_rate, learning_rate_schedule, warmup_fraction)
    if (
        steps <= 0
        or batch_size <= 0
        or action_coefficient <= 0
        or not 0 <= manipulation_focus_probability <= 1
        or box_coefficient < 0
        or offscreen_center_coefficient < 0
        or not 0 <= obstacle_input_start_probability <= 1
        or not 0 <= obstacle_input_end_probability <= 1
        or obstacle_input_decay_steps <= 0
        or not 0 <= obstacle_sample_probability <= 1
        or checkpoint_interval <= 0
        or validation_interval <= 0
    ):
        raise ValueError(
            "steps, batch size, and action coefficient must be positive; "
            "contrast coefficient must be non-negative; manipulation focus "
            "probability must be within [0, 1]; box coefficients must be "
            "non-negative; obstacle input probabilities must be within "
            "[0, 1], their decay steps must be positive, and obstacle "
            "sampling probability must be within [0, 1]"
        )
    if paired_batches and batch_size % 2:
        raise ValueError("Paired batches require an even batch size")
    if supervised_validation_probe_enabled and validation_split is None:
        raise ValueError("Supervised validation requires a frozen validation split")
    if output.exists() and not (training_state is not None and training_state.exists()):
        raise FileExistsError(f"Fresh SFT output already exists: {output}")
    split = json.loads(validation_split.read_text()) if validation_split is not None else None
    excluded = tuple(split["validation"]) if split is not None else ()
    catalog = json.loads((collection / "collection.json").read_text())["successful_episodes"]
    if not set(excluded).issubset(catalog):
        raise ValueError("Frozen validation sources are missing from the requested collection")
    resume_state = (torch.load(training_state, map_location="cpu", weights_only=False)
                    if training_state is not None and training_state.exists() else None)
    if resume_state is not None:
        validate_state(resume_state, allow_untrained=True)
    device_rng_plan = (plan_device_rng(
        resume_state, device, allow_device_rng_migration, device_rng_seed,
    ) if resume_state is not None else None)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    entity_config = EntityModelConfig(history_steps=history_steps, depth_input=depth_mode != "blind")
    source_checkpoint = pretrained
    normalization = None
    if resume is None:
        base = SmolVLAPolicy.from_pretrained(pretrained)
        attach_entity_modules(base, entity_config)
        policy = base.to(device).train()
    else:
        policy, loaded_config, normalization = load_entity_checkpoint(
            resume, device
        )
        if loaded_config != entity_config:
            raise ValueError(
                f"Resume config {loaded_config} does not match requested config {entity_config}"
            )
        source_checkpoint = json.loads((resume / "metadata.json").read_text())[
            "source_checkpoint"
        ]
        policy.train()
    if resume_state is not None and "log" in resume_state:
        normalization = resume_state["log"]["dataset"]["normalization"]
    if trainable_scope == "action-output":
        for name, parameter in policy.named_parameters():
            parameter.requires_grad = name.startswith(
                "model.action_out_proj."
            )
    elif trainable_scope != "all":
        raise ValueError(f"Unknown trainable scope {trainable_scope}")
    policy.model.vlm_with_expert.lm_expert.float()
    dataset = EntityTrajectoryDataset(
        collection,
        policy.config.chunk_size,
        history_steps,
        entity_config.max_entities,
        tasks,
        normalization,
        phase_prompts,
        excluded,
        storage_path=episode_storage,
    )
    output.mkdir(parents=True, exist_ok=True)
    if training_state is None:
        training_state = output / "training_state.pt"
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-10)
    start_step = 1
    log = {
        "kind": "sft", "steps": [], "validation": [],
        "run_config": {
            "collection": str(collection), "output": str(output), "pretrained": pretrained,
            "resume": str(resume) if resume is not None else None,
            "steps": steps, "batch_size": batch_size, "history_steps": history_steps,
            "depth_mode": depth_mode, "device": device, "learning_rate": learning_rate,
            "learning_rate_schedule": learning_rate_schedule, "seed": seed,
            "warmup_fraction": warmup_fraction,
            "paired_batches": paired_batches, "trainable_scope": trainable_scope,
            "action_coefficient": action_coefficient,
            "box_coefficient": box_coefficient, "offscreen_center_coefficient": offscreen_center_coefficient,
            "manipulation_focus_probability": manipulation_focus_probability,
            "phase_prompts": phase_prompts, "validation_interval": validation_interval,
            "validation_split": str(validation_split) if validation_split is not None else None,
            "validation_sources": list(excluded),
            "collection_sources_sha256": hashlib.sha256("\n".join(catalog).encode()).hexdigest(),
            "validation_seed": validation_seed, "validation_noise_scale": .1,
            "validation_horizon": 300, "validation_action_steps": 16,
            "training_state": str(training_state), "checkpoint_interval": checkpoint_interval,
            "supervised_validation_probe": supervised_validation_probe_enabled,
            "retain_validation_checkpoints": retain_validation_checkpoints,
        },
        "dataset": {
            "episodes": len(dataset.episodes), "frames": len(dataset),
            "held_out_episodes": len(excluded), "normalization": dataset.normalization(),
            "episodes_by_command": {task: len(indices) for task, indices in dataset.task_episode_indices.items()},
        },
    }
    validation_interval_changed = False
    if resume_state is not None:
        assert device_rng_plan is not None
        state = resume_state
        start_step = int(state["step"]) + 1
        validation_interval_changed = (
            state["log"]["run_config"]["validation_interval"] != validation_interval
        )
        log = restore_training_state(
            state, policy, optimizer, rng, log["run_config"], device_rng_plan,
        )
        del state
        del resume_state
    log["memory_optimization"] = {
        "method": "microbatch-activation-checkpointing" if batch_size > 16 else "none",
        "micro_batch_size": min(batch_size, 16), "optimizer_batch_size": batch_size,
    }
    if episode_storage is not None:
        log["episode_storage"] = {
            "path": str(episode_storage), "format": "sqlite-episode-bytes",
            "version": 1,
        }
    else:
        log.pop("episode_storage", None)
    training_sources = {
        task: [str(dataset.episodes[index][0]) for index in indices]
        for task, indices in dataset.task_episode_indices.items()
    }
    atomic_json(output / "data_split.json", {
        "training": training_sources, "validation": list(excluded),
        **({"expert_bounds": split["expert_bounds"]} if split is not None else {}),
    })
    log["status"] = "training"
    atomic_json(output / "sft_training.json", log)
    controller, sim, validation = None, None, []
    probe_dataset, probe_indices = None, []
    probe_seed = validation_seed + 100000
    if supervised_validation_probe_enabled:
        held_out = set(excluded)
        probe_dataset = EntityTrajectoryDataset(
            collection, policy.config.chunk_size, history_steps,
            entity_config.max_entities, normalization=dataset.normalization(),
            phase_prompts=phase_prompts,
            excluded_sources=tuple(source for source in catalog if source not in held_out),
            storage_path=episode_storage,
        )
        episode_indices = {str(source): i for i, (source, _, _) in enumerate(probe_dataset.episodes)}
        probe_rng = np.random.default_rng(probe_seed)
        for source in excluded:
            candidates = probe_dataset.episode_sample_indices[episode_indices[source]]
            probe_indices.extend(int(index) for index in probe_rng.choice(
                candidates, size=4, replace=len(candidates) < 4
            ))
        log["supervised_probe"] = {
            "seed": probe_seed, "samples": len(probe_indices),
            "frames": [
                {"source": str(probe_dataset.episodes[probe_dataset.samples[i][0]][0]),
                 "frame_index": probe_dataset.samples[i][1]} for i in probe_indices
            ],
            "normalization": "training-only",
        }

    if initialize_only:
        log["status"] = "initialized"
        save_training_state(training_state, policy, optimizer, 0, rng, log)
        atomic_json(output / "sft_training.json", log)
        return

    def evaluate_at(step):
        optimizer.zero_grad(set_to_none=True)
        log["status"] = "evaluating"
        atomic_json(output / "sft_training.json", log)
        started = time.monotonic()
        metrics = evaluate_without_training_rng(
            controller, sim, validation, split["expert_bounds"], validation_seed
        )
        if probe_dataset is not None:
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            metrics.update(supervised_validation_probe(
                policy, probe_dataset, probe_indices, entity_config, device,
                depth_mode, probe_seed,
                (action_coefficient, box_coefficient, offscreen_center_coefficient),
            ))
        metrics.update(step=step, samples_seen=step * batch_size,
                       evaluation_seconds=time.monotonic() - started)
        log["validation"].append(metrics)
        log["status"] = "training"
        atomic_json(output / "sft_training.json", log)
        if retain_validation_checkpoints:
            save_training_state(
                output / "checkpoints" / f"training_state_{step:06d}.pt",
                policy, optimizer, step, rng, log,
            )

    if split is not None:
        controller = EntityPolicy(output, device, entity_config.depth_input, phase_prompts=phase_prompts,
                                  loaded_policy=(policy, entity_config, dataset.normalization()))
        sim = BeamSimulation(MODEL_ROOT, render=True)
        validation = [CorpusEpisode(Path(source), json.loads(dataset.read_bytes(Path(source), "episode.json")))
                      for source in excluded]
        if (start_step == 1 and not log["validation"]) or validation_interval_changed:
            evaluate_at(start_step - 1)
    if start_step == 1:
        save_training_state(training_state, policy, optimizer, 0, rng, log)
    completed_step = start_step - 1
    for step in range(start_step, steps + 1):
        if (output / "STOP").exists():
            break
        step_start = time.monotonic()
        current_lr = scheduled_learning_rate(step, steps, learning_rate, learning_rate_schedule, warmup_fraction)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        if paired_batches and dataset.task_contrast_groups:
            matched_phase = step % 3
            matched_step = matched_phase in (1, 2)
        elif paired_batches:
            matched_phase = 1 if step % 2 else 0
            matched_step = matched_phase == 1
        else:
            matched_phase = 0
            matched_step = False
        if matched_phase == 1:
            batch = default_collate([
                dataset[index]
                for index in dataset.paired_indices(
                    rng, batch_size, manipulation_focus_probability
                )
            ])
        elif matched_phase == 2:
            batch = default_collate([
                dataset[index]
                for index in dataset.task_contrast_indices(
                    rng, batch_size, manipulation_focus_probability
                )
            ])
        else:
            batch = default_collate([
                dataset[index]
                for index in dataset.obstacle_balanced_indices(
                    rng,
                    batch_size,
                    obstacle_sample_probability,
                    manipulation_focus_probability,
                )
            ])
        obstacle_input_probability = linear_probability(
            step,
            obstacle_input_start_probability,
            obstacle_input_end_probability,
            obstacle_input_decay_steps,
        )
        losses, input_entity_mask, obstacle_slots = supervised_batch_losses(
            policy, batch, entity_config, device, depth_mode, matched_step,
            obstacle_input_probability, action_coefficient, box_coefficient,
            offscreen_center_coefficient,
        )
        loss = losses["loss"]
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite supervised loss; previous checkpoint preserved")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 10))
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite supervised gradients; previous checkpoint preserved")
        optimizer.step()
        completed_step = step
        elapsed = time.monotonic() - step_start
        row = {
            "step": step, "samples_seen": step * batch_size,
            **{name: float(value.detach()) for name, value in losses.items()},
            "learning_rate": current_lr,
            "gradient_norm": gradient_norm, "step_seconds": elapsed, "completed_at": time.time(),
            "batch_size": batch_size, "micro_batch_size": min(batch_size, 16),
        }
        log["steps"].append(row)
        row["eta_seconds"] = float(np.mean([item["step_seconds"] for item in log["steps"][-25:]])) * (steps - step)
        atomic_json(output / "sft_training.json", log)
        if step == 1 or step % 25 == 0 or step == steps:
            print(json.dumps({
                **row,
                "obstacle_input_probability": (
                    obstacle_input_probability
                ),
                "obstacle_input_fraction": float(
                    (
                        input_entity_mask[obstacle_slots].float().mean()
                        if obstacle_slots.any()
                        else torch.tensor(0., device=device)
                    ).detach()
                ),
                "obstacle_sample_probability": (
                    obstacle_sample_probability
                ),
            }), flush=True)
        # Do not keep a completed training graph alive during the larger fixed probe.
        del losses, loss, input_entity_mask, obstacle_slots
        if controller is not None and (step % validation_interval == 0 or step == steps):
            evaluate_at(step)
        if (
            training_state is not None
            and (step == start_step or step % checkpoint_interval == 0 or step == steps)
        ):
            save_training_state(
                training_state,
                policy,
                optimizer,
                step,
                rng,
                log,
            )
    log["status"] = "stopped" if completed_step < steps else "completed"
    atomic_json(output / "sft_training.json", log)
    save_training_state(training_state, policy, optimizer, completed_step, rng, log)
    if sim is not None:
        sim.close()
    save_entity_checkpoint(
        output,
        policy,
        entity_config,
        source_checkpoint,
        dataset.normalization(),
        exist_ok=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--episode-storage", type=Path,
                        help="Immutable indexed episode container; retain the original collection catalog")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pretrained",
        default="outputs/smolvla_target_color_combined_2000/checkpoints/001000/pretrained_model",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--depth-mode", choices=("known", "blind", "mixed"), default="mixed")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--learning-rate-schedule",
                        choices=("constant", "triangular", "warmup-constant"), default="constant")
    parser.add_argument("--warmup-fraction", type=float, default=.1)
    parser.add_argument("--validation-split", type=Path)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--validation-seed", type=int, default=961000)
    parser.add_argument("--supervised-validation-probe", action="store_true")
    parser.add_argument("--retain-validation-checkpoints", action="store_true")
    parser.add_argument("--action-coefficient", type=float, default=1)
    parser.add_argument("--box-coefficient", type=float, default=1)
    parser.add_argument(
        "--offscreen-center-coefficient", type=float, default=.1
    )
    parser.add_argument(
        "--manipulation-focus-probability",
        type=float,
        default=.75,
        help=(
            "Probability of sampling push-contact or post-grasp place frames "
            "instead of uniformly sampling the full episode."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--paired-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Alternate same-scene, different-target batches with shuffled batches.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("grasp", "view", "push", "place"),
    )
    parser.add_argument(
        "--phase-prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--obstacle-input-start-probability",
        type=float,
        default=1.,
    )
    parser.add_argument(
        "--obstacle-input-end-probability",
        type=float,
        default=.5,
    )
    parser.add_argument(
        "--obstacle-input-decay-steps",
        type=int,
    )
    parser.add_argument(
        "--obstacle-sample-probability",
        type=float,
        default=0.,
    )
    parser.add_argument("--training-state", type=Path)
    parser.add_argument("--initialize-only", action="store_true",
                        help="Save fresh seeded step-zero model/Adam/RNG state without evaluation or updates")
    parser.add_argument("--allow-device-rng-migration", action="store_true")
    parser.add_argument("--device-rng-seed", type=int)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument(
        "--trainable-scope",
        choices=("all", "action-output"),
        default="all",
    )
    args = parser.parse_args()
    train(
        args.collection,
        args.output,
        args.pretrained,
        args.steps,
        args.batch_size,
        args.history_steps,
        args.depth_mode,
        args.device,
        args.learning_rate,
        args.seed,
        args.paired_batches,
        args.resume,
        tuple(args.tasks) if args.tasks else None,
        args.action_coefficient,
        args.manipulation_focus_probability,
        args.box_coefficient,
        args.offscreen_center_coefficient,
        args.phase_prompts,
        args.obstacle_input_start_probability,
        args.obstacle_input_end_probability,
        args.obstacle_input_decay_steps or args.steps,
        args.obstacle_sample_probability,
        args.training_state,
        args.checkpoint_interval,
        args.trainable_scope,
        args.learning_rate_schedule,
        args.validation_split,
        args.validation_interval,
        args.validation_seed,
        args.warmup_fraction,
        args.supervised_validation_probe,
        args.retain_validation_checkpoints,
        args.allow_device_rng_migration,
        args.device_rng_seed,
        args.episode_storage,
        args.initialize_only,
    )


if __name__ == "__main__":
    main()
