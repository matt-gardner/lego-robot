"""Real GPU qualification in an isolated container; never advance the original trial."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import gc
import json
from pathlib import Path
import random
import sys
import tempfile

import mujoco
import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from so101.cloud_checkpoint import sha256, validate_state
from so101.control_model.modeling_entity_smolvla import EntityModelConfig, attach_entity_modules
from so101.resume_rng import plan_device_rng
from so101.train_entity_sim import restore_training_state, save_training_state


def tiny_recovery(device: str, directory: Path) -> None:
    model = torch.nn.Linear(2, 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    sampling = np.random.default_rng(61000)
    config = {"device": device, "learning_rate": .001, "validation_interval": 1}
    log = {"run_config": config, "steps": [{"step": 1}]}
    def advance(policy, adam):
        adam.zero_grad(set_to_none=True)
        factor = random.random() + np.random.random() + sampling.random() + torch.rand(()).item()
        policy(torch.rand(2, 2, device=device)).square().sum().mul(factor).backward()
        adam.step()
    advance(model, optimizer)
    path = directory / "tiny-state.pt"
    save_training_state(path, model, optimizer, 1, sampling, log)
    state = torch.load(path, map_location="cpu", weights_only=False)
    validate_state(state)
    advance(model, optimizer)
    expected_model = deepcopy(model.state_dict())
    expected_adam = deepcopy(optimizer.state_dict())
    restored = torch.nn.Linear(2, 2).to(device)
    adam = torch.optim.AdamW(restored.parameters(), lr=.001)
    restore_training_state(state, restored, adam, sampling, deepcopy(config),
                           plan_device_rng(state, device))
    advance(restored, adam)
    if any(not torch.equal(value, restored.state_dict()[name])
           for name, value in expected_model.items()):
        raise ValueError("Full-state recovery did not reproduce the next optimizer update")
    actual_adam = adam.state_dict()
    for key, values in expected_adam["state"].items():
        for name, value in values.items():
            actual = actual_adam["state"][key][name]
            if isinstance(value, torch.Tensor):
                if not torch.equal(value, actual):
                    raise ValueError("Recovered Adam state differs after the next update")
            elif value != actual:
                raise ValueError("Recovered Adam metadata differs")


def restore_original(path: Path, migration_seed: int) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    validate_state(state, allow_untrained=True)
    config = state["log"]["run_config"]
    if config["resume"] is not None or config["pretrained"] != "lerobot/smolvla_base":
        raise ValueError("Qualification supports only the approved base-initialized SFT recovery")
    policy = SmolVLAPolicy.from_pretrained(config["pretrained"])
    attach_entity_modules(policy, EntityModelConfig(
        history_steps=config["history_steps"], depth_input=config["depth_mode"] != "blind",
    ))
    policy = policy.to("cuda").train()
    if config["trainable_scope"] == "action-output":
        for name, parameter in policy.named_parameters():
            parameter.requires_grad = name.startswith("model.action_out_proj.")
    elif config["trainable_scope"] != "all":
        raise ValueError("Unknown original trainable scope")
    policy.model.vlm_with_expert.lm_expert.float()
    parameters = {name: value for name, value in policy.named_parameters() if value.requires_grad}
    if list(parameters) != list(state["model"]):
        raise ValueError("Original checkpoint trainable parameter layout differs from runtime")
    optimizer = torch.optim.AdamW(list(parameters.values()), lr=config["learning_rate"],
                                 weight_decay=1e-10)
    requested = {**config, "device": "cuda"}
    restore_training_state(state, policy, optimizer, np.random.default_rng(), requested,
                           plan_device_rng(state, "cuda", True, migration_seed))
    for name, value in state["model"].items():
        if not torch.equal(value, parameters[name].detach().cpu()):
            raise ValueError("Original trained parameter did not restore exactly on CUDA")
    saved_adam = state["optimizer"]["state"]
    restored_adam = optimizer.state_dict()["state"]
    for key, values in saved_adam.items():
        for name, value in values.items():
            actual = restored_adam[key][name]
            if isinstance(value, torch.Tensor):
                if not torch.equal(value.cpu(), actual.cpu()):
                    raise ValueError("Original Adam tensors did not restore exactly on CUDA")
            elif value != actual:
                raise ValueError("Original Adam metadata changed")
    return state["step"]


def qualify(original: Path, migration_seed: int) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Qualification requires exactly one visible CUDA GPU")
    if torch.version.cuda is None or not torch.version.cuda.startswith("13.0"):
        raise ValueError("Qualification requires the exact CUDA 13.0 PyTorch runtime")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support the required BF16 prefix")
    value = torch.ones(8, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    (value @ value).sum().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(value.grad).all():
        raise ValueError("CUDA BF16 gradient check failed")
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        renderer.update_scene(data)
        image = renderer.render()
        if image.shape != (64, 64, 3) or image.dtype != np.uint8:
            raise ValueError("EGL did not render the required RGB frame")
    finally:
        renderer.close()
    with tempfile.TemporaryDirectory() as directory:
        tiny_recovery("cuda", Path(directory))
    gc.collect()
    torch.cuda.empty_cache()
    original_digest = sha256(original)
    step = restore_original(original, migration_seed)
    if sha256(original) != original_digest:
        raise ValueError("Original checkpoint changed during read-only qualification")
    return {"status": "RUNTIME_PASS", "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "bf16": True, "egl": True,
            "tiny_next_update_exact": True, "original_state_restored": True,
            "original_step": step, "original_sha256": original_digest,
            "migration_seed": migration_seed, "original_trial_advanced": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--migration-seed", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.migration_seed < 2**64:
        raise ValueError("Qualification migration seed must match explicit approved migration")
    with redirect_stdout(sys.stderr):
        result = qualify(args.original, args.migration_seed)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
