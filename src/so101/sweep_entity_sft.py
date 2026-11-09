"""Serial, resumable, sample-matched local batch-size/learning-rate experiments."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from so101.cloud_checkpoint import atomic_json
from so101.rl_dashboard import active_run_outputs


COLLECTION = "data/simulation/entity_commands_with_difficulty_v1"
SPLIT = "outputs/entity_smolvla_trajectory_rl_all_assisted_lr1e4_before_step243_rerun/data_split.json"
PREVIOUS = Path("outputs/entity_smolvla_sft_batch_comparison/comparison.json")
GRID = ((128, 1e-5), (64, 1e-5), (128, 3e-6), (64, 3e-6),
        (128, 3e-5), (64, 3e-5))


def trials(samples: int, output: Path) -> list[dict]:
    if samples < 1536 or samples % 1536:
        raise ValueError("Sample budget must be a positive multiple of 1536")
    return [
        {"batch_size": batch, "learning_rate": rate, "steps": samples // batch,
         "samples": samples, "validation_interval": samples // batch // 4,
         "output": str(output / f"entity_smolvla_sft_b{batch}_lr{rate:.0e}")}
        for batch, rate in GRID
    ]


def command(trial: dict, device: str = "mps", allow_device_rng_migration: bool = False,
            device_rng_seed: int | None = None, episode_storage: Path | None = None) -> list[str]:
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("Unsupported sweep device")
    if allow_device_rng_migration and (
        type(device_rng_seed) is not int or not 0 <= device_rng_seed < 2**64
    ):
        raise ValueError("Device RNG migration requires an explicit valid device RNG seed")
    args = [
        sys.executable, "-u", "-m", "so101.train_entity_sim",
        "--collection", COLLECTION, "--output", trial["output"],
        "--pretrained", "lerobot/smolvla_base", "--device", device,
        "--steps", str(trial["steps"]), "--batch-size", str(trial["batch_size"]),
        "--history-steps", "4", "--depth-mode", "known",
        "--learning-rate", str(trial["learning_rate"]),
        "--learning-rate-schedule", "warmup-constant", "--warmup-fraction", ".1",
        "--seed", "61000", "--validation-split", SPLIT,
        "--validation-interval", str(trial["validation_interval"]),
        "--validation-seed", "961000", "--supervised-validation-probe",
        "--retain-validation-checkpoints",
        "--training-state", str(Path(trial["output"]) / "training_state.pt"),
        "--checkpoint-interval", str(max(1, trial["validation_interval"] // 4)),
        "--obstacle-input-end-probability", "1",
    ]
    if allow_device_rng_migration:
        args.extend(["--allow-device-rng-migration", "--device-rng-seed", str(device_rng_seed)])
    if episode_storage is not None:
        args.extend(["--episode-storage", str(episode_storage)])
    return args


def verify_log(log: dict, trial: dict, reference: dict) -> None:
    config = log["run_config"]
    if log["status"] != "completed":
        raise RuntimeError(f"{trial['output']}: incomplete training")
    if [row["step"] for row in log["steps"]] != list(range(1, trial["steps"] + 1)):
        raise RuntimeError("Optimizer history is not complete and continuous")
    for name in ("steps", "batch_size", "learning_rate", "validation_interval"):
        if config[name] != trial[name]:
            raise RuntimeError(f"Incorrect trial configuration: {name}")
    if config["pretrained"] != "lerobot/smolvla_base" or config["resume"] is not None:
        raise RuntimeError("Trial must initialize fresh from the public base")
    if config["learning_rate_schedule"] != "warmup-constant":
        raise RuntimeError("Incorrect sweep learning-rate schedule")
    if not config["supervised_validation_probe"] or not config["retain_validation_checkpoints"]:
        raise RuntimeError("Probe or checkpoint retention is disabled")
    for name in (
        "collection", "collection_sources_sha256", "depth_mode", "history_steps",
        "paired_batches", "seed", "warmup_fraction", "trainable_scope",
        "action_coefficient", "box_coefficient", "offscreen_center_coefficient",
        "manipulation_focus_probability", "phase_prompts", "validation_sources",
        "validation_seed", "validation_noise_scale", "validation_horizon",
        "validation_action_steps",
    ):
        if config[name] != reference["run_config"][name]:
            raise RuntimeError(f"Reference configuration changed: {name}")
    if log["dataset"] != reference["dataset"]:
        raise RuntimeError("Training split or normalization changed")
    expected_steps = list(range(0, trial["steps"] + 1, trial["validation_interval"]))
    if [row["step"] for row in log["validation"]] != expected_steps:
        raise RuntimeError("Evaluations are not aligned to sample-budget quarters")
    for row in log["steps"]:
        if (row["samples_seen"] != row["step"] * trial["batch_size"]
                or row["batch_size"] != trial["batch_size"]
                or row["micro_batch_size"] != min(
                    trial["batch_size"], config.get("micro_batch_size", 16))
                or not all(math.isfinite(row[name]) for name in ("loss", "gradient_norm"))):
            raise RuntimeError("Incorrect exposure, microbatching, or non-finite update")
    if not math.isclose(log["steps"][-1]["learning_rate"], trial["learning_rate"], rel_tol=1e-12):
        raise RuntimeError("Final learning rate should remain at its plateau")
    for row in log["validation"]:
        if (row["episodes"] != 40 or row["sampler"] != "sde"
                or row["denoising_noise_scale"] != .1
                or not math.isfinite(row["supervised_loss"])
                or [r["source"] for r in row["rollouts"]] != config["validation_sources"]
                or [r["seed"] for r in row["rollouts"]] != list(range(961000, 961040))):
            raise RuntimeError("Frozen evaluation or supervised probe changed")
    probe = log["supervised_probe"]
    if probe["samples"] != 160 or probe["seed"] != 1061000:
        raise RuntimeError("Incorrect fixed held-out probe")
    if [row["source"] for row in probe["frames"][::4]] != config["validation_sources"]:
        raise RuntimeError("Probe is not four frames per frozen problem")


def verify_state(output: Path, step: int) -> None:
    import torch

    state = torch.load(output / "training_state.pt", map_location="cpu", weights_only=False)
    if state["step"] != step or not state["optimizer"]["state"] or "rng" not in state:
        raise RuntimeError("Incomplete durable optimizer/RNG checkpoint")
    values = list(state["model"].values()) + [
        value for item in state["optimizer"]["state"].values()
        for value in item.values() if isinstance(value, torch.Tensor)
    ]
    if not all(torch.isfinite(value).all() for value in values):
        raise RuntimeError("Non-finite saved model or Adam state")
    for name in ("model.pt", "entity_config.json", "metadata.json"):
        if not (output / name).is_file():
            raise RuntimeError(f"Missing final export: {name}")
    del values, state


def summary(log: dict, elapsed: float) -> dict:
    final = log["validation"][-1]
    recent = log["steps"][-max(3, len(log["steps"]) // 10):]
    return {
        "output": log["run_config"]["output"],
        "batch_size": log["run_config"]["batch_size"],
        "learning_rate": log["run_config"]["learning_rate"],
        "samples": log["steps"][-1]["samples_seen"],
        "successes": sum(row["success"] for row in final["rollouts"]),
        "by_command": final["by_task"],
        "completion_efficiency": final["completion_efficiency"],
        "rms_jerk": final["rms_jerk"],
        "wrong_object_rate": final["wrong_object_rate"],
        "held_out_loss": final["supervised_loss"],
        "recent_mean_training_loss": statistics.mean(row["loss"] for row in recent),
        "training_seconds": sum(row["step_seconds"] for row in log["steps"]),
        "evaluation_seconds": sum(row["evaluation_seconds"] for row in log["validation"]),
        "driver_session_seconds": elapsed,
        "training_curve": [{name: row[name] for name in
                            ("samples_seen", "loss", "action_loss", "box_loss", "step_seconds")}
                           for row in log["steps"]],
        "validation_curve": [{name: row[name] for name in
                              ("samples_seen", "success_rate", "supervised_loss")}
                             for row in log["validation"]],
        "rollouts": final["rollouts"],
        "supervised_probe": log["supervised_probe"],
    }


def common_success_jerk(results: list[dict]) -> list[dict]:
    comparisons = []
    for i, left in enumerate(results):
        reference = {row["source"]: row for row in left["rollouts"] if row["success"]}
        for right in results[i + 1:]:
            common = [row for row in right["rollouts"]
                      if row["success"] and row["source"] in reference]
            comparisons.append({
                "left": left["output"], "right": right["output"], "problems": len(common),
                "left_rms_jerk": statistics.mean(
                    reference[row["source"]]["rms_jerk"] for row in common) if common else None,
                "right_rms_jerk": statistics.mean(row["rms_jerk"] for row in common) if common else None,
                "note": "Each successful rollout's full duration, not matched-length windows.",
            })
    return comparisons


def plots(output: Path, results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for result in results:
        label = f"B{result['batch_size']}, LR {result['learning_rate']:.0e}"
        rows = result["training_curve"]
        cycles = [rows[i:i + 3] for i in range(0, len(rows), 3)]
        axes[0].loglog([group[-1]["samples_seen"] for group in cycles],
                       [statistics.mean(row["loss"] for row in group) for group in cycles],
                       label=label)
        validation = result["validation_curve"][1:]
        axes[1].loglog([row["samples_seen"] for row in validation],
                       [row["supervised_loss"] for row in validation], marker="o", label=label)
    for ax, title in zip(axes, ("Training mean loss (three-update cycles)", "Fixed held-out mean loss")):
        ax.set(title=title, xlabel="Samples processed", ylabel="Mean loss")
        ax.legend()
        ax.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output / "loss_vs_samples.png", dpi=150)
    plt.close(figure)
    figure, ax = plt.subplots(figsize=(7, 4))
    for result in results:
        rows = result["validation_curve"]
        ax.plot([row["samples_seen"] for row in rows],
                [row["success_rate"] for row in rows], marker="o",
                label=f"B{result['batch_size']}, LR {result['learning_rate']:.0e}")
    ax.set(xlabel="Samples processed", ylabel="Frozen40 success rate", ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output / "success_vs_samples.png", dpi=150)
    plt.close(figure)


def run(output: Path, samples: int, device: str = "mps",
        allow_device_rng_migration: bool = False, device_rng_seed: int | None = None,
        episode_storage: Path | None = None) -> None:
    plan = trials(samples, output)
    previous = json.loads(PREVIOUS.read_text())
    if previous["status"] != "completed" or active_run_outputs():
        raise RuntimeError("Original comparison must be complete and all trainers inactive")
    reference = json.loads(Path("outputs/entity_smolvla_sft_batch128_1000/sft_training.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "comparison.json"
    existing = json.loads(status_path.read_text()) if status_path.exists() else None
    if existing is not None and existing["trials"] != plan:
        raise ValueError("Existing sweep has a different fixed sample budget/grid")
    payload = {
        "driver_pid": os.getpid(), "trials": plan, "results": [],
        "notes": [
            "Sample exposure is a compute proxy, not an exact FLOP or wall-time match.",
            "All trials use microbatch16, fresh seed61000, and training-only normalization.",
            "Same sampling distribution/seed does not imply an identical sample stream across batches.",
            "Adam betas are unchanged; their effective sample half-life changes with batch size.",
            "10% warmup then constant LR; quarters are aligned by sample exposure.",
            "Frozen40 is a tuning set, not an independent final test set.",
            "Single-seed pilot rankings need confirmation; no model deployed automatically.",
        ],
    }

    def write(status: str, **fields):
        payload.update(status=status, updated_at=time.time(), **fields)
        atomic_json(status_path, payload)

    with (output / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            env = {**os.environ, "PYTORCH_MPS_HIGH_WATERMARK_RATIO": ".65",
                   "PYTORCH_MPS_LOW_WATERMARK_RATIO": ".5", "OMP_NUM_THREADS": "2"}
            for trial in plan:
                if active_run_outputs():
                    raise RuntimeError("Refusing a second concurrent policy/trainer")
                target = Path(trial["output"])
                log_path = target / "sft_training.json"
                log = json.loads(log_path.read_text()) if log_path.exists() else None
                started = time.monotonic()
                if log is None or log["status"] != "completed":
                    write("training", current_trial=trial)
                    with (output / f"{target.name}.stdout.log").open("a") as stream:
                        child = subprocess.Popen(command(
                            trial, device, allow_device_rng_migration, device_rng_seed,
                            episode_storage,
                        ), env=env, stdout=stream,
                                                 stderr=subprocess.STDOUT)
                        write("training", trainer_pid=child.pid)
                        code = child.wait()
                    if code:
                        raise RuntimeError(f"{target.name} exited {code}; inspect its stdout log")
                    log = json.loads(log_path.read_text())
                verify_log(log, trial, reference)
                verify_state(target, trial["steps"])
                for row in log["validation"]:
                    snapshot = target / "checkpoints" / f"training_state_{row['step']:06d}.pt"
                    if not snapshot.is_file():
                        raise RuntimeError(f"Missing retained evaluation checkpoint: {snapshot}")
                result = summary(log, time.monotonic() - started)
                if payload["results"] and result["supervised_probe"] != payload["results"][0]["supervised_probe"]:
                    raise RuntimeError("Fixed held-out frames/noise differ between trials")
                payload["results"].append(result)
                payload["common_success_jerk"] = common_success_jerk(payload["results"])
                plots(output, payload["results"])
                write("trial_completed")
            write("completed")
        except (RuntimeError, ValueError, OSError, KeyError) as error:
            write("failed", error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/entity_smolvla_sft_sweep_24576"))
    parser.add_argument("--samples", type=int, default=24576)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--allow-device-rng-migration", action="store_true")
    parser.add_argument("--device-rng-seed", type=int)
    parser.add_argument("--episode-storage", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        plan = trials(args.samples, args.output)
        print(json.dumps({"trials": plan, "commands": [
            command(trial, args.device, args.allow_device_rng_migration, args.device_rng_seed,
                    args.episode_storage)
            for trial in plan
        ]}, indent=2))
    else:
        run(args.output, args.samples, args.device, args.allow_device_rng_migration,
            args.device_rng_seed, args.episode_storage)


if __name__ == "__main__":
    main()
