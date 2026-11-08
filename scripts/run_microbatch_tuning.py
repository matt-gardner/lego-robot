"""Benchmark microbatches serially on the existing GPU, then replay a matched interval."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def feasible_candidates(candidates):
    feasible = [
        item for item in candidates
        if item["status"] == "completed"
        and item["max_reserved_bytes"] <= .9 * item["total_memory_bytes"]
    ]
    if not feasible:
        raise RuntimeError("No tested microbatch retained ten-percent memory headroom")
    return feasible


def select_candidates(candidates):
    feasible = feasible_candidates(candidates)
    selected = min(feasible, key=lambda item: item["steady_mean_seconds"])["microbatch"]
    alternatives = [item for item in feasible if item["microbatch"] != 16]
    if not alternatives:
        raise RuntimeError("No changed microbatch is feasible for the requested sensitivity comparison")
    comparison = min(alternatives, key=lambda item: item["steady_mean_seconds"])["microbatch"]
    return selected, comparison


def run(root, image, job_id, no_checkpointing_only=False, accumulation_only=False,
        accumulation_microbatches=(16,)):
    inputs = Path("/home/ubuntu/lego-sweep/inputs")
    sweep = Path("/home/ubuntu/lego-sweep/output")
    scripts = Path("/home/ubuntu/lego-sweep/microbatch-scripts")
    request = json.loads((inputs / "initialization-commands.json").read_text())[0]
    argv = request["gpu_argv"]
    directory_name = Path(argv[argv.index("--output") + 1]).name
    reference = sweep / directory_name
    base = reference / "checkpoints/training_state_000048.pt"
    if not base.is_file() or not (sweep / "b128-lr3e6-export.json").is_file():
        raise ValueError("Completed baseline export and its real step-48 checkpoint are required")
    with base.open("rb") as stream:
        base_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    root.mkdir(exist_ok=True)
    with (root / "tuning.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def continuation(label, microbatch, until_step, activation_checkpointing=True, accumulation=False):
            destination = root / label
            if destination.exists():
                receipt = json.loads((destination / "benchmark.json").read_text())
                if (receipt["microbatch"] != microbatch or receipt["base_sha256"] != base_sha
                        or receipt.get("activation_checkpointing", True) != activation_checkpointing
                        or receipt.get("gradient_accumulation", False) != accumulation
                        or (receipt["status"] == "completed" and receipt["until_step"] != until_step)):
                    raise ValueError("Existing benchmark receipt belongs to a different continuation")
                return receipt
            destination.mkdir()
            output = destination / directory_name
            output.mkdir()
            shutil.copyfile(base, output / "training_state.pt")
            mounts = [
                (inputs / "collection.json", "/opt/so101/data/simulation/entity_commands_with_difficulty_v1/collection.json"),
                (inputs / "episodes.sqlite", "/opt/so101/data/episodes.sqlite"),
                (inputs / "data-split.json", "/opt/so101/outputs/entity_smolvla_trajectory_rl_all_assisted_lr1e4_before_step243_rerun/data_split.json"),
                (inputs / "hf_cache", "/work/cache/huggingface"),
                (inputs / "data_model/model", "/opt/so101/data/simulation/model"),
                (scripts / "benchmark_gpu_microbatch.py", "/microbatch.py"),
            ]
            if accumulation:
                mounts.append((scripts / "train_entity_sim.py",
                               "/opt/so101/src/so101/train_entity_sim.py"))
            command = ["docker", "run", "--rm", "--network", "none", "--read-only",
                       "--user", "0:0", "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g"]
            for source, target in mounts:
                command += ["--mount", f"type=bind,src={source},dst={target},readonly"]
            command += [
                "--mount", f"type=bind,src={destination},dst=/opt/so101/outputs/entity_smolvla_sft_lr_sweep",
                "--gpus", "all", "--name", f"lego-microbatch-{job_id}-{label}", image,
                "python", "-u", "/microbatch.py", "--microbatch", str(microbatch),
                "--until-step", str(until_step), "--receipt",
                "/opt/so101/outputs/entity_smolvla_sft_lr_sweep/benchmark.json", *argv,
            ]
            if not activation_checkpointing:
                command.append("--no-activation-checkpointing")
            if accumulation:
                command.append("--true-gradient-accumulation")
            write(root / "progress.json", {
                "phase": "tuning", "label": label, "microbatch": microbatch,
                "base_step": 48, "until_step": until_step,
                "activation_checkpointing": activation_checkpointing,
                "gradient_accumulation": accumulation,
            })
            with (destination / "runner.log").open("wb") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode not in (0, 86):
                raise RuntimeError(f"{label} failed with exit {result.returncode}; see {destination / 'runner.log'}")
            receipt = json.loads((destination / "benchmark.json").read_text())
            if result.returncode == 86 and receipt["status"] != "out_of_memory":
                raise ValueError("OOM exit did not produce an explicit OOM receipt")
            return receipt

        if accumulation_only:
            candidates = []
            for microbatch in accumulation_microbatches:
                receipt = continuation(
                    f"probe-{microbatch}-accumulation", microbatch, 54, False, True)
                if (candidates and receipt["status"] == "completed"
                        and receipt["sample_fingerprints"] != candidates[0]["sample_fingerprints"]):
                    raise ValueError("Accumulation candidates sampled different examples")
                candidates.append(receipt)
                write(root / "accumulation_candidates.json", candidates)
                if receipt["status"] == "out_of_memory":
                    break
            if len(accumulation_microbatches) == 1:
                write(root / "gradient_accumulation.json", candidates[0])
            feasible = feasible_candidates(candidates)
            fastest = min(feasible, key=lambda item: item["steady_mean_seconds"])
            write(root / "accumulation_selection.json", {
                "microbatch": fastest["microbatch"],
                "steady_mean_seconds": fastest["steady_mean_seconds"],
                "effective_batch": 128,
                "tested": [item["microbatch"] for item in candidates],
                "not_tested": list(accumulation_microbatches[len(candidates):]),
                "criterion": "Fastest five-update post-warmup mean with ten-percent reserved-memory headroom.",
                "larger_candidates": "Not run after the first genuine CUDA OOM.",
            })
            write(root / "progress.json", {
                "phase": "completed", "microbatch": fastest["microbatch"],
                "gradient_accumulation": True,
                "activation_checkpointing": False, "status": "completed",
            })
            return
        if no_checkpointing_only:
            receipt = continuation("probe-16-no-checkpointing", 16, 54, False)
            write(root / "activation_checkpointing.json", receipt)
            write(root / "progress.json", {
                "phase": "completed", "microbatch": 16,
                "activation_checkpointing": False, "status": receipt["status"],
            })
            return

        candidates = []
        for microbatch in (16, 32, 64, 128):
            candidates.append(continuation(f"probe-{microbatch}", microbatch, 54))
            write(root / "candidates.json", candidates)
        selected, comparison_microbatch = select_candidates(candidates)
        write(root / "selection.json", {
            "microbatch": selected, "effective_batch": 128,
            "comparison_microbatch": comparison_microbatch,
            "criterion": "Lowest mean time over five measured updates after one warmup, with ten-percent memory headroom.",
        })
        baseline = continuation("comparison-16", 16, 96)
        tuned = continuation(f"comparison-tuned-{comparison_microbatch}", comparison_microbatch, 96)
        if baseline["status"] != "completed" or tuned["status"] != "completed":
            raise RuntimeError("Matched 48-update comparison did not finish")
        if (baseline["base_sha256"] != tuned["base_sha256"]
                or baseline["sample_fingerprints"] != tuned["sample_fingerprints"]):
            raise ValueError("Comparison checkpoint or sampled examples differ")
        command = [
            "docker", "run", "--rm", "--network", "none", "--user", "0:0",
            "--mount", f"type=bind,src={scripts},dst=/scripts,readonly",
            "--mount", f"type=bind,src={reference},dst=/reference,readonly",
            "--mount", f"type=bind,src={root},dst=/tuning",
            image, "python", "/scripts/compare_gpu_microbatches.py",
            "--baseline", f"/tuning/comparison-16/{directory_name}",
            "--tuned", f"/tuning/comparison-tuned-{comparison_microbatch}/{directory_name}",
            "--reference", "/reference", "--output", "/tuning/comparison.json",
        ]
        subprocess.run(command, check=True)
        write(root / "progress.json", {
            "phase": "completed", "microbatch": selected,
            "comparison_microbatch": comparison_microbatch,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--no-checkpointing-only", action="store_true")
    parser.add_argument("--accumulation-only", action="store_true")
    parser.add_argument("--accumulation-microbatch", type=int, action="append",
                        choices=(16, 24, 32, 48, 64, 96, 128))
    args = parser.parse_args()
    try:
        if args.no_checkpointing_only and args.accumulation_only:
            raise ValueError("Choose one isolated execution-mode benchmark")
        run(args.root, args.image, args.job_id, args.no_checkpointing_only,
            args.accumulation_only, args.accumulation_microbatch or (16,))
    except Exception as error:
        write(args.root / "failure.json", {"error": str(error)})
        raise


if __name__ == "__main__":
    main()
