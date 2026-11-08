"""Start the next approved fresh cell on the existing GPU without terminating it."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess


def write(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o644)
    temporary.replace(path)


def run(root, image, job_id):
    inputs = Path("/home/ubuntu/lego-sweep/inputs")
    original_output = Path("/home/ubuntu/lego-sweep/output")
    requests = json.loads((inputs / "initialization-commands.json").read_text())
    matches = [request for request in requests if request["id"] == "b64-lr3e5"]
    if len(matches) != 1:
        raise ValueError("Exactly one frozen B64/LR3e-5 request is required")
    argv = matches[0]["gpu_argv"]
    name = Path(argv[argv.index("--output") + 1]).name
    if (argv[argv.index("--batch-size") + 1] != "64"
            or float(argv[argv.index("--learning-rate") + 1]) != 3e-5
            or argv[argv.index("--steps") + 1] != "384"):
        raise ValueError("Next cell must preserve the approved 24,576-sample recipe")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "cell.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = root / "output"
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        state = directory / "training_state.pt"
        if not state.exists():
            shutil.copyfile(inputs / "initial_b64_lr3e5.pt", state)
        logical = original_output / name
        if logical.is_symlink():
            if logical.resolve() != directory.resolve():
                raise ValueError("Existing dashboard path points at another cell")
        elif logical.exists():
            raise ValueError("Existing cell directory must not be overwritten")
        else:
            logical.symlink_to(directory, target_is_directory=True)
        mounts = [
            (inputs / "collection.json", "/opt/so101/data/simulation/entity_commands_with_difficulty_v1/collection.json"),
            (inputs / "episodes.sqlite", "/opt/so101/data/episodes.sqlite"),
            (inputs / "data-split.json", "/opt/so101/outputs/entity_smolvla_trajectory_rl_all_assisted_lr1e4_before_step243_rerun/data_split.json"),
            (inputs / "hf_cache", "/work/cache/huggingface"),
            (inputs / "data_model/model", "/opt/so101/data/simulation/model"),
            (Path("/home/ubuntu/lego-sweep/microbatch-scripts/train_entity_sim.py"),
             "/opt/so101/src/so101/train_entity_sim.py"),
        ]
        command = ["docker", "run", "--rm", "--network", "none", "--read-only",
                   "--user", "0:0", "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g"]
        for source, target in mounts:
            command += ["--mount", f"type=bind,src={source},dst={target},readonly"]
        command += [
            "--mount", f"type=bind,src={output},dst=/opt/so101/outputs/entity_smolvla_sft_lr_sweep",
            "--gpus", "all", "--name", f"lego-sweep-{job_id}-b64-lr3e5",
            image, "python", "-u", "-m", "so101.train_entity_sim", *argv,
            "--gradient-accumulation", "--micro-batch-size", "8",
        ]
        write(root / "progress.json", {
            "phase": "training", "arm": "b64-lr3e5", "job_id": job_id,
            "batch_size": 64, "learning_rate": 3e-5, "micro_batch_size": 8,
            "samples": 24576, "output": str(directory),
            "gradient_accumulation": True, "activation_checkpointing": False,
        })
        with (root / "trainer.log").open("ab") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        write(root / "progress.json", {
            "phase": "training-complete-awaiting-export",
            "arm": "b64-lr3e5", "job_id": job_id, "output": str(directory),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    try:
        run(args.root, args.image, args.job_id)
    except Exception as error:
        args.root.mkdir(parents=True, exist_ok=True)
        write(args.root / "failure.json", {"error": str(error)})
        raise


if __name__ == "__main__":
    main()
