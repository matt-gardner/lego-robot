"""Restore immutable inputs and continue the selected run on a new GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
from urllib.parse import urlsplit


def execute(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download(artifact, destination):
    if destination.exists() and digest(destination) == artifact["sha256"]:
        return
    uri = urlsplit(artifact["uri"])
    temporary = destination.with_suffix(".download")
    execute(["aws", "s3api", "get-object", "--region", "us-west-2",
             "--expected-bucket-owner", "133961745003",
             "--bucket", uri.netloc, "--key", uri.path.lstrip("/"),
             "--version-id", artifact["version"], str(temporary)],
            stdout=subprocess.DEVNULL)
    if temporary.stat().st_size != artifact["size"] or digest(temporary) != artifact["sha256"]:
        raise ValueError(f"Immutable input checksum/size mismatch: {destination.name}")
    temporary.replace(destination)


def stage(root, bundle, image):
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 30_000_000_000:
        raise ValueError("Insufficient space for frozen inputs and retained checkpoints")
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    manifest = json.loads((bundle / "inputs-manifest.json").read_text())
    for name in ("collection.json", "episodes.sqlite", "data-split.json",
                 "hf-cache.tar", "model.tar"):
        print(json.dumps({"phase": "restoring", "artifact": name}), flush=True)
        download(manifest["artifacts"][name], inputs / name)
    for name, directory in (("hf-cache.tar", "hf_cache"), ("model.tar", "data_model")):
        target = inputs / directory
        marker = target / ".restored"
        if not marker.exists():
            target.mkdir(exist_ok=True)
            with tarfile.open(inputs / name) as archive:
                archive.extractall(target, filter="data")
            marker.touch()
    arm = json.loads((bundle / "arm-manifest.json").read_text())
    artifact = arm["files"]["training_state.pt"]
    download(artifact, inputs / "baseline_384.pt")
    password = execute(["aws", "ecr", "get-login-password", "--region", "us-west-2"],
                       capture_output=True, text=True).stdout
    execute(["docker", "login", "--username", "AWS", "--password-stdin",
             image.split("/")[0]], input=password, text=True)
    execute(["docker", "pull", image])
    output = root / "output"
    if not (output / "continuation.json").exists():
        execute(["docker", "run", "--rm", "--network", "none", "--read-only", "--user", "0:0",
                 "--cpus", "4", "--tmpfs", "/tmp:rw,size=1g",
                 "--mount", f"type=bind,src={bundle / 'source'},dst=/long-source,readonly",
                 "--mount", f"type=bind,src={bundle / 'source/src/so101/train_entity_sim.py'},dst=/opt/so101/src/so101/train_entity_sim.py,readonly",
                 "--mount", f"type=bind,src={root},dst=/long-run",
                 image, "python", "/long-source/scripts/prepare_extended_training.py",
                 "--source", "/long-run/inputs/baseline_384.pt",
                 "--destination", "/long-run/output", "--steps", "16384"])
    receipt = json.loads((output / "continuation.json").read_text())
    requests = json.loads((bundle / "initialization-commands.json").read_text())
    argv = list(next(row for row in requests if row["id"] == "b64-lr3e5")["gpu_argv"])
    for flag, value in (
        ("--output", receipt["output"]), ("--training-state", receipt["output"] + "/training_state.pt"),
        ("--steps", receipt["steps"]), ("--warmup-fraction", receipt["warmup_fraction"]),
    ):
        argv[argv.index(flag) + 1] = str(value)
    mounts = [
        (bundle / "source", "/long-source", True),
        (bundle / "source/src/so101/train_entity_sim.py", "/opt/so101/src/so101/train_entity_sim.py", True),
        (inputs / "collection.json", "/opt/so101/data/simulation/entity_commands_with_difficulty_v1/collection.json", True),
        (inputs / "episodes.sqlite", "/opt/so101/data/episodes.sqlite", True),
        (inputs / "data-split.json", "/opt/so101/outputs/entity_smolvla_trajectory_rl_all_assisted_lr1e4_before_step243_rerun/data_split.json", True),
        (inputs / "hf_cache", "/work/cache/huggingface", True),
        (inputs / "data_model/model", "/opt/so101/data/simulation/model", True),
        (output, "/opt/so101/" + receipt["output"], False),
    ]
    command = ["docker", "run", "--rm", "--network", "none", "--read-only",
               "--user", "0:0", "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g",
               "--gpus", "all", "--name", f"lego-long-{root.name}"]
    for source, target, readonly in mounts:
        command += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
    command += [image, "python", "-u", "-m", "so101.train_entity_sim", *argv,
                "--gradient-accumulation", "--micro-batch-size", "8"]
    (root / "command.json").write_text(json.dumps(command))
    (root / "setup-complete.json").write_text(json.dumps(receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    try:
        if args.run:
            command = json.loads((args.root / "command.json").read_text())
            with (args.root / "trainer.log").open("ab") as stream:
                execute(command, stdout=stream, stderr=subprocess.STDOUT)
        else:
            stage(args.root, args.bundle, args.image)
            if args.start:
                execute(["tmux", "new-session", "-d", "-s", f"train-{args.root.name}",
                         shlex.join(["python3", str(Path(__file__).resolve()),
                                     "--root", str(args.root), "--bundle", str(args.bundle),
                                     "--image", args.image, "--run"])])
    except Exception as error:
        (args.root / "failure.json").write_text(json.dumps({"error": str(error)}))
        raise


if __name__ == "__main__":
    main()
