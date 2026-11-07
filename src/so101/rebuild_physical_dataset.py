"""Resumably regenerate legacy expert scenarios with checked physical contacts, on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import cv2
import mujoco
import numpy as np
import torch

from so101.cloud_checkpoint import atomic_json, sha256
from so101.collect_difficulty_sim import configure_scene
from so101.control_model.simulation_dataset import EntityTrajectoryDataset
from so101.robot_sim import (
    BeamSimulation, BEAM_HALF_HEIGHT, INITIAL_PENETRATION_TOLERANCE, MODEL_COMMIT,
    MODEL_ROOT, OBJECT_NAMES, SCENE_VERSION, TABLE_Z, TARGET_COLORS, restore_episode,
)
from so101.simulation_corpus import CorpusEpisode, expert_bounds, load_corpus
from so101.simulation_playback import model_fingerprint

CONTACT_LIMIT_M = .001
TABLE_CONTACT_LIMIT_M = .0003


def generator_fingerprint(model_root: Path) -> str:
    digest = hashlib.sha256(
        f"{mujoco.__version__}:{np.__version__}:{cv2.__version__}".encode()
    )
    digest.update(model_fingerprint(model_root, SCENE_VERSION).encode())
    for name in ("robot_sim.py", "collect_difficulty_sim.py", "rebuild_physical_dataset.py",
                 "simulation_corpus.py"):
        digest.update(sha256(Path(__file__).with_name(name)).encode())
    digest.update(sha256(Path(__file__).parent / "control_model" / "simulation_dataset.py").encode())
    return digest.hexdigest()


def configure_replacement(sim: BeamSimulation, source: dict, seed: int, original_start: bool) -> dict:
    initial = source["initial"]
    difficulty = source.get("difficulty")
    if difficulty:
        config = configure_scene(sim, source["task"], int(difficulty["level"]), seed)
    else:
        appearance = initial.get("appearance", {})
        sim.reset(
            seed, varied_layout=bool(initial.get("varied_layout", True)),
            real_scene=bool(appearance.get("real_scene", True)), color_targets=True,
        )
        config = {
            "target": source["target"], "destination": source.get("destination"),
            "obstacles": list(source.get("obstacles", [])),
            "constraints": dict(source.get("constraints", {})),
            "perception": dict(source.get("perception", {})),
        }
        if original_start:
            for name in sim.distractors:
                sim.set_body_pose(name, [0., 0., -1.])
            for name in OBJECT_NAMES:
                yaw = float(initial["yaws"][name])
                address = sim.model.joint(f"{name}_free").qposadr[0]
                position = initial["objects"][name]
                sim.data.qpos[address:address + 7] = [
                    *position[:2], TABLE_Z + BEAM_HALF_HEIGHT + .001,
                    math.cos(yaw / 2), 0., 0., math.sin(yaw / 2),
                ]
                sim.yaws[name] = yaw
            arm = sim.decode(np.asarray(initial["arm_state"]))
            sim.data.qpos[sim.qadr] = arm
            sim.data.ctrl[sim.actuators] = arm
            rng = np.random.default_rng(seed)
            for name, position in appearance.get("distractors", {}).items():
                sim.place_prop(name, position, [1., 0., 0., 0.], rng)
            for name, position in appearance.get("destinations", {}).items():
                sim.model.body_pos[sim.model.body(name).id] = [
                    *position[:2], TABLE_Z + .00002,
                ]
        for name in config["obstacles"]:
            if name == "obstacle_box" and source["task"] == "grasp":
                sim.place_grasp_obstacle(config["target"])
            else:
                position = appearance["obstacles"][name]
                sim.model.body_pos[sim.model.body(name).id] = position
        sim.settle_initial_scene()
    labels = initial.get("appearance", {}).get("target_labels", {})
    sim.target_labels.update(labels)
    for name, label in labels.items():
        color = label.split()[0]
        if color in TARGET_COLORS:
            sim.model.geom_rgba[sim.beam_geoms[name], :3] = TARGET_COLORS[color]
    sim.initial["appearance"]["target_labels"] = dict(sim.target_labels)
    entities = [(config["target"], "object")]
    if config["destination"] is not None:
        entities.append((config["destination"], "destination"))
    entities.extend((name, "obstacle") for name in config["obstacles"])
    sim.configure_command(source["task"], entities)
    sim.episode_constraints = config["constraints"]
    sim.hide_unseen_centers = bool(config["perception"].get("hide_unseen_centers", False))
    sim.snapshot_initial_physics()
    sim.frame_start = sim.data.time
    return config


def demonstrate(sim: BeamSimulation, task: str, config: dict) -> dict:
    result = (
        sim.demonstrate_obstacle_grasp(config["target"])
        if task == "grasp" and config["obstacles"] else
        sim.demonstrate_command(task, config["target"], config["destination"])
    )
    if task in ("push", "place"):
        tolerance = config["constraints"].get(
            "destination_tolerance_m", .07 if task == "push" else .04,
        )
        result["success"] = bool(result["success"]) and result["destination_error_m"] <= tolerance
    return result


def validate_episode(root: Path) -> tuple[dict, int]:
    metadata = json.loads((root / "episode.json").read_text())
    if metadata["outcome"] != "success" or metadata["initial"]["scene_version"] != SCENE_VERSION:
        raise ValueError(f"Unqualified corrected episode: {root}")
    count = 0
    with (root / "frames.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if len(row["qpos"]) != 62 or not np.isfinite(row["qpos"]).all():
                raise ValueError(f"Invalid physics snapshot in {root}")
            if (row["action_interval_max_contact_penetration_m"] > CONTACT_LIMIT_M
                    or row["action_interval_max_table_penetration_m"] > TABLE_CONTACT_LIMIT_M):
                raise ValueError(f"Excessive physical penetration in {root}")
            for key in ("command_boxes", "command_visibility", "command_depth_m",
                        "command_depth_valid", "command_roles", "command_entity_ids"):
                if len(row[key]) != len(row["command_roles"]):
                    raise ValueError(f"Inconsistent entity labels in {root}")
            image = (root / row["image"]).resolve()
            if not image.is_relative_to(root.resolve()) or not image.is_file():
                raise ValueError(f"Missing/invalid recorded image in {root}")
            if abs(row["elapsed"] - count / metadata["fps"]) > 1e-7:
                raise ValueError(f"Incorrect pre-action frame timing in {root}")
            count += 1
    if count < 2 or count != metadata["frames"]:
        raise ValueError(f"Incomplete recorded trajectory in {root}")
    return metadata, count


def qualify_consumers(report: dict) -> dict:
    groups = {}
    for path in report["successful_episodes"]:
        metadata = json.loads((Path(path) / "episode.json").read_text())
        groups.setdefault((metadata["task"], metadata["pair_id"]), []).append(path)
    selected = {}
    counts = Counter()
    for (task, _), values in groups.items():
        if len(values) >= 2:
            selected.setdefault(task, values[:2])
            counts[task] += 1
    if set(selected) != set(report["task_counts"]):
        raise ValueError("Corrected dataset is missing paired command groups for one or more tasks")
    with tempfile.TemporaryDirectory(prefix="so101-data-qualification-") as temporary:
        root = Path(temporary)
        paths = [path for values in selected.values() for path in values]
        atomic_json(root / "collection.json", {**report, "successful_episodes": paths})
        dataset = EntityTrajectoryDataset(root, horizon=16, history_steps=4)
        for indices in dataset.episode_sample_indices:
            sample = dataset[indices[0]]
            for value in sample.values():
                if isinstance(value, torch.Tensor) and (
                    value.device.type != "cpu" or not torch.isfinite(value).all()
                ):
                    raise ValueError("Corrected supervised data produced invalid/non-CPU tensors")
        corpus = load_corpus(root)
        return {
            "sft_episode_count": len(dataset.episodes), "sft_frame_count": len(dataset.samples),
            "rl_tasks": {task: len(values) for task, values in corpus.items()},
            "full_dataset_pair_groups_by_task": dict(counts),
        }


def regenerate(source_collection: Path, output: Path, model_root: Path = MODEL_ROOT,
               attempts_per_source: int = 12, limit: int | None = None,
               validation_split: Path | None = None, publish_catalog: bool = False) -> dict:
    if attempts_per_source <= 0 or (limit is not None and limit <= 0):
        raise ValueError("Attempts and optional limit must be positive")
    if source_collection.resolve() == output.resolve():
        raise ValueError("Corrected data must be separate from the historical source")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".rebuild.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot_path = output / "historical" / "collection.json"
        if not snapshot_path.exists():
            atomic_json(snapshot_path, json.loads((source_collection / "collection.json").read_text()))
        original = json.loads(snapshot_path.read_text())
        sources = original["successful_episodes"]
        if len(sources) != len(set(sources)) or not sources:
            raise ValueError("Source catalog must contain distinct successful episodes")
        fingerprint = generator_fingerprint(model_root)
        report_path = output / "collection.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if (report["generator_fingerprint"] != fingerprint
                    or report["source_catalog_sha256"] != sha256(snapshot_path)):
                raise ValueError("Regeneration source/physics changed; use a new output dataset")
        else:
            report = {
                "simulation": True, "status": "building", "scene_version": SCENE_VERSION,
                "source_collection": str(source_collection), "source_catalog_sha256": sha256(snapshot_path),
                "generator_fingerprint": fingerprint, "requested_episodes": len(sources),
                "successful_episodes": [], "replacements": {}, "attempts": [],
                "contact_limit_m": CONTACT_LIMIT_M, "table_contact_limit_m": TABLE_CONTACT_LIMIT_M,
                "initial_penetration_limit_m": INITIAL_PENETRATION_TOLERANCE,
            }
        started = time.monotonic()
        sim = BeamSimulation(model_root)
        sim.contact_penetration_limit_m = CONTACT_LIMIT_M
        sim.table_penetration_limit_m = TABLE_CONTACT_LIMIT_M
        try:
            for index, source_name in enumerate(sources[:limit]):
                source_root = Path(source_name)
                source_hash = sha256(source_root / "episode.json")
                if source_name in report["replacements"]:
                    replacement = Path(report["replacements"][source_name])
                    metadata, _ = validate_episode(replacement)
                    if metadata["provenance"]["source_episode_sha256"] != source_hash:
                        raise ValueError(f"Historical source changed during rebuild: {source_root}")
                    continue
                source = json.loads((source_root / "episode.json").read_text())
                previous = sum(row["source"] == source_name for row in report["attempts"])
                for attempt in range(previous, attempts_per_source):
                    seed = int(source["initial"]["seed"]) + attempt * 1_000_003
                    target_root = output / f"episode_{index:05}" / f"attempt_{attempt:02}"
                    sim.episode_root = None
                    entry = {"source": source_name, "attempt": attempt, "seed": seed}
                    try:
                        config = configure_replacement(sim, source, seed, original_start=attempt == 0)
                        dry = demonstrate(sim, source["task"], config)
                        if not dry["success"]:
                            raise ValueError(f"Corrected expert verification failed: {dry}")
                        initial = sim.initial
                        metadata = {
                            "simulation": True, "model_commit": MODEL_COMMIT,
                            "task": source["task"], **config, "initial": initial,
                            "scene_seed": seed, "fps": sim.fps,
                            "target_label": sim.target_labels[config["target"]],
                            "pair_id": f"physical-v{SCENE_VERSION}:{source.get('pair_id', source_name)}:{seed}",
                            "provenance": {
                                "source": source_name, "source_episode_sha256": source_hash,
                                "generator_fingerprint": fingerprint,
                                "start": "repaired legacy metadata" if attempt == 0 and not source.get("difficulty")
                                         else "new seeded physical scene",
                                "legacy_recovery_prefix": source.get("recovery_prefix")
                                                          or source["initial"].get("recovery_prefix"),
                                "policy_recovery_replayed": False,
                            },
                        }
                        restore_episode(sim, metadata)
                        sim.maximum_contact_penetration_m = 0.
                        sim.maximum_table_penetration_m = 0.
                        sim.frame_start = sim.data.time
                        if not target_root.resolve().is_relative_to(output.resolve()):
                            raise ValueError("Corrected output episode escapes dataset root")
                        target_root.mkdir(parents=True, exist_ok=True)
                        wrist = target_root / "wrist"
                        if not wrist.resolve().is_relative_to(output.resolve()):
                            raise ValueError("Corrected wrist output escapes dataset root")
                        wrist.mkdir(exist_ok=True)
                        sim.overview(target_root / "diagnostic_overview.jpg")
                        if not cv2.imwrite(str(target_root / "initial_wrist.jpg"),
                                           cv2.cvtColor(sim.image(), cv2.COLOR_RGB2BGR)):
                            raise OSError("Cannot record corrected initial wrist image")
                        sim.episode_root = target_root
                        result = demonstrate(sim, source["task"], config)
                        if not result["success"]:
                            raise ValueError(f"Recorded corrected expert verification failed: {result}")
                        metadata.update(
                            outcome="success", frames=len(sim.rows), verification=result,
                            duration=len(sim.rows) / sim.fps,
                            maximum_contact_penetration_m=sim.maximum_contact_penetration_m,
                            maximum_table_penetration_m=sim.maximum_table_penetration_m,
                        )
                        with (target_root / "frames.jsonl").open("w") as stream:
                            for row in sim.rows:
                                stream.write(json.dumps(row, allow_nan=False) + "\n")
                        atomic_json(target_root / "episode.json", metadata)
                        validate_episode(target_root)
                        report["replacements"][source_name] = str(target_root)
                        report["successful_episodes"].append(str(target_root))
                        entry.update(outcome="success", replacement=str(target_root))
                    except (ValueError, RuntimeError) as error:
                        entry.update(outcome="failed", error=str(error))
                    report["attempts"].append(entry)
                    report["wall_seconds"] = report.get("wall_seconds", 0.) + time.monotonic() - started
                    started = time.monotonic()
                    atomic_json(report_path, report)
                    print(json.dumps({**entry, "completed": len(report["replacements"]),
                                      "requested": len(sources)}), flush=True)
                    if entry["outcome"] == "success":
                        break
            report["missing_sources"] = [source for source in sources if source not in report["replacements"]]
            report["status"] = "complete" if not report["missing_sources"] else "incomplete"
            report["task_counts"] = dict(Counter(
                json.loads((Path(path) / "episode.json").read_text())["task"]
                for path in report["successful_episodes"]
            ))
            if report["status"] == "complete":
                frames = 0
                for path in report["successful_episodes"]:
                    _, count = validate_episode(Path(path))
                    frames += count
                report["frames"] = frames
                if validation_split is not None:
                    old_split = json.loads(validation_split.read_text())
                    new_split = dict(old_split)
                    new_split["validation"] = [
                        report["replacements"][source] for source in old_split["validation"]
                    ]
                    if isinstance(old_split["training"], dict):
                        new_split["training"] = {
                            task: [report["replacements"][source] for source in values]
                            for task, values in old_split["training"].items()
                        }
                        new_split["expert_bounds"] = expert_bounds({
                            task: [
                                CorpusEpisode(Path(path), json.loads((Path(path) / "episode.json").read_text()))
                                for path in values
                            ] for task, values in new_split["training"].items()
                        })
                        new_split["replay_counts"] = {
                            task: len(values) for task, values in new_split["training"].items()
                        }
                    else:
                        new_split["training"] = [
                            report["replacements"][source] for source in old_split["training"]
                        ]
                    new_split["provenance"] = {
                        "source_split": str(validation_split), "source_split_sha256": sha256(validation_split),
                        "scene_version": SCENE_VERSION,
                        "historical_metrics_comparable": False,
                    }
                    atomic_json(output / "data_split.json", new_split)
                if publish_catalog:
                    report["consumer_qualification"] = qualify_consumers(report)
            atomic_json(report_path, report)
            if publish_catalog:
                if report["status"] != "complete":
                    raise ValueError("Cannot publish an incomplete corrected dataset")
                current = source_collection / "collection.json"
                if json.loads(current.read_text()) != original:
                    existing = json.loads(current.read_text())
                    if (existing.get("generator_fingerprint") != fingerprint
                            or existing.get("corrected_dataset") != str(output)):
                        raise ValueError("Canonical catalog changed concurrently; refusing replacement")
                atomic_json(current, {**report, "historical_catalog": str(snapshot_path),
                                      "corrected_dataset": str(output)})
            return report
        finally:
            sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--attempts-per-source", type=int, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validation-split", type=Path)
    parser.add_argument("--publish-catalog", action="store_true")
    args = parser.parse_args()
    report = regenerate(args.source_collection, args.output, args.model_root,
                        args.attempts_per_source, args.limit, args.validation_split, args.publish_catalog)
    if report["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
