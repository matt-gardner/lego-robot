"""Collect compact VIEW references; policy difficulty remains separately measured."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

import mujoco
import numpy as np

from so101.robot_sim import BeamSimulation, restore_episode
from so101.simulation_corpus import load_catalog


ADMISSION_BYTES = 10 * 1024**3
MAX_OUTPUT_BYTES = 64 * 1024**2


class ReplayStateMismatch(ValueError):
    """A finite candidate trajectory fails strict independent state replay."""


def scene_family(metadata: dict) -> str:
    return hashlib.sha256(json.dumps(
        [metadata["initial"]["seed"], metadata["initial"].get("scene_version", 1)],
    ).encode()).hexdigest()


def spatial_recipe(index: int) -> dict:
    """Predeclared geometry strata, independent of policy/development outcomes."""
    return {
        "target_xy": [.16 + .04 * (index % 4),
                      (-1 if index % 2 else 1) * (.10 + .035 * (index // 4 % 3))],
        "target_yaw": (-1 if index // 2 % 2 else 1) * .9,
        "joint_offsets": [
            (-1 if index % 2 else 1) * .65,
            (-1 if index // 4 % 2 else 1) * .30,
            0., 0., (-1 if index // 8 % 2 else 1) * .55, 0.,
        ],
    }


def balanced_schedule(sources: list, recipe_indices: list[int],
                      families_per_split: int) -> list:
    if (not recipe_indices or len(set(recipe_indices)) != len(recipe_indices)
            or any(index < 0 or index >= 64 for index in recipe_indices)
            or families_per_split < 1):
        raise ValueError("Require unique recipe indices in 0..63 and positive family count")
    families = {}
    for source, metadata in sorted(sources, key=lambda item: str(item[0])):
        families.setdefault(scene_family(metadata), (source, metadata))
    selected = []
    for development in (False, True):
        eligible = [
            families[family] for family in sorted(families)
            if (int(family[:8], 16) % 5 == 0) == development
        ]
        if len(eligible) < families_per_split:
            raise ValueError("Insufficient fresh families for the predeclared balanced experiment")
        selected.extend(eligible[:families_per_split])
    return [(source, metadata, recipe) for source, metadata in selected
            for recipe in recipe_indices]


def excluded_families(manifests: list[Path]) -> set[str]:
    return {
        entry["scene_family_id"] for path in manifests
        for entry in json.loads(path.read_text())["episodes"]
    }


class ReferenceSimulation(BeamSimulation):
    """Keep the legacy image writer disabled while capturing numeric actions."""

    def step(self, target: np.ndarray, phase: str):
        if self.episode_root is not None:
            raise RuntimeError("Compact reference simulation must not enable the image writer")
        action = self.encode(target).astype(np.float64)
        # Execute exactly the serialized control path, including float32 encoding.
        target = self.decode(action)
        row = {
            "index": len(self.rows), "elapsed": float(self.data.time - self.frame_start),
            "state": self.state().tolist(), "action": action.tolist(),
            "phase": phase,
            **{name: getattr(self.data, name).tolist()
               for name in ("qpos", "qvel", "ctrl", "act")},
        }
        super().step(target, phase)
        self.rows.append(row)


def check_storage(path: Path) -> None:
    if shutil.disk_usage(path).free < ADMISSION_BYTES:
        raise RuntimeError("VIEW collection requires 10GiB available; storage guard unchanged")


def activate_render_context(sim: BeamSimulation) -> None:
    renderer = sim.renderer
    if renderer is None or renderer._gl_context is None or renderer._mjr_context is None:
        raise ValueError("VIEW qualification requires a live renderer context")
    # Match the trainer's context/FBO activation without importing its Torch stack.
    renderer._gl_context.make_current()
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, renderer._mjr_context)


def view_metrics(sim: BeamSimulation, target: str) -> dict:
    activate_render_context(sim)
    box = sim.target_oriented_box(target)
    return {
        "mask_pixels": int(np.count_nonzero(sim.entity_mask(target))),
        "center": list(box[:2]),
        "center_error": float(np.linalg.norm(np.asarray(box[:2]) - .5)),
        "visible": bool(box[5]),
    }


def meets_view(metrics: dict, executed: int) -> bool:
    return executed >= 30 and metrics["visible"] and metrics["center_error"] < .3


def replay_reference(sim: BeamSimulation, metadata: dict, rows: list[dict]) -> dict:
    activate_render_context(sim)
    target, _, _ = restore_episode(sim, metadata)
    sim.episode_root = None
    stable = 0
    success = False
    maximum_state_error = 0.
    for executed, row in enumerate(rows, 1):
        error = float(np.max(np.abs(sim.data.qpos - np.asarray(row["qpos"]))))
        maximum_state_error = max(maximum_state_error, error)
        if not np.isfinite(error):
            raise ValueError(f"Nonfinite independent replay state at frame {executed}")
        if error > 1e-7:
            raise ReplayStateMismatch(
                f"Independent replay state mismatch at frame {executed}: {error}"
            )
        sim.step(sim.decode(np.asarray(row["action"], dtype=float)), row["phase"])
        stable = stable + 1 if meets_view(view_metrics(sim, target), executed) else 0
        success |= stable >= sim.fps
    return {"success": bool(success), "maximum_qpos_error": maximum_state_error,
            "frames": len(rows), "final": view_metrics(sim, target)}


def collect(collection: Path, model_root: Path, output: Path, *,
            training_sources: Path | None = None, attempts: int = 16,
            profile: str = "base-v1", exclude_manifests: list[Path] | None = None,
            recipe_indices: list[int] | None = None, families_per_split: int = 4) -> dict:
    if not 2 <= attempts <= 64:
        raise ValueError("Require 2..64 bounded attempts")
    check_storage(output.parent)
    if training_sources is None:
        raise ValueError("An explicit frozen-split training-source allowlist is required")
    allowed = json.loads(training_sources.read_text())
    if not isinstance(allowed, list) or not allowed or not all(
        isinstance(name, str) and Path(name).is_absolute() for name in allowed
    ):
        raise ValueError("Training sources must be a nonempty JSON list of absolute episode paths")
    allowed = {Path(name).resolve() for name in allowed}
    if profile not in ("base-v1", "spatial-v2"):
        raise ValueError("Unknown VIEW candidate profile")
    if profile == "spatial-v2" and not exclude_manifests:
        raise ValueError("Spatial candidates require the previous manifest for family exclusion")
    excluded = excluded_families(exclude_manifests or [])
    report = load_catalog(collection)
    sources = []
    for name in report["successful_episodes"]:
        source = Path(name)
        metadata = json.loads((source / "episode.json").read_text())
        if (metadata["task"] == "view" and source.resolve() in allowed
                and scene_family(metadata) not in excluded):
            sources.append((source, metadata))
    if not sources:
        raise ValueError("Source collection has no successful VIEW episodes")
    if any(metadata["initial"].get("renderer") is not None for _, metadata in sources):
        raise ValueError("Collector currently requires the existing default renderer identity")
    if recipe_indices is not None:
        if profile != "spatial-v2":
            raise ValueError("Explicit recipes require spatial-v2")
        schedule = balanced_schedule(sources, recipe_indices, families_per_split)
        if len(schedule) != attempts:
            raise ValueError("Attempts must equal recipes times families-per-split times two")
    else:
        schedule = [(*sources[index % len(sources)], index) for index in range(attempts)]
    output.mkdir(exist_ok=False)
    manifest = {
        "schema_version": 1, "kind": "harder-view-references",
        "status": "building", "task": "view", "policy_difficulty_measured": False,
        "sft_qualified": False, "episodes": [], "attempts": [],
        "candidate_profile": profile, "excluded_families": sorted(excluded),
        "schedule": [
            {"source_episode": str(source.resolve()), "scene_family_id": scene_family(metadata),
             "recipe_index": recipe}
            for source, metadata, recipe in schedule
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    sim = ReferenceSimulation(model_root, render=True)
    replay = None
    try:
        replay = BeamSimulation(model_root, render=True)
        # Render only transient segmentation for the existing goal, never raster files.
        sim.record_images = False
        replay.record_images = False
        for index in range(attempts):
            check_storage(output)
            source, original, recipe_index = schedule[index]
            metadata = copy.deepcopy(original)
            activate_render_context(sim)
            target, _, _ = restore_episode(sim, metadata)
            recipe = spatial_recipe(recipe_index) if profile == "spatial-v2" else {
                "joint_offsets": [
                    (-1 if index % 2 else 1) * (.35 + .15 * (index // 2 % 4)),
                    0., 0., 0., 0., 0.,
                ],
            }
            arm = sim.data.qpos[sim.qadr] + np.asarray(recipe["joint_offsets"])
            if np.any(arm < sim.limits[:, 0] + .01) or np.any(
                arm > sim.limits[:, 1] - .01
            ):
                manifest["attempts"].append({"index": index, "reason": "joint_limit"})
                continue
            sim.data.qpos[sim.qadr] = arm
            if profile == "spatial-v2":
                joint = sim.model.joint(f"{target}_free")
                address = int(joint.qposadr[0])
                yaw = recipe["target_yaw"]
                sim.data.qpos[address:address + 2] = recipe["target_xy"]
                sim.data.qpos[address + 3:address + 7] = [
                    math.cos(yaw / 2), 0., 0., math.sin(yaw / 2),
                ]
                sim.data.qvel[int(joint.dofadr[0]):int(joint.dofadr[0]) + 6] = 0.
                sim.yaws[target] = yaw
            sim.data.qvel[sim.dofs] = 0.
            sim.data.ctrl[sim.actuators] = sim.data.qpos[sim.qadr]
            mujoco.mj_forward(sim.model, sim.data)
            if sim.initial_penetrations():
                manifest["attempts"].append({"index": index, "reason": "initial_penetration"})
                continue
            sim.initial = copy.deepcopy(metadata["initial"])
            sim.snapshot_initial_physics()
            metadata["initial"] = copy.deepcopy(sim.initial)
            initial = view_metrics(sim, target)
            if meets_view(initial, 30):
                manifest["attempts"].append({"index": index, "reason": "already_in_goal"})
                continue
            sim.rows = []
            sim.frame_start = sim.data.time
            episode = output / f"candidate_{index:03}"
            sim.episode_root = None
            try:
                result = sim.demonstrate_view(target)
            except RuntimeError as error:
                if not str(error).startswith("No vertical IK solution for "):
                    raise
                manifest["attempts"].append({
                    "index": index, "reason": "unreachable_expert_vantage",
                    "detail": str(error), "candidate_recipe": recipe,
                })
                continue
            if not result["success"]:
                manifest["attempts"].append({"index": index, "reason": "expert_failed"})
                continue
            try:
                verification = replay_reference(replay, metadata, sim.rows)
            except ReplayStateMismatch as error:
                manifest["attempts"].append({
                    "index": index, "reason": "finite_replay_state_mismatch",
                    "detail": str(error),
                })
                continue
            if not verification["success"]:
                manifest["attempts"].append({"index": index, "reason": "replay_goal_failed"})
                continue
            family = scene_family(metadata)
            split = "development" if int(family[:8], 16) % 5 == 0 else "train"
            metadata.update(
                outcome="success", task="view", frames=len(sim.rows), fps=sim.fps,
                initial_view_metrics=initial, reference_replay=verification,
                source_episode=str(source.resolve()), scene_family_id=family, split=split,
                sft_qualified=False,
                candidate_profile=profile, candidate_recipe=recipe,
                candidate_recipe_index=recipe_index,
            )
            frame_bytes = "".join(json.dumps(row, allow_nan=False) + "\n"
                                  for row in sim.rows).encode()
            episode_bytes = json.dumps(metadata, allow_nan=False, sort_keys=True).encode()
            used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            if used + len(frame_bytes) + len(episode_bytes) > MAX_OUTPUT_BYTES:
                raise RuntimeError("Compact VIEW output exceeds 64MiB budget")
            check_storage(output)
            episode.mkdir()
            (episode / "frames.jsonl").write_bytes(frame_bytes)
            (episode / "episode.json").write_bytes(episode_bytes)
            manifest["episodes"].append({
                "path": str(episode.resolve()), "split": split, "scene_family_id": family,
                "sha256": {"episode.json": hashlib.sha256(episode_bytes).hexdigest(),
                           "frames.jsonl": hashlib.sha256(frame_bytes).hexdigest()},
            })
        manifest["status"] = "complete" if manifest["episodes"] else "no_qualified_references"
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
        if not manifest["episodes"]:
            raise RuntimeError(f"No qualified VIEW references; inspect {output / 'manifest.json'}")
        return manifest
    finally:
        sim.close()
        if replay is not None:
            replay.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--training-sources", type=Path, required=True,
                        help="JSON absolute episode-path list from the frozen training split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=16)
    parser.add_argument("--profile", choices=("base-v1", "spatial-v2"), default="base-v1")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--recipe-indices", type=int, nargs="+")
    parser.add_argument("--families-per-split", type=int, default=4)
    args = parser.parse_args()
    collect(args.collection, args.model_root, args.output,
            training_sources=args.training_sources, attempts=args.attempts,
            profile=args.profile, exclude_manifests=args.exclude_manifest,
            recipe_indices=args.recipe_indices, families_per_split=args.families_per_split)


if __name__ == "__main__":
    main()
