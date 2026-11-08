"""Record a paired catalog expert attempt; never overwrite historical datasets."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import shutil

import cv2
import mujoco
import numpy as np

from so101.catalog_sim import CatalogSimulation
from so101.cloud_checkpoint import atomic_json
from so101.lego_curriculum import load_catalog, prepare_scene
from so101.lego_scene import capture_scene_state
from so101.renderer_appearance import DEFAULT_PROFILE, RendererAppearance
from so101.renderer_appearance_v2 import DEFAULT_PROFILE_V2, WristAppearanceV2
from so101.renderer_appearance_v3 import DEFAULT_PROFILE_V3, WristAppearanceV3
from so101.renderer_appearance_v4 import WristAppearanceV4
from so101.renderer_appearance_v6 import WristAppearanceV6
from so101.simulation_corpus import validate_clean_expert


LOGGER = logging.getLogger(__name__)


def load_appearance(path: Path) -> RendererAppearance:
    profile_id = json.loads(path.read_bytes()).get("profile_id")
    implementations = {
        "wrist-soft-v1": RendererAppearance,
        "wrist-soft-v2": WristAppearanceV2,
        "wrist-soft-v3": WristAppearanceV3,
        "wrist-soft-v4": WristAppearanceV4,
        "wrist-cad-v6": WristAppearanceV6,
    }
    if profile_id not in implementations:
        raise ValueError(f"Unsupported catalog recording renderer profile: {profile_id!r}")
    return implementations[profile_id](path)


def validate_recording(root: Path, metadata: dict, rows: list[dict], nq: int) -> None:
    validate_clean_expert(metadata, rows)
    fps = metadata.get("fps")
    if type(fps) not in (int, float) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid catalog recording FPS")
    if len(rows) < 2:
        raise ValueError("Catalog recording contains fewer than two frames")
    identities = rows[0].get("command_entity_ids")
    roles = rows[0].get("command_roles")
    if not identities or not roles or len(identities) != len(roles):
        raise ValueError("Missing catalog command identities or roles")
    for index, row in enumerate(rows):
        if row.get("command_entity_ids") != identities or row.get("command_roles") != roles:
            raise ValueError("Command identities or roles changed within recording")
        for key, count in (("state", 6), ("action", 6), ("qpos", nq)):
            values = np.asarray(row.get(key), dtype=float)
            if values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid catalog recording {key}")
        if (not np.isfinite(row["elapsed"])
                or abs(row["elapsed"] - index / metadata["fps"]) > 1e-7):
            raise ValueError("Catalog recording has incorrect pre-action timing")
        for key in ("command_boxes", "command_visibility", "command_depth_m",
                    "command_depth_valid"):
            values = np.asarray(row.get(key), dtype=float)
            shape = (len(identities), 5) if key == "command_boxes" else (len(identities),)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid catalog recording {key}")
        for key, limit in (("action_interval_max_contact_penetration_m", .001),
                           ("action_interval_max_table_penetration_m", .0003)):
            depth = row.get(key, float("inf"))
            if not np.isfinite(depth) or not 0 <= depth <= limit:
                raise ValueError(f"Invalid or excessive catalog recording {key}")
        for key in ("image", "external_image"):
            name = row.get(key)
            if not isinstance(name, str):
                raise ValueError(f"Missing catalog recording {key}")
            path = root / name
            if (Path(name).is_absolute() or ".." in Path(name).parts
                    or not path.resolve().is_relative_to(root.resolve()) or path.is_symlink()):
                raise ValueError("Catalog image escapes episode root")
            image = cv2.imread(str(path))
            if image is None or image.shape != (240, 320, 3):
                raise ValueError(f"Missing or malformed catalog image: {path}")
            if np.ptp(image) == 0:
                raise ValueError(f"Uniform catalog image: {path}")


def record_pair(sim: CatalogSimulation, output: Path) -> dict:
    """Accept a scene pair only when both independently recorded commands pass."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "model.xml").write_text(sim.xml)
    mujoco.mj_saveModel(sim.model, str(output / "model.mjb"), None)
    atomic_json(output / "scene_plan.json", sim.plan)
    atomic_json(output / "settled_state.json", sim.settled_snapshot)
    fingerprint = hashlib.sha256()
    source = output / "source"
    source.mkdir()
    source_hashes = {}
    if sim.catalog.manifest_path is None:
        raise ValueError("Recording requires the original parts catalog manifest")
    catalog_bytes = sim.catalog.manifest_path.read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != sim.catalog.manifest_sha256:
        raise ValueError("Parts catalog manifest changed before recording")
    (source / "parts_catalog.json").write_bytes(catalog_bytes)
    source_hashes["parts_catalog.json"] = sim.catalog.manifest_sha256
    for name in ("catalog_sim.py", "lego_scene.py", "lego_curriculum.py",
                 "robot_sim.py", "collect_catalog_sim.py", "simulation_corpus.py",
                 "scene_metrics.py"):
        content = Path(__file__).with_name(name).read_bytes()
        fingerprint.update(content)
        (source / name).write_bytes(content)
        source_hashes[name] = hashlib.sha256(content).hexdigest()
    if sim.appearance is not None:
        shutil.copyfile(sim.appearance.path, source / "renderer_profile.json")
        source_hashes["renderer_profile.json"] = hashlib.sha256(
            (source / "renderer_profile.json").read_bytes()
        ).hexdigest()
        profiles = source / "renderer_profiles"
        profiles.mkdir()
        shutil.copyfile(sim.appearance.path, profiles / sim.appearance.path.name)
        source_hashes[f"renderer_profiles/{sim.appearance.path.name}"] = source_hashes[
            "renderer_profile.json"
        ]
        for key, parent in (("parent_profile_sha256", DEFAULT_PROFILE_V2),
                            ("parent_v3_profile_sha256", DEFAULT_PROFILE_V3)):
            parent_hash = sim.appearance.identity().get(key)
            if parent_hash is not None:
                parent_bytes = parent.read_bytes()
                if hashlib.sha256(parent_bytes).hexdigest() != parent_hash:
                    raise ValueError("Renderer parent profile differs from recorded identity")
                (profiles / parent.name).write_bytes(parent_bytes)
                source_hashes[f"renderer_profiles/{parent.name}"] = parent_hash
        for kind in type(sim.appearance).__mro__:
            if kind is object:
                continue
            path = Path(inspect.getfile(kind))
            shutil.copyfile(path, source / path.name)
            source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(source / "hashes.json", source_hashes)
    archive_hashes = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ("model.xml", "model.mjb", "scene_plan.json", "settled_state.json")
    }
    report = {
        "schema_version": 1, "generator": "catalog-curriculum-v1", "status": "building",
        "scene_version": 4, "scene_id": sim.plan["scene_id"],
        "catalog_id": sim.catalog.catalog_id, "catalog_sha256": sim.catalog.manifest_sha256,
        "generator_sha256": fingerprint.hexdigest(),
        "renderer": sim.initial.get("renderer"),
        "scene_archive_sha256": archive_hashes,
        "tags": sim.plan["tags"], "attempts": [], "successful_episodes": [],
    }
    report["release_id"] = hashlib.sha256(json.dumps(
        [report["scene_id"], report["generator_sha256"], report["renderer"], archive_hashes],
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    if "scene_family_id" in sim.plan or "split" in sim.plan:
        if not sim.plan.get("scene_family_id") or sim.plan.get("split") not in (
            "train", "development", "sealed-test"
        ):
            raise ValueError("Invalid reserved curriculum family membership")
        report.update(scene_family_id=sim.plan["scene_family_id"], split=sim.plan["split"])
    atomic_json(output / "collection.json", report)
    candidates = []
    for index, command in enumerate(sim.plan["commands"]):
        sim.restore_initial()
        root = output / f"command_{index}"
        (root / "wrist").mkdir(parents=True)
        (root / "external").mkdir()
        for name in archive_hashes:
            os.link(output / name, root / name)
        shutil.copytree(source, root / "source", copy_function=os.link)
        sim.episode_root = root
        metadata = {
            "simulation": True, "task": command["task"],
            "pair_id": sim.plan["pair_id"], "scene_seed": sim.plan["seed"],
            "example_id": command["command_id"], "command_id": command["command_id"],
            "release_id": report["release_id"],
            "scene_id": sim.plan["scene_id"], "initial": sim.initial,
            "target": command["object_instance_id"],
            "target_label": sim.target_labels[command["object_instance_id"]],
            "destination": command["destination_id"], "fps": sim.fps,
            "generator_sha256": report["generator_sha256"], "tags": report["tags"],
            "scene_model": "model.xml", "compiled_scene_model": "model.mjb",
            "scene_state": "settled_state.json", "scene_archive_sha256": archive_hashes,
            "simulation_backend": "catalog-v1",
        }
        if "scene_family_id" in report:
            metadata.update(scene_family_id=report["scene_family_id"], split=report["split"])
        try:
            sim.observe_recovery_start(
                command["object_instance_id"], command["task"], command["destination_id"],
            )
            result = sim.demonstrate_scene_command(command)
            metadata["outcome"] = "success" if result["success"] else "failed"
            metadata["verification"] = result
            if result["success"]:
                validate_recording(root, metadata, sim.rows, sim.model.nq)
        except (ValueError, RuntimeError) as error:
            LOGGER.error("Catalog command %s failed: %s", command["command_id"], error)
            metadata["outcome"] = "failed"
            metadata["verification"] = {"success": False, "reason": str(error)}
        metadata.update(frames=len(sim.rows), duration=len(sim.rows) / sim.fps)
        if metadata["outcome"] == "failed":
            try:
                state = capture_scene_state(sim.model, sim.data, sim.xml, sim.plan,
                                            status="failed-attempt")
                state["parent_example_id"] = command["command_id"]
                atomic_json(root / "failure_state.json", state)
                metadata["failure_state"] = "failure_state.json"
            except ValueError as error:
                LOGGER.error("No usable recovery state for %s: %s", command["command_id"], error)
                metadata["recovery_state_error"] = str(error)
        with (root / "frames.jsonl").open("w") as stream:
            for row in sim.rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        if metadata["outcome"] == "success":
            metadata["generated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(root / "episode.json", metadata)
        if metadata["outcome"] == "success":
            candidates.append(str(root.resolve()))
        report["attempts"].append({
            "source": str(root.resolve()), "example_id": command["command_id"],
            "outcome": metadata["outcome"], "verification": metadata["verification"],
        })
        atomic_json(output / "collection.json", report)
    if len(candidates) == 2:
        report["successful_episodes"] = candidates
        report["status"] = "complete"
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
    else:
        report["status"] = "incomplete"
    atomic_json(output / "collection.json", report)
    sim.episode_root = None
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--category", choices=("separated", "cluster", "overlap", "pile"), required=True)
    parser.add_argument("--task", choices=("grasp", "view", "push", "place"), required=True)
    parser.add_argument("--object-count", type=int, required=True)
    parser.add_argument("--appearance", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--part-id", action="append",
                        help="Sample only these verified IDs; repeat to select multiple parts.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    catalog = load_catalog(args.catalog)
    if args.part_id:
        requested = set(args.part_id)
        available = {part.part_id for part in catalog.parts}
        if requested - available:
            parser.error(f"Requested parts are not verified: {sorted(requested - available)}")
        catalog = replace(catalog, parts=tuple(
            part for part in catalog.parts if part.part_id in requested
        ))
    plan = prepare_scene(catalog, seed=args.seed, sequence=args.sequence,
                         category=args.category, task=args.task, object_count=args.object_count)
    appearance = load_appearance(args.appearance) if args.appearance else None
    sim = CatalogSimulation(catalog, plan, appearance=appearance)
    try:
        report = record_pair(sim, args.output)
    finally:
        sim.close()
    print(json.dumps(report, allow_nan=False))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
