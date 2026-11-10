"""Appearance-only CPU rerenders of immutable catalog episode archives."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from so101.cloud_checkpoint import atomic_json
from so101.episode_storage import EpisodeStorage, consolidate
from so101.pathtrace_pile import DEFAULT_PROFILE, file_hash, linear_rgb
from so101.robot_sim import BeamSimulation, MODEL_ROOT, scene_xml
from so101.simulation_corpus import validate_clean_expert


LOGGER = logging.getLogger(__name__)


def implementation_inputs() -> dict:
    directory = Path(__file__).parent
    return {
        str(directory / name): file_hash(directory / name)
        for name in ("pathtrace_recordings.py", "pathtrace_pile.py", "robot_sim.py",
                     "robot_control.py", "episode_storage.py", "simulation_corpus.py",
                     "control_model/dataset.py")
    }


def eligible(metadata: dict, since: datetime) -> bool:
    stamp = metadata.get("generated_at")
    if not stamp:
        raise ValueError("Qualified episode has no generated_at date")
    generated = datetime.fromisoformat(stamp)
    if generated.tzinfo is None or since.tzinfo is None:
        raise ValueError("Eligibility dates must include time zones")
    return generated >= since


def inventory(root: Path, since: datetime) -> dict:
    releases = {}
    excluded = []
    for qualification in sorted(root.rglob("qualification.json")):
        archive = qualification.parent / "episodes.sqlite"
        if not archive.is_file():
            continue
        with EpisodeStorage(archive) as storage:
            catalog = storage.catalog
            if catalog.get("status") != "complete":
                continue
            selected = []
            for source in catalog["successful_episodes"]:
                metadata = json.loads(storage.read(Path(source), "episode.json"))
                if metadata["outcome"] != "success":
                    raise ValueError(f"Qualified archive contains failed episode: {archive}")
                if not metadata.get("generated_at"):
                    excluded.append({"archive": str(archive), "reason": "Unknown episode date"})
                    selected = []
                    break
                if eligible(metadata, since):
                    rows = storage.read(Path(source), "frames.jsonl")
                    selected.append({
                        "source": source, "example_id": metadata["example_id"],
                        "frames_sha256": hashlib.sha256(rows).hexdigest(),
                        "frames": len(rows.splitlines()), "task": metadata["task"],
                        "generated_at": metadata["generated_at"],
                    })
            if not selected or len(selected) != len(catalog["successful_episodes"]):
                continue
            identity = catalog["release_id"]
            signature = [(e["example_id"], e["frames_sha256"]) for e in selected]
            if identity in releases:
                if releases[identity]["signature"] != signature:
                    raise ValueError(f"Conflicting duplicate release: {identity}")
                continue
            releases[identity] = {
                "archive": str(archive), "release_id": identity,
                "split": catalog.get("split"), "episodes": selected,
                "signature": signature,
                "expected_archive_sha256": json.loads(qualification.read_text()).get("storage_sha256"),
            }
    examples = set()
    for release in releases.values():
        for episode in release["episodes"]:
            examples.add((release["release_id"], episode["example_id"]))
    return {
        "since": since.isoformat(), "releases": list(releases.values()),
        "episodes": len(examples),
        "frames": sum(e["frames"] for r in releases.values() for e in r["episodes"]),
        "excluded": excluded,
        "scope": "Derived appearance of qualified source archives; original splits and audits retained.",
    }


def cad_render_xml(xml: str, original_robot_xml: str) -> str:
    root, original = ET.fromstring(xml), ET.fromstring(original_robot_xml)
    for body, material in (
        ("gripper", "wrist_roll_follower_so101_v1_material"),
        ("moving_jaw_so101_v1", "moving_jaw_so101_v1_material"),
    ):
        selector = f".//body[@name='{body}']/geom[@material='{material}']"
        destination, source = root.find(selector), original.find(selector)
        if destination is None or source is None or source.get("class") != "visual":
            raise ValueError(f"Missing original CAD visual jaw: {body}")
        destination.attrib.clear()
        destination.attrib.update(source.attrib)
    return ET.tostring(root, encoding="unicode")


def restore_pose(model, data, row: dict) -> None:
    pose = np.asarray(row["qpos"], dtype=float)
    if pose.shape != (model.nq,) or not np.isfinite(pose).all():
        raise ValueError("Recorded pose does not match render model")
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    if not np.array_equal(data.qpos, pose):
        raise RuntimeError("Forward rendering changed recorded pose")


def visible_geoms(model) -> list[int]:
    groups = mujoco.MjvOption().geomgroup
    return [
        geom for geom in range(model.ngeom)
        if groups[model.geom_group[geom]] and model.geom_rgba[geom, 3] > 0
    ]


class FrameRenderer:
    def __init__(self, model, profile: dict, *, variant: str = "llvm_ad_rgb"):
        # Optional heavy backend, isolated from normal recorder imports.
        import drjit as dr
        import mitsuba as mi

        if variant not in ("llvm_ad_rgb", "cuda_ad_rgb"):
            raise ValueError(f"Unsupported pathtrace variant: {variant}")
        mi.set_variant(variant)
        if variant == "llvm_ad_rgb":
            dr.set_thread_count(profile["threads"])
        self.variant = variant
        self.mi, self.model, self.profile = mi, model, profile
        self.visible = visible_geoms(model)
        self.meshes = {}
        for geom in self.visible:
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = int(model.geom_dataid[geom])
            va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
            na = int(model.mesh_normaladr[mesh])
            faces = model.mesh_face[fa:fa + fn]
            normals = model.mesh_facenormal[fa:fa + fn]
            self.meshes[geom] = (
                model.mesh_vert[va:va + vn][faces].reshape(-1, 3).astype(float),
                model.mesh_normal[na + normals].reshape(-1, 3).astype(float),
            )

    def render(self, data, frame: int) -> np.ndarray:
        mi, model, profile = self.mi, self.model, self.profile
        if mi.variant() != self.variant:
            raise RuntimeError("Pathtrace backend changed while a renderer was live")
        camera = model.camera("wrist_cam").id
        rotation = data.cam_xmat[camera].reshape(3, 3)
        origin = data.cam_xpos[camera]
        transform = mi.ScalarTransform4f
        scene = {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": profile["max_depth"], "rr_depth": 5},
            "sensor": {
                "type": "perspective", "fov": float(model.cam_fovy[camera]), "fov_axis": "y",
                "near_clip": float(model.vis.map.znear * model.stat.extent),
                "far_clip": float(model.vis.map.zfar * model.stat.extent),
                "to_world": transform.look_at(origin=origin, target=origin - rotation[:, 2],
                                             up=rotation[:, 1]),
                "sampler": {"type": "independent", "sample_count": profile["samples"]},
                "film": {"type": "hdrfilm", "width": 320, "height": 240,
                         "pixel_format": "rgb", "rfilter": {"type": "tent"}},
            },
            "environment": {"type": "constant", "radiance": profile["environment_radiance"]},
            "key": {
                "type": "rectangle",
                "to_world": transform.look_at(origin=profile["key_origin"], target=[.22, 0, -.003],
                                             up=[0, 1, 0])
                @ transform.scale([profile["key_half_size"]] * 2 + [1]),
                "emitter": {"type": "area", "radiance": profile["key_radiance"]},
            },
        }
        for geom in self.visible:
            material = int(model.geom_matid[geom])
            rgb = model.mat_rgba[material, :3] if material >= 0 else model.geom_rgba[geom, :3]
            table = model.geom(geom).name == "table"
            bsdf = {
                "type": "roughplastic", "distribution": "ggx",
                "alpha": profile["table_roughness"] if table else profile["plastic_roughness"],
                "int_ior": profile["plastic_ior"], "ext_ior": 1.,
                "diffuse_reflectance": {
                    "type": "rgb", "value": profile["table_linear_rgb"] if table
                    else linear_rgb(rgb).tolist(),
                },
            }
            position, rotation = data.geom_xpos[geom], data.geom_xmat[geom].reshape(3, 3)
            matrix = np.eye(4)
            matrix[:3, :3], matrix[:3, 3] = rotation, position
            world = transform(matrix)
            kind = model.geom_type[geom]
            if geom in self.meshes:
                vertices, normals = self.meshes[geom]
                props = mi.Properties()
                props["bsdf"] = mi.load_dict(bsdf)
                mesh = mi.Mesh(f"geom_{geom}", len(vertices), len(vertices) // 3,
                               props, has_vertex_normals=True)
                parameters = mi.traverse(mesh)
                parameters["vertex_positions"] = (vertices @ rotation.T + position).astype("float32").ravel()
                parameters["vertex_normals"] = (normals @ rotation.T).astype("float32").ravel()
                parameters["faces"] = np.arange(len(vertices), dtype="uint32")
                parameters.update()
                scene[f"geom_{geom}"] = mesh
                continue
            size = model.geom_size[geom]
            if kind == mujoco.mjtGeom.mjGEOM_PLANE:
                shape = {"type": "rectangle", "to_world": world @ transform.scale([2, 2, 1])}
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                shape = {"type": "cube", "to_world": world @ transform.scale(size)}
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                shape = {"type": "sphere", "center": position, "radius": float(size[0])}
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                shape = {
                    "type": "cylinder", "p0": position - rotation[:, 2] * size[1],
                    "p1": position + rotation[:, 2] * size[1], "radius": float(size[0]),
                }
                for side in (-1, 1):
                    cap = world @ transform.translate([0, 0, side * size[1]])
                    if side < 0:
                        cap = cap @ transform.rotate([1, 0, 0], 180)
                    scene[f"cap_{geom}_{side}"] = {
                        "type": "disk", "to_world": cap @ transform.scale([size[0]] * 2 + [1]),
                        "bsdf": bsdf,
                    }
            else:
                raise ValueError(f"Unsupported visible geometry: {model.geom(geom).name}, type {kind}")
            scene[f"geom_{geom}"] = {**shape, "bsdf": bsdf}
        image = mi.render(mi.load_dict(scene), seed=profile["seed"] + frame,
                          spp=profile["samples"])
        bitmap = mi.Bitmap(image * profile["exposure_gain"]).convert(
            mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True,
        )
        return np.asarray(bitmap).copy()


def rerender_release(entry: dict, output: Path, profile: dict) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    archive = Path(entry["archive"])
    archive_hash = file_hash(archive)
    if entry.get("expected_archive_sha256") not in (None, archive_hash):
        raise ValueError(f"Archive differs from source qualification: {archive}")
    original_robot = scene_xml(MODEL_ROOT)
    implementation = implementation_inputs()
    with EpisodeStorage(archive) as storage:
        storage.verify()
        collection = dict(storage.catalog)
        collection["source_release_id"] = collection["release_id"]
        collection["release_id"] = hashlib.sha256(json.dumps(
            [collection["source_release_id"], profile, file_hash(Path(__file__))],
            sort_keys=True,
        ).encode()).hexdigest()
        destinations = []
        for episode in entry["episodes"]:
            source = Path(episode["source"])
            metadata = json.loads(storage.read(source, "episode.json"))
            rows = [json.loads(line) for line in storage.read(source, "frames.jsonl").splitlines()]
            validate_clean_expert(metadata, rows)
            xml = storage.read(source, "model.xml").decode()
            expected = metadata["scene_archive_sha256"]
            for name, digest in expected.items():
                if hashlib.sha256(storage.read(source, name)).hexdigest() != digest:
                    raise ValueError(f"Recorded model/state hash mismatch: {source}/{name}")
            render_xml = cad_render_xml(xml, original_robot)
            model = mujoco.MjModel.from_xml_string(render_xml)
            plan = json.loads(storage.read(source, "scene_plan.json"))
            names = tuple(obj["instance_id"] for obj in plan["objects"])
            sim = BeamSimulation(model=model, render=True, object_names=names, distractor_names=())
            destination = output / f"episode_{len(destinations):04d}"
            destination.mkdir(exist_ok=True)
            destinations.append(str(destination.resolve()))
            try:
                completed = destination / "rerender_receipt.json"
                if completed.exists():
                    receipt = json.loads(completed.read_text())
                    if (receipt["source_archive_sha256"] != archive_hash
                            or receipt["profile"] != profile
                            or receipt["implementation_inputs_sha256"] != implementation):
                        raise ValueError("Completed episode has a different rerender identity")
                    for name, digest in receipt["images_sha256"].items():
                        if file_hash(destination / name) != digest:
                            raise ValueError("Completed episode image checksum mismatch")
                    if file_hash(destination / "frames.jsonl") != receipt["frames_sha256"]:
                        raise ValueError("Completed episode labels checksum mismatch")
                    continue
                if (destination / "frames.jsonl").exists():
                    LOGGER.warning("Restarting incomplete derived episode from its saved poses: %s", destination)
                for name, content in storage.connection.execute(
                    "SELECT name,content FROM files WHERE source=?", (str(source),)
                ):
                    path = Path(name)
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError(f"Unsafe archive member: {name}")
                    if path.parts[0] == "wrist" or name in ("episode.json", "frames.jsonl"):
                        continue
                    target = destination / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                (destination / "original_episode.json").write_bytes(storage.read(source, "episode.json"))
                (destination / "original_frames.jsonl").write_bytes(storage.read(source, "frames.jsonl"))
                (destination / "render_model.xml").write_text(render_xml)
                mujoco.mj_saveModel(model, str(destination / "render_model.mjb"), None)
                renderer = FrameRenderer(model, profile)
                sim.hide_unseen_centers = bool(metadata.get("perception", {}).get("hide_unseen_centers", False))
                images = {}
                with (destination / "frames.jsonl").open("w") as stream:
                    for index, row in enumerate(rows):
                        restore_pose(model, sim.data, row)
                        sim.configure_command(metadata["task"], list(zip(
                            row["command_entity_ids"], row["command_roles"], strict=True,
                        )))
                        rendered = renderer.render(sim.data, index)
                        image = destination / row["image"]
                        if image.is_absolute() and not image.resolve().is_relative_to(destination.resolve()):
                            raise ValueError("Frame image escapes destination")
                        image.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(image), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)):
                            raise OSError(f"Cannot save rendered frame: {image}")
                        images[row["image"]] = file_hash(image)
                        revised = {**row, **sim._recording_command_labels()}
                        stream.write(json.dumps(revised, allow_nan=False) + "\n")
                        atomic_json(output / "progress.json", {
                            "status": "rendering", "episode": len(destinations), "frame": index + 1,
                            "episode_frames": len(rows), "source": str(source),
                        })
                identity = {
                    "backend": "mitsuba-llvm_ad_rgb", "profile": profile,
                    "implementation_sha256": file_hash(Path(__file__)),
                    "implementation_inputs_sha256": implementation,
                    "render_model_sha256": file_hash(destination / "render_model.xml"),
                    "source_archive_sha256": archive_hash, "source": str(source),
                    "original_renderer": metadata.get("initial", {}).get("renderer"),
                    "images_sha256": images, "labels_recomputed": True,
                    "frames_sha256": file_hash(destination / "frames.jsonl"),
                    "qualification": "Appearance derivation only. Original trajectory, physics verdict, "
                    "split and historical audits unchanged. External images retained, not rerendered. "
                    "No physics stepping, calibration, transfer or training-promotion claim.",
                }
                metadata["source_release_id"] = metadata["release_id"]
                metadata["release_id"] = collection["release_id"]
                metadata["appearance_rerender"] = identity
                atomic_json(destination / "episode.json", metadata)
                atomic_json(destination / "rerender_receipt.json", identity)
            finally:
                sim.close()
        collection["successful_episodes"] = destinations
        replacements = dict(zip((e["source"] for e in entry["episodes"]), destinations, strict=True))
        collection["attempts"] = [
            {**attempt, "original_source": attempt["source"],
             "source": replacements.get(attempt["source"], attempt["source"])}
            for attempt in collection.get("attempts", [])
        ]
        collection["appearance_rerender"] = {
            "profile": profile, "source_archive": str(archive), "source_archive_sha256": archive_hash,
            "original_split_and_qualification_retained": True,
        }
        atomic_json(output / "collection.json", collection)
    if file_hash(archive) != archive_hash:
        raise RuntimeError("Original archive changed during rerender")
    if implementation_inputs() != implementation:
        raise RuntimeError("Renderer or label implementation changed during rerender")
    container = output / "episodes.sqlite"
    if container.exists():
        with EpisodeStorage(container) as stored:
            stored.verify()
            stored.verify_sources(output)
        result = {"episodes": len(destinations), "container_bytes": container.stat().st_size}
    else:
        result = consolidate(output, container)
    atomic_json(output / "qualification.json", {
        **result, "source_archive_sha256": archive_hash, "status": "complete",
        "qualification": "Appearance rerender verified; inherits original physics and split only.",
    })
    atomic_json(output / "progress.json", {"status": "complete", **result})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--since", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--samples", type=int, default=1024)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("Samples must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "inventory.json"
    if manifest_path.exists() and not args.inventory_only:
        manifest = json.loads(manifest_path.read_text())
        if manifest["since"] != datetime.fromisoformat(args.since).isoformat():
            raise ValueError("Output inventory has a different date cutoff")
    else:
        manifest = inventory(args.root, datetime.fromisoformat(args.since))
        atomic_json(manifest_path, manifest)
    if args.inventory_only:
        print(json.dumps({key: manifest[key] for key in ("episodes", "frames", "since")}))
        return
    profile = json.loads(DEFAULT_PROFILE.read_text())
    profile.update(version="curriculum-wrist-pathtrace-v1", width=320, height=240,
                   samples=args.samples)
    profile_path = args.output / "profile.json"
    if profile_path.exists() and json.loads(profile_path.read_text()) != profile:
        raise ValueError("Cannot change the profile in an existing rerender batch")
    atomic_json(args.output / "profile.json", profile)
    identity = {"profile": profile, "implementation_inputs_sha256": implementation_inputs()}
    identity_path = args.output / "batch_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Existing batch was rendered by a different implementation")
    atomic_json(identity_path, identity)
    for entry in manifest["releases"]:
        destination = args.output / entry["release_id"]
        if destination.exists():
            if (destination / "qualification.json").exists():
                with EpisodeStorage(destination / "episodes.sqlite") as stored:
                    stored.verify()
                    stored.verify_sources(destination)
                continue
        rerender_release(entry, destination, profile)
    atomic_json(args.output / "completion.json", {
        "status": "complete", "episodes": manifest["episodes"], "frames": manifest["frames"],
        "releases": len(manifest["releases"]),
    })


if __name__ == "__main__":
    main()
