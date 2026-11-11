"""Optional CPU or CUDA appearance rendering of a pinned world-space mesh export."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_PROFILE = Path(__file__).with_name("renderer_profiles") / "pile_pathtrace_v1.json"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def camera_frame(camera: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin, target, up = (
        np.asarray(camera[key], dtype=float) for key in ("origin", "target", "up")
    )
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in (origin, target, up)):
        raise ValueError("Camera vectors must contain three finite coordinates")
    forward = target - origin
    if np.linalg.norm(forward) == 0 or np.linalg.norm(np.cross(forward, up)) == 0:
        raise ValueError("Camera direction and up must define a nondegenerate frame")
    return origin, target, up


def linear_rgb(rgb) -> np.ndarray:
    values = np.asarray(rgb, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Color must be three finite sRGB components in [0, 1]")
    return np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)


def check_segmentation_alignment(mi, dr, loaded, description, segmentation_path, width, height):
    with np.load(segmentation_path) as archive:
        segmentation = archive["segmentation"]
    if segmentation.shape != (height, width, 2):
        raise ValueError("Segmentation dimensions differ from the appearance camera")
    ys, xs = np.meshgrid(np.arange(2, height, 5), np.arange(2, width, 5), indexing="ij")
    pixel = mi.Point2f((xs.ravel() + .5) / width, (ys.ravel() + .5) / height)
    rays, _ = loaded.sensors()[0].sample_ray(0, .5, pixel, mi.Point2f(.5))
    intersections = loaded.ray_intersect(rays)
    measured = segmentation[ys.ravel(), xs.ravel()]
    expected = np.full(len(xs.ravel()), -1)
    actual = np.full_like(expected, -1)
    names = {"table": [description["table_geom_id"]]}
    names.update({f"part_{i}": obj["geom_ids"] for i, obj in enumerate(description["objects"])})
    for index, shape in enumerate(loaded.shapes()):
        if shape.id() not in names:
            continue
        expected[(measured[:, 1] == 5) & np.isin(measured[:, 0], names[shape.id()])] = index
        actual[np.asarray(intersections.shape == shape)] = index
    mismatch = float(np.mean(expected != actual))
    if mismatch > .01:
        raise ValueError(f"Same-camera mesh/segmentation alignment failed: {mismatch:.4%} mismatched rays")
    return {"method": "Mitsuba pixel-center primary rays versus MuJoCo geom segmentation",
            "sampled_pixels": len(expected), "mismatched_pixels": int(np.count_nonzero(expected != actual)),
            "mismatch_fraction": mismatch, "maximum_mismatch_fraction": .01}


def render_pathtrace(
    scene_path: Path, output: Path, profile_path: Path = DEFAULT_PROFILE,
    mesh_root: Path | None = None,
    variant: str = "llvm_ad_rgb",
) -> dict:
    if variant not in ("llvm_ad_rgb", "cuda_ad_rgb"):
        raise ValueError(f"Unsupported appearance backend: {variant}")
    # Mitsuba is an optional heavy backend; normal simulation imports must not require it.
    import drjit as dr
    import mitsuba as mi

    scene_path, output, profile_path = map(Path, (scene_path, output, profile_path))
    description = json.loads(scene_path.read_text())
    profile = json.loads(profile_path.read_text())
    source = Path(description["source"])
    if not source.is_absolute():
        source = scene_path.parent / source
    inputs = [
        scene_path, profile_path, Path(__file__), source / "model.xml",
        source / description.get("source_state_file", "settled-state.npz"),
    ]
    input_hashes = {str(path): file_hash(path) for path in inputs}
    origin, target, up = camera_frame(description["camera"])
    meshes = []
    for obj in description["objects"]:
        path = Path(obj["mesh"])
        if mesh_root is not None:
            path = Path(mesh_root) / path.name
        elif not path.is_absolute():
            path = scene_path.parent / path
        meshes.append((path, file_hash(path), linear_rgb(obj["rgb"])))
    if not meshes:
        raise ValueError("Scene contains no visual meshes")
    table_z = float(description["table_z_m"])
    if not np.isfinite(table_z):
        raise ValueError("Table height must be finite")
    mi.set_variant(variant)
    dr.set_thread_count(profile["threads"])
    scene = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": profile["max_depth"], "rr_depth": 5},
        "sensor": {
            "type": "perspective", "fov": description["camera"]["fovy"], "fov_axis": "y",
            "to_world": mi.ScalarTransform4f.look_at(origin=origin, target=target, up=up),
            "sampler": {"type": "independent", "sample_count": profile["samples"]},
            "film": {"type": "hdrfilm", "width": profile["width"], "height": profile["height"],
                     "pixel_format": "rgb", "rfilter": {"type": "tent"}},
        },
        "table": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate([0, 0, table_z])
            @ mi.ScalarTransform4f.scale([2, 2, 1]),
            "bsdf": {"type": "roughplastic", "alpha": profile["table_roughness"],
                     "diffuse_reflectance": {"type": "rgb", "value": profile["table_linear_rgb"]}},
        },
        "environment": {"type": "constant", "radiance": profile["environment_radiance"]},
        "key": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=profile["key_origin"], target=[0, 0, table_z], up=[0, 1, 0],
            ) @ mi.ScalarTransform4f.scale([profile["key_half_size"]] * 2 + [1]),
            "emitter": {"type": "area", "radiance": profile["key_radiance"]},
        },
    }
    for key in ("near_clip", "far_clip"):
        if key in description["camera"]:
            scene["sensor"][key] = float(description["camera"][key])
    if "aperture_radius_m" in profile:
        focus = float(profile["focus_distance_m"])
        aperture = float(profile["aperture_radius_m"])
        if not np.isfinite([focus, aperture]).all() or focus <= 0 or aperture <= 0:
            raise ValueError("Thin-lens focus distance and aperture must be positive and finite")
        scene["sensor"].update(type="thinlens", focus_distance=focus, aperture_radius=aperture)
    for index, (path, _, rgb) in enumerate(meshes):
        scene[f"part_{index}"] = {
            "type": "ply", "filename": str(path), "face_normals": False,
            "bsdf": {"type": "roughplastic", "distribution": "ggx",
                     "alpha": profile["plastic_roughness"], "int_ior": profile["plastic_ior"],
                     "ext_ior": 1., "diffuse_reflectance": {"type": "rgb", "value": rgb.tolist()}},
        }
    loaded = mi.load_dict(scene)
    alignment = None
    if "segmentation" in description:
        segmentation_path = scene_path.parent / description["segmentation"]
        input_hashes[str(segmentation_path)] = file_hash(segmentation_path)
        alignment = check_segmentation_alignment(
            mi, dr, loaded, description, segmentation_path, profile["width"], profile["height"],
        )
    output.mkdir(parents=True, exist_ok=False)
    image = mi.render(loaded, seed=profile["seed"], spp=profile["samples"])
    mi.util.write_bitmap(
        str(output / "pile.png"), image * profile["exposure_gain"], write_async=False,
    )
    for path, digest in input_hashes.items():
        if file_hash(Path(path)) != digest:
            raise RuntimeError(f"Input changed while rendering: {path}")
    for path, digest, _ in meshes:
        if file_hash(path) != digest:
            raise RuntimeError(f"Mesh changed while rendering: {path}")
    receipt = {
        "backend": f"mitsuba-{variant}", "mitsuba_version": mi.__version__,
        "scene_sha256": file_hash(scene_path), "profile_sha256": file_hash(profile_path),
        "implementation_sha256": file_hash(Path(__file__)),
        "mesh_sha256": {str(path): digest for path, digest, _ in meshes},
        "mesh_count": len(meshes), "camera": description["camera"], "settings": profile,
        "input_sha256": input_hashes,
        "image_sha256": file_hash(output / "pile.png"),
        "segmentation_alignment": alignment,
        "qualification": "Appearance only; input collision qualification is not upgraded. "
        "No simulation stepping, silhouette bevels, synchronized labels, calibration or transfer claim.",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--mesh-root", type=Path, help="Explicit relocation directory for mesh basenames")
    parser.add_argument("--variant", choices=("llvm_ad_rgb", "cuda_ad_rgb"), default="llvm_ad_rgb")
    args = parser.parse_args()
    render_pathtrace(args.scene, args.output, args.profile, args.mesh_root, args.variant)


if __name__ == "__main__":
    main()
