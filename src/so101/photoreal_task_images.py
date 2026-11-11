"""CPU appearance derivatives of exact INITIAL pile resets, with separate MuJoCo labels."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
from PIL import Image

from so101.capture_pile_task_images import integration_state, target_annotation
from so101.cloud_checkpoint import atomic_json
from so101.lego_scene import runtime_model_sha256
from so101.pathtrace_pile import DEFAULT_PROFILE, file_hash, render_pathtrace
from so101.pile_task_runtime import create_simulation, load_problem, reset_problem
from so101.task_image_roles import role_annotations


def mono_camera(scene):
    eyes = scene.camera
    origin = np.mean([eye.pos for eye in eyes], axis=0)
    forward = np.mean([eye.forward for eye in eyes], axis=0)
    forward /= np.linalg.norm(forward)
    up = np.mean([eye.up for eye in eyes], axis=0)
    up /= np.linalg.norm(up)
    eye = eyes[0]
    if scene.stereo != mujoco.mjtStereo.mjSTEREO_NONE or eye.orthographic:
        raise ValueError("External derivative requires a mono perspective camera")
    if not np.isclose(eye.frustum_top, -eye.frustum_bottom, atol=1e-7):
        raise ValueError("Off-axis vertical projection is unsupported")
    return {
        "origin": origin.tolist(), "target": (origin + forward).tolist(), "up": up.tolist(),
        "fovy": float(np.rad2deg(2 * np.arctan(
            (eye.frustum_top - eye.frustum_bottom) / (2 * eye.frustum_near),
        ))),
        "near_clip": float(eye.frustum_near), "far_clip": float(eye.frustum_far),
        "eye_transforms": [
            {"origin": eye.pos.tolist(), "forward": eye.forward.tolist(), "up": eye.up.tolist()}
            for eye in eyes
        ],
    }


def world_mesh(model, data, geom):
    # Only the optional CPU export command needs the primitive tessellator.
    import trimesh

    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom])
        va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        na = int(model.mesh_normaladr[mesh])
        faces = model.mesh_face[fa:fa + fn]
        vertices = model.mesh_vert[va:va + vn][faces].reshape(-1, 3).astype(float)
        normals = model.mesh_normal[na + model.mesh_facenormal[fa:fa + fn]].reshape(-1, 3)
    else:
        size = model.geom_size[geom]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=2 * size)
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=128)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            mesh = trimesh.creation.icosphere(subdivisions=4, radius=size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            mesh = trimesh.creation.capsule(radius=size[0], height=2 * size[1], count=(64, 64))
        else:
            raise ValueError(f"Unsupported visible geometry type {kind}: {geom}")
        vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            normals = np.repeat(mesh.face_normals, 3, axis=0)
        else:
            normals = mesh.vertex_normals[mesh.faces].reshape(-1, 3)
    rotation = data.geom_xmat[geom].reshape(3, 3)
    # Compiled vertices already include MuJoCo's mesh centering transform.
    return vertices @ rotation.T + data.geom_xpos[geom], normals @ rotation.T


def write_ply(path, vertices, normals, mesh_root=None):
    rows = np.column_stack((vertices, normals)).astype("<f4")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        f"element face {len(vertices) // 3}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    faces = np.empty(len(vertices) // 3, dtype=[("count", "u1"), ("vertices", "<i4", (3,))])
    faces["count"] = 3
    faces["vertices"] = np.arange(len(vertices)).reshape(-1, 3)
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(rows.tobytes())
        stream.write(faces.tobytes())
    if mesh_root is not None:
        mesh_root.mkdir(parents=True, exist_ok=True)
        digest = file_hash(path)
        shared = mesh_root / f"{digest}.ply"
        if shared.exists():
            if file_hash(shared) != digest:
                raise ValueError("Content-addressed world mesh changed")
            path.unlink()
            os.link(shared, path)
        else:
            os.link(path, shared)


def export_scene(sim, root, output, camera, mesh_root=None):
    output.mkdir(parents=True, exist_ok=False)
    model, data = sim.model, sim.data
    table = model.geom("table").id
    if (model.geom_type[table] != mujoco.mjtGeom.mjGEOM_PLANE
            or not np.allclose(data.geom_xmat[table].reshape(3, 3)[:, 2], [0, 0, 1])):
        raise ValueError("Appearance recipe requires the actual horizontal table plane")
    groups = mujoco.MjvOption().geomgroup
    objects, grouped = [], {}
    for geom in range(model.ngeom):
        if geom == table or not groups[model.geom_group[geom]] or model.geom_rgba[geom, 3] == 0:
            continue
        material = int(model.geom_matid[geom])
        rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom]
        if rgba[3] != 1:
            raise ValueError("Transparent visual geometry requires a separate appearance contract")
        vertices, normals = world_mesh(model, data, geom)
        key = (int(model.geom_bodyid[geom]), tuple(rgba[:3]))
        group = grouped.setdefault(key, {"vertices": [], "normals": [], "geom_ids": []})
        group["vertices"].append(vertices)
        group["normals"].append(normals)
        group["geom_ids"].append(geom)
    for (body, rgb), group in grouped.items():
        path = output / f"body-{body}-material-{len(objects)}.ply"
        write_ply(path, np.concatenate(group["vertices"]), np.concatenate(group["normals"]), mesh_root)
        objects.append({"mesh": path.name, "rgb": list(map(float, rgb)), "geom_ids": group["geom_ids"]})
    description = {
        "source": str(root), "source_state_file": "problem.json",
        "integration_state_sha256": hashlib.sha256(integration_state(sim).tobytes()).hexdigest(),
        "table_z_m": float(data.geom_xpos[table, 2]), "camera": camera, "objects": objects,
        "segmentation": "../segmentation.npz", "table_geom_id": table,
    }
    atomic_json(output / "scene.json", description)
    return output / "scene.json"


def group_task_resets(sim, entry, tasks):
    groups = {}
    for task in tasks:
        reset_problem(sim, load_problem(entry, task["task_id"]))
        digest = hashlib.sha256(integration_state(sim).tobytes()).hexdigest()
        groups.setdefault(digest, []).append(task)
    return groups


def admit_render(output):
    if shutil.disk_usage(output).free < 10 * 1024**3:
        raise RuntimeError("CPU rendering paused: less than 10 GiB output disk available")
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(memory["MemAvailable"].split()[0]) * 1024
    cgroup = Path("/sys/fs/cgroup")
    if (cgroup / "memory.max").exists():
        limit = (cgroup / "memory.max").read_text().strip()
        if limit != "max":
            available = min(available, int(limit) - int((cgroup / "memory.current").read_text()))
    if available < 2 * 1024**3:
        raise RuntimeError("CPU rendering paused: less than 2 GiB memory headroom available")


def render_reset_group(sim, entry, tasks, state_sha, output, profile_path, contract,
                       variant="llvm_ad_rgb"):
    started = time.monotonic()
    profile = json.loads(profile_path.read_bytes())
    problem = load_problem(entry, tasks[0]["task_id"])
    reset_problem(sim, problem)
    before, physical = integration_state(sim), runtime_model_sha256(sim.model)
    if hashlib.sha256(before.tobytes()).hexdigest() != state_sha:
        raise ValueError("Selected task reset changed before image capture")
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.mean([sim.position(name) for name in sim.registry], axis=0)
    camera.distance, camera.azimuth, camera.elevation = .40, 135., -65.
    sim.renderer.update_scene(sim.data, camera=camera)
    geometry_camera = mono_camera(sim.renderer.scene)
    scene_output = output / "_scenes" / entry["scene_id"] / state_sha
    scene_output.mkdir(parents=True, exist_ok=True)
    raster = sim.renderer.render().copy()
    Image.fromarray(raster).save(scene_output / "mujoco-rgb.png")
    sim.renderer.enable_segmentation_rendering()
    try:
        segmentation = sim.renderer.render().copy()
    finally:
        sim.renderer.disable_segmentation_rendering()
    np.savez_compressed(scene_output / "segmentation.npz", segmentation=segmentation)
    export = scene_output / "export"
    if export.exists():
        raise ValueError(f"Unpublished export already exists; preserve and inspect it: {export}")
    scene_path = export_scene(sim, problem["root"], export, geometry_camera, output / "_meshes")
    rendered = render_pathtrace(scene_path, scene_output / "pathtrace", profile_path, variant=variant)
    if (not np.array_equal(before, integration_state(sim))
            or physical != runtime_model_sha256(sim.model)):
        raise ValueError("Appearance or labels changed the source physical model/state")
    image = scene_output / "pathtrace" / "pile.png"
    image_sha = file_hash(image)
    destination = output / f"{image_sha}.png"
    if not destination.exists():
        shutil.copyfile(image, destination)
    rows = []
    for task in tasks:
        reset_problem(sim, load_problem(entry, task["task_id"]))
        if not np.array_equal(before, integration_state(sim)):
            raise ValueError("Task-specific INITIAL reset differs from rendered scene state")
        target = task["target_instance_id"]
        geoms = np.flatnonzero(sim.model.geom_bodyid == sim.registry[target])
        mask = ((segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
                & np.isin(segmentation[:, :, 0], geoms))
        source = str(problem["root"] / "tasks" / task["task_id"])
        identity = hashlib.sha256(source.encode()).hexdigest()
        receipt = {
            "problem_id": identity, "episode_source": source, **contract,
            "task_type": task["task_type"],
            "capture_implementation_sha256": contract["implementation_sha256"]["photoreal_task_images.py"],
            "target_instance_id": target, "image_file": destination.name, "sha256": image_sha,
            "width": profile["width"], "height": profile["height"], "camera": "external",
            "camera_provenance": geometry_camera, "backend": rendered["backend"],
            "samples": profile["samples"], "physics_unchanged": True, "assistance": 0,
            "is_gpu_policy_frame": False, "initial_state": "exact-admitted-integration-reset",
            "scene_files_sha256": entry["files_sha256"], "integration_state_sha256": state_sha,
            "segmentation_sha256": file_hash(scene_output / "segmentation.npz"),
            "segmentation_file": str((scene_output / "segmentation.npz").relative_to(output)),
            "appearance_receipt": str((scene_output / "pathtrace" / "receipt.json").relative_to(output)),
            "target_annotations": role_annotations(sim.model, sim.data, task, segmentation),
            "capture_wall_seconds": time.monotonic() - started,
            "scene_sha256": rendered["scene_sha256"], **target_annotation(mask),
        }
        rows.append(receipt)
    if physical != runtime_model_sha256(sim.model):
        raise ValueError("Task resets changed the physical model")
    for receipt in rows:
        atomic_json(output / f'{receipt["problem_id"]}.json', receipt)
    return rows


def capture_pool(pool_path, output, profile_path=DEFAULT_PROFILE, limit_scenes=None, limit_states=None):
    pool_bytes = pool_path.read_bytes()
    pool_sha = hashlib.sha256(pool_bytes).hexdigest()
    pool = json.loads(pool_bytes)
    if (pool["kind"] != "lego-pile-task-pool" or pool["status"] != "complete"
            or any(entry["split"] != "train" for entry in pool["scenes"])):
        raise ValueError("Rendering requires an immutable completed TRAIN pool")
    profile = json.loads(profile_path.read_bytes())
    implementation = {
        name: file_hash(Path(__file__).with_name(name))
        for name in ("photoreal_task_images.py", "task_image_roles.py", "pathtrace_pile.py", "pile_task_runtime.py",
                     "lego_scene.py", "capture_pile_task_images.py")
    }
    contract = {"pool_sha256": pool_sha, "profile_sha256": file_hash(profile_path),
                "implementation_sha256": implementation, "schema_version": 1}
    meta = output / "_meta"
    meta.mkdir(parents=True, exist_ok=True)
    contract_path = meta / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_bytes()) != contract:
        raise ValueError("Existing image population has a different immutable render contract")
    atomic_json(contract_path, contract)
    coverage = Counter()
    completed, rendered_states = 0, 0
    population = Counter()
    for entry in pool["scenes"]:
        scene = json.loads((Path(entry["root"]) / "problem.json").read_bytes())
        population.update((str(scene["tags"]["piece_count"]), task["task_type"]) for task in scene["tasks"])
    selected_tasks = sum(population.values())
    atomic_json(meta / "population.json", {
        **contract, "selected_tasks": selected_tasks, "selected_scenes": len(pool["scenes"]),
        "buckets": [{"piece_count": key[0], "task": key[1], "tasks": count}
                    for key, count in sorted(population.items())],
    })
    for index, entry in enumerate(pool["scenes"]):
        if limit_scenes is not None and index >= limit_scenes:
            break
        admit_render(output)
        scene = json.loads((Path(entry["root"]) / "problem.json").read_bytes())
        tasks = scene["tasks"]
        identities = {
            task["task_id"]: hashlib.sha256(
                str(Path(entry["root"]) / "tasks" / task["task_id"]).encode(),
            ).hexdigest() for task in tasks
        }
        receipts = [output / f'{identities[task["task_id"]]}.json' for task in tasks]
        if all(path.exists() for path in receipts):
            for path in receipts:
                row = json.loads(path.read_bytes())
                if row["pool_sha256"] != pool_sha or file_hash(output / row["image_file"]) != row["sha256"]:
                    raise ValueError("Completed image receipt or asset changed")
                coverage[(str(scene["tags"]["piece_count"]), row["task_type"], row["target_bbox_status"])] += 1
            completed += len(tasks)
            continue
        problem = load_problem(entry, tasks[0]["task_id"])
        sim = create_simulation(problem)
        try:
            groups = group_task_resets(sim, entry, tasks)
            sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, profile["width"])
            sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, profile["height"])
            sim.renderer = mujoco.Renderer(sim.model, height=profile["height"], width=profile["width"])
            for state_sha, group in groups.items():
                paths = [output / f'{identities[task["task_id"]]}.json' for task in group]
                if all(path.exists() for path in paths):
                    rows = [json.loads(path.read_bytes()) for path in paths]
                    if any(row["integration_state_sha256"] != state_sha
                           or file_hash(output / row["image_file"]) != row["sha256"] for row in rows):
                        raise ValueError("Completed task image differs from its exact reset")
                else:
                    if limit_states is not None and rendered_states >= limit_states:
                        return {"completed_tasks": completed, "selected_tasks": selected_tasks,
                                "pool_sha256": pool_sha}
                    admit_render(output)
                    rows = render_reset_group(sim, entry, group, state_sha, output, profile_path, contract)
                    rendered_states += 1
                completed += len(rows)
                for row in rows:
                    coverage[(str(scene["tags"]["piece_count"]), row["task_type"], row["target_bbox_status"])] += 1
                status = {
                    **contract, "completed_tasks": completed, "selected_tasks": selected_tasks,
                    "selected_scenes": len(pool["scenes"]), "new_rendered_states": rendered_states,
                    "coverage": [
                        {"piece_count": key[0], "task": key[1], "status": status, "tasks": (
                            count - sum(coverage[(key[0], key[1], measured)]
                                        for measured in ("visible", "not_visible"))
                            if status == "unmeasured" else coverage[(key[0], key[1], status)]
                        )}
                        for key, count in sorted(population.items())
                        for status in ("visible", "not_visible", "unmeasured")
                    ],
                }
                atomic_json(meta / "progress.json", status)
                print(json.dumps(status), flush=True)
        finally:
            sim.close()
    return {"completed_tasks": completed, "pool_sha256": pool_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--limit-scenes", type=int)
    parser.add_argument("--limit-states", type=int)
    args = parser.parse_args()
    os.nice(10)
    print(json.dumps(capture_pool(args.pool, args.output, args.profile, args.limit_scenes, args.limit_states)))


if __name__ == "__main__":
    main()
