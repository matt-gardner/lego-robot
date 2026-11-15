"""CPU-only pile preview images: painter's-algorithm mesh drawing, no OpenGL required."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from so101.pile_task_runtime import create_simulation, load_problems


def camera(eye, target):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(right, forward), forward])


def draw(model, data, bodies, eye, target, size, scale):
    mujoco.mj_forward(model, data)
    rotation = camera(np.asarray(eye), np.asarray(target))
    light = np.array([.3, -.4, .87])
    polygons = []
    for geom in np.flatnonzero(np.isin(model.geom_bodyid, bodies)):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom] != 2:
            continue
        mesh = model.geom_dataid[geom]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh]:
                                   model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        faces = model.mesh_face[model.mesh_faceadr[mesh]:
                                model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]
        world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        triangles = world[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        view = (triangles - eye) @ rotation.T
        facing = np.einsum("ij,ij->i", normals, triangles[:, 0] - eye) < 0
        shade = .35 + .65 * np.clip(normals @ light, 0, 1)
        color = model.geom_rgba[geom][:3]
        for tri, depth_tri, ok, s in zip(view, view[:, :, 2], facing, shade):
            if ok and depth_tri.min() > 0:
                xy = tri[:, :2] / tri[:, 2:3] * scale
                points = [(size[0] / 2 + x, size[1] / 2 - y) for x, y in xy]
                polygons.append((depth_tri.mean(), points, tuple(int(255 * c * s) for c in color)))
    image = Image.new("RGB", size, (235, 235, 230))
    canvas = ImageDraw.Draw(image)
    for _, points, fill in sorted(polygons, key=lambda item: -item[0]):
        canvas.polygon(points, fill=fill)
    return image


def render_scene(scene: Path, output: Path) -> None:
    entry = json.loads((scene / "admission.json").read_text())
    problem = load_problems(entry)[0]
    sim = create_simulation(problem)
    bodies = [sim.model.body(i["body_name"]).id for i in problem["scene"]["instances"]]
    centers = np.array([sim.data.xpos[b] for b in bodies])
    # Median framing keeps a lone piece that bounced far away from shrinking the pile.
    target = np.median(centers, axis=0)
    distances = np.linalg.norm(centers[:, :2] - target[:2], axis=1)
    radius = max(.03, float(np.percentile(distances, 90)) + .015)
    views = [draw(sim.model, sim.data, bodies, target + np.array(offset) * radius, target,
                  (480, 360), 640)
             for offset in ([-2.2, -2.2, 2.0], [0, -.001, 3.6])]
    sheet = Image.new("RGB", (960, 360))
    for index, view in enumerate(views):
        sheet.paste(view, (480 * index, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    sheet.save(temporary)
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Producer root containing piles/")
    args = parser.parse_args()
    for scene in sorted((args.root / "piles").glob("scene_*")):
        output = args.root / "previews" / f"{scene.name}.png"
        if (scene / "admission.json").exists() and not output.exists():
            render_scene(scene, output)
            print(output, flush=True)


if __name__ == "__main__":
    main()
