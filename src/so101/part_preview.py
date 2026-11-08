"""Visual-only previews from the catalog's hash-pinned MJCF mesh assets."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET


def visual_geometry(content: bytes) -> dict:
    if b"<!DOCTYPE" in content or b"<!ENTITY" in content:
        raise ValueError("MJCF declarations are not supported in previews")
    root = ET.fromstring(content)
    if any(any(body.get(key) for key in ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis"))
           for body in root.findall(".//body")):
        raise ValueError("Transformed part bodies require a published preview adapter")
    assets = {mesh.get("name"): mesh for mesh in root.findall("./asset/mesh")}
    meshes = []
    for geom in root.findall(".//geom"):
        if geom.get("group") != "2" or geom.get("type") != "mesh":
            continue
        if geom.get("pos") or geom.get("quat") or geom.get("euler"):
            raise ValueError("Transformed visual geoms require a published preview adapter")
        mesh = assets.get(geom.get("mesh"))
        if mesh is None or mesh.get("file") or mesh.get("scale"):
            raise ValueError("Preview requires embedded, part-local visual meshes")
        values = [float(value) for value in mesh.get("vertex", "").split()]
        indices = [int(value) for value in mesh.get("face", "").split()]
        color = [float(value) for value in geom.get("rgba", "0.8 0.25 0.1 1").split()]
        if (not values or len(values) % 3 or len(indices) % 3 or not indices
                or not all(math.isfinite(value) for value in values)
                or len(color) != 4 or not all(math.isfinite(value) and 0 <= value <= 1 for value in color)):
            raise ValueError("Invalid visual mesh data")
        vertices = [values[index:index + 3] for index in range(0, len(values), 3)]
        if any(index < 0 or index >= len(vertices) for index in indices):
            raise ValueError("Visual face index outside vertex array")
        meshes.append({"vertices": vertices,
                       "faces": [indices[index:index + 3] for index in range(0, len(indices), 3)],
                       "color": color})
    if not meshes:
        raise FileNotFoundError("No prepared visual geometry")
    return {"meshes": meshes}


def thumbnail(geometry: dict) -> bytes:
    vertices = [vertex for mesh in geometry["meshes"] for vertex in mesh["vertices"]]
    lower = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
    upper = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
    center = [(lo + hi) / 2 for lo, hi in zip(lower, upper)]
    radius = max(math.dist(vertex, center) for vertex in vertices)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Degenerate visual geometry")
    yaw, pitch = -0.65, 0.55
    triangles = []
    for mesh in geometry["meshes"]:
        projected = []
        for vertex in mesh["vertices"]:
            x, y, z = [(value - origin) / radius for value, origin in zip(vertex, center)]
            horizontal = math.cos(yaw) * x - math.sin(yaw) * y
            depth = math.sin(yaw) * x + math.cos(yaw) * y
            vertical = math.cos(pitch) * z - math.sin(pitch) * depth
            depth = math.cos(pitch) * depth + math.sin(pitch) * z
            projected.append((160 + horizontal * 125, 120 - vertical * 100, depth))
        for face in mesh["faces"]:
            points = [projected[index] for index in face]
            a, b, c = [mesh["vertices"][index] for index in face]
            u, v = [b[axis] - a[axis] for axis in range(3)], [c[axis] - a[axis] for axis in range(3)]
            normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                      u[0] * v[1] - u[1] * v[0]]
            length = math.sqrt(sum(value * value for value in normal))
            shade = 0.4 + 0.6 * abs(sum(value * light for value, light in
                                       zip(normal, (0.35, -0.5, 0.79))) / length) if length else 0.4
            color = ",".join(str(round(channel * shade * 255)) for channel in mesh["color"][:3])
            polygon = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
            triangles.append((sum(point[2] for point in points) / 3,
                              f'<polygon points="{polygon}" fill="rgb({color})" stroke="rgb({color})" stroke-width=".35"/>'))
    triangles.sort(key=lambda triangle: triangle[0])
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 240">'
            '<rect width="320" height="240" rx="14" fill="#eef2f6"/>'
            + "".join(polygon for _, polygon in triangles) + "</svg>").encode()
