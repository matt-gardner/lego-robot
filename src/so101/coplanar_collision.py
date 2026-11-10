"""Merge adjacent planar triangles only when their union is convex and hole-free."""
from __future__ import annotations

from itertools import combinations

import numpy as np


def merge_coplanar_triangles(triangles: np.ndarray, tolerance_m: float = 1e-12) -> list[np.ndarray]:
    """Keep original vertices; only remove a shared edge inside a convex planar union."""
    triangles = np.asarray(triangles, dtype=float)
    if (triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or triangles.size == 0
            or not np.isfinite(triangles).all()
            or not np.isfinite(tolerance_m) or tolerance_m <= 0):
        raise ValueError("Expected finite triangles and positive coplanarity tolerance")
    points, indices, polygons, normals = [], {}, {}, {}
    for index, triangle in enumerate(triangles):
        polygon = []
        for point in triangle:
            key = tuple(point)
            if key not in indices:
                indices[key] = len(points)
                points.append(point.copy())
            polygon.append(indices[key])
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = float(np.linalg.norm(normal))
        if not np.isfinite(length) or length == 0:
            raise ValueError("Degenerate source triangle")
        polygons[index] = polygon
        normals[index] = normal / length
    vertices = np.asarray(points)

    def edges(polygon):
        return list(zip(polygon, polygon[1:] + polygon[:1]))

    def area(polygon, normal):
        points = vertices[polygon]
        centered = points - points[0]
        return float(np.sum(np.cross(centered, np.roll(centered, -1, axis=0)) @ normal) / 2)

    def union(first, second, normal):
        a, b = polygons[first], polygons[second]
        dot = float(normal @ normals[second])
        if abs(abs(dot) - 1.) > 1e-12:
            return None
        if np.max(np.abs((vertices[b] - vertices[a[0]]) @ normal)) > tolerance_m:
            return None
        if dot < 0:
            b = list(reversed(b))
        ae, be = edges(a), edges(b)
        shared = [(u, v) for u, v in ae if (v, u) in be]
        if len(shared) != 1:
            return None
        u, v = shared[0]
        boundary = [edge for edge in ae if edge != (u, v)]
        boundary += [edge for edge in be if edge != (v, u)]
        following = dict(boundary)
        if len(following) != len(boundary) or len(set(following.values())) != len(boundary):
            return None
        start = boundary[0][0]
        merged, current = [], start
        while current not in merged:
            merged.append(current)
            current = following[current]
        if current != start or len(merged) != len(boundary):
            return None
        points = vertices[merged]
        sides = np.roll(points, -1, axis=0) - points
        turns = np.cross(sides, np.roll(sides, -1, axis=0)) @ normal
        allowance = tolerance_m * float(np.max(np.linalg.norm(sides, axis=1)))
        if np.min(turns) < -allowance:
            return None
        expected = area(a, normal) + area(b, normal)
        if expected <= 0 or abs(area(merged, normal) - expected) > allowance * len(merged):
            return None
        return merged

    while True:
        owners = {}
        for index, polygon in polygons.items():
            for u, v in edges(polygon):
                owners.setdefault(tuple(sorted((u, v))), []).append(index)
        changed = False
        for adjacent in owners.values():
            for first, second in combinations(adjacent, 2):
                merged = union(first, second, normals[first])
                if merged is not None:
                    polygons[first] = merged
                    del polygons[second]
                    del normals[second]
                    changed = True
                    break
            if changed:
                break
        if not changed:
            return [vertices[polygon].copy() for polygon in polygons.values()]
