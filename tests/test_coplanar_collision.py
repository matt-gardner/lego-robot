import numpy as np
import pytest

from so101.coplanar_collision import merge_coplanar_triangles


def cell(x, y):
    a, b, c, d = [(x, y, 0), (x + 1, y, 0), (x + 1, y + 1, 0), (x, y + 1, 0)]
    return [a, b, c], [a, c, d]


def inside(point, polygon):
    sides = np.roll(polygon, -1, axis=0) - polygon
    return np.all(np.cross(sides, point - polygon)[:, 2] >= -1e-10)


@pytest.mark.parametrize("cells", [
    [(0, 0), (1, 0)],
    [(0, 0), (1, 0), (0, 1)],
    [(x, y) for x in range(3) for y in range(3) if (x, y) != (1, 1)],
    [(0, 0), (3, 0)],
])
def test_convex_merges_preserve_union_including_holes_concavities_and_gaps(cells):
    triangles = np.asarray([triangle for x, y in cells for triangle in cell(x, y)], dtype=float)
    rotation = np.array([[1., 0., 0.], [0., .6, -.8], [0., .8, .6]])
    offset = np.array([.2, -.4, .1])
    polygons = [(p - offset) @ rotation
                for p in merge_coplanar_triangles(triangles @ rotation.T + offset)]
    assert len(polygons) < len(triangles)
    assert sum(abs(np.cross(p, np.roll(p, -1, axis=0))[:, 2].sum()) / 2
               for p in polygons) == pytest.approx(len(cells))
    for x in np.arange(-.125, 4.25, .25):
        for y in np.arange(-.125, 3.25, .25):
            point = np.array([x, y, 0.])
            assert any(inside(point, p) for p in polygons) == any(
                inside(point, triangle) for triangle in triangles)


def test_folded_faces_and_nearby_disconnected_surfaces_are_not_flattened():
    triangles = np.array([
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
        [[1., 0., 0.], [0., 0., 0.], [0., 0., 1.]],
    ])
    assert len(merge_coplanar_triangles(triangles)) == 2


def test_opposite_winding_does_not_change_surface_union():
    triangles = np.array(cell(0, 0), dtype=float)
    triangles[1] = triangles[1, ::-1]
    polygons = merge_coplanar_triangles(triangles)
    assert len(polygons) == 1
    assert len(polygons[0]) == 4


def test_degenerate_triangles_are_explicit_errors():
    with pytest.raises(ValueError, match="Degenerate"):
        merge_coplanar_triangles(np.zeros((1, 3, 3)))
