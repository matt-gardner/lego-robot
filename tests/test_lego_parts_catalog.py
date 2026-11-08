import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from so101.lego_parts_catalog import (
    Library, decoration, prepare, publish, safe_path, settling_height, validate,
)
from so101 import lego_parts_catalog as catalog_module


def test_recursive_transform_scale_frame_and_provenance(tmp_path):
    (tmp_path / "parts").mkdir()
    (tmp_path / "p").mkdir()
    (tmp_path / "parts/1.dat").write_text(
        "0 Test\n0 !LICENSE Test license\n"
        "1 16 10 20 30 2 0 0 0 2 0 0 0 2 tri.dat\n")
    (tmp_path / "p/tri.dat").write_text(
        "0 Triangle\n3 16 0 0 0 1 0 0 0 1 0\n")
    triangles, deps = Library(tmp_path).triangles("1")
    np.testing.assert_allclose(triangles[0],
                               [[.004, .012, -.008], [.0048, .012, -.008],
                                [.004, .012, -.0088]])
    assert len(deps) == 2
    assert deps[0]["licenses"] == ["Test license"]
    assert all(len(d["sha256"]) == 64 for d in deps)


def test_dependency_cycle_and_missing_are_explicit(tmp_path):
    (tmp_path / "parts").mkdir()
    path = tmp_path / "parts/1.dat"
    path.write_text("0 Cycle\n1 16 0 0 0 1 0 0 0 1 0 0 0 1 1.dat\n")
    with pytest.raises(ValueError, match="Cyclic"):
        Library(tmp_path).triangles("1")
    path.write_text("0 Missing\n1 16 0 0 0 1 0 0 0 1 0 0 0 1 absent.dat\n")
    with pytest.raises(ValueError, match="Missing"):
        Library(tmp_path).triangles("1")


@pytest.mark.parametrize("path", ["../evil", "/evil", "a/../../evil"])
def test_path_traversal_rejected(path):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_path(path)


def test_import_publication_hashes_and_immutability(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("ldraw/parts/1.dat",
                        "0 Official test\n0 Author: Test\n0 !LICENSE Test license\n")
        bundle.writestr("ldraw/CAreadme.txt", "License")
    output = tmp_path / "catalogs"
    path = publish(archive, output, "release1", ())
    catalog = json.loads(path.read_text())
    assert catalog["parts"][0]["status"] == "imported"
    assert catalog["source"]["file_count"] == 2
    validate(path.parent)
    with pytest.raises(FileExistsError, match="Immutable"):
        publish(archive, output, "release1", ())
    source = path.parent / catalog["parts"][0]["source"]["path"]
    source.chmod(0o644)
    source.write_text("corrupted")
    with pytest.raises(ValueError, match="corrupt"):
        validate(path.parent)


def test_surface_prisms_preserve_cavity_and_convert_y_height_to_z(tmp_path):
    library_root = tmp_path / "ldraw"
    (library_root / "parts").mkdir(parents=True)
    (library_root / "parts/1.dat").write_text(
        "0 Tetrahedron\n0 BFC CERTIFY CCW\n"
        "3 16 0 0 0 0 20 0 10 0 0\n"
        "3 16 0 0 0 10 0 0 0 0 10\n"
        "3 16 0 0 0 0 0 10 0 20 0\n"
        "3 16 10 0 0 0 20 0 0 0 10\n")
    release = tmp_path / "release"
    entry = prepare(Library(library_root), release, "1")
    model = mujoco.MjModel.from_xml_path(str(release / entry["mjcf"]))
    assert model.ngeom == 5
    assert entry["geometry"]["collision_method"] == "compound_triangle_prisms"
    np.testing.assert_allclose(entry["dimensions_m"], [.004, .004, .008])
    # Every convex collider stays near its source face, not across the cavity.
    triangles, _ = Library(library_root).triangles("1")
    meshes = ET.parse(release / entry["mjcf"]).findall("asset/mesh")[1:]
    for triangle, mesh in zip(triangles, meshes, strict=True):
        vertices = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3)
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        offsets = np.abs((vertices - triangle[0]) @ normal)
        np.testing.assert_allclose(offsets, .00002, atol=1e-12)


def test_reflected_dependency_preserves_oriented_volume(tmp_path):
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts/1.dat").write_text(
        "0 Reflected\n1 16 0 0 0 -1 0 0 0 1 0 0 0 1 2.dat\n")
    (tmp_path / "parts/2.dat").write_text(
        "0 Triangle\n0 BFC CERTIFY CCW\n3 16 0 0 1 1 0 1 0 1 1\n")
    library = Library(tmp_path)
    original, _ = library.triangles("2")
    reflected, _ = library.triangles("1")
    def volume(triangles):
        return np.einsum("ij,ij->i", triangles[:, 0],
                         np.cross(triangles[:, 1], triangles[:, 2])).sum()
    assert volume(original) == pytest.approx(volume(reflected))


def test_unavailable_is_not_verified_and_archive_escape_rejected(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("ldraw/parts/1.dat",
                        "0 Missing\n1 16 0 0 0 1 0 0 0 1 0 0 0 1 missing.dat\n")
    path = publish(archive, tmp_path / "catalog", "failed", ("1",))
    entry = json.loads(path.read_text())["parts"][0]
    assert entry["status"] == "unavailable"
    assert entry["failure"]["stage"] == "preparation"
    assert "Missing LDraw dependency" in entry["failure"]["reason"]
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        publish(archive, tmp_path / "catalog", "unsafe", ())
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(("description", "part_id", "expected"), [
    ("Brick 1 x 1", "3005", "undecorated"),
    ("Minifig Head with Smile Pattern", "3626bp01", "print"),
    ("Tile with Sticker", "3068bd01", "sticker"),
    ("Alias", "973p123", "print"),
    ("Technic Pin", "3673", "undecorated"),
])
def test_decoration_priority_classification(description, part_id, expected):
    assert decoration(description, part_id) == expected


def test_decorated_conversion_is_deferred(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("ldraw/parts/1p01.dat", "0 Tile with Pattern\n")
    with pytest.raises(ValueError, match="Decorated conversion deferred"):
        publish(archive, tmp_path / "catalog", "decorated", ("1p01",))
    assert not (tmp_path / "catalog/decorated").exists()


def test_next_undecorated_skips_prints_and_requires_same_source(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("ldraw/parts/1p01.dat", "0 Tile with Pattern\n")
        bundle.writestr("ldraw/parts/2.dat", "0 Brick\n")
    output = tmp_path / "catalog"
    first = publish(archive, output, "first", ())
    second = publish(archive, output, "second", (), first, 1)
    entries = json.loads(second.read_text())["parts"]
    assert entries[0]["status"] == "imported"
    assert entries[1]["status"] == "unavailable"
    third = publish(archive, output, "third", (), second, 1)
    assert json.loads(third.read_text())["parts"] == entries
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr("ldraw/parts/3.dat", "0 Tile\n")
    with pytest.raises(ValueError, match="different official source"):
        publish(archive, output, "changed-source", (), first)


def test_non_library_archive_is_not_successfully_imported(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("unrelated.txt", "not an official library")
    with pytest.raises(ValueError, match="lacks the official"):
        publish(archive, tmp_path / "catalog", "invalid", ())


def test_releases_share_immutable_sources_and_require_explicit_requalification(tmp_path, monkeypatch):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("ldraw/parts/1.dat", "0 Brick\n")
    output = tmp_path / "catalog"
    first = publish(archive, output, "first", ())
    second = publish(archive, output, "second", (), first)
    for relative in ("source/complete.zip", "source/ldraw/parts/1.dat"):
        a, b = first.parent / relative, second.parent / relative
        assert a.stat().st_ino == b.stat().st_ino
        assert a.stat().st_dev == b.stat().st_dev
        assert not b.stat().st_mode & 0o222
    original_sha256 = catalog_module.sha256
    def changed_converter(path):
        return ("f" * 64 if path == Path(catalog_module.__file__)
                else original_sha256(path))
    monkeypatch.setattr(catalog_module, "sha256", changed_converter)
    with pytest.raises(ValueError, match="different converter"):
        publish(archive, output, "blocked", (), second)
    third = publish(archive, output, "requalified", (), second,
                    requalify_previous=True)
    assert json.loads(third.read_text())["converter_sha256"] == "f" * 64


def test_retry_unlinks_shared_assets_and_clears_failure_metadata(tmp_path, monkeypatch):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("ldraw/parts/1.dat",
                        "0 Tetrahedron\n0 BFC CERTIFY CCW\n"
                        "3 16 0 0 0 0 20 0 10 0 0\n"
                        "3 16 0 0 0 10 0 0 0 0 10\n"
                        "3 16 0 0 0 0 0 10 0 20 0\n"
                        "3 16 10 0 0 0 20 0 0 0 10\n")
    output = tmp_path / "catalog"
    monkeypatch.setattr(catalog_module, "qualify", lambda *args: {"test_fixture": True})
    first = publish(archive, output, "first", ("1",))
    def fail_qualification(*args):
        raise ValueError("Fixture contact failure")
    monkeypatch.setattr(catalog_module, "qualify", fail_qualification)
    second = publish(archive, output, "failed", (), first, requalify_previous=True)
    old_asset = second.parent / "parts/1/part.xml"
    old_hash = catalog_module.sha256(old_asset)
    assert json.loads(second.read_text())["parts"][0]["status"] == "unavailable"
    monkeypatch.setattr(catalog_module, "qualify", lambda *args: {"test_fixture": True})
    third = publish(archive, output, "retry", ("1",), second)
    retried = json.loads(third.read_text())["parts"][0]
    assert retried["status"] == "verified"
    assert "failure" not in retried
    assert (third.parent / "parts/1/part.xml").stat().st_ino != old_asset.stat().st_ino
    assert catalog_module.sha256(old_asset) == old_hash
    validate(first.parent)
    validate(second.parent)
    validate(third.parent)


@pytest.mark.parametrize(("quaternion", "expected_height"), [
    ((1., 0., 0., 0.), .006),
    ((2 ** -.5, 2 ** -.5, 0., 0.), .008),
    ((0., 1., 0., 0.), .084),
])
def test_long_parts_start_four_mm_above_table_in_each_orientation(quaternion, expected_height):
    bounds = np.array([[-.005, -.004, -.002], [.005, .004, .08]])
    assert settling_height(bounds, quaternion) == pytest.approx(expected_height)
