"""Import official LDraw sources and publish immutable, qualified MuJoCo shells.

Run with --archive complete.zip --output /mnt/data/lego-catalog --catalog-id ID.
Use --previous CATALOG_JSON to retain verified parts in the next release.
Use --next-undecorated N with --previous to select the next N unverified plain
definitions in stable ID order. Decorated definitions are imported but never
converted by this initial, undecorated-only converter.
Each release includes the complete source library, not just converted parts.
Contact geometry is a compound of thin convex triangle prisms, not a whole-part
convex hull. This preserves recesses at the cost of many contact geoms. It is
not a claim of robot graspability or calibrated LEGO material properties.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np


SOURCE_URL = "https://library.ldraw.org/library/updates/complete.zip"
CONVERTER_VERSION = "surface-prisms-v2"
LDU_M = 0.0004
CONTACT_HALF_THICKNESS_M = 0.00002
FRAME = np.diag([1.0, -1.0, -1.0])
DEFAULT_PARTS = ("54200", "3062b", "3626c", "3005", "3024", "3070b", "3700")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe source path: {name}")
    return path


def metadata(text: str) -> dict:
    lines = text.splitlines()
    result = {"description": lines[0][2:].strip() if lines else "",
              "authors": [], "licenses": [], "ldraw_org": []}
    for line in lines:
        for prefix, field in (("0 Author:", "authors"), ("0 !LICENSE ", "licenses"),
                              ("0 !LDRAW_ORG ", "ldraw_org")):
            if line.startswith(prefix):
                result[field].append(line[len(prefix):].strip())
    return result


def family(description: str) -> str:
    text = description.lower()
    for word, label in (("minifig", "minifigure"), ("technic", "technic"),
                        ("slope", "slope"), ("round", "round"),
                        ("tile", "tile"), ("plate", "plate"), ("brick", "brick")):
        if word in text:
            return label
    return "other"


def decoration(description: str, part_id: str = "") -> str:
    if re.search(r"\b(sticker|stickers|stickered)\b", description, re.IGNORECASE):
        return "sticker"
    if (re.search(r"\b(pattern|patterned|printed|printing)\b", description, re.IGNORECASE)
            or re.fullmatch(r"\d+[a-z]*p[0-9a-z]+", part_id, re.IGNORECASE)):
        return "print"
    return "undecorated"


class Library:
    def __init__(self, root: Path):
        self.root = root
        self.files = {str(p.relative_to(root)).lower(): p
                      for p in root.rglob("*") if p.is_file()}

    def resolve(self, reference: str) -> str:
        name = str(safe_path(reference)).lower()
        for candidate in ("parts/" + name, "p/" + name, name):
            if candidate in self.files:
                return candidate
        raise ValueError(f"Missing LDraw dependency: {reference}")

    def triangles(self, part_id: str) -> tuple[np.ndarray, list[dict]]:
        dependencies: dict[str, dict] = {}

        def visit(key: str, transform: np.ndarray, offset: np.ndarray,
                  inverted: bool, stack: tuple[str, ...]) -> list[np.ndarray]:
            if key in stack:
                raise ValueError(f"Cyclic LDraw dependency: {' -> '.join((*stack, key))}")
            path = self.files[key]
            text = path.read_text(encoding="utf-8-sig")
            dependencies[key] = {"path": "source/ldraw/" + path.relative_to(self.root).as_posix(),
                                 "sha256": sha256(path), **metadata(text)}
            triangles = []
            clockwise = False
            invert_next = False
            for line in text.splitlines():
                fields = line.split()
                if not fields:
                    continue
                kind = fields[0]
                if kind == "0":
                    if "!TEXMAP" in fields or "!LPE" in fields:
                        raise ValueError(f"Unsupported textured/generated geometry in {key}")
                    if fields[1:3] == ["BFC", "INVERTNEXT"]:
                        invert_next = True
                    elif fields[1:2] == ["BFC"]:
                        if "CW" in fields:
                            clockwise = True
                        elif "CCW" in fields:
                            clockwise = False
                    continue
                if kind == "1":
                    if len(fields) < 15:
                        raise ValueError(f"Malformed subfile line in {key}: {line}")
                    values = np.array([float(v) for v in fields[2:14]])
                    matrix = values[3:].reshape(3, 3)
                    child = self.resolve(" ".join(fields[14:]))
                    triangles.extend(visit(
                        child, transform @ matrix, offset + transform @ values[:3],
                        inverted ^ invert_next ^ (np.linalg.det(matrix) < 0),
                        (*stack, key)))
                    invert_next = False
                elif kind in ("3", "4"):
                    count = int(kind)
                    if len(fields) != 2 + 3 * count:
                        raise ValueError(f"Malformed face in {key}: {line}")
                    points = np.array([float(v) for v in fields[2:]]).reshape(count, 3)
                    points = points @ transform.T + offset
                    for indices in ((0, 1, 2),) if count == 3 else ((0, 1, 2), (0, 2, 3)):
                        triangle = points[list(indices)]
                        if clockwise ^ inverted:
                            triangle = triangle[[0, 2, 1]]
                        triangles.append(triangle)
                elif kind not in ("2", "5"):
                    raise ValueError(f"Unsupported LDraw line type {kind} in {key}")
            return triangles

        triangles = np.asarray(visit(self.resolve(part_id + ".dat"),
                                    np.eye(3), np.zeros(3), False, ()))
        if triangles.size == 0 or not np.isfinite(triangles).all():
            raise ValueError(f"No finite surface geometry for {part_id}")
        triangles = triangles @ FRAME.T * LDU_M
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0]), axis=1)
        triangles = triangles[areas > 1e-14]
        # Duplicate faces only increase contacts, not geometric fidelity.
        unique = {}
        for triangle in triangles:
            key = tuple(sorted(tuple(v) for v in np.round(triangle, 10)))
            unique.setdefault(key, triangle)
        return np.asarray(list(unique.values())), list(dependencies.values())


def numbers(values: np.ndarray) -> str:
    return " ".join(f"{float(v):.12g}" for v in values.flat)


def prepare(library: Library, release: Path, part_id: str) -> dict:
    triangles, dependencies = library.triangles(part_id)
    low = triangles.min(axis=(0, 1))
    high = triangles.max(axis=(0, 1))
    dimensions = high - low
    if np.any(dimensions < 0.0001) or np.any(dimensions > 1):
        raise ValueError(f"Implausible dimensions: {dimensions.tolist()}")
    signed_volume = np.einsum("ij,ij->i", triangles[:, 0],
                             np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
    volume = abs(float(signed_volume))
    if not 0 < volume <= float(np.prod(dimensions)) * 1.01:
        raise ValueError(f"Invalid oriented surface volume: {volume}")
    mass = volume * 1050
    center = (low + high) / 2
    # Explicit nominal inertia avoids summing overlapping surface-prism masses.
    inertia = mass / 12 * np.array([dimensions[1] ** 2 + dimensions[2] ** 2,
                                   dimensions[0] ** 2 + dimensions[2] ** 2,
                                   dimensions[0] ** 2 + dimensions[1] ** 2])
    directory = release / "parts" / part_id
    directory.mkdir(parents=True)
    root = ET.Element("mujoco", model="ldraw_" + part_id)
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="visual_0",
                  vertex=numbers(triangles.reshape(-1, 3)),
                  face=" ".join(str(i) for i in range(triangles.size // 3)))
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name="part")
    ET.SubElement(body, "freejoint", name="part_free")
    ET.SubElement(body, "inertial", pos=numbers(center), mass=f"{mass:.12g}",
                  diaginertia=numbers(inertia))
    ET.SubElement(body, "geom", name="visual_0", type="mesh", mesh="visual_0",
                  contype="0", conaffinity="0", group="2", rgba="0.8 0.25 0.1 1")
    for i, triangle in enumerate(triangles):
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal *= CONTACT_HALF_THICKNESS_M / np.linalg.norm(normal)
        vertices = np.concatenate((triangle + normal, triangle - normal))
        name = f"collision_{i}"
        ET.SubElement(asset, "mesh", name=name, vertex=numbers(vertices))
        ET.SubElement(body, "geom", name=name, type="mesh", mesh=name,
                      group="3", rgba="0.8 0.25 0.1 0",
                      friction="0.5 0.005 0.0001", condim="3",
                      solref="0.004 1", solimp="0.99 0.999 0.0001")
    path = directory / "part.xml"
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return {"mjcf": str(path.relative_to(release)),
            "assets": [{"path": str(path.relative_to(release)), "sha256": sha256(path)}],
            "dimensions_m": dimensions.tolist(),
            "source_dependencies": dependencies,
            "physics": {"density_kg_m3": 1050, "mass_kg": mass,
                        "volume_m3": volume, "inertia_model": "bounding_box_at_bbox_center",
                        "center_of_mass_m": center.tolist(),
                        "friction": [0.5, 0.005, 0.0001],
                        "contact_solref": [0.004, 1],
                        "contact_solimp": [0.99, 0.999, 0.0001]},
            "geometry": {"visual_triangles": len(triangles),
                         "collision_method": "compound_triangle_prisms",
                         "surface_half_thickness_m": CONTACT_HALF_THICKNESS_M,
                         "frame_ldraw_to_mujoco": FRAME.tolist(), "metres_per_ldu": LDU_M,
                         "bounds_m": [low.tolist(), high.tolist()]}}


def qualify(release: Path, entry: dict) -> dict:
    path = release / entry["mjcf"]
    compile_started = time.perf_counter()
    standalone = mujoco.MjModel.from_xml_path(str(path))
    compile_wall_s = time.perf_counter() - compile_started
    if standalone.nq != 7 or standalone.nv != 6 or standalone.nbody != 2:
        raise ValueError("Part template must contain exactly one free dynamic body")
    root = ET.parse(path).getroot()
    ET.SubElement(root, "option", timestep="0.001", gravity="0 0 -9.81",
                  integrator="implicitfast", iterations="100")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="qualification_table", type="plane",
                  size="1 1 0.01", friction="0.5 0.005 0.0001",
                  solref="0.004 1", solimp="0.99 0.999 0.0001")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "qualification_table")
    bounds = np.array(entry["geometry"]["bounds_m"])
    radius = float(np.linalg.norm(bounds, axis=1).max())
    runs = []
    for quaternion in ((1, 0, 0, 0), (math.sqrt(.5), math.sqrt(.5), 0, 0),
                       (0, 1, 0, 0)):
        data = mujoco.MjData(model)
        data.qpos[:3] = [0, 0, radius + .004]
        data.qpos[3:7] = quaternion
        contact_steps = 0
        maximum_contacts = 0
        max_penetration = 0.0
        started = time.perf_counter()
        for step in range(3000):
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise ValueError("Non-finite qualification dynamics")
            contacts = [c for c in data.contact if table in (c.geom1, c.geom2)]
            maximum_contacts = max(maximum_contacts, len(contacts))
            if contacts:
                contact_steps += 1
                max_penetration = max(max_penetration, max(-float(c.dist) for c in contacts))
            if any(w.number for w in data.warning):
                raise ValueError("MuJoCo warning during qualification")
        speed = float(np.linalg.norm(data.qvel))
        if contact_steps < 100 or data.ncon == 0 or speed > .02 or max_penetration > .001:
            raise ValueError(f"Failed settling/contact qualification: speed={speed}, "
                             f"contact_steps={contact_steps}, penetration={max_penetration}")
        runs.append({"initial_quaternion": list(quaternion), "duration_s": 3,
                     "steps": 3000, "wall_seconds": time.perf_counter() - started,
                     "maximum_table_contacts": maximum_contacts,
                     "final_table_contacts": len(contacts),
                     "final_velocity_norm": speed, "table_contact_steps": contact_steps,
                     "max_penetration_m": max_penetration,
                     "final_qpos": data.qpos.tolist()})
    return {"version": "table-shell-v2", "mujoco_version": mujoco.__version__,
            "compile_wall_seconds": compile_wall_s,
            "collision_geom_count": standalone.ngeom - 1,
            "loadable": True, "finite_dynamics": True, "settling_runs": runs,
            "cavity_fidelity": {
                "method": "one_convex_prism_per_source_triangle",
                "maximum_surface_offset_m": CONTACT_HALF_THICKNESS_M,
                "whole_part_convex_hull": False},
            "limitations": ["Nominal ABS density and friction, not measured materials",
                            "Bounding-box inertia approximation",
                            "Thin surface contacts, not interlocking/stud clutch qualification",
                            "No robot graspability qualification"]}


def publish(archive: Path, output: Path, catalog_id: str,
            part_ids: tuple[str, ...] = DEFAULT_PARTS,
            previous: Path | None = None,
            next_undecorated: int | None = None) -> Path:
    if str(safe_path(catalog_id)) != catalog_id or "/" in catalog_id:
        raise ValueError("catalog_id must be a single directory name")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / catalog_id
    if destination.exists():
        raise FileExistsError(f"Immutable catalog already exists: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=".catalog-staging-", dir=output))
    try:
        source = staging / "source"
        source.mkdir()
        shutil.copyfile(archive, source / "complete.zip")
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                relative = safe_path(info.filename)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Source archive symlink: {info.filename}")
                path = source.joinpath(*relative.parts)
                if info.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(info) as reader, path.open("xb") as writer:
                        shutil.copyfileobj(reader, writer)
        library = Library(source / "ldraw")
        inventory = []
        entries = []
        for key, path in sorted(library.files.items()):
            details = metadata(path.read_text(encoding="utf-8-sig")) if key.endswith(".dat") else {}
            relative_source = "source/ldraw/" + path.relative_to(library.root).as_posix()
            inventory.append({"path": relative_source,
                              "sha256": sha256(path), **details})
            if not key.startswith("parts/") or "/" in key[len("parts/"):] or not key.endswith(".dat"):
                continue
            entries.append({"part_id": path.stem, **details,
                            "family": family(details["description"]), "status": "imported",
                            "decoration": decoration(details["description"], path.stem),
                            "source": {"path": relative_source, "sha256": sha256(path)}})
        write_json(source / "files.json", inventory)
        if previous is not None:
            validate(previous.parent)
            prior = json.loads(previous.read_text())
            if prior["source"]["archive_sha256"] != sha256(archive):
                raise ValueError("Previous catalog uses a different official source archive")
            if prior["converter_sha256"] != sha256(Path(__file__)):
                raise ValueError("Previous catalog uses a different converter; requalify parts")
            retained = {e["part_id"]: e for e in prior["parts"]
                        if e["status"] in ("verified", "prepared", "unavailable")}
            for index, entry in enumerate(entries):
                if entry["part_id"] not in retained:
                    continue
                old = retained[entry["part_id"]]
                entries[index] = old
                for asset in old.get("assets", []):
                    destination_asset = staging.joinpath(*safe_path(asset["path"]).parts)
                    destination_asset.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(previous.parent / asset["path"], destination_asset)
        if next_undecorated is not None:
            if next_undecorated <= 0:
                raise ValueError("next_undecorated must be positive")
            selected = {e["part_id"] for e in entries
                        if e["decoration"] == "undecorated" and e["status"] == "imported"}
            selected = set(sorted(selected)[:next_undecorated])
        else:
            selected = set(part_ids)
        absent = selected - {e["part_id"] for e in entries}
        if absent:
            raise ValueError(f"Unknown requested official part IDs: {sorted(absent)}")
        decorated = [e["part_id"] for e in entries
                     if e["part_id"] in selected and e["decoration"] != "undecorated"]
        if decorated:
            raise ValueError("Decorated conversion deferred until all undecorated variants "
                             f"are qualified; visual decoration is not yet supported: {decorated}")
        for entry in entries:
            if entry["part_id"] not in selected:
                continue
            if entry["status"] == "verified":
                continue
            print(f"Preparing {entry['part_id']}", flush=True)
            try:
                entry.update(prepare(library, staging, entry["part_id"]))
                entry["status"] = "prepared"
                entry["qualification"] = qualify(staging, entry)
                entry["status"] = "verified"
            except (ValueError, RuntimeError) as error:
                entry["status"] = "unavailable"
                entry["failure"] = {"stage": "qualification" if "mjcf" in entry else "preparation",
                                    "reason": str(error), "exception": type(error).__name__}
                print(f"Unavailable {entry['part_id']}: {error}", flush=True)
        license_files = [{"path": "source/ldraw/" + path.relative_to(library.root).as_posix(),
                          "sha256": sha256(path)}
                         for key, path in library.files.items()
                         if "readme" in key and "/" not in key]
        write_json(staging / "catalog.json", {
            "schema_version": 1, "catalog_id": catalog_id, "parts": entries,
            "converter_version": CONVERTER_VERSION,
            "converter_sha256": sha256(Path(__file__)),
            "source": {"url": SOURCE_URL, "archive_path": "source/complete.zip",
                       "archive_sha256": sha256(archive), "license_files": license_files,
                       "inventory": {"path": "source/files.json", "sha256": sha256(source / "files.json")},
                       "file_count": len(library.files)},
            "conversion_priority": "all_undecorated_before_decorated",
            "status_counts": {status: sum(e["status"] == status for e in entries)
                              for status in ("imported", "prepared", "verified", "unavailable")}})
        validate(staging)
        # Publish only complete artifacts. Consumers never observe staging files.
        for path in staging.rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        staging.chmod(0o555)
        os.rename(staging, destination)
        return destination / "catalog.json"
    except BaseException:
        for path in staging.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
        staging.chmod(0o755)
        shutil.rmtree(staging)
        raise


def validate(release: Path) -> None:
    catalog = json.loads((release / "catalog.json").read_text())
    if catalog["schema_version"] != 1:
        raise ValueError("Unsupported catalog schema")
    ids = set()
    for entry in catalog["parts"]:
        if entry["part_id"] in ids:
            raise ValueError(f"Duplicate part ID: {entry['part_id']}")
        ids.add(entry["part_id"])
        for asset in [entry["source"], *entry.get("source_dependencies", []),
                      *entry.get("assets", [])]:
            path = release.joinpath(*safe_path(asset["path"]).parts)
            if any(parent.is_symlink() for parent in (path, *path.parents)):
                raise ValueError(f"Symlink asset: {path}")
            if not path.is_file() or sha256(path) != asset["sha256"]:
                raise ValueError(f"Missing/corrupt asset: {path}")
        if entry["status"] == "verified":
            if entry["mjcf"] not in {a["path"] for a in entry["assets"]}:
                raise ValueError("MJCF missing from asset manifest")
            if len(entry["dimensions_m"]) != 3 or not entry.get("qualification"):
                raise ValueError("Verified part lacks qualification/dimensions")
            template = ET.parse(release / entry["mjcf"]).getroot()
            if template.findall(".//*[@file]") or template.findall(".//include"):
                raise ValueError("Converter templates must embed all MJCF mesh assets")
    archive = release / catalog["source"]["archive_path"]
    if sha256(archive) != catalog["source"]["archive_sha256"]:
        raise ValueError("Corrupt official source archive")
    inventory_asset = catalog["source"].get("inventory")
    if inventory_asset:
        inventory_path = release.joinpath(*safe_path(inventory_asset["path"]).parts)
        if inventory_path.is_symlink() or sha256(inventory_path) != inventory_asset["sha256"]:
            raise ValueError("Corrupt source inventory")
        inventory = json.loads(inventory_path.read_text())
        if len(inventory) != catalog["source"]["file_count"]:
            raise ValueError("Incomplete source inventory")
        for item in inventory:
            path = release.joinpath(*safe_path(item["path"]).parts)
            if path.is_symlink() or not path.is_file() or sha256(path) != item["sha256"]:
                raise ValueError(f"Corrupt inventoried source: {item['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--parts", nargs="*", default=list(DEFAULT_PARTS))
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--next-undecorated", type=int)
    args = parser.parse_args()
    path = publish(args.archive, args.output, args.catalog_id, tuple(args.parts),
                   args.previous, args.next_undecorated)
    print(json.dumps({"catalog": str(path), "sha256": sha256(path)}), flush=True)


if __name__ == "__main__":
    main()
