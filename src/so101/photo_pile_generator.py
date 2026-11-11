"""Generate fresh 20+ piece piles from photo templates or a pinned verified catalog."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import signal
import threading
import time
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from so101.cloud_checkpoint import atomic_json
from so101.lego_curriculum import load_catalog
from so101.lego_parts_catalog import catalog_read_lock
from so101.lego_scene import assemble_scene, restore_scene, settle_data
from so101.pile_task_generator import resource_check


LOGGER = logging.getLogger(__name__)
PROFILE = "photo-demo-visual-hull-v1"
PHOTO_TEMPLATE_SHA256 = "ae154b5b6c938c8b55867dea647338de1eb52aec4d3ca6ee8ce5ab1117e64543"
PHOTO_PART_IDS = ("3001", "3002", "3003", "3004", "3020", "3022", "3700", "3702")
CATALOG_PROFILE = "catalog-native-surface-piles-v1"


def pin_catalog(manifest: Path, destination: Path) -> Path:
    pinned = destination / "catalog.json"
    if pinned.exists():
        load_catalog(pinned)
        return pinned
    destination.parent.mkdir(parents=True, exist_ok=True)
    with catalog_read_lock(manifest):
        raw = manifest.read_bytes()
        report = json.loads(raw)
        entries = catalog_candidates(report)
        with TemporaryDirectory(prefix=".catalog-", dir=destination.parent) as directory:
            staging = Path(directory) / "catalog"
            staging.mkdir()
            for entry in entries:
                for asset in entry["assets"]:
                    relative = Path(asset["path"])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Catalog asset escapes snapshot")
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(manifest.parent / relative, target)
            atomic_json(staging / "catalog.json", {
                **report, "parts": entries,
                "parent_manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "selection_profile": "catalog-diverse-v1",
            })
            load_catalog(staging / "catalog.json")
            staging.rename(destination)
    return pinned


def catalog_candidates(report: dict) -> list[dict]:
    excluded = ("duplo", "modulex", "quatro", "mursten", "assembly", "electric",
                "cloth", "flexible", "sticker")
    return [
        part for part in report["parts"]
        if part["status"] == "verified"
        and part.get("decoration") == "undecorated"
        and any(value.startswith("Part UPDATE") for value in part.get("ldraw_org", []))
        and not any(word in part["description"].lower() for word in excluded)
        and part["family"] in (
            "brick", "plate", "tile", "slope", "round", "technic", "minifigure", "other")
        and .004 <= max(part["dimensions_m"]) <= .065
        and min(part["dimensions_m"]) >= .002
        and part["qualification"]["collision_geom_count"] <= 200
    ]


def prepare_catalog_model(tree: ET.Element, manifest: Path, count: int, seed: int):
    catalog = load_catalog(manifest)
    entries = catalog_candidates(json.loads(manifest.read_bytes()))
    if len(entries) < count:
        raise ValueError("Diverse piles require at least one distinct eligible part per piece")
    rng = np.random.default_rng(seed)
    families = {}
    for entry in entries:
        families.setdefault(entry["family"], []).append(entry)
    chosen = []
    while len(chosen) < count:
        for family in rng.permutation(sorted(families)):
            candidates = families[family]
            if candidates and len(chosen) < count:
                chosen.append(candidates.pop(int(rng.integers(len(candidates)))))
    rng.shuffle(chosen)
    world = tree.find("worldbody")
    for body in list(world.findall("body")):
        world.remove(body)
    assets = tree.find("asset")
    if assets is not None:
        for asset in assets.findall("mesh"):
            assets.remove(asset)
    compiler = tree.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(tree, "compiler")
    compiler.set("angle", "radian")
    size = tree.find("size")
    if size is None:
        size = ET.SubElement(tree, "size")
    size.attrib.clear()
    size.set("memory", "512M")
    option = tree.find("option")
    option.set("jacobian", "sparse")
    option.set("cone", "pyramidal")
    option.set("noslip_iterations", "0")
    objects = []
    for index, entry in enumerate(chosen):
        radius = .07 * math.sqrt(float(rng.random()))
        angle, yaw = rng.uniform(-math.pi, math.pi, 2)
        objects.append({
            "instance_id": f"brick_{index}", "part_id": entry["part_id"],
            "position_relative_to_scene_m": [
                radius * math.cos(angle), radius * math.sin(angle), .04 + .025 * index],
            "yaw_rad": float(yaw), "color": "gray",
        })
    plan = {"catalog_sha256": catalog.manifest_sha256, "catalog_id": catalog.catalog_id,
            "status": "proposed", "objects": objects}
    xml = assemble_scene(ET.tostring(tree, encoding="unicode"), catalog, plan,
                         scene_origin_m=(0., 0., 0.))
    placements = [
        {"instance_id": obj["instance_id"], "part_id": obj["part_id"],
         "position": obj["position_relative_to_scene_m"],
         "quaternion": [math.cos(obj["yaw_rad"] / 2), 0., 0., math.sin(obj["yaw_rad"] / 2)]}
        for obj in objects
    ]
    return xml, placements, catalog.manifest_sha256


class RejectedPile(ValueError):
    """A fresh placement failed the unchanged photo-demo physical criteria."""


def prepare_model(source: Path, count: int, seed: int,
                  sampling: str = "original", catalog_path: Path | None = None
                  ) -> tuple[str, dict, list[dict]]:
    if not 20 <= count <= 90:
        raise ValueError("Photo piles must contain between 20 and 90 pieces")
    original = json.loads((source / "receipt.json").read_text())
    source_xml = (source / "model.xml").read_text()
    if hashlib.sha256(source_xml.encode()).hexdigest() != original["model_xml_sha256"]:
        raise ValueError("Photo source model does not match its receipt")
    tree = ET.fromstring(source_xml)
    if sampling == "catalog-diverse-v1":
        if catalog_path is None:
            raise ValueError("Diverse sampling requires a pinned catalog")
        xml, placements, catalog_sha = prepare_catalog_model(tree, catalog_path, count, seed)
        return xml, {**original, "catalog_sha256": catalog_sha}, placements
    world = tree.find("worldbody")
    if world is None:
        raise ValueError("Photo model has no worldbody")
    bodies = world.findall("body")
    if len(bodies) != 90:
        raise ValueError("Expected the retained 90-piece photo model")
    if sampling not in ("original", "common-basic-v1"):
        raise ValueError("Unknown photo-pile sampling profile")
    rng = np.random.default_rng(seed)
    if sampling == "common-basic-v1":
        if original["model_xml_sha256"] != PHOTO_TEMPLATE_SHA256:
            raise ValueError("Common-piece mapping requires the exact verified original photo model")
        basic_count = round(.9 * count)
        choices = np.r_[rng.integers(0, 6, size=basic_count),
                        rng.integers(6, 8, size=count - basic_count)]
        rng.shuffle(choices)
        selected = []
        for index, choice in enumerate(choices):
            body = copy.deepcopy(bodies[int(choice)])
            old_name = body.attrib["name"]
            for node in body.iter():
                if "name" in node.attrib:
                    node.set("name", node.attrib["name"].replace(old_name, f"brick_{index}", 1))
            selected.append(body)
    else:
        choices = np.arange(count) % len(PHOTO_PART_IDS)
        selected = bodies[:count]
    for body in bodies:
        world.remove(body)
    for body in selected:
        world.append(body)
    used_meshes = {geom.attrib["mesh"] for geom in world.iter("geom") if "mesh" in geom.attrib}
    assets = tree.find("asset")
    if assets is not None:
        for mesh in assets.findall("mesh"):
            if mesh.attrib["name"] not in used_meshes:
                assets.remove(mesh)
    placements = []
    for index, body in enumerate(selected):
        radius = .07 * math.sqrt(float(rng.random()))
        angle, yaw = rng.uniform(-math.pi, math.pi, 2)
        position = [radius * math.cos(angle), radius * math.sin(angle), .04 + .025 * index]
        quaternion = [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]
        body.set("pos", " ".join(map(str, position)))
        body.set("quat", " ".join(map(str, quaternion)))
        placements.append({"instance_id": body.attrib["name"], "position": position,
                           "quaternion": quaternion,
                           "part_id": (PHOTO_PART_IDS[int(choices[index])]
                                       if original["model_xml_sha256"] == PHOTO_TEMPLATE_SHA256
                                       else "catalog-demo")})
    return ET.tostring(tree, encoding="unicode"), original, placements


def vertical_bounds(model, data, geom: int) -> tuple[float, float]:
    rotation = data.geom_xmat[geom].reshape(3, 3)[2]
    center = data.geom_xpos[geom, 2] + rotation @ model.geom_aabb[geom, :3]
    half = np.abs(rotation) @ model.geom_aabb[geom, 3:]
    return float(center - half), float(center + half)


def verify_scene(output: Path) -> dict:
    receipt = json.loads((output / "receipt.json").read_text())
    xml = (output / "model.xml").read_text()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    restore_scene(model, data, receipt, xml)
    with np.load(output / "settled-state.npz") as saved:
        if not (np.array_equal(data.qpos, saved["qpos"])
                and np.array_equal(data.qvel, saved["qvel"])):
            raise ValueError("Published photo-pile state differs from its receipt")
    if model.nbody - 1 != receipt["piece_count"] or receipt["piece_count"] < 20:
        raise ValueError("Published photo-pile piece count is invalid")
    return receipt


def generate_scene(source: Path, output: Path, *, count: int, seed: int,
                   check_stop=lambda: None, progress=lambda value: None,
                   sampling: str = "original", color_mode: str = "original",
                   catalog_path: Path | None = None) -> dict:
    xml, original, placements = prepare_model(source, count, seed, sampling, catalog_path)
    if color_mode not in ("original", "single", "few", "many"):
        raise ValueError("Unknown photo-pile color mode")
    if color_mode != "original":
        tree = ET.fromstring(xml)
        palette = (".8 .04 .04 1", ".04 .48 .12 1", ".04 .2 .8 1",
                   ".95 .72 .04 1", ".93 .34 .04 1", ".38 .39 .4 1")
        color_rng = np.random.default_rng(np.random.SeedSequence([seed, 913]))
        palette = [palette[i] for i in color_rng.permutation(len(palette))]
        limit = {"single": 1, "few": 2, "many": len(palette)}[color_mode]
        for body, placement in zip(tree.findall("worldbody/body"), placements):
            color = palette[int(color_rng.integers(limit))]
            placement["color_rgba"] = [float(value) for value in color.split()]
            for geom in body.findall("geom"):
                geom.set("rgba", color)
        xml = ET.tostring(tree, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    if model.nbody != count + 1 or model.nq != count * 7 or model.nv != count * 6:
        raise ValueError("Photo source must contain only free dynamic piece bodies")
    for index in range(count):
        data.qpos[index * 7:index * 7 + 3] = [3. + index, 0., 1.]
    parked = data.qpos.copy()
    started = time.monotonic()
    cpu_started = time.process_time()
    cpu_budget = (30. if sampling == "catalog-diverse-v1" else 6.) * count
    wall_budget = (120. if sampling == "catalog-diverse-v1" else 24.) * count
    drop_interval = .15
    settling_seconds = .5 * count

    def check_budget():
        check_stop()
        if time.process_time() - cpu_started > cpu_budget:
            raise RejectedPile(f"{cpu_budget:g}-second CPU budget reached")
        if time.monotonic() - started > wall_budget:
            raise RejectedPile(f"{wall_budget:g}-second wall budget reached")

    maximum_penetration = 0.
    for index, placement in enumerate(placements):
        check_budget()
        data.qpos[index * 7:index * 7 + 3] = placement["position"]
        data.qpos[index * 7 + 2] = 0.
        mujoco.mj_forward(model, data)
        active_top = 0.
        own_bottom = math.inf
        own_bounds = []
        active_bounds = []
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            if body == 0 or body > index + 1:
                continue
            lower, upper = vertical_bounds(model, data, geom)
            if body == index + 1:
                own_bottom = min(own_bottom, lower)
            else:
                active_top = max(active_top, upper)
            if sampling == "catalog-diverse-v1":
                rotation = data.geom_xmat[geom].reshape(3, 3)
                center = data.geom_xpos[geom] + rotation @ model.geom_aabb[geom, :3]
                half = np.abs(rotation) @ model.geom_aabb[geom, 3:]
                (own_bounds if body == index + 1 else active_bounds).append(
                    (center - half, center + half))
        if not math.isfinite(own_bottom):
            raise ValueError("Photo piece has no collision geometry")
        if sampling == "catalog-diverse-v1":
            own_lower = np.min([bounds[0] for bounds in own_bounds], axis=0)
            own_upper = np.max([bounds[1] for bounds in own_bounds], axis=0)
            active_top = max((
                float(upper[2]) for lower, upper in active_bounds
                if np.all(np.minimum(own_upper[:2], upper[:2])
                          > np.maximum(own_lower[:2], lower[:2]))
            ), default=0.)
        data.qpos[index * 7 + 2] = active_top - own_bottom + .0001
        placement["insertion_position"] = data.qpos[index * 7:index * 7 + 3].tolist()
        deadline = data.time + drop_interval
        steps = 0
        while data.time < deadline:
            if steps % 100 == 0:
                check_budget()
            mujoco.mj_forward(model, data)
            penetration = max((max(0., -float(c.dist)) for c in data.contact), default=0.)
            maximum_penetration = max(maximum_penetration, penetration)
            if penetration > .005:
                raise RejectedPile(f"Insertion {index + 1}: penetration {penetration}m")
            if not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc)):
                raise RejectedPile("Nonfinite insertion physics")
            if any(warning.number for warning in data.warning):
                raise RejectedPile("MuJoCo warning during insertion")
            mujoco.mj_step(model, data)
            # Pending pieces are not inserted yet; reset only those bodies after each step.
            data.qpos[(index + 1) * 7:] = parked[(index + 1) * 7:]
            data.qvel[(index + 1) * 6:] = 0.
            steps += 1
        progress({"inserted": index + 1, "piece_count": count,
                  "phase": "adding", "simulated_seconds": data.time, "contacts": data.ncon})
    objects = [{"instance_id": p["instance_id"], "part_id": p["part_id"]} for p in placements]
    plan = {"scene_id": output.name, "catalog_sha256": original["catalog_sha256"],
            "objects": objects}
    try:
        progress({"inserted": count, "piece_count": count, "phase": "final_settling",
                  "simulated_seconds": data.time, "contacts": data.ncon})
        _, _, receipt = settle_data(
            model, data, xml, plan, maximum_seconds=settling_seconds, stationary_seconds=.5,
            velocity_tolerance=.0005, contact_limit_m=.005,
            stationarity_metric="surface-speed-v2", contact_solver="model-default",
            check_stop=check_budget,
        )
    except ValueError as error:
        if not str(error).startswith(("Scene contact penetration", "Non-finite catalog",
                                      "MuJoCo emitted", "Catalog scene did not settle")):
            raise
        raise RejectedPile(str(error)) from error
    receipt.update({
        "profile": CATALOG_PROFILE if sampling == "catalog-diverse-v1" else PROFILE,
        "source": str(source),
        "source_model_sha256": original["model_xml_sha256"],
        "source_receipt_sha256": hashlib.sha256((source / "receipt.json").read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "piece_count": count, "seed": seed, "initial_placements": placements,
        "sampling": sampling,
        "color_mode": color_mode,
        "color_sampling": "independent-per-instance-v1" if color_mode != "original" else "source-template",
        "replayed_from_initial_xml": False, "restored_original_integration_state": False,
        "initialization": "Collective low-height drops; only pending pieces parked/reset until insertion",
        "generation_recipe": (
            "collective-local-support-v1" if sampling == "catalog-diverse-v1"
            else "collective-low-drop-v1"),
        "drop_interval_simulated_seconds": drop_interval,
        "final_settling_budget_simulated_seconds": settling_seconds,
        "cpu_budget_seconds": cpu_budget, "wall_budget_seconds": wall_budget,
        "cpu_seconds": time.process_time() - cpu_started,
        "catalog_path": str(catalog_path) if catalog_path else None,
        "catalog_manifest_sha256": original["catalog_sha256"],
        "unique_part_count": len({p["part_id"] for p in placements}),
        "qualification": (
            "Verified catalog triangle-prism surfaces; pile acceptance is not solid-volume, clutch, RL or SFT qualification"
            if sampling == "catalog-diverse-v1" else
            "Photo-demo convex visual hulls; not native cavity, clutch, RL or SFT qualification"),
        "maximum_insertion_penetration_m": maximum_penetration,
        "wall_seconds": time.monotonic() - started,
    })
    check_stop()
    with TemporaryDirectory(prefix=".photo-", dir=output.parent) as temporary:
        staging = Path(temporary) / "scene"
        staging.mkdir()
        (staging / "model.xml").write_text(xml)
        np.savez(staging / "settled-state.npz", qpos=data.qpos, qvel=data.qvel)
        atomic_json(staging / "receipt.json", receipt)
        verify_scene(staging)
        staging.rename(output)
    return receipt


def run(root: Path, source: Path, *, seed: int, count: int, attempts: int = 0,
        stopping: threading.Event | None = None) -> None:
    if not 20 <= count <= 90 or seed < 0 or attempts < 0:
        raise ValueError("Require 20..90 pieces, nonnegative seed and attempt limit")
    root.mkdir(parents=True, exist_ok=True)
    stopping = stopping or threading.Event()
    config = {"profile": PROFILE, "source": str(source.resolve()), "seed": seed,
              "piece_count": count, "source_model_sha256":
              hashlib.sha256((source / "model.xml").read_bytes()).hexdigest(),
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (root / ".generator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = root / "generator.json"
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError("Existing photo pool belongs to a different generator configuration")
        else:
            atomic_json(config_path, config)
        sequence = 0
        while (root / f"attempt_{sequence:09d}.json").exists():
            sequence += 1

        next_resource_check = 0.

        def check_stop():
            nonlocal next_resource_check
            if stopping.is_set() or (root / "STOP").exists() or (root.parent / "STOP").exists():
                raise InterruptedError("Photo-pile producer stopped; original scenes preserved")
            if time.monotonic() >= next_resource_check:
                resource_check(root)
                next_resource_check = time.monotonic() + 2.

        completed = 0
        try:
            while not attempts or completed < attempts:
                try:
                    check_stop()
                except BlockingIOError as error:
                    atomic_json(root / "status.json", {"status": "paused", "reason": str(error)})
                    stopping.wait(5)
                    continue
                output = root / f"scene_{sequence:09d}"
                attempt = {"sequence": sequence, "seed": seed + sequence, "piece_count": count}

                def progress(value):
                    atomic_json(root / "status.json", {
                        "status": "generating", "pid": os.getpid(), **attempt, **value})

                progress({"inserted": 0})
                try:
                    receipt = (verify_scene(output) if output.exists() else generate_scene(
                        source, output, count=count, seed=seed + sequence,
                        check_stop=check_stop, progress=progress))
                    if any(receipt[key] != value for key, value in {
                        "profile": PROFILE, "seed": seed + sequence, "piece_count": count,
                        "source_model_sha256": config["source_model_sha256"],
                        "implementation_sha256": config["implementation_sha256"],
                    }.items()):
                        raise ValueError("Published photo scene does not match the proposal")
                    attempt.update(status="complete", scene=str(output),
                                   model_xml_sha256=receipt["model_xml_sha256"])
                except BlockingIOError as error:
                    atomic_json(root / "status.json", {"status": "paused", "reason": str(error)})
                    stopping.wait(5)
                    continue
                except RejectedPile as error:
                    LOGGER.warning("Photo proposal %s rejected: %s", sequence, error)
                    attempt.update(status="rejected", reason=str(error))
                atomic_json(root / f"attempt_{sequence:09d}.json", attempt)
                atomic_json(root / "status.json", {"pid": os.getpid(), **attempt})
                completed += 1
                sequence += 1
        except InterruptedError as error:
            atomic_json(root / "status.json", {"status": "stopped", "reason": str(error)})
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--pieces", type=int, default=20)
    parser.add_argument("--attempts", type=int, default=0, help="0 continuously generates")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    os.nice(19)
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    try:
        run(args.root, args.source, seed=args.seed, count=args.pieces,
            attempts=args.attempts, stopping=stopping)
    except (OSError, ValueError):
        LOGGER.exception("Photo-pile producer failed")
        raise


if __name__ == "__main__":
    main()
