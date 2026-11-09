"""Export standalone native pile starts, without demonstrations or training admission."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from so101.catalog_sim import CatalogSimulation
from so101.lego_curriculum import load_catalog, prepare_scene
from so101.lego_scene import capture_scene_state, restore_scene
from so101.robot_sim import MODEL_ROOT


def support_graph(model: mujoco.MjModel, data: mujoco.MjData, names: list[str]) -> dict:
    """Measure contact connectivity to the table, not graspability or clutch fit."""
    table = model.geom("table").id
    bodies = {model.body(name).id: name for name in names}
    edges = set()
    maximum_penetration = 0.
    for contact in data.contact:
        maximum_penetration = max(maximum_penetration, max(0., -float(contact.dist)))
        if contact.dist > .0001:
            continue
        endpoints = [
            "table" if geom == table else bodies.get(int(model.geom_bodyid[geom]))
            for geom in (int(contact.geom1), int(contact.geom2))
        ]
        if all(endpoint is not None for endpoint in endpoints) and endpoints[0] != endpoints[1]:
            edges.add(tuple(sorted(endpoints)))
    reached = {"table"}
    while True:
        expanded = reached | {node for edge in edges if reached.intersection(edge) for node in edge}
        if expanded == reached:
            break
        reached = expanded
    return {
        "metric": "table-contact-connectivity-v1",
        "edges": sorted(edges), "table_connected_objects": sorted(reached - {"table"}),
        "unsupported_objects": sorted(set(names) - reached),
        "object_object_edges": sum("table" not in edge for edge in edges),
        "maximum_final_penetration_m": maximum_penetration,
    }


def export_pile(catalog, output: Path, *, count: int, seed: int, sequence: int) -> dict:
    plan = prepare_scene(catalog, seed=seed, sequence=sequence,
                         category="pile", task="grasp", object_count=count)
    # Keep the existing nonintersecting drop heights, but compact the footprint.
    for obj in plan["objects"]:
        obj["position_relative_to_scene_m"][:2] = [
            coordinate * .25 for coordinate in obj["position_relative_to_scene_m"][:2]
        ]
    plan["pile_generator"] = "native-settled-piles-v1"
    plan["scene_id"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["pair_id"] = f"{plan['scene_id']}:{plan['task']}"
    for index, command in enumerate(plan["commands"]):
        command["command_id"] = f"{plan['scene_id']}:{index}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "scene_plan.json").write_text(json.dumps(plan, sort_keys=True))
    # Native triangle-shell stacks need a larger constraint arena, not altered contacts.
    with TemporaryDirectory(prefix="native-pile-model-") as directory:
        model_root = Path(directory)
        robot = ET.parse(MODEL_ROOT / "so101.xml")
        size = robot.getroot().find("size")
        if size is None:
            size = ET.SubElement(robot.getroot(), "size")
        size.attrib.clear()
        size.set("memory", "256M")
        robot.write(model_root / "so101.xml")
        (model_root / "revision.json").symlink_to((MODEL_ROOT / "revision.json").resolve())
        (model_root / "assets").symlink_to((MODEL_ROOT / "assets").resolve())
        sim = CatalogSimulation(catalog, plan, model_root=model_root,
                                render=False, record_images=False,
                                contact_solver="pyramidal-v1",
                                stationarity_metric="surface-speed-v2")
    try:
        mujoco.mj_forward(sim.model, sim.data)
        names = [obj["instance_id"] for obj in plan["objects"]]
        support = support_graph(sim.model, sim.data, names)
        if support["unsupported_objects"]:
            raise ValueError(f"Objects not connected to table: {support}")
        if support["object_object_edges"] == 0:
            raise ValueError("Settled proposal has no measured inter-object contacts")
        (output / "model.xml").write_text(sim.xml)
        mujoco.mj_saveModel(sim.model, str(output / "model.mjb"), None)
        snapshot = {
            **sim.settled_snapshot,
            **capture_scene_state(sim.model, sim.data, sim.xml, plan, status="settled"),
        }
        (output / "integration_state.json").write_text(json.dumps(snapshot, sort_keys=True))
        restored_model = mujoco.MjModel.from_binary_path(str(output / "model.mjb"))
        restored_data = mujoco.MjData(restored_model)
        restore_scene(restored_model, restored_data, snapshot, sim.xml)
        for field in ("qpos", "qvel", "ctrl", "act"):
            if not np.array_equal(getattr(restored_data, field), getattr(sim.data, field)):
                raise ValueError(f"Saved pile round trip changed {field}")
        if support_graph(restored_model, restored_data, names) != support:
            raise ValueError("Saved pile round trip changed support contacts")
        result = {
            "schema_version": 1, "generator": "native-settled-piles-v1",
            "scene_id": plan["scene_id"], "object_count": count,
            "part_ids": [obj["part_id"] for obj in plan["objects"]],
            "catalog_sha256": catalog.manifest_sha256,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "constraint_memory_bytes": 256 * 1024 * 1024,
            "physics_version": mujoco.__version__,
            "stationarity": snapshot["stationarity"], "support": support,
            "configuration": sim.configuration_metrics(),
            "files_sha256": {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in ("model.xml", "model.mjb", "integration_state.json", "scene_plan.json")
            },
            "qualification": (
                "Standalone simulated settled start with table-connected contacts and exact "
                "binary-model state restoration. Not a successful action reference, RL reset "
                "adapter, grasp/goal solvability, clutch fit, calibrated material or transfer claim. "
                "XML references pinned catalog/robot assets; binary model contains runtime settings."
            ),
        }
        (output / "receipt.json").write_text(json.dumps(result, sort_keys=True))
        return result
    finally:
        sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", nargs="+", required=True)
    parser.add_argument("--counts", nargs="+", type=int, default=[3, 5, 8])
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    catalog = load_catalog(args.catalog)
    requested = set(args.parts)
    selected = tuple(part for part in catalog.parts if part.part_id in requested)
    if {part.part_id for part in selected} != requested:
        raise ValueError("Requested parts are not all verified catalog members")
    if any(count < 2 for count in args.counts):
        raise ValueError("Pile counts must be at least two")
    catalog = replace(catalog, parts=selected)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for sequence, count in enumerate(args.counts):
        result = export_pile(catalog, args.output / f"pile_{sequence:04d}",
                             count=count, seed=args.seed, sequence=sequence)
        results.append(result)
        print(json.dumps({"scene_id": result["scene_id"], "count": count}), flush=True)
    (args.output / "completion.json").write_text(json.dumps({
        "schema_version": 1, "generator": "native-settled-piles-v1",
        "catalog_path": str(args.catalog.resolve()), "piles": results,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
