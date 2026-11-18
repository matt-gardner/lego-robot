"""Export the checked current inventory into canonical per-pile task directories."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path

from so101.bucket_store import current_image, current_inventory

SOURCE_FILES = ("model.mjb", "model.xml", "problem.json", "integration_state.json", "scene_plan.json")


@contextmanager
def initial_import(piles: Path):
    marker = piles / ".incomplete"
    marker.touch()
    yield
    marker.unlink()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    atomic_bytes(path, json.dumps(value, separators=(",", ":"), allow_nan=False).encode())


def atomic_bytes(path: Path, body: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    parent = path.parent.stat()
    os.chown(temporary, parent.st_uid, parent.st_gid)
    temporary.chmod(0o640)
    temporary.replace(path)


def make_directory(path: Path, root: Path) -> None:
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Canonical directory escapes its root")
    path.mkdir(parents=True, exist_ok=True)
    owner = root.stat()
    os.chown(path, owner.st_uid, owner.st_gid)
    path.chmod(0o2770)


def source_root(record: dict) -> Path:
    source = record["source"]
    path = Path(source.get("episode_source") or source["path"])
    if path.parent.name == "tasks":
        return path.parent.parent
    if path.name == "problem.json":
        return path.parent
    raise ValueError("Task source has no recognized exact scene root")


def arm_joint_addresses(model_path: Path) -> dict[str, int]:
    from mujoco import MjModel  # pyright: ignore[reportAttributeAccessIssue]

    model = MjModel.from_binary_path(str(model_path))
    result = {}
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        if joint < 0:
            raise ValueError("Robot actuator does not reference a joint")
        name = model.joint(joint).name
        if not name or int(model.jnt_type[joint]) not in (2, 3):
            raise ValueError("Robot actuator must reference a named scalar joint")
        result[name] = int(model.jnt_qposadr[joint])
    if not result:
        raise ValueError("Pile model has no actuated arm joints")
    del model
    gc.collect()
    return result


def hardlink(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError("Canonical source must be a regular non-symlink file")
    if destination.exists():
        if source.stat().st_ino != destination.stat().st_ino or source.stat().st_dev != destination.stat().st_dev:
            raise ValueError("Canonical destination already contains different bytes")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)


def receipts_by_task(roots: list[Path]) -> dict:
    receipts = defaultdict(list)
    for root in roots:
        for path in sorted((root / "receipts").glob("*.json")):
            body = path.read_bytes()
            receipt = json.loads(body)
            binding = receipt["source_binding"]
            if receipt["status"] not in ("complete", "unavailable", "timeout", "error"):
                raise ValueError("Assistance receipt is not a finalized outcome")
            levels = receipt["levels"]
            if receipt["status"] == "complete":
                if [row["value"] for row in levels] != [index / 20 for index in range(20)]:
                    raise ValueError("Complete assistance receipt has an unexpected level grid")
            elif levels:
                raise ValueError("Non-complete assistance receipt contains checkpoint levels")
            receipts[(binding["scene_id"], binding["task_id"])].append(
                (path, hashlib.sha256(body).hexdigest()))
    return receipts


def saved_checkpoints(document: dict) -> dict:
    checkpoints = defaultdict(list)
    for pilot in document["distance_pilots"]:
        for record in pilot["problems"]:
            checkpoints[record["tags"]["parent_task_id"]].append(record)
    return checkpoints


def attach_receipt(task: dict, directory: Path, root: Path,
                   path: Path, sha: str) -> bool:
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != sha:
        raise ValueError("Assistance receipt changed during export")
    receipt = json.loads(body)
    definition = task["task_definition"]
    binding = receipt["source_binding"]
    if (binding["scene_id"] != task["source"]["scene_id"]
            or binding["task_id"] != definition["task_id"]
            or binding["task_sha256"] != identity(definition)):
        raise ValueError("Assistance receipt belongs to a different source task")
    target = directory / "assistance" / f"{sha}.json"
    make_directory(target.parent, root)
    if not target.exists():
        atomic_bytes(target, body)
    if digest(target) != sha:
        raise ValueError("Assistance receipt changed during export")
    reference = {
        "path": f"../assistance/{sha}.json", "sha256": sha,
        "measurement_inventory_sha256": receipt["inventory_sha256"],
        "status": receipt["status"], "source_binding": binding,
        "retained_from": receipt.get("retained_from"),
        "checkpoint_states_persisted": False,
    }
    for existing in task["assistance_breakdowns"]:
        if existing["sha256"] == sha:
            if existing != reference:
                raise ValueError("Existing assistance reference differs from finalized receipt")
            return False
    task["assistance_breakdowns"].append(reference)
    return True


def refresh_assistance(root: Path, assistance_roots: list[Path]) -> dict:
    from so101.pile_directory import PILE_NAME, TASK_NAME, contained, read_json

    piles = contained(root, root / "piles")
    if (piles / ".incomplete").exists():
        raise ValueError("Canonical pile import is incomplete")
    read_json(contained(root, piles / "metadata.json"))
    pending = receipts_by_task(assistance_roots)
    result = {"receipts": sum(len(rows) for rows in pending.values()),
              "attached": 0, "tasks_updated": 0}
    for directory in sorted(piles.iterdir()):
        if not PILE_NAME.fullmatch(directory.name):
            continue
        contained(root, directory)
        tasks = contained(root, directory / "tasks")
        for path in sorted(tasks.iterdir()):
            if not TASK_NAME.fullmatch(path.name):
                continue
            task = read_json(contained(root, path))
            key = (task["source"]["scene_id"], task["task_definition"]["task_id"])
            rows = pending.pop(key, [])
            changed = False
            for receipt_path, sha in rows:
                if attach_receipt(task, directory, root, receipt_path, sha):
                    result["attached"] += 1
                    changed = True
            if changed:
                atomic_json(path, task)
                result["tasks_updated"] += 1
    if pending:
        raise ValueError(f"{len(pending)} finalized receipt tasks have no canonical parent")
    return result


def export_checkpoint(record: dict, directory: Path, root: Path) -> dict:
    source = source_root(record)
    raw = json.loads((source / "problem.json").read_bytes())
    scene = record["source"]["scene_id"]
    if raw["scene_id"] != scene:
        raise ValueError("Checkpoint source scene identity differs")
    definition = next(task for task in raw["tasks"]
                      if task["task_id"] == record["source"]["task_id"])
    state = definition.get("initial_state")
    if state is None:
        state = json.loads((source / "integration_state.json").read_bytes())
    if state["scene_id"] != scene:
        raise ValueError("Checkpoint initial state belongs to a different scene")
    destination = directory / "checkpoints" / record["id"]
    make_directory(destination, root)
    hashes = {}
    for name in SOURCE_FILES:
        hardlink(source / name, destination / name)
        hashes[name] = digest(destination / name)
    return {
        "id": record["id"], "directory": f"../checkpoints/{record['id']}",
        "source": record["source"], "files_sha256": hashes,
        "task_definition": definition, "initial_state": state,
        "task_assistance": record["tags"]["task_assistance"],
        "qualification": raw.get("qualification"),
        "physics_version": raw.get("physics_version"),
    }


def export(legacy: Path, root: Path, assistance_roots: list[Path]) -> dict:
    document = current_inventory(legacy)
    groups = defaultdict(list)
    for record in document["problems"]["problems"]:
        scene = record["source"].get("scene_id")
        if not scene:
            raise ValueError("Canonical tasks require an explicit source scene identity")
        groups[scene].append(record)
    rendered = {pile["scene_id"]: pile for pile in document["admitted_piles"]}
    receipts = receipts_by_task(assistance_roots)
    checkpoints = saved_checkpoints(document)
    parent_ids = {record["source"]["task_id"]
                  for records in groups.values() for record in records}
    if set(checkpoints) - parent_ids:
        raise ValueError("Saved assistance checkpoint has no primary parent task")
    root.mkdir(parents=True, exist_ok=True)
    piles = root / "piles"
    make_directory(piles, root)
    if (piles / "metadata.json").exists():
        raise ValueError("Initial import cannot replace an existing canonical inventory")
    result = {"piles": 0, "tasks": 0, "rendered": 0, "assistance_receipts": 0,
              "assistance_checkpoints": 0, "source_models_bytes": 0}
    with initial_import(piles):
        for index, (scene, records) in enumerate(sorted(groups.items()), 1):
            roots = {source_root(record) for record in records}
            if len(roots) != 1:
                raise ValueError("Scene identity references multiple source roots")
            source = roots.pop()
            raw = json.loads((source / "problem.json").read_bytes())
            state = json.loads((source / "integration_state.json").read_bytes())
            if raw["scene_id"] != scene:
                raise ValueError("Scene source identity differs")
            if state["scene_id"] != scene:
                plan = json.loads((source / "scene_plan.json").read_bytes())
                admission = json.loads((source / "admission.json").read_bytes())
                if (state["scene_id"] != plan.get("scene_id")
                        or any(task.get("initial_state", {}).get("scene_id") != scene
                               for task in raw["tasks"])
                        or state.get("model_xml_sha256") != digest(source / "model.xml")
                        or any(admission["files_sha256"][name] != digest(source / name)
                               for name in SOURCE_FILES)):
                    raise ValueError("Scene source/state identity differs without exact task overrides")
            definitions = {task["task_id"]: task for task in raw["tasks"]}
            addresses = arm_joint_addresses(source / "model.mjb")
            directory = piles / f"pile_{index:04d}"
            make_directory(directory, root)
            task_directory = directory / "tasks"
            make_directory(task_directory, root)
            hashes = {}
            for name in SOURCE_FILES:
                hardlink(source / name, directory / name)
                hashes[name] = digest(source / name)
            result["source_models_bytes"] += (source / "model.mjb").stat().st_blocks * 512
            instances = {instance["instance_id"]: instance for instance in raw["instances"]}
            for task_index, record in enumerate(records, 1):
                task_id = record["source"].get("task_id") or Path(record["source"]["episode_source"]).name
                definition = definitions[task_id]
                snapshot = definition.get("initial_state", state)
                if snapshot.get("scene_id") != scene:
                    raise ValueError("Task initial state belongs to a different pile")
                positions = {name: snapshot["qpos"][address] for name, address in addresses.items()}
                if any(not isinstance(value, (int, float)) or not math.isfinite(value)
                       for value in positions.values()):
                    raise ValueError("Arm joint position is not finite")
                instance = instances[definition["target_instance_id"]]
                exported = {
                    **record, "schema_version": 1, "task_key": f"task_{task_index:03d}",
                    "pile_id": directory.name, "task_definition": definition,
                    "piece_id": definition["target_instance_id"],
                    "part_id": instance.get("part_id"),
                    "initial_arm_position": positions,
                    "initial_state": definition.get("initial_state", {
                        "path": "../integration_state.json", "sha256": hashes["integration_state.json"]}),
                    "target_location": definition["goal"].get("destination_xy"),
                    "assistance_checkpoints": [],
                    "assistance_checkpoint_status": "not_persisted",
                    "assistance_breakdowns": [],
                }
                for checkpoint in checkpoints.get(task_id, []):
                    exported["assistance_checkpoints"].append(
                        export_checkpoint(checkpoint, directory, root))
                    result["assistance_checkpoints"] += 1
                if exported["assistance_checkpoints"]:
                    exported["assistance_checkpoint_status"] = "persisted"
                seen = set()
                for path, sha in receipts.get((scene, task_id), []):
                    if sha in seen:
                        continue
                    seen.add(sha)
                    attach_receipt(exported, directory, root, path, sha)
                    result["assistance_receipts"] += 1
                atomic_json(task_directory / f"task_{task_index:03d}.json", exported)
                result["tasks"] += 1
            pile = {
                "schema_version": 1, "id": directory.name, "scene_id": scene,
                "tags": {**raw.get("tags", {}), "piece_count": len(instances)},
                "qualification": "Source TRAIN reset; not expert/SFT or grasp-hold qualification",
                "source": {"original_root": str(source), "files_sha256": hashes},
                "model": "model.mjb", "model_xml": "model.xml",
                "initial_state": "integration_state.json", "rendered": None,
                "piece_coordinates": {}, "task_count": len(records),
            }
            if scene in rendered:
                original = rendered[scene]
                body, media = current_image(legacy, original["id"], 0)
                if media != "image/png":
                    raise ValueError("Canonical rendered.png requires the original PNG")
                destination = directory / "rendered.png"
                destination.write_bytes(body)
                pile["rendered"] = {
                    key: value for key, value in original["initial_images"][0].items()
                    if key not in ("path", "root_relative", "receipt_path")}
                pile["piece_coordinates"] = {
                    key: value for key, value in original.items()
                    if key == "pieces" or key.startswith("piece_")}
                result["rendered"] += 1
            atomic_json(directory / "pile.json", pile)
            result["piles"] += 1
        metadata = {
            "schema_version": 1, "manifest": document["manifest"],
            "buckets": document["problems"]["buckets"], "calibration": document["calibration"],
            "distance_pilots": document["distance_pilots"],
        }
        atomic_json(piles / "metadata.json", metadata)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dashboard-root", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assistance-root", type=Path, action="append", default=[])
    parser.add_argument("--refresh-assistance", action="store_true")
    args = parser.parse_args()
    if args.refresh_assistance:
        if not args.assistance_root:
            parser.error("--refresh-assistance requires --assistance-root")
        result = refresh_assistance(args.root, args.assistance_root)
    else:
        if args.legacy_dashboard_root is None:
            parser.error("initial import requires --legacy-dashboard-root")
        result = export(args.legacy_dashboard_root, args.root, args.assistance_root)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
