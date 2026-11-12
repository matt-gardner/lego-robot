"""Freeze paired, scene-stratified GRASP assistance rollout inputs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import tarfile
import xml.etree.ElementTree as ET


def select_entries(entries, per_size, seed):
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    for entry, scene in entries:
        if entry["split"] != "train":
            continue
        tasks = scene["tasks"]
        parents = {task.get("parent_task_id") for task in tasks}
        levels = [task.get("assistance", {}).get("value") for task in tasks]
        if (len(parents) != 1 or None in parents or len(tasks) != 20
                or sorted(levels) != [tick / 20 for tick in range(20)]
                or any(task["task_type"] != "grasp" or
                       task["assistance"].get("kind") != "original-pose-grasp-v1"
                       for task in tasks)):
            raise ValueError("Admission is not one complete original-pose GRASP parent")
        parent = scene["provenance"]["parent_entry"]
        identity = next(iter(parents))
        if identity in seen:
            raise ValueError("Duplicate original parent in admission population")
        seen.add(identity)
        groups[entry["tags"]["piece_count"]][parent["scene_family_id"]].append((entry, scene))
    rng = random.Random(seed)
    selected = []
    for size, families in sorted(groups.items()):
        keys = sorted(families)
        rng.shuffle(keys)
        candidates_by_family = []
        for key in keys:
            candidates = sorted(families[key], key=lambda item: item[0]["root"])
            rng.shuffle(candidates)
            candidates_by_family.append(candidates)
        choices = [rows[index] for index in range(max(map(len, candidates_by_family)))
                   for rows in candidates_by_family if index < len(rows)]
        selected.extend(entry for entry, _ in choices[:per_size])
    if not selected:
        raise ValueError("No TRAIN GRASP admissions")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admissions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parents-per-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17551000)
    args = parser.parse_args()
    if args.parents_per_size < 1:
        raise ValueError("Need a positive matched-parent population")
    content = args.admissions.read_bytes()
    index = json.loads(content)
    rows = []
    files = {args.admissions}
    for name in index["admissions"]:
        admission = Path(name)
        entry = json.loads(admission.read_bytes())
        root = Path(entry["root"])
        if "task-grasp-assistance-v1" in root.parts:
            raise ValueError("Duplicate pilot admissions are forbidden")
        scene_bytes = (root / "problem.json").read_bytes()
        if hashlib.sha256(scene_bytes).hexdigest() != entry["files_sha256"]["problem.json"]:
            raise ValueError(f"Changed admitted scene: {root}")
        rows.append((entry, json.loads(scene_bytes)))
    selected = select_entries(rows, args.parents_per_size, args.seed)
    for entry in selected:
        root = Path(entry["root"])
        for name, expected in entry["files_sha256"].items():
            path = root / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Changed admitted input: {path}")
            files.add(path)
        tree = ET.fromstring((root / "model.xml").read_bytes())
        compiler = tree.find("compiler")
        for kind, setting in (("mesh", "meshdir"), ("texture", "texturedir")):
            directory = compiler.get(setting, "") if compiler is not None else ""
            for asset in tree.findall(f".//asset/{kind}"):
                if asset.get("file"):
                    files.add((root / directory / asset.get("file")).resolve(strict=True))
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {
        "kind": "paired-original-pose-grasp-rollouts-v1",
        "admissions_sha256": hashlib.sha256(content).hexdigest(),
        "admitted_parent_population": len(rows),
        "selection_seed": args.seed, "parents_per_size": args.parents_per_size,
        "entries": selected,
        "levels": [tick / 20 for tick in range(20)],
        "scope": "Constructible TRAIN parents only; not all attempted parents or held-out accuracy",
        "files_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(files)},
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    with tarfile.open(args.output / "inputs.tar.gz", "w:gz") as archive:
        for path in sorted(files):
            archive.add(path, arcname=str(path).lstrip("/"), recursive=False)
    print(json.dumps({"parents": len(selected), "levels": 20,
                      "initial_rollouts": len(selected) * 20,
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
