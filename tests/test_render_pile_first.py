import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "render_pile_first", Path(__file__).parents[1] / "scripts/render_pile_first.py",
)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def test_uncovered_piles_and_new_admissions_preempt_task_expansion(tmp_path):
    entries = []
    for index in range(3):
        root = tmp_path / str(index)
        root.mkdir()
        problem = root / "problem.json"
        problem.write_text(json.dumps({"tasks": [{"task_id": "first"}, {"task_id": "second"}]}))
        entry = {"root": str(root), "files_sha256": {"problem.json": queue.digest(problem)}}
        entries.append((entry, root, str(index), "selected" if index < 2 else "admitted-pile"))
    assert queue.next_job(entries)["entry"] == entries[0][0]
    for entry, output, sha, _ in entries:
        image = output / "image.png"
        image.write_bytes(b"image")
        queue.receipt_path(output, entry, {"task_id": "first"}).write_text(json.dumps({
            "pool_sha256": sha, "image_file": image.name, "sha256": queue.digest(image),
        }))
        job = queue.next_job(entries)
        if sha == "0":
            assert job["entry"] == entries[1][0]
        elif sha == "1":
            assert job["entry"] == entries[2][0]
            assert job["phase"] == "piles"
    job = queue.next_job(entries)
    assert job["phase"] == "tasks"
    assert job["entry"] == entries[0][0]
    assert job["task"]["task_id"] == "second"


def test_discovery_includes_only_atomic_admissions_from_both_generators(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "current-task-pools.json").write_text(json.dumps({"pools": []}))
    output = tmp_path / "output" / "initial-photo-external-v2"
    output.mkdir(parents=True)
    for index, name in enumerate(("photo-task-progression-collective-v1", "catalog-diverse-20-v1")):
        scene = data / name / "pieces-20/tasks/scene_0"
        scene.mkdir(parents=True)
        (scene / "admission.json").write_text(json.dumps({
            "scene_id": str(index), "split": "train", "root": str(scene),
        }))
        (scene / "admission.json.tmp").write_text("unfinished")
    entries = queue.discover(data, output)
    assert {entry[0]["scene_id"] for entry in entries} == {"0", "1"}
    assert all(entry[3] == "admitted-pile" for entry in entries)
