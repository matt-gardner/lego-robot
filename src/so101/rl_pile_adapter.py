"""Use native pile resets and goals with the existing robot/policy action contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import mujoco

from so101.pile_task_runtime import PileSimulation, load_problem, reset_problem, task_success
from so101.rl_renderer import PathtraceSimulation
from so101.robot_sim import BeamSimulation
from so101.simulation_corpus import CorpusEpisode, TASKS


def load_reference_rows(episode: CorpusEpisode) -> list[dict]:
    reference = episode.metadata.get("native_reference")
    if reference is None:
        raise ValueError("Reference-free native tasks do not support gold-prefix assistance")
    content = Path(reference["frames"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != reference["frames_sha256"]:
        raise ValueError("Native reference actions changed after qualification")
    rows = json.loads(content)
    if len(rows) < 2:
        raise ValueError("Native reference needs at least two actions")
    for row in rows:
        action = np.asarray(row["action"], dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Native reference action is not six finite policy coordinates")
    return rows


def attach_native_references(
    episodes: dict[str, list[CorpusEpisode]], path: Path, *, pool_sha256: str,
) -> None:
    manifest = json.loads(path.read_text())
    if (manifest.get("schema_version") != 1
            or manifest.get("kind") != "pile-expert-references"
            or manifest.get("success_condition") != "terminal-native-dwell-v1"
            or manifest.get("pool_sha256") != pool_sha256
            or manifest.get("complete") is not True):
        raise ValueError("Native reference manifest is not a completed qualification")
    by_source = {str(episode.source): episode for rows in episodes.values() for episode in rows}
    seen = set()
    for reference in manifest["references"]:
        source = reference["source"]
        if source in seen or source not in by_source:
            raise ValueError("Reference duplicates a problem or crosses the selected native pool")
        seen.add(source)
        episode = by_source[source]
        error = reference["maximum_qpos_error"]
        if (reference["accepted"] is not True or reference["replay_success"] is not True
                or not np.isfinite(error) or not 0 <= error <= 1e-7
                or reference["task"] != episode.task
                or reference["scene_files_sha256"] != episode.metadata["native_entry"]["files_sha256"]):
            raise ValueError("Native reference lacks successful, bound exact-reset replay evidence")
        episode.metadata["native_reference"] = reference
        load_reference_rows(episode)
        episode.metadata["reference_available"] = True


def load_native_pool(path: Path, excluded_families: set[str]) -> dict[str, list[CorpusEpisode]]:
    manifest = json.loads(path.read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "lego-pile-task-pool"
            or manifest.get("status") != "complete" or manifest.get("expert_references") is not False):
        raise ValueError("Unsupported native problem pool")
    episodes = {task: [] for task in TASKS}
    seen = set()
    for entry in manifest["scenes"]:
        if (entry["split"] != "train" or entry["scene_family_id"] in excluded_families
                or entry["scene_id"] in seen):
            raise ValueError("Native scene crosses held-out families, duplicates, or is not TRAIN")
        seen.add(entry["scene_id"])
        scene = json.loads((Path(entry["root"]) / "problem.json").read_text())
        if scene["capabilities"] != {
            "reset": "mujoco-integration-v1", "target_labels": "instance-segmentation-v1",
            "expert_references": False, "sft_qualified": False,
        }:
            raise ValueError("Native task reset/label capabilities are unsupported")
        if any(item["body_name"] != item["instance_id"] for item in scene["instances"]):
            raise ValueError("Native robot adapter requires body-bound persistent instance names")
        task_ids = set()
        for task in scene["tasks"]:
            problem = load_problem(entry, task["task_id"])
            if task["task_id"] in task_ids:
                raise ValueError("Duplicate native task identity")
            task_ids.add(task["task_id"])
            episodes[task["task_type"]].append(CorpusEpisode(
                problem["root"] / "tasks" / task["task_id"],
                {"task": task["task_type"], "target": task["target_instance_id"],
                 "native_entry": entry, "native_task_id": task["task_id"],
                 "scene_family_id": entry["scene_family_id"],
                 **({"parent_task_id": task["parent_task_id"],
                     "task_assistance": task["assistance"]}
                    if "parent_task_id" in task else {}),
                 "reference_available": False},
            ))
    if not all(episodes.values()):
        raise ValueError("Native pool must include trainable problems for every supported task")
    return episodes


class NativePileSimulation(PathtraceSimulation):
    def __init__(self, episode, model_root, backend="mujoco", samples=16, threads=4):
        self.problem = load_problem(
            episode.metadata["native_entry"], episode.metadata["native_task_id"],
        )
        model = mujoco.MjModel.from_binary_path(str(self.problem["root"] / "model.mjb"))
        for instance in self.problem["scene"]["instances"]:
            body = model.body(instance["body_name"]).id
            actual = {model.geom(int(index)).name for index in np.flatnonzero(model.geom_bodyid == body)}
            if actual != set(instance["geom_names"]):
                raise ValueError("Native instance registry differs from its compiled geometry")
        force_ranges = model.actuator_forcerange.copy()
        names = tuple(item["instance_id"] for item in self.problem["scene"]["instances"])
        if backend == "mujoco":
            BeamSimulation.__init__(
                self, model_root, model=model, object_names=names, distractor_names=(),
            )
        else:
            super().__init__(
                model_root, model=model, object_names=names, distractor_names=(),
                visual_xml=(self.problem["root"] / "model.xml").read_text(),
                backend=backend, samples=samples, threads=threads,
            )
        self.native_backend = backend
        # Compiled artifacts already contain the collector's actuator calibration.
        self.model.actuator_forcerange[:] = force_ranges
        self.native = PileSimulation(
            self.model, self.data, self.renderer,
            {name: self.model.body(name).id for name in names},
        )
        self.object_names = names
        self.hide_unseen_centers = True
        try:
            self.reset_native(episode)
        except Exception:
            self.close()
            raise

    def reset_native(self, episode):
        problem = load_problem(episode.metadata["native_entry"], episode.metadata["native_task_id"])
        if problem["root"] != self.problem["root"]:
            raise ValueError("A different compiled scene requires a different simulation")
        self.problem = problem
        reset_problem(self.native, problem)
        task = problem["task"]
        target = task["target_instance_id"]
        destination = task["goal"].get("destination_entity")
        if task["task_type"] in ("push", "place"):
            if destination not in self.destination_geoms or not np.allclose(
                self.position(destination)[:2], task["goal"]["destination_xy"], atol=1e-8, rtol=0.,
            ):
                raise ValueError("Native task's visual destination differs from its graded goal")
        self.initial = {
            "seed": problem["scene"]["seed"], "scene_version": 4,
            "scene_id": problem["scene"]["scene_id"],
            "scene_family_id": problem["scene"]["scene_family_id"],
        }
        self.active_target = target
        entities = [(target, "object")]
        if destination is not None:
            entities.append((destination, "destination"))
        self.command_phrase = task["task_type"]
        self.command_entities = entities
        return target, destination, entities

    def native_success(self, executed, stable):
        result = task_success(self.native, self.problem, executed, stable)
        return result["success"], result["stable"]

    def position(self, name):
        if hasattr(self, "native") and name in self.native.registry:
            return self.native.position(name)
        return super().position(name)

    def _activate_context(self):
        if self.renderer is None or self.renderer._gl_context is None:
            raise RuntimeError("Native policy observations require a live GL renderer")
        self.renderer._gl_context.make_current()
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.renderer._mjr_context)

    def _render_wrist_rgb(self):
        self._activate_context()
        if self.native_backend == "mujoco":
            return BeamSimulation._render_wrist_rgb(self)
        return super()._render_wrist_rgb()

    def _segmentation_image(self):
        self._activate_context()
        return super()._segmentation_image()

    def entity_mask(self, entity):
        geoms = self.beam_geoms.get(entity)
        if geoms is None:
            geoms = self.destination_geoms.get(entity)
        if geoms is None:
            raise ValueError(f"Unknown native policy entity: {entity}")
        segmentation = self._segmentation_image()
        return ((segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
                & np.isin(segmentation[:, :, 0], geoms)).astype(np.uint8)
