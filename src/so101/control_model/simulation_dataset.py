from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from so101.control_model.command import (
    ROLE_PROMPTS,
    TASK_PROMPTS,
    camera_point_from_projection,
    task_phase_prompt,
)


class EntityTrajectoryDataset(Dataset):
    def __init__(
        self,
        collection: Path,
        horizon: int,
        history_steps: int,
        max_entities: int = 4,
        tasks: tuple[str, ...] | None = None,
        normalization: dict[str, list[float]] | None = None,
        phase_prompts: bool = False,
        excluded_sources: tuple[str, ...] = (),
    ):
        if horizon <= 0 or history_steps <= 0:
            raise ValueError("horizon and history_steps must be positive")
        report = json.loads((collection / "collection.json").read_text())
        self.horizon = horizon
        self.history_steps = history_steps
        self.max_entities = max_entities
        self.phase_prompts = phase_prompts
        self.episodes = []
        self.samples = []
        self.episode_sample_indices: list[list[int]] = []
        self.episode_focus_sample_indices: list[list[int]] = []
        self.task_episode_indices: dict[str, list[int]] = {}
        self.obstacle_episode_indices: list[int] = []
        pair_groups: dict[tuple, list[int]] = {}
        task_contrast_groups: dict[tuple, list[int]] = {}
        states, actions = [], []
        excluded = {Path(source).resolve() for source in excluded_sources}
        for source_index, source_name in enumerate(report["successful_episodes"]):
            source = Path(source_name)
            if source.resolve() in excluded:
                continue
            episode = json.loads((source / "episode.json").read_text())
            if tasks is not None and episode["task"] not in tasks:
                continue
            rows = [
                json.loads(line)
                for line in (source / "frames.jsonl").read_text().splitlines()
            ]
            if not rows or any(
                key not in row
                for row in rows
                for key in (
                    "command_boxes",
                    "command_visibility",
                    "command_depth_m",
                    "command_depth_valid",
                    "command_roles",
                )
            ):
                raise ValueError(f"{source}: missing entity trajectory labels")
            entity_count = len(rows[0]["command_roles"])
            if not 0 < entity_count <= max_entities or any(
                len(row[key]) != entity_count
                for row in rows
                for key in (
                    "command_boxes",
                    "command_visibility",
                    "command_depth_m",
                    "command_depth_valid",
                    "command_roles",
                )
            ):
                raise ValueError(f"{source}: inconsistent command entities")
            episode_index = len(self.episodes)
            self.episodes.append((source, episode, rows))
            self.task_episode_indices.setdefault(episode["task"], []).append(
                episode_index
            )
            if episode.get("obstacles"):
                self.obstacle_episode_indices.append(episode_index)
            first_sample = len(self.samples)
            self.samples.extend((episode_index, index) for index in range(len(rows)))
            self.episode_sample_indices.append(
                list(range(first_sample, first_sample + len(rows)))
            )
            focus_phases = {
                "push": {"push_lower", "push", "push_realign"},
                "place": {
                    "lift",
                    "place_carry",
                    "place_lower",
                    "place_release",
                },
            }.get(episode["task"], set())
            if episode.get("obstacles"):
                focus_phases = {
                    "avoid_obstacle_1",
                    "avoid_obstacle_2",
                    "approach",
                    "descend",
                }
            self.episode_focus_sample_indices.append([
                first_sample + index
                for index, row in enumerate(rows)
                if row.get("phase") in focus_phases
            ])
            scene_seed = episode.get(
                "scene_seed", episode.get("initial", {}).get("seed")
            )
            pair_id = (
                ("explicit", str(episode["pair_id"]))
                if "pair_id" in episode
                else ("adjacent", str(source.parent), source_index // 2,
                      episode["task"], scene_seed)
            )
            pair_groups.setdefault(pair_id, []).append(episode_index)
            if scene_seed is not None:
                if episode["task"] in ("grasp", "view"):
                    contrast_key = (
                        "unary",
                        scene_seed,
                        episode.get("target"),
                    )
                else:
                    contrast_key = (
                        "binary",
                        scene_seed,
                        episode.get("target"),
                        episode.get("destination"),
                    )
                task_contrast_groups.setdefault(contrast_key, []).append(
                    episode_index
                )
            states.extend(row["state"] for row in rows)
            actions.extend(row["action"] for row in rows)
        if not self.samples:
            raise ValueError("Collection contains no samples")
        self.pair_groups_by_task: dict[str, list[list[int]]] = {}
        for indices in pair_groups.values():
            if len(indices) < 2:
                continue
            task = self.episodes[indices[0]][1]["task"]
            if any(self.episodes[index][1]["task"] != task for index in indices):
                raise ValueError("Paired command group mixes task phrases")
            self.pair_groups_by_task.setdefault(task, []).append(indices)
        self.pair_groups = [
            group
            for groups in self.pair_groups_by_task.values()
            for group in groups
        ]
        if not self.pair_groups:
            raise ValueError("Entity collection must contain paired command episodes")
        self.task_contrast_groups = [
            indices
            for indices in task_contrast_groups.values()
            if len({self.episodes[index][1]["task"] for index in indices}) >= 2
        ]
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if normalization is None:
            self.state_mean = states.mean(axis=0)
            self.state_std = np.maximum(
                states.std(axis=0), np.ones(6, dtype=np.float32)
            )
            self.action_mean = actions.mean(axis=0)
            self.action_std = np.maximum(
                actions.std(axis=0), np.ones(6, dtype=np.float32)
            )
        else:
            self.state_mean = np.asarray(
                normalization["state_mean"], dtype=np.float32
            )
            self.state_std = np.asarray(
                normalization["state_std"], dtype=np.float32
            )
            self.action_mean = np.asarray(
                normalization["action_mean"], dtype=np.float32
            )
            self.action_std = np.asarray(
                normalization["action_std"], dtype=np.float32
            )
            if any(
                value.shape != (6,)
                for value in (
                    self.state_mean,
                    self.state_std,
                    self.action_mean,
                    self.action_std,
                )
            ) or np.any(self.state_std <= 0) or np.any(self.action_std <= 0):
                raise ValueError("Invalid normalization override")

    def __len__(self) -> int:
        return len(self.samples)

    def _image(self, source: Path, row: dict) -> torch.Tensor:
        bgr = cv2.imread(str(source / row["image"]))
        if bgr is None:
            raise RuntimeError(f"Missing image {source / row['image']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        episode_index, index = self.samples[item]
        source, episode, rows = self.episodes[episode_index]
        history_indices = np.arange(index - self.history_steps + 1, index + 1).clip(0)
        future_indices = np.arange(index, index + self.horizon).clip(max=len(rows) - 1)
        box_indices = np.arange(index + 1, index + self.horizon + 1).clip(max=len(rows) - 1)
        future_pad = np.arange(index, index + self.horizon) >= len(rows)
        history_images = torch.stack([self._image(source, rows[i]) for i in history_indices])
        history_state = np.asarray([rows[i]["state"] for i in history_indices], dtype=np.float32)
        prior_indices = np.maximum(history_indices - 1, 0)
        history_action = np.asarray([rows[i]["action"] for i in prior_indices], dtype=np.float32)
        future_actions = np.asarray([rows[i]["action"] for i in future_indices], dtype=np.float32)
        entity_count = len(rows[index]["command_roles"])
        entity_order = np.random.permutation(entity_count)

        def pad(values: np.ndarray, fill: float = 0) -> np.ndarray:
            shape = (*values.shape[:-2], self.max_entities, values.shape[-1])
            padded = np.full(shape, fill, dtype=np.float32)
            padded[..., :entity_count, :] = values
            return padded

        future_boxes = pad(np.asarray([
            np.column_stack((
                rows[i]["command_boxes"],
                rows[i]["command_visibility"],
            ))[entity_order]
            for i in box_indices
        ], dtype=np.float32))
        future_depth = np.zeros(
            (self.horizon, self.max_entities), dtype=np.float32
        )
        future_depth[:, :entity_count] = np.asarray(
            [
                np.asarray(rows[i]["command_depth_m"])[entity_order]
                for i in box_indices
            ],
            dtype=np.float32,
        )
        future_depth_valid = np.zeros(
            (self.horizon, self.max_entities), dtype=np.float32
        )
        future_depth_valid[:, :entity_count] = np.asarray(
            [
                np.asarray(rows[i]["command_depth_valid"])[entity_order]
                for i in box_indices
            ],
            dtype=np.float32,
        )
        future_center_valid = np.zeros(
            (self.horizon, self.max_entities), dtype=np.float32
        )
        future_center_valid[:, :entity_count] = np.asarray(
            [
                np.asarray(rows[i].get(
                    "command_center_valid",
                    rows[i]["command_visibility"],
                ))[entity_order]
                for i in box_indices
            ],
            dtype=np.float32,
        )
        current_box = np.zeros((self.max_entities, 5), dtype=np.float32)
        current_box[:entity_count] = np.asarray(
            rows[index]["command_boxes"], dtype=np.float32
        )[entity_order]
        current_visibility = np.zeros(self.max_entities, dtype=np.float32)
        current_visibility[:entity_count] = np.asarray(
            rows[index]["command_visibility"], dtype=np.float32
        )[entity_order]
        current_depth = np.zeros(self.max_entities, dtype=np.float32)
        current_depth[:entity_count] = np.asarray(
            rows[index]["command_depth_m"], dtype=np.float32
        )[entity_order]
        current_depth_valid = np.zeros(self.max_entities, dtype=np.float32)
        current_depth_valid[:entity_count] = np.asarray(
            rows[index]["command_depth_valid"], dtype=np.float32
        )[entity_order]
        current_camera_position = np.zeros(
            (self.max_entities, 3), dtype=np.float32
        )
        for entity_index in range(entity_count):
            if current_depth_valid[entity_index] > .5:
                current_camera_position[entity_index] = (
                    camera_point_from_projection(
                        float(current_box[entity_index, 0]),
                        float(current_box[entity_index, 1]),
                        float(current_depth[entity_index]),
                    )
                )
        roles = [
            ROLE_PROMPTS[rows[index]["command_roles"][entity_index]]
            for entity_index in entity_order
        ] + [ROLE_PROMPTS["object"]] * (self.max_entities - entity_count)
        return {
            "images": history_images,
            "history_state": torch.from_numpy(
                (history_state - self.state_mean) / self.state_std
            ),
            "history_action": torch.from_numpy(
                (history_action - self.action_mean) / self.action_std
            ),
            "actions": torch.from_numpy(
                (future_actions - self.action_mean) / self.action_std
            ),
            "action_is_pad": torch.from_numpy(future_pad),
            "entity_box": torch.from_numpy(current_box),
            "entity_visibility": torch.from_numpy(current_visibility),
            "entity_depth_m": torch.from_numpy(current_depth),
            "entity_depth_valid": torch.from_numpy(current_depth_valid),
            "entity_camera_position": torch.from_numpy(
                current_camera_position
            ),
            "entity_mask": torch.arange(self.max_entities) < entity_count,
            "future_box": torch.from_numpy(future_boxes[..., :5]),
            "future_visibility": torch.from_numpy(future_boxes[..., 5]),
            "future_depth_m": torch.from_numpy(future_depth),
            "future_depth_valid": torch.from_numpy(future_depth_valid),
            "future_center_valid": torch.from_numpy(future_center_valid),
            "task": (
                task_phase_prompt(episode["task"], rows[index].get("phase", ""))
                if self.phase_prompts
                else TASK_PROMPTS.get(episode["task"], episode["task"])
            ),
            "roles": roles,
        }

    def normalization(self) -> dict[str, list[float]]:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }

    def _sample_pool(
        self,
        rng: np.random.Generator,
        episode: int,
        focus_probability: float,
    ) -> list[int]:
        focused = self.episode_focus_sample_indices[episode]
        if focused and rng.random() < focus_probability:
            return focused
        return self.episode_sample_indices[episode]

    def paired_indices(
        self,
        rng: np.random.Generator,
        batch_size: int,
        focus_probability: float = 0.,
    ) -> list[int]:
        if batch_size <= 0 or batch_size % 2:
            raise ValueError("Paired batches require a positive even batch size")
        indices = []
        tasks = tuple(self.pair_groups_by_task)
        for _ in range(batch_size // 2):
            task = tasks[int(rng.integers(len(tasks)))]
            groups = self.pair_groups_by_task[task]
            group = groups[int(rng.integers(len(groups)))]
            selected = rng.choice(group, size=2, replace=False)
            first = self._sample_pool(
                rng, int(selected[0]), focus_probability
            )
            second = self._sample_pool(
                rng, int(selected[1]), focus_probability
            )
            frame = int(rng.integers(min(len(first), len(second))))
            indices.extend((first[frame], second[frame]))
        return indices

    def balanced_indices(
        self,
        rng: np.random.Generator,
        batch_size: int,
        focus_probability: float = 0.,
    ) -> list[int]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        tasks = tuple(self.task_episode_indices)
        indices = []
        for _ in range(batch_size):
            task = tasks[int(rng.integers(len(tasks)))]
            episodes = self.task_episode_indices[task]
            episode = episodes[int(rng.integers(len(episodes)))]
            samples = self._sample_pool(rng, episode, focus_probability)
            indices.append(samples[int(rng.integers(len(samples)))])
        return indices

    def obstacle_balanced_indices(
        self,
        rng: np.random.Generator,
        batch_size: int,
        obstacle_probability: float,
        focus_probability: float = 0.,
    ) -> list[int]:
        if (
            batch_size <= 0
            or not 0 <= obstacle_probability <= 1
        ):
            raise ValueError(
                "batch_size must be positive and obstacle_probability "
                "must be within [0, 1]"
            )
        tasks = tuple(self.task_episode_indices)
        indices = []
        for _ in range(batch_size):
            if (
                self.obstacle_episode_indices
                and rng.random() < obstacle_probability
            ):
                episodes = self.obstacle_episode_indices
            else:
                task = tasks[int(rng.integers(len(tasks)))]
                episodes = self.task_episode_indices[task]
            episode = episodes[int(rng.integers(len(episodes)))]
            samples = self._sample_pool(
                rng, episode, focus_probability
            )
            indices.append(samples[int(rng.integers(len(samples)))])
        return indices

    def task_contrast_indices(
        self,
        rng: np.random.Generator,
        batch_size: int,
        focus_probability: float = 0.,
    ) -> list[int]:
        if batch_size <= 0 or batch_size % 2:
            raise ValueError("Task-contrast batches require a positive even batch size")
        if not self.task_contrast_groups:
            raise ValueError("Collection contains no cross-task matched scenes")
        indices = []
        for _ in range(batch_size // 2):
            group = self.task_contrast_groups[
                int(rng.integers(len(self.task_contrast_groups)))
            ]
            first = int(rng.integers(len(group)))
            alternatives = [
                index
                for index in range(len(group))
                if self.episodes[group[index]][1]["task"]
                != self.episodes[group[first]][1]["task"]
            ]
            second = alternatives[int(rng.integers(len(alternatives)))]
            first_samples = self._sample_pool(
                rng, group[first], focus_probability
            )
            second_samples = self._sample_pool(
                rng, group[second], focus_probability
            )
            frame = int(rng.integers(min(len(first_samples), len(second_samples))))
            indices.extend((first_samples[frame], second_samples[frame]))
        return indices
