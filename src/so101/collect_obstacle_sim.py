from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from so101.robot_sim import BeamSimulation, MODEL_COMMIT, MODEL_ROOT


def box_points(
    box: list[float],
    width: int,
    height: int,
) -> np.ndarray:
    center_x, center_y, mean_extent, anisotropy_x, anisotropy_y = box
    diagonal = float(np.hypot(width, height))
    half_difference = float(np.hypot(anisotropy_x, anisotropy_y))
    major = mean_extent + half_difference
    minor = max(mean_extent - half_difference, 0)
    angle = np.rad2deg(
        np.arctan2(anisotropy_y, anisotropy_x) / 2
    )
    return cv2.boxPoints((
        (float(center_x * width), float(center_y * height)),
        (float(major * diagonal), float(minor * diagonal)),
        float(angle),
    ))


def draw_box(
    image: np.ndarray,
    box: list[float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    height, width = image.shape[:2]
    points = box_points(box, width, height).round().astype(np.int32)
    cv2.polylines(image, [points], True, color, 2)
    origin = (
        int(np.clip(points[:, 0].min(), 4, width - 1)),
        int(np.clip(points[:, 1].min() - 5, 16, height - 1)),
    )
    cv2.putText(
        image,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        .45,
        color,
        1,
        cv2.LINE_AA,
    )


def edge_indicator_point(
    center: tuple[float, float],
    width: int,
    height: int,
    margin: int = 8,
) -> tuple[int, int]:
    center_x, center_y = center
    direction_x = center_x - .5
    direction_y = center_y - .5
    if direction_x == 0 and direction_y == 0:
        return width // 2, height // 2
    limits = []
    if direction_x:
        limits.append((.5 - margin / width) / abs(direction_x))
    if direction_y:
        limits.append((.5 - margin / height) / abs(direction_y))
    scale = min(limits)
    return (
        round((.5 + direction_x * scale) * width),
        round((.5 + direction_y * scale) * height),
    )


def render_video(
    sim: BeamSimulation,
    rows: list[dict],
    seed: int,
    target: str,
    path: Path,
) -> None:
    sim.episode_root = None
    sim.reset(
        seed,
        varied_layout=True,
        real_scene=True,
        color_targets=True,
    )
    sim.configure_command(
        "grasp", [(target, "object"), ("obstacle_box", "obstacle")]
    )
    sim.place_grasp_obstacle(target)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        sim.fps,
        (640, 240),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    try:
        for row in rows:
            overview = sim.overview_image()
            wrist = sim.image()
            target_box = sim.target_oriented_box(target)
            if target_box[5]:
                draw_box(wrist, target_box[:5], (0, 255, 0), "target")
            elif not (
                0 <= target_box[0] <= 1
                and 0 <= target_box[1] <= 1
            ):
                point = edge_indicator_point(
                    (target_box[0], target_box[1]),
                    wrist.shape[1],
                    wrist.shape[0],
                )
                cv2.circle(wrist, point, 7, (0, 255, 0), -1)
                cv2.circle(wrist, point, 7, (0, 0, 0), 1)
            frame = cv2.cvtColor(
                np.concatenate((overview, wrist), axis=1),
                cv2.COLOR_RGB2BGR,
            )
            cv2.putText(
                frame,
                f"{row['phase']} | target: {target} | obstacle: red box",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                .45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
            sim.step(
                sim.decode(np.asarray(row["action"], dtype=np.float32)),
                row["phase"],
            )
    finally:
        writer.release()


def collect(
    root: Path,
    seeds: list[int],
    target: str,
    model_root: Path = MODEL_ROOT,
    video_limit: int = 1,
) -> None:
    if not seeds or video_limit < 0:
        raise ValueError(
            "At least one seed is required and video_limit must be nonnegative"
        )
    root.mkdir(parents=True, exist_ok=False)
    sim = BeamSimulation(model_root)
    attempts = []
    successful = []
    started = time.monotonic()
    try:
        for index, seed in enumerate(seeds):
            sim.reset(
                seed,
                varied_layout=True,
                real_scene=True,
                color_targets=True,
            )
            episode = root / f"episode_{index:05}"
            (episode / "wrist").mkdir(parents=True)
            sim.episode_root = episode
            try:
                result = sim.demonstrate_obstacle_grasp(target)
            except (RuntimeError, ValueError) as exception:
                result = {
                    "success": False,
                    "reason": "Obstacle-avoidance expert failed",
                    "error": str(exception),
                    "obstacle_contacts": len(
                        sim.obstacle_contact_events
                    ),
                }
            rows = list(sim.rows)
            report = {
                "simulation": True,
                "model_commit": MODEL_COMMIT,
                "outcome": "success" if result["success"] else "failed",
                "fps": sim.fps,
                "frames": len(rows),
                "task": "grasp",
                "scene_seed": seed,
                "target": target,
                "target_label": sim.target_labels[target],
                "destination": None,
                "obstacles": ["obstacle_box"],
                "initial": json.loads(json.dumps(sim.initial)),
                "verification": result,
                "duration": len(rows) / sim.fps,
                "timing_acceptable": len(rows) > 1 and bool(np.allclose(
                    np.diff([row["elapsed"] for row in rows]),
                    1 / sim.fps,
                    atol=1e-8,
                    rtol=0,
                )),
            }
            (episode / "frames.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (episode / "episode.json").write_text(
                json.dumps(report, indent=2)
            )
            attempts.append(report)
            if result["success"]:
                successful.append(str(episode))
                if len(successful) <= video_limit:
                    render_video(
                        sim,
                        rows,
                        seed,
                        target,
                        episode / "rollout.mp4",
                    )
            print(json.dumps({
                "episode": episode.name,
                "seed": seed,
                "target": target,
                **result,
            }), flush=True)
            (root / "collection.json").write_text(json.dumps({
                "simulation": True,
                "task": "grasp",
                "target": target,
                "obstacle": "obstacle_box",
                "requested_seeds": seeds,
                "successful_episodes": successful,
                "attempts": attempts,
                "wall_seconds": time.monotonic() - started,
            }, indent=2))
    finally:
        sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--target",
        choices=("short", "wide"),
        default="wide",
    )
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--video-limit", type=int, default=1)
    args = parser.parse_args()
    collect(
        args.root,
        args.seeds,
        args.target,
        args.model_root,
        args.video_limit,
    )


if __name__ == "__main__":
    main()
