from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal


BasePhrase = Literal["grasp", "push", "view", "place"]
EntityRole = Literal["object", "destination", "reference", "obstacle"]
SUPPORTED_PHRASES = frozenset(("grasp", "push", "view", "place"))
SUPPORTED_ROLES = frozenset(("object", "destination", "reference", "obstacle"))
TASK_PROMPTS = {
    "grasp": "grasp",
    "push": "push",
    "view": "view",
    "place": "place",
}
ROLE_PROMPTS = {
    "object": "object",
    "destination": "destination",
    "reference": "reference",
    "obstacle": "obstacle",
}
WRIST_CAMERA_FOV_Y_DEGREES = 61.
WRIST_IMAGE_ASPECT = 320 / 240


def camera_point_from_projection(
    center_x: float, center_y: float, depth_m: float
) -> tuple[float, float, float]:
    if not all(math.isfinite(value) for value in (center_x, center_y, depth_m)):
        raise ValueError("Projected center and depth must be finite")
    if depth_m <= 0:
        raise ValueError("Depth must be positive")
    half_height = depth_m * math.tan(
        math.radians(WRIST_CAMERA_FOV_Y_DEGREES) / 2
    )
    half_width = half_height * WRIST_IMAGE_ASPECT
    return (
        (center_x - .5) * 2 * half_width,
        (.5 - center_y) * 2 * half_height,
        -depth_m,
    )


def task_phase_prompt(task: str, phase: str) -> str:
    if task == "grasp":
        coarse_phase = "lift" if phase in {"lift", "verify"} else "acquire"
    elif task == "view":
        coarse_phase = "view"
    elif task == "push":
        coarse_phase = (
            "push"
            if phase == "push"
            else "lower"
            if phase == "push_lower"
            else "finish"
            if phase in {"push_retreat", "push_verify"}
            else "approach"
        )
    elif task == "place":
        coarse_phase = {
            "lift": "lift",
            "place_carry": "carry",
            "place_lower": "lower",
            "place_release": "release",
            "place_retreat": "release",
            "place_verify": "release",
        }.get(phase, "acquire")
    else:
        raise ValueError(f"Unsupported task: {task}")
    return f"{TASK_PROMPTS[task]} {coarse_phase}"


@dataclass(frozen=True)
class CommandEntity:
    entity_id: str
    role: EntityRole
    box_cxcywhr: tuple[float, float, float, float, float]
    camera_depth_m: float | None = None

    def __post_init__(self) -> None:
        if not self.entity_id.strip():
            raise ValueError("entity_id must not be empty")
        if self.role not in SUPPORTED_ROLES:
            raise ValueError(f"Unsupported entity role: {self.role}")
        if len(self.box_cxcywhr) != 5 or not all(
            math.isfinite(value) for value in self.box_cxcywhr
        ):
            raise ValueError("box_cxcywhr must contain five finite values")
        center_x, center_y, major, minor, rotation = self.box_cxcywhr
        if not (
            -4 <= center_x <= 5
            and -4 <= center_y <= 5
            and 0 < minor <= major <= 1
            and -math.pi / 2 <= rotation < math.pi / 2
        ):
            raise ValueError(
                "box_cxcywhr must contain a projected center in [-4, 5], "
                "canonical extents, and rotation in [-pi/2, pi/2)"
            )
        if self.camera_depth_m is not None and (
            not math.isfinite(self.camera_depth_m) or self.camera_depth_m <= 0
        ):
            raise ValueError("camera_depth_m must be finite and positive when provided")

    @property
    def box_features(self) -> tuple[float, float, float, float, float]:
        center_x, center_y, major, minor, rotation = self.box_cxcywhr
        mean_extent = (major + minor) / 2
        half_difference = (major - minor) / 2
        return (
            center_x,
            center_y,
            mean_extent,
            half_difference * math.cos(2 * rotation),
            half_difference * math.sin(2 * rotation),
        )


@dataclass(frozen=True)
class BaseCommand:
    command_id: str
    scene_id: str
    source_camera: str
    source_timestamp: float
    phrase: BasePhrase
    entities: tuple[CommandEntity, ...]

    def __post_init__(self) -> None:
        for name in ("command_id", "scene_id", "source_camera"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if not math.isfinite(self.source_timestamp) or self.source_timestamp < 0:
            raise ValueError("source_timestamp must be finite and non-negative")
        if self.phrase not in SUPPORTED_PHRASES:
            raise ValueError(f"Unsupported base phrase: {self.phrase}")
        if not self.entities:
            raise ValueError("Command must contain at least one entity")
        ids = [entity.entity_id for entity in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("Entity IDs must be unique within a command")
        role_counts = {
            role: sum(entity.role == role for entity in self.entities)
            for role in SUPPORTED_ROLES
        }
        if self.phrase == "grasp" and (
            len(self.entities) != 1 or role_counts["object"] != 1
        ):
            raise ValueError("grasp requires exactly one object entity")
        if self.phrase == "push" and (
            len(self.entities) != 2
            or role_counts["object"] != 1
            or role_counts["destination"] != 1
        ):
            raise ValueError("push requires one object and one destination")
        if self.phrase == "view" and not all(
            entity.role in ("object", "reference") for entity in self.entities
        ):
            raise ValueError("view accepts only object or reference entities")
        if self.phrase == "place" and role_counts["destination"] != 1:
            raise ValueError("place requires exactly one destination")

    def require_scene(self, scene_id: str, source_camera: str | None = None) -> None:
        if scene_id != self.scene_id:
            raise ValueError(
                f"Command references scene {self.scene_id!r}, not current scene {scene_id!r}"
            )
        if source_camera is not None and source_camera != self.source_camera:
            raise ValueError(
                f"Command references camera {self.source_camera!r}, not {source_camera!r}"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> BaseCommand:
        required = {
            "command_id",
            "scene_id",
            "source_camera",
            "source_timestamp",
            "phrase",
            "entities",
        }
        if set(value) != required:
            missing = sorted(required - set(value))
            extra = sorted(set(value) - required)
            raise ValueError(f"Invalid command fields; missing={missing}, extra={extra}")
        raw_entities = value["entities"]
        if not isinstance(raw_entities, (list, tuple)):
            raise ValueError("entities must be a list or tuple")
        entities = []
        for raw in raw_entities:
            if not isinstance(raw, dict):
                raise ValueError("Each entity must be an object")
            entities.append(CommandEntity(
                entity_id=str(raw["entity_id"]),
                role=str(raw["role"]),  # type: ignore[arg-type]
                box_cxcywhr=tuple(
                    float(item) for item in raw["box_cxcywhr"]  # type: ignore[arg-type]
                ),
                camera_depth_m=(
                    None
                    if raw.get("camera_depth_m") is None
                    else float(raw["camera_depth_m"])
                ),
            ))
        return cls(
            command_id=str(value["command_id"]),
            scene_id=str(value["scene_id"]),
            source_camera=str(value["source_camera"]),
            source_timestamp=float(value["source_timestamp"]),
            phrase=str(value["phrase"]),  # type: ignore[arg-type]
            entities=tuple(entities),
        )
