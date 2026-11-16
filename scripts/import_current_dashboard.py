"""Import one checked live inventory into the current dashboard database."""
import argparse
import hashlib
import json
from pathlib import Path

from so101.bucket_dashboard import validate_annotation
from so101.bucket_store import MAX_INVENTORY_BYTES, import_current


def migrate(root: Path, source: Path, expected_sha256: str, pieces: Path) -> dict:
    with source.open("rb") as stream:
        body = stream.read(MAX_INVENTORY_BYTES + 1)
    if len(body) > MAX_INVENTORY_BYTES or hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError("Live inventory size or pinned digest differs")
    document = json.loads(body)
    by_scene = {}
    for path in sorted(pieces.glob("*.json")):
        scene = json.loads(path.read_bytes())
        identity = scene["scene_id"]
        if identity in by_scene:
            raise ValueError("Duplicate piece-coordinate scene")
        by_scene[identity] = scene
    migrated = 0
    for pile in document.get("admitted_piles", []):
        if pile.get("image_kind") != "admitted-pile-camera":
            continue
        scene = by_scene[pile["scene_id"]]
        image = pile["initial_images"][0]
        if scene["pile_image_sha256"] != image["sha256"]:
            raise ValueError("Piece coordinates do not match the retained pile image")
        receipt_path = (root / pile["snapshot_directory"] / image["receipt_path"]).resolve()
        if not receipt_path.is_relative_to(root.resolve()):
            raise ValueError("Pile receipt escapes the dashboard root")
        receipt = json.loads(receipt_path.read_bytes())["pile"]
        if (scene["piece_camera"] != receipt["camera_provenance"]
                or scene["scene_files_sha256"] != receipt["scene_files_sha256"]
                or scene["piece_image_width"] != image["width"]
                or scene["piece_image_height"] != image["height"]):
            raise ValueError("Piece coordinate camera or source binding differs")
        pieces_for_scene = scene["pieces"]
        if (len(pieces_for_scene) != pile["tags"]["piece_count"]
                or len({piece["entity_id"] for piece in pieces_for_scene}) != len(pieces_for_scene)):
            raise ValueError("Piece coordinate population differs from admitted pile")
        for piece in pieces_for_scene:
            validate_annotation(piece, image["width"], image["height"])
            if (not isinstance(piece["position_m"], list) or len(piece["position_m"]) != 3):
                raise ValueError("Piece position must contain three world coordinates")
        pile["pieces"] = pieces_for_scene
        pile["piece_camera"] = scene["piece_camera"]
        pile["piece_image_width"] = scene["piece_image_width"]
        pile["piece_image_height"] = scene["piece_image_height"]
        pile["piece_coordinate_provenance"] = {
            key: value for key, value in scene.items()
            if key not in ("pieces", "piece_camera", "piece_image_width", "piece_image_height")
        }
        migrated += 1
    if migrated != len(by_scene):
        raise ValueError("Piece coordinates cross the current pile population")
    path = import_current(root, document)
    return {"store": str(path), "piles_with_pieces": migrated,
            "task_records": len(document["problems"]["problems"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--pieces", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.root, args.inventory, args.sha256, args.pieces)))


if __name__ == "__main__":
    main()
