import hashlib
import json

import pytest

from so101.publish_task_images import validate_live_images


def test_retired_callback_validates_live_assets_without_writing_publications(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"already-rendered-asset")
    receipt = tmp_path / "task.json"
    receipt.write_text(json.dumps({
        "image_file": image.name, "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "physics_unchanged": True,
    }))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert validate_live_images(tmp_path) == {
        "mode": "live-file-validation", "validated_receipts": 1, "published": False,
    }
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        validate_live_images(tmp_path)
