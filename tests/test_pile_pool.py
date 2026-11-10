import json

import pytest

from so101.pile_pool import publish_pool


def test_current_and_snapshot_match_and_changed_snapshot_is_rejected(tmp_path):
    result = publish_pool(tmp_path)
    snapshot = tmp_path / result["manifest_path"]
    assert json.loads(snapshot.read_text()) == json.loads(
        (tmp_path / "current-pool.json").read_text()
    )
    assert publish_pool(tmp_path) == result
    snapshot.write_text("{}")
    with pytest.raises(ValueError, match="snapshot changed"):
        publish_pool(tmp_path)


def test_non_training_scene_cannot_replace_current(tmp_path):
    current = tmp_path / "current-pool.json"
    current.write_text('{"previous": true}')
    scene = tmp_path / "scene_000"
    scene.mkdir()
    (scene / "admission.json").write_text(json.dumps({
        "split": "development", "scene_id": "heldout",
    }))
    with pytest.raises(ValueError, match="non-TRAIN"):
        publish_pool(tmp_path)
    assert json.loads(current.read_text()) == {"previous": True}
