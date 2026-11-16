import json
import os

import mujoco
import numpy as np
import pytest

from so101.cloud_checkpoint import atomic_json
from so101.lego_scene import capture_scene_state
from so101.pathtrace_pile import DEFAULT_PROFILE, file_hash
from so101.pile_directory import current_image
from so101.render_canonical_piles import render_one, source_hashes


def test_changed_frozen_source_is_rejected_before_render(tmp_path):
    with pytest.raises(ValueError, match="Frozen source changed"):
        directory = tmp_path / "pile_0001"
        directory.mkdir()
        for name in ("model.mjb", "model.xml", "integration_state.json",
                     "problem.json", "scene_plan.json"):
            (directory / name).write_bytes(b"original")
        expected = source_hashes(directory)
        (directory / "integration_state.json").write_bytes(b"changed")
        render_one(directory, DEFAULT_PROFILE, tmp_path, expected)


@pytest.mark.skipif(not os.environ.get("LEGO_PATHTRACE_PYTHON"), reason="Optional CPU backend absent")
def test_canonical_render_preserves_sources_and_reuses_verified_image(tmp_path):
    directory = tmp_path / "piles" / "pile_0001"
    directory.mkdir(parents=True)
    xml = """<mujoco><worldbody>
      <geom name="table" type="plane" pos=".08 .02 0" euler="0 0 15" size=".12 .1 .01"/>
      <body name="piece" pos=".02 .01 .03"><freejoint/>
      <geom type="box" size=".03 .02 .03" rgba=".2 .5 .8 1"/></body>
      </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    (directory / "model.xml").write_text(xml)
    mujoco.mj_saveModel(model, str(directory / "model.mjb"))
    snapshot = capture_scene_state(model, data, xml, {
        "scene_id": "test", "catalog_sha256": "test",
    }, status="settled")
    atomic_json(directory / "integration_state.json", snapshot)
    atomic_json(directory / "problem.json", {
        "instances": [{"instance_id": "piece", "body_name": "piece", "part_id": "test"}],
    })
    atomic_json(directory / "scene_plan.json", {})
    atomic_json(directory / "pile.json", {"rendered": None, "preserved": "metadata"})
    expected = source_hashes(directory)
    config = json.loads(DEFAULT_PROFILE.read_text())
    config.update(width=64, height=48, samples=4, threads=1)
    profile = tmp_path / "profile.json"
    atomic_json(profile, config)
    assert render_one(directory, profile, tmp_path, expected) == "rendered"
    assert source_hashes(directory) == expected
    pile = json.loads((directory / "pile.json").read_text())
    assert pile["preserved"] == "metadata"
    assert pile["rendered"]["renderer"] == "mitsuba-llvm_ad_rgb"
    coords = pile["piece_coordinates"]
    assert coords["pile_image_sha256"] == file_hash(directory / "rendered.png")
    box = coords["pieces"][0]["target_bbox_xyxy"]
    assert 0 <= box[0] < box[2] <= 64
    assert 0 <= box[1] < box[3] <= 48
    body, mime = current_image(tmp_path, "pile_0001", 0)
    assert mime == "image/png" and body.startswith(b"\x89PNG")
    assert render_one(directory, profile, tmp_path, expected) == "reused"
    np.testing.assert_array_equal(snapshot["qpos"], data.qpos)
    (directory / "rendered.png").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Published image/metadata differs"):
        render_one(directory, profile, tmp_path, expected)
