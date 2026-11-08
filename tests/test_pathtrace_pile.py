import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from so101.pathtrace_pile import DEFAULT_PROFILE, camera_frame, linear_rgb


def test_camera_rejects_degenerate_frame():
    with pytest.raises(ValueError, match="nondegenerate"):
        camera_frame({"origin": [0, 0, 0], "target": [0, 0, 1], "up": [0, 0, 2]})
    with pytest.raises(ValueError, match="finite"):
        camera_frame({"origin": [0, 0, float("nan")], "target": [0, 0, 1], "up": [0, 1, 0]})


def test_camera_preserves_mono_export_without_eye_offset():
    camera = {"origin": [.1, -.3, .4], "target": [0, 0, .03], "up": [0, 0, 1]}
    origin, target, up = camera_frame(camera)
    np.testing.assert_array_equal(origin, camera["origin"])
    np.testing.assert_array_equal(target, camera["target"])
    np.testing.assert_array_equal(up, camera["up"])


def test_linear_material_color_preserves_dark_branch_and_rejects_bad_input():
    np.testing.assert_allclose(linear_rgb([0, .02, .5]), [0, .02 / 12.92, .21404114048])
    with pytest.raises(ValueError, match="sRGB"):
        linear_rgb([1.01, 0, 0])


@pytest.mark.skipif(not os.environ.get("LEGO_PATHTRACE_PYTHON"), reason="Optional CPU backend absent")
def test_actual_cpu_render_receipt_hashes_completed_bitmap(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.xml").write_text("<mujoco/>")
    np.savez(source / "settled-state.npz", qpos=[], qvel=[])
    mesh = tmp_path / "triangle.ply"
    mesh.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
        "property float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n"
        "-.1 -.1 .01\n.1 -.1 .01\n0 .1 .01\n3 0 1 2\n"
    )
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps({
        "source": str(source), "table_z_m": 0,
        "objects": [{"mesh": str(mesh), "rgb": [.2, .4, .6]}],
        "camera": {"origin": [0, -.3, .3], "target": [0, 0, 0], "up": [0, 0, 1], "fovy": 45},
    }))
    config = json.loads(DEFAULT_PROFILE.read_text())
    config.update(width=64, height=48, samples=8)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(config))
    output = tmp_path / "render"
    subprocess.run([
        os.environ["LEGO_PATHTRACE_PYTHON"], "-m", "so101.pathtrace_pile",
        "--scene", str(scene), "--output", str(output), "--profile", str(profile),
    ], check=True, capture_output=True, text=True)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["image_sha256"] == hashlib.sha256((output / "pile.png").read_bytes()).hexdigest()
    for filename, digest in receipt["input_sha256"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest
