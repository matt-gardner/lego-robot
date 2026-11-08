import numpy as np
import pytest

from so101.pathtrace_pile import camera_frame, linear_rgb


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
