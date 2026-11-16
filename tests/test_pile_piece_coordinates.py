from types import SimpleNamespace
from unittest.mock import patch
import unittest

import mujoco
import numpy as np

from so101.pile_piece_coordinates import piece_coordinates


class PieceCoordinatesTest(unittest.TestCase):
    def test_exact_pile_segmentation_and_hidden_piece(self):
        labels = np.full((10, 12, 2), -1, dtype=int)
        labels[2:5, 3:7, 0] = 0
        labels[2:5, 3:7, 1] = mujoco.mjtObj.mjOBJ_GEOM
        renderer = SimpleNamespace(
            enable_segmentation_rendering=lambda: None,
            disable_segmentation_rendering=lambda: None,
            render=lambda: labels, scene=None)
        model = SimpleNamespace(
            body=lambda name: SimpleNamespace(id={"a": 0, "b": 1}[name]),
            geom_bodyid=np.array([0, 1]))
        sim = SimpleNamespace(model=model, data=SimpleNamespace(xpos=np.array([[1., 2., 3.], [4., 5., 6.]])))
        with patch("so101.pile_piece_coordinates.mono_camera", return_value={"camera": "pile"}):
            result = piece_coordinates(sim, renderer, [
                {"instance_id": "first", "body_name": "a"},
                {"instance_id": "hidden", "body_name": "b"}])
        first, hidden = result["pieces"]
        self.assertEqual(first["target_bbox_xyxy"], [3, 2, 7, 5])
        self.assertEqual(first["target_visible_pixels"], 12)
        self.assertEqual(first["position_m"], [1., 2., 3.])
        self.assertEqual(hidden["target_bbox_status"], "not_visible")
        self.assertIsNone(hidden["target_bbox_xyxy"])
        self.assertEqual(result["piece_image_width"], 12)
        self.assertEqual(result["piece_image_height"], 10)


if __name__ == "__main__":
    unittest.main()
