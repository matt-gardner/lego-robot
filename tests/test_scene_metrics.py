import numpy as np
import pytest

from so101.scene_metrics import configuration_metrics


def test_rotated_envelopes_describe_actual_separation_not_proposed_category():
    rotation = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    result = configuration_metrics([
        {"instance_id": "a", "position_m": [0, 0, .01],
         "rotation": rotation, "dimensions_m": [.04, .01, .02]},
        {"instance_id": "b", "position_m": [.03, 0, .01],
         "rotation": np.eye(3), "dimensions_m": [.01, .01, .02]},
    ])
    assert result["objects"][0]["envelope_upper_m"] == pytest.approx([.005, .02, .02])
    assert result["pairs"][0]["horizontal_envelope_overlap_m"] == pytest.approx([-.02, .01])
    assert result["pairs"][0]["center_distance_m"] == pytest.approx(.03)


def test_nonfinite_pose_is_not_a_valid_measurement():
    with pytest.raises(ValueError, match="Invalid"):
        configuration_metrics([{"instance_id": "a", "position_m": [0, 0, float("nan")],
                                "rotation": np.eye(3), "dimensions_m": [.01] * 3}])
