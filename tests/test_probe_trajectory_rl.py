import json
from types import SimpleNamespace

import pytest

from so101.probe_trajectory_rl import probe, success_interval


@pytest.mark.parametrize("plan,calibrate,message", [
    ([], False, "contain problems"),
    ([{"task": "view", "source": "a", "assistance": 0.}] * 2, False, "duplicates"),
    ([{"task": "view", "source": "a", "assistance": .5}], True, "without assistance"),
])
def test_invalid_measurement_population_fails_before_loading_policy(tmp_path, plan, calibrate, message):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    args = SimpleNamespace(
        plan=path, native_pool=None, native_references=None,
        calibrate_assistance=calibrate, native_only=True, initial_only=True,
    )
    with pytest.raises(ValueError, match=message):
        probe(args)


def test_probe_intervals_preserve_uncertainty_at_extremes():
    low, high = success_interval(0, 16)
    assert low == pytest.approx(0)
    assert .19 < high < .20
    low, high = success_interval(16, 16)
    assert .80 < low < .81
    assert high == pytest.approx(1)
    low, high = success_interval(8, 16)
    assert low < .5 < high
    wider = high - low
    low, high = success_interval(16, 32)
    assert high - low < wider
