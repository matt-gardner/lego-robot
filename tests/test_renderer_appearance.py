import copy
import os

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from so101.renderer_appearance import RendererAppearance
from so101.robot_sim import BeamSimulation, MODEL_ROOT, restore_episode


@pytest.fixture
def pair():
    if not (MODEL_ROOT / "revision.json").exists():
        pytest.skip("Pinned SO101 assets unavailable")
    baseline = BeamSimulation(render=False)
    styled = BeamSimulation(render=False, appearance=RendererAppearance())
    try:
        yield baseline, styled
    finally:
        baseline.close()
        styled.close()


def test_profile_preserves_contacts_joints_and_seeded_dynamics(pair):
    baseline, styled = pair
    for field in ("body_mass", "body_inertia", "geom_contype", "geom_conaffinity",
                  "geom_friction", "geom_solref", "geom_solimp", "jnt_axis",
                  "jnt_range", "actuator_gainprm", "actuator_forcerange"):
        np.testing.assert_array_equal(getattr(baseline.model, field), getattr(styled.model, field))
    baseline.reset(10)
    styled.reset(10)
    for _ in range(100):
        mujoco.mj_step(baseline.model, baseline.data)
        mujoco.mj_step(styled.model, styled.data)
    np.testing.assert_array_equal(baseline.data.qpos, styled.data.qpos)
    np.testing.assert_array_equal(baseline.data.qvel, styled.data.qvel)
    assert baseline.data.ncon == styled.data.ncon
    assert "renderer" not in baseline.initial
    assert styled.initial["renderer"] == styled.appearance.identity()


def test_recorded_profile_cannot_be_silently_replaced(pair):
    baseline, styled = pair
    styled.reset(10)
    metadata = {"initial": copy.deepcopy(styled.initial), "target": "short",
                "task": "grasp"}
    with pytest.raises(ValueError, match="renderer identity"):
        restore_episode(baseline, metadata)
    restore_episode(styled, metadata)
    metadata["initial"].pop("renderer")
    with pytest.raises(ValueError, match="renderer identity"):
        restore_episode(styled, metadata)


def test_photometric_processing_is_reproducible_without_spatial_warp():
    profile = RendererAppearance()
    image = np.full((240, 320, 3), 128, dtype=np.uint8)
    first = profile.process_rgb(image, seed=10, frame=1)
    assert first.shape == image.shape and first.dtype == image.dtype
    np.testing.assert_array_equal(first, profile.process_rgb(image, seed=10, frame=1))
    assert not np.array_equal(first, profile.process_rgb(image, seed=10, frame=2))


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") not in ("osmesa", "egl"),
                    reason="Requires explicit headless rendering backend")
def test_rgb_area_lighting_preserves_segmentation_depth_and_model():
    sim = BeamSimulation(appearance=RendererAppearance())
    try:
        sim.reset(10)
        positions = sim.model.light_pos.copy()
        segmentation = sim._segmentation_image()
        depth = sim.entity_camera_depth("studded")
        image = sim.image()
        np.testing.assert_array_equal(image, sim.image())
        np.testing.assert_array_equal(positions, sim.model.light_pos)
        np.testing.assert_array_equal(segmentation, sim._segmentation_image())
        assert depth == sim.entity_camera_depth("studded")
        assert sim.entity_mask("studded").sum() > 0
    finally:
        sim.close()
