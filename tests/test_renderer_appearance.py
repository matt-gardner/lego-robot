import copy
import os

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from so101.renderer_appearance import RendererAppearance
from so101.renderer_appearance_v2 import WristAppearanceV2
from so101.robot_sim import BeamSimulation, MODEL_ROOT, restore_episode


@pytest.fixture(params=[RendererAppearance, WristAppearanceV2])
def pair(request):
    if not (MODEL_ROOT / "revision.json").exists():
        pytest.skip("Pinned SO101 assets unavailable")
    baseline = BeamSimulation(render=False)
    styled = BeamSimulation(render=False, appearance=request.param())
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


def test_v2_desaturates_without_changing_gray_or_parent_identity():
    baseline = RendererAppearance()
    profile = WristAppearanceV2()
    assert profile.identity()["parent_implementation_sha256"] == baseline.implementation_sha256
    image = np.full((240, 320, 3), [220, 60, 20], dtype=np.uint8)
    processed = profile.process_rgb(image, seed=10, frame=1).mean(axis=(0, 1))
    original = baseline.process_rgb(image, seed=10, frame=1).mean(axis=(0, 1))
    assert np.ptp(processed) < np.ptp(original) * .8
    gray = np.full((240, 320, 3), 128, dtype=np.uint8)
    assert abs(profile.process_rgb(gray, seed=10, frame=1).mean()
               - baseline.process_rgb(gray, seed=10, frame=1).mean()) < .1


def test_visual_cable_is_explicit_noncontact_world_context():
    profile = WristAppearanceV2()
    source = "<mujoco><worldbody/></mujoco>"
    points = np.array([[0, 0, .0015], [.03, .01, .0015], [.06, .01, .0015]])
    model = mujoco.MjModel.from_xml_string(profile.add_visual_cable(source, points))
    assert model.nq == 0
    assert not model.geom_contype.any() and not model.geom_conaffinity.any()
    assert model.body("appearance_cable").mass == 0
    for invalid in (points[:1], np.full((2, 3), np.nan), np.zeros((2, 3))):
        with pytest.raises(ValueError, match="cable"):
            profile.add_visual_cable(source, invalid)


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") not in ("osmesa", "egl"),
                    reason="Requires explicit headless rendering backend")
@pytest.mark.parametrize("profile_type", [RendererAppearance, WristAppearanceV2])
def test_rgb_area_lighting_preserves_segmentation_depth_and_model(profile_type):
    sim = BeamSimulation(appearance=profile_type())
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
