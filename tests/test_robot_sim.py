import json
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from so101.robot_sim import (
    BeamSimulation, DISTRACTOR_SIZES, IMAGE_SHAPE, INITIAL_PENETRATION_TOLERANCE,
    LEGACY_SCENE_VERSION, MODEL_ROOT, SCENE_VERSION, _prepare_recovery_start,
    restore_episode, training_command,
)
from so101.collect_difficulty_sim import configure_scene
from so101.simulation_corpus import validate_clean_expert


def test_exact_failed_first_push_is_not_accepted_after_recovery():
    # Recorded example 0d5a7a3eda4b771683b5: first pass 17.03cm, recovery 2.17cm.
    def run(version):
        position = np.array([.22562779, 0., 0.])
        phases = []
        def move(goal, phase, **kwargs):
            phases.append(phase)
            if phase == "push":
                position[0] = .17034389 if phases.count("push") == 1 else .02171213
        fake = SimpleNamespace(
            scene_version=version, limits=np.array([[0., 1.]] * 6),
            data=SimpleNamespace(qpos=np.zeros(6), ctrl=np.zeros(6)),
            qadr=np.arange(6), actuators=np.arange(6),
            configure_command=lambda *args: None,
            position=lambda name: position.copy() if name == "short" else np.zeros(3),
            push_ik=lambda *args: np.zeros(6), move=move, step=lambda *args: None,
        )
        return BeamSimulation.demonstrate_push(fake, "short", "goal_left"), phases
    legacy, legacy_phases = run(LEGACY_SCENE_VERSION)
    assert legacy["success"] and legacy_phases.count("push") == 2
    corrected, phases = run(SCENE_VERSION)
    assert not corrected["success"] and phases.count("push") == 1
    assert corrected["destination_error_m"] == pytest.approx(.17034389)


@pytest.mark.parametrize("phases", [
    ["push", "push_retreat", "push_realign", "push"],
    ["close", "retry_open", "grasp_retry", "lift"],
    ["push", "push_retreat", "push"],
])
def test_clean_expert_guard_rejects_failed_prefix_even_with_final_success(phases):
    metadata = {"outcome": "success", "initial": {"expert_acceptance_version": 1}}
    with pytest.raises(ValueError, match="pass|retry"):
        validate_clean_expert(metadata, [{"phase": phase} for phase in phases])
    validate_clean_expert({"outcome": "success", "initial": {}},
                          [{"phase": phase} for phase in phases])


def test_failed_first_grasp_cannot_recover_into_clean_place(sim, monkeypatch):
    sim.reset(0)
    sim.scene_version = SCENE_VERSION
    calls = []
    def contacts(target):
        calls.append(target)
        return set() if len(calls) == 1 else {"fixed", "moving"}
    monkeypatch.setattr(sim, "contacts", contacts)
    result = sim.demonstrate_place("studded", "goal_left")
    assert not result["success"] and len(calls) == 1


@pytest.fixture
def sim():
    if not (MODEL_ROOT / "revision.json").exists():
        pytest.skip("Download the pinned model with robot_sim.py setup")
    environment = BeamSimulation(render=False, scene_version=LEGACY_SCENE_VERSION)
    try:
        yield environment
    finally:
        environment.close()


def test_seeded_resets_vary_arm_and_objects_without_exposing_object_state(sim):
    sim.reset(10)
    first = sim.initial
    assert sim.state().shape == (6,)
    sim.reset(11)
    assert not np.allclose(first["arm_state"], sim.initial["arm_state"], atol=.5)
    assert first["objects"] != sim.initial["objects"]
    sim.reset(10)
    assert sim.initial == first
    assert sim.model.neq == 0
    assert sim.model.body("moving_jaw_so101_v1").id != sim.model.body("gripper").id


def test_physical_props_and_studs_generate_contacts_without_duplicate_mass():
    environment = BeamSimulation(render=False)
    legacy = BeamSimulation(render=False, scene_version=LEGACY_SCENE_VERSION)
    try:
        environment.reset(10, real_scene=True)
        assert environment.model.body("studded").mass == pytest.approx(legacy.model.body("studded").mass)
        for name in DISTRACTOR_SIZES:
            assert environment.model.body(name).mass > 0
            assert environment.model.joint(f"{name}_free").type == mujoco.mjtJoint.mjJNT_FREE
            environment.set_body_pose(name, [.45, 0., -.003])
            mujoco.mj_forward(environment.model, environment.data)
            geom = environment.model.geom(f"{name}_geom").id
            assert any(
                geom in (contact.geom1, contact.geom2) and contact.dist < 0
                for contact in environment.data.contact
            )
            environment.set_body_pose(name, [.45, 0., .03])
            for _ in range(round(.4 / environment.model.opt.timestep)):
                mujoco.mj_step(environment.model, environment.data)
            assert environment.position(name)[2] == pytest.approx(
                -.003 + DISTRACTOR_SIZES[name][2], abs=INITIAL_PENETRATION_TOLERANCE,
            )
            environment.set_body_pose(name, [0., 0., -1.])
        stud = int(environment.studded_studs[0])
        point = environment.data.geom_xpos[stud].copy()
        point[2] += .0065
        environment.set_body_pose("green_piece", point)
        mujoco.mj_forward(environment.model, environment.data)
        assert any(stud in (contact.geom1, contact.geom2) for contact in environment.data.contact)
    finally:
        environment.close()
        legacy.close()


def test_physical_initialization_rejects_overlap_and_preserves_full_restore():
    environment = BeamSimulation(render=False)
    try:
        for seed in range(12):
            environment.reset(seed, varied_layout=True, real_scene=True, color_targets=True)
            assert not environment.initial_penetrations()
            assert environment.initial["scene_version"] == SCENE_VERSION
        expected = environment.data.qpos.copy()
        expected_velocity = environment.data.qvel.copy()
        expected_control = environment.data.ctrl.copy()
        metadata = {"initial": environment.initial, "target": "short", "task": "grasp"}
        environment.step(environment.data.ctrl[environment.actuators], "hold")
        restore_episode(environment, metadata)
        np.testing.assert_array_equal(environment.data.qpos, expected)
        np.testing.assert_array_equal(environment.data.qvel, expected_velocity)
        np.testing.assert_array_equal(environment.data.ctrl, expected_control)
        environment.set_body_pose("green_piece", environment.position("short"))
        with pytest.raises(ValueError, match="Initial scene interpenetrates"):
            environment.settle_initial_scene()
        assert max(value["depth_m"] for value in environment.initial_penetrations()) > INITIAL_PENETRATION_TOLERANCE
    finally:
        environment.close()


def test_untagged_legacy_episode_keeps_legacy_physics_and_pose_dimensions(sim):
    sim.reset(10, real_scene=True)
    metadata = {"initial": sim.initial, "target": "short", "task": "grasp"}
    environment = BeamSimulation(render=False)
    try:
        restore_episode(environment, metadata)
        assert environment.scene_version == LEGACY_SCENE_VERSION
        assert environment.model.nq == 34
        for name in DISTRACTOR_SIZES:
            assert environment.model.geom(f"{name}_geom").contype == 0
        assert np.all(environment.model.geom_contype[environment.studded_studs] == 0)
        for name, position in sim.initial["objects"].items():
            np.testing.assert_allclose(environment.position(name), position)
    finally:
        environment.close()


@pytest.mark.parametrize("task", ["grasp", "view", "push", "place"])
@pytest.mark.parametrize("level", [0, 3])
def test_corrected_scene_experts_with_valid_solid_starts(task, level):
    environment = BeamSimulation()
    try:
        case = configure_scene(environment, task, level, 90000 + level)
        assert not environment.initial_penetrations()
        assert case["difficulty"]["level"] == level
        if task == "view" and level == 3:
            assert case["difficulty"]["measurements"]["initial_visible_pixels"] == 0
        result = (
            environment.demonstrate_obstacle_grasp(case["target"])
            if task == "grasp" and level == 3 else
            environment.demonstrate_command(task, case["target"], case["destination"])
        )
        if task == "push" and level == 3:
            assert not result["success"]
            assert result["destination_error_m"] > .07
        else:
            assert result["success"], result
        if task in ("push", "place") and result["success"]:
            assert result["destination_error_m"] <= case["constraints"]["destination_tolerance_m"]
        if task == "grasp" and level == 3:
            assert result["obstacle_contacts"] == 0
    finally:
        environment.close()


@pytest.mark.parametrize("version", [2, SCENE_VERSION])
def test_dynamic_contact_guard_rejects_soft_table_but_new_physics_lifts_cleanly(version):
    environment = BeamSimulation(render=False, scene_version=version)
    try:
        environment.reset(21000, varied_layout=True, real_scene=True, color_targets=True)
        environment.contact_penetration_limit_m = .001
        environment.table_penetration_limit_m = .0003
        if version == 2:
            with pytest.raises(RuntimeError, match="Physical contact penetration"):
                environment.demonstrate("studded")
        else:
            assert environment.demonstrate("studded")["success"]
            assert environment.maximum_contact_penetration_m <= .001
            assert environment.maximum_table_penetration_m <= .0003
    finally:
        environment.close()

def test_joint_units_and_gripper_mapping_round_trip(sim):
    for fraction in (0, .5, 1):
        radians = sim.limits[:, 0] + fraction * np.ptp(sim.limits, axis=1)
        action = sim.encode(radians)
        assert action[-1] == pytest.approx(fraction * 100)
        np.testing.assert_allclose(sim.decode(action), radians, atol=1e-6)


def test_varied_layout_swaps_piece_sides_without_revealing_them_to_policy(sim):
    sides = set()
    for seed in range(12):
        sim.reset(seed, varied_layout=True)
        sides.add(np.sign(sim.position("studded")[1]))
        assert sim.initial["varied_layout"]
        assert sim.state().shape == (6,)
        assert np.sign(sim.position("studded")[1]) != np.sign(sim.position("hollow")[1])
    assert sides == {-1, 1}


def test_real_scene_adds_white_table_and_randomized_visual_distractors(sim):
    sim.reset(10, varied_layout=True, real_scene=True)
    first = sim.initial["appearance"]
    assert first["real_scene"]
    assert min(first["table_rgba"][:3]) >= .9
    assert all(position[2] > -0.1 for position in first["distractors"].values())
    sim.reset(11, varied_layout=True, real_scene=True)
    assert sim.initial["appearance"]["distractors"] != first["distractors"]
    sim.reset(10, varied_layout=True, real_scene=False)
    assert all(position[2] == -1 for position in sim.initial["appearance"]["distractors"].values())


def test_color_targets_assign_distinct_reproducible_labels_and_colors(sim):
    sim.reset(10, varied_layout=True, real_scene=True, color_targets=True)
    first = sim.initial
    labels = first["appearance"]["target_labels"]
    assert labels["studded"] != labels["hollow"]
    assert set(labels) == {"studded", "hollow", "short", "wide"}
    assert all(label.endswith(f"{name} beam") for name, label in labels.items())
    studded_color = sim.model.geom_rgba[sim.beam_geoms["studded"][0], :3].copy()
    hollow_color = sim.model.geom_rgba[sim.beam_geoms["hollow"][0], :3].copy()
    assert not np.allclose(studded_color, hollow_color)
    sim.reset(10, varied_layout=True, real_scene=True, color_targets=True)
    assert sim.initial == first
    np.testing.assert_allclose(sim.model.geom_rgba[sim.beam_geoms["studded"][0], :3], studded_color)


def test_visual_target_cue_requires_and_annotates_active_target():
    if not (MODEL_ROOT / "revision.json").exists():
        pytest.skip("Download the pinned model with robot_sim.py setup")
    environment = BeamSimulation()
    try:
        environment.reset(10, real_scene=True, color_targets=True, visual_cue_target=True)
        with pytest.raises(RuntimeError, match="active target"):
            environment.image()
        environment.active_target = "studded"
        image = environment.image()
        assert image.shape == IMAGE_SHAPE
        assert np.count_nonzero(np.all(image == 255, axis=2)) > 0
    finally:
        environment.close()


def test_exact_target_box_uses_simulator_object_identity():
    environment = BeamSimulation()
    try:
        environment.reset(10, real_scene=True, color_targets=True)
        environment.active_target = "studded"
        box = environment.target_box()
        assert len(box) == 5
        assert box[-1] == 1
        assert all(0 <= value <= 1 for value in box)
        assert box[2] > 0 and box[3] > 0
        oriented = environment.target_oriented_box()
        assert len(oriented) == 6
        assert oriented[-1] == 1
        assert all(0 <= value <= 1 for value in oriented[:3])
        assert oriented[2] > np.hypot(oriented[3], oriented[4]) > 0
        assert 0 < environment.target_depth() < 1
    finally:
        environment.close()


def test_offscreen_entity_keeps_projected_center_coordinates():
    environment = BeamSimulation()
    try:
        environment.reset(10, varied_layout=True, real_scene=True)
        box = environment.target_oriented_box("wide")
        assert not box[5]
        assert box[0] != 0
        assert box[1] < 0
    finally:
        environment.close()


def test_recorded_frames_include_the_active_target_box(tmp_path):
    environment = BeamSimulation()
    try:
        environment.reset(10)
        environment.active_target = "hollow"
        (tmp_path / "wrist").mkdir()
        environment.episode_root = tmp_path
        environment.step(environment.data.ctrl[environment.actuators].copy(), "hold")
        assert environment.rows[0]["target_box"] == pytest.approx(
            environment.target_box("hollow")
        )
        assert environment.rows[0]["target_oriented_box"] == pytest.approx(
            environment.target_oriented_box("hollow"), abs=1e-4
        )
        assert environment.rows[0]["target_depth_m"] == pytest.approx(
            environment.target_depth("hollow"), abs=1e-5
        )
        assert environment.rows[0]["command_roles"] == ["object"]
    finally:
        environment.close()


def test_recorded_push_frames_include_object_and_destination_entities(tmp_path, monkeypatch):
    environment = BeamSimulation()
    try:
        environment.reset(12, varied_layout=True, real_scene=True, color_targets=True)
        environment.configure_command(
            "push", [("short", "object"), ("goal_left", "destination")]
        )
        (tmp_path / "wrist").mkdir()
        environment.episode_root = tmp_path
        segmentation = environment._segmentation_image
        segmentation_calls = 0

        def counted_segmentation():
            nonlocal segmentation_calls
            segmentation_calls += 1
            return segmentation()

        monkeypatch.setattr(environment, "_segmentation_image", counted_segmentation)
        environment.step(
            environment.data.ctrl[environment.actuators].copy(), "hold"
        )
        row = environment.rows[0]
        assert segmentation_calls == 1
        assert environment._recording_segmentation is None
        assert row["command_entity_ids"] == ["short", "goal_left"]
        assert row["command_roles"] == ["object", "destination"]
        assert len(row["command_boxes"]) == len(row["command_depth_m"]) == 2
        assert len(row["command_depth_valid"]) == 2
        assert len(row["command_center_valid"]) == 2
    finally:
        environment.close()


def test_recorded_obstacle_grasp_includes_optional_obstacle_entity(tmp_path):
    environment = BeamSimulation()
    try:
        environment.reset(
            31003,
            varied_layout=True,
            real_scene=True,
            color_targets=True,
        )
        environment.configure_command(
            "grasp",
            [("wide", "object"), ("obstacle_box", "obstacle")],
        )
        environment.place_grasp_obstacle("wide")
        (tmp_path / "wrist").mkdir()
        environment.episode_root = tmp_path
        environment.step(
            environment.data.ctrl[environment.actuators].copy(), "hold"
        )
        row = environment.rows[0]
        assert row["command_entity_ids"] == ["wide", "obstacle_box"]
        assert row["command_roles"] == ["object", "obstacle"]
        assert row["command_visibility"] == pytest.approx([
            environment.target_oriented_box("wide")[5],
            environment.target_oriented_box("obstacle_box")[5],
        ])
        assert environment.active_obstacles == ["obstacle_box"]
    finally:
        environment.close()


def test_push_expert_keeps_wrist_roll_fixed(sim):
    sim.reset(32001, varied_layout=True, real_scene=True, color_targets=True)
    initial_wrist_roll = sim.data.qpos[sim.qadr[4]]
    result = sim.demonstrate_push("hollow", "goal_right")
    assert result["success"]
    assert sim.data.qpos[sim.qadr[4]] == pytest.approx(initial_wrist_roll)


@pytest.mark.parametrize("action", [
    np.zeros(5), np.array([np.nan, 0, 0, 0, 0, 50]),
    np.array([0, 0, 0, 0, 0, 101]), np.array([1000, 0, 0, 0, 0, 50]),
])
def test_invalid_policy_targets_do_not_move_the_simulator(sim, action):
    before = sim.data.qpos.copy()
    with pytest.raises(ValueError):
        sim.decode(action)
    np.testing.assert_array_equal(sim.data.qpos, before)


def test_small_policy_boundary_noise_is_projected_and_reported(sim):
    action = sim.encode(np.mean(sim.limits, axis=1))
    action[-1] = -0.5
    target, correction = sim.bounded_policy_action(action)
    assert correction == pytest.approx(.5)
    assert target[-1] == pytest.approx(sim.limits[-1, 0])
    assert sim.encode(target)[-1] == pytest.approx(0)


@pytest.mark.parametrize("action", [
    np.zeros(5), np.array([np.nan, 0, 0, 0, 0, 50]),
    np.array([1000, 0, 0, 0, 0, 50]),
])
def test_large_or_nonfinite_policy_errors_are_not_projected(sim, action):
    before = sim.data.qpos.copy()
    with pytest.raises(ValueError):
        sim.bounded_policy_action(action)
    np.testing.assert_array_equal(sim.data.qpos, before)


@pytest.mark.parametrize(("gripper", "bounded_gripper"), [(-20, 0), (120, 100)])
def test_gripper_targets_saturate_without_rejecting_valid_arm_targets(
    sim, gripper, bounded_gripper
):
    action = sim.encode(np.mean(sim.limits, axis=1))
    action[-1] = gripper
    target, correction = sim.bounded_policy_action(action)
    assert correction == pytest.approx(20)
    assert sim.encode(target)[-1] == pytest.approx(bounded_gripper)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("varied_layout", [False, True])
def test_both_beams_are_lifted_by_two_jaw_contact_and_retained(sim, seed, varied_layout):
    target = ("studded", "hollow")[seed % 2]
    sim.reset(seed, varied_layout=varied_layout)
    result = sim.demonstrate(target)
    assert result["success"], result
    assert result["pinch"] == ["fixed", "moving"]
    assert result["minimum_lift_m"] >= .05
    assert result["retained_samples"] == result["retention_samples"] == 10
    other = "hollow" if target == "studded" else "studded"
    assert not sim.contacts(other)
    assert sim.position(other)[2] < .01


def test_missing_pinch_prevents_lift_and_success(sim, monkeypatch):
    sim.reset(0)
    monkeypatch.setattr(sim, "contacts", lambda _: set())
    phases = []
    move = sim.move

    def observed_move(target, phase, **kwargs):
        phases.append(phase)
        return move(target, phase, **kwargs)

    monkeypatch.setattr(sim, "move", observed_move)
    result = sim.demonstrate("studded")
    assert not result["success"]
    assert "lift" not in phases


def test_recovery_prefix_is_not_recorded_as_expert_actions(sim, tmp_path):
    tasks = []

    class HoldingPolicy:
        def reset(self, seed):
            pass

        def action(self, environment, target, task=None):
            tasks.append(task)
            return environment.encode(environment.data.ctrl[environment.actuators])

    sim.reset(0, varied_layout=True)
    before = sim.data.time
    sim.episode_root = tmp_path
    _prepare_recovery_start(
        sim, HoldingPolicy(), "studded", 0, "Pick up the teal studded beam."
    )
    assert sim.initial["recovery_prefix"]["steps"] >= 20
    assert sim.initial["recovery_prefix"]["error"] is None
    assert sim.frame_start == sim.data.time > before
    assert sim.episode_root is None and sim.rows == []
    assert list(tmp_path.iterdir()) == []
    assert set(tasks) == {"Pick up the teal studded beam."}
    np.testing.assert_allclose(sim.initial["arm_state"], sim.state())
    for name in ("studded", "hollow"):
        np.testing.assert_allclose(sim.initial["objects"][name], sim.position(name))


def test_collection_can_filter_requested_target_color(sim, tmp_path, monkeypatch):
    import so101.robot_sim as robot_sim

    monkeypatch.setattr(robot_sim, "BeamSimulation", lambda *args: sim)
    monkeypatch.setattr(sim, "image", lambda: np.zeros(IMAGE_SHAPE, dtype=np.uint8))
    monkeypatch.setattr(sim, "overview", lambda path: None)
    root = tmp_path / "teal_collection"
    robot_sim.collect(
        root, MODEL_ROOT, 2, 0, 2, color_targets=True, target_color="teal",
        base_phrase="grasp",
    )
    report = json.loads((root / "collection.json").read_text())
    assert len(report["successful_episodes"]) == 2
    assert [attempt["target"] for attempt in report["attempts"]] == ["studded", "hollow"]
    assert all(attempt["target_label"].startswith("teal ") for attempt in report["attempts"])
    assert all(attempt["target_color"] == "teal" for attempt in report["attempts"])
    assert all(attempt["task"] == "grasp" for attempt in report["attempts"])
    assert all(attempt["base_phrase"] == "grasp" for attempt in report["attempts"])


def test_evaluation_stops_and_reports_a_wrong_piece_lift(sim, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import so101.robot_sim as robot_sim

    original_position = sim.position

    class WrongPiecePolicy:
        model = SimpleNamespace(config=SimpleNamespace(n_action_steps=16))

        def reset(self, seed):
            monkeypatch.setattr(sim, "position", lambda name: (
                original_position(name) + np.array([0, 0, .06 if name == "hollow" else 0])
            ))
            monkeypatch.setattr(sim, "contacts", lambda name: (
                {"fixed", "moving"} if name == "hollow" else set()
            ))

        def action(self, environment, target, task=None):
            return environment.encode(environment.data.ctrl[environment.actuators])

    monkeypatch.setattr(robot_sim, "_WristPolicy", lambda *args: WrongPiecePolicy())
    monkeypatch.setattr(robot_sim, "BeamSimulation", lambda *args: sim)
    monkeypatch.setattr(sim, "image", lambda: np.zeros(IMAGE_SHAPE, dtype=np.uint8))
    monkeypatch.setattr(sim, "overview", lambda path: None)
    root = tmp_path / "evaluation"
    robot_sim.evaluate(tmp_path / "checkpoint", root, MODEL_ROOT, 1, 0, 5, "cpu")
    result = json.loads((root / "evaluation.json").read_text())["episodes"][0]
    assert not result["success"]
    assert result["steps"] == 1
    assert result["error"] == "Policy lifted the unrequested hollow beam"
    row = json.loads((root / "episode_000/frames.jsonl").read_text().strip())
    assert row["other_elevation"] >= .05
    assert row["target_contacts"] == []


def test_evaluation_can_filter_requested_target_color(sim, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import so101.robot_sim as robot_sim

    class HoldPolicy:
        model = SimpleNamespace(config=SimpleNamespace(n_action_steps=16))

        def reset(self, seed):
            pass

        def action(self, environment, target, task=None):
            return environment.encode(environment.data.ctrl[environment.actuators])

    monkeypatch.setattr(robot_sim, "_WristPolicy", lambda *args: HoldPolicy())
    monkeypatch.setattr(robot_sim, "BeamSimulation", lambda *args: sim)
    monkeypatch.setattr(sim, "image", lambda: np.zeros(IMAGE_SHAPE, dtype=np.uint8))
    monkeypatch.setattr(sim, "overview", lambda path: None)
    root = tmp_path / "teal_evaluation"
    robot_sim.evaluate(
        tmp_path / "checkpoint", root, MODEL_ROOT, 2, 0, 1, "cpu",
        color_targets=True, target_color="teal",
    )
    report = json.loads((root / "evaluation.json").read_text())
    assert report["target_color"] == "teal"
    assert [episode["target"] for episode in report["episodes"]] == ["studded", "hollow"]
    assert all(episode["target_label"].startswith("teal ") for episode in report["episodes"])


def test_frames_pair_pre_action_state_with_next_target_and_physics_clock(sim, tmp_path, monkeypatch):
    sim.reset(0)
    sim.episode_root = tmp_path
    (tmp_path / "wrist").mkdir()
    monkeypatch.setattr(sim, "image", lambda: np.zeros(IMAGE_SHAPE, dtype=np.uint8))
    before = sim.state()
    before_qpos = sim.data.qpos.copy()
    target = sim.data.ctrl[sim.actuators].copy()
    target[0] += .01
    sim.step(target, "test")
    sim.step(target, "test")
    np.testing.assert_allclose(sim.rows[0]["state"], before)
    np.testing.assert_allclose(sim.rows[0]["qpos"], before_qpos)
    np.testing.assert_allclose(sim.rows[0]["action"], sim.encode(target))
    assert sim.rows[0]["elapsed"] == pytest.approx(0)
    assert sim.rows[1]["elapsed"] == pytest.approx(.1)
    assert set(sim.rows[0]) == {
        "index", "elapsed", "camera_elapsed", "camera_age", "state", "action", "image", "phase", "qpos"
    }


def test_training_is_local_wrist_only_and_rejects_hardware_datasets(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    info = {"robot_type": "so101_sim", "features": {"observation.images.wrist": {}}}
    path = dataset / "meta/info.json"
    path.write_text(json.dumps(info))
    command = training_command(dataset, tmp_path / "training", 2, 1, "cpu")
    assert "--policy.push_to_hub=false" in command
    assert "--wandb.enable=false" in command
    assert "--steps=2" in command
    assert "--policy.input_features=null" in command
    assert "--policy.n_action_steps=16" in command
    assert "--policy.chunk_size=16" in command
    assert "--save_freq=1000" in command
    assert "--num_workers=4" in command
    assert "--persistent_workers=true" in command
    continued = training_command(dataset, tmp_path / "continued", 100, 8, "cpu", "/local/checkpoint", 3e-5)
    assert "--policy.path=/local/checkpoint" in continued
    assert "--policy.optimizer_lr=3e-05" in continued
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Learning rate"):
            training_command(dataset, tmp_path / "bad", 100, 8, "cpu", learning_rate=invalid)
    info["robot_type"] = "so101_follower"
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="simulation dataset"):
        training_command(dataset, tmp_path / "training", 2, 1, "cpu")
