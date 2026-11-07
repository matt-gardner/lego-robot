import json
import shutil
import subprocess

import cv2
import mujoco
import numpy as np
import pytest

import so101.simulation_playback as playback
import so101.robot_sim as robot_sim


@pytest.fixture
def model_root(tmp_path, monkeypatch):
    root = tmp_path / "model"
    root.mkdir()
    xml = '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
    (root / "model.xml").write_text(xml)
    monkeypatch.setattr(playback, "scene_xml", lambda path, version=1: (path / "model.xml").read_text())
    return root


def test_compiled_playback_preserves_model_and_avoids_xml_compilation(model_root, tmp_path, monkeypatch):
    cache = tmp_path / "playback.mjb"
    playback.prepare_model_cache(model_root, cache)
    expected = mujoco.MjModel.from_xml_string(playback.scene_xml(model_root))

    def unexpected_compile(*args):
        pytest.fail("Playback must not recompile XML inside the dashboard")

    monkeypatch.setattr(mujoco.MjModel, "from_xml_string", unexpected_compile)
    actual = playback.load_model_cache(model_root, cache)
    np.testing.assert_array_equal(actual.body_mass, expected.body_mass)
    np.testing.assert_array_equal(actual.qpos0, expected.qpos0)
    assert actual.nv == expected.nv


@pytest.mark.parametrize("change", ["asset", "xml", "version"])
def test_changed_model_or_mujoco_requires_explicit_cache_rebuild(model_root, tmp_path, monkeypatch, change):
    cache = tmp_path / "playback.mjb"
    playback.prepare_model_cache(model_root, cache)
    if change == "asset":
        (model_root / "mesh.stl").write_bytes(b"changed mesh")
    elif change == "xml":
        path = model_root / "model.xml"
        path.write_text(path.read_text().replace('size=".1"', 'size=".2"'))
    else:
        monkeypatch.setattr(mujoco, "__version__", "different-version")
    with pytest.raises(ValueError, match="cache is stale"):
        playback.load_model_cache(model_root, cache)


def test_corrupt_compiled_model_is_not_loaded(model_root, tmp_path):
    cache = tmp_path / "playback.mjb"
    playback.prepare_model_cache(model_root, cache)
    cache.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="cache is corrupt"):
        playback.load_model_cache(model_root, cache)


def test_model_changed_during_compilation_is_not_published(model_root, tmp_path, monkeypatch):
    cache = tmp_path / "playback.mjb"
    fingerprints = iter(["before", "after"])
    monkeypatch.setattr(playback, "model_fingerprint", lambda root, version=1: next(fingerprints))
    with pytest.raises(ValueError, match="changed during compilation"):
        playback.prepare_model_cache(model_root, cache)
    assert not cache.exists()
    assert not cache.with_suffix(".partial.mjb").exists()
    assert not cache.with_suffix(".json").exists()


def test_cache_manifest_records_binary_integrity(model_root, tmp_path):
    cache = tmp_path / "playback.mjb"
    playback.prepare_model_cache(model_root, cache)
    metadata = json.loads(cache.with_suffix(".json").read_text())
    assert metadata["fingerprint"] == playback.model_fingerprint(model_root)
    assert len(metadata["sha256"]) == 64


def test_scene_versions_cannot_share_a_compiled_playback_model(model_root, tmp_path):
    cache = tmp_path / "playback.mjb"
    playback.prepare_model_cache(model_root, cache, robot_sim.SCENE_VERSION)
    with pytest.raises(ValueError, match="wrong scene version"):
        playback.load_model_cache(model_root, cache)
    assert playback.load_model_cache(model_root, cache, robot_sim.SCENE_VERSION).nq == 7


@pytest.mark.parametrize("version", [robot_sim.LEGACY_SCENE_VERSION, robot_sim.SCENE_VERSION])
def test_real_recorded_playback_uses_its_version_and_preserves_source_files(tmp_path, version):
    if shutil.which("ffmpeg") is None:
        pytest.skip("Playback requires the documented ffmpeg runtime")
    source = tmp_path / "episode"
    (source / "wrist").mkdir(parents=True)
    environment = robot_sim.BeamSimulation(scene_version=version)
    try:
        environment.reset(10, real_scene=True, color_targets=True)
        initial = environment.initial
        environment.episode_root = source
        for _ in range(2):
            environment.step(environment.data.ctrl[environment.actuators], "hold")
        rows = environment.rows
    finally:
        environment.close()
    del environment
    (source / "episode.json").write_text(json.dumps({
        "initial": initial, "task": "grasp", "target": "short", "fps": 10,
    }))
    if version == robot_sim.LEGACY_SCENE_VERSION:
        rows = [{key: value for key, value in row.items() if key != "qpos"} for row in rows]
    (source / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = {path: playback.file_hash(path) for path in source.rglob("*") if path.is_file()}
    output = tmp_path / "movie.mp4"
    base_cache = tmp_path / "playback.mjb"
    cache = base_cache if version == robot_sim.LEGACY_SCENE_VERSION else (
        tmp_path / f"playback-v{version}.mjb"
    )
    playback.prepare_model_cache(robot_sim.MODEL_ROOT, cache, version)
    playback.render(source, output, robot_sim.MODEL_ROOT, model_cache=base_cache)
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(output), "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-",
    ], capture_output=True, check=True)
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, 264, 640, 3)
    assert len(frames) == 2
    original = cv2.cvtColor(cv2.imread(str(source / rows[0]["image"])), cv2.COLOR_BGR2RGB)
    assert np.mean(np.abs(frames[0, 24:, 320:].astype(float) - original)) < 8
    assert before == {path: playback.file_hash(path) for path in before}


@pytest.mark.parametrize("supplied", [False, True])
def test_simulation_constructor_preserves_default_and_injected_model(model_root, monkeypatch, supplied):
    model = mujoco.MjModel.from_xml_string(playback.scene_xml(model_root))
    compiled_paths = []

    def compile_scene(path):
        compiled_paths.append(path)
        return playback.scene_xml(path)

    def start_data(actual):
        np.testing.assert_array_equal(actual.body_mass, model.body_mass)
        if supplied:
            assert actual is model
        raise RuntimeError("Reached simulation data initialization")

    monkeypatch.setattr(robot_sim, "scene_xml", compile_scene)
    monkeypatch.setattr(mujoco, "MjData", start_data)
    with pytest.raises(RuntimeError, match="Reached simulation data initialization"):
        robot_sim.BeamSimulation(model_root, render=False, model=model if supplied else None)
    assert compiled_paths == ([] if supplied else [model_root])
