import hashlib
import json
import threading

import mujoco
import numpy as np
import pytest

from so101 import photo_pile_generator as generator


@pytest.fixture
def photo_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    bodies = "".join(
        f'<body name="brick_{index}" pos="0 0 .1"><freejoint/>'
        '<geom type="box" size=".004 .004 .004" mass=".002" '
        'solref=".005 1" friction=".5 .005 .0001"/></body>'
        for index in range(90)
    )
    xml = (
        '<mujoco><option timestep=".0005" iterations="100" noslip_iterations="20" '
        'tolerance="1e-10" cone="elliptic" impratio="10" gravity="0 0 -9.81" '
        'integrator="implicitfast"/><worldbody>'
        '<geom type="plane" size="1 1 .01"/>'
        f'{bodies}</worldbody></mujoco>'
    )
    (source / "model.xml").write_text(xml)
    (source / "receipt.json").write_text(json.dumps({
        "model_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "catalog_sha256": "test-catalog",
    }))
    return source


def test_fresh_twenty_settles_without_source_saved_state(photo_source, tmp_path):
    output = tmp_path / "scene"
    stages = []
    receipt = generator.generate_scene(
        photo_source, output, count=20, seed=20261111, progress=stages.append)
    verified = generator.verify_scene(output)
    assert receipt == verified
    assert [stage["inserted"] for stage in stages] == list(range(1, 21))
    assert receipt["restored_original_integration_state"] is False
    assert receipt["stationarity"]["maximum_final_speed"] <= .0005
    assert receipt["maximum_insertion_penetration_m"] <= .005
    model = mujoco.MjModel.from_xml_path(str(output / "model.xml"))
    assert model.nbody == 21
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.noslip_iterations == 20
    assert np.array_equal(model.opt.gravity, [0., 0., -9.81])


def test_minimum_count_and_source_hash_protection(photo_source):
    with pytest.raises(ValueError, match="20 and 90"):
        generator.prepare_model(photo_source, 19, 1)
    with (photo_source / "model.xml").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="does not match"):
        generator.prepare_model(photo_source, 20, 1)


def test_rejections_resume_and_stop_preserve_attempts(photo_source, tmp_path, monkeypatch):
    root = tmp_path / "pool"
    seeds = []
    monkeypatch.setattr(generator, "resource_check", lambda root: None)

    def rejected(source, output, *, seed, **kwargs):
        seeds.append(seed)
        raise generator.RejectedPile("expected physical timeout")

    monkeypatch.setattr(generator, "generate_scene", rejected)
    generator.run(root, photo_source, seed=11, count=20, attempts=1)
    first = (root / "attempt_000000000.json").read_bytes()
    generator.run(root, photo_source, seed=11, count=20, attempts=1)
    assert seeds == [11, 12]
    assert (root / "attempt_000000000.json").read_bytes() == first
    assert json.loads((root / "attempt_000000001.json").read_text())["status"] == "rejected"
    stopping = threading.Event()
    stopping.set()
    generator.run(root, photo_source, seed=11, count=20, attempts=1, stopping=stopping)
    assert seeds == [11, 12]
    assert json.loads((root / "status.json").read_text())["status"] == "stopped"
    with pytest.raises(ValueError, match="different generator"):
        generator.run(root, photo_source, seed=11, count=21, attempts=1)
