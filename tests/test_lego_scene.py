import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from so101.lego_curriculum import load_catalog, prepare_scene
from so101.lego_scene import assemble_scene, capture_scene_state, restore_scene, settle_scene
from so101.catalog_sim import CatalogSimulation


BASE = """<mujoco><compiler angle="radian"/>
<option timestep=".001" gravity="0 0 -9.81" integrator="implicitfast" iterations="100"/>
<worldbody><geom name="table" type="plane" size="1 1 .1"
solref=".004 1" solimp=".99 .999 .0001"/></worldbody></mujoco>"""
PART = """<mujoco><compiler angle="radian"/><asset>
<material name="plastic" rgba="1 0 0 1"/></asset><worldbody>
<body name="part"><freejoint name="part_free"/>
<geom name="collision_0" type="box" size=".004 .004 .003"
mass=".001" material="plastic" solref=".004 1" solimp=".99 .999 .0001"/>
</body></worldbody></mujoco>"""


class LegoSceneTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "part.xml").write_text(PART)
        manifest = {
            "schema_version": 1, "catalog_id": "fixture-only",
            "parts": [{
                "part_id": "fixture", "description": "Synthetic test box",
                "family": "brick", "status": "verified",
                "dimensions_m": [.008, .008, .006], "mjcf": "part.xml",
                "geometry": {"frame_ldraw_to_mujoco": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                             "metres_per_ldu": .0004},
                "assets": [{"path": "part.xml", "sha256": hashlib.sha256(PART.encode()).hexdigest()}],
            }],
        }
        (self.root / "catalog.json").write_text(json.dumps(manifest))
        self.catalog = load_catalog(self.root / "catalog.json")
        self.plan = prepare_scene(self.catalog, seed=10, sequence=0,
                                  category="separated", task="grasp", object_count=3)

    def assemble(self, plan=None):
        return assemble_scene(BASE, self.catalog, self.plan if plan is None else plan,
                              scene_origin_m=(0., 0., 0.))

    def test_dynamic_instances_have_distinct_assets_joints_and_materials(self):
        xml = self.assemble()
        model = mujoco.MjModel.from_xml_string(xml)
        self.assertEqual((model.nq, model.nv), (21, 18))
        self.assertEqual(model.nmat, 3)
        for obj in self.plan["objects"]:
            name = obj["instance_id"]
            self.assertGreater(model.body(name).id, 0)
            geom = model.geom(name + "__collision_0")
            self.assertEqual(model.material(name + "__plastic").id, model.geom_matid[geom.id])

    def test_settle_snapshot_restores_same_next_step(self):
        xml = self.assemble()
        model, data, snapshot = settle_scene(xml, self.plan)
        self.assertEqual(snapshot["status"], "settled")
        self.assertEqual(len(snapshot["objects"]), 3)
        self.assertLessEqual(snapshot["maximum_penetration_m"], .001)
        restored = mujoco.MjData(model)
        restore_scene(model, restored, snapshot, xml)
        mujoco.mj_step(model, data)
        mujoco.mj_step(model, restored)
        np.testing.assert_allclose(data.qpos, restored.qpos, rtol=0, atol=1e-12)
        np.testing.assert_allclose(data.qvel, restored.qvel, rtol=0, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "different scene model"):
            restore_scene(model, restored, snapshot, xml + " ")
        before = restored.qpos.copy()
        inconsistent = copy.deepcopy(snapshot)
        inconsistent["qpos"][0] += 1.
        with self.assertRaisesRegex(ValueError, "disagrees"):
            restore_scene(model, restored, inconsistent, xml)
        np.testing.assert_array_equal(restored.qpos, before)
        model.body_mass[1] *= 2
        with self.assertRaisesRegex(ValueError, "runtime physics"):
            restore_scene(model, restored, snapshot, xml)
        model.body_mass[1] /= 2
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        with self.assertRaisesRegex(ValueError, "runtime physics"):
            restore_scene(model, restored, snapshot, xml)

    def test_invalid_instances_and_catalog_binding_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan["objects"][1]["instance_id"] = plan["objects"][0]["instance_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.assemble(plan)
        plan = copy.deepcopy(self.plan)
        plan["catalog_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.assemble(plan)
        (self.root / "part.xml").write_text(PART + " ")
        with self.assertRaisesRegex(ValueError, "changed after catalog validation"):
            self.assemble()

    def test_settle_rejects_penetration_and_timeout(self):
        plan = copy.deepcopy(self.plan)
        plan["objects"][0]["position_relative_to_scene_m"][2] = 0.
        with self.assertRaisesRegex(ValueError, "penetration"):
            settle_scene(self.assemble(plan), plan)
        with self.assertRaisesRegex(ValueError, "did not settle"):
            settle_scene(self.assemble(), self.plan, maximum_seconds=.001)

    def test_template_cannot_change_global_physics(self):
        template = ET.fromstring(PART)
        ET.SubElement(template, "option", gravity="0 0 0")
        raw = ET.tostring(template)
        (self.root / "part.xml").write_bytes(raw)
        manifest = json.loads((self.root / "catalog.json").read_text())
        manifest["parts"][0]["assets"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
        (self.root / "catalog.json").write_text(json.dumps(manifest))
        self.catalog = load_catalog(self.root / "catalog.json")
        self.plan = prepare_scene(self.catalog, seed=10, sequence=0,
                                  category="separated", task="grasp", object_count=3)
        with self.assertRaisesRegex(ValueError, "global settings"):
            self.assemble()

    def test_robot_registry_and_recovery_use_actual_state_without_failed_prefix(self):
        sim = CatalogSimulation(self.catalog, self.plan, render=False)
        self.addCleanup(sim.close)
        self.assertEqual(set(sim.beam_geoms), {obj["instance_id"] for obj in self.plan["objects"]})
        name = self.plan["objects"][0]["instance_id"]
        joint = sim.model.joint(name + "__part_free")
        address = sim.model.jnt_qposadr[joint.id]
        sim.data.qpos[address + 3:address + 7] = [np.cos(.6), 0., 0., np.sin(.6)]
        mujoco.mj_forward(sim.model, sim.data)
        snapshot = capture_scene_state(sim.model, sim.data, sim.xml, sim.plan,
                                       status="failed-attempt")
        sim.rows = [{"phase": "failed_prefix"}]
        sim.use_recovery_start(snapshot, parent_example_id="failed-parent")
        self.assertAlmostEqual(sim.initial["yaws"][name], 1.2)
        self.assertEqual(sim.plan["scene_family_id"], self.plan["scene_id"])
        self.assertNotEqual(sim.plan["scene_id"], self.plan["scene_id"])
        sim.restore_initial()
        self.assertEqual(sim.rows, [])
        observations = []
        sim.step = lambda hold, phase: observations.append((hold.copy(), phase))
        sim.observe_recovery_start(name, "grasp", None)
        self.assertEqual([phase for _, phase in observations], ["recovery_observe"] * 4)
        self.assertEqual(sim.command_entities, [(name, "object")])
        for hold, _ in observations:
            np.testing.assert_allclose(hold, sim.data.qpos[sim.qadr])
        self.assertEqual(sim.initial["recovery_parent_example_id"], "failed-parent")

    def test_grasp_selects_equivalent_heading_before_recording_any_action(self):
        sim = CatalogSimulation(self.catalog, self.plan, render=False)
        self.addCleanup(sim.close)
        name = self.plan["objects"][0]["instance_id"]
        rotation = sim.data.body(name).xmat.reshape(3, 3)
        primary = np.arctan2(rotation[1, 0], rotation[0, 0]) + np.pi / 2
        events = []
        def ik(position, yaw):
            events.append(("ik", yaw))
            if abs(yaw - primary) < 1e-9:
                raise RuntimeError("Unreachable original heading")
            return sim.data.qpos[sim.qadr].copy()
        sim.ik = ik
        sim.move = lambda goal, phase: events.append(("move", phase))
        sim.contacts = lambda target: {"fixed", "moving"}
        _, result = sim._grasp(name)
        self.assertTrue(result["success"])
        self.assertEqual([event[0] for event in events[:3]], ["ik"] * 3)
        self.assertAlmostEqual(events[1][1], primary + np.pi)
        self.assertEqual([value for kind, value in events if kind == "move"],
                         ["approach", "descend", "close", "lift"])


if __name__ == "__main__":
    unittest.main()
