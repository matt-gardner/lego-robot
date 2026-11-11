import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import xml.etree.ElementTree as ET

from so101.piece_recognition_data import (
    assign_splits, digest, discover, prepare, read_bound, visual_signature,
)


class RecognitionDataTests(unittest.TestCase):
    def test_body_lookup_matches_geometry_masks_and_ignores_non_geometry(self):
        import mujoco
        import numpy as np
        from so101.piece_recognition_render import segmented_bodies

        model = SimpleNamespace(ngeom=4, geom_bodyid=np.array([0, 1, 1, 2]))
        segmentation = np.array([
            [[0, mujoco.mjtObj.mjOBJ_GEOM], [1, mujoco.mjtObj.mjOBJ_GEOM],
             [2, mujoco.mjtObj.mjOBJ_GEOM], [3, mujoco.mjtObj.mjOBJ_GEOM]],
            [[-1, -1], [1, mujoco.mjtObj.mjOBJ_SITE],
             [4, mujoco.mjtObj.mjOBJ_GEOM], [-1, mujoco.mjtObj.mjOBJ_GEOM]],
        ], dtype=np.int32)
        actual = segmented_bodies(model, segmentation)
        for body in (0, 1, 2):
            expected = ((segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
                        & np.isin(segmentation[:, :, 0],
                                  np.flatnonzero(model.geom_bodyid == body)))
            np.testing.assert_array_equal(actual == body, expected)

    def test_label_context_restores_rgb_samples_and_closes_after_failure(self):
        from so101.piece_recognition_render import label_context

        model = SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=4)))
        renderer = MagicMock()
        renderer.__enter__.return_value = renderer

        def create(*args, **kwargs):
            self.assertEqual(model.vis.quality.offsamples, 0)
            return renderer

        with patch("so101.piece_recognition_render.mujoco.Renderer", side_effect=create):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                with label_context(model):
                    self.assertEqual(model.vis.quality.offsamples, 4)
                    raise RuntimeError("capture failed")
        renderer.__exit__.assert_called_once()
        renderer.enable_segmentation_rendering.assert_called_once()
        with patch("so101.piece_recognition_render.mujoco.Renderer",
                   side_effect=RuntimeError("context failed")):
            with self.assertRaisesRegex(RuntimeError, "context failed"):
                with label_context(model):
                    self.fail("Context creation should fail")
        self.assertEqual(model.vis.quality.offsamples, 4)

    def test_split_unions_raw_source_retask_and_exact_model_without_class_leakage_fixes(self):
        scenes = [
            {"source": "raw", "group_keys": ["receipt:a", "model:raw"]},
            {"source": "task", "group_keys": ["receipt:a", "family:b"]},
            {"source": "retask", "group_keys": ["family:b", "model:task"]},
            {"source": "copy", "group_keys": ["model:task"]},
            *[{"source": f"other-{i}", "group_keys": [f"receipt:other-{i}"]}
              for i in range(12)],
        ]
        result = assign_splits(copy.deepcopy(scenes), 61000)
        self.assertEqual(len({row["family"] for row in result[:4]}), 1)
        self.assertEqual(len({row["split"] for row in result[:4]}), 1)
        self.assertEqual({row["split"] for row in result}, {"train", "development", "test"})
        self.assertEqual(result, assign_splits(copy.deepcopy(scenes), 61000))
        with self.assertRaisesRegex(ValueError, "three independent"):
            assign_splits(scenes[:4], 61000)

    def test_geometry_recovery_ignores_body_pose_color_names_but_not_shape(self):
        xml = """<mujoco><asset><mesh name="shape" vertex="0 0 0 1 0 0 0 1 0"
                 face="0 1 2"/></asset><worldbody><body name="brick" pos="1 2 3">
                 <geom name="visual" type="mesh" mesh="shape" group="2" rgba="1 0 0 1"/>
                 </body></worldbody></mujoco>"""
        tree = ET.fromstring(xml)
        body = tree.find("./worldbody/body")
        original = visual_signature(tree, body)
        body.set("pos", "9 8 7")
        body.set("name", "renamed")
        body.find("geom").set("rgba", "0 1 0 1")
        self.assertEqual(original, visual_signature(tree, body))
        tree.find("./asset/mesh").set("vertex", "0 0 0 2 0 0 0 1 0")
        self.assertNotEqual(original, visual_signature(tree, body))

    def test_inventory_recovers_placeholders_only_from_matching_geometry_and_records_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml = """<mujoco><asset><mesh name="m" vertex="0 0 0 1 0 0 0 1 0"
                     face="0 1 2"/></asset><worldbody><body name="brick_0">
                     <geom type="mesh" mesh="m" group="2"/></body></worldbody></mujoco>"""
            tree = ET.fromstring(xml)
            signature = visual_signature(tree, tree.find("./worldbody/body"))
            for name in ("known", "unresolved", ".staging"):
                pile = root / name
                pile.mkdir()
                (pile / "model.xml").write_text(xml if name != "unresolved" else
                                                xml.replace("1 0 0", "2 0 0"))
                (pile / "receipt.json").write_text(json.dumps({
                    "status": "settled", "integration_state": [],
                    "objects": [{"instance_id": "brick_0", "part_id": "catalog-demo"}],
                }))
            scenes, excluded = discover([root], {signature: "3001"})
            self.assertEqual(len(scenes), 1)
            self.assertEqual(scenes[0]["labels"][0]["part_id"], "3001")
            self.assertEqual(scenes[0]["recovered_identities"], 1)
            self.assertEqual(excluded, [{"source": str(root / "unresolved"),
                                         "reason": "unresolved_exact_part_identity"}])

    def test_launch_preparation_is_only_argv_and_requires_immutable_image(self):
        script = Path(__file__).parents[1] / "scripts" / "prepare_piece_recognition_launch.py"
        spec = importlib.util.spec_from_file_location("recognition_launch", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        paths = (Path("/work/experiment"), Path("/checkpoints/model"), Path("/cache/hf"))
        with self.assertRaisesRegex(ValueError, "immutable"):
            module.command("worker:latest", *paths, "recognition-test")
        argv = module.command("worker@sha256:" + "a" * 64, *paths, "recognition-test")
        self.assertEqual(argv[:3], ["docker", "run", "--detach"])
        self.assertIn("so101.train_piece_recognition", argv)
        self.assertNotIn("--resume", argv)
        self.assertEqual(argv[argv.index("python") - 1], "worker@sha256:" + "a" * 64)
        frozen = module.command("worker@sha256:" + "a" * 64, *paths, "recognition-frozen",
                                freeze_backbone=True)
        self.assertIn("output-frozen", frozen[frozen.index("--output") + 1])

    def test_preparation_pins_used_assets_and_never_renders_or_loads_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog" / "current"
            catalog.mkdir(parents=True)
            (catalog.parent / ".catalog.lock").touch()
            part = catalog / "part.xml"
            part.write_text("""<mujoco><asset><mesh name="m" vertex="0 0 0 1 0 0 0 1 0"
                            face="0 1 2"/></asset><worldbody><body>
                            <geom type="mesh" mesh="m" group="2"/></body></worldbody></mujoco>""")
            (catalog / "catalog.json").write_text(json.dumps({
                "parts": [{"part_id": "3001", "status": "verified", "family": "brick",
                           "description": "Brick 2 x 4", "mjcf": "part.xml",
                           "assets": [{"path": "part.xml", "sha256": digest(part)}]}],
            }))
            piles = root / "piles"
            for index in range(3):
                pile = piles / str(index)
                pile.mkdir(parents=True)
                (pile / "model.xml").write_text(f'<mujoco model="pile-{index}"/>')
                (pile / "receipt.json").write_text(json.dumps({
                    "status": "settled", "integration_state": [index],
                    "objects": [{"instance_id": "body", "part_id": "3001"}],
                }))
            output = root / "prepared"
            with patch("so101.piece_recognition_data.photo_identity_map", return_value={}):
                report = prepare([piles], catalog / "catalog.json", root / "unused", output, 7)
            self.assertEqual(len(report["scenes"]), 3)
            self.assertEqual(set(report["parts"]), {"3001"})
            self.assertEqual({scene["split"] for scene in report["scenes"]},
                             {"train", "development", "test"})
            self.assertEqual((output / "catalog" / "part.xml").read_bytes(), part.read_bytes())
            expected = digest(part)
            part.write_text("changed by publisher")
            with self.assertRaisesRegex(ValueError, "Changed experiment input"):
                read_bound(part, expected)
            self.assertEqual(digest(output / "catalog" / "part.xml"), expected)


if __name__ == "__main__":
    unittest.main()
