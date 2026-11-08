import hashlib
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

from so101.lego_curriculum import (
    CatalogUnavailable, load_catalog, prepare_scene, visual_audit_due,
)


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.asset = self.root / "part.xml"
        self.asset.write_text("<mujoco/>")
        self.manifest = self.root / "catalog.json"
        self.report = {
            "schema_version": 1, "catalog_id": "test-only",
            "parts": [{
                "part_id": "test-slope", "description": "Test fixture",
                "family": "slope", "status": "verified",
                "dimensions_m": [.008, .008, .006],
                "geometry": {"frame_ldraw_to_mujoco": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                             "metres_per_ldu": .0004},
                "mjcf": "part.xml",
                "assets": [{"path": "part.xml",
                            "sha256": hashlib.sha256(self.asset.read_bytes()).hexdigest()}],
            }],
        }

    def write_catalog(self):
        self.manifest.write_text(json.dumps(self.report))
        return load_catalog(self.manifest)

    def plan(self, catalog, **changes):
        options = dict(seed=123, sequence=0, category="pile", task="grasp", object_count=20)
        return prepare_scene(catalog, **(options | changes))

    def test_only_verified_parts_are_sampled_and_empty_catalog_waits(self):
        self.report["parts"].append({
            "part_id": "not-ready", "description": "Not converted",
            "family": "round", "status": "imported",
        })
        catalog = self.write_catalog()
        plan = self.plan(catalog)
        self.assertEqual({obj["part_id"] for obj in plan["objects"]}, {"test-slope"})
        self.report["parts"][0]["status"] = "prepared"
        with self.assertRaisesRegex(CatalogUnavailable, "No verified"):
            self.write_catalog()

    def test_scene_uses_distinct_parts_before_repeating(self):
        other = dict(self.report["parts"][0], part_id="test-brick", family="brick")
        self.report["parts"].append(other)
        catalog = self.write_catalog()
        for sequence in range(8):
            plan = self.plan(catalog, sequence=sequence, object_count=8)
            for index in range(0, 8, 2):
                self.assertEqual(
                    {obj["part_id"] for obj in plan["objects"][index:index + 2]},
                    {"test-slope", "test-brick"},
                )

    def test_mixed_height_pile_has_disjoint_initial_vertical_envelopes(self):
        self.report["parts"].append(dict(
            self.report["parts"][0], part_id="test-tall",
            dimensions_m=[.008, .008, .02],
        ))
        catalog = self.write_catalog()
        dimensions = {part.part_id: part.dimensions_m for part in catalog.parts}
        for sequence in range(8):
            plan = self.plan(catalog, sequence=sequence, object_count=8)
            previous_top = 0.
            for obj in plan["objects"]:
                half_height = dimensions[obj["part_id"]][2] / 2
                center = obj["position_relative_to_scene_m"][2]
                self.assertGreaterEqual(center - half_height + 1e-12, previous_top + .002)
                previous_top = center + half_height

    def test_off_center_source_bounds_keep_mixed_pile_initially_disjoint(self):
        original = self.report["parts"][0]
        original["geometry"] = dict(
            original["geometry"],
            bounds_m=[[-.004, -.004, -.0048], [.004, .004, .0012]],
        )
        self.report["parts"].append(dict(
            original, part_id="test-tall", dimensions_m=[.008, .008, .02],
            geometry=dict(original["geometry"],
                          bounds_m=[[-.004, -.004, -.0188], [.004, .004, .0012]]),
        ))
        catalog = self.write_catalog()
        bounds = {part.part_id: part.vertical_bounds_m for part in catalog.parts}
        for sequence in range(8):
            plan = self.plan(catalog, sequence=sequence, object_count=8)
            previous_top = 0.
            for obj in plan["objects"]:
                lower, upper = bounds[obj["part_id"]]
                center = obj["position_relative_to_scene_m"][2]
                self.assertGreaterEqual(center + lower + 1e-12, previous_top + .002)
                previous_top = center + upper

    def test_cluster_proposals_do_not_start_with_intersecting_planar_bounds(self):
        self.report["parts"][0]["geometry"]["bounds_m"] = [
            [-.002, -.004, -.003], [.014, .004, .003],
        ]
        catalog = self.write_catalog()
        radius = math.hypot(.014, .004)
        for sequence in range(32):
            plan = self.plan(catalog, sequence=sequence, category="cluster", object_count=4)
            objects = plan["objects"]
            for index, obj in enumerate(objects):
                for other in objects[:index]:
                    distance = math.dist(obj["position_relative_to_scene_m"][:2],
                                         other["position_relative_to_scene_m"][:2])
                    self.assertGreaterEqual(distance, 2 * radius + .0001)

    def test_deterministic_pairing_and_catalog_version_binding(self):
        catalog = self.write_catalog()
        original = self.plan(catalog)
        self.assertEqual(original, self.plan(catalog))
        self.assertNotEqual(original["scene_id"], self.plan(catalog, sequence=1)["scene_id"])
        self.assertNotEqual(original["commands"][0]["object_instance_id"],
                            original["commands"][1]["object_instance_id"])
        push = self.plan(catalog, task="push")
        self.assertEqual(push["commands"][0]["object_instance_id"],
                         push["commands"][1]["object_instance_id"])
        self.assertNotEqual(push["commands"][0]["destination_id"],
                            push["commands"][1]["destination_id"])
        self.report["catalog_id"] = "different-release"
        newer = self.plan(self.write_catalog())
        self.assertNotEqual(original["scene_id"], newer["scene_id"])
        self.assertEqual(original["status"], "proposed")
        self.assertTrue(original["requires_physical_settling"])
        alternate = replace(catalog, parts=(replace(catalog.parts[0], part_id="other-part"),))
        self.assertNotEqual(original["scene_id"], self.plan(alternate)["scene_id"])

    def test_corrupt_verified_asset_rejected(self):
        self.write_catalog()
        self.asset.write_text("changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_catalog(self.manifest)

    def test_paired_target_colors_remain_distinct_across_seeded_scenes(self):
        catalog = self.write_catalog()
        for task in ("grasp", "view"):
            for sequence in range(20):
                plan = self.plan(catalog, task=task, sequence=sequence)
                colors = {obj["instance_id"]: obj["color"] for obj in plan["objects"]}
                targets = [colors[command["object_instance_id"]] for command in plan["commands"]]
                self.assertNotEqual(*targets)

    def test_separated_small_parts_leave_hand_clearance_and_center_partial_grid(self):
        plan = self.plan(self.write_catalog(), category="separated", object_count=2)
        first, second = [obj["position_relative_to_scene_m"] for obj in plan["objects"]]
        self.assertGreaterEqual(abs(second[0] - first[0]), .04)
        self.assertEqual((first[1], second[1]), (0., 0.))

    def test_asset_escape_and_symlink_rejected(self):
        self.report["parts"][0]["mjcf"] = "../outside.xml"
        with self.assertRaisesRegex(ValueError, "escapes catalog"):
            self.write_catalog()
        self.report["parts"][0]["mjcf"] = "link.xml"
        (self.root / "link.xml").symlink_to(self.asset)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.write_catalog()

    def test_duplicate_ids_and_invalid_dimensions_rejected(self):
        self.report["parts"].append(dict(self.report["parts"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate catalog part_id"):
            self.write_catalog()
        self.report["parts"].pop()
        self.report["parts"][0]["dimensions_m"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid dimensions_m"):
            self.write_catalog()

    def test_visual_audit_one_per_category_block_is_resumable(self):
        for category in ("grasp-pile-slope", "push-cluster-round"):
            selected = [
                count for count in range(1, 51)
                if visual_audit_due(category=category, accepted_count=count, seed=42)
            ]
            self.assertEqual(len(selected), 5)
            self.assertEqual([(count - 1) // 10 for count in selected], list(range(5)))
            resumed = [
                count for count in range(21, 51)
                if visual_audit_due(category=category, accepted_count=count, seed=42)
            ]
            self.assertEqual(resumed, selected[2:])

    def test_old_y_down_sign_flip_is_not_z_up_conversion(self):
        self.report["parts"][0]["geometry"]["frame_ldraw_to_mujoco"] = [
            [1, 0, 0], [0, -1, 0], [0, 0, -1],
        ]
        with self.assertRaisesRegex(ValueError, "coordinate frame"):
            self.write_catalog()


if __name__ == "__main__":
    unittest.main()
