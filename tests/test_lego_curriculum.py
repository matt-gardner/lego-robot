import hashlib
import json
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

    def test_corrupt_verified_asset_rejected(self):
        self.write_catalog()
        self.asset.write_text("changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_catalog(self.manifest)

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


if __name__ == "__main__":
    unittest.main()
