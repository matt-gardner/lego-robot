# Workstream 1: MuJoCo-ready LEGO catalog

Approved by Matt November 8, 2026. A dedicated worker owns catalog acquisition,
conversion and physical qualification. Curriculum ownership is separate.

## Outcome

Import the complete official LDraw library, pin the release and download hash,
and progressively publish usable parts. Importing all definitions is not a
requirement to verify every part before other workers can proceed.

Track imported, prepared, verified and unavailable entries with explicit
failure reasons. Preserve LDraw IDs, descriptions, authors, per-file licenses,
dependency hashes and conversion versions. Start broad: bricks, plates, tiles,
cheese slopes, round pieces, Technic and minifigure parts, not just beams.

Official source: https://library.ldraw.org/updates?latest .
Verified examples: 54200.dat, 3062b.dat and 3626c.dat.

## Preparation and qualification

Resolve recursive subfiles and transforms; convert the LDraw coordinate frame
and approximate 0.4 mm/LDU scale to MuJoCo metres. Separate visual triangles
from contact geometry. A single convex hull that fills a hollow part is not
an acceptable universal conversion. Preserve cavities and grasp-relevant
surfaces using suitable convex decomposition or compound primitives.

Publish an MJCF template for one free, dynamic part with no robot, table,
cameras or global defaults. Qualify scale, loadability, finite dynamics,
settling, plausible table contacts and important geometry/cavity fidelity.
Do not claim a part graspable by the current robot merely because it loads.
Record nominal LEGO physics; arbitrary material randomization is not requested.

## Shared catalog interface

Publish an immutable versioned directory with `catalog.json`, schema version 1:

```json
{
  "schema_version": 1,
  "catalog_id": "ldraw-2026-10-converter-v1",
  "parts": [
    {
      "part_id": "54200",
      "description": "Cheese slope",
      "family": "slope",
      "status": "verified",
      "dimensions_m": [0.008, 0.008, 0.006],
      "geometry": {
        "frame_ldraw_to_mujoco": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
        "metres_per_ldu": 0.0004
      },
      "mjcf": "parts/54200/part.xml",
      "assets": [
        {"path": "parts/54200/part.xml", "sha256": "64 lowercase hex digits"}
      ]
    }
  ]
}
```

Dimensions are the converted part-local bounding-box extents, not its scene
pose. Every MJCF and transitive asset must appear in `assets` with its exact
hash. Paths are relative to the catalog directory, contained within it, and
not symlinks. All entries have stable IDs, descriptions, families and status;
only verified entries need the prepared geometry fields. Add qualification,
source/license provenance and failure metadata without changing those fields.
The example dimensions are illustrative, not a qualification measurement.

The frame is `x'=x, y'=z, z'=-y`: LDraw's vertical axis must become MuJoCo Z,
not merely have its sign flipped. The curriculum reader rejects the older
diagonal sign-flip frame even if a historical manifest labels it verified.
Qualification must check known physical dimensions on the appropriate axes,
not just stationary contacts; a sideways part can still settle successfully.

MJCF naming/asset namespaces and grasp geometry must be coordinated with the
curriculum worker before scene integration. Publish a small verified release
early, then new immutable releases; never mutate a release already in use.

## Boundaries

Own catalog files and conversion tests, not `robot_sim.py`, dashboard UI,
renderer calibration or curriculum policy. Use isolated tests and scratch;
preserve existing datasets, checkpoints, STOP markers and shared edits.
Send the curriculum and dashboard workers the manifest path and file hash.
