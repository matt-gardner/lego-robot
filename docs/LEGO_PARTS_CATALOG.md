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

Publish one current directory with `catalog.json`, schema version 1. Matt clarified
November 10 that historical catalog versions are not required. Use
`--catalog-id current --previous OUTPUT/current/catalog.json` for subsequent
batches. Stage and validate additions, leave unchanged assets in place, then
atomically replace only `catalog.json`; do not accumulate releases. Newly converted
MJCF assets use content-hashed filenames. A failed index installation leaves the
previous index usable. Superseded part assets are removed only under the exclusive
asset lock after the new index is installed.
Consumers resolve the current catalog at problem creation and record part IDs and
content/asset hashes, rejecting incompatible assets rather than silently replacing
geometry in existing problems. Hash provenance does not require retaining old
catalog directories.

Readers must hold `so101.lego_parts_catalog.catalog_read_lock(manifest)` across
loading the manifest and all asset reads, snapshot creation or MuJoCo compilation.
The yielded path is the manifest path. Releasing it does not pin filesystem assets:
finish copying/compiling before leaving the context. Preparation and qualification
hold a shared `.catalog.lock`, allowing consumers to read the current catalog.
Publication upgrades to an exclusive lock only for installation and superseded-tree
cleanup, waiting for existing asset readers to finish. A separate
`.catalog-publish.lock` serializes publishers throughout staging and installation;
no historical catalog trees are required.

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
early, then refresh the single current catalog with coordinated asset readers.

## Boundaries

Own catalog files and conversion tests, not `robot_sim.py`, dashboard UI,
renderer calibration or curriculum policy. Use isolated tests and scratch;
preserve existing datasets, checkpoints, STOP markers and shared edits.
Send the curriculum and dashboard workers the manifest path and file hash.
