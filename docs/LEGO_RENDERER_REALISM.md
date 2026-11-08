# Workstream 3: Real wrist-camera simulation appearance

Approved by Matt November 8, 2026. A dedicated worker owns renderer/camera
appearance and versioned configuration, not parts conversion or curriculum.

The explicit recorder factory also supports opt-in `WristAppearanceV4` from
`so101.renderer_appearance_v4`. Independently checked source/profile hashes are
`46c1b5bad9dcf3809dcc8f6d139ba95640e148d4312c9067af5cddfe81fbf114` and
`ef9d9726fc6158803ad7798e490bbf94d6514f5642b35d5865f76ecf00507c15`.
The immutable receipt is
`/mnt/dockyard/lego-robot-data/data/lego-renderer/b887f204-wrist-soft-v4/`.
Recordings archive both v2/v3 parent JSON profiles and all four inherited
implementation sources. Wider camera composition, sharper cosmetic jaws and
lower blur are renderer changes, not grasp or transfer qualification. The
5 mm wrist-height/0.5-opening reference fixture is an uncalibrated hypothesis,
not expert-physics defaults. Existing recordings and the running pinned-v3
producer are unchanged.

## Reference and target

Inspect the original three images in
`data/real_robot/wrist_camera/`: `autopilot_scene_wrist.jpg`,
`target_teal_current_wrist.jpg`, and `wrist_current.jpg`.

They show a pale textured tabletop, soft low-contrast illumination, image
softness/noise, prominent white jaws near the lens, partial/cropped LEGO pieces
and, in the teal view, a black cable. Match these characteristics before adding
generic visual randomization. Three stills do not establish camera intrinsics,
extrinsics, material properties or physical manipulation success.

## Implementation

Improve gripper visual geometry/material, camera composition, table appearance,
lighting/shadows and image processing. Keep physical contacts, joint conventions
and privileged state out of learned-policy inputs. Any geometric camera change
must keep images, boxes, visibility and metric depth mutually consistent.

Provide reproducible versioned renderer configurations with source/reference
hashes and documented parameters. Share the exact MJCF/camera integration
interface with curriculum and catalog workers. Preserve legacy playback and
pin old generations to their original renderer; do not relabel old images.

## Integration interface

The opt-in source interface is `RendererAppearance` in
`src/so101/renderer_appearance.py`. Apply `apply_xml(assembled_xml)` before
MuJoCo compilation, `apply_model(model)` before creating a renderer, and
`apply_lighting(model)` after reset appearance changes. Use `render_rgb` and
`process_rgb(image, seed=scene_seed, frame=frame_index)` for RGB only; render
segmentation and metric depth with the same geometric camera without photometric
processing. The profile currently requires the SO101 base's `table`,
`wrist_cam` and named white-jaw materials, not a standalone part scene.

Persist `identity()` with each generation: it includes both profile and
implementation hashes. Do not bind to a mutable filename or replay with a
mismatched identity. Before recording a qualified renderer release here,
verify its final hashes and CPU render evidence from an artifact path readable
by the curriculum worker. A producer's container-local `/mnt/data` path is not
necessarily visible to other workers; use the actual shared project data mount
for cross-worker deliverables.

The first shared CPU render receipt is
`/mnt/dockyard/lego-robot-data/data/lego-renderer/b887f204-wrist-soft-v1/`.
The curriculum worker independently read its qualification and comparison image
and verified these hashes:

| Artifact | SHA-256 |
| --- | --- |
| `wrist_soft_v1.json` | `ad190884a753fabe3f2e663ac4458fce6bdffc4d4b90b2bb08f4435438f104c5` |
| `renderer_appearance.py` | `e1f2c7b79963fd5809c5ef77c341ee2bd7384ff36ed49baef907bafff055e6ce` |
| `qualification.json` | `b3b106e9e380c94418731b82d068cbf10be7597e3a47d9f826580f337c1fec9e` |
| `reference_vs_simulation.png` | `68f21263e096452da2337b62fe1ced7c3fbf9cf4cbb71414f8d7ae7f2565d538` |

The receipt records MuJoCo 3.17.0 and Mesa llvmpipe. It uses three hand-composed
legacy-beam render fixtures, not settled catalog scenes or accepted training
episodes. Table appearance is closer, but sharp angular jaws, saturated proxy
objects, missing cable and different part geometry remain visible differences.
This is an available appearance baseline, not calibrated camera matching or
real-robot transfer evidence.

The independently readable v2 release is
`/mnt/dockyard/lego-robot-data/data/lego-renderer/b887f204-wrist-soft-v2/`.
Instantiate `WristAppearanceV2()` from `so101.renderer_appearance_v2`, not the
v1 class with a v2 JSON file. It inherits the integration API and records all
three identities: profile
`27e45fc310d66bcca018113c1ef05cb7ebc41ec705e91a77a39c1d2170c0e870`,
v2 implementation
`437f2b4a881ab720b4e5e5b5b180197000e606fd1f21f1ff391f6345fb6dc9e1`
and the unchanged v1 parent implementation above.

V2 replaces non-contact jaw visuals with outward-tapered sleeves and softens
jaw highlights, blur and saturation. `add_visual_cable(xml, points_m)` optionally
adds fixed, world-space non-contact context; bind the points and resulting XML
hash, and never treat it as a physical obstacle. Curriculum independently
verified the source/profile, comparison, qualification and molded-parts fixture
hashes. The latter uses real catalog 3003/3022/3700 geometry, but remains a
hand-composed render, not a settled expert episode.

The v2 comparison is softer and less saturated than v1; its exterior still
looks angular/triangular compared with the reference's molded white jaws.
The fixture gripper opening is a composition hypothesis, not robot calibration.
V1/v2 remain immutable and opt-in; neither receipt qualifies SFT, physical
manipulation or existing sweep data. New renderer versions require independent
receipts rather than silent substitution into an already-recording attempt.

The independently verified v3 receipt is
`/mnt/dockyard/lego-robot-data/data/lego-renderer/b887f204-wrist-soft-v3/`.
Use `WristAppearanceV3` from `so101.renderer_appearance_v3` with the same inherited
integration API. Profile SHA is
`416aaf2a85acfa582a6cf315eaf2a24c8e428cb5ab0ac57f78c962aa70d3124f`;
implementation SHA is
`438eed9f8d74090fe10aba237a302d107ca98f46e0626c4594972330682504f8`.
`identity()` also binds the exact v2 profile/implementation and v1 implementation.
V3 rounds the non-contact sleeve surfaces with elliptic sections; the matched
comparison has rounder tips but still different jaw silhouette and proxy parts.
Its hand-composed CPU render qualification does not establish dynamic catalog
acceptance or calibration. Any separate static visual of physics-failed 96874
must retain that diagnostic-only status, never become a verified catalog entry
or expert example through renderer integration.

The separately verified render-only diagnostic at
`/mnt/dockyard/lego-robot-data/data/lego-renderer/b887f204-reference-static-v3/`
uses actual 96874 source geometry as a fixed non-contact visual, alongside three
distinct 3702 instances and 3003. Qualification hash
`4e6d866e29c2ce22e0321419fcc3317d3bd70ef7244f4146acd3c2217d5b20f2`
records the failed 96874 physical check explicitly; comparison hash
`0d77b62dd20599a06e470362c23120c6b749013e0c6dfaf5a4274527efd4ffda`
shows a more recognizable molded separator than the earlier proxy. Composition,
orientation and jaw occlusion still differ from the photo. This diagnostic
cannot enter the verified-only curriculum loader or establish expert acceptance.

The later, separately verified
`/mnt/dockyard/lego-robot-data/data/lego-renderer/b887f204-reference-verified-v4/`
uses the new 43-entry catalog v4, including its newly table-qualified 96874,
without the static diagnostic override. Qualification SHA is
`06d7d88e88fd06026b2b4019427a75cd063ebc632ef489572fe020d55d3ea2b7`;
comparison SHA is
`bdba00fb22aa41355cb4e0399ee6ccfafebb5a5279883c1db1589a63ccd1c312`.
The source catalog uses a new matched 2 ms part/table contact profile and only
qualifies three low-drop tabletop orientations, not robot grasping or clutch.
This does not change the earlier v3 failure/static labels. The new composition
still is a hand-composed render, with imperfect photo orientation, crop and
jaw occlusion rather than calibrated real-robot transfer evidence.

## Evidence and boundaries

Render representative matched compositions, including near-field jaw occlusion
and edge-of-frame targets. Compare them visually against the original references;
distinguish observed resemblance from unknown calibration. Verify entity-label
alignment and unchanged physical behavior where appearance-only changes apply.

Own renderer integration, focused tests and CPU qualification. Do not change
the active sweep recipe, GPU images or existing datasets/checkpoints/STOP.
Coordinate `robot_sim.py` ownership before edits. Tell the curriculum worker
which renderer configuration is available and how to instantiate it.
