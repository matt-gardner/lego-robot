# Workstream 3: Real wrist-camera simulation appearance

Approved by Matt November 8, 2026. A dedicated worker owns renderer/camera
appearance and versioned configuration, not parts conversion or curriculum.

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

## Evidence and boundaries

Render representative matched compositions, including near-field jaw occlusion
and edge-of-frame targets. Compare them visually against the original references;
distinguish observed resemblance from unknown calibration. Verify entity-label
alignment and unchanged physical behavior where appearance-only changes apply.

Own renderer integration, focused tests and CPU qualification. Do not change
the active sweep recipe, GPU images or existing datasets/checkpoints/STOP.
Coordinate `robot_sim.py` ownership before edits. Tell the curriculum worker
which renderer configuration is available and how to instantiate it.
