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

## Evidence and boundaries

Render representative matched compositions, including near-field jaw occlusion
and edge-of-frame targets. Compare them visually against the original references;
distinguish observed resemblance from unknown calibration. Verify entity-label
alignment and unchanged physical behavior where appearance-only changes apply.

Own renderer integration, focused tests and CPU qualification. Do not change
the active sweep recipe, GPU images or existing datasets/checkpoints/STOP.
Coordinate `robot_sim.py` ownership before edits. Tell the curriculum worker
which renderer configuration is available and how to instantiate it.
