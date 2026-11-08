# Saved-pile appearance rendering

`so101.render_pile` rerenders an existing `model.xml` and
`settled-state.npz` without stepping, repositioning or freezing objects.
It is independent of the versioned wrist-camera profiles and does not
change their identities or producer defaults.

```bash
PYTHONPATH=src MUJOCO_GL=osmesa python -m so101.render_pile \
  --source /path/to/saved-pile \
  --output /path/to/new-render
```

Use the repository's isolated Linux simulation environment and installed
OSMesa libraries. Do not rebuild the host-mounted virtual environment.
The output directory must not already exist.

## Appearance profile

The default profile is `src/so101/renderer_profiles/pile_photo_v1.json`.
`--profile` selects an explicit alternative. Published profile/source bytes
must remain pinned; introduce a new version for later release changes.

The renderer fits the full visible pile to the requested frame occupancy,
replaces the checkerboard with deterministic fine surface grain, and adds
plastic specular highlights. Explicit corner normals smooth shallow mesh
faceting while retaining sharp molded edges; mesh vertices and faces are
unchanged. Stratified light samples approximate a broad light source.
Supersampling and slight final softness reduce aliasing without added noise.

## Outputs and checks

- `pile.png`: appearance image.
- `synchronized_labels.npz`: segmentation and metric depth from the same
  geometric camera at the output resolution.
- `receipt.json`: source model/state and profile/implementation hashes,
  physical model hash, actual visible bounding box and frame occupancy.

Compilation must preserve the original physical model fingerprint, and
rendering must preserve the saved qpos/qvel exactly. Clipped pile geometry
and nonfinite depth are errors. Contact geometry, masses, inertia, joints
and solver settings are not restyled.

The first input pile uses detailed catalog visual meshes but convex-hull
collision approximations. Its appearance rerender does not establish native
cavity, clutch or training-data qualification. Lighting is an OpenGL
area-source approximation, not path-traced global illumination. Better
appearance is not camera calibration or evidence of real-robot transfer.
