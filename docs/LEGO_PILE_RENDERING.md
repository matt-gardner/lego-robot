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

## Optional CPU path tracing

`so101.pathtrace_pile` consumes a pinned `scene.json` world-space PLY
export, not arbitrary MuJoCo XML. Install `requirements/pathtracing.txt`
in a separate isolated environment; LLVM must be available to Mitsuba. It uses CPU
`llvm_ad_rgb`, four threads, indirect illumination, a broad area source
and rough plastic. It does not replace the raster or wrist renderer.

```bash
PYTHONPATH=src python -m so101.pathtrace_pile \
  --scene /path/to/export/scene.json --output /path/to/new-render
```

The manifest contains `objects` with `mesh` and normalized sRGB `rgb`,
`table_z_m`, and `camera` with `origin`, `target`, `up`, `fovy`.
Relative mesh paths resolve beside the manifest.
For an explicitly relocated export, `--mesh-root` selects the directory
containing its mesh basenames; resolved meshes are hashed in the receipt.
Export world-space
visual vertices/normals without applying MuJoCo mesh centering twice.
For mono camera matching, average both MuJoCo eye transforms, not just
the left eye. Use the source plane's actual height to avoid hovering.

The pinned profile is `src/so101/renderer_profiles/pile_pathtrace_v1.json`.
The output receipt records mesh, manifest, implementation, profile and
image hashes plus backend/settings. This appearance-only backend does
not produce synchronized labels/depth or simulate physics. Existing
saved-state and collision qualification must accompany the export;
photorealism does not upgrade it. The plastic response is an appearance
assumption, not measured ABS. True molded silhouette bevels, measured
table material and calibrated optics remain absent.

On the saved 90-piece example, a coplanar 80-micrometer normal-bevel
experiment did not visibly improve the molded edges, so it was not
adopted. A 1-millimeter aperture-radius thin-lens probe added foreground
blur; the default retains the inspectable pinhole view rather than
assuming unmeasured optics. Further substantial gains require actual
rounded visual geometry and material/optical measurements, not heavier
blur or arbitrary noise.
