"""Selectable online RGB rendering; physics and recorded-reset identity stay separate."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from so101.pathtrace_pile import DEFAULT_PROFILE
from so101.pathtrace_recordings import FrameRenderer, cad_render_xml, visible_geoms
from so101.robot_sim import BeamSimulation, scene_xml


class PathtraceSimulation(BeamSimulation):
    """Path tracing of a forward-only CAD visual twin, never the physics model."""

    def __init__(self, *args, samples: int | None = None, threads: int | None = None,
                 backend: str | None = None, visual_xml: str | None = None, **kwargs):
        self.pathtrace_backend = backend if backend is not None else getattr(
            self, "pathtrace_backend", "cpu-pathtrace",
        )
        if self.pathtrace_backend not in ("cpu-pathtrace", "gpu-pathtrace"):
            raise ValueError(f"Unknown pathtrace backend: {self.pathtrace_backend}")
        self.pathtrace_samples = samples if samples is not None else getattr(
            self, "pathtrace_samples", 16,
        )
        self.pathtrace_threads = threads if threads is not None else getattr(
            self, "pathtrace_threads", 4,
        )
        if self.pathtrace_samples < 1 or self.pathtrace_threads < 1:
            raise ValueError("Pathtrace samples and threads must be positive")
        if (kwargs.get("appearance") is not None
                or kwargs.get("model") is not None and visual_xml is None):
            raise ValueError("Online path tracing requires the standard recorded scene identity")
        super().__init__(*args, **kwargs)
        try:
            profile = json.loads(DEFAULT_PROFILE.read_bytes())
            profile.update(version="rl-wrist-pathtrace-v1", width=320, height=240,
                           samples=self.pathtrace_samples,
                           threads=self.pathtrace_threads)
            xml = cad_render_xml(
                visual_xml if visual_xml is not None else scene_xml(self.model_root, self.scene_version),
                (self.model_root / "so101.xml").read_text(),
            )
            self.rgb_model = mujoco.MjModel.from_xml_string(xml)
            if (self.rgb_model.nq != self.model.nq or self.rgb_model.ngeom != self.model.ngeom
                    or any(self.rgb_model.geom(i).name != self.model.geom(i).name
                           for i in range(self.model.ngeom))):
                raise ValueError("CAD visual twin does not match live scene topology")
            self.rgb_data = mujoco.MjData(self.rgb_model)
            variant = "cuda_ad_rgb" if self.pathtrace_backend == "gpu-pathtrace" else "llvm_ad_rgb"
            self.pathtracer = FrameRenderer(self.rgb_model, profile, variant=variant,
                                           reuse_meshes=True)
            self.rgb_identity = {
                "backend": self.pathtrace_backend, "variant": variant, "profile": profile,
                "reuse_meshes": True,
                "mujoco_version": mujoco.__version__,
                "mitsuba_version": self.pathtracer.mi.__version__,
                "profile_sha256": hashlib.sha256(DEFAULT_PROFILE.read_bytes()).hexdigest(),
                "implementation_sha256": {
                    name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ("rl_renderer.py", "pathtrace_recordings.py", "pathtrace_pile.py",
                                 "robot_sim.py")
                },
                "robot_xml_sha256": hashlib.sha256(
                    (self.model_root / "so101.xml").read_bytes(),
                ).hexdigest(),
                "scene_xml_sha256": hashlib.sha256(
                    (visual_xml if visual_xml is not None
                     else scene_xml(self.model_root, self.scene_version)).encode(),
                ).hexdigest(),
                "reset_renderer": None,
                "scope": "Appearance-only RGB; not camera calibration or transfer qualification",
            }
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        # MuJoCo frees GL names in the current context, not necessarily this rollout's.
        if self.renderer is not None and self.renderer._gl_context is not None:
            self.renderer._gl_context.make_current()
        super().close()

    def _render_wrist_rgb(self) -> np.ndarray:
        if self.renderer is None:
            raise RuntimeError("Rendering was not enabled")
        # Reset changes these model fields; CAD-local geom transforms stay in the visual twin.
        for field in ("body_pos", "body_quat", "geom_rgba", "mat_rgba", "geom_size",
                      "cam_pos", "cam_quat", "cam_fovy"):
            getattr(self.rgb_model, field)[:] = getattr(self.model, field)
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model, state_spec))
        mujoco.mj_getState(self.model, self.data, state, state_spec)
        mujoco.mj_setState(self.rgb_model, self.rgb_data, state, state_spec)
        mujoco.mj_forward(self.rgb_model, self.rgb_data)
        self.pathtracer.visible = visible_geoms(self.rgb_model)
        frame = round(self.data.time / self.model.opt.timestep)
        seed = int(np.random.SeedSequence(
            [int(self.initial.get("seed", 0)), frame, 1108],
        ).generate_state(1)[0])
        started = perf_counter()
        image = self.pathtracer.render(self.rgb_data, seed - self.pathtracer.profile["seed"])
        self.last_rgb_seconds = perf_counter() - started
        return image


def create_simulation(model_root: Path, *, backend: str = "mujoco",
                      samples: int = 16, threads: int = 4) -> BeamSimulation:
    if backend == "mujoco":
        return BeamSimulation(model_root, render=True)
    if backend in ("cpu-pathtrace", "gpu-pathtrace"):
        return PathtraceSimulation(model_root, render=True, samples=samples, threads=threads,
                                   backend=backend)
    raise ValueError(f"Unknown RL RGB backend: {backend}")
