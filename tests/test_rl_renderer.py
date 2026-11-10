"""Bounded local integration checks; run with the optional CPU rendering runtime."""

import os
import random
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import mujoco
import numpy as np

from so101.rl_renderer import PathtraceSimulation, compile_visual_model, create_simulation
from so101.robot_sim import restore_episode


class RendererSelectionTest(unittest.TestCase):
    def test_external_visual_mesh_preserves_compiled_geometry(self):
        xml = """<mujoco><asset><mesh name="part" scale="2 3 4"
          vertex="0 0 0 1 0 0 0 1 0 0 0 1"
          face="0 2 1 0 1 3 0 3 2 1 2 3"/></asset>
          <worldbody><geom name="part_geom" type="mesh" mesh="part"/></worldbody></mujoco>"""
        expected = mujoco.MjModel.from_xml_string(xml)
        actual = compile_visual_model(xml)
        self.assertEqual(actual.ngeom, expected.ngeom)
        self.assertEqual(actual.geom(0).name, expected.geom(0).name)
        np.testing.assert_array_equal(actual.mesh_face, expected.mesh_face)
        for field in ("mesh_vert", "mesh_pos", "mesh_quat", "geom_pos", "geom_quat"):
            np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), atol=1e-7)

    def test_close_activates_its_own_gl_context_before_freeing_resources(self):
        sim = PathtraceSimulation.__new__(PathtraceSimulation)
        sim.renderer = Mock()
        sim.close()
        self.assertEqual(sim.renderer.mock_calls, [call._gl_context.make_current(), call.close()])
        sim.renderer._gl_context = None
        sim.renderer.reset_mock()
        sim.close()
        sim.renderer.close.assert_called_once_with()

    def test_default_and_invalid_backend(self):
        with patch("so101.rl_renderer.BeamSimulation") as simulation:
            create_simulation(Path("model"))
            simulation.assert_called_once_with(Path("model"), render=True)
        with self.assertRaisesRegex(ValueError, "Unknown RL RGB backend"):
            create_simulation(Path("model"), backend="filter")

    def test_invalid_budget_before_model_creation(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            PathtraceSimulation(samples=0)

    def test_gpu_selection_has_no_cpu_fallback(self):
        with patch("so101.rl_renderer.PathtraceSimulation") as simulation:
            simulation.side_effect = RuntimeError("CUDA unavailable")
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                create_simulation(Path("model"), backend="gpu-pathtrace", samples=8)
            simulation.assert_called_once_with(
                Path("model"), render=True, samples=8, threads=4, backend="gpu-pathtrace",
            )

    @unittest.skipUnless(os.environ.get("SO101_TEST_PATHTRACE") == "1",
                         "requires optional Mitsuba/LLVM runtime and pinned robot assets")
    def test_real_observation_preserves_physics_rng_and_scene_switch(self):
        sim = create_simulation(Path("data/simulation/model"), backend="cpu-pathtrace",
                                samples=1, threads=1)
        try:
            sim.reset(17, varied_layout=True, color_targets=True)
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            before = np.empty(mujoco.mj_stateSize(sim.model, spec))
            after = np.empty_like(before)
            mujoco.mj_getState(sim.model, sim.data, before, spec)
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            image = sim.image()
            np.testing.assert_array_equal(sim.model.cam_fovy, sim.rgb_model.cam_fovy)
            np.testing.assert_allclose(sim.data.cam_xpos, sim.rgb_data.cam_xpos, atol=1e-12)
            np.testing.assert_allclose(sim.data.cam_xmat, sim.rgb_data.cam_xmat, atol=1e-12)
            mujoco.mj_getState(sim.model, sim.data, after, spec)
            np.testing.assert_array_equal(before, after)
            self.assertEqual(python_rng, random.getstate())
            actual_rng = np.random.get_state()
            self.assertEqual(numpy_rng[0], actual_rng[0])
            np.testing.assert_array_equal(numpy_rng[1], actual_rng[1])
            self.assertEqual(numpy_rng[2:], actual_rng[2:])
            self.assertEqual(image.shape, (240, 320, 3))
            self.assertEqual(image.dtype, np.uint8)
            np.testing.assert_array_equal(image, sim.image())
            self.assertIsNot(sim.model, sim.rgb_model)
            self.assertIsNot(sim.data, sim.rgb_data)
            with self.assertRaisesRegex(ValueError, "renderer identity mismatch"):
                restore_episode(sim, {"initial": {"renderer": {"unknown": "profile"}}})
            sim.use_scene_version(1)
            self.assertEqual(sim.pathtrace_samples, 1)
            self.assertEqual(sim.pathtrace_threads, 1)
            sim.reset(17)
            self.assertEqual(sim.image().shape, image.shape)
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()
