import unittest
from unittest.mock import Mock

from so101.resume_rng import plan_device_rng, restore_device_rng


class DeviceRNGTests(unittest.TestCase):
    def state(self, source="mps", legacy=False):
        state = {
            "step": 304, "rng": {"device": Mock()},
            "log": {"run_config": {"device": source}},
        }
        if not legacy:
            state["rng"]["backend"] = source
        return state

    def test_foreign_rng_requires_opt_in_and_explicit_valid_seed(self):
        for allow, seed in ((False, None), (False, 61000), (True, None),
                            (True, True), (True, -1), (True, 2**64)):
            with self.subTest(allow=allow, seed=seed), self.assertRaises(ValueError):
                plan_device_rng(self.state(), "cuda", allow, seed)

    def test_legacy_backend_comes_from_saved_log_not_target(self):
        state = self.state(legacy=True)
        plan = plan_device_rng(state, "cuda", True, 61000)
        torch = Mock()
        restore_device_rng(plan, state, torch)
        torch.cuda.manual_seed.assert_called_once_with(61000)
        torch.cuda.set_rng_state.assert_not_called()
        state["rng"]["device"].cpu.assert_not_called()
        self.assertEqual(state["log"]["device_rng_migrations"], [{
            "after_step": 304, "source": "mps", "target": "cuda",
            "seed": 61000, "bit_exact": False,
        }])

    def test_same_backend_restores_bytes_never_reseeds(self):
        for backend in ("mps", "cuda"):
            with self.subTest(backend=backend):
                state = self.state(backend)
                plan = plan_device_rng(state, backend, True, 61000)
                torch = Mock()
                restore_device_rng(plan, state, torch)
                api = getattr(torch, backend)
                api.set_rng_state.assert_called_once_with(state["rng"]["device"].cpu.return_value)
                api.manual_seed.assert_not_called()
                self.assertNotIn("device_rng_migrations", state["log"])

    def test_cpu_to_gpu_needs_explicit_seed_even_without_device_bytes(self):
        state = self.state("cpu")
        state["rng"]["device"] = None
        with self.assertRaises(ValueError):
            plan_device_rng(state, "cuda")
        plan = plan_device_rng(state, "cuda", True, 61000)
        torch = Mock()
        restore_device_rng(plan, state, torch)
        torch.cuda.manual_seed.assert_called_once_with(61000)

    def test_missing_backend_or_same_backend_state_is_not_silently_seeded(self):
        for state in (
            {"rng": {"device": None}}, {"rng": {}},
            {"rng": {"backend": "cuda", "device": None}},
        ):
            with self.subTest(state=state), self.assertRaises(ValueError):
                plan_device_rng(state, "cuda", True, 61000)

    def test_cpu_same_backend_does_not_touch_accelerator_apis(self):
        state = self.state("cpu")
        state["rng"]["device"] = None
        torch = Mock()
        restore_device_rng(plan_device_rng(state, "cpu"), state, torch)
        self.assertEqual(torch.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
