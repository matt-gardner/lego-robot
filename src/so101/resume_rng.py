"""Backend-aware device RNG recovery, separate from model/optimizer restoration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class BackendRNG(Protocol):
    def manual_seed(self, seed: int) -> Any: ...
    def set_rng_state(self, state: Any) -> None: ...


class TorchRNG(Protocol):
    cuda: BackendRNG
    mps: BackendRNG


@dataclass(frozen=True)
class DeviceRNGPlan:
    source: str
    target: str
    seed: int | None

    @property
    def migration(self) -> bool:
        return self.source != self.target


def plan_device_rng(state: dict, target: str, allow_migration: bool = False,
                    seed: int | None = None) -> DeviceRNGPlan:
    rng = state.get("rng")
    if not isinstance(rng, dict) or "device" not in rng:
        raise ValueError("Full-state recovery requires saved device RNG metadata")
    source = rng.get("backend", state.get("log", {}).get("run_config", {}).get("device"))
    if source not in ("cpu", "mps", "cuda") or target not in ("cpu", "mps", "cuda"):
        raise ValueError("Full-state recovery requires a known source and target RNG backend")
    if source != target:
        if not allow_migration or type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError(
                f"Device RNG backend {source!r} cannot restore on {target}; "
                "explicitly allow migration and supply a device RNG seed"
            )
    elif target != "cpu" and rng["device"] is None:
        raise ValueError("Same-backend accelerator recovery requires saved device RNG state")
    return DeviceRNGPlan(source, target, seed)


def restore_device_rng(plan: DeviceRNGPlan, state: dict, torch: TorchRNG) -> None:
    if plan.migration:
        if plan.seed is None:
            raise ValueError("Device RNG migration requires a seed")
        if plan.target == "cuda":
            torch.cuda.manual_seed(plan.seed)
        elif plan.target == "mps":
            torch.mps.manual_seed(plan.seed)
        state["log"].setdefault("device_rng_migrations", []).append({
            "after_step": state["step"], "source": plan.source, "target": plan.target,
            "seed": plan.seed, "bit_exact": False,
        })
    elif plan.target == "cuda":
        torch.cuda.set_rng_state(state["rng"]["device"].cpu())
    elif plan.target == "mps":
        torch.mps.set_rng_state(state["rng"]["device"].cpu())
