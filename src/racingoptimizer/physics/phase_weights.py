"""Phase -> sub-utilization weights (spec §6, frozen table).

Six sub-utilizations from spec §6: grip, balance, stability, traction,
aero_eff, platform. Each phase weights them differently per the spec's
shape: braking weights grip+stability+platform; mid-corner weights
grip+balance; exit weights traction+balance+aero_eff; straight weights
aero_eff+platform; trail-brake is a hand-off blend.

Each row sums to 1.0; the unit test in tests/physics/test_phase_weights.py
asserts the invariant. Numbers are pinned per the spec — do not tune.
"""
from __future__ import annotations

from racingoptimizer.corner import Phase

SUB_UTILIZATIONS: tuple[str, ...] = (
    "grip", "balance", "stability", "traction", "aero_eff", "platform",
)


PHASE_WEIGHTS: dict[Phase, dict[str, float]] = {
    Phase.BRAKING: {
        "grip": 0.35, "stability": 0.30, "platform": 0.25,
        "balance": 0.10, "traction": 0.0, "aero_eff": 0.0,
    },
    Phase.TRAIL_BRAKE: {
        "grip": 0.30, "balance": 0.25, "stability": 0.20,
        "platform": 0.15, "traction": 0.05, "aero_eff": 0.05,
    },
    Phase.MID_CORNER: {
        "grip": 0.40, "balance": 0.35, "platform": 0.10,
        "stability": 0.10, "traction": 0.05, "aero_eff": 0.0,
    },
    Phase.EXIT: {
        "traction": 0.35, "balance": 0.25, "aero_eff": 0.15,
        "grip": 0.15, "platform": 0.05, "stability": 0.05,
    },
    Phase.STRAIGHT: {
        "aero_eff": 0.55, "platform": 0.35, "stability": 0.10,
        "grip": 0.0, "balance": 0.0, "traction": 0.0,
    },
}


__all__ = ["PHASE_WEIGHTS", "SUB_UTILIZATIONS"]
