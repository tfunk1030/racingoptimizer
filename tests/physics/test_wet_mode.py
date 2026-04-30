"""Wet-mode classification + branching baselines (S4.7)."""
from __future__ import annotations

from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner import Phase
from racingoptimizer.physics.baselines import baselines_for
from racingoptimizer.physics.phase_weights import PHASE_WEIGHTS
from racingoptimizer.physics.wet_mode import (
    classify_conditions,
    wet_baselines,
    wet_phase_weights,
)


def _env(track_wetness: float, *, declared_wet: bool = False, precip: int = -1) -> EnvironmentFrame:
    return EnvironmentFrame(
        track_wetness=track_wetness,
        weather_declared_wet=declared_wet,
        precip_type=precip,
    )


def test_classify_conditions_dry() -> None:
    assert classify_conditions(_env(0.0)) == "dry"
    assert classify_conditions(_env(0.04)) == "dry"


def test_classify_conditions_damp() -> None:
    # Wetness in damp band.
    assert classify_conditions(_env(0.05)) == "damp"
    assert classify_conditions(_env(0.2)) == "damp"
    # Or weather declared wet but track not yet wet.
    assert classify_conditions(_env(0.01, declared_wet=True)) == "damp"


def test_classify_conditions_wet() -> None:
    assert classify_conditions(_env(0.3)) == "wet"
    assert classify_conditions(_env(0.5)) == "wet"
    assert classify_conditions(_env(0.69)) == "wet"


def test_classify_conditions_full_rain() -> None:
    # Wetness above the full-rain threshold.
    assert classify_conditions(_env(0.7)) == "full_rain"
    assert classify_conditions(_env(0.95)) == "full_rain"
    # Or active precipitation regardless of wetness.
    assert classify_conditions(_env(0.0, precip=2)) == "full_rain"


def test_wet_baselines_dry_returns_unchanged() -> None:
    base = baselines_for("bmw")
    wet = wet_baselines("bmw", "dry")
    assert wet == base


def test_wet_baselines_wet_lowers_lateral_g_and_aero() -> None:
    base = baselines_for("bmw")
    wet = wet_baselines("bmw", "wet")
    assert wet.max_lateral_g < base.max_lateral_g
    assert wet.aero_grip_baseline_g < base.aero_grip_baseline_g
    # Wheelspin tolerance increases (driver manages throttle in wet).
    assert wet.wheelspin_scale_ms > base.wheelspin_scale_ms


def test_wet_phase_weights_dry_is_identity() -> None:
    assert wet_phase_weights("dry") == PHASE_WEIGHTS


def test_wet_phase_weights_wet_lowers_braking_aero_eff() -> None:
    dry = PHASE_WEIGHTS[Phase.BRAKING]
    wet = wet_phase_weights("wet")[Phase.BRAKING]
    assert wet["aero_eff"] < dry["aero_eff"] or (
        wet["aero_eff"] == 0.0 and dry["aero_eff"] == 0.0
    )
    # The straight phase has the most aero_eff weight; verify the shift there too.
    dry_straight = PHASE_WEIGHTS[Phase.STRAIGHT]
    wet_straight = wet_phase_weights("wet")[Phase.STRAIGHT]
    assert wet_straight["aero_eff"] < dry_straight["aero_eff"]
