"""Wet-mode classification + branching baselines for VISION §10 conditions.

Dry vs wet vs full-rain has fundamentally different physics: grip
collapses, braking distances stretch, aero balance shifts (low-speed
mechanical grip becomes more important than high-speed aero). The
optimiser needs different baselines + priorities in those regimes.

This module provides:
- `classify_conditions(env)` — classify the EnvironmentFrame into one of
  {"dry", "damp", "wet", "full_rain"}.
- `wet_baselines(car, regime)` — returns wet-adjusted CarBaselines.
- `wet_phase_weights(regime)` — returns wet-adjusted phase weight table.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Literal

from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner import Phase
from racingoptimizer.physics.baselines import CarBaselines, baselines_for
from racingoptimizer.physics.phase_weights import PHASE_WEIGHTS

WetRegime = Literal["dry", "damp", "wet", "full_rain"]

# Wetness thresholds (TrackWetness is a 0..1 IBT channel).
_FULL_RAIN_WETNESS = 0.7
_WET_WETNESS = 0.3
_DAMP_WETNESS = 0.05

# Precipitation type >= 2 = active rain (per IBT Precipitation enum).
_PRECIP_RAIN_THRESHOLD = 2

# Per-regime grip scale applied to lateral G + aero baseline. Wheelspin
# tolerance scales inversely (wet => more spin tolerated by drivers).
_REGIME_SCALE: dict[str, float] = {
    "damp": 0.92,
    "wet": 0.75,
    "full_rain": 0.55,
}

# Phase-weight shift away from aero_eff and toward platform + grip.
_AERO_SHIFT: dict[str, float] = {
    "wet": 0.15,
    "full_rain": 0.25,
}


def classify_conditions(env: EnvironmentFrame) -> WetRegime:
    """Classify wet/dry regime from EnvironmentFrame.

    Thresholds:
    - dry: track_wetness < 0.05 AND not weather_declared_wet
    - damp: track_wetness in [0.05, 0.3) OR weather_declared_wet AND wetness < 0.3
    - wet: track_wetness in [0.3, 0.7)
    - full_rain: track_wetness >= 0.7 OR precip_type >= 2
    """
    if env.track_wetness >= _FULL_RAIN_WETNESS or env.precip_type >= _PRECIP_RAIN_THRESHOLD:
        return "full_rain"
    if env.track_wetness >= _WET_WETNESS:
        return "wet"
    if env.track_wetness >= _DAMP_WETNESS or env.weather_declared_wet:
        return "damp"
    return "dry"


def wet_baselines(car: str, regime: WetRegime) -> CarBaselines:
    """Adjust per-car baselines for wet regime.

    Wet → lower max lateral G, lower aero baseline (downforce less
    effective on wet tyres), higher tolerance for wheelspin (managed
    throttle is the norm in wet).
    """
    base = baselines_for(car)
    if regime == "dry":
        return base
    scale = _REGIME_SCALE[regime]
    return replace(
        base,
        max_lateral_g=base.max_lateral_g * scale,
        aero_grip_baseline_g=base.aero_grip_baseline_g * scale,
        # More wheelspin tolerance: scale of 0.55 (full_rain) -> factor of 1.45.
        wheelspin_scale_ms=base.wheelspin_scale_ms * (2.0 - scale),
    )


def wet_phase_weights(regime: WetRegime) -> dict[Phase, dict[str, float]]:
    """Adjust phase-weight table for wet regime.

    Dry: aero efficiency dominates straights.
    Wet: mechanical grip + platform stability dominate; aero matters less.
    Damp: no aero shift (mostly-dry behaviour); the table is cloned defensively.
    """
    if regime == "dry":
        return PHASE_WEIGHTS
    shift = _AERO_SHIFT.get(regime, 0.0)
    adjusted: dict[Phase, dict[str, float]] = {}
    for phase, weights in PHASE_WEIGHTS.items():
        new_weights = dict(weights)
        if shift > 0.0:
            new_weights["aero_eff"] = max(0.0, new_weights["aero_eff"] - shift)
            new_weights["platform"] = new_weights["platform"] + shift / 2
            new_weights["grip"] = new_weights["grip"] + shift / 2
        adjusted[phase] = new_weights
    return adjusted


__all__ = [
    "WetRegime",
    "classify_conditions",
    "wet_baselines",
    "wet_phase_weights",
]
