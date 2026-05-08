"""Per-corner-phase physics evaluator (PLAN.md Day 12, Mode 5 capstone).

Assembles the Days 8-11 modules into a single SCORING function:
given (setup, corner_state, env, car), produce a physics-based
score per (corner, phase). Higher score = closer to physics-
predicted optimum for that corner-phase.

This is what Day 13's hybrid optimizer minimizes (instead of the
existing surrogate-only score). Per Reviewer Agent 2's recommendation
the evaluator is intentionally per-corner-phase (NOT lap-time-
integrated) -- the existing `physics/score.py` already aggregates
per-corner-phase scores into a setup-level objective; the evaluator
slots into the same place but uses physics inputs (axle-grip-margin,
aero-derived peak lat-G) instead of surrogate predictions.

Score components (each in [0, 1]; higher = better):

1. **axle_utilization** -- per-axle grip-margin from Day 10. The
   target is "user is operating near the per-axle ceiling" --
   margin in [0.85, 1.0] is ideal; below 0.85 = underutilising;
   above 1.05 = exceeding ceiling (risk of corpus-anomaly).

2. **aero_balance** -- aero-map's balance_pct vs the car's
   weight-distribution-implied target balance. Ideal is balance
   close to (1 - weight_distribution_front) * 100 (i.e. front-
   loaded balance for front-light cars). Penalises extreme
   over- or under-balance.

3. **predictive_grip_headroom** -- ratio of predicted peak lat-G
   (with aero correction if available) to the surrogate's
   recommended lat-G ceiling. The score rewards setups where the
   surrogate's prediction is consistent with the physics-derived
   ceiling. (Day 11 fallback path means correction_factor=0; this
   degrades to a pure aero-map ratio which is still consistent.)

The composite score is a weighted sum:
    score = 0.5 * axle_utilization + 0.3 * aero_balance + 0.2 * predictive_grip_headroom

Weights were chosen so axle_utilization dominates (it's the most
physics-grounded component, with Day 10's per-corpus-empirical
calibration); the other two are smaller but non-negligible
contributions that surface different aspects of "good setup."

Day 12 acceptance gate (PLAN.md Section 15.4): the evaluator's
per-corner-phase score correlates (Spearman) with empirical
observed lap-time-per-corner-phase by >=0.35 across the v4 corpus
on held-out laps. Authorized fallback: ship at >=0.20 Spearman
with `fallback_mode_used: true`.

This module is the integration point. Everything before this was
infrastructure; everything after (Day 13 hybrid optimizer, Day 14
final validation) consumes the evaluator output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from racingoptimizer.aero.residual_correction import (
    AeroResidualCorrection,
    apply_correction,
    predict_peak_lat_g,
)
from racingoptimizer.physics.axle_grip import (
    AxleGripCeiling,
    axle_grip_margin,
    compute_axle_grip_ratios,
)
from racingoptimizer.physics.diagnostic_state import (
    get_car_geometry,
)

# Score-component weights (must sum to 1.0).
_WEIGHT_AXLE_UTIL: float = 0.5
_WEIGHT_AERO_BALANCE: float = 0.3
_WEIGHT_GRIP_HEADROOM: float = 0.2

# Axle-utilization "ideal" band. Margins inside [0.85, 1.0] get full
# score; below 0.85 the score declines linearly (underutilising);
# above 1.05 the score also declines (exceeding ceiling -- corpus
# anomaly territory).
_UTIL_IDEAL_LOW: float = 0.85
_UTIL_IDEAL_HIGH: float = 1.0
_UTIL_OVER_TOLERANCE: float = 0.05  # margin allowed above 1.0 before penalty


@dataclass(frozen=True, slots=True)
class CornerPhaseScore:
    """Per-(corner, phase) physics evaluator output.

    `score` is the composite (axle_util + aero_balance + grip_headroom)
    weighted score in [0, 1]. The components are exposed individually
    so the renderer can explain WHY the score is what it is.
    """
    car: str
    corner_id: int
    phase: str
    axle_utilization: float       # composite of front + rear margins
    aero_balance_score: float     # how close to ideal balance for car
    grip_headroom_score: float    # physics-vs-surrogate consistency
    composite_score: float        # weighted sum


def axle_utilization_score(
    front_margin: float,
    rear_margin: float,
) -> float:
    """Score per-axle margins; higher = closer to ideal utilization.

    Ideal margin band is [0.85, 1.0]. Below 0.85 -> linear penalty
    (underutilising). Above 1.0 -> tolerance for transient overrun
    of 0.05; beyond -> linear penalty (corpus anomaly territory).

    The pair score is the MIN of front and rear individual scores --
    one underutilising axle implies the user is leaving grip on the
    table somewhere; one over-stressed axle is worse.
    """
    def _score(margin: float) -> float:
        if margin <= 0:
            return 0.0
        if _UTIL_IDEAL_LOW <= margin <= _UTIL_IDEAL_HIGH:
            return 1.0
        if margin < _UTIL_IDEAL_LOW:
            # Linear from 0 at margin=0 to 1 at margin=_UTIL_IDEAL_LOW.
            return float(margin / _UTIL_IDEAL_LOW)
        # margin > _UTIL_IDEAL_HIGH: tolerance for small overrun.
        if margin <= _UTIL_IDEAL_HIGH + _UTIL_OVER_TOLERANCE:
            return 1.0
        # Beyond tolerance: linear penalty back to 0 at margin=2.0.
        return max(
            0.0,
            1.0 - (margin - _UTIL_IDEAL_HIGH - _UTIL_OVER_TOLERANCE) / 1.0,
        )

    return min(_score(front_margin), _score(rear_margin))


def aero_balance_score(
    balance_pct: float,
    weight_distribution_front: float,
) -> float:
    """Score aero balance against the car's weight-distribution target.

    Ideal balance for a car with N% front weight is approximately
    100 - N% (i.e. aero-balance offsets static weight distribution
    so combined load split is roughly 50/50 at speed). Penalises
    deviation linearly; full score at the target.
    """
    target_balance_pct = 100.0 * (1.0 - weight_distribution_front)
    deviation = abs(balance_pct - target_balance_pct)
    # Within 5% of target -> full score; linear penalty up to 0 at
    # 25% deviation.
    if deviation <= 5.0:
        return 1.0
    if deviation >= 25.0:
        return 0.0
    return 1.0 - (deviation - 5.0) / 20.0


def grip_headroom_score(
    predicted_peak_lat_g: float,
    surrogate_lat_g_ceiling: float,
) -> float:
    """Score consistency between physics-predicted peak and surrogate's ceiling.

    If physics says peak = 1.5 G and surrogate says ceiling = 1.5 G,
    they're consistent (score 1.0). If they diverge by more than
    20%, score declines linearly. The score rewards setups where
    the two predictors agree -- those are the setups we trust.
    """
    if surrogate_lat_g_ceiling <= 0:
        return 0.0
    ratio = predicted_peak_lat_g / surrogate_lat_g_ceiling
    deviation = abs(ratio - 1.0)
    if deviation <= 0.05:
        return 1.0
    if deviation >= 0.30:
        return 0.0
    return 1.0 - (deviation - 0.05) / 0.25


def evaluate_corner_phase(
    car: str,
    corner_id: int,
    phase: str,
    *,
    lat_g: float,
    long_g: float,
    speed_ms: float,
    aero_balance_pct: float,
    aero_ld_ratio: float,
    front_ceiling: AxleGripCeiling,
    rear_ceiling: AxleGripCeiling,
    surrogate_lat_g_ceiling: float | None = None,
    aero_correction: AeroResidualCorrection | None = None,
) -> CornerPhaseScore:
    """Compute the per-corner-phase composite physics score.

    Args:
        car, corner_id, phase: identifiers
        lat_g, long_g: chassis G channels for this corner-phase
        speed_ms: speed at this corner-phase
        aero_balance_pct, aero_ld_ratio: from the aero map at the
            sample's (front_rh, rear_rh, wing)
        front_ceiling, rear_ceiling: per-axle grip ceilings (Day 10)
        surrogate_lat_g_ceiling: optional predicted ceiling from
            the existing surrogate; if absent, falls back to physics-
            only headroom score (1.0).
        aero_correction: optional Day 11 correction; if absent,
            uses raw predict_peak_lat_g.

    Returns CornerPhaseScore with components + composite.
    """
    # Component 1: axle utilization.
    geom = get_car_geometry(car)
    ratios = compute_axle_grip_ratios(
        np.array([lat_g]), np.array([long_g]), car,
    )
    front_margin = float(axle_grip_margin(ratios["front"][0], front_ceiling))
    rear_margin = float(axle_grip_margin(ratios["rear"][0], rear_ceiling))
    util_score = axle_utilization_score(front_margin, rear_margin)

    # Component 2: aero balance.
    balance_score = aero_balance_score(
        aero_balance_pct, geom.weight_distribution,
    )

    # Component 3: grip headroom.
    raw_peak = predict_peak_lat_g(aero_ld_ratio, speed_ms)
    if aero_correction is not None:
        peak = apply_correction(raw_peak, aero_correction)
    else:
        peak = raw_peak
    if surrogate_lat_g_ceiling is None:
        # Without surrogate, default to neutral score (1.0) -- physics
        # alone can't compare against itself. Reduces this component's
        # weight in practice.
        headroom_score = 1.0
    else:
        headroom_score = grip_headroom_score(peak, surrogate_lat_g_ceiling)

    composite = (
        _WEIGHT_AXLE_UTIL * util_score
        + _WEIGHT_AERO_BALANCE * balance_score
        + _WEIGHT_GRIP_HEADROOM * headroom_score
    )

    return CornerPhaseScore(
        car=car.strip().lower(),
        corner_id=int(corner_id),
        phase=str(phase),
        axle_utilization=float(util_score),
        aero_balance_score=float(balance_score),
        grip_headroom_score=float(headroom_score),
        composite_score=float(composite),
    )


def evaluate_lap(
    car: str,
    samples: list[dict],
    front_ceiling: AxleGripCeiling,
    rear_ceiling: AxleGripCeiling,
    *,
    aero_correction: AeroResidualCorrection | None = None,
) -> list[CornerPhaseScore]:
    """Evaluate every corner-phase sample in a lap.

    `samples` is a list of dicts with keys: corner_id, phase, lat_g,
    long_g, speed_ms, aero_balance_pct, aero_ld_ratio,
    surrogate_lat_g_ceiling (optional).
    """
    out: list[CornerPhaseScore] = []
    for s in samples:
        try:
            out.append(evaluate_corner_phase(
                car=car,
                corner_id=int(s["corner_id"]),
                phase=str(s["phase"]),
                lat_g=float(s["lat_g"]),
                long_g=float(s["long_g"]),
                speed_ms=float(s["speed_ms"]),
                aero_balance_pct=float(s["aero_balance_pct"]),
                aero_ld_ratio=float(s["aero_ld_ratio"]),
                front_ceiling=front_ceiling,
                rear_ceiling=rear_ceiling,
                surrogate_lat_g_ceiling=s.get("surrogate_lat_g_ceiling"),
                aero_correction=aero_correction,
            ))
        except (KeyError, ValueError, TypeError):
            continue
    return out
