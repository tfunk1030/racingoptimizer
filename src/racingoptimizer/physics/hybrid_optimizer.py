"""Hybrid optimizer: physics + surrogate scoring (PLAN.md Day 13).

Day 13 builds on Day 12b's per-car-calibrated weights + Day 13's
investigation finding: PHASE-AWARE physics weighting is the right
design. The investigation (`scripts/day_13_investigate_responsiveness.py`)
showed:

    Phase            Cross-car mean Spearman vs -duration_s
    -----            ---------------------------------------
    mid_corner       +0.233   <-- strong physics signal
    exit             +0.087
    braking          +0.083
    straight         +0.070
    trail_brake      +0.008

Mid-corner has 3-30x stronger physics signal than other phases
because that's where setup directly determines steady-state cornering
behaviour. Other phases are dominated by driver inputs (brake apply,
throttle modulation, trail-brake release timing) that the physics
evaluator cannot see.

Hybrid score:
    hybrid = w_phase * physics_evaluator + (1 - w_phase) * surrogate

with phase-specific weights:
    mid_corner    -> w = 0.4 (physics-heavy where physics has signal)
    braking       -> w = 0.1
    exit          -> w = 0.1
    trail_brake   -> w = 0.05
    straight      -> w = 0.05  (minimal cornering forces)

PLUS guardrails as a hard constraint: any sample with
axle_grip_margin > 1.0 (exceeding empirical ceiling) gets a penalty
on the hybrid score. The penalty is additive (not multiplicative) so
it can be tuned without affecting the surrogate's relative-ranking.

This module is the integration point Day 13 ships. The recommender's
DE search consumes the hybrid score; the briefing renderer can show
guardrail violations as warnings.
"""
from __future__ import annotations

from dataclasses import dataclass

# Phase-aware physics weights. Empirically derived from
# `scripts/day_13_investigate_responsiveness.py` -- mid_corner is
# where physics has the strongest signal; other phases get small
# physics weight + mostly-surrogate.
_PHASE_PHYSICS_WEIGHTS: dict[str, float] = {
    "mid_corner": 0.40,
    "braking": 0.10,
    "exit": 0.10,
    "trail_brake": 0.05,
    "straight": 0.05,
}
_DEFAULT_PHASE_WEIGHT: float = 0.10  # for unknown phases


# Guardrail penalty applied to hybrid score when a sample's
# axle_grip_margin exceeds the empirical ceiling. Subtractive
# (penalty units in the same scale as the score, [0, 1]).
_GUARDRAIL_PENALTY_OVER_CEILING: float = 0.15


@dataclass(frozen=True, slots=True)
class HybridScore:
    """Per-(corner, phase) hybrid optimizer score.

    `hybrid_score` is the weighted combination of physics + surrogate,
    minus any guardrail penalties. Components exposed for renderer
    explainability.
    """
    car: str
    corner_id: int
    phase: str
    physics_score: float       # composite from physics evaluator
    surrogate_score: float     # composite from surrogate (caller-provided)
    physics_weight: float      # phase-aware w (0.05-0.40)
    raw_hybrid_score: float    # w * physics + (1-w) * surrogate
    guardrail_penalty: float   # penalty for axle_util > 1.0, etc.
    hybrid_score: float        # raw_hybrid_score - guardrail_penalty


def get_phase_physics_weight(phase: str) -> float:
    """Return the physics weight for the given phase, default 0.10."""
    return _PHASE_PHYSICS_WEIGHTS.get(
        phase.strip().lower(), _DEFAULT_PHASE_WEIGHT,
    )


def hybrid_score(
    *,
    car: str,
    corner_id: int,
    phase: str,
    physics_score: float,
    surrogate_score: float,
    over_axle_ceiling: bool = False,
    severely_off_balance: bool = False,
    grip_inconsistency: bool = False,
    physics_weight_override: float | None = None,
) -> HybridScore:
    """Combine physics and surrogate scores with phase-aware weighting.

    Args:
        car: car identifier (kept for provenance)
        corner_id, phase: identifiers
        physics_score: composite from `physics/evaluator.evaluate_corner_phase`
            in [0, 1]
        surrogate_score: composite from the existing surrogate model
            in [0, 1] (caller's responsibility to normalize)
        over_axle_ceiling: from `guardrail_check`; if True, applies
            penalty
        severely_off_balance: from `guardrail_check`; if True, applies
            smaller penalty
        grip_inconsistency: from `guardrail_check`; if True, applies
            a smaller penalty (physics aero peak vs corpus reference)
        physics_weight_override: optional explicit weight; bypasses
            phase-aware default

    Returns HybridScore with all components exposed.
    """
    if physics_weight_override is not None:
        w = float(physics_weight_override)
    else:
        w = get_phase_physics_weight(phase)
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"physics weight {w} outside [0, 1]")

    raw = w * physics_score + (1.0 - w) * surrogate_score
    penalty = 0.0
    if over_axle_ceiling:
        penalty += _GUARDRAIL_PENALTY_OVER_CEILING
    if severely_off_balance:
        penalty += _GUARDRAIL_PENALTY_OVER_CEILING / 2.0
    if grip_inconsistency:
        penalty += _GUARDRAIL_PENALTY_OVER_CEILING / 4.0
    final = max(0.0, raw - penalty)

    return HybridScore(
        car=car.strip().lower(),
        corner_id=int(corner_id),
        phase=str(phase),
        physics_score=float(physics_score),
        surrogate_score=float(surrogate_score),
        physics_weight=w,
        raw_hybrid_score=float(raw),
        guardrail_penalty=float(penalty),
        hybrid_score=float(final),
    )


def hybrid_score_lap(
    samples: list[dict],
) -> list[HybridScore]:
    """Apply hybrid scoring to a list of per-corner-phase samples.

    Each sample dict needs: car, corner_id, phase, physics_score,
    surrogate_score; optional: over_axle_ceiling, severely_off_balance,
    physics_weight_override.
    """
    out: list[HybridScore] = []
    for s in samples:
        try:
            out.append(hybrid_score(
                car=s["car"],
                corner_id=int(s["corner_id"]),
                phase=str(s["phase"]),
                physics_score=float(s["physics_score"]),
                surrogate_score=float(s["surrogate_score"]),
                over_axle_ceiling=bool(s.get("over_axle_ceiling", False)),
                severely_off_balance=bool(s.get("severely_off_balance", False)),
                grip_inconsistency=bool(s.get("grip_inconsistency", False)),
                physics_weight_override=s.get("physics_weight_override"),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out
