"""Day 13 of physics-rebuild: hybrid optimizer tests.

Per PLAN.md §15.5 + Day 13 investigation finding: phase-aware
physics weighting (mid_corner gets 0.40, other phases 0.05-0.10) +
guardrail penalty for setups exceeding empirical ceiling.
"""
from __future__ import annotations

import pytest

from racingoptimizer.physics.hybrid_optimizer import (
    _DEFAULT_PHASE_WEIGHT,
    _GUARDRAIL_PENALTY_OVER_CEILING,
    _PHASE_PHYSICS_WEIGHTS,
    get_phase_physics_weight,
    hybrid_score,
    hybrid_score_lap,
)

# ---- Phase-aware weighting ---------------------------------------------


def test_mid_corner_gets_highest_weight() -> None:
    """Per investigation, mid_corner has 0.40 (largest physics weight)."""
    assert get_phase_physics_weight("mid_corner") == 0.40


def test_per_phase_weights_documented() -> None:
    """Pin the empirically-derived per-phase weights."""
    assert _PHASE_PHYSICS_WEIGHTS["mid_corner"] == 0.40
    assert _PHASE_PHYSICS_WEIGHTS["braking"] == 0.10
    assert _PHASE_PHYSICS_WEIGHTS["exit"] == 0.10
    assert _PHASE_PHYSICS_WEIGHTS["trail_brake"] == 0.05
    assert _PHASE_PHYSICS_WEIGHTS["straight"] == 0.05


def test_unknown_phase_gets_default_weight() -> None:
    assert get_phase_physics_weight("totally_made_up_phase") == _DEFAULT_PHASE_WEIGHT


def test_phase_weight_normalized_case() -> None:
    """Phase comparison is case-insensitive."""
    assert get_phase_physics_weight("MID_CORNER") == 0.40
    assert get_phase_physics_weight("Mid_Corner") == 0.40


# ---- hybrid_score core --------------------------------------------------


def test_hybrid_score_mid_corner_physics_heavy() -> None:
    """Mid_corner: hybrid = 0.4 * physics + 0.6 * surrogate."""
    score = hybrid_score(
        car="bmw", corner_id=5, phase="mid_corner",
        physics_score=0.8, surrogate_score=0.5,
    )
    expected_raw = 0.4 * 0.8 + 0.6 * 0.5  # 0.62
    assert score.raw_hybrid_score == pytest.approx(expected_raw)
    assert score.hybrid_score == pytest.approx(expected_raw)  # no penalty
    assert score.physics_weight == 0.4
    assert score.guardrail_penalty == 0.0


def test_hybrid_score_braking_surrogate_heavy() -> None:
    """Braking: hybrid = 0.1 * physics + 0.9 * surrogate."""
    score = hybrid_score(
        car="bmw", corner_id=5, phase="braking",
        physics_score=0.8, surrogate_score=0.5,
    )
    expected = 0.1 * 0.8 + 0.9 * 0.5  # 0.53
    assert score.hybrid_score == pytest.approx(expected)
    assert score.physics_weight == 0.1


def test_hybrid_score_override_weight() -> None:
    """`physics_weight_override` bypasses phase default."""
    score = hybrid_score(
        car="bmw", corner_id=5, phase="mid_corner",
        physics_score=0.8, surrogate_score=0.5,
        physics_weight_override=0.0,  # surrogate-only
    )
    assert score.hybrid_score == pytest.approx(0.5)
    assert score.physics_weight == 0.0


def test_hybrid_score_override_invalid_raises() -> None:
    with pytest.raises(ValueError, match="outside"):
        hybrid_score(
            car="bmw", corner_id=5, phase="mid_corner",
            physics_score=0.8, surrogate_score=0.5,
            physics_weight_override=1.5,
        )


# ---- Guardrail penalty -------------------------------------------------


def test_hybrid_score_axle_ceiling_penalty() -> None:
    """over_axle_ceiling=True applies the documented penalty."""
    score = hybrid_score(
        car="bmw", corner_id=5, phase="mid_corner",
        physics_score=0.8, surrogate_score=0.5,
        over_axle_ceiling=True,
    )
    expected_raw = 0.4 * 0.8 + 0.6 * 0.5
    assert score.raw_hybrid_score == pytest.approx(expected_raw)
    assert score.guardrail_penalty == _GUARDRAIL_PENALTY_OVER_CEILING
    assert score.hybrid_score == pytest.approx(
        max(0.0, expected_raw - _GUARDRAIL_PENALTY_OVER_CEILING),
    )


def test_hybrid_score_balance_penalty_smaller() -> None:
    """severely_off_balance penalty is half the axle-ceiling penalty."""
    score = hybrid_score(
        car="bmw", corner_id=5, phase="mid_corner",
        physics_score=0.8, surrogate_score=0.5,
        severely_off_balance=True,
    )
    assert score.guardrail_penalty == _GUARDRAIL_PENALTY_OVER_CEILING / 2.0


def test_hybrid_score_grip_inconsistency_penalty_quarter() -> None:
    """grip_inconsistency penalty is one quarter of the axle-ceiling penalty."""
    score = hybrid_score(
        car="bmw", corner_id=5, phase="mid_corner",
        physics_score=0.8, surrogate_score=0.5,
        grip_inconsistency=True,
    )
    assert score.guardrail_penalty == _GUARDRAIL_PENALTY_OVER_CEILING / 4.0


def test_hybrid_score_both_penalties_stack() -> None:
    """Both guardrails firing -> stacked penalty."""
    score = hybrid_score(
        car="bmw", corner_id=5, phase="mid_corner",
        physics_score=0.8, surrogate_score=0.5,
        over_axle_ceiling=True,
        severely_off_balance=True,
    )
    expected_penalty = (
        _GUARDRAIL_PENALTY_OVER_CEILING
        + _GUARDRAIL_PENALTY_OVER_CEILING / 2.0
    )
    assert score.guardrail_penalty == pytest.approx(expected_penalty)


def test_hybrid_score_floors_at_zero() -> None:
    """If penalty exceeds raw score, hybrid_score floors at 0."""
    score = hybrid_score(
        car="bmw", corner_id=5, phase="mid_corner",
        physics_score=0.0, surrogate_score=0.05,  # tiny raw
        over_axle_ceiling=True,
        severely_off_balance=True,
    )
    assert score.hybrid_score == 0.0


# ---- hybrid_score_lap (bulk wrapper) -----------------------------------


def test_hybrid_score_lap_skips_invalid() -> None:
    samples = [
        {  # valid
            "car": "bmw", "corner_id": 1, "phase": "mid_corner",
            "physics_score": 0.7, "surrogate_score": 0.5,
        },
        {"corner_id": 2},  # invalid
        {  # valid
            "car": "bmw", "corner_id": 3, "phase": "exit",
            "physics_score": 0.6, "surrogate_score": 0.4,
            "over_axle_ceiling": True,
        },
    ]
    out = hybrid_score_lap(samples)
    assert len(out) == 2
    assert out[0].corner_id == 1
    assert out[1].corner_id == 3
    assert out[1].guardrail_penalty > 0


# ---- Canary -------------------------------------------------------------


def test_canary_zero_weight_collapses_to_surrogate() -> None:
    """Broken-model canary: if all phase weights collapse to 0,
    hybrid_score = surrogate_score regardless of physics. The hybrid
    optimizer reduces to the existing surrogate path -- which means
    the hybrid added nothing. If a future commit accidentally zeros
    all weights, this canary detects it.
    """
    score = hybrid_score(
        car="bmw", corner_id=5, phase="mid_corner",
        physics_score=0.99, surrogate_score=0.30,
        physics_weight_override=0.0,
    )
    assert score.hybrid_score == pytest.approx(0.30)
    # Inverse: with weight 1.0, hybrid = physics.
    score2 = hybrid_score(
        car="bmw", corner_id=5, phase="mid_corner",
        physics_score=0.99, surrogate_score=0.30,
        physics_weight_override=1.0,
    )
    assert score2.hybrid_score == pytest.approx(0.99)


def test_per_phase_distinct_weights_real() -> None:
    """The per-phase weight pattern (mid_corner > others) reflects
    investigation finding. If a future commit makes all phases
    equally-weighted, this canary detects the loss of phase-aware
    behaviour.
    """
    weights = [_PHASE_PHYSICS_WEIGHTS[ph] for ph in
               ("mid_corner", "braking", "exit", "trail_brake", "straight")]
    spread = max(weights) - min(weights)
    assert spread > 0.30, (
        f"per-phase weight spread {spread:.2f} too small; phase-aware "
        f"design is being eroded"
    )


def test_constants_documented() -> None:
    """Pin the design constants."""
    assert _DEFAULT_PHASE_WEIGHT == 0.10
    assert _GUARDRAIL_PENALTY_OVER_CEILING == 0.15
