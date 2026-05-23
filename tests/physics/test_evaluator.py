"""Day 12 of physics-rebuild: per-corner-phase physics evaluator.

PLAN.md Section 15.4: capstone module assembling Days 8-11 into a
single physics-based scoring function. Day 13's hybrid optimizer
consumes this; Day 14 validates end-to-end.

Acceptance gate: per-corner-phase score correlates (Spearman) with
empirical lap-time-per-corner-phase by >=0.35 on held-out laps;
fallback at >=0.20.

Broken-model canary: Constant score regardless of setup. Spearman -> 0.
"""
from __future__ import annotations

import pytest

from racingoptimizer.physics.axle_grip import AxleGripCeiling
from racingoptimizer.physics.evaluator import (
    _UTIL_IDEAL_HIGH,
    _UTIL_IDEAL_LOW,
    _WEIGHT_AERO_BALANCE,
    _WEIGHT_AXLE_UTIL,
    _WEIGHT_GRIP_HEADROOM,
    CornerPhaseScore,
    aero_balance_score,
    axle_utilization_score,
    evaluate_corner_phase,
    evaluate_lap,
    grip_headroom_score,
)

# ---- axle_utilization_score --------------------------------------------


def test_util_score_ideal_band_max() -> None:
    """Within [0.85, 1.00], full score."""
    assert axle_utilization_score(0.90, 0.92) == 1.0
    assert axle_utilization_score(_UTIL_IDEAL_LOW, _UTIL_IDEAL_HIGH) == 1.0


def test_util_score_underutilising_linear_penalty() -> None:
    """Below 0.85, linear decline to 0 at margin=0."""
    # margin=0.5 -> score = 0.5 / 0.85 ~ 0.588
    score = axle_utilization_score(0.5, 0.5)
    assert 0.55 < score < 0.62


def test_util_score_min_of_axles() -> None:
    """Pair score = MIN of front + rear individual scores."""
    # Front = ideal (1.0); rear = 0.5 (underutilising).
    # Pair score should match rear's score (~0.59).
    score = axle_utilization_score(0.95, 0.5)
    assert 0.55 < score < 0.62


def test_util_score_overrun_tolerance() -> None:
    """Margin in (1.0, 1.05] -> still full score (transient tolerance)."""
    assert axle_utilization_score(1.03, 1.03) == 1.0


def test_util_score_beyond_tolerance_penalised() -> None:
    """Margin > 1.05 -> declining score."""
    # margin = 1.5 -> score = 1.0 - (1.5 - 1.05) / 1.0 = 0.55
    score = axle_utilization_score(1.5, 1.5)
    assert 0.50 < score < 0.60


def test_util_score_zero_margin() -> None:
    """Zero margin (no lat force) -> score 0."""
    assert axle_utilization_score(0.0, 0.0) == 0.0


# ---- aero_balance_score ------------------------------------------------


def test_balance_score_at_target_full() -> None:
    """For 46% front weight, ideal aero balance ~54% -> full score."""
    assert aero_balance_score(54.0, 0.46) == 1.0
    # Within 5% of target also full score.
    assert aero_balance_score(58.0, 0.46) == 1.0


def test_balance_score_far_from_target_zero() -> None:
    """25%+ deviation -> zero score."""
    assert aero_balance_score(20.0, 0.46) == 0.0
    assert aero_balance_score(85.0, 0.46) == 0.0


def test_balance_score_linear_decline() -> None:
    """Between 5% and 25% deviation, linear decline."""
    # Target=54%; deviation=15% -> score = 1.0 - (15-5)/20 = 0.5
    score = aero_balance_score(69.0, 0.46)
    assert 0.45 < score < 0.55


# ---- grip_headroom_score -----------------------------------------------


def test_headroom_score_consistent_full() -> None:
    """Physics and surrogate agreeing -> score 1.0."""
    assert grip_headroom_score(1.5, 1.5) == 1.0
    # Within 5% -> still full.
    assert grip_headroom_score(1.5, 1.55) == 1.0


def test_headroom_score_divergence_penalised() -> None:
    """30%+ divergence -> zero score."""
    assert grip_headroom_score(1.0, 2.0) == 0.0


def test_headroom_score_zero_ceiling() -> None:
    """Surrogate ceiling = 0 -> zero score (defensive)."""
    assert grip_headroom_score(1.5, 0.0) == 0.0


# ---- evaluate_corner_phase ---------------------------------------------


def _ceiling(car: str, axle: str, mu: float = 1.5) -> AxleGripCeiling:
    return AxleGripCeiling(
        car=car, axle=axle, mu_peak=mu,
        n_samples=1000, n_above_ceiling=10, percentile_used=99.0,
    )


def test_evaluate_corner_phase_returns_score() -> None:
    """Smoke: evaluate produces a valid CornerPhaseScore."""
    score = evaluate_corner_phase(
        car="bmw",
        corner_id=5,
        phase="mid_corner",
        lat_g=1.5,
        long_g=0.0,
        speed_ms=50.0,
        aero_balance_pct=54.0,
        aero_ld_ratio=4.0,
        front_ceiling=_ceiling("bmw", "front"),
        rear_ceiling=_ceiling("bmw", "rear"),
        surrogate_lat_g_ceiling=1.5,
    )
    assert isinstance(score, CornerPhaseScore)
    assert 0.0 <= score.composite_score <= 1.0
    assert 0.0 <= score.axle_utilization <= 1.0
    assert 0.0 <= score.aero_balance_score <= 1.0
    assert 0.0 <= score.grip_headroom_score <= 1.0


def test_evaluate_corner_phase_composite_weighted_sum() -> None:
    """Composite uses per-car calibrated weights (Day 12 follow-up).

    BMW calibrated weights are (0.2, 0.8, 0.0) per the evaluator
    docstring; this test pins that contract.
    """
    from racingoptimizer.physics.evaluator import get_weights_for_car
    score = evaluate_corner_phase(
        car="bmw",
        corner_id=1, phase="mid_corner",
        lat_g=1.5, long_g=0.0, speed_ms=50.0,
        aero_balance_pct=54.0,  # ideal -> balance_score=1.0
        aero_ld_ratio=4.0,
        front_ceiling=_ceiling("bmw", "front"),
        rear_ceiling=_ceiling("bmw", "rear"),
        surrogate_lat_g_ceiling=1.5,  # consistent -> headroom=1.0
    )
    wu, wb, wh = get_weights_for_car("bmw")
    expected = (
        wu * score.axle_utilization
        + wb * score.aero_balance_score
        + wh * score.grip_headroom_score
    )
    assert score.composite_score == pytest.approx(expected)


def test_per_car_calibrated_weights_documented() -> None:
    """Pin the per-car weight values so a future commit doesn't
    silently drift them."""
    from racingoptimizer.physics.evaluator import get_weights_for_car
    assert get_weights_for_car("bmw") == (0.2, 0.8, 0.0)
    assert get_weights_for_car("cadillac") == (0.2, 0.3, 0.5)
    assert get_weights_for_car("ferrari") == (0.0, 0.0, 1.0)
    assert get_weights_for_car("porsche") == (0.0, 0.5, 0.5)
    # Acura falls back to default (0.5, 0.3, 0.2).
    assert get_weights_for_car("acura") == (
        _WEIGHT_AXLE_UTIL, _WEIGHT_AERO_BALANCE, _WEIGHT_GRIP_HEADROOM,
    )


def test_guardrail_check_axle_over_ceiling() -> None:
    """Flag setups where front or rear margin exceeds 1.0."""
    from racingoptimizer.physics.evaluator import guardrail_check
    score = evaluate_corner_phase(
        car="bmw", corner_id=1, phase="mid_corner",
        lat_g=2.5, long_g=0.0, speed_ms=70.0,  # high lat_g
        aero_balance_pct=54.0, aero_ld_ratio=4.0,
        front_ceiling=_ceiling("bmw", "front", mu=0.5),  # very low ceiling
        rear_ceiling=_ceiling("bmw", "rear", mu=0.5),
        surrogate_lat_g_ceiling=1.5,
    )
    # Front margin will be > 1.0 for sure with mu=0.5 and high lat_g.
    report = guardrail_check(score, front_margin=1.5, rear_margin=1.4)
    assert report.over_axle_ceiling is True
    assert report.flagged is True
    assert "exceeds empirical grip ceiling" in (report.reason or "")


def test_guardrail_check_balance_off_target() -> None:
    """Flag setups with aero balance > 17% off target."""
    from racingoptimizer.physics.evaluator import guardrail_check
    score = evaluate_corner_phase(
        car="bmw", corner_id=1, phase="mid_corner",
        lat_g=1.0, long_g=0.0, speed_ms=50.0,
        aero_balance_pct=80.0,  # very off-target (target ~54)
        aero_ld_ratio=4.0,
        front_ceiling=_ceiling("bmw", "front"),
        rear_ceiling=_ceiling("bmw", "rear"),
        surrogate_lat_g_ceiling=1.5,
    )
    report = guardrail_check(score, front_margin=0.7, rear_margin=0.7)
    assert report.severely_off_balance is True
    assert report.flagged is True


def test_guardrail_check_clean_setup_no_flags() -> None:
    """A well-behaved setup has no guardrail flags."""
    from racingoptimizer.physics.evaluator import guardrail_check
    score = evaluate_corner_phase(
        car="bmw", corner_id=1, phase="mid_corner",
        lat_g=1.4, long_g=0.0, speed_ms=50.0,
        aero_balance_pct=54.0,  # ideal
        aero_ld_ratio=4.0,
        front_ceiling=_ceiling("bmw", "front"),
        rear_ceiling=_ceiling("bmw", "rear"),
        surrogate_lat_g_ceiling=2.2,  # matches predict_peak_lat_g(4.0, 50.0) ~ 2.2
    )
    # Margin around 0.93 (under 1.0)
    report = guardrail_check(score, front_margin=0.93, rear_margin=0.93)
    assert report.flagged is False
    assert report.reason is None


def test_evaluate_corner_phase_no_surrogate_neutral_headroom() -> None:
    """Without surrogate ceiling, headroom defaults to neutral 1.0.

    A speed-anchored proxy was tested in Day 12b and rejected by
    external judge as tautological. The neutral default is honest
    when standalone (no surrogate available); the recommender
    integration always has surrogates, so the default is rarely hit
    in production paths.
    """
    score = evaluate_corner_phase(
        car="bmw",
        corner_id=1, phase="mid_corner",
        lat_g=1.5, long_g=0.0, speed_ms=50.0,
        aero_balance_pct=54.0, aero_ld_ratio=4.0,
        front_ceiling=_ceiling("bmw", "front"),
        rear_ceiling=_ceiling("bmw", "rear"),
        surrogate_lat_g_ceiling=None,
    )
    assert score.grip_headroom_score == 1.0


# ---- evaluate_lap (bulk wrapper) ---------------------------------------


def test_evaluate_lap_skips_invalid_samples() -> None:
    """Samples missing required keys are silently skipped."""
    samples = [
        {  # valid
            "corner_id": 1, "phase": "mid_corner",
            "lat_g": 1.5, "long_g": 0.0, "speed_ms": 50.0,
            "aero_balance_pct": 54.0, "aero_ld_ratio": 4.0,
            "surrogate_lat_g_ceiling": 1.5,
        },
        {"corner_id": 2},  # invalid
        {  # valid
            "corner_id": 3, "phase": "exit",
            "lat_g": 1.0, "long_g": 0.5, "speed_ms": 60.0,
            "aero_balance_pct": 53.0, "aero_ld_ratio": 4.2,
            "surrogate_lat_g_ceiling": 1.4,
        },
    ]
    scores = evaluate_lap(
        "bmw", samples,
        _ceiling("bmw", "front"), _ceiling("bmw", "rear"),
    )
    assert len(scores) == 2  # invalid sample skipped
    assert scores[0].corner_id == 1
    assert scores[1].corner_id == 3


# ---- Canary -------------------------------------------------------------


def test_canary_constant_inputs_yield_same_score() -> None:
    """Broken-model canary: if the evaluator produces the SAME score
    regardless of inputs (e.g. always returns 0.5), Spearman with
    empirical lap-time would be 0.

    This canary tests the inverse: with VARIED inputs, the score
    should VARY. If a future commit breaks this (e.g. hardcodes
    composite_score = 0.5), the gate's Spearman correlation with
    lap-time would collapse, fulfilling PLAN.md's canary.
    """
    front = _ceiling("bmw", "front", mu=1.5)
    rear = _ceiling("bmw", "rear", mu=1.5)
    # Two samples with different lat_g -> should produce different scores.
    s1 = evaluate_corner_phase(
        car="bmw", corner_id=1, phase="mid_corner",
        lat_g=0.5, long_g=0.0, speed_ms=30.0,  # underutilising
        aero_balance_pct=70.0,  # off target
        aero_ld_ratio=3.0,
        front_ceiling=front, rear_ceiling=rear,
        surrogate_lat_g_ceiling=1.5,
    )
    s2 = evaluate_corner_phase(
        car="bmw", corner_id=2, phase="mid_corner",
        lat_g=2.0, long_g=0.0, speed_ms=70.0,  # at limit
        aero_balance_pct=54.0,  # ideal
        aero_ld_ratio=4.5,
        front_ceiling=front, rear_ceiling=rear,
        surrogate_lat_g_ceiling=2.0,
    )
    # Scores must differ.
    assert s1.composite_score != s2.composite_score, (
        "two clearly different (lat_g, balance, speed) samples produced "
        "the same composite score; the canary's premise is broken"
    )


def test_constants_sum_to_one() -> None:
    """The three score-component weights must sum to exactly 1.0."""
    total = _WEIGHT_AXLE_UTIL + _WEIGHT_AERO_BALANCE + _WEIGHT_GRIP_HEADROOM
    assert total == pytest.approx(1.0)
