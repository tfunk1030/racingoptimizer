"""Day 6 of the physics-rebuild plan: lap-time-weighted samples (Mode 3).

Background (PLAN.md Section 14.4):
- Mode 3 (driver-bias inheritance): a slow recent stint pulls the
  baseline conservative because the per-session contribution to
  `baseline_setup` is unweighted -- recent sessions count the same as
  fast ones.
- Cheap fix: weight per-session contribution by 1/(session_best_lap -
  track_min + 0.5s). Sessions with valid laps and best-pace get the
  highest weight; sessions without laps fall back to weight 1.0.

Acceptance gate (per PLAN.md Section 14.4): for the BMW Sebring corpus
(37 sessions), the fitter's per-parameter `baseline_setup` shifts toward
the values used in the user's known-fast laps (top quartile by lap time)
by >= 0.3 *step* units on at least 5 fittable parameters.

Broken-model canary: set all weights to 1.0 (effectively disable the
weighting); gate must FAIL.

Day 6 unit tests cover the helpers + a synthetic Mode 3 case + the
canary. The full BMW Sebring gate runs in `scripts/day_06_gate.py`.
"""
from __future__ import annotations

from statistics import median, pstdev

import pytest

from racingoptimizer.physics.fitter import (
    _LAP_WEIGHT_EPSILON_S,
    _weighted_median,
    _weighted_std,
)

# ---- _weighted_median ---------------------------------------------------


def test_weighted_median_uniform_weights_matches_plain_median() -> None:
    """When all weights are equal, weighted_median = plain median."""
    values = [10.0, 11.0, 12.0, 13.0, 14.0]
    weights = [1.0] * 5
    assert _weighted_median(values, weights) == pytest.approx(median(values))


def test_weighted_median_pulls_toward_heavy_weight() -> None:
    """A single heavy weight at one extreme pulls the median to it.

    Synthetic Mode 3 case: 3 sessions with setup values {10, 15, 20},
    weights {2.0, 0.18, 0.095} (corresponding to fast/medium/slow laps).
    Plain median = 15. Weighted median should be 10 (heaviest weight
    is at value 10, accumulating past half-total before reaching 15).
    """
    values = [10.0, 15.0, 20.0]
    weights = [2.0, 0.18, 0.095]
    assert _weighted_median(values, weights) == pytest.approx(10.0)


def test_weighted_median_zero_weights_falls_back_to_plain() -> None:
    """If all weights are zero, fall back to plain median (defensive)."""
    values = [10.0, 11.0, 12.0]
    weights = [0.0, 0.0, 0.0]
    assert _weighted_median(values, weights) == pytest.approx(11.0)


def test_weighted_median_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        _weighted_median([], [])


def test_weighted_median_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="len"):
        _weighted_median([1.0, 2.0], [1.0])


def test_weighted_median_stable_under_permutation() -> None:
    """Same inputs in different order yield the same weighted median."""
    values_a = [10.0, 15.0, 20.0]
    weights_a = [2.0, 1.0, 0.5]
    values_b = [20.0, 10.0, 15.0]
    weights_b = [0.5, 2.0, 1.0]
    assert _weighted_median(values_a, weights_a) == _weighted_median(values_b, weights_b)


# ---- _weighted_std ------------------------------------------------------


def test_weighted_std_uniform_weights_matches_pstdev() -> None:
    values = [10.0, 12.0, 14.0]
    weights = [1.0, 1.0, 1.0]
    assert _weighted_std(values, weights) == pytest.approx(pstdev(values))


def test_weighted_std_concentrated_weight_shrinks() -> None:
    """Weighting all mass at one value should yield a small std."""
    values = [10.0, 100.0]
    weights = [1000.0, 0.001]
    out = _weighted_std(values, weights)
    plain = pstdev(values)
    assert out < plain * 0.1


def test_weighted_std_single_value_returns_zero() -> None:
    assert _weighted_std([42.0], [1.0]) == 0.0


def test_weighted_std_zero_weights_falls_back_to_pstdev() -> None:
    values = [10.0, 12.0, 14.0]
    weights = [0.0, 0.0, 0.0]
    assert _weighted_std(values, weights) == pytest.approx(pstdev(values))


# ---- Lap-time weight formula constants ----------------------------------


def test_epsilon_constant_documented() -> None:
    """The 0.5s epsilon floor is the documented PLAN.md value."""
    assert _LAP_WEIGHT_EPSILON_S == 0.5


# ---- Synthetic Mode 3 closure -------------------------------------------


def test_mode_3_synthetic_baseline_shifts_toward_fast_lap_setup() -> None:
    """Canonical Mode 3 case: 3 sessions with known fast-lap times and
    setup values. Verify the weighted baseline matches the fast-lap
    setup, NOT the corpus median.

    Maps to PLAN.md Section 14.4: a conservative recent stint should not
    pull the baseline toward a conservative setup if the user's faster
    historical sessions used a different setup.

    Setup: 3 BMW sessions on Sebring.
    - session A: best_lap = 100.0, setup_X = 10  (FASTEST)
    - session B: best_lap = 102.5, setup_X = 15  (medium)
    - session C: best_lap = 110.0, setup_X = 20  (slow)

    Track min = 100.0; gaps = (0, 2.5, 10.0); weights = 1/(gap+0.5).
    Weighted median pulls toward the fast session's setup (10).
    Plain median would be 15 (the middle value).
    """
    session_best = {"A": 100.0, "B": 102.5, "C": 110.0}
    track_min = 100.0
    weights = {
        sid: 1.0 / (session_best[sid] - track_min + _LAP_WEIGHT_EPSILON_S)
        for sid in session_best
    }
    setup_values = [10.0, 15.0, 20.0]
    setup_weights = [weights["A"], weights["B"], weights["C"]]
    weighted = _weighted_median(setup_values, setup_weights)
    plain = median(setup_values)
    assert weighted == pytest.approx(10.0), (
        f"weighted median {weighted} did not pull to fast-session setup 10"
    )
    assert plain == pytest.approx(15.0)
    # Plain != weighted -- the entire point of the weighting.
    assert weighted != plain


def test_mode_3_canary_uniform_weights_no_shift() -> None:
    """Broken-model canary: with all weights = 1.0, the weighted median
    collapses to the plain median (no Mode 3 closure)."""
    setup_values = [10.0, 15.0, 20.0]
    uniform_weights = [1.0, 1.0, 1.0]
    weighted = _weighted_median(setup_values, uniform_weights)
    plain = median(setup_values)
    assert weighted == pytest.approx(plain), (
        "uniform weights should produce plain-median behaviour; "
        "if this test fails, the weighting code may have a bias not "
        "tied to weight values (suspicious)"
    )


def test_weighted_median_three_step_shift_threshold() -> None:
    """The acceptance gate threshold is >= 0.3 step. Verify the weighted
    median can produce shifts of that magnitude on representative inputs.

    Setup parameter with step = 1.0 (e.g. spring_rate_n_per_mm in 1 N/mm
    increments). 6 sessions with two clusters at 10 and 13 (delta = 3
    steps); fastest 2 in the 10 cluster, others in the 13 cluster.
    Plain median (over 6 evenly-weighted) = 11.5.
    Weighted should pull toward 10 (the fast cluster).
    Shift = 11.5 - 10 = 1.5 steps; well above the 0.3 threshold.
    """
    values = [10.0, 10.0, 13.0, 13.0, 13.0, 13.0]
    # First two are fastest; weighting them heavily.
    weights = [10.0, 10.0, 1.0, 1.0, 1.0, 1.0]
    weighted = _weighted_median(values, weights)
    plain = median(values)  # median of [10,10,13,13,13,13] = 13.0
    assert weighted == 10.0
    assert plain == 13.0
    shift = abs(plain - weighted)
    assert shift >= 0.3
