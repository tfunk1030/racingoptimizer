"""Day 9 of physics-rebuild: per-car damper curve refit (T4.4).

PLAN.md Section 15.2: replace the pre-Day-9 seeded damper coefficients
(uniform 5.5-6.5 N*s/mm across all 5 GTP cars, knee fixed at 100 mm/s)
with per-car-fitted values calibrated from the corpus's shock-velocity
distributions.

Acceptance gate: per-car damper curve fit residual < 8% on held-out
laps for all 5 cars; refit baseline beats seeded baseline on residual
MAE.

Broken-model canary: use the seeded curves; gate must FAIL on residual
MAE comparison.
"""
from __future__ import annotations

import numpy as np
import pytest

from racingoptimizer.physics.damper_force import (
    _ANCHOR_PERCENTILE,
    _K_LOW_SPEED_MAX_NS_PER_MM,
    _K_LOW_SPEED_MIN_NS_PER_MM,
    _KNEE_PERCENTILE,
    _TARGET_FORCE_AT_P95_N,
    DAMPER_COEFFICIENT_NS_PER_MM,
    DIGRESSIVE_KNEE_MM_S,
    DamperCurve,
    damper_coefficient,
    estimate_damper_force_n,
    fit_damper_curve_from_velocities,
)

# ---- backward compatibility (pre-Day-9 path still works) ----------------


def test_damper_coefficient_lookup_unchanged() -> None:
    """The seeded per-car constants are still accessible."""
    assert damper_coefficient("bmw") == DAMPER_COEFFICIENT_NS_PER_MM["bmw"]
    assert damper_coefficient("ferrari") == DAMPER_COEFFICIENT_NS_PER_MM["ferrari"]


def test_estimate_force_with_seeded_car_unchanged() -> None:
    """Pre-Day-9 callers (`car=...` only) get the seeded curve."""
    v = np.array([10.0, 50.0, 100.0, 200.0])
    f_seeded = estimate_damper_force_n(v, car="bmw")
    # Same numerics as pre-Day-9.
    k = DAMPER_COEFFICIENT_NS_PER_MM["bmw"]
    expected_low = k * 10.0  # below 100 mm/s knee, linear
    assert f_seeded[0] == pytest.approx(expected_low)


# ---- DamperCurve dataclass + estimate_damper_force_n with curve ----------


def test_estimate_force_with_curve_uses_curve_params() -> None:
    """Pass a DamperCurve and the function uses its (k, knee) directly."""
    curve = DamperCurve(
        car="bmw", k_low_speed_ns_per_mm=8.0, knee_mm_s=50.0,
        n_samples=10000, p30_velocity_mm_s=50.0, p95_velocity_mm_s=200.0,
    )
    v = np.array([10.0, 50.0, 100.0, 200.0])
    f = estimate_damper_force_n(v, curve=curve)
    # At v=10, below knee=50, force = 8 * 10 = 80.
    assert f[0] == pytest.approx(80.0)


def test_estimate_force_curve_overrides_car_kwarg() -> None:
    """Curve takes precedence over car when both provided (curve is the
    refitted truth source)."""
    curve = DamperCurve(
        car="cadillac", k_low_speed_ns_per_mm=4.0, knee_mm_s=80.0,
        n_samples=10000, p30_velocity_mm_s=80.0, p95_velocity_mm_s=200.0,
    )
    v = np.array([10.0])
    # Despite car="bmw" (seeded k=6), the curve's k=4 wins.
    f = estimate_damper_force_n(v, car="bmw", curve=curve)
    assert f[0] == pytest.approx(40.0)


def test_damper_curve_immutable() -> None:
    from dataclasses import FrozenInstanceError
    curve = DamperCurve(
        car="bmw", k_low_speed_ns_per_mm=6.0, knee_mm_s=100.0,
        n_samples=10000, p30_velocity_mm_s=100.0, p95_velocity_mm_s=300.0,
    )
    with pytest.raises(FrozenInstanceError):
        curve.k_low_speed_ns_per_mm = 7.0  # type: ignore[misc]


# ---- fit_damper_curve_from_velocities (the core refit) -------------------


def test_fit_synthesises_distinct_per_car_curves() -> None:
    """Two cars with different velocity distributions yield distinct
    DamperCurve params -- the WHOLE POINT of the refit."""
    rng = np.random.default_rng(0)
    # Car A: most samples around 50 mm/s, p95 ~150 mm/s.
    car_a_vel = np.abs(rng.normal(50, 30, 5000))
    # Car B: most samples around 100 mm/s, p95 ~250 mm/s.
    car_b_vel = np.abs(rng.normal(100, 60, 5000))
    curve_a = fit_damper_curve_from_velocities("car_a", car_a_vel)
    curve_b = fit_damper_curve_from_velocities("car_b", car_b_vel)
    # Different knees (anchored at p30).
    assert curve_a.knee_mm_s != pytest.approx(curve_b.knee_mm_s, abs=2.0)
    # Different k (anchored at p95).
    assert curve_a.k_low_speed_ns_per_mm != pytest.approx(
        curve_b.k_low_speed_ns_per_mm, abs=0.1,
    )


def test_fit_anchors_knee_at_p30() -> None:
    """The fit's knee equals the 30th-percentile of |velocity|."""
    rng = np.random.default_rng(1)
    vel = np.abs(rng.normal(100, 50, 5000))
    curve = fit_damper_curve_from_velocities("test", vel)
    expected_knee = float(np.percentile(np.abs(vel), _KNEE_PERCENTILE))
    assert curve.knee_mm_s == pytest.approx(expected_knee, abs=0.01)
    assert curve.p30_velocity_mm_s == pytest.approx(expected_knee, abs=0.01)


def test_fit_anchors_k_at_p95_target_force() -> None:
    """k_low_speed * p95 == TARGET_FORCE (the 95th-percentile anchor)."""
    rng = np.random.default_rng(2)
    vel = np.abs(rng.normal(80, 40, 5000))
    curve = fit_damper_curve_from_velocities("test", vel)
    expected_p95 = float(np.percentile(np.abs(vel), _ANCHOR_PERCENTILE))
    expected_k = _TARGET_FORCE_AT_P95_N / expected_p95
    assert curve.p95_velocity_mm_s == pytest.approx(expected_p95, abs=0.01)
    assert curve.k_low_speed_ns_per_mm == pytest.approx(expected_k, abs=0.01)


def test_fit_rejects_too_few_samples() -> None:
    """Fewer than 100 samples -> ValueError; caller should use seeded."""
    with pytest.raises(ValueError, match="too few samples"):
        fit_damper_curve_from_velocities("test", np.array([1.0, 2.0, 3.0]))


def test_fit_rejects_out_of_physical_range_k() -> None:
    """If the corpus's velocity distribution would yield k outside
    [3, 10] N*s/mm, the fit raises -- caller should fall back to the
    seeded value rather than ship a pathological curve."""
    rng = np.random.default_rng(3)
    # Tiny velocities -> very small p95 -> k = TARGET / p95 is huge.
    too_small = np.abs(rng.normal(10, 5, 1000))
    with pytest.raises(ValueError, match="outside physical range"):
        fit_damper_curve_from_velocities("test", too_small)


def test_fit_rejects_zero_p95() -> None:
    """All-zero (or near-zero) velocity samples -> p95 = 0 -> rejected."""
    with pytest.raises(ValueError, match="(degenerate|outside physical range)"):
        fit_damper_curve_from_velocities("test", np.zeros(1000))


def test_fit_finite_filter_drops_nans() -> None:
    """NaN/inf in input are filtered out before percentile computation."""
    rng = np.random.default_rng(4)
    clean = np.abs(rng.normal(100, 50, 4900))
    polluted = np.concatenate([
        clean,
        np.full(50, np.nan),
        np.full(50, np.inf),
    ])
    curve_clean = fit_damper_curve_from_velocities("test", clean)
    curve_polluted = fit_damper_curve_from_velocities("test", polluted)
    # The polluted fit should match the clean fit (NaN/inf filtered).
    assert curve_clean.knee_mm_s == pytest.approx(
        curve_polluted.knee_mm_s, abs=0.5,
    )


# ---- Canary --------------------------------------------------------------


def test_canary_seeded_values_uniform_across_cars() -> None:
    """Broken-model canary: the seeded values are nearly uniform
    (5.5-6.5 N*s/mm spread = 1.0 N*s/mm). The refit should produce
    spreads >= 1.5 N*s/mm on synthetic distinct-per-car distributions.

    If a future commit reverts the refit and re-uses seeded values,
    the per-car distinction collapses -- this canary catches it.
    """
    seeded_values = list(DAMPER_COEFFICIENT_NS_PER_MM.values())
    seeded_spread = max(seeded_values) - min(seeded_values)
    assert seeded_spread <= 1.5, (
        "seeded spread is unexpectedly wide; the canary's premise "
        "(seeded was uniform-ish) no longer holds"
    )

    # Synthesise 3 distinct distributions; refit should spread > 1.5.
    rng = np.random.default_rng(0)
    refits = [
        fit_damper_curve_from_velocities(
            f"car_{i}", np.abs(rng.normal(70 + i * 30, 30, 5000)),
        )
        for i in range(3)
    ]
    refit_ks = [c.k_low_speed_ns_per_mm for c in refits]
    refit_spread = max(refit_ks) - min(refit_ks)
    assert refit_spread > 1.5, (
        f"refit k spread {refit_spread:.3f} <= 1.5; per-car distinction "
        f"is missing on synthetic distinct distributions -- canary fired"
    )


# ---- Constants ----------------------------------------------------------


def test_constants_match_plan_md() -> None:
    """The 30/95 percentile anchors and the 600 N target are the
    Day 9 design choices. If a future commit changes these without
    updating the gate or this test, the test fires."""
    assert _KNEE_PERCENTILE == 30.0
    assert _ANCHOR_PERCENTILE == 95.0
    assert _TARGET_FORCE_AT_P95_N == 600.0
    assert _K_LOW_SPEED_MIN_NS_PER_MM == 3.0
    assert _K_LOW_SPEED_MAX_NS_PER_MM == 10.0


def test_digressive_knee_constant_kept() -> None:
    """The pre-Day-9 fixed knee constant is retained for the
    backward-compat seeded path."""
    assert DIGRESSIVE_KNEE_MM_S == 100.0
