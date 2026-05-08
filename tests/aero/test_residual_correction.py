"""Day 11 of physics-rebuild: per-car aero-map residual correction.

Per PLAN.md Section 15.3 (second half): fit a per-car scalar
correction on the aero-map-derived peak lat-G prediction so corrected
predictions better match observed corpus values. Fallback authorized
if correction doesn't beat raw.
"""
from __future__ import annotations

import numpy as np
import pytest

from racingoptimizer.aero.residual_correction import (
    _CORRECTION_BOUND_PCT,
    AeroResidualCorrection,
    apply_correction,
    fit_residual_correction,
    improvement_pct,
    predict_peak_lat_g,
)

# ---- predict_peak_lat_g -------------------------------------------------


def test_predict_peak_at_zero_speed_equals_mu() -> None:
    """At zero speed, no aero downforce -> peak = mu (gravitational only)."""
    out = predict_peak_lat_g(ld_ratio=4.0, speed_ms=0.0, mu=1.5)
    assert out == pytest.approx(1.5)


def test_predict_peak_increases_with_speed() -> None:
    """Higher speed -> more downforce -> higher peak."""
    speeds = [10.0, 30.0, 60.0, 90.0]
    peaks = [predict_peak_lat_g(ld_ratio=4.0, speed_ms=v) for v in speeds]
    for i in range(len(peaks) - 1):
        assert peaks[i + 1] > peaks[i]


def test_predict_peak_increases_with_ld_ratio() -> None:
    """At fixed speed, higher ld_ratio -> more downforce -> higher peak."""
    lds = [2.0, 3.0, 4.0, 5.0]
    peaks = [predict_peak_lat_g(ld_ratio=ld, speed_ms=60.0) for ld in lds]
    for i in range(len(peaks) - 1):
        assert peaks[i + 1] > peaks[i]


# ---- fit_residual_correction --------------------------------------------


def _synthetic_samples(
    n: int = 200, true_correction: float = 0.10, noise_sigma: float = 0.02,
) -> list[dict]:
    """Build n samples where observed = predicted * (1 + true_correction) +
    noise. The fit should recover ~true_correction."""
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(n):
        ld = float(rng.uniform(2.5, 4.5))
        speed = float(rng.uniform(40.0, 90.0))
        pred = predict_peak_lat_g(ld_ratio=ld, speed_ms=speed)
        obs = pred * (1.0 + true_correction) + rng.normal(0, noise_sigma)
        samples.append({"ld_ratio": ld, "speed_ms": speed, "observed_lat_g": obs})
    return samples


def test_fit_recovers_true_correction_on_synthetic() -> None:
    """With +10% systematic offset on synthetic data, fit recovers ~+0.10."""
    samples = _synthetic_samples(n=500, true_correction=0.10, noise_sigma=0.01)
    correction = fit_residual_correction("test", samples)
    assert abs(correction.correction_factor - 0.10) < 0.02
    assert correction.fallback_mode_used is False
    assert correction.fit_mae_corrected_g < correction.fit_mae_raw_g


def test_fit_falls_back_when_no_systematic_offset() -> None:
    """Pure noise (no systematic offset) -> correction can't beat raw on
    MAE, fallback triggered."""
    rng = np.random.default_rng(1)
    samples = []
    for _ in range(200):
        ld = float(rng.uniform(2.5, 4.5))
        speed = float(rng.uniform(40.0, 90.0))
        pred = predict_peak_lat_g(ld_ratio=ld, speed_ms=speed)
        # Symmetric noise; no systematic offset.
        obs = pred + rng.normal(0, 0.05)
        samples.append({"ld_ratio": ld, "speed_ms": speed, "observed_lat_g": obs})
    correction = fit_residual_correction("test", samples)
    # The fitted scalar should be near 0 (no real bias). Corrected MAE
    # may be slightly worse than raw on this random sample; fallback
    # path triggers if so.
    assert abs(correction.correction_factor) < 0.05 or correction.fallback_mode_used


def test_fit_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="too few samples"):
        fit_residual_correction("test", [{"ld_ratio": 4.0, "speed_ms": 60.0,
                                          "observed_lat_g": 1.5}])


def test_fit_caps_extreme_corrections() -> None:
    """If the corpus suggests a correction beyond ±30%, fit returns
    fallback (correction_factor=0) -- avoids extrapolating massive
    biases that probably indicate a bug elsewhere."""
    rng = np.random.default_rng(2)
    samples = []
    for _ in range(200):
        ld = float(rng.uniform(2.5, 4.5))
        speed = float(rng.uniform(40.0, 90.0))
        pred = predict_peak_lat_g(ld_ratio=ld, speed_ms=speed)
        # Extreme +100% offset -- exceeds bound.
        obs = pred * 2.0 + rng.normal(0, 0.01)
        samples.append({"ld_ratio": ld, "speed_ms": speed, "observed_lat_g": obs})
    correction = fit_residual_correction("test", samples)
    assert correction.fallback_mode_used is True
    assert correction.correction_factor == 0.0


def test_fit_filters_invalid_samples() -> None:
    """NaN, negative, zero-speed samples are filtered out."""
    rng = np.random.default_rng(3)
    samples = []
    # 100 valid samples.
    for _ in range(100):
        ld = float(rng.uniform(2.5, 4.5))
        speed = float(rng.uniform(40.0, 90.0))
        pred = predict_peak_lat_g(ld_ratio=ld, speed_ms=speed)
        samples.append({
            "ld_ratio": ld, "speed_ms": speed,
            "observed_lat_g": pred * 1.10,
        })
    # 50 polluted samples.
    samples.extend([
        {"ld_ratio": float("nan"), "speed_ms": 60.0, "observed_lat_g": 1.5}
    ] * 25)
    samples.extend([
        {"ld_ratio": 4.0, "speed_ms": 0.0, "observed_lat_g": 1.5}
    ] * 25)
    correction = fit_residual_correction("test", samples)
    # Should have used the 100 clean samples; fit ~+10%.
    assert correction.n_samples == 100
    assert abs(correction.correction_factor - 0.10) < 0.01


# ---- apply_correction --------------------------------------------------


def test_apply_correction_scales_prediction() -> None:
    correction = AeroResidualCorrection(
        car="test", correction_factor=0.10, n_samples=200,
        fit_mae_raw_g=0.05, fit_mae_corrected_g=0.04,
        fallback_mode_used=False,
    )
    out = apply_correction(1.5, correction)
    assert out == pytest.approx(1.65)


def test_apply_correction_vectorised() -> None:
    correction = AeroResidualCorrection(
        car="test", correction_factor=-0.05, n_samples=200,
        fit_mae_raw_g=0.05, fit_mae_corrected_g=0.04,
        fallback_mode_used=False,
    )
    out = apply_correction(np.array([1.0, 2.0, 3.0]), correction)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, np.array([0.95, 1.90, 2.85]))


def test_apply_correction_fallback_zero_correction() -> None:
    """When fallback was triggered (correction_factor=0), apply is no-op."""
    correction = AeroResidualCorrection(
        car="test", correction_factor=0.0, n_samples=200,
        fit_mae_raw_g=0.05, fit_mae_corrected_g=0.05,
        fallback_mode_used=True,
    )
    assert apply_correction(1.5, correction) == pytest.approx(1.5)


# ---- improvement_pct ---------------------------------------------------


def test_improvement_pct_zero_when_fallback() -> None:
    correction = AeroResidualCorrection(
        car="test", correction_factor=0.0, n_samples=200,
        fit_mae_raw_g=0.05, fit_mae_corrected_g=0.05,
        fallback_mode_used=True,
    )
    assert improvement_pct(correction) == 0.0


def test_improvement_pct_positive_on_real_correction() -> None:
    correction = AeroResidualCorrection(
        car="test", correction_factor=0.10, n_samples=200,
        fit_mae_raw_g=0.05, fit_mae_corrected_g=0.04,
        fallback_mode_used=False,
    )
    assert improvement_pct(correction) == pytest.approx(20.0)


# ---- Constants ---------------------------------------------------------


def test_correction_bound_documented() -> None:
    """The +/-30% bound on per-car correction magnitude."""
    assert _CORRECTION_BOUND_PCT == 30.0
