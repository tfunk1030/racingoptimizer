"""P2.2 -- closed-form Bayes random intercepts on surrogate residuals.

Mathematical contracts covered:
1. Two-track corpus where one track has clean (near-zero) residuals and
   the other has a systematic positive residual mean -- the clean track
   shrinks toward zero, the divergent track retains its intercept with
   low shrinkage.
2. Single-track corpus (below ``min_tracks``) -- degraded path returns
   the per-track empirical mean with zero shrinkage (caller can decide
   whether to trust it).
3. Tracks below ``min_samples_per_track`` are dropped.
4. ``predict_correction`` returns ``(0.0, 0.0)`` on every None/missing
   path so legacy pickles + cross-car-borrowed schedules pass through
   without surrogate disturbance.
5. ``fit_all_channels`` aggregates per-channel fits into the flat
   ``(channel, track) -> TrackIntercept`` shape PhysicsModel persists.
"""
from __future__ import annotations

import pytest

from racingoptimizer.physics.track_random_intercepts import (
    TrackIntercept,
    fit_all_channels,
    fit_per_channel,
    predict_correction,
)


def test_clean_track_shrinks_to_zero_while_divergent_retains_intercept() -> None:
    # Sebring: surrogate is calibrated -- residuals oscillate around 0.
    # Spa: surrogate systematically under-predicts by ~0.5.
    res = fit_per_channel(
        {
            "sebring": [
                0.01, -0.02, 0.005, -0.01, 0.02, -0.015, 0.008, 0.0,
                0.012, -0.005,
            ],
            "spa": [
                0.48, 0.52, 0.49, 0.51, 0.50, 0.47, 0.53, 0.49, 0.50, 0.51,
            ],
        },
        channel_name="accel_lat_g_max",
    )
    assert set(res.keys()) == {"sebring", "spa"}
    assert res["sebring"].intercept == abs(res["sebring"].intercept)
    # Sebring intercept ~ 0 (within noise floor).
    assert abs(res["sebring"].intercept) < 0.05, (
        f"clean track intercept should be ~0, got {res['sebring'].intercept}"
    )
    # Spa intercept stays near the observed 0.5 -- between-track variance
    # is large vs within-track noise, so shrinkage is small.
    assert 0.40 < res["spa"].intercept < 0.55
    assert res["spa"].shrinkage < 0.20
    # Std reflects partial-pooled uncertainty.
    assert res["spa"].intercept_std > 0.0


def test_constant_residuals_across_tracks_all_shrink_to_zero() -> None:
    # No systematic per-track bias -- all residuals near zero. Each
    # track's posterior should be ~0 with shrinkage > 0.5 (the maths
    # decides the within-track noise dominates the between-track signal).
    res = fit_per_channel(
        {
            "sebring": [-0.001, 0.002, -0.001, 0.0, 0.001, -0.002, 0.001],
            "spa":     [0.001, -0.001, 0.0, 0.001, -0.001, 0.0, 0.001],
            "monza":   [0.0, 0.001, -0.001, 0.0, 0.0, 0.001, -0.001],
        },
        channel_name="accel_lat_g_max",
    )
    assert set(res.keys()) == {"sebring", "spa", "monza"}
    for t in res.values():
        assert abs(t.intercept) < 0.005


def test_single_track_degrades_to_empirical_mean() -> None:
    # Below `min_tracks`, the partial-pooling math degrades to the
    # per-track empirical mean with shrinkage=0 (no cross-track signal
    # to estimate tau^2 from).
    res = fit_per_channel(
        {"spa": [0.3, 0.35, 0.32, 0.28]},
        channel_name="accel_lat_g_max",
    )
    assert "spa" in res
    intercept = res["spa"]
    # mean(values) = (0.3+0.35+0.32+0.28)/4 = 0.3125
    assert abs(intercept.intercept - 0.3125) < 1e-9
    assert intercept.shrinkage == 0.0
    assert intercept.n_samples == 4
    assert intercept.intercept_std > 0.0


def test_min_samples_per_track_drops_thin_tracks() -> None:
    res = fit_per_channel(
        {
            "sebring": [0.01, 0.02, -0.01, 0.0, 0.01, -0.02, 0.01],
            "spa":     [0.3],  # single sample; dropped.
            "monza":   [0.05, 0.06, 0.04, 0.05],
        },
        channel_name="ch",
        min_samples_per_track=3,
    )
    assert "spa" not in res
    assert {"sebring", "monza"} <= set(res.keys())


def test_predict_correction_none_or_missing_returns_zero() -> None:
    fit = {
        ("accel_lat_g_max", "spa"): TrackIntercept(
            channel="accel_lat_g_max", track="spa", intercept=0.5,
            intercept_std=0.05, n_samples=10, shrinkage=0.1,
        ),
    }
    # Happy path.
    intercept, std = predict_correction(fit, "accel_lat_g_max", "spa")
    assert intercept == 0.5
    assert std == pytest.approx(0.10)  # 0.05 * W6 OOF inflation (2x)
    # Track absent from fit.
    assert predict_correction(fit, "accel_lat_g_max", "monza") == (0.0, 0.0)
    # Channel absent from fit.
    assert predict_correction(fit, "brake_max", "spa") == (0.0, 0.0)
    # No fit dict at all (legacy pickle).
    assert predict_correction(None, "accel_lat_g_max", "spa") == (0.0, 0.0)
    # Empty fit dict.
    assert predict_correction({}, "accel_lat_g_max", "spa") == (0.0, 0.0)
    # None track (caller didn't thread the target).
    assert predict_correction(fit, "accel_lat_g_max", None) == (0.0, 0.0)


def test_fit_all_channels_flattens_per_channel_dicts() -> None:
    fits = fit_all_channels(
        {
            "accel_lat_g_max": {
                "sebring": [0.01, -0.01, 0.0, 0.01, 0.02, -0.02, 0.0],
                "spa":     [0.45, 0.5, 0.48, 0.52, 0.51, 0.47, 0.49],
            },
            "brake_max": {
                "sebring": [0.0, 0.001, -0.001, 0.0, 0.001, 0.0, -0.001],
                "spa":     [-0.2, -0.18, -0.22, -0.21, -0.19, -0.2, -0.21],
            },
        },
    )
    # Both channels x both tracks -> 4 entries.
    assert set(fits.keys()) == {
        ("accel_lat_g_max", "sebring"),
        ("accel_lat_g_max", "spa"),
        ("brake_max", "sebring"),
        ("brake_max", "spa"),
    }
    # Spa intercept is positive for lat-G, negative for brake_max --
    # signs propagate through the maths.
    assert fits[("accel_lat_g_max", "spa")].intercept > 0.4
    assert fits[("brake_max", "spa")].intercept < -0.15


def test_empty_input_returns_empty_dict() -> None:
    assert fit_per_channel({}) == {}
    assert fit_per_channel({"sebring": []}, min_samples_per_track=1) == {}
    assert fit_all_channels({}) == {}
