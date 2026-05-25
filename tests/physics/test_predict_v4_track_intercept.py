"""P2.2 -- ``_predict_v4`` applies the track random intercept additively.

Stub ``FitRecord`` + ``PhysicsModel`` so we can verify in isolation that:

1. With no ``track_random_intercepts`` slot, the surrogate's mu passes
   through unchanged (legacy behaviour).
2. With an intercept fitted for ``(channel, target_track)``, the
   prediction's value is shifted by the intercept amount.
3. With ``track=None`` at predict time, the correction is suppressed
   even if the slot has data (caller must opt in).
4. The CI std widens by the intercept's posterior std in quadrature.
"""
from __future__ import annotations

import numpy as np

from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner import CornerPhaseKey, Phase
from racingoptimizer.physics.fitter import _ENV_COLUMNS
from racingoptimizer.physics.fitters import FitterBase
from racingoptimizer.physics.model import FitRecord, PhysicsModel
from racingoptimizer.physics.track_random_intercepts import TrackIntercept


class _ConstantFitter(FitterBase):
    """Fitter that returns a fixed ``(mu, sigma)`` regardless of input."""

    def __init__(self, mu: float = 1.5, sigma: float = 0.1) -> None:
        super().__init__()
        self._mu = float(mu)
        self._sigma = float(sigma)
        self.is_trained = True
        self.n_samples = 100

    def fit(self, X, y, *, sample_weight=None):  # type: ignore[override]
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[override]
        n = X.shape[0]
        return (
            np.full(n, self._mu, dtype=np.float64),
            np.full(n, self._sigma, dtype=np.float64),
        )


def _make_model(*, intercepts=None) -> PhysicsModel:
    feature_names = (
        "rear_wing_angle_deg", *_ENV_COLUMNS,
        "corner_apex_speed_ms", "corner_peak_lat_g",
        "corner_max_speed_ms", "corner_min_speed_ms",
        "corner_duration_s", "corner_compression_demand_mms",
        "phase_duration_s",
    )
    record = FitRecord(
        fitter=_ConstantFitter(mu=1.5, sigma=0.10),
        n_samples=100,
        cv_residual_std=0.05,
        signal_std=0.50,
        feature_names=feature_names,
        bootstrap_std=0.0,
    )
    return PhysicsModel(
        car="bmw",
        session_ids=("sid0",),
        track_models_used={"sid0": "sebring_international"},
        fitters={("mid_corner", "accel_lat_g_max"): record},
        baseline_setup={"rear_wing_angle_deg": 15.0},
        feature_schema_version=7,
        track_random_intercepts=intercepts or {},
    )


def _env() -> EnvironmentFrame:
    return EnvironmentFrame(
        air_temp_c=20.0, air_density=1.18, air_pressure_mbar=1013.0,
        relative_humidity=0.5, wind_vel_ms=0.0, wind_dir_deg=0.0,
        fog_level=0.0, track_temp_c=30.0, track_wetness=0.0,
        weather_declared_wet=False, precip_type=-1, skies=-1,
    )


def _archetype() -> dict[str, float]:
    return {
        "corner_apex_speed_ms": 35.0,
        "corner_peak_lat_g": 1.2,
        "corner_max_speed_ms": 60.0,
        "corner_min_speed_ms": 35.0,
        "corner_duration_s": 3.0,
        "corner_compression_demand_mms": 200.0,
        "phase_duration_s": 1.0,
    }


def _key() -> CornerPhaseKey:
    return CornerPhaseKey(
        session_id="<test>", lap_index=0, corner_id=3, phase=Phase.MID_CORNER,
    )


def test_no_intercepts_means_no_correction() -> None:
    model = _make_model(intercepts=None)
    setup = {"rear_wing_angle_deg": 15.0}
    state = model.predict(
        setup, _env(), _key(), corner_archetype=_archetype(),
        track="spa_2024_up",
    )
    conf = state.states["accel_lat_g_max"]
    # Stub fitter returns mu=1.5; with no intercept, value passes through.
    assert abs(conf.value - 1.5) < 1e-9
    # Confidence.derive widens the bracket by a regime-derived factor of
    # the std; the bracket exists but its exact width depends on the
    # confidence regime. We only need a non-zero finite width here.
    assert conf.hi > conf.value
    assert conf.lo < conf.value


def test_intercept_applied_when_track_matches() -> None:
    intercepts = {
        ("accel_lat_g_max", "spa_2024_up"): TrackIntercept(
            channel="accel_lat_g_max", track="spa_2024_up",
            intercept=0.30, intercept_std=0.05,
            n_samples=50, shrinkage=0.10,
        ),
    }
    model = _make_model(intercepts=intercepts)
    state = model.predict(
        {"rear_wing_angle_deg": 15.0}, _env(), _key(),
        corner_archetype=_archetype(),
        track="spa_2024_up",
    )
    conf = state.states["accel_lat_g_max"]
    # mu (1.5) + intercept (0.30) = 1.80.
    assert abs(conf.value - 1.80) < 1e-9


def test_intercept_suppressed_when_track_is_none() -> None:
    """Even with an intercept available, the caller must opt in by
    passing ``track``. The recommend path threads it; callers that
    don't have a target track (e.g. counterfactual probes in the
    narrative renderer) keep surrogate-only output by passing None.
    """
    intercepts = {
        ("accel_lat_g_max", "spa_2024_up"): TrackIntercept(
            channel="accel_lat_g_max", track="spa_2024_up",
            intercept=0.30, intercept_std=0.05,
            n_samples=50, shrinkage=0.10,
        ),
    }
    model = _make_model(intercepts=intercepts)
    state = model.predict(
        {"rear_wing_angle_deg": 15.0}, _env(), _key(),
        corner_archetype=_archetype(),
        track=None,
    )
    conf = state.states["accel_lat_g_max"]
    # No correction; pass-through value.
    assert abs(conf.value - 1.5) < 1e-9


def test_intercept_not_applied_when_track_absent_from_fit() -> None:
    """Track outside the fit (e.g. cross-car-borrowed schedule, target
    track has no training data) gets no correction. The recommender
    falls back to surrogate-only gracefully.
    """
    intercepts = {
        ("accel_lat_g_max", "spa_2024_up"): TrackIntercept(
            channel="accel_lat_g_max", track="spa_2024_up",
            intercept=0.30, intercept_std=0.05,
            n_samples=50, shrinkage=0.10,
        ),
    }
    model = _make_model(intercepts=intercepts)
    state = model.predict(
        {"rear_wing_angle_deg": 15.0}, _env(), _key(),
        corner_archetype=_archetype(),
        track="hockenheim_gp",  # no intercept for this track.
    )
    conf = state.states["accel_lat_g_max"]
    assert abs(conf.value - 1.5) < 1e-9


def test_intercept_widens_ci_in_quadrature() -> None:
    intercepts = {
        ("accel_lat_g_max", "spa_2024_up"): TrackIntercept(
            channel="accel_lat_g_max", track="spa_2024_up",
            intercept=0.30, intercept_std=0.20,  # large posterior std.
            n_samples=10, shrinkage=0.50,
        ),
    }
    model_no_correction = _make_model(intercepts=None)
    model_with_correction = _make_model(intercepts=intercepts)

    args = (
        {"rear_wing_angle_deg": 15.0}, _env(), _key(),
    )
    state_a = model_no_correction.predict(
        *args, corner_archetype=_archetype(), track="spa_2024_up",
    )
    state_b = model_with_correction.predict(
        *args, corner_archetype=_archetype(), track="spa_2024_up",
    )
    width_a = state_a.states["accel_lat_g_max"].hi - state_a.states["accel_lat_g_max"].value
    width_b = state_b.states["accel_lat_g_max"].hi - state_b.states["accel_lat_g_max"].value
    # Quadrature widening: sqrt(0.10^2 + 0.20^2) > 0.10.
    assert width_b > width_a
