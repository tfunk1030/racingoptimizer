"""Per-car baseline derivation (gap #15)."""
from __future__ import annotations

import numpy as np
import polars as pl

from racingoptimizer.physics.baselines import (
    DEFAULT_BASELINES,
    CarBaselines,
    baselines_for,
    derive_baselines,
)


def _frame(rows: list[dict[str, float]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_default_baselines_exist_for_every_gtp_car() -> None:
    expected = {"acura", "bmw", "cadillac", "ferrari", "porsche"}
    assert expected.issubset(set(DEFAULT_BASELINES.keys()))
    for car, base in DEFAULT_BASELINES.items():
        assert isinstance(base, CarBaselines)
        assert base.car == car
        # Every scale must be a positive finite float.
        for field in (
            base.max_lateral_g, base.understeer_scale_rad,
            base.yaw_rate_scale_rad_s, base.wheelspin_scale_ms,
            base.ride_height_variance_scale_mm, base.shock_defl_scale_mm,
            base.aero_grip_baseline_g,
        ):
            assert np.isfinite(field) and field > 0.0


def test_derive_baselines_empty_corpus_returns_defaults() -> None:
    base = derive_baselines("bmw", None)
    assert base == DEFAULT_BASELINES["bmw"]
    base_empty = derive_baselines("bmw", pl.DataFrame())
    assert base_empty == DEFAULT_BASELINES["bmw"]


def test_derive_baselines_uses_99th_percentile_of_observed_channels() -> None:
    # 100 rows so the 99th percentile is well-defined and visibly different
    # from the per-car cold-start defaults.
    n = 100
    lat = np.linspace(0.0, 3.0, n)            # 99p ≈ 2.97
    us = np.linspace(0.0, 0.4, n)             # 99p ≈ 0.396
    yaw = np.linspace(0.0, 4.0, n)            # 99p ≈ 3.96
    rh = np.tile(np.array([10.0, 20.0, 30.0, 40.0]), (n, 1))
    shock = np.linspace(0.0, 30.0, n)

    frame = pl.DataFrame({
        "accel_lat_g_max": lat,
        "understeer_angle_mean_rad": us,
        "yaw_rate_max_rad_s": yaw,
        "lf_ride_height_mean_mm": rh[:, 0],
        "rf_ride_height_mean_mm": rh[:, 1],
        "lr_ride_height_mean_mm": rh[:, 2],
        "rr_ride_height_mean_mm": rh[:, 3],
        "lf_shock_defl_p99_mm": shock,
        "rf_shock_defl_p99_mm": shock,
        "lr_shock_defl_p99_mm": shock,
        "rr_shock_defl_p99_mm": shock,
    })

    base = derive_baselines("bmw", frame)
    assert base.car == "bmw"
    # Each scale should match np.quantile(values, 0.99) ± float-precision.
    assert abs(base.max_lateral_g - float(np.quantile(np.abs(lat), 0.99))) < 1e-9
    assert abs(base.understeer_scale_rad - float(np.quantile(np.abs(us), 0.99))) < 1e-9
    assert abs(base.yaw_rate_scale_rad_s - float(np.quantile(np.abs(yaw), 0.99))) < 1e-9
    # The four ride-height columns are constant per row, so per-row
    # variance is constant; quantile collapses to that value (~125.0).
    expected_rh_var = float(np.var(np.array([10.0, 20.0, 30.0, 40.0]), ddof=0))
    assert abs(base.ride_height_variance_scale_mm - expected_rh_var) < 1e-9
    expected_shock = float(np.quantile(np.abs(shock), 0.99))
    assert abs(base.shock_defl_scale_mm - expected_shock) < 1e-9


def test_derive_baselines_missing_channel_falls_back_to_default() -> None:
    # Frame has lateral G but no shock-deflection channels (the Acura case).
    frame = pl.DataFrame({"accel_lat_g_max": np.linspace(0.0, 2.5, 50)})
    base = derive_baselines("acura", frame)
    default = DEFAULT_BASELINES["acura"]
    # Lateral G is derived; shock baseline falls back to default.
    assert base.max_lateral_g != default.max_lateral_g
    assert base.shock_defl_scale_mm == default.shock_defl_scale_mm
    assert base.ride_height_variance_scale_mm == default.ride_height_variance_scale_mm


def test_baselines_for_dispatcher_matches_derive() -> None:
    frame = pl.DataFrame({"accel_lat_g_max": np.linspace(0.0, 2.0, 50)})
    assert baselines_for("bmw", frame) == derive_baselines("bmw", frame)
    assert baselines_for("ferrari", None) == DEFAULT_BASELINES["ferrari"]
