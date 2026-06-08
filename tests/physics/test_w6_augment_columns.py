"""W6 Forest augment columns land in FitRecord.feature_names."""

from __future__ import annotations

import polars as pl

from racingoptimizer.physics.aero_fit_features import aero_fit_column_names
from racingoptimizer.physics.fitter import (
    CORNER_ARCHETYPE_COLUMNS,
    DRIVER_CONTROL_COLUMNS,
    _fit_one_quadruple,
)


def test_forest_fit_includes_driver_and_aero_augment_columns() -> None:
    sub = pl.DataFrame(
        {
            "accel_lat_g_max": [1.0, 1.2, 1.1, 1.3, 1.0],
            "heave_spring_rate_n_per_mm": [50.0, 55.0, 52.0, 54.0, 51.0],
            "air_density": [1.225] * 5,
            "air_temp_c": [20.0] * 5,
            "air_pressure_pa": [101325.0] * 5,
            "relative_humidity": [0.5] * 5,
            "wind_vel_ms": [0.0] * 5,
            "wind_dir_rad": [0.0] * 5,
            "fog_level": [0.0] * 5,
            "track_temp_c": [30.0] * 5,
            "track_wetness": [0.0] * 5,
            "weather_declared_wet": [0] * 5,
            "precip_type": [0] * 5,
            "skies": [0] * 5,
            "steering_mean_rad": [0.1, 0.2, 0.15, 0.18, 0.12],
            "brake_mean": [0.0, 0.1, 0.0, 0.2, 0.0],
            "throttle_mean": [0.8, 0.7, 0.75, 0.72, 0.78],
            "aero_map_ld_ratio": [3.5, 3.4, 3.6, 3.5, 3.4],
            "aero_map_balance_pct": [48.0, 49.0, 47.0, 48.5, 48.0],
            **{c: [1.0] * 5 for c in CORNER_ARCHETYPE_COLUMNS},
        }
    )
    rec = _fit_one_quadruple(
        sub=sub,
        parameters=["heave_spring_rate_n_per_mm"],
        output_channel="accel_lat_g_max",
        family_kind="rf",
        seed=0,
        cv_seed=1,
        k_folds=2,
        archetype_columns=CORNER_ARCHETYPE_COLUMNS,
        augment_columns=DRIVER_CONTROL_COLUMNS + aero_fit_column_names(),
    )
    assert rec is not None
    for col in DRIVER_CONTROL_COLUMNS + aero_fit_column_names():
        assert col in rec.feature_names
