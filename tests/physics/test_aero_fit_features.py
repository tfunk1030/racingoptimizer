"""W6 aero-map fit-time features."""

from __future__ import annotations

import polars as pl

from racingoptimizer.physics.aero_fit_features import (
    aero_fit_column_names,
    aero_map_features_for_predict,
    attach_aero_map_features,
)


def test_aero_fit_column_names_stable() -> None:
    assert aero_fit_column_names() == (
        "aero_map_ld_ratio",
        "aero_map_balance_pct",
    )


def test_attach_aero_map_features_no_surface_zeros() -> None:
    frame = pl.DataFrame(
        {
            "session_id": ["s1"],
            "aero_platform_front_rh_mean_mm": [15.0],
            "aero_platform_rear_rh_mean_mm": [40.0],
        }
    )
    out = attach_aero_map_features(frame, "acura", {"s1": {}}, None)
    assert out["aero_map_ld_ratio"][0] == 0.0
    assert out["aero_map_balance_pct"][0] == 0.0


def test_aero_map_features_for_predict_uses_static_readouts() -> None:
    from racingoptimizer.aero import load_aero_maps

    surface = load_aero_maps("acura")
    feats = aero_map_features_for_predict(
        car="acura",
        setup={"rear_wing_angle_deg": 8.0},  # ontology leaf name
        aero_surface=surface,
        air_density=1.225,
        static_readouts={
            "setup_static_lf_ride_height_mm": 15.0,
            "setup_static_rf_ride_height_mm": 15.0,
            "setup_static_lr_ride_height_mm": 40.0,
            "setup_static_rr_ride_height_mm": 40.0,
        },
    )
    assert feats["aero_map_ld_ratio"] > 2.0
    assert 0.0 < feats["aero_map_balance_pct"] < 100.0


def test_archetype_dict_from_row_after_attach() -> None:
    from racingoptimizer.physics.corner_schedule import (
        ARCHETYPE_KEYS,
        archetype_dict_from_row,
    )
    from racingoptimizer.physics.fitter import _attach_corner_archetypes

    frame = pl.DataFrame(
        {
            "session_id": ["s1", "s1"],
            "corner_id": [1, 1],
            "phase": ["mid_corner", "exit"],
            "speed_min_ms": [30.0, 28.0],
            "speed_max_ms": [55.0, 50.0],
            "accel_lat_g_max": [1.5, 1.2],
            "t_start_s": [0.0, 2.0],
            "t_end_s": [2.0, 4.0],
        }
    )
    attached = _attach_corner_archetypes(frame)
    row = attached.to_dicts()[0]
    arch = archetype_dict_from_row(row)
    assert "corner_apex_speed_ms" in arch
    for key in ARCHETYPE_KEYS:
        if key in arch:
            assert arch[key] >= 0.0
