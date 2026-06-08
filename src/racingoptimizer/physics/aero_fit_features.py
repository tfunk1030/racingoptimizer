"""Aero-map features for per-car surrogate fit + predict (W6 P3).

At fit time each training row carries observed ``aero_platform_*_rh_mean_mm``
from telemetry; we query the car's aero map at those ride heights + wing +
air density and append ``aero_map_ld_ratio`` / ``aero_map_balance_pct`` to the
joint feature vector so grip-balance channels are not asked to learn downforce
implicitly from setup alone.

At predict time telemetry RH is unavailable; we approximate platform RH from
deterministic static-RH readouts (``predict_setup_readouts`` / kinematic fit)
as the map query point. This is intentionally conservative -- the surrogate
still owns the dynamic-RH output channels; the aero features supply a
physics-structured prior on downforce trim.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from racingoptimizer.physics.ontology import setup_value

if TYPE_CHECKING:
    from racingoptimizer.aero.interpolator import AeroSurface

_AERO_FIT_COLUMNS: tuple[str, ...] = (
    "aero_map_ld_ratio",
    "aero_map_balance_pct",
)

_AIR_DENSITY_REF: float = 1.225


def aero_fit_column_names() -> tuple[str, ...]:
    return _AERO_FIT_COLUMNS


def _wing_from_setup(car: str, setup: dict, row: dict | None) -> float | None:
    if row is not None:
        for key in ("rear_wing_angle_deg", "rear_wing_deg"):
            v = row.get(key)
            if v is not None:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fv):
                    return fv
    for name in ("rear_wing_angle_deg",):
        v = setup_value(car, name, setup)
        if v is not None:
            return float(v)
    return None


def _air_density_from_row(row: dict) -> float:
    for key in ("air_density", "AirDensity"):
        v = row.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv) and fv > 0.0:
            return fv
    return _AIR_DENSITY_REF


def _query_aero_map(
    aero_surface: AeroSurface,
    *,
    front_rh_mm: float,
    rear_rh_mm: float,
    wing_deg: float,
    air_density: float,
) -> tuple[float, float] | None:
    try:
        balance_pct, ld_ratio = aero_surface.interpolate(
            front_rh_mm, rear_rh_mm, wing_deg, air_density,
        )
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(balance_pct) and np.isfinite(ld_ratio)):
        return None
    if ld_ratio <= 0.0:
        return None
    return float(ld_ratio), float(balance_pct)


def attach_aero_map_features(
    frame: pl.DataFrame,
    car: str,
    setups_by_session: dict[str, dict],
    aero_surface: AeroSurface | None,
) -> pl.DataFrame:
    """Append ``aero_map_*`` columns to a training joint frame."""
    if aero_surface is None or frame.height == 0:
        return frame.with_columns(
            pl.lit(0.0).alias("aero_map_ld_ratio"),
            pl.lit(0.0).alias("aero_map_balance_pct"),
        )
    required = {
        "session_id",
        "aero_platform_front_rh_mean_mm",
        "aero_platform_rear_rh_mean_mm",
    }
    if not required.issubset(set(frame.columns)):
        return frame.with_columns(
            pl.lit(0.0).alias("aero_map_ld_ratio"),
            pl.lit(0.0).alias("aero_map_balance_pct"),
        )

    default_wing = float(
        aero_surface.bounds.wing_angles[len(aero_surface.bounds.wing_angles) // 2]
    )
    ld_vals: list[float] = []
    bal_vals: list[float] = []
    for row in frame.to_dicts():
        sid = str(row.get("session_id") or "")
        setup = setups_by_session.get(sid, {})
        front = float(row.get("aero_platform_front_rh_mean_mm") or 0.0)
        rear = float(row.get("aero_platform_rear_rh_mean_mm") or 0.0)
        wing = _wing_from_setup(car, setup, row) or default_wing
        rho = _air_density_from_row(row)
        hit = _query_aero_map(
            aero_surface,
            front_rh_mm=front,
            rear_rh_mm=rear,
            wing_deg=wing,
            air_density=rho,
        )
        if hit is None:
            ld_vals.append(0.0)
            bal_vals.append(0.0)
        else:
            ld_vals.append(hit[0])
            bal_vals.append(hit[1])
    return frame.with_columns(
        pl.Series("aero_map_ld_ratio", ld_vals, dtype=pl.Float64),
        pl.Series("aero_map_balance_pct", bal_vals, dtype=pl.Float64),
    )


def aero_map_features_for_predict(
    *,
    car: str,
    setup: dict[str, float],
    aero_surface: AeroSurface | None,
    air_density: float,
    static_readouts: dict[str, float] | None = None,
) -> dict[str, float]:
    """Approximate aero-map features at predict time from setup readouts."""
    if aero_surface is None:
        return {"aero_map_ld_ratio": 0.0, "aero_map_balance_pct": 0.0}
    readouts = static_readouts or {}
    front = (
        readouts.get("setup_static_lf_ride_height_mm"),
        readouts.get("setup_static_rf_ride_height_mm"),
    )
    rear = (
        readouts.get("setup_static_lr_ride_height_mm"),
        readouts.get("setup_static_rr_ride_height_mm"),
    )
    front_vals = [float(v) for v in front if v is not None and np.isfinite(v)]
    rear_vals = [float(v) for v in rear if v is not None and np.isfinite(v)]
    if not front_vals or not rear_vals:
        return {"aero_map_ld_ratio": 0.0, "aero_map_balance_pct": 0.0}
    front_rh = sum(front_vals) / len(front_vals)
    rear_rh = sum(rear_vals) / len(rear_vals)
    wing = _wing_from_setup(car, setup, None)
    if wing is None:
        wing = float(
            aero_surface.bounds.wing_angles[len(aero_surface.bounds.wing_angles) // 2]
        )
    hit = _query_aero_map(
        aero_surface,
        front_rh_mm=front_rh,
        rear_rh_mm=rear_rh,
        wing_deg=wing,
        air_density=air_density if air_density > 0.0 else _AIR_DENSITY_REF,
    )
    if hit is None:
        return {"aero_map_ld_ratio": 0.0, "aero_map_balance_pct": 0.0}
    return {"aero_map_ld_ratio": hit[0], "aero_map_balance_pct": hit[1]}


__all__ = [
    "aero_fit_column_names",
    "aero_map_features_for_predict",
    "attach_aero_map_features",
]
