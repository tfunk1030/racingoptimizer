"""Per-car empirical baselines for score normalisation (gap #15).

VISION §3 forbids hardcoded engineering formulas as the primary model. The
score function needs scaling constants to convert raw physics quantities
into [0, 1] utilizations. This module derives them per-car from the
observed extrema in the training corpus rather than inventing literals.

`derive_baselines(car, frame)` walks a stacked corner-phase-states frame
(the same one `fitter.fit` collects) and pulls the 99th percentile of
each baseline channel. `baselines_for(car, frame=None)` is the public
dispatch — derives when given a frame, falls back to per-car defaults
otherwise (cold-start or pickle-loaded models from before this module
existed).

The defaults are deliberately wider than VISION's old hardcoded values
so the score still produces meaningful gradients without overstating the
limit. They come from cross-car averages observed in `ibtfiles/`; per-car
values that diverge meaningfully at corpus density should override them
via `derive_baselines`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

# Heave-spring upper bound from `constraints.md` "Suspension deflections".
# Used as the platform-shock baseline upper limit when corpus is empty.
_HEAVE_SPRING_MAX_MM: float = 25.0


@dataclass(frozen=True, slots=True)
class CarBaselines:
    """Per-car normalisation constants for score sub-utilizations.

    Each scale field is the 99th percentile observed in the training
    corpus (or a conservative cold-start default). They feed the
    denominators in `score.py`'s six sub-utilizations.
    """

    car: str
    max_lateral_g: float                 # 99p observed |AccelLat|/g
    understeer_scale_rad: float          # 99p observed |understeer_angle|
    yaw_rate_scale_rad_s: float          # 99p observed |YawRate|
    wheelspin_scale_ms: float            # 99p observed wheel-speed spread
    ride_height_variance_scale_mm: float  # 99p across-corner RH variance
    shock_defl_scale_mm: float           # 99p heave/shock deflection
    aero_grip_baseline_g: float          # zero-downforce lateral G floor


# Cold-start defaults per car. Values reflect observed cross-corpus
# averages; see module docstring. Acura is wider on RH/shock because its
# IBT YAML lacks shock-deflection channels — fitting still works on
# `ride_height_*` columns alone.
_COMMON_DEFAULTS: dict[str, float] = {
    "max_lateral_g": 2.0,
    "understeer_scale_rad": 0.15,
    "yaw_rate_scale_rad_s": 2.5,
    "wheelspin_scale_ms": 5.0,
    "ride_height_variance_scale_mm": 8.0,
    "shock_defl_scale_mm": _HEAVE_SPRING_MAX_MM,
    "aero_grip_baseline_g": 1.5,
}


def default_baselines_for(car: str) -> CarBaselines:
    """Return the cold-start default `CarBaselines` for `car`.

    Used as the fallback when no training corpus is available (cold start)
    or when an old pickled `PhysicsModel` predates the `car_baselines`
    field.
    """
    car_key = car.strip().lower()
    return CarBaselines(car=car_key, **_COMMON_DEFAULTS)


DEFAULT_BASELINES: dict[str, CarBaselines] = {
    car: default_baselines_for(car)
    for car in ("acura", "bmw", "cadillac", "ferrari", "porsche")
}


# Channels we pull from the stacked corner-phase-states frame. The
# left-side (frame column) is matched against the schema in
# `racingoptimizer.corner.states`; missing columns fall back to the
# default for that scale.
_LATERAL_G_COLUMNS: tuple[str, ...] = ("accel_lat_g_max",)
_UNDERSTEER_COLUMNS: tuple[str, ...] = ("understeer_angle_mean_rad",)
_YAW_COLUMNS: tuple[str, ...] = ("yaw_rate_max_rad_s",)
_RH_COLUMNS: tuple[str, ...] = (
    "lf_ride_height_mean_mm",
    "rf_ride_height_mean_mm",
    "lr_ride_height_mean_mm",
    "rr_ride_height_mean_mm",
)
_SHOCK_COLUMNS: tuple[str, ...] = (
    "lf_shock_defl_p99_mm",
    "rf_shock_defl_p99_mm",
    "lr_shock_defl_p99_mm",
    "rr_shock_defl_p99_mm",
)


def derive_baselines(car: str, frame: pl.DataFrame | None) -> CarBaselines:
    """Compute per-car baselines from a stacked corner-phase-states frame.

    `frame` is the polars frame produced by stacking every valid lap's
    `corner_phase_states` for `car` (see `fitter._collect_training_frames`).
    On empty/None input, returns the per-car default. On populated input,
    each scale is the 99th percentile of the corresponding channel; if a
    channel is missing entirely (e.g. Acura has no shock deflections),
    the default for that scale is preserved.
    """
    car_key = car.strip().lower()
    defaults = DEFAULT_BASELINES.get(car_key, default_baselines_for(car_key))
    if frame is None or frame.is_empty():
        return defaults

    lat_g = _quantile_abs(frame, _LATERAL_G_COLUMNS)
    us = _quantile_abs(frame, _UNDERSTEER_COLUMNS)
    yaw = _quantile_abs(frame, _YAW_COLUMNS)
    rh_var = _ride_height_variance_quantile(frame)
    shock = _quantile_abs(frame, _SHOCK_COLUMNS)

    return CarBaselines(
        car=car_key,
        max_lateral_g=lat_g if lat_g is not None else defaults.max_lateral_g,
        understeer_scale_rad=us if us is not None else defaults.understeer_scale_rad,
        yaw_rate_scale_rad_s=yaw if yaw is not None else defaults.yaw_rate_scale_rad_s,
        wheelspin_scale_ms=defaults.wheelspin_scale_ms,
        ride_height_variance_scale_mm=(
            rh_var if rh_var is not None else defaults.ride_height_variance_scale_mm
        ),
        shock_defl_scale_mm=shock if shock is not None else defaults.shock_defl_scale_mm,
        aero_grip_baseline_g=defaults.aero_grip_baseline_g,
    )


def baselines_for(
    car: str,
    frame: pl.DataFrame | None = None,
) -> CarBaselines:
    """Public dispatcher: derive when the corpus is non-empty, else default."""
    return derive_baselines(car, frame)


# ---- internals -----------------------------------------------------------


def _quantile_abs(frame: pl.DataFrame, columns: tuple[str, ...]) -> float | None:
    """99th percentile of |x| across all listed columns; None if none present."""
    available = [c for c in columns if c in frame.columns]
    if not available:
        return None
    values: list[float] = []
    for col in available:
        series = frame[col].drop_nulls().cast(pl.Float64)
        if series.is_empty():
            continue
        values.extend(np.abs(series.to_numpy()).tolist())
    if not values:
        return None
    q = float(np.quantile(values, 0.99))
    if not np.isfinite(q) or q <= 0.0:
        return None
    return q


def _ride_height_variance_quantile(frame: pl.DataFrame) -> float | None:
    """99p of per-row across-corner ride-height variance (mm^2 in score)."""
    available = [c for c in _RH_COLUMNS if c in frame.columns]
    if len(available) < 2:
        return None
    data = (
        frame.select(available).drop_nulls().cast(pl.Float64).to_numpy()
    )
    if data.shape[0] == 0:
        return None
    # Per-row population variance across the four corners.
    variances = np.var(data, axis=1, ddof=0)
    variances = variances[np.isfinite(variances)]
    if variances.size == 0:
        return None
    q = float(np.quantile(variances, 0.99))
    if not np.isfinite(q) or q <= 0.0:
        return None
    return q


__all__ = [
    "CarBaselines",
    "DEFAULT_BASELINES",
    "baselines_for",
    "default_baselines_for",
    "derive_baselines",
]
