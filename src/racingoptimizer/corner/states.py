"""Corner-phase aggregator: `segment_lap` + `corner_phase_states`.

Spec: docs/superpowers/specs/2026-04-28-corner-phase-design.md (slice B-3).

`segment_lap` is a pure function that labels every sample of a single lap
with its `corner_id` and `phase` by chaining the U4 building blocks
(`detect_corners` + `assign_phases`).

`corner_phase_states` is the public aggregator that pulls one lap from
slice A's parquet via `lap_data`, runs `segment_lap`, then collapses each
`(corner_id, phase)` group into one wide row of derived physics + per-phase
environmental means.

Channel-name handling: slice A's parquet preserves raw IBT channel names
(`LatAccel`, `LongAccel`, ...). The corner module's public surface is
defined in g-axis terms (`AccelLat`, `AccelLon`) so the synthetic-signal
unit tests stay column-agnostic. `corner_phase_states` performs the
rename internally before handing the frame off to `segment_lap`.

The `track_model` keyword on `segment_lap` is reserved for slice D (Wave 3
unit U8). Passing a non-`None` value today raises `NotImplementedError`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from racingoptimizer.corner.boundaries import assign_phases
from racingoptimizer.corner.config import (
    DEFAULT_THRESHOLDS,
    G_MS2,
    PhaseThresholds,
)
from racingoptimizer.corner.detect import detect_corners
from racingoptimizer.corner.phase import Phase
from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.api import lap_data
from racingoptimizer.ingest.detect import UnknownCarError, normalize_car_key
from racingoptimizer.ingest.paths import (
    catalog_path,
    parquet_path,
    resolve_corpus_root,
)

# Stage-2 empirical understeer-signal coefficients (rad / m·s⁻²).
#
# VISION §3 forbids the textbook bicycle-model `Speed^2` denominator that
# slice B used as a placeholder. Replace it with a per-car linear yaw-deficiency
# proxy: `understeer_signal = SteeringWheelAngle - k(car) * AccelLat`.
#
# The seeded coefficients below are stepping-stones — they are an order-of-
# magnitude calibration anchored to typical GTP steering racks (~0.06 rad of
# wheel angle per m/s² of lateral demand). Stage 3 will refine `k_car`
# empirically per session out of the physics fitter; until then these constants
# keep the column finite and per-car-distinguishable.
STEERING_GEOMETRY_COEFFICIENT: dict[str, float] = {
    "bmw":      0.06,
    "acura":    0.07,
    "cadillac": 0.06,
    "ferrari":  0.065,
    "porsche":  0.065,
}
DEFAULT_STEERING_GEOMETRY_COEFFICIENT: float = 0.065


def steering_geometry_for(car: str | None) -> float:
    """Return the per-car understeer-signal coefficient (rad / m·s⁻²).

    Accepts either a canonical car key (``"bmw"``) — which is what the catalog
    persists on :class:`~racingoptimizer.ingest.catalog.SessionRow.car` — or a
    raw iRacing identifier (``"bmwlmdh"``, ``"acuraarx06gtp"``) which is then
    normalised via :func:`~racingoptimizer.ingest.detect.normalize_car_key`.
    Lookups are case-insensitive.

    Falls back to :data:`DEFAULT_STEERING_GEOMETRY_COEFFICIENT` when ``car``
    is ``None``, empty, or does not resolve to a known canonical car key.
    """
    if not car:
        return DEFAULT_STEERING_GEOMETRY_COEFFICIENT
    key = car.strip().lower()
    if key in STEERING_GEOMETRY_COEFFICIENT:
        return STEERING_GEOMETRY_COEFFICIENT[key]
    try:
        key = normalize_car_key(key)
    except UnknownCarError:
        return DEFAULT_STEERING_GEOMETRY_COEFFICIENT
    return STEERING_GEOMETRY_COEFFICIENT.get(key, DEFAULT_STEERING_GEOMETRY_COEFFICIENT)


# Curated default channel pull for `corner_phase_states`. The names match
# slice A's parquet column names (raw IBT channel names + the snake-cased
# columns the writer materialises). The aggregator renames LatAccel/LongAccel
# to AccelLat/AccelLon internally before invoking `segment_lap`.
DEFAULT_CHANNELS: tuple[str, ...] = (
    # core required
    "t_s",
    "lap_index",
    "lap_dist_pct",
    "data_quality_mask",
    "Speed",
    "LatAccel",
    "LongAccel",
    "Brake",
    "Throttle",
    "SteeringWheelAngle",
    # optional dynamics
    "YawRate",
    "Roll",
    "RollRate",
    # shock / ride-height (spec §6 derived state)
    "LFshockDefl",
    "RFshockDefl",
    "LRshockDefl",
    "RRshockDefl",
    "LFrideHeight",
    "RFrideHeight",
    "LRrideHeight",
    "RRrideHeight",
    # per-wheel speed (spec section 6 traction utilisation).
    "LFspeed",
    "RFspeed",
    "LRspeed",
    "RRspeed",
    # VISION section 10 12-channel environment set (spec section 7 EnvironmentFrame contract).
    # Atmospheric floats:
    "AirTemp",
    "AirDensity",
    "AirPressure",
    "RelativeHumidity",
    "WindVel",
    "WindDir",
    "FogLevel",
    # Track surface floats:
    "TrackTempCrew",
    "TrackWetness",
    # Discrete weather state (bool/int channels):
    "WeatherDeclaredWet",
    "Precipitation",
    "Skies",
)


# Phase ordering for the output sort. STRAIGHT lives between corners but
# we still rank it last per (corner_id, phase) so the in-corner phases lead.
_PHASE_ORDER: dict[str, int] = {
    Phase.BRAKING.value: 0,
    Phase.TRAIL_BRAKE.value: 1,
    Phase.MID_CORNER.value: 2,
    Phase.EXIT.value: 3,
    Phase.STRAIGHT.value: 4,
}

_REQUIRED_SEGMENT_COLUMNS: tuple[str, ...] = (
    "t_s",
    "AccelLat",
    "Brake",
    "Throttle",
    "SteeringWheelAngle",
)

# (source_channel, output_alias) pairs for the four-corner damper / ride-height
# blocks. Defined once so `_aggregate` and `_empty_frame` stay in lock-step on
# which alias is emitted for which source.
_SHOCK_DEFL_P99_COLUMNS: tuple[tuple[str, str], ...] = (
    ("LFshockDefl", "lf_shock_defl_p99_mm"),
    ("RFshockDefl", "rf_shock_defl_p99_mm"),
    ("LRshockDefl", "lr_shock_defl_p99_mm"),
    ("RRshockDefl", "rr_shock_defl_p99_mm"),
)
_RIDE_HEIGHT_MEAN_COLUMNS: tuple[tuple[str, str], ...] = (
    ("LFrideHeight", "lf_ride_height_mean_mm"),
    ("RFrideHeight", "rf_ride_height_mean_mm"),
    ("LRrideHeight", "lr_ride_height_mean_mm"),
    ("RRrideHeight", "rr_ride_height_mean_mm"),
)


def segment_lap(
    lap_df: pl.DataFrame,
    *,
    thresholds: PhaseThresholds | None = None,
    track_model: Any | None = None,
) -> pl.DataFrame:
    """Label every sample with its corner_id and phase.

    Pure function. Returns a NEW frame with two appended columns:
    - ``corner_id``: ``Int32`` (-1 outside any corner, 0..N-1 inside).
    - ``phase``: ``Utf8`` (one of the ``Phase.value`` strings).

    Required input columns: ``t_s``, ``AccelLat``, ``Brake``, ``Throttle``,
    ``SteeringWheelAngle``. Missing columns raise ``ValueError`` naming the
    missing column.

    The ``track_model`` keyword is reserved for slice D's track-position
    detector. Passing a non-``None`` value raises ``NotImplementedError``.
    """
    if track_model is not None:
        raise NotImplementedError(
            "track_model integration deferred to Wave 3 U8"
        )

    missing = [c for c in _REQUIRED_SEGMENT_COLUMNS if c not in lap_df.columns]
    if missing:
        raise ValueError(
            f"segment_lap: missing required column(s): {', '.join(missing)}"
        )

    th = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    corner_ids = detect_corners(lap_df, thresholds=th)
    phases = assign_phases(lap_df, corner_ids, thresholds=th)

    return lap_df.with_columns(
        pl.Series("corner_id", corner_ids, dtype=pl.Int32),
        pl.Series("phase", phases, dtype=pl.Utf8),
    )


def corner_phase_states(
    session_id: str,
    lap_index: int,
    *,
    channels: list[str] | None = None,
    thresholds: PhaseThresholds | None = None,
    corpus_root: Path | str | None = None,
) -> pl.DataFrame:
    """Aggregate one row per ``(corner_id, phase)`` for a single lap.

    Loads the lap via slice A's :func:`racingoptimizer.ingest.api.lap_data`,
    runs :func:`segment_lap`, then groups by ``(corner_id, phase)`` and
    materialises the columns described in spec §6. Rows with
    ``corner_id == -1`` (samples outside any detected corner) are dropped.

    The ``lap_index == -1`` pre-grid sentinel is rejected up-front; ingest
    flags warm-up samples with that value and they should never reach the
    fitter.
    """
    if lap_index == -1:
        raise ValueError(
            "lap_index=-1 is the pre-grid sentinel; pass a real lap_index"
        )

    # The session lookup gives us both `car` (for the steering-geom coefficient)
    # and the parquet path used by the schema-only column intersect below.
    sess = _get_session(session_id, corpus_root)
    if channels is None:
        # Some IBT recordings (e.g. Acura ARX-06 telemetry) drop the shock
        # deflection set entirely. Filter the curated default against the
        # parquet's actual schema so optional channels stay optional.
        root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
        pq = parquet_path(root, car=sess.car, track=sess.track, session_id=session_id)
        available = set(pl.scan_parquet(pq).collect_schema().names())
        pull = [c for c in DEFAULT_CHANNELS if c in available]
    else:
        pull = list(channels)
    df = lap_data(session_id, lap_index, channels=pull, corpus_root=corpus_root)

    # Rename raw IBT names to the g-axis names that `segment_lap` consumes.
    rename_map: dict[str, str] = {}
    if "LatAccel" in df.columns and "AccelLat" not in df.columns:
        rename_map["LatAccel"] = "AccelLat"
    if "LongAccel" in df.columns and "AccelLon" not in df.columns:
        rename_map["LongAccel"] = "AccelLon"
    if rename_map:
        df = df.rename(rename_map)

    th = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    labeled = segment_lap(df, thresholds=th)

    return _aggregate(labeled, session_id=session_id, lap_index=lap_index, car=sess.car)


def _get_session(session_id: str, corpus_root: Path | str | None) -> cat.SessionRow:
    """Look up the catalog row for ``session_id``, raising on unknown ids."""
    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    with cat.open_catalog(catalog_path(root)) as conn:
        sess = cat.get_session(conn, session_id)
    if sess is None:
        raise KeyError(f"unknown session_id: {session_id}")
    return sess


# ---- internal aggregation ------------------------------------------------


def _aggregate(
    df: pl.DataFrame, *, session_id: str, lap_index: int, car: str | None = None
) -> pl.DataFrame:
    """Collapse a labeled lap into one row per ``(corner_id, phase)``."""
    cols = set(df.columns)

    has = {name: name in cols for name in (
        "lap_dist_pct", "Speed", "AccelLat", "AccelLon", "Brake", "Throttle",
        "SteeringWheelAngle", "YawRate", "Roll", "RollRate",
        "LFshockDefl", "RFshockDefl", "LRshockDefl", "RRshockDefl",
        "LFrideHeight", "RFrideHeight", "LRrideHeight", "RRrideHeight",
        "LFspeed", "RFspeed", "LRspeed", "RRspeed",
        # VISION section 10 12-channel env set.
        "AirTemp", "AirDensity", "AirPressure", "RelativeHumidity",
        "WindVel", "WindDir", "FogLevel",
        "TrackTempCrew", "TrackWetness",
        "WeatherDeclaredWet", "Precipitation", "Skies",
        "data_quality_mask",
    )}

    # Spec §6 derived-column gates. Each conditional block below assumes the
    # whole quad is present — Acura ARX-06 telemetry drops the *shockDefl set
    # entirely so these columns are simply omitted for that car.
    has_shocks = all(
        has[c] for c in ("LFshockDefl", "RFshockDefl", "LRshockDefl", "RRshockDefl")
    )
    has_ride_heights = all(
        has[c] for c in ("LFrideHeight", "RFrideHeight", "LRrideHeight", "RRrideHeight")
    )
    has_wheel_speeds = all(
        has[c] for c in ("LFspeed", "RFspeed", "LRspeed", "RRspeed")
    )

    # Drop out-of-corner samples up front; defaults exclude them per spec.
    inner = df.filter(pl.col("corner_id") != -1)
    if inner.height == 0:
        return _empty_frame(has)

    # Materialise per-sample damper velocity (mm/s) before group_by so the
    # diff stays within the lap. Raw shockDefl is in METERS; * 1000 -> mm/s.
    # The leading sample's diff is null; Polars aggregates ignore nulls.
    if has_shocks:
        dt = pl.col("t_s").diff()
        inner = inner.with_columns(
            [
                ((pl.col(src).diff() / dt).abs() * 1000.0).alias(f"_v_{src}")
                for src in ("LFshockDefl", "RFshockDefl", "LRshockDefl", "RRshockDefl")
            ]
        )

    aggs: list[pl.Expr] = [
        pl.len().cast(pl.UInt32).alias("n_samples"),
        pl.col("t_s").min().cast(pl.Float64).alias("t_start_s"),
        pl.col("t_s").max().cast(pl.Float64).alias("t_end_s"),
    ]

    if has["lap_dist_pct"]:
        aggs.extend(
            [
                pl.col("lap_dist_pct").first().cast(pl.Float32).alias("lap_dist_pct_start"),
                pl.col("lap_dist_pct").last().cast(pl.Float32).alias("lap_dist_pct_end"),
            ]
        )

    if has["Speed"]:
        aggs.extend(
            [
                pl.col("Speed").min().cast(pl.Float32).alias("speed_min_ms"),
                pl.col("Speed").max().cast(pl.Float32).alias("speed_max_ms"),
                pl.col("Speed").mean().cast(pl.Float32).alias("speed_mean_ms"),
            ]
        )

    # AccelLat is required by segment_lap, so always present.
    aggs.extend(
        [
            (pl.col("AccelLat").abs() / G_MS2).max().cast(pl.Float32).alias("accel_lat_g_max"),
            (pl.col("AccelLat") / G_MS2).mean().cast(pl.Float32).alias("accel_lat_g_mean"),
        ]
    )

    if has["AccelLon"]:
        aggs.extend(
            [
                (pl.col("AccelLon") / G_MS2).min().cast(pl.Float32).alias("accel_lon_g_min"),
                (pl.col("AccelLon") / G_MS2).max().cast(pl.Float32).alias("accel_lon_g_max"),
            ]
        )

    aggs.extend(
        [
            pl.col("Brake").max().cast(pl.Float32).alias("brake_max"),
            pl.col("Brake").mean().cast(pl.Float32).alias("brake_mean"),
            pl.col("Throttle").max().cast(pl.Float32).alias("throttle_max"),
            pl.col("Throttle").mean().cast(pl.Float32).alias("throttle_mean"),
            pl.col("SteeringWheelAngle").abs().max().cast(pl.Float32).alias("steering_max_rad"),
            pl.col("SteeringWheelAngle").mean().cast(pl.Float32).alias("steering_mean_rad"),
        ]
    )

    if has["YawRate"]:
        aggs.append(
            pl.col("YawRate").abs().max().cast(pl.Float32).alias("yaw_rate_max_rad_s")
        )
    if has["Roll"]:
        aggs.append(
            pl.col("Roll").abs().max().cast(pl.Float32).alias("roll_max_rad")
        )
        aggs.append(
            pl.col("Roll").mean().cast(pl.Float32).alias("roll_angle_mean_rad")
        )
    if has["RollRate"]:
        aggs.append(
            pl.col("RollRate").abs().max().cast(pl.Float32).alias("roll_rate_max_rad_s")
        )

    # Empirical yaw-deficiency signal (S2.10 — gap #10 in the VISION-completion
    # plan). VISION §3 forbids the textbook bicycle-model `Speed^2` denominator
    # the placeholder used. Stage 2 replaces it with the per-car linear proxy
    #     understeer_signal = SteeringWheelAngle - k(car) * AccelLat
    # where k(car) is seeded in `STEERING_GEOMETRY_COEFFICIENT`. Stage 3 will
    # refine `k_car` empirically per session inside the physics fitter.
    k_car = steering_geometry_for(car)
    understeer_expr = pl.col("SteeringWheelAngle") - k_car * pl.col("AccelLat")
    aggs.append(understeer_expr.mean().cast(pl.Float32).alias("understeer_angle_mean_rad"))

    for src, alias in _SHOCK_DEFL_P99_COLUMNS:
        if has[src]:
            aggs.append(
                pl.col(src).abs().quantile(0.99).cast(pl.Float32).alias(alias)
            )

    for src, alias in _RIDE_HEIGHT_MEAN_COLUMNS:
        if has[src]:
            aggs.append(pl.col(src).mean().cast(pl.Float32).alias(alias))

    # Spec section 6: load-transfer asymmetry. Sign convention:
    #   positive = right-front + left-rear loaded.
    # Raw shockDefl is meters; convert to mm via * 1000.
    if has_shocks:
        asym_mm = (
            (pl.col("LFshockDefl") + pl.col("RRshockDefl"))
            - (pl.col("RFshockDefl") + pl.col("LRshockDefl"))
        ) * 1000.0
        aggs.append(
            asym_mm.mean().cast(pl.Float32).alias("load_transfer_asymmetry_mean")
        )

    # Spec section 6: traction utilisation.
    # (max(*Speed) - min(*Speed)) / max(Speed, eps), clipped to [0, 1].
    if has_wheel_speeds and has["Speed"]:
        wheels = (pl.col("LFspeed"), pl.col("RFspeed"), pl.col("LRspeed"), pl.col("RRspeed"))
        wheel_max = pl.max_horizontal(*wheels)
        wheel_min = pl.min_horizontal(*wheels)
        speed_floor = pl.max_horizontal(pl.col("Speed"), pl.lit(1e-6))
        traction_util = ((wheel_max - wheel_min) / speed_floor).clip(0.0, 1.0)
        aggs.append(
            traction_util.mean().cast(pl.Float32).alias("traction_util_mean")
        )

    # Spec section 6: aero-platform front/rear ride-height means + pitch.
    # Raw rideHeight is meters; convert to mm via * 1000.
    # Sign convention: pitch_mm = rear_rh - front_rh (positive = nose-down rake).
    if has_ride_heights:
        front_rh_mm = ((pl.col("LFrideHeight") + pl.col("RFrideHeight")) / 2.0) * 1000.0
        rear_rh_mm = ((pl.col("LRrideHeight") + pl.col("RRrideHeight")) / 2.0) * 1000.0
        aggs.append(
            front_rh_mm.mean().cast(pl.Float32).alias("aero_platform_front_rh_mean_mm")
        )
        aggs.append(
            rear_rh_mm.mean().cast(pl.Float32).alias("aero_platform_rear_rh_mean_mm")
        )
        aggs.append(
            (rear_rh_mm - front_rh_mm)
            .mean()
            .cast(pl.Float32)
            .alias("aero_platform_pitch_mean_mm")
        )

    # Spec section 6: damper velocities.
    # p99 = max p99 across the four corners; mean = mean of per-corner means.
    # Pre-computed `_v_*shockDefl` columns are already abs(mm/s).
    if has_shocks:
        v_cols = ("_v_LFshockDefl", "_v_RFshockDefl", "_v_LRshockDefl", "_v_RRshockDefl")
        max_p99 = pl.max_horizontal(*[pl.col(c).quantile(0.99) for c in v_cols])
        mean_of_means = pl.mean_horizontal(*[pl.col(c).mean() for c in v_cols])
        aggs.append(max_p99.cast(pl.Float32).alias("damper_velocity_p99_mms"))
        aggs.append(mean_of_means.cast(pl.Float32).alias("damper_velocity_mean_mms"))

    # VISION section 10 12-channel env set. Atmospheric floats aggregate as
    # means; discrete weather state aggregates as max so a transient flag
    # (e.g. a rain start mid-corner) survives the reduction.
    if has["AirTemp"]:
        aggs.append(pl.col("AirTemp").mean().cast(pl.Float32).alias("air_temp_c_mean"))
    if has["AirDensity"]:
        aggs.append(pl.col("AirDensity").mean().cast(pl.Float32).alias("air_density_mean"))
    if has["AirPressure"]:
        aggs.append(
            pl.col("AirPressure").mean().cast(pl.Float32).alias("air_pressure_mbar_mean")
        )
    if has["RelativeHumidity"]:
        aggs.append(
            pl.col("RelativeHumidity").mean().cast(pl.Float32).alias("relative_humidity_mean")
        )
    if has["WindVel"]:
        aggs.append(pl.col("WindVel").mean().cast(pl.Float32).alias("wind_vel_ms_mean"))
    if has["FogLevel"]:
        aggs.append(pl.col("FogLevel").mean().cast(pl.Float32).alias("fog_level_mean"))

    if has["TrackTempCrew"]:
        aggs.append(
            pl.col("TrackTempCrew").mean().cast(pl.Float32).alias("track_temp_c_mean")
        )
    if has["TrackWetness"]:
        aggs.append(
            pl.col("TrackWetness").mean().cast(pl.Float32).alias("track_wetness_mean")
        )

    # Discrete weather flags: max captures any sample that flipped to wet.
    if has["WeatherDeclaredWet"]:
        aggs.append(
            pl.col("WeatherDeclaredWet").max().cast(pl.Boolean).alias("weather_declared_wet_max")
        )
    if has["Precipitation"]:
        aggs.append(pl.col("Precipitation").max().cast(pl.Int32).alias("precip_type_max"))
    if has["Skies"]:
        aggs.append(pl.col("Skies").max().cast(pl.Int32).alias("skies_max"))

    if has["data_quality_mask"]:
        # Cast bool->float before averaging; the placeholder mask is all-True
        # today and slice D wires the real per-sample mask in later. Spec §6
        # column name is `data_quality_clean_frac` (0..1 fraction of clean
        # samples in the phase) — NOT a 0..100 percentage.
        aggs.append(
            pl.col("data_quality_mask")
            .cast(pl.Float32)
            .mean()
            .cast(pl.Float32)
            .alias("data_quality_clean_frac")
        )

    grouped = inner.group_by(["corner_id", "phase"]).agg(aggs)

    # Circular mean of WindDir is not a Polars-native group_by aggregation, so
    # compute it via the mean of unit vectors and atan2 in a separate pass.
    if has["WindDir"]:
        wind = (
            inner.with_columns(
                [
                    pl.col("WindDir").radians().sin().alias("_wd_sin"),
                    pl.col("WindDir").radians().cos().alias("_wd_cos"),
                ]
            )
            .group_by(["corner_id", "phase"])
            .agg(
                [
                    pl.col("_wd_sin").mean().alias("_wd_sin_mean"),
                    pl.col("_wd_cos").mean().alias("_wd_cos_mean"),
                ]
            )
        )
        # atan2(sin_mean, cos_mean) → degrees, mod 360 to land in [0, 360).
        wind = wind.with_columns(
            (
                (
                    pl.arctan2(pl.col("_wd_sin_mean"), pl.col("_wd_cos_mean")).degrees()
                    + 360.0
                )
                % 360.0
            )
            .cast(pl.Float32)
            .alias("wind_dir_deg_mean")
        ).select(["corner_id", "phase", "wind_dir_deg_mean"])
        grouped = grouped.join(wind, on=["corner_id", "phase"], how="left")

    # Derived: duration_s and the broadcast scalars.
    grouped = grouped.with_columns(
        [
            (pl.col("t_end_s") - pl.col("t_start_s")).cast(pl.Float64).alias("duration_s"),
            pl.lit(session_id, dtype=pl.Utf8).alias("session_id"),
            pl.lit(lap_index, dtype=pl.Int32).alias("lap_index"),
        ]
    )

    # Spec §6 edge case: a phase the walker passes through instantaneously is
    # omitted from the output, not emitted with NaN. group_by never emits
    # zero-row groups so this is a belt-and-braces filter.
    grouped = grouped.filter(pl.col("n_samples") > 0)

    phase_order_expr = pl.col("phase").replace_strict(_PHASE_ORDER, return_dtype=pl.Int8)
    grouped = (
        grouped.with_columns(phase_order_expr.alias("_phase_order"))
        .sort(["corner_id", "_phase_order"])
        .drop("_phase_order")
    )

    return grouped.select(_output_columns(grouped.columns))


def _output_columns(present: list[str]) -> list[str]:
    """Pin column order; only emit columns that are actually present."""
    canonical = [
        "session_id",
        "lap_index",
        "corner_id",
        "phase",
        "n_samples",
        "t_start_s",
        "t_end_s",
        "duration_s",
        "lap_dist_pct_start",
        "lap_dist_pct_end",
        "speed_min_ms",
        "speed_max_ms",
        "speed_mean_ms",
        "accel_lat_g_max",
        "accel_lat_g_mean",
        "accel_lon_g_min",
        "accel_lon_g_max",
        "brake_max",
        "brake_mean",
        "throttle_max",
        "throttle_mean",
        "steering_max_rad",
        "steering_mean_rad",
        "yaw_rate_max_rad_s",
        "roll_max_rad",
        "roll_angle_mean_rad",
        "roll_rate_max_rad_s",
        "understeer_angle_mean_rad",
        "load_transfer_asymmetry_mean",
        "traction_util_mean",
        "lf_shock_defl_p99_mm",
        "rf_shock_defl_p99_mm",
        "lr_shock_defl_p99_mm",
        "rr_shock_defl_p99_mm",
        "lf_ride_height_mean_mm",
        "rf_ride_height_mean_mm",
        "lr_ride_height_mean_mm",
        "rr_ride_height_mean_mm",
        "aero_platform_front_rh_mean_mm",
        "aero_platform_rear_rh_mean_mm",
        "aero_platform_pitch_mean_mm",
        "damper_velocity_p99_mms",
        "damper_velocity_mean_mms",
        # VISION section 10 12-channel env aggregates.
        "air_temp_c_mean",
        "air_density_mean",
        "air_pressure_mbar_mean",
        "relative_humidity_mean",
        "wind_vel_ms_mean",
        "wind_dir_deg_mean",
        "fog_level_mean",
        "track_temp_c_mean",
        "track_wetness_mean",
        "weather_declared_wet_max",
        "precip_type_max",
        "skies_max",
        "data_quality_clean_frac",
    ]
    present_set = set(present)
    return [c for c in canonical if c in present_set]


def _empty_frame(has: dict[str, bool]) -> pl.DataFrame:
    """Empty frame with the canonical schema for the columns that would have
    been emitted given the input's channel coverage."""
    schema: dict[str, pl.DataType] = {
        "session_id": pl.Utf8,
        "lap_index": pl.Int32,
        "corner_id": pl.Int32,
        "phase": pl.Utf8,
        "n_samples": pl.UInt32,
        "t_start_s": pl.Float64,
        "t_end_s": pl.Float64,
        "duration_s": pl.Float64,
        "accel_lat_g_max": pl.Float32,
        "accel_lat_g_mean": pl.Float32,
        "brake_max": pl.Float32,
        "brake_mean": pl.Float32,
        "throttle_max": pl.Float32,
        "throttle_mean": pl.Float32,
        "steering_max_rad": pl.Float32,
        "steering_mean_rad": pl.Float32,
        "understeer_angle_mean_rad": pl.Float32,
    }
    if has["lap_dist_pct"]:
        schema["lap_dist_pct_start"] = pl.Float32
        schema["lap_dist_pct_end"] = pl.Float32
    if has["Speed"]:
        schema["speed_min_ms"] = pl.Float32
        schema["speed_max_ms"] = pl.Float32
        schema["speed_mean_ms"] = pl.Float32
    if has["AccelLon"]:
        schema["accel_lon_g_min"] = pl.Float32
        schema["accel_lon_g_max"] = pl.Float32
    if has["YawRate"]:
        schema["yaw_rate_max_rad_s"] = pl.Float32
    if has["Roll"]:
        schema["roll_max_rad"] = pl.Float32
        schema["roll_angle_mean_rad"] = pl.Float32
    if has["RollRate"]:
        schema["roll_rate_max_rad_s"] = pl.Float32
    for src, alias in _SHOCK_DEFL_P99_COLUMNS + _RIDE_HEIGHT_MEAN_COLUMNS:
        if has[src]:
            schema[alias] = pl.Float32

    # Spec section 6 derived columns gated by the full four-corner channel quad.
    has_shocks_e = all(
        has[c] for c in ("LFshockDefl", "RFshockDefl", "LRshockDefl", "RRshockDefl")
    )
    has_ride_heights_e = all(
        has[c] for c in ("LFrideHeight", "RFrideHeight", "LRrideHeight", "RRrideHeight")
    )
    has_wheel_speeds_e = all(
        has[c] for c in ("LFspeed", "RFspeed", "LRspeed", "RRspeed")
    )
    if has_shocks_e:
        schema["load_transfer_asymmetry_mean"] = pl.Float32
        schema["damper_velocity_p99_mms"] = pl.Float32
        schema["damper_velocity_mean_mms"] = pl.Float32
    if has_wheel_speeds_e and has["Speed"]:
        schema["traction_util_mean"] = pl.Float32
    if has_ride_heights_e:
        schema["aero_platform_front_rh_mean_mm"] = pl.Float32
        schema["aero_platform_rear_rh_mean_mm"] = pl.Float32
        schema["aero_platform_pitch_mean_mm"] = pl.Float32
    if has["AirTemp"]:
        schema["air_temp_c_mean"] = pl.Float32
    if has["AirDensity"]:
        schema["air_density_mean"] = pl.Float32
    if has["AirPressure"]:
        schema["air_pressure_mbar_mean"] = pl.Float32
    if has["RelativeHumidity"]:
        schema["relative_humidity_mean"] = pl.Float32
    if has["WindVel"]:
        schema["wind_vel_ms_mean"] = pl.Float32
    if has["WindDir"]:
        schema["wind_dir_deg_mean"] = pl.Float32
    if has["FogLevel"]:
        schema["fog_level_mean"] = pl.Float32
    if has["TrackTempCrew"]:
        schema["track_temp_c_mean"] = pl.Float32
    if has["TrackWetness"]:
        schema["track_wetness_mean"] = pl.Float32
    if has["WeatherDeclaredWet"]:
        schema["weather_declared_wet_max"] = pl.Boolean
    if has["Precipitation"]:
        schema["precip_type_max"] = pl.Int32
    if has["Skies"]:
        schema["skies_max"] = pl.Int32
    if has["data_quality_mask"]:
        schema["data_quality_clean_frac"] = pl.Float32

    ordered = _output_columns(list(schema.keys()))
    return pl.DataFrame(schema={k: schema[k] for k in ordered})


__all__ = [
    "DEFAULT_CHANNELS",
    "DEFAULT_STEERING_GEOMETRY_COEFFICIENT",
    "STEERING_GEOMETRY_COEFFICIENT",
    "corner_phase_states",
    "segment_lap",
    "steering_geometry_for",
]
