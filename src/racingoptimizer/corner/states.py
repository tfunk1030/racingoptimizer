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
from racingoptimizer.ingest.paths import (
    catalog_path,
    parquet_path,
    resolve_corpus_root,
)

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
    # environment quintet (spec §7 EnvironmentFrame contract)
    "AirDensity",
    "TrackTempCrew",
    "WindVel",
    "WindDir",
    "TrackWetness",
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

    if channels is None:
        # Some IBT recordings (e.g. Acura ARX-06 telemetry) drop the shock
        # deflection set entirely. Filter the curated default against the
        # parquet's actual schema so optional channels stay optional.
        available = _parquet_columns(session_id, corpus_root)
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

    return _aggregate(labeled, session_id=session_id, lap_index=lap_index)


def _parquet_columns(session_id: str, corpus_root: Path | str | None) -> set[str]:
    """Cheap schema-only lookup of the columns persisted for ``session_id``."""
    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    with cat.open_catalog(catalog_path(root)) as conn:
        sess = cat.get_session(conn, session_id)
        if sess is None:
            raise KeyError(f"unknown session_id: {session_id}")
    pq = parquet_path(root, car=sess.car, track=sess.track, session_id=session_id)
    return set(pl.scan_parquet(pq).collect_schema().names())


# ---- internal aggregation ------------------------------------------------


def _aggregate(
    df: pl.DataFrame, *, session_id: str, lap_index: int
) -> pl.DataFrame:
    """Collapse a labeled lap into one row per ``(corner_id, phase)``."""
    cols = set(df.columns)

    has = {name: name in cols for name in (
        "lap_dist_pct", "Speed", "AccelLat", "AccelLon", "Brake", "Throttle",
        "SteeringWheelAngle", "YawRate", "Roll", "RollRate",
        "LFshockDefl", "RFshockDefl", "LRshockDefl", "RRshockDefl",
        "LFrideHeight", "RFrideHeight", "LRrideHeight", "RRrideHeight",
        "AirDensity", "TrackTempCrew", "WindVel", "WindDir", "TrackWetness",
        "data_quality_mask",
    )}

    # Drop out-of-corner samples up front; defaults exclude them per spec.
    inner = df.filter(pl.col("corner_id") != -1)
    if inner.height == 0:
        return _empty_frame(has)

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
    if has["RollRate"]:
        aggs.append(
            pl.col("RollRate").abs().max().cast(pl.Float32).alias("roll_rate_max_rad_s")
        )

    # Understeer angle: SteeringWheelAngle - 1.0 * AccelLat / max(Speed^2, 1.0)
    # `1.0` is the steering_geom placeholder spec §6 / open-question 4 calls out;
    # slice E will fit the per-car coefficient.
    if has["Speed"]:
        denom = pl.max_horizontal(pl.col("Speed").pow(2), pl.lit(1.0))
        understeer_expr = pl.col("SteeringWheelAngle") - 1.0 * pl.col("AccelLat") / denom
    else:
        understeer_expr = pl.col("SteeringWheelAngle") - 1.0 * pl.col("AccelLat")
    aggs.append(understeer_expr.mean().cast(pl.Float32).alias("understeer_angle_mean_rad"))

    for src, alias in _SHOCK_DEFL_P99_COLUMNS:
        if has[src]:
            aggs.append(
                pl.col(src).abs().quantile(0.99).cast(pl.Float32).alias(alias)
            )

    for src, alias in _RIDE_HEIGHT_MEAN_COLUMNS:
        if has[src]:
            aggs.append(pl.col(src).mean().cast(pl.Float32).alias(alias))

    if has["AirDensity"]:
        aggs.append(pl.col("AirDensity").mean().cast(pl.Float32).alias("air_density_mean"))
    if has["TrackTempCrew"]:
        aggs.append(
            pl.col("TrackTempCrew").mean().cast(pl.Float32).alias("track_temp_c_mean")
        )
    if has["WindVel"]:
        aggs.append(pl.col("WindVel").mean().cast(pl.Float32).alias("wind_vel_ms_mean"))

    if has["TrackWetness"]:
        aggs.append(
            pl.col("TrackWetness").mean().cast(pl.Float32).alias("track_wetness_mean")
        )

    if has["data_quality_mask"]:
        # Cast bool->float before averaging; the placeholder mask is all-True
        # today and slice D wires the real per-sample mask in later.
        aggs.append(
            (pl.col("data_quality_mask").cast(pl.Float32).mean() * 100.0)
            .cast(pl.Float32)
            .alias("data_quality_pct")
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
        "roll_rate_max_rad_s",
        "understeer_angle_mean_rad",
        "lf_shock_defl_p99_mm",
        "rf_shock_defl_p99_mm",
        "lr_shock_defl_p99_mm",
        "rr_shock_defl_p99_mm",
        "lf_ride_height_mean_mm",
        "rf_ride_height_mean_mm",
        "lr_ride_height_mean_mm",
        "rr_ride_height_mean_mm",
        "air_density_mean",
        "track_temp_c_mean",
        "wind_vel_ms_mean",
        "wind_dir_deg_mean",
        "track_wetness_mean",
        "data_quality_pct",
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
    if has["RollRate"]:
        schema["roll_rate_max_rad_s"] = pl.Float32
    for src, alias in _SHOCK_DEFL_P99_COLUMNS + _RIDE_HEIGHT_MEAN_COLUMNS:
        if has[src]:
            schema[alias] = pl.Float32
    if has["AirDensity"]:
        schema["air_density_mean"] = pl.Float32
    if has["TrackTempCrew"]:
        schema["track_temp_c_mean"] = pl.Float32
    if has["WindVel"]:
        schema["wind_vel_ms_mean"] = pl.Float32
    if has["WindDir"]:
        schema["wind_dir_deg_mean"] = pl.Float32
    if has["TrackWetness"]:
        schema["track_wetness_mean"] = pl.Float32
    if has["data_quality_mask"]:
        schema["data_quality_pct"] = pl.Float32

    ordered = _output_columns(list(schema.keys()))
    return pl.DataFrame(schema={k: schema[k] for k in ordered})


__all__ = ["DEFAULT_CHANNELS", "corner_phase_states", "segment_lap"]
