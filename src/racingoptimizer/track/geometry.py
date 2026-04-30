"""Per-bin elevation / camber proxies (S4.4, VISION §9).

VISION §9 lists "elevation changes and camber (derived from
lateral/longitudinal G vs steering angle relationships)" as part of every
track's surface characterisation. The IBT may not expose explicit
elevation or camber channels — and even when it does, the exposed values
depend on the iRacing build. This module derives both proxies from the
60 Hz accelerometer / steering signals that every IBT carries.

Elevation gradient proxy
    If the IBT exposes ``AccelVert`` (rare), use its absolute value as a
    direct hill-acceleration signal. Otherwise compute a residual:
    ``AccelLon - expected(AccelLon | Speed)`` where the expected long-G
    is a linear regression of ``LongAccel`` against ``Speed`` fit across
    every sample of the corpus. The fit captures the speed-dependent
    drag + braking baseline; the residual is what the baseline cannot
    explain — most plausibly hill crests (large negative residual: car
    accelerating uphill less than expected) and dips (large positive
    residual: car accelerating downhill more than expected). Aggregated
    per ``track_pos_m`` bin as the median of per-sample absolute
    residuals across every clean lap.

Camber ratio proxy
    For samples in the mid-corner phase (peak lateral load), compute
    ``observed_lat_g / expected_lat_g_for_steering`` where the expected
    value comes from a linear fit of ``|LatAccel| / g`` against
    ``|SteeringWheelAngle|`` across every mid-corner sample of the
    corpus. A ratio > 1.0 at a bin means the car is generating MORE
    lateral G than the steering input alone explains — banking is
    helping. A ratio < 1.0 at a bin means the steering produces less
    lateral G than expected — adverse camber or off-camber surface.
    Aggregated per bin as the median of per-sample ratios across every
    clean mid-corner lap sample. Bins with no mid-corner samples
    receive a NaN ratio so callers can distinguish "no data" from
    "neutral camber".

Cold-start (< 3 sessions) returns an empty frame — a single session's
fits are too noisy for the residuals to mean anything.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from racingoptimizer.corner.boundaries import assign_phases
from racingoptimizer.corner.config import G_MS2
from racingoptimizer.corner.detect import detect_corners
from racingoptimizer.corner.phase import Phase
from racingoptimizer.ingest import api as ingest_api
from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.paths import (
    catalog_path,
    parquet_path,
    resolve_corpus_root,
)
from racingoptimizer.track.bins import DEFAULT_BIN_SIZE_M, bin_index, track_pos_m_from_pct

_LOG = logging.getLogger(__name__)
_COLD_START_THRESHOLD = 3

# Slice A's parquet preserves raw IBT names (LongAccel / LatAccel); the
# corner phase machine consumes AccelLat. We pull the raw names here and
# rename LatAccel -> AccelLat before handing to assign_phases. data_quality_mask
# is reserved by slice A and rewritten by slice D's apply_quality_mask;
# pulling it lets us drop curb / off-track samples before the corpus-wide fits.
_NEEDED_CHANNELS: tuple[str, ...] = (
    "lap_dist_pct",
    "Speed",
    "LongAccel",
    "LatAccel",
    "Brake",
    "Throttle",
    "SteeringWheelAngle",
    "data_quality_mask",
)
_VERT_CHANNEL = "AccelVert"

GEOMETRY_SCHEMA: dict[str, type[pl.DataType]] = {
    "bin_index": pl.Int32,
    "track_pos_m": pl.Float64,
    "elevation_gradient_proxy": pl.Float64,
    "camber_ratio_proxy": pl.Float64,
    "n_samples": pl.Int64,
    "n_sessions": pl.Int64,
}


def empty_geometry_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=GEOMETRY_SCHEMA)


def compute_track_geometry(
    track: str,
    session_ids: list[str],
    *,
    bin_size_m: float = DEFAULT_BIN_SIZE_M,
    corpus_root: Path | str | None = None,
) -> pl.DataFrame:
    """Per-bin elevation gradient + camber ratio proxies for a track.

    Walks every session's clean laps once, fits the corpus-wide
    ``LongAccel ~ Speed`` and ``|LatAccel|/g ~ |SteeringWheelAngle|``
    baselines, then aggregates per-bin medians. Cold-start (fewer than
    ``_COLD_START_THRESHOLD = 3`` usable sessions) returns an empty
    frame — the fits below need cross-session coverage to mean anything.
    """
    sorted_ids = sorted(session_ids)
    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    if len(sorted_ids) < _COLD_START_THRESHOLD:
        return empty_geometry_frame()

    samples = _collect_samples(
        sorted_ids, bin_size_m=bin_size_m, corpus_root=root, track=track
    )
    if samples is None or samples.height == 0:
        return empty_geometry_frame()

    n_sessions_used = samples["session_id"].n_unique()
    if n_sessions_used < _COLD_START_THRESHOLD:
        return empty_geometry_frame()

    elev_per_sample = _elevation_residual(samples)
    camber_per_sample = _camber_ratio(samples)

    enriched = samples.with_columns(
        pl.Series("_elev", elev_per_sample, dtype=pl.Float64),
        pl.Series("_camber", camber_per_sample, dtype=pl.Float64),
    )

    elev_per_bin = (
        enriched.group_by("bin_index")
        .agg(
            pl.col("_elev").drop_nulls().median().alias("elevation_gradient_proxy"),
            pl.len().cast(pl.Int64).alias("n_samples"),
            pl.col("session_id").n_unique().cast(pl.Int64).alias("n_sessions"),
        )
    )
    camber_per_bin = (
        enriched.filter(pl.col("_is_mid_corner"))
        .group_by("bin_index")
        .agg(pl.col("_camber").drop_nulls().median().alias("camber_ratio_proxy"))
    )

    out = (
        elev_per_bin.join(camber_per_bin, on="bin_index", how="left")
        .with_columns(
            (pl.col("bin_index").cast(pl.Float64) * bin_size_m).alias("track_pos_m"),
        )
        .select(list(GEOMETRY_SCHEMA.keys()))
        .sort("bin_index", maintain_order=True)
    )
    return out


# ---- internals ----


def _collect_samples(
    session_ids: list[str],
    *,
    bin_size_m: float,
    corpus_root: Path,
    track: str,
) -> pl.DataFrame | None:
    """Stack every clean sample from every session into one long frame.

    Columns: ``session_id, bin_index, Speed, LongAccel, lat_g, steer_abs,
    AccelVert (or NaN), _is_mid_corner``. The mid-corner flag is computed
    once here via the corner-phase state machine so the camber aggregate
    can filter without re-running it.
    """
    from racingoptimizer.track.builder import _lap_length_for_session

    frames: list[pl.DataFrame] = []
    with cat.open_catalog(catalog_path(corpus_root)) as conn:
        for sid in session_ids:
            sess = cat.get_session(conn, sid)
            if sess is None or sess.parquet_path is None:
                continue
            lap_length_m = _lap_length_for_session(conn, sid, corpus_root=corpus_root)
            if lap_length_m is None or lap_length_m <= 0.0:
                continue
            session_frame = _collect_one_session(
                sid,
                car=sess.car,
                bin_size_m=bin_size_m,
                lap_length_m=lap_length_m,
                corpus_root=corpus_root,
                track=track,
            )
            if session_frame.height > 0:
                frames.append(session_frame)
    if not frames:
        return None
    return pl.concat(frames, how="vertical_relaxed")


def _collect_one_session(
    session_id: str,
    *,
    car: str,
    bin_size_m: float,
    lap_length_m: float,
    corpus_root: Path,
    track: str,
) -> pl.DataFrame:
    lap_rows = ingest_api.laps(
        session_id=session_id, valid_only=True, corpus_root=corpus_root
    )
    if lap_rows.height == 0:
        return _empty_collected()

    # Schema-only probe of the parquet so the optional AccelVert and the
    # data_quality_mask column degrade gracefully without paying the
    # cost of materialising one full lap up-front (real Spa laps are
    # ~25 MB each).
    pq = parquet_path(corpus_root, car=car, track=track, session_id=session_id)
    available = set(pl.scan_parquet(pq).collect_schema().names())
    pull = [c for c in _NEEDED_CHANNELS if c in available]
    has_vert = _VERT_CHANNEL in available
    if has_vert:
        pull.append(_VERT_CHANNEL)

    parts: list[pl.DataFrame] = []
    for lap_idx in lap_rows["lap_index"].to_list():
        try:
            df = ingest_api.lap_data(
                session_id, int(lap_idx), channels=pull, corpus_root=corpus_root
            )
        except pl.exceptions.ColumnNotFoundError as exc:
            _LOG.warning(
                "track=%s session=%s: skipped geometry (channel absent: %s)",
                track,
                session_id,
                exc,
            )
            return _empty_collected()
        if df.height == 0:
            continue

        # Slice D's data_quality_mask flips False on curb / off-track samples;
        # those are exactly the samples whose accelerometer signals lie about
        # surface geometry. Drop them before the corpus-wide fit so curbs
        # cannot inflate the elevation residual or the steering slope.
        if "data_quality_mask" in df.columns:
            df = df.filter(pl.col("data_quality_mask"))
        if df.height == 0:
            continue

        # assign_phases consumes AccelLat; slice A's parquet calls it LatAccel.
        if "LatAccel" in df.columns and "AccelLat" not in df.columns:
            df = df.rename({"LatAccel": "AccelLat"})

        # Sanity: the four channels we depend on for the regression / phase
        # detection. Acura is the known Achilles-heel car; it lacks the per-
        # corner shock channels but DOES expose LongAccel + LatAccel + Brake +
        # Throttle + SteeringWheelAngle, so we expect this guard to never fire
        # in practice. Belt-and-braces: if any future car drops one of these,
        # we skip cleanly rather than crashing mid-aggregation.
        required = ("LongAccel", "AccelLat", "Brake", "Throttle", "SteeringWheelAngle")
        if not all(c in df.columns for c in required):
            return _empty_collected()

        corner_ids = detect_corners(df)
        phases = assign_phases(df, corner_ids)
        is_mid = phases == Phase.MID_CORNER.value

        track_pos = track_pos_m_from_pct(df["lap_dist_pct"].to_numpy(), lap_length_m)
        idx = bin_index(track_pos, bin_size_m=bin_size_m)

        n = df.height
        parts.append(
            pl.DataFrame(
                {
                    "session_id": pl.Series("session_id", [session_id] * n, dtype=pl.Utf8),
                    "bin_index": idx.astype(np.int32),
                    "Speed": df["Speed"].to_numpy().astype(np.float64),
                    "LongAccel": df["LongAccel"].to_numpy().astype(np.float64),
                    "lat_g": np.abs(df["AccelLat"].to_numpy()) / G_MS2,
                    "steer_abs": np.abs(df["SteeringWheelAngle"].to_numpy()),
                    "AccelVert": (
                        df[_VERT_CHANNEL].to_numpy().astype(np.float64)
                        if has_vert
                        else np.full(n, np.nan, dtype=np.float64)
                    ),
                    "_is_mid_corner": is_mid,
                }
            ).filter(pl.col("bin_index") >= 0)
        )

    if not parts:
        return _empty_collected()
    return pl.concat(parts, how="vertical_relaxed")


_COLLECTED_SCHEMA: dict[str, type[pl.DataType]] = {
    "session_id": pl.Utf8,
    "bin_index": pl.Int32,
    "Speed": pl.Float64,
    "LongAccel": pl.Float64,
    "lat_g": pl.Float64,
    "steer_abs": pl.Float64,
    "AccelVert": pl.Float64,
    "_is_mid_corner": pl.Boolean,
}


def _empty_collected() -> pl.DataFrame:
    return pl.DataFrame(schema=_COLLECTED_SCHEMA)


def _elevation_residual(samples: pl.DataFrame) -> np.ndarray:
    """Per-sample |elevation gradient proxy| in m/s².

    Uses ``|AccelVert|`` directly when present for any session in the
    corpus. Otherwise fits ``LongAccel = a * Speed + b`` across the whole
    corpus and returns ``|LongAccel - (a * Speed + b)|`` per sample. The
    fit's intercept absorbs the per-car drag baseline; the slope absorbs
    the speed-linear part of drag + power. What's left is whatever the
    linear baseline cannot explain — most plausibly hills.
    """
    vert = samples["AccelVert"].to_numpy().astype(np.float64)
    if np.isfinite(vert).any():
        # Mix of finite and NaN samples: prefer AccelVert where present,
        # fall back to the residual for the rest. In practice the channel
        # is either present for every session or absent for every session,
        # so this is rarely a partial mix — but defensive against the
        # mixed case is cheap.
        out = np.abs(vert)
        if np.isnan(out).any():
            residual = _long_accel_residual(samples)
            mask = np.isnan(out)
            out[mask] = residual[mask]
        return out
    return _long_accel_residual(samples)


def _long_accel_residual(samples: pl.DataFrame) -> np.ndarray:
    speed = samples["Speed"].to_numpy().astype(np.float64)
    long_g = samples["LongAccel"].to_numpy().astype(np.float64)
    finite = np.isfinite(speed) & np.isfinite(long_g)
    if finite.sum() < 2 or np.var(speed[finite]) < 1e-9:
        # Degenerate fit — fall back to absolute LongAccel itself.
        return np.abs(np.where(finite, long_g, 0.0))
    a, b = np.polyfit(speed[finite], long_g[finite], 1)
    expected = a * speed + b
    return np.abs(long_g - expected)


def _camber_ratio(samples: pl.DataFrame) -> np.ndarray:
    """Per-sample lat_g / expected_lat_g_for_steering.

    Fits ``|lat_g| = a * |steering| + b`` across mid-corner samples only
    (peak lateral load) so the slope is dominated by the actual lateral-G
    response of the car rather than the small-steering noise around the
    straights. Samples where the predicted expected lateral-G is below a
    1e-3 g floor (essentially zero steering) get NaN to avoid divide-by-
    near-zero amplifying noise into a bogus 1000× ratio.
    """
    lat_g = samples["lat_g"].to_numpy().astype(np.float64)
    steer = samples["steer_abs"].to_numpy().astype(np.float64)
    is_mid = samples["_is_mid_corner"].to_numpy().astype(bool)

    fit_mask = is_mid & np.isfinite(lat_g) & np.isfinite(steer)
    if fit_mask.sum() < 2 or np.var(steer[fit_mask]) < 1e-9:
        return np.full(samples.height, np.nan, dtype=np.float64)

    a, b = np.polyfit(steer[fit_mask], lat_g[fit_mask], 1)
    expected = a * steer + b
    out = np.full(samples.height, np.nan, dtype=np.float64)
    valid = expected > 1e-3
    out[valid] = lat_g[valid] / expected[valid]
    return out
