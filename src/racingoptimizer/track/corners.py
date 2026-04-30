"""Per-corner braking / apex / exit landmarks (S4.2, VISION §9).

VISION §9 commits to "precise braking points, apex positions, and exit points
for each corner (averaged and refined across hundreds of laps)." This module
computes those three positions for every corner the lap-detector finds, then
collapses across all valid laps in every passed-in session into one row per
corner.

Definitions (per the S4.2 task brief):

- **apex_m** — `track_pos_m` where `|AccelLat|` is largest within a corner.
- **braking_point_m** — first sample within ±50 m of the apex where
  `Brake > 0.05`. Defaults to the corner-window's start position when no
  brake input crosses the threshold (e.g. flat-out kinks).
- **exit_point_m** — first post-apex sample where `Throttle > 0.5` AND
  `|AccelLat|/g < 0.5`. Defaults to the corner-window's end when the lap
  never relaxes lateral load before the next corner starts.

Corner-id stability across laps: per lap, corners are sorted by apex
position and assigned 0..N-1. Cross-lap aggregation groups by that index,
which works as long as a track's corner count is consistent across laps.
We cap to the per-lap minimum so a single noisy lap that splits one corner
into two does not poison the count for everything downstream.

`n_observations` reports how many lap-corner pairs contributed to each
landmark — useful for gating downstream consumers (sparse → don't trust).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from racingoptimizer.corner.config import G_MS2
from racingoptimizer.corner.detect import detect_corners
from racingoptimizer.ingest import api as ingest_api
from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.paths import catalog_path, resolve_corpus_root
from racingoptimizer.track.bins import track_pos_m_from_pct

_LOG = logging.getLogger(__name__)

_BRAKE_THRESHOLD = 0.05
_THROTTLE_THRESHOLD = 0.5
_EXIT_LATERAL_G_THRESHOLD = 0.5
_BRAKING_SEARCH_RADIUS_M = 50.0

_CORNER_LANDMARKS_SCHEMA: dict[str, type[pl.DataType]] = {
    "corner_id": pl.Int32,
    "braking_point_m": pl.Float64,
    "apex_m": pl.Float64,
    "exit_point_m": pl.Float64,
    "n_observations": pl.Int64,
}

# Channels the per-lap landmark scan needs. `LatAccel` is the iRacing raw name
# the writer persists; the corner detector reads it after a rename.
_REQUIRED_CHANNELS: tuple[str, ...] = ("lap_dist_pct", "LatAccel", "Brake", "Throttle")


@dataclass(frozen=True)
class _LapLandmark:
    """One corner, observed in one lap."""

    corner_position_index: int  # 0..N-1 within the lap, position-sorted
    braking_point_m: float
    apex_m: float
    exit_point_m: float


def _empty_landmarks_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_CORNER_LANDMARKS_SCHEMA)


def compute_corner_landmarks(
    track: str,
    session_ids: list[str],
    corpus_root: Path | str | None = None,
) -> pl.DataFrame:
    """Cross-session braking / apex / exit landmarks for every corner on `track`.

    Pulls the per-lap raw telemetry from each session's parquet via
    `ingest.api.lap_data`, detects corners per lap, computes the three
    landmarks per corner per lap, then averages by position-sorted corner
    index across every contributing lap. Returns one row per corner.

    Returns an empty frame when no session yielded a usable lap (e.g. all
    sessions missing one of the required channels).
    """
    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)

    per_lap_records: list[_LapLandmark] = []
    with cat.open_catalog(catalog_path(root)) as conn:
        for sid in session_ids:
            sess = cat.get_session(conn, sid)
            if sess is None or sess.parquet_path is None:
                continue
            lap_length_m = _lap_length_for_session(conn, sid, corpus_root=root)
            if lap_length_m is None or lap_length_m <= 0.0:
                continue
            lap_rows = ingest_api.laps(
                session_id=sid, valid_only=True, corpus_root=root
            )
            if lap_rows.height == 0:
                continue
            for lap_idx in lap_rows["lap_index"].to_list():
                lap_records = _landmarks_for_lap(
                    sid,
                    lap_index=int(lap_idx),
                    lap_length_m=lap_length_m,
                    corpus_root=root,
                    track=track,
                )
                per_lap_records.extend(lap_records)

    if not per_lap_records:
        return _empty_landmarks_frame()

    return _aggregate_across_laps(per_lap_records)


# ---- internals ----


def _landmarks_for_lap(
    session_id: str,
    *,
    lap_index: int,
    lap_length_m: float,
    corpus_root: Path,
    track: str,
) -> list[_LapLandmark]:
    """Detect corners and emit one `_LapLandmark` per corner found in the lap."""
    try:
        df = ingest_api.lap_data(
            session_id,
            lap_index,
            channels=list(_REQUIRED_CHANNELS),
            corpus_root=corpus_root,
        )
    except pl.exceptions.ColumnNotFoundError as exc:
        _LOG.warning(
            "track=%s session=%s lap=%d: skipped corner-landmark scan (%s)",
            track,
            session_id,
            lap_index,
            exc,
        )
        return []
    if df.height == 0:
        return []

    # `detect_corners` reads `AccelLat`; the parquet's IBT-native name is
    # `LatAccel`. Rename in-place rather than persisting two copies.
    df = df.rename({"LatAccel": "AccelLat"})
    corner_ids = detect_corners(df)
    if not np.any(corner_ids >= 0):
        return []

    track_pos = track_pos_m_from_pct(df["lap_dist_pct"].to_numpy(), lap_length_m)
    lat_g = np.abs(df["AccelLat"].to_numpy()) / G_MS2
    brake = df["Brake"].to_numpy()
    throttle = df["Throttle"].to_numpy()

    unique_ids = np.unique(corner_ids[corner_ids >= 0])

    # Build per-corner records, then sort by apex position so the position
    # index is stable across laps regardless of detect_corners' internal
    # numbering convention.
    raw: list[tuple[float, float, float]] = []
    for cid in unique_ids:
        mask = corner_ids == cid
        idxs = np.flatnonzero(mask)
        apex_m, braking_m, exit_m = _landmarks_for_corner(
            idxs, track_pos=track_pos, lat_g=lat_g, brake=brake, throttle=throttle
        )
        raw.append((apex_m, braking_m, exit_m))

    raw.sort(key=lambda r: r[0])
    return [
        _LapLandmark(
            corner_position_index=position_index,
            braking_point_m=braking_m,
            apex_m=apex_m,
            exit_point_m=exit_m,
        )
        for position_index, (apex_m, braking_m, exit_m) in enumerate(raw)
    ]


def _landmarks_for_corner(
    corner_indices: np.ndarray,
    *,
    track_pos: np.ndarray,
    lat_g: np.ndarray,
    brake: np.ndarray,
    throttle: np.ndarray,
) -> tuple[float, float, float]:
    """Apex / braking / exit positions for one detected corner.

    Apex is taken as the absolute lateral-G peak inside the corner window.
    Braking point is the first brake application within ±50 m of the apex
    — searched across the whole lap, not just inside the corner window, so
    a brake release that begins before the lateral-G entry threshold still
    counts. Exit is the first post-apex sample anywhere downstream where
    the throttle is on AND lateral load has dropped below 0.5 g — driver
    re-application of throttle frequently happens after the corner-window
    detector has already closed (lateral G falls past the exit hysteresis).

    Falls back to the corner-window edges when no event crosses its
    threshold (very-low-G kink, throttle-stab corner without brake, etc.).
    """
    apex_within = corner_indices[np.argmax(lat_g[corner_indices])]
    apex_m = float(track_pos[apex_within])

    # Braking: search the whole lap and pick the first brake input within
    # the radius of the apex.
    brake_mask = brake > _BRAKE_THRESHOLD
    near_apex = np.abs(track_pos - apex_m) <= _BRAKING_SEARCH_RADIUS_M
    candidate_brake_idx = np.flatnonzero(brake_mask & near_apex)
    if candidate_brake_idx.size > 0:
        braking_m = float(track_pos[candidate_brake_idx[0]])
    else:
        braking_m = float(track_pos[corner_indices[0]])

    # Exit: per the S4.2 task brief, the first post-apex sample where
    # `Throttle > 0.5 AND |AccelLat|/g < 0.5`. The throttle re-application can
    # legitimately fall past the corner-detector's lat_g_exit boundary (the
    # detector closes the window once lateral load relaxes; exit is when the
    # driver gets back on power), so search the whole post-apex slice.
    n = track_pos.size
    post_apex_idx = np.arange(int(apex_within), n, dtype=np.int64)
    exit_candidates = post_apex_idx[
        (throttle[post_apex_idx] > _THROTTLE_THRESHOLD)
        & (lat_g[post_apex_idx] < _EXIT_LATERAL_G_THRESHOLD)
    ]
    if exit_candidates.size > 0:
        exit_m = float(track_pos[exit_candidates[0]])
    else:
        exit_m = float(track_pos[corner_indices[-1]])

    return apex_m, braking_m, exit_m


def _aggregate_across_laps(records: list[_LapLandmark]) -> pl.DataFrame:
    """Average per-corner landmarks across every contributing lap.

    Drops any corner_id whose observation count is below the maximum: a
    noisy lap that splits one corner into two produces a phantom corner
    seen in only that lap, and a phantom corner with one observation is
    worse than no landmark at that position (VISION §9 — "averaged and
    refined across hundreds of laps").
    """
    if not records:
        return _empty_landmarks_frame()

    aggregated = (
        pl.DataFrame(
            {
                "corner_id": [r.corner_position_index for r in records],
                "braking_point_m": [r.braking_point_m for r in records],
                "apex_m": [r.apex_m for r in records],
                "exit_point_m": [r.exit_point_m for r in records],
            }
        )
        .group_by("corner_id")
        .agg(
            pl.col("braking_point_m").mean(),
            pl.col("apex_m").mean(),
            pl.col("exit_point_m").mean(),
            pl.len().cast(pl.Int64).alias("n_observations"),
        )
    )
    max_observations = aggregated["n_observations"].max()
    return (
        aggregated.filter(pl.col("n_observations") == max_observations)
        .with_columns(pl.col("corner_id").cast(pl.Int32))
        .select(list(_CORNER_LANDMARKS_SCHEMA.keys()))
        .sort("corner_id", maintain_order=True)
    )


def _lap_length_for_session(
    conn,
    session_id: str,
    *,
    corpus_root: Path,
) -> float | None:
    """Reuse builder's lap-length resolution so corner positions match track-model bins.

    Imported lazily to keep `racingoptimizer.track.corners` independent of
    `builder` at module import (avoids a circular import via __init__).
    """
    from racingoptimizer.track.builder import _lap_length_for_session as _resolve

    return _resolve(conn, session_id, corpus_root=corpus_root)
