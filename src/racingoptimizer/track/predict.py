"""Per-bin channel expectation lookups (S4.1, VISION §9).

Once the track model has compounded across multiple sessions, every
`track_pos_m` bin has a known distribution of shock-velocity p99,
lateral-G p95, and lateral-G median. `expected(track_pos_m, channel)`
returns the `(mean, p99)` of that distribution across sessions for the
bin containing the queried position. Cold-start (< 3 sessions) returns
None — there is no model to query.

Reads from the per-session cache parquet (one row per
`session_id × bin_index` written by `build_track_model`). The cache file
is small (a few thousand rows for a typical track / corpus pair) so a
direct `pl.read_parquet` per call is fine — no streaming or lazy frames
needed at the volumes we expect for slice D consumers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from racingoptimizer.track.bins import DEFAULT_BIN_SIZE_M, bin_index

PredictableChannel = str  # "shock_v_p99_mm_s" | "lateral_g_p95" | "lateral_g_median"

PREDICTABLE_CHANNELS: tuple[str, ...] = (
    "shock_v_p99_mm_s",
    "lateral_g_p95",
    "lateral_g_median",
)


@dataclass(frozen=True)
class Expected:
    """Cross-session distribution of a channel for one track-position bin."""

    mean: float
    p99: float
    n_sessions: int


def expected_from_cache(
    cache_df: pl.DataFrame,
    *,
    track_pos_m: float,
    channel: PredictableChannel,
    bin_size_m: float = DEFAULT_BIN_SIZE_M,
) -> Expected | None:
    """Return `(mean, p99)` of `channel` across sessions for the bin at `track_pos_m`.

    Returns None when:
    - `channel` is not in PREDICTABLE_CHANNELS
    - the cache has < 3 rows for the requested bin (cold-start short-circuit
      consistent with `_COLD_START_THRESHOLD` in `builder.py`)
    """
    if channel not in PREDICTABLE_CHANNELS:
        return None
    if cache_df.height == 0 or channel not in cache_df.columns:
        return None

    target_bin = int(bin_index(np.array([track_pos_m]), bin_size_m=bin_size_m)[0])
    target_pos = float(target_bin) * bin_size_m
    matching = cache_df.filter(pl.col("track_pos_m") == target_pos)
    if matching.height < 3:
        return None

    values = matching[channel].to_numpy().astype(np.float64)
    return Expected(
        mean=float(np.mean(values)),
        p99=float(np.quantile(values, 0.99)),
        n_sessions=int(matching.height),
    )
