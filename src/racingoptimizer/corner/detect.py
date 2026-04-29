"""Lateral-G corner detector with Schmitt-trigger hysteresis.

Pure function over a single lap's Polars frame: returns one corner_id per
sample, -1 outside any corner. See spec §4 for the algorithm.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from racingoptimizer.corner.config import (
    DEFAULT_THRESHOLDS,
    G_MS2,
    PhaseThresholds,
    ms_to_samples,
)


def detect_corners(
    lap_df: pl.DataFrame,
    *,
    thresholds: PhaseThresholds = DEFAULT_THRESHOLDS,
) -> np.ndarray:
    n = lap_df.height
    out = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return out

    lat_g = np.abs(lap_df["AccelLat"].to_numpy().astype(np.float64)) / G_MS2

    hz = thresholds.sample_rate_hz
    exit_hold_samples = ms_to_samples(thresholds.exit_hold_ms, hz, minimum=1)
    min_duration_samples = ms_to_samples(thresholds.min_corner_duration_ms, hz)
    min_gap_samples = ms_to_samples(thresholds.min_gap_ms, hz)

    candidates: list[tuple[int, int]] = []
    inside = False
    start = 0
    below_run = 0
    for i in range(n):
        g = lat_g[i]
        if not inside:
            if g > thresholds.lat_g_entry:
                inside = True
                start = i
                below_run = 0
        else:
            if g < thresholds.lat_g_exit:
                below_run += 1
                if below_run >= exit_hold_samples:
                    end = i - below_run + 1
                    candidates.append((start, end))
                    inside = False
                    below_run = 0
            else:
                below_run = 0
    if inside:
        candidates.append((start, n))

    accepted: list[tuple[int, int]] = []
    last_end = -10**9
    for s, e in candidates:
        if (e - s) < min_duration_samples:
            continue
        if s - last_end < min_gap_samples:
            continue
        accepted.append((s, e))
        last_end = e

    for cid, (s, e) in enumerate(accepted):
        out[s:e] = cid
    return out
