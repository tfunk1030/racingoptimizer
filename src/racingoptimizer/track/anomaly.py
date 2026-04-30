"""Per-sample anomaly flagging against the compounded track model (S4.1, VISION §9).

Once `expected(track_pos_m, channel)` knows the cross-session distribution
of a channel at a position, observed values that fall far outside that
distribution are anomalies. The classifier produces a `pl.DataFrame` of
`(sample_idx, channel, observed, expected, z_score, label)` rows, one per
flagged sample × channel. Threshold: `|z_score| > 3.0`.

Heuristic labels:
- `data_noise` — extreme outlier (`|z| > 10`) of a single sample, no
  neighbours flagged. Likely a sensor glitch or telemetry decode burp.
- `setup_problem` — cluster of consecutive flagged samples on the same
  channel. The car is consistently behaving differently than the track
  model expects in this region — points at a setup change vs the
  population.
- `driver_error` — everything else: a moderate spike at a non-curb bin,
  bracketed by clean samples. Most likely a one-off line / brake error
  rather than a setup or data issue.

The flagger is robust to channel absence and cold-start: if the track
model's per-session cache has < 3 sessions for a bin, `expected` returns
None and the sample contributes no anomaly rows.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from racingoptimizer.track.bins import DEFAULT_BIN_SIZE_M, bin_index, track_pos_m_from_pct
from racingoptimizer.track.masks import _max_abs_shock_vel, shock_vel_channels
from racingoptimizer.track.predict import PREDICTABLE_CHANNELS

_GRAVITY_M_S2 = 9.80665
_Z_THRESHOLD = 3.0
_NOISE_Z_THRESHOLD = 10.0
_CLUSTER_MIN_RUN = 3  # consecutive flagged samples on same channel → setup_problem
_MIN_SESSIONS_FOR_EXPECTATION = 3  # matches predict.expected_from_cache cold-start floor

_ANOMALY_SCHEMA: dict[str, type[pl.DataType]] = {
    "sample_idx": pl.Int64,
    "channel": pl.Utf8,
    "observed": pl.Float64,
    "expected": pl.Float64,
    "z_score": pl.Float64,
    "label": pl.Utf8,
}


def _empty_anomaly_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_ANOMALY_SCHEMA)


def flag_anomalies_from_cache(
    lap_df: pl.DataFrame,
    cache_df: pl.DataFrame,
    *,
    lap_length_m: float,
    bin_size_m: float = DEFAULT_BIN_SIZE_M,
    car: str | None = None,
    z_threshold: float = _Z_THRESHOLD,
) -> pl.DataFrame:
    """Flag samples whose observed channel value deviates far from the bin's expectation.

    Computes per-sample observed channel values, looks up the bin's
    cross-session expectation from the per-session cache, and emits a
    row per (sample, channel) pair where `|z_score| > z_threshold`.

    Returns an empty frame (with schema) when the model is in cold-start
    for every bin, or when the lap is empty.
    """
    if lap_df.height == 0 or cache_df.height == 0:
        return _empty_anomaly_frame()
    if "lap_dist_pct" not in lap_df.columns:
        return _empty_anomaly_frame()

    track_pos = track_pos_m_from_pct(lap_df["lap_dist_pct"].to_numpy(), lap_length_m)
    bins = bin_index(track_pos, bin_size_m=bin_size_m)
    n = lap_df.height

    observed_by_channel = _compute_observed(lap_df, car=car)
    rows: list[dict[str, object]] = []
    for channel, observed in observed_by_channel.items():
        if observed is None:
            continue
        expectations = _expected_per_sample(
            cache_df, bins, channel=channel, bin_size_m=bin_size_m
        )
        if expectations is None:
            continue

        valid = expectations["valid"]
        means = expectations["mean"]
        p99s = expectations["p99"]
        z = np.zeros(n, dtype=np.float64)
        spread = np.maximum(p99s - means, 1.0)
        z[valid] = (observed[valid] - means[valid]) / spread[valid]

        flagged_idx = np.flatnonzero(valid & (np.abs(z) > z_threshold))
        if flagged_idx.size == 0:
            continue
        labels = _classify(flagged_idx, np.abs(z[flagged_idx]))
        for sample_idx, label in zip(flagged_idx, labels, strict=True):
            rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "channel": channel,
                    "observed": float(observed[sample_idx]),
                    "expected": float(means[sample_idx]),
                    "z_score": float(z[sample_idx]),
                    "label": label,
                }
            )

    if not rows:
        return _empty_anomaly_frame()
    return pl.DataFrame(rows, schema=_ANOMALY_SCHEMA).sort(
        ["sample_idx", "channel"], maintain_order=True
    )


# ---- internals ----

def _compute_observed(
    lap_df: pl.DataFrame, *, car: str | None
) -> dict[str, np.ndarray | None]:
    """Per-sample observed values for each PREDICTABLE_CHANNEL.

    `shock_v_p99_mm_s` is mapped to instantaneous max-abs shock-velocity
    (mm/s) — same transform as `_max_abs_shock_vel`. `lateral_g_*` map to
    `|LatAccel|/g`. Returns None for channels whose source columns are
    absent — graceful degradation per spec §9.
    """
    out: dict[str, np.ndarray | None] = {ch: None for ch in PREDICTABLE_CHANNELS}

    channels = shock_vel_channels(car)
    if all(c in lap_df.columns for c in channels):
        out["shock_v_p99_mm_s"] = _max_abs_shock_vel(lap_df, channels)

    if "LatAccel" in lap_df.columns:
        lat_g = np.abs(lap_df["LatAccel"].to_numpy()) / _GRAVITY_M_S2
        out["lateral_g_p95"] = lat_g
        out["lateral_g_median"] = lat_g

    return out


def _expected_per_sample(
    cache_df: pl.DataFrame,
    bins: np.ndarray,
    *,
    channel: str,
    bin_size_m: float,
) -> dict[str, np.ndarray] | None:
    """Vectorised per-sample (mean, p99, valid) lookup from the per-session cache.

    Returns None if the cache cannot answer for this channel at all.
    Otherwise marks `valid=False` for samples whose bin has < 3 sessions
    (cold-start short-circuit, matching `expected_from_cache`).

    Pre-aggregates the cache once with a single ``group_by("track_pos_m")`` so
    a lap with ~1700 unique bins (Spa) does ~1 polars op rather than 1700
    ``filter`` scans of the whole cache.
    """
    if channel not in cache_df.columns:
        return None

    aggregated = (
        cache_df.group_by("track_pos_m")
        .agg(
            pl.col(channel).mean().alias("_mean"),
            pl.col(channel).quantile(0.99, "linear").alias("_p99"),
            pl.len().alias("_n_sessions"),
        )
        .filter(pl.col("_n_sessions") >= _MIN_SESSIONS_FOR_EXPECTATION)
    )
    if aggregated.height == 0:
        return None

    bin_to_pos = (aggregated["track_pos_m"].to_numpy() / bin_size_m).astype(np.int64)
    mean_lookup = dict(zip(bin_to_pos, aggregated["_mean"].to_numpy(), strict=True))
    p99_lookup = dict(zip(bin_to_pos, aggregated["_p99"].to_numpy(), strict=True))

    n = bins.size
    means = np.zeros(n, dtype=np.float64)
    p99s = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for i, b in enumerate(bins):
        b_int = int(b)
        if b_int in mean_lookup:
            means[i] = mean_lookup[b_int]
            p99s[i] = p99_lookup[b_int]
            valid[i] = True
    return {"mean": means, "p99": p99s, "valid": valid}


def _classify(flagged_idx: np.ndarray, abs_z: np.ndarray) -> list[str]:
    """Heuristic per-flag label.

    See module docstring for the full rule. Operates on the indices of
    flagged samples for one channel — a sample's neighbour status is
    determined by whether the next/prev flagged sample is adjacent.
    """
    if flagged_idx.size == 0:
        return []
    flagged_set = set(flagged_idx.tolist())
    labels: list[str] = []
    for i, z in zip(flagged_idx.tolist(), abs_z, strict=True):
        run_len = 1
        left = i - 1
        while left in flagged_set:
            run_len += 1
            left -= 1
        right = i + 1
        while right in flagged_set:
            run_len += 1
            right += 1

        if z > _NOISE_Z_THRESHOLD and run_len == 1:
            labels.append("data_noise")
        elif run_len >= _CLUSTER_MIN_RUN:
            labels.append("setup_problem")
        else:
            labels.append("driver_error")
    return labels
