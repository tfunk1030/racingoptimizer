"""Curb / bump / off-track detectors and mask APIs (slice D-2, unit U7).

Pure functions over polars frames + numpy arrays. No I/O. The aggregators
compose: per-session p99 → cross-session likelihoods → per-sample masks at
lap_data() time. See `docs/superpowers/specs/2026-04-28-track-model-design.md`
§4 + §5 for the algorithm narrative.

Thresholds default to the spec §5 values; callers can override for unit tests
or future tuning. Mutual exclusivity (curb suppresses bump_likelihood) is
enforced at aggregate time, not at sample time.

Per-car shock-velocity channels: the four-corner standard
(`LFshockVel`/`RFshockVel`/`LRshockVel`/`RRshockVel`) is what BMW, Cadillac,
Ferrari, and Porsche expose. Acura ARX-06 IBT files instead expose two heave
channels (`HFshockVel`, `TRshockVel`) and two roll channels (`FROLLshockVel`,
`RROLLshockVel`) — its suspension geometry has no per-corner shock channels.
`shock_vel_channels(car)` resolves the right names; `_max_abs_shock_vel`
takes their elementwise maximum just like the four-corner case.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.ndimage import median_filter

from racingoptimizer.track.bins import DEFAULT_BIN_SIZE_M, bin_index, track_pos_m_from_pct

DEFAULT_SHOCK_VEL_CHANNELS: tuple[str, ...] = (
    "LFshockVel",
    "RFshockVel",
    "LRshockVel",
    "RRshockVel",
)

# Per-car overrides: cars whose IBT exports diverge from the four-corner standard.
# Verified 2026-04-29 against `ibtfiles/acuraarx06gtp_*.ibt`.
_PER_CAR_SHOCK_VEL_CHANNELS: dict[str, tuple[str, ...]] = {
    "acura": ("HFshockVel", "TRshockVel", "FROLLshockVel", "RROLLshockVel"),
}


def shock_vel_channels(car: str | None) -> tuple[str, ...]:
    """Return the IBT shock-velocity channel names for `car` (lowercase key).

    Falls back to the four-corner default for unknown cars and for `None` —
    the latter happens in synthetic tests that don't carry car metadata.
    """
    if car is None:
        return DEFAULT_SHOCK_VEL_CHANNELS
    return _PER_CAR_SHOCK_VEL_CHANNELS.get(car.lower(), DEFAULT_SHOCK_VEL_CHANNELS)


def _max_abs_shock_vel(df: pl.DataFrame, channels: tuple[str, ...]) -> np.ndarray:
    """Elementwise max(|channel|) across the given shock-velocity columns.

    iRacing emits ``*shockVel`` channels in m/s. Slice D's threshold constants
    (``T_CURB_*_MM_S``, ``BUMP_RANGE_*_MM_S``) and persisted ``shock_v_p99_mm_s``
    column are in mm/s — convert at the read site so every downstream
    aggregation and comparison stays in mm/s units.
    """
    return np.maximum.reduce([np.abs(df[c].to_numpy()) * 1000.0 for c in channels])

T_CURB_SESSION_MM_S: float = 350.0
T_CURB_AGGREGATE_MM_S: float = 400.0
CURB_AGREEMENT_FRACTION: float = 0.6

# Per-car overrides for the curb-agreement fraction. The default 0.6 is
# calibrated against the four-corner shock signal (BMW/Cadillac/Ferrari/Porsche),
# where any one wheel hitting a curb pushes that bin's session p99 above the
# trigger. Acura ARX-06 instead exposes two HEAVE channels and two ROLL
# channels — the heave signal aggregates symmetric loading across both wheels
# of an axle, so the per-session p99 is lower-amplitude and only one or two
# sessions out of a small N agree on which bins trigger. With as few as 3
# sessions in the corpus the agreement values are quantised at {0, ⅓, ⅔, 1}
# so 0.6 disqualifies a 2-of-3 (≈0.67) majority but 0.3 catches the single-
# session sightings that BMW's four-corner signal would have caught with one
# session. Empirically tuned against the Acura Hockenheim 3-session corpus.
_PER_CAR_CURB_AGREEMENT_FRACTION: dict[str, float] = {
    "acura": 0.3,
}


def curb_agreement_fraction_for(car: str | None) -> float:
    """Return the per-car curb-agreement fraction (defaults to 0.6)."""
    if car is None:
        return CURB_AGREEMENT_FRACTION
    return _PER_CAR_CURB_AGREEMENT_FRACTION.get(
        car.strip().lower(), CURB_AGREEMENT_FRACTION
    )


BUMP_RANGE_MIN_MM_S: float = 150.0
BUMP_RANGE_MAX_MM_S: float = 350.0
OFFTRACK_GRIP_LOSS_RATIO: float = 0.5
OFFTRACK_GRIP_HISTORY_MS: int = 100
OFFTRACK_WHEELSPEED_RATIO: float = 3.0
# Total mask window centred on the trigger sample, in seconds. Per
# `docs/superpowers/specs/2026-04-28-track-model-design.md` §4.4 / §8 this is
# 0.5 s = ±15 samples at 60 Hz around the trigger. The dilation kernel uses
# the half-window (window_samples = WINDOW * rate / 2) so its size is
# 2*window_samples + 1 ≈ rate * WINDOW total samples flagged.
OFFTRACK_MASK_WINDOW_S: float = 0.5

_GRAVITY_M_S2 = 9.80665

_PER_SESSION_P99_SCHEMA: dict[str, type[pl.DataType]] = {
    "bin_index": pl.Int32,
    "shock_v_p99_mm_s": pl.Float64,
    "n_samples": pl.UInt32,
}

_BUMP_MAP_SCHEMA: dict[str, type[pl.DataType]] = {
    "bin_index": pl.Int32,
    "shock_v_p99_mm_s": pl.Float64,
    "n_sessions": pl.Int64,
    "curb_likelihood": pl.Float64,
    "bump_likelihood": pl.Float64,
}

_GRIP_MAP_SCHEMA: dict[str, type[pl.DataType]] = {
    "bin_index": pl.Int32,
    "lateral_g_p95": pl.Float64,
    "lateral_g_median": pl.Float64,
    "n_samples": pl.UInt32,
    "n_sessions": pl.Int64,
}


def compute_session_shock_v_p99_per_bin(
    lap_df: pl.DataFrame,
    *,
    lap_length_m: float,
    bin_size_m: float = DEFAULT_BIN_SIZE_M,
    car: str | None = None,
) -> pl.DataFrame:
    if lap_df.height == 0:
        return pl.DataFrame(schema=_PER_SESSION_P99_SCHEMA)

    df = lap_df
    if "lap_index" in df.columns:
        df = df.filter(pl.col("lap_index") != -1)
    if "data_quality_mask" in df.columns:
        df = df.filter(pl.col("data_quality_mask"))
    if df.height == 0:
        return pl.DataFrame(schema=_PER_SESSION_P99_SCHEMA)

    track_pos = track_pos_m_from_pct(df["lap_dist_pct"].to_numpy(), lap_length_m)
    idx = bin_index(track_pos, bin_size_m=bin_size_m)
    shock = _max_abs_shock_vel(df, shock_vel_channels(car))
    samples = pl.DataFrame(
        {
            "bin_index": idx.astype(np.int32),
            "shock_v_p99_mm_s": shock.astype(np.float64),
        }
    ).filter(pl.col("bin_index") >= 0)
    if samples.height == 0:
        return pl.DataFrame(schema=_PER_SESSION_P99_SCHEMA)

    out = (
        samples.group_by("bin_index")
        .agg(
            pl.col("shock_v_p99_mm_s").quantile(0.99, "linear").alias("shock_v_p99_mm_s"),
            pl.len().cast(pl.UInt32).alias("n_samples"),
        )
        .sort("bin_index", maintain_order=True)
    )
    return out.select(list(_PER_SESSION_P99_SCHEMA.keys()))


def aggregate_curb_likelihood(
    per_session_p99: list[pl.DataFrame],
    *,
    t_session: float = T_CURB_SESSION_MM_S,
    t_aggregate: float = T_CURB_AGGREGATE_MM_S,
    agreement: float | None = None,
    car: str | None = None,
) -> pl.DataFrame:
    """Aggregate per-session p99 frames into a cross-session curb/bump map.

    The persistent-curb / suppress-bump rule fires when a bin's
    cross-session agreement >= ``agreement`` AND its max p99 >= ``t_aggregate``.
    When ``agreement`` is left as ``None`` the per-car default is resolved via
    :func:`curb_agreement_fraction_for` (Acura uses a lower threshold to
    accommodate its heave/roll-shock signal — see the table comment).
    """
    if agreement is None:
        agreement = curb_agreement_fraction_for(car)
    if not per_session_p99:
        return pl.DataFrame(schema=_BUMP_MAP_SCHEMA)

    tagged: list[pl.DataFrame] = []
    for frame in per_session_p99:
        if frame.height == 0:
            continue
        tagged.append(
            frame.with_columns(
                (pl.col("shock_v_p99_mm_s") > t_session).alias("is_curb_session"),
            )
        )
    if not tagged:
        return pl.DataFrame(schema=_BUMP_MAP_SCHEMA)

    stacked = pl.concat(tagged, how="vertical_relaxed")
    grouped = (
        stacked.group_by("bin_index")
        .agg(
            pl.col("shock_v_p99_mm_s").max().alias("shock_v_p99_mm_s"),
            pl.col("is_curb_session").sum().cast(pl.Int64).alias("_n_curb_sessions"),
            pl.len().cast(pl.Int64).alias("n_sessions"),
        )
        .with_columns(
            (
                pl.col("_n_curb_sessions").cast(pl.Float64)
                / pl.max_horizontal(pl.col("n_sessions").cast(pl.Float64), pl.lit(1.0))
            ).alias("curb_likelihood"),
        )
    )
    persistently_curb = (pl.col("curb_likelihood") >= agreement) & (
        pl.col("shock_v_p99_mm_s") >= t_aggregate
    )
    bump_raw = (
        (pl.col("shock_v_p99_mm_s") - BUMP_RANGE_MIN_MM_S)
        / (BUMP_RANGE_MAX_MM_S - BUMP_RANGE_MIN_MM_S)
    ).clip(0.0, 1.0)
    out = grouped.with_columns(
        pl.when(persistently_curb)
        .then(pl.lit(0.0, dtype=pl.Float64))
        .otherwise(bump_raw)
        .alias("bump_likelihood"),
    )
    return out.select(list(_BUMP_MAP_SCHEMA.keys())).sort("bin_index", maintain_order=True)


def aggregate_grip_map(per_session_grip: list[pl.DataFrame]) -> pl.DataFrame:
    if not per_session_grip:
        return pl.DataFrame(schema=_GRIP_MAP_SCHEMA)

    frames = [f for f in per_session_grip if f.height > 0]
    if not frames:
        return pl.DataFrame(schema=_GRIP_MAP_SCHEMA)

    stacked = pl.concat(frames, how="vertical_relaxed")
    out = (
        stacked.group_by("bin_index")
        .agg(
            pl.col("lateral_g_p95").median().alias("lateral_g_p95"),
            pl.col("lateral_g_median").median().alias("lateral_g_median"),
            pl.col("n_samples").sum().cast(pl.UInt32).alias("n_samples"),
            pl.col("n_sessions").sum().cast(pl.Int64).alias("n_sessions"),
        )
        .sort("bin_index", maintain_order=True)
    )
    return out.select(list(_GRIP_MAP_SCHEMA.keys()))


def compute_curb_mask(
    lap_df: pl.DataFrame,
    bump_map: pl.DataFrame,
    *,
    lap_length_m: float,
    bin_size_m: float = DEFAULT_BIN_SIZE_M,
    car: str | None = None,
) -> np.ndarray:
    n = lap_df.height
    if n == 0 or bump_map.height == 0:
        return np.zeros(n, dtype=bool)

    threshold = curb_agreement_fraction_for(car)
    curb_bins = (
        bump_map.filter(pl.col("curb_likelihood") >= threshold)["bin_index"]
        .to_numpy()
        .astype(np.int32)
    )
    if curb_bins.size == 0:
        return np.zeros(n, dtype=bool)

    track_pos = track_pos_m_from_pct(lap_df["lap_dist_pct"].to_numpy(), lap_length_m)
    idx = bin_index(track_pos, bin_size_m=bin_size_m)
    return np.isin(idx, curb_bins)


def compute_off_track_mask(
    lap_df: pl.DataFrame,
    grip_map: pl.DataFrame,
    *,
    lap_length_m: float,
    bin_size_m: float = DEFAULT_BIN_SIZE_M,
    sample_rate_hz: int = 60,
) -> np.ndarray:
    n = lap_df.height
    if n == 0 or grip_map.height == 0:
        return np.zeros(n, dtype=bool)

    track_pos = track_pos_m_from_pct(lap_df["lap_dist_pct"].to_numpy(), lap_length_m)
    idx = bin_index(track_pos, bin_size_m=bin_size_m)

    grip_lookup = dict(
        zip(
            grip_map["bin_index"].to_numpy().astype(np.int32),
            grip_map["lateral_g_p95"].to_numpy().astype(np.float64),
            strict=True,
        )
    )
    bin_p95 = np.array([grip_lookup.get(int(b), np.nan) for b in idx], dtype=np.float64)

    triggers = np.zeros(n, dtype=bool)

    # Detector 1: sudden grip loss after a sustained high-grip window.
    lat_g = np.abs(lap_df["LatAccel"].to_numpy()) / _GRAVITY_M_S2
    history_samples = max(1, int(round(OFFTRACK_GRIP_HISTORY_MS * sample_rate_hz / 1000)))
    high_grip = lat_g >= 0.8 * bin_p95
    grip_loss = lat_g < OFFTRACK_GRIP_LOSS_RATIO * bin_p95
    if history_samples > 0:
        window = np.ones(history_samples, dtype=np.float64)
        high_count = np.convolve(high_grip.astype(np.float64), window, mode="full")[
            : n
        ]
        history_ok = np.zeros(n, dtype=bool)
        if n > history_samples:
            history_ok[history_samples:] = (
                high_count[history_samples - 1 : n - 1] >= 0.5 * history_samples
            )
        valid_p95 = ~np.isnan(bin_p95)
        triggers |= grip_loss & history_ok & valid_p95

    # Detector 2: wheel-speed differential spike vs forward 1s rolling-median baseline.
    speeds = np.stack(
        [
            lap_df["LFspeed"].to_numpy().astype(np.float64),
            lap_df["RFspeed"].to_numpy().astype(np.float64),
            lap_df["LRspeed"].to_numpy().astype(np.float64),
            lap_df["RRspeed"].to_numpy().astype(np.float64),
        ]
    )
    diff = speeds.max(axis=0) - speeds.min(axis=0)
    baseline_window = max(1, int(round(sample_rate_hz)))  # 1 second
    baseline = _rolling_median_forward(diff, baseline_window)
    spike = diff > OFFTRACK_WHEELSPEED_RATIO * baseline
    triggers |= spike

    # Half-window in samples; kernel size = 2*half + 1 ≈ OFFTRACK_MASK_WINDOW_S * rate.
    window_samples = int(round(OFFTRACK_MASK_WINDOW_S * sample_rate_hz / 2))
    return _dilate(triggers, window_samples)


# ---- internals ----

def _rolling_median_forward(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling median: out[i] = median(values[max(0, i-window+1) : i+1]).

    Steady-state (i ≥ window-1) is delegated to ``scipy.ndimage.median_filter``
    with a kernel size rounded up to the nearest odd integer and the origin
    shifted so the kernel is right-aligned at sample i. The warm-up prefix
    (i < window-1) is computed per-sample so the first samples remain defined
    over the available data, matching the original loop's edge behaviour.

    The naive O(n²) loop this replaces dominated ``compute_off_track_mask``
    runtime at realistic ingest scale (~10⁶ samples per build); switching to
    the C-implemented filter is ~25× faster than the equivalent
    ``sliding_window_view`` + ``np.median`` vectorisation.

    Note: for even-sized windows the true median averages the two middle
    values, while ``median_filter`` picks one of them. We round the kernel up
    by one sample to keep the implementation exact for odd ``window`` and
    arbitrarily close for even ``window`` — the call site uses this as a
    spike-detection baseline where the sub-1% difference is immaterial.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    w = max(1, int(window))
    out = np.empty(n, dtype=np.float64)
    prefix = min(w - 1, n)
    for i in range(prefix):
        out[i] = float(np.median(arr[: i + 1]))
    if n >= w:
        kernel = w if w % 2 == 1 else w + 1
        # Right-align the kernel at sample i so the window covers [i-w+1 .. i].
        origin = kernel // 2
        filtered = median_filter(arr, size=kernel, mode="nearest", origin=origin)
        out[w - 1 :] = filtered[w - 1 :]
    return out


def _dilate(mask: np.ndarray, window: int) -> np.ndarray:
    if window <= 0 or mask.size == 0:
        return mask.astype(bool, copy=True)
    kernel = np.ones(2 * window + 1, dtype=np.int32)
    return np.convolve(mask.astype(np.int32), kernel, mode="same") > 0
