"""compute_off_track_mask — synthetic single-lap detection windows (U7)."""
from __future__ import annotations

import numpy as np
import polars as pl

from racingoptimizer.track import compute_off_track_mask

_GRAVITY = 9.80665
_BIN_SIZE = 5.0
_LAP_LEN = 350.0  # 70 samples * 5 m / sample at lap_dist_pct step ≈ 1/70


def _build_lap(*, accel_lat: np.ndarray, wheel_diff: np.ndarray) -> pl.DataFrame:
    n = accel_lat.size
    lap_dist_pct = np.linspace(0.0, 1.0 - 1e-9, n)
    lf = np.full(n, 50.0)
    rf = lf - wheel_diff
    lr = np.full(n, 50.0)
    rr = np.full(n, 50.0)
    return pl.DataFrame(
        {
            "lap_dist_pct": lap_dist_pct,
            "LatAccel": accel_lat,
            "LFspeed": lf,
            "RFspeed": rf,
            "LRspeed": lr,
            "RRspeed": rr,
        }
    )


def _grip_map_for_lap(n: int, p95: float) -> pl.DataFrame:
    bin_indices = np.arange(0, int(np.ceil(_LAP_LEN / _BIN_SIZE)) + 1, dtype=np.int32)
    return pl.DataFrame(
        {
            "bin_index": bin_indices,
            "lateral_g_p95": np.full(bin_indices.size, p95, dtype=np.float64),
            "lateral_g_median": np.full(bin_indices.size, p95 * 0.5, dtype=np.float64),
            "n_samples": np.full(bin_indices.size, 100, dtype=np.uint32),
            "n_sessions": np.full(bin_indices.size, 5, dtype=np.int64),
        },
        schema={
            "bin_index": pl.Int32,
            "lateral_g_p95": pl.Float64,
            "lateral_g_median": pl.Float64,
            "n_samples": pl.UInt32,
            "n_sessions": pl.Int64,
        },
    )


def test_sudden_grip_loss_dilates_to_half_second_window():
    # Spec §4.4 / §8: total mask window is 0.5 s centred on the trigger,
    # i.e. ±15 samples at 60 Hz.
    n = 70
    accel = np.zeros(n)
    accel[30:51] = 8.0       # high grip 30..50 inclusive
    accel[51:56] = 1.0       # sudden drop 51..55
    accel[56:70] = 8.0       # recover

    df = _build_lap(accel_lat=accel, wheel_diff=np.zeros(n))
    grip_map = _grip_map_for_lap(n, p95=8.0 / _GRAVITY)

    mask = compute_off_track_mask(
        df, grip_map, lap_length_m=_LAP_LEN, bin_size_m=_BIN_SIZE, sample_rate_hz=60
    )
    assert mask.shape == (n,)
    flagged = np.where(mask)[0]
    assert flagged.size > 0
    # First trigger fires at sample 51; ±15 sample dilation flags samples 36..66.
    assert int(flagged.min()) <= 36
    assert int(flagged.max()) >= 55
    # Pre-trigger samples (more than half-window before the first trigger) stay clean.
    assert not mask[:35].any()


def test_wheel_speed_spike_dilates_to_half_second_window():
    # Spec §4.4 / §8: total mask window is 0.5 s centred on the trigger,
    # i.e. ±15 samples at 60 Hz.
    n = 70
    diff = np.zeros(n)
    diff[40:43] = 200.0  # 200 m/s differential

    # LatAccel 0 keeps grip-loss detector quiet (no high-grip history ever).
    df = _build_lap(accel_lat=np.zeros(n), wheel_diff=diff)
    grip_map = _grip_map_for_lap(n, p95=8.0 / _GRAVITY)

    mask = compute_off_track_mask(
        df, grip_map, lap_length_m=_LAP_LEN, bin_size_m=_BIN_SIZE, sample_rate_hz=60
    )
    assert mask.shape == (n,)
    flagged = np.where(mask)[0]
    assert flagged.size > 0
    # First trigger fires at sample 40; ±15 sample dilation flags samples 25..57.
    assert int(flagged.min()) <= 25
    assert int(flagged.max()) >= 42
    # Pre-trigger samples (more than half-window before the first trigger) stay clean.
    assert not mask[:24].any()


def test_cold_start_empty_grip_map_returns_all_false():
    n = 70
    df = _build_lap(accel_lat=np.full(n, 8.0), wheel_diff=np.zeros(n))
    empty_grip = pl.DataFrame(
        schema={
            "bin_index": pl.Int32,
            "lateral_g_p95": pl.Float64,
            "lateral_g_median": pl.Float64,
            "n_samples": pl.UInt32,
            "n_sessions": pl.Int64,
        }
    )
    mask = compute_off_track_mask(
        df, empty_grip, lap_length_m=_LAP_LEN, bin_size_m=_BIN_SIZE, sample_rate_hz=60
    )
    assert mask.shape == (n,)
    assert not mask.any()
