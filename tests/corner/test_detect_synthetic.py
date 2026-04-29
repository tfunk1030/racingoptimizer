import numpy as np
import polars as pl

from racingoptimizer.corner import detect_corners
from racingoptimizer.corner.config import G_MS2, PhaseThresholds

SAMPLE_HZ = 60


def _frame(accel_lat_g: np.ndarray) -> pl.DataFrame:
    n = accel_lat_g.size
    return pl.DataFrame(
        {
            "t_s": pl.Series("t_s", np.arange(n) / SAMPLE_HZ, dtype=pl.Float64),
            "AccelLat": pl.Series("AccelLat", accel_lat_g * G_MS2, dtype=pl.Float32),
        }
    )


def _hump(width_s: float, peak_g: float) -> np.ndarray:
    n = int(round(width_s * SAMPLE_HZ))
    return peak_g * np.sin(np.linspace(0.0, np.pi, n, endpoint=False))


def _flat(width_s: float) -> np.ndarray:
    return np.zeros(int(round(width_s * SAMPLE_HZ)))


def test_five_humps_yield_five_corners():
    parts = []
    parts.append(_flat(0.5))
    for _ in range(5):
        parts.append(_hump(1.0, 0.8))
        parts.append(_flat(1.0))
    signal = np.concatenate(parts)
    df = _frame(signal)
    ids = detect_corners(df)
    distinct = sorted(set(ids[ids >= 0].tolist()))
    assert distinct == [0, 1, 2, 3, 4]
    # Between humps at least one sample must be -1.
    assert -1 in ids.tolist()


def test_subthreshold_hump_detected_zero_corners():
    signal = np.concatenate([_flat(0.5), _hump(1.5, 0.49), _flat(0.5)])
    ids = detect_corners(_frame(signal))
    assert (ids == -1).all()


def test_short_spike_rejected_by_min_duration():
    # 200 ms spike at 0.6g — above entry threshold but only 12 samples.
    spike = np.full(int(round(0.2 * SAMPLE_HZ)), 0.6)
    signal = np.concatenate([_flat(0.5), spike, _flat(0.5)])
    ids = detect_corners(_frame(signal))
    assert (ids == -1).all()


def test_min_gap_drops_second_corner():
    # Construct a case where corner A closes (short exit_hold) and corner B's
    # entry sample lands within the min-gap window so the second is dropped.
    # Default exit_hold_ms == min_gap_ms (200 ms) makes this scenario
    # unreachable; bump the gap threshold to isolate the rule.
    thresholds = PhaseThresholds(exit_hold_ms=100, min_gap_ms=400)
    n_hump = 30
    n_valley = 7   # 7 samples ≈ 117 ms — exceeds 100 ms hold so corner A closes
    sig = np.concatenate(
        [
            _flat(0.5),
            np.full(n_hump, 0.8),
            np.zeros(n_valley),
            np.full(n_hump, 0.8),
            _flat(0.5),
        ]
    )
    ids = detect_corners(_frame(sig), thresholds=thresholds)
    distinct = sorted(set(ids[ids >= 0].tolist()))
    assert distinct == [0]


def test_pure_function():
    signal = np.concatenate([_flat(0.5), _hump(1.0, 0.8), _flat(0.5)])
    df = _frame(signal)
    a = detect_corners(df)
    b = detect_corners(df)
    np.testing.assert_array_equal(a, b)
