"""Unit tests for `segment_lap` (slice B-3)."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from racingoptimizer.corner import segment_lap
from racingoptimizer.corner.config import G_MS2

SAMPLE_HZ = 60


def _five_corner_frame() -> pl.DataFrame:
    """Build a 60-Hz lap with 5 sinusoidal lateral-G humps (peak 0.8g)."""
    flat_pre = np.zeros(int(0.5 * SAMPLE_HZ))
    parts: list[np.ndarray] = [flat_pre]
    for _ in range(5):
        # 1.0 s hump, peak 0.8 g, then 1.5 s flat between corners.
        n = int(1.0 * SAMPLE_HZ)
        hump = 0.8 * np.sin(np.linspace(0.0, np.pi, n, endpoint=False))
        parts.append(hump)
        parts.append(np.zeros(int(1.5 * SAMPLE_HZ)))
    accel_lat_g = np.concatenate(parts)
    n = accel_lat_g.size

    return pl.DataFrame(
        {
            "t_s": pl.Series("t_s", np.arange(n) / SAMPLE_HZ, dtype=pl.Float64),
            "AccelLat": pl.Series(
                "AccelLat", accel_lat_g * G_MS2, dtype=pl.Float32
            ),
            "Brake": pl.Series("Brake", np.zeros(n), dtype=pl.Float32),
            "Throttle": pl.Series("Throttle", np.zeros(n), dtype=pl.Float32),
            "SteeringWheelAngle": pl.Series(
                "SteeringWheelAngle", np.zeros(n), dtype=pl.Float32
            ),
        }
    )


def _five_corner_frame_at_rate(sample_hz: int) -> pl.DataFrame:
    """Build the same 5-corner timing profile at an arbitrary sample rate."""
    flat_pre = np.zeros(int(0.5 * sample_hz))
    parts: list[np.ndarray] = [flat_pre]
    for _ in range(5):
        n = int(1.0 * sample_hz)
        hump = 0.8 * np.sin(np.linspace(0.0, np.pi, n, endpoint=False))
        parts.append(hump)
        parts.append(np.zeros(int(1.5 * sample_hz)))
    accel_lat_g = np.concatenate(parts)
    n = accel_lat_g.size
    return pl.DataFrame(
        {
            "t_s": pl.Series("t_s", np.arange(n) / sample_hz, dtype=pl.Float64),
            "AccelLat": pl.Series(
                "AccelLat", accel_lat_g * G_MS2, dtype=pl.Float32
            ),
            "Brake": pl.Series("Brake", np.zeros(n), dtype=pl.Float32),
            "Throttle": pl.Series("Throttle", np.zeros(n), dtype=pl.Float32),
            "SteeringWheelAngle": pl.Series(
                "SteeringWheelAngle", np.zeros(n), dtype=pl.Float32
            ),
        }
    )


def test_segment_lap_appends_corner_id_and_phase() -> None:
    df = _five_corner_frame()
    n_in = df.height
    cols_in = list(df.columns)

    out = segment_lap(df)

    assert out.height == n_in
    assert "corner_id" in out.columns
    assert "phase" in out.columns
    # Original columns unchanged (and still present).
    for c in cols_in:
        assert c in out.columns
        # Same values & dtype.
        assert_frame_equal(out.select(c), df.select(c))

    # Five distinct in-corner ids (-1 between humps).
    distinct = sorted({c for c in out["corner_id"].to_list() if c >= 0})
    assert distinct == [0, 1, 2, 3, 4]
    # Phase column is Utf8 with values from the Phase enum vocabulary.
    phases = set(out["phase"].to_list())
    assert phases.issubset(
        {"straight", "braking", "trail_brake", "mid_corner", "exit"}
    )


def test_segment_lap_does_not_mutate_input() -> None:
    df = _five_corner_frame()
    cols_before = list(df.columns)
    segment_lap(df)
    # No new columns leaked into the caller's frame.
    assert list(df.columns) == cols_before


def test_segment_lap_track_model_raises_not_implemented() -> None:
    df = _five_corner_frame()
    sentinel = object()
    with pytest.raises(NotImplementedError, match="track_model"):
        segment_lap(df, track_model=sentinel)


def test_segment_lap_missing_column_raises_value_error() -> None:
    df = _five_corner_frame().drop("Brake")
    with pytest.raises(ValueError, match="Brake"):
        segment_lap(df)


def test_segment_lap_missing_multiple_columns_lists_all() -> None:
    df = _five_corner_frame().drop(["Brake", "Throttle"])
    with pytest.raises(ValueError) as info:
        segment_lap(df)
    msg = str(info.value)
    assert "Brake" in msg
    assert "Throttle" in msg


def test_segment_lap_is_pure_two_calls_identical() -> None:
    df = _five_corner_frame()
    a = segment_lap(df)
    b = segment_lap(df)
    assert_frame_equal(a, b)


def test_segment_lap_derives_sample_rate_from_time_axis() -> None:
    sample_hz = 360
    flat_pre = np.zeros(int(0.5 * sample_hz))
    # 0.2 s above threshold is shorter than the 400 ms minimum corner
    # duration. A stale 60 Hz conversion sees 72 samples >= 24 and accepts it;
    # deriving 360 Hz from t_s correctly requires 144 samples and rejects it.
    short_hump = 0.8 * np.sin(np.linspace(0.0, np.pi, int(0.2 * sample_hz)))
    accel_lat_g = np.concatenate([flat_pre, short_hump, flat_pre])
    n = accel_lat_g.size
    df = pl.DataFrame(
        {
            "t_s": pl.Series("t_s", np.arange(n) / sample_hz, dtype=pl.Float64),
            "AccelLat": pl.Series(
                "AccelLat", accel_lat_g * G_MS2, dtype=pl.Float32
            ),
            "Brake": pl.Series("Brake", np.zeros(n), dtype=pl.Float32),
            "Throttle": pl.Series("Throttle", np.zeros(n), dtype=pl.Float32),
            "SteeringWheelAngle": pl.Series(
                "SteeringWheelAngle", np.zeros(n), dtype=pl.Float32
            ),
        }
    )

    out = segment_lap(df)

    distinct = sorted({c for c in out["corner_id"].to_list() if c >= 0})
    assert distinct == []
