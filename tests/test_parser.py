from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from racingoptimizer.ingest.parser import EXCLUDED_CHANNEL_PATTERNS, ParseResult, parse_ibt


def test_parses_real_ibt(small_ibt: Path) -> None:
    result = parse_ibt(small_ibt)
    assert isinstance(result, ParseResult)

    # Car/track come from the YAML header.
    assert result.yaml_car  # raw IBT car id, e.g. "bmwlmdh"
    assert result.yaml_track  # raw IBT track id, non-empty

    # We expect at least the obvious physics channels.
    expected_channels = {"Speed", "Brake", "Throttle", "LapDistPct", "Lap"}
    missing = expected_channels - set(result.channels)
    assert not missing, f"missing expected channels: {missing}"

    # Channels are 1-D float arrays of consistent length.
    sample_count = result.channels["Speed"].shape[0]
    assert sample_count > 100
    assert all(arr.shape == (sample_count,) for arr in result.channels.values())

    # Lap spans look reasonable.
    assert any(s.valid == 1 for s in result.lap_spans), "expected at least one valid lap"

    # Setup is parsed and is a non-empty dict.
    assert isinstance(result.setup, dict) and result.setup

    # Weather summary contains at least the most important fields.
    for key in ("AirTemp_c_mean", "TrackTempCrew_c_mean"):
        assert key in result.weather_summary


def test_excluded_channels_are_dropped(small_ibt: Path) -> None:
    result = parse_ibt(small_ibt)
    for name in result.channels:
        for pattern in EXCLUDED_CHANNEL_PATTERNS:
            assert pattern not in name, f"channel {name!r} matched excluded pattern {pattern!r}"


def test_dtypes_are_float32_for_telemetry(small_ibt: Path) -> None:
    result = parse_ibt(small_ibt)
    assert result.channels["Speed"].dtype == np.float32
