from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from racingoptimizer.ingest.parser import (
    DEFAULT_SAMPLE_RATE_HZ,
    EXCLUDED_CHANNEL_PATTERNS,
    ParseResult,
    _detect_sample_rate,
    parse_ibt,
)


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


def test_dropped_channels_are_audited(small_ibt: Path) -> None:
    """VISION §1: every silent drop must show up in the audit trail.

    The two drop classes today are `EXCLUDED_CHANNEL_PATTERNS` (CarIdx*, per-tyre
    spot temps) and multi-element scalar arrays (`header.count > 1`). Real IBTs
    contain at least the CarIdx family, so the dict must be non-empty and every
    entry must carry a human-readable reason.
    """
    result = parse_ibt(small_ibt)
    assert isinstance(result.dropped_channels, dict)
    assert result.dropped_channels, "expected at least one dropped channel (CarIdx*) in real IBT"
    # Every recorded drop must have a non-empty reason string.
    for name, reason in result.dropped_channels.items():
        assert isinstance(name, str) and name
        assert isinstance(reason, str) and reason
    # Pattern-excluded names must each carry the pattern reason.
    pattern_drops = {
        n: r for n, r in result.dropped_channels.items()
        if any(p in n for p in EXCLUDED_CHANNEL_PATTERNS)
    }
    assert pattern_drops, "expected CarIdx*/Temp* pattern drops in BMW Sebring fixture"
    for reason in pattern_drops.values():
        assert "EXCLUDED_CHANNEL_PATTERNS" in reason
    # Drops and kept channels are disjoint.
    assert set(result.dropped_channels).isdisjoint(set(result.channels))


def test_sample_rate_auto_detected(small_ibt: Path) -> None:
    """`sample_rate_hz` must be detected from the IBT header, not hardcoded.

    BMW Sebring fixture is recorded at iRacing's default 60 Hz, so the
    detected rate exercises the `header.tick_rate` branch of the fallback
    chain. `duration_s` and the per-sample t-axis must agree with the rate.
    """
    result = parse_ibt(small_ibt)
    # Detected rate must be positive and within iRacing's documented range
    # (60 Hz default, 360 Hz max via app.ini).
    assert 1.0 < result.sample_rate_hz <= 720.0
    # BMW Sebring fixture is 60 Hz; allow a hair of float drift if the
    # disk-header back-computed branch ever wins.
    assert result.sample_rate_hz == pytest.approx(60.0, rel=1e-3)
    # duration_s must be consistent with sample count / detected rate.
    n = result.channels["LapDistPct"].shape[0]
    assert result.duration_s == pytest.approx(n / result.sample_rate_hz, rel=1e-9)


def test_detect_sample_rate_falls_back_to_default() -> None:
    """When neither the header nor YAML expose a rate, the documented
    fallback constant (`DEFAULT_SAMPLE_RATE_HZ` = 60.0) wins."""

    class _BareIBT:
        _header = None
        _disk_header = None

    rate = _detect_sample_rate(_BareIBT(), info={})
    assert rate == DEFAULT_SAMPLE_RATE_HZ == 60.0


def test_detect_sample_rate_uses_header_tick_rate() -> None:
    """`header.tick_rate` is the canonical source — used even when YAML
    advertises something different."""

    class _Header:
        tick_rate = 360

    class _IBT:
        _header = _Header()
        _disk_header = None

    rate = _detect_sample_rate(_IBT(), info={"SessionInfo": {"SessionTickRate": 60}})
    assert rate == 360.0


def test_detect_sample_rate_yaml_fallback() -> None:
    """When the header omits a tick rate, fall through to YAML SessionInfo."""

    class _Header:
        tick_rate = 0  # missing/zero — must skip

    class _IBT:
        _header = _Header()
        _disk_header = None

    rate = _detect_sample_rate(_IBT(), info={"SessionInfo": {"SessionTickRate": 360}})
    assert rate == 360.0


def test_dtypes_are_float32_for_telemetry(small_ibt: Path) -> None:
    result = parse_ibt(small_ibt)
    assert result.channels["Speed"].dtype == np.float32


def test_weather_summary_covers_vision_section_10_channels(small_ibt: Path) -> None:
    """VISION section 10 lists 12 weather/track channels. The summary must
    surface every channel present in the IBT (S2.2 expanded the set from 5
    to 12).

    Real BMW Sebring fixtures carry the full set, so every key for which
    the parser saw a channel must appear; the parser only emits a summary
    key when its source channel is present, which keeps the contract
    honest if iRacing ever drops a channel.
    """
    result = parse_ibt(small_ibt)
    expected_keys = {
        # Atmospheric:
        "AirTemp_c_mean",
        "AirDensity_kgm3_mean",
        "AirPressure_mbar_mean",
        "RelativeHumidity_mean",
        "WindVel_ms_mean",
        "WindDir_deg_mean",
        "FogLevel_max",
        # Track surface:
        "TrackTempCrew_c_mean",
        "TrackWetness_max",
        # Discrete weather state:
        "WeatherDeclaredWet_max",
        "PrecipType_max",
        "Skies_max",
    }
    raw_to_summary = {
        "AirTemp": "AirTemp_c_mean",
        "AirDensity": "AirDensity_kgm3_mean",
        "AirPressure": "AirPressure_mbar_mean",
        "RelativeHumidity": "RelativeHumidity_mean",
        "WindVel": "WindVel_ms_mean",
        "WindDir": "WindDir_deg_mean",
        "FogLevel": "FogLevel_max",
        "TrackTempCrew": "TrackTempCrew_c_mean",
        "TrackWetness": "TrackWetness_max",
        "WeatherDeclaredWet": "WeatherDeclaredWet_max",
        "Precipitation": "PrecipType_max",
        "Skies": "Skies_max",
    }
    for raw, summary_key in raw_to_summary.items():
        if raw in result.channels:
            assert summary_key in result.weather_summary, (
                f"channel {raw!r} present in IBT but {summary_key!r} missing "
                f"from weather_summary; got {sorted(result.weather_summary)}"
            )
    # Every emitted key must be in the expected set (no stray fields).
    assert set(result.weather_summary).issubset(expected_keys)
