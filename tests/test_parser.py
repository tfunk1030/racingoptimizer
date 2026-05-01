from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from racingoptimizer.ingest.parser import (
    DEFAULT_SAMPLE_RATE_HZ,
    EXCLUDED_CHANNEL_PATTERNS,
    ParseResult,
    _detect_sample_rate,
    convert_track_wetness_enum,
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


# ---------------------------------------------------------------------------
# TrackWetness enum -> 0..1 fraction conversion (VISION §10).
# iRacing's `TrackWetness` channel is enum-coded (1=Dry .. 7=Extremely Wet),
# not a 0..1 fraction. The parser remaps at ingest time so every downstream
# module sees the spec-canonical scale. Without the remap, every dry
# (enum=1) session was being classified as `full_rain` by `wet_mode`.
# ---------------------------------------------------------------------------

def test_convert_track_wetness_enum_dry_maps_to_zero() -> None:
    raw = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    out = convert_track_wetness_enum(raw)
    assert np.allclose(out, 0.0)
    assert out.dtype == np.float32


def test_convert_track_wetness_enum_extremely_wet_maps_to_one() -> None:
    raw = np.array([7.0, 7.0], dtype=np.float32)
    out = convert_track_wetness_enum(raw)
    assert np.allclose(out, 1.0)


def test_convert_track_wetness_enum_full_table() -> None:
    """Every documented enum value maps to the published fraction."""
    # (enum_value, expected_fraction) — must stay aligned with
    # `_TRACK_WETNESS_ENUM_TO_FRACTION` in parser.py.
    expected = {
        0: 0.00,  # Unknown (treated as dry)
        1: 0.00,  # Dry
        2: 0.10,  # Mostly Dry
        3: 0.25,  # Very Lightly Wet
        4: 0.40,  # Lightly Wet
        5: 0.60,  # Moderately Wet
        6: 0.80,  # Very Wet
        7: 1.00,  # Extremely Wet
    }
    raw = np.array(list(expected), dtype=np.float32)
    out = convert_track_wetness_enum(raw)
    assert np.allclose(out, np.array(list(expected.values()), dtype=np.float32))


def test_convert_track_wetness_enum_clips_out_of_range() -> None:
    """Malformed IBTs must not produce NaN or out-of-range fractions."""
    raw = np.array([-3.0, 0.0, 8.0, 99.0], dtype=np.float32)
    out = convert_track_wetness_enum(raw)
    assert out[0] == 0.0   # negative -> Unknown -> dry
    assert out[1] == 0.0   # 0 -> Unknown -> dry
    assert out[2] == 1.0   # over-range -> Extremely Wet
    assert out[3] == 1.0
    assert not np.isnan(out).any()


def test_convert_track_wetness_enum_rounds_intermediate_floats() -> None:
    """Float values from pyirsdk are integer-valued; defensive rounding
    handles any float-precision drift around enum boundaries."""
    raw = np.array([1.49, 1.51, 4.999], dtype=np.float32)
    out = convert_track_wetness_enum(raw)
    # 1.49 rounds to 1 (Dry -> 0.0); 1.51 rounds to 2 (Mostly Dry -> 0.10);
    # 4.999 rounds to 5 (Moderately Wet -> 0.60).
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.10, abs=1e-5)
    assert out[2] == pytest.approx(0.60, abs=1e-5)


def test_real_ibt_track_wetness_in_canonical_range(small_ibt: Path) -> None:
    """End-to-end: a parsed real IBT must expose `TrackWetness` in [0, 1].

    Pre-fix, every dry session in the corpus reported `TrackWetness == 1.0`
    because the raw enum value (1=Dry) was passed through unchanged. Pin the
    contract so a regression here can never poison `wet_mode` again.
    """
    result = parse_ibt(small_ibt)
    if "TrackWetness" not in result.channels:
        pytest.skip("fixture IBT does not expose TrackWetness")
    arr = result.channels["TrackWetness"]
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0
    if "TrackWetness_max" in result.weather_summary:
        assert 0.0 <= result.weather_summary["TrackWetness_max"] <= 1.0
