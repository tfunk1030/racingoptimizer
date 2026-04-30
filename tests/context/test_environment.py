"""Tests for EnvironmentFrame dataclass and from_row / from_partial_row adapters.

VISION section 10 expanded the env contract from 5 channels to 12 (S2.2).
The strict `from_row` raises on any missing channel; `from_partial_row`
is the degraded-mode adapter for older IBT versions / synthetic fixtures
that don't carry every channel.
"""
from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from racingoptimizer.context import EnvironmentFrame


def _row() -> dict[str, float | bool | int]:
    """Full 12-channel IBT row matching VISION section 10."""
    return {
        "AirTemp": 22.0,
        "AirDensity": 1.225,
        "AirPressure": 1013.0,
        "RelativeHumidity": 0.55,
        "WindVel": 3.2,
        "WindDir": 145.0,
        "FogLevel": 0.0,
        "TrackTempCrew": 32.5,
        "TrackWetness": 0.0,
        "WeatherDeclaredWet": False,
        "Precipitation": 0,
        "Skies": 1,
    }


def test_fields_round_trip() -> None:
    env = EnvironmentFrame(
        air_temp_c=22.0,
        air_density=1.18,
        air_pressure_mbar=1015.0,
        relative_humidity=0.6,
        wind_vel_ms=2.4,
        wind_dir_deg=87.5,
        fog_level=0.0,
        track_temp_c=28.0,
        track_wetness=0.1,
        weather_declared_wet=False,
        precip_type=0,
        skies=1,
    )
    assert env.air_temp_c == 22.0
    assert env.air_density == 1.18
    assert env.air_pressure_mbar == 1015.0
    assert env.relative_humidity == 0.6
    assert env.wind_vel_ms == 2.4
    assert env.wind_dir_deg == 87.5
    assert env.fog_level == 0.0
    assert env.track_temp_c == 28.0
    assert env.track_wetness == 0.1
    assert env.weather_declared_wet is False
    assert env.precip_type == 0
    assert env.skies == 1


def test_default_constructor_uses_missing_sentinels() -> None:
    """Bare `EnvironmentFrame()` is the all-missing degraded sentinel.

    Floats default to NaN, the bool flag to False, the int channels to -1.
    Slice E inspects these to decide whether env was actually present.
    """
    env = EnvironmentFrame()
    for f in (
        env.air_temp_c, env.air_density, env.air_pressure_mbar,
        env.relative_humidity, env.wind_vel_ms, env.wind_dir_deg,
        env.fog_level, env.track_temp_c, env.track_wetness,
    ):
        assert math.isnan(f), "float channels default to NaN sentinel"
    assert env.weather_declared_wet is False
    assert env.precip_type == -1
    assert env.skies == -1


def test_frame_is_frozen() -> None:
    env = EnvironmentFrame(air_density=1.225, track_temp_c=25.0)
    with pytest.raises(FrozenInstanceError):
        env.air_density = 1.0  # type: ignore[misc]


def test_from_row_happy_path() -> None:
    env = EnvironmentFrame.from_row(_row())
    assert env.air_temp_c == 22.0
    assert env.air_density == 1.225
    assert env.air_pressure_mbar == 1013.0
    assert env.relative_humidity == 0.55
    assert env.wind_vel_ms == 3.2
    assert env.wind_dir_deg == 145.0
    assert env.fog_level == 0.0
    assert env.track_temp_c == 32.5
    assert env.track_wetness == 0.0
    assert env.weather_declared_wet is False
    assert env.precip_type == 0
    assert env.skies == 1


def test_from_row_missing_key_raises_keyerror() -> None:
    """Strict adapter — every channel must be present."""
    row = _row()
    row.pop("WindDir")
    with pytest.raises(KeyError, match="WindDir"):
        EnvironmentFrame.from_row(row)


def test_from_row_missing_new_channel_raises_keyerror() -> None:
    """Strictness covers the seven channels added in S2.2."""
    row = _row()
    row.pop("RelativeHumidity")
    with pytest.raises(KeyError, match="RelativeHumidity"):
        EnvironmentFrame.from_row(row)


def test_from_row_missing_skies_raises_keyerror() -> None:
    row = _row()
    row.pop("Skies")
    with pytest.raises(KeyError, match="Skies"):
        EnvironmentFrame.from_row(row)


def test_from_row_ignores_extra_keys() -> None:
    row = _row()
    row["Speed"] = 72.4
    env = EnvironmentFrame.from_row(row)
    assert env.air_density == 1.225
    assert env.track_wetness == 0.0


def test_from_partial_row_fills_missing_with_sentinels() -> None:
    """Degraded-mode adapter fills missing channels with NaN/False/-1."""
    partial = {
        "AirDensity": 1.225,
        "TrackTempCrew": 32.5,
        "WindVel": 3.2,
        "WindDir": 145.0,
        "TrackWetness": 0.0,
    }
    env = EnvironmentFrame.from_partial_row(partial)
    # Provided floats survive.
    assert env.air_density == 1.225
    assert env.track_temp_c == 32.5
    assert env.wind_vel_ms == 3.2
    assert env.wind_dir_deg == 145.0
    assert env.track_wetness == 0.0
    # Missing floats become NaN.
    assert math.isnan(env.air_temp_c)
    assert math.isnan(env.air_pressure_mbar)
    assert math.isnan(env.relative_humidity)
    assert math.isnan(env.fog_level)
    # Missing bool / int channels get the sentinel.
    assert env.weather_declared_wet is False
    assert env.precip_type == -1
    assert env.skies == -1


def test_from_partial_row_full_row_matches_from_row() -> None:
    """Given a full row, the partial adapter agrees with the strict adapter."""
    row = _row()
    assert EnvironmentFrame.from_partial_row(row) == EnvironmentFrame.from_row(row)


def test_from_partial_row_empty_is_all_missing() -> None:
    """An empty row degrades to the all-sentinel default frame.

    NaN != NaN by IEEE-754, so we cannot use `==` against the default
    constructor; check each field individually.
    """
    env = EnvironmentFrame.from_partial_row({})
    for f in (
        env.air_temp_c, env.air_density, env.air_pressure_mbar,
        env.relative_humidity, env.wind_vel_ms, env.wind_dir_deg,
        env.fog_level, env.track_temp_c, env.track_wetness,
    ):
        assert math.isnan(f)
    assert env.weather_declared_wet is False
    assert env.precip_type == -1
    assert env.skies == -1
