"""Tests for EnvironmentFrame dataclass and from_row adapter."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from racingoptimizer.context import EnvironmentFrame


def _row() -> dict[str, float]:
    return {
        "AirDensity": 1.225,
        "TrackTempCrew": 32.5,
        "WindVel": 3.2,
        "WindDir": 145.0,
        "TrackWetness": 0.0,
    }


def test_fields_round_trip() -> None:
    env = EnvironmentFrame(
        air_density=1.18,
        track_temp_c=28.0,
        wind_vel_ms=2.4,
        wind_dir_deg=87.5,
        track_wetness=0.1,
    )
    assert env.air_density == 1.18
    assert env.track_temp_c == 28.0
    assert env.wind_vel_ms == 2.4
    assert env.wind_dir_deg == 87.5
    assert env.track_wetness == 0.1


def test_frame_is_frozen() -> None:
    env = EnvironmentFrame(1.225, 25.0, 2.0, 90.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        env.air_density = 1.0  # type: ignore[misc]


def test_from_row_happy_path() -> None:
    env = EnvironmentFrame.from_row(_row())
    assert env == EnvironmentFrame(
        air_density=1.225,
        track_temp_c=32.5,
        wind_vel_ms=3.2,
        wind_dir_deg=145.0,
        track_wetness=0.0,
    )


def test_from_row_missing_key_raises_keyerror() -> None:
    row = _row()
    row.pop("WindDir")
    with pytest.raises(KeyError, match="WindDir"):
        EnvironmentFrame.from_row(row)


def test_from_row_ignores_extra_keys() -> None:
    row = _row()
    row["Speed"] = 72.4
    env = EnvironmentFrame.from_row(row)
    assert env.air_density == 1.225
    assert env.track_wetness == 0.0
