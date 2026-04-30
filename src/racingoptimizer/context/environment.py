"""Per-sample atmospheric + track-surface context.

Master plan §2 minimum contract for slice B's `EnvironmentFrame`. Frozen so
it is hashable and safe to cache. `from_row()` is the canonical adapter
from a single Polars row (raw IBT channel names) to the dataclass.

VISION §10 channel set: 12 weather/conditions channels at 60 Hz.
- Floats (NaN sentinel for missing): air_temp_c, air_density,
  air_pressure_mbar, relative_humidity, wind_vel_ms, wind_dir_deg,
  fog_level, track_temp_c, track_wetness.
- Bool (False sentinel for missing): weather_declared_wet.
- Ints (-1 sentinel for missing): precip_type, skies.

`from_row` is strict — every channel must be present (per slice E's
"never silently zero-fill" contract). `from_partial_row` is the degraded-
mode adapter for laps/sessions that only carry a subset (e.g. older IBT
versions or synthetic test fixtures).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

# IBT channel name -> EnvironmentFrame field name. Single source of truth
# so `from_row` and `from_partial_row` stay in sync, and so the parser /
# corner aggregator can iterate over the same set without drift.
_FLOAT_CHANNELS: tuple[tuple[str, str], ...] = (
    ("AirTemp", "air_temp_c"),
    ("AirDensity", "air_density"),
    ("AirPressure", "air_pressure_mbar"),
    ("RelativeHumidity", "relative_humidity"),
    ("WindVel", "wind_vel_ms"),
    ("WindDir", "wind_dir_deg"),
    ("FogLevel", "fog_level"),
    ("TrackTempCrew", "track_temp_c"),
    ("TrackWetness", "track_wetness"),
)
_BOOL_CHANNELS: tuple[tuple[str, str], ...] = (
    ("WeatherDeclaredWet", "weather_declared_wet"),
)
_INT_CHANNELS: tuple[tuple[str, str], ...] = (
    ("Precipitation", "precip_type"),
    ("Skies", "skies"),
)


@dataclass(slots=True, frozen=True)
class EnvironmentFrame:
    # Atmospheric (floats; NaN means missing).
    air_temp_c: float = float("nan")
    air_density: float = float("nan")
    air_pressure_mbar: float = float("nan")
    relative_humidity: float = float("nan")
    wind_vel_ms: float = float("nan")
    wind_dir_deg: float = float("nan")
    fog_level: float = float("nan")
    # Track surface (floats; NaN means missing).
    track_temp_c: float = float("nan")
    track_wetness: float = float("nan")
    # Discrete weather state (sentinel-based missing).
    weather_declared_wet: bool = False
    precip_type: int = -1
    skies: int = -1

    @classmethod
    def from_row(cls, row: Mapping[str, float]) -> EnvironmentFrame:
        """Build from a row keyed by raw IBT channel names (strict).

        Raises KeyError naming the missing channel; never silently zero-fills.
        Slice E's confidence path needs to know when env was unavailable.
        """
        kwargs: dict[str, float | bool | int] = {}
        for ibt_name, field_name in _FLOAT_CHANNELS:
            kwargs[field_name] = float(row[ibt_name])
        for ibt_name, field_name in _BOOL_CHANNELS:
            kwargs[field_name] = bool(row[ibt_name])
        for ibt_name, field_name in _INT_CHANNELS:
            kwargs[field_name] = int(row[ibt_name])
        return cls(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_partial_row(cls, row: Mapping[str, float]) -> EnvironmentFrame:
        """Build from a row that may be missing channels (degraded mode).

        Missing floats fill with NaN, bools with False, ints with -1. Use
        when ingesting older IBT versions or synthetic fixtures that don't
        carry the full 12-channel set.
        """
        kwargs: dict[str, float | bool | int] = {}
        for ibt_name, field_name in _FLOAT_CHANNELS:
            value = row.get(ibt_name)
            kwargs[field_name] = float(value) if value is not None else math.nan
        for ibt_name, field_name in _BOOL_CHANNELS:
            value = row.get(ibt_name)
            kwargs[field_name] = bool(value) if value is not None else False
        for ibt_name, field_name in _INT_CHANNELS:
            value = row.get(ibt_name)
            kwargs[field_name] = int(value) if value is not None else -1
        return cls(**kwargs)  # type: ignore[arg-type]
