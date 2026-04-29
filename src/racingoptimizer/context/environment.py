"""Per-sample atmospheric + track-surface context.

Master plan §2 minimum contract for slice B's `EnvironmentFrame`. Frozen so
it is hashable and safe to cache. `from_row()` is the canonical adapter
from a single Polars row (raw IBT channel names) to the dataclass.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class EnvironmentFrame:
    air_density: float
    track_temp_c: float
    wind_vel_ms: float
    wind_dir_deg: float
    track_wetness: float

    @classmethod
    def from_row(cls, row: Mapping[str, float]) -> EnvironmentFrame:
        """Build from a row keyed by raw IBT channel names.

        Raises KeyError naming the missing channel; never silently zero-fills.
        Slice E's confidence path needs to know when env was unavailable.
        """
        return cls(
            air_density=row["AirDensity"],
            track_temp_c=row["TrackTempCrew"],
            wind_vel_ms=row["WindVel"],
            wind_dir_deg=row["WindDir"],
            track_wetness=row["TrackWetness"],
        )
