"""Phase-detector thresholds.

All threshold defaults come straight from the spec
(docs/superpowers/specs/2026-04-28-corner-phase-design.md §4 + §5).
`AccelLat` is consumed in m/s^2 — `G_MS2` converts to g for the lateral
thresholds whose values are quoted in g.
"""
from __future__ import annotations

from dataclasses import dataclass

from racingoptimizer.ingest.detect import normalize_car_key

G_MS2: float = 9.80665


@dataclass(frozen=True, slots=True)
class PhaseThresholds:
    lat_g_entry: float = 0.5
    lat_g_exit: float = 0.3
    exit_hold_ms: int = 200
    min_corner_duration_ms: int = 400
    min_gap_ms: int = 200
    sample_rate_hz: int = 60
    brake_threshold: float = 0.05
    brake_off_threshold: float = 0.02
    brake_off_hold_ms: int = 50
    steering_active_rad: float = 0.05
    throttle_active_threshold: float = 0.10
    throttle_straight_threshold: float = 0.50
    accel_lat_decreasing_window_ms: int = 100


DEFAULT_THRESHOLDS = PhaseThresholds()

PER_CAR: dict[str, PhaseThresholds] = {}


def thresholds_for(car: str) -> PhaseThresholds:
    return PER_CAR.get(normalize_car_key(car), DEFAULT_THRESHOLDS)


def ms_to_samples(ms: int, sample_rate_hz: int, *, minimum: int = 0) -> int:
    return max(minimum, int(round(ms * sample_rate_hz / 1000)))
