"""Phantom corner-0 guardrail filter.

Regression for ``docs/accuracy-rebuild-2026-05-24/PLAN.md`` P0.4.
The start/finish straight (and pit-out segments) occasionally leak
into the per-track schedule as ``corner_id=0`` with all five phases.
Those slots produce nonsense guardrail spam like
``T0 mid_corner: physics-vs-surrogate divergence`` because the
surrogate's predicted lat-G on straight-line features sits near
zero. The fix is an archetype-based predicate; this module pins
that filter.
"""
from __future__ import annotations

from racingoptimizer.physics.corner_schedule import (
    CornerScheduleEntry,
    is_real_corner_archetype,
)


def _straight_archetype() -> dict[str, float]:
    """Archetype that mimics a start/finish straight slot."""
    return {
        "corner_apex_speed_ms": 80.0,
        "corner_peak_lat_g": 0.05,
        "corner_max_speed_ms": 82.0,
        "corner_min_speed_ms": 80.0,
        "corner_duration_s": 10.0,
        "corner_compression_demand_mms": 0.0,
        "phase_duration_s": 2.0,
    }


def _real_corner_archetype() -> dict[str, float]:
    """Archetype that mimics a real medium-speed corner."""
    return {
        "corner_apex_speed_ms": 35.0,
        "corner_peak_lat_g": 1.8,
        "corner_max_speed_ms": 65.0,
        "corner_min_speed_ms": 33.0,
        "corner_duration_s": 4.5,
        "corner_compression_demand_mms": 60.0,
        "phase_duration_s": 1.2,
    }


def test_straight_archetype_rejected() -> None:
    assert is_real_corner_archetype(_straight_archetype()) is False


def test_real_corner_archetype_accepted() -> None:
    assert is_real_corner_archetype(_real_corner_archetype()) is True


def test_missing_archetype_rejected() -> None:
    assert is_real_corner_archetype(None) is False
    assert is_real_corner_archetype({}) is False


def test_corner_with_marginal_lat_g_below_threshold_rejected() -> None:
    archetype = _real_corner_archetype()
    archetype["corner_peak_lat_g"] = 0.30
    assert is_real_corner_archetype(archetype) is False


def test_corner_without_slowdown_rejected() -> None:
    """``apex_speed ~= max_speed`` means the car didn't slow -- not a corner."""
    archetype = _real_corner_archetype()
    archetype["corner_apex_speed_ms"] = 80.0
    archetype["corner_max_speed_ms"] = 82.0
    archetype["corner_peak_lat_g"] = 1.2  # still has lat-G (e.g., banking)
    assert is_real_corner_archetype(archetype) is False


def test_schedule_entry_with_straight_archetype_filtered() -> None:
    """End-to-end: a CornerScheduleEntry built for the straight is rejected."""
    entry = CornerScheduleEntry(
        corner_id=0, phase="mid_corner", archetype=_straight_archetype(),
    )
    assert is_real_corner_archetype(entry.archetype) is False
