"""AeroSurface out-of-domain accounting (AUDIT H2 instrumentation).

GTPs run below the aero maps' 25 mm front-RH floor; every clamped query
silently substitutes envelope-edge aero. `AeroClampStats` makes that
visible so the recommend pipeline can warn + downgrade confidence.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from racingoptimizer.aero.interpolator import BASELINE_AIR_DENSITY, AeroSurface
from racingoptimizer.aero.loader import load_aero_map_data


@pytest.fixture
def surface(aero_dir: Path) -> AeroSurface:
    return AeroSurface(load_aero_map_data("cadillac", aero_dir=aero_dir))


def test_in_domain_query_counts_without_clamps(surface: AeroSurface) -> None:
    surface.interpolate(50.0, 25.0, 14.0, BASELINE_AIR_DENSITY)
    stats = surface.clamp_stats
    assert stats.queries == 1
    assert stats.front_clamped == 0
    assert stats.rear_clamped == 0
    assert stats.wing_clamped == 0
    assert stats.front_clamp_fraction == 0.0


def test_below_floor_front_query_records_clamp_and_excursion(
    surface: AeroSurface,
) -> None:
    # The documented Cadillac case: ~8.43 mm vs the 25 mm map floor.
    surface.interpolate(8.43, 25.0, 14.0, BASELINE_AIR_DENSITY)
    stats = surface.clamp_stats
    assert stats.queries == 1
    assert stats.front_clamped == 1
    assert stats.front_clamp_fraction == 1.0
    assert stats.max_front_excursion_mm == pytest.approx(25.0 - 8.43)


def test_excursion_tracks_the_worst_query(surface: AeroSurface) -> None:
    surface.interpolate(20.0, 25.0, 14.0, BASELINE_AIR_DENSITY)
    surface.interpolate(8.0, 25.0, 14.0, BASELINE_AIR_DENSITY)
    surface.interpolate(22.0, 25.0, 14.0, BASELINE_AIR_DENSITY)
    stats = surface.clamp_stats
    assert stats.front_clamped == 3
    assert stats.max_front_excursion_mm == pytest.approx(17.0)


def test_rear_and_wing_axes_count_independently(surface: AeroSurface) -> None:
    surface.interpolate(50.0, 1.0, 99.0, BASELINE_AIR_DENSITY)
    stats = surface.clamp_stats
    assert stats.front_clamped == 0
    assert stats.rear_clamped == 1
    assert stats.wing_clamped == 1
    assert stats.max_rear_excursion_mm == pytest.approx(4.0)


def test_reset_zeroes_the_window(surface: AeroSurface) -> None:
    surface.interpolate(8.0, 25.0, 14.0, BASELINE_AIR_DENSITY)
    surface.reset_clamp_stats()
    stats = surface.clamp_stats
    assert stats.queries == 0
    assert stats.front_clamped == 0
    assert stats.max_front_excursion_mm == 0.0
