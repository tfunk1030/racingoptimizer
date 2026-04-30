from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from racingoptimizer.aero.interpolator import (
    BASELINE_AIR_DENSITY,
    AeroBounds,
    AeroSurface,
)
from racingoptimizer.aero.loader import load_aero_map_data


@pytest.fixture
def porsche_surface(aero_dir: Path) -> AeroSurface:
    return AeroSurface(load_aero_map_data("porsche", aero_dir=aero_dir))


@pytest.fixture
def acura_surface(aero_dir: Path) -> AeroSurface:
    return AeroSurface(load_aero_map_data("acura", aero_dir=aero_dir))


# --- bounds ------------------------------------------------------------------

def test_bounds_envelope(porsche_surface: AeroSurface) -> None:
    b: AeroBounds = porsche_surface.bounds
    assert b.front_rh_mm == (25.0, 75.0)
    assert b.rear_rh_mm == (5.0, 50.0)
    assert b.wing_deg == (12.0, 17.0)
    assert b.wing_angles == (12.0, 13.0, 14.0, 15.0, 16.0, 17.0)


def test_bounds_acura_half_degree_steps(acura_surface: AeroSurface) -> None:
    assert acura_surface.bounds.wing_deg == (6.0, 10.0)
    assert acura_surface.bounds.wing_angles == (
        6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
    )


# --- exact node lookup -------------------------------------------------------

def test_interpolate_at_grid_node_returns_stored_value(
    porsche_surface: AeroSurface, aero_dir: Path
) -> None:
    raw = load_aero_map_data("porsche", aero_dir=aero_dir)
    fi, ri, wi = 5, 10, 1
    front = float(raw.front_rh_mm[fi])
    rear = float(raw.rear_rh_mm[ri])
    wing = float(raw.wing_angles[wi])
    bal, ld = porsche_surface.interpolate(front, rear, wing, BASELINE_AIR_DENSITY)
    assert bal == pytest.approx(raw.balance_pct[wi, fi, ri], abs=1e-12)
    assert ld == pytest.approx(raw.ld_ratio[wi, fi, ri], abs=1e-12)


# --- midway-on-wing axis -----------------------------------------------------

def test_interpolate_midway_between_wings_at_grid_node(
    porsche_surface: AeroSurface, aero_dir: Path
) -> None:
    raw = load_aero_map_data("porsche", aero_dir=aero_dir)
    fi, ri = 12, 8
    w_lo = float(raw.wing_angles[2])
    w_hi = float(raw.wing_angles[3])
    front = float(raw.front_rh_mm[fi])
    rear = float(raw.rear_rh_mm[ri])
    bal, ld = porsche_surface.interpolate(
        front, rear, 0.5 * (w_lo + w_hi), BASELINE_AIR_DENSITY
    )
    bal_ref = 0.5 * (raw.balance_pct[2, fi, ri] + raw.balance_pct[3, fi, ri])
    ld_ref = 0.5 * (raw.ld_ratio[2, fi, ri] + raw.ld_ratio[3, fi, ri])
    assert bal == pytest.approx(bal_ref, abs=1e-12)
    assert ld == pytest.approx(ld_ref, abs=1e-12)


# --- bilinear at fixed wing --------------------------------------------------

def test_interpolate_bilinear_in_rh_at_fixed_wing(
    porsche_surface: AeroSurface, aero_dir: Path
) -> None:
    raw = load_aero_map_data("porsche", aero_dir=aero_dir)
    wi = 2
    fi = 5
    ri = 10
    f0, f1 = float(raw.front_rh_mm[fi]), float(raw.front_rh_mm[fi + 1])
    r0, r1 = float(raw.rear_rh_mm[ri]), float(raw.rear_rh_mm[ri + 1])
    f_query = 0.5 * (f0 + f1)
    r_query = 0.5 * (r0 + r1)
    expected_bal = 0.25 * (
        raw.balance_pct[wi, fi, ri]
        + raw.balance_pct[wi, fi + 1, ri]
        + raw.balance_pct[wi, fi, ri + 1]
        + raw.balance_pct[wi, fi + 1, ri + 1]
    )
    expected_ld = 0.25 * (
        raw.ld_ratio[wi, fi, ri]
        + raw.ld_ratio[wi, fi + 1, ri]
        + raw.ld_ratio[wi, fi, ri + 1]
        + raw.ld_ratio[wi, fi + 1, ri + 1]
    )
    bal, ld = porsche_surface.interpolate(
        f_query, r_query, float(raw.wing_angles[wi]), BASELINE_AIR_DENSITY
    )
    assert bal == pytest.approx(expected_bal, abs=1e-12)
    assert ld == pytest.approx(expected_ld, abs=1e-12)


# --- air-density correction --------------------------------------------------

def test_air_density_baseline_returns_raw_ld(
    porsche_surface: AeroSurface, aero_dir: Path
) -> None:
    raw = load_aero_map_data("porsche", aero_dir=aero_dir)
    fi, ri, wi = 5, 10, 1
    bal, ld = porsche_surface.interpolate(
        float(raw.front_rh_mm[fi]),
        float(raw.rear_rh_mm[ri]),
        float(raw.wing_angles[wi]),
        BASELINE_AIR_DENSITY,
    )
    assert ld == pytest.approx(raw.ld_ratio[wi, fi, ri], abs=1e-12)


def test_ld_ratio_is_density_invariant(
    porsche_surface: AeroSurface, aero_dir: Path
) -> None:
    """S2.9 audit: ld_ratio is a dimensionless lift/drag ratio.

    Both lift and drag scale linearly with air density, so the ratio
    cancels and `interpolate(..., air_density)` must return the same
    `ld_ratio` regardless of the air-density argument. Callers needing
    an absolute downforce must apply rho at their own use-site (see
    `racingoptimizer.physics.score.grip`).
    """
    raw = load_aero_map_data("porsche", aero_dir=aero_dir)
    fi, ri, wi = 5, 10, 1
    front = float(raw.front_rh_mm[fi])
    rear = float(raw.rear_rh_mm[ri])
    wing = float(raw.wing_angles[wi])
    bal_base, ld_base = porsche_surface.interpolate(
        front, rear, wing, BASELINE_AIR_DENSITY
    )
    bal_2x, ld_2x = porsche_surface.interpolate(
        front, rear, wing, 2 * BASELINE_AIR_DENSITY
    )
    bal_half, ld_half = porsche_surface.interpolate(
        front, rear, wing, 0.5 * BASELINE_AIR_DENSITY
    )
    assert ld_2x == pytest.approx(ld_base, abs=1e-12)
    assert ld_half == pytest.approx(ld_base, abs=1e-12)
    assert bal_2x == pytest.approx(bal_base, abs=1e-12)
    assert bal_half == pytest.approx(bal_base, abs=1e-12)


def test_zero_or_negative_air_density_raises(porsche_surface: AeroSurface) -> None:
    with pytest.raises(ValueError):
        porsche_surface.interpolate(40.0, 20.0, 14.0, 0.0)
    with pytest.raises(ValueError):
        porsche_surface.interpolate(40.0, 20.0, 14.0, -1.0)


# --- clamp behaviour ---------------------------------------------------------

def test_out_of_envelope_front_rh_clamps_to_edge(
    porsche_surface: AeroSurface, aero_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw = load_aero_map_data("porsche", aero_dir=aero_dir)
    rear = float(raw.rear_rh_mm[10])
    wing = float(raw.wing_angles[2])
    expected_bal, expected_ld = porsche_surface.interpolate(
        75.0, rear, wing, BASELINE_AIR_DENSITY
    )
    with caplog.at_level(logging.WARNING, logger="racingoptimizer.aero"):
        bal, ld = porsche_surface.interpolate(200.0, rear, wing, BASELINE_AIR_DENSITY)
    assert bal == pytest.approx(expected_bal, abs=1e-12)
    assert ld == pytest.approx(expected_ld, abs=1e-12)
    assert any("front_rh_mm" in r.message for r in caplog.records)


def test_out_of_envelope_wing_clamps_to_edge(
    porsche_surface: AeroSurface, caplog: pytest.LogCaptureFixture
) -> None:
    front, rear = 42.5, 22.5
    expected_bal, expected_ld = porsche_surface.interpolate(
        front, rear, 17.0, BASELINE_AIR_DENSITY
    )
    with caplog.at_level(logging.WARNING, logger="racingoptimizer.aero"):
        bal, ld = porsche_surface.interpolate(front, rear, 25.0, BASELINE_AIR_DENSITY)
    assert bal == pytest.approx(expected_bal, abs=1e-12)
    assert ld == pytest.approx(expected_ld, abs=1e-12)
    assert any("wing_deg" in r.message for r in caplog.records)


def test_out_of_envelope_does_not_raise(porsche_surface: AeroSurface) -> None:
    bal, ld = porsche_surface.interpolate(1000.0, -50.0, 0.0, BASELINE_AIR_DENSITY)
    assert np.isfinite(bal) and np.isfinite(ld)


# --- car attribute -----------------------------------------------------------

def test_aero_surface_exposes_car(porsche_surface: AeroSurface) -> None:
    assert porsche_surface.car == "porsche"
