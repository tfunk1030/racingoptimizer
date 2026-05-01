"""End-to-end smoke tests against the real aero-maps/ corpus."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from racingoptimizer.aero import AeroBounds, AeroSurface, load_aero_maps
from racingoptimizer.aero.interpolator import BASELINE_AIR_DENSITY
from racingoptimizer.aero.loader import load_aero_map_data

CARS_AND_BOUNDS = [
    ("acura", (6.0, 10.0), 9),
    ("bmw", (12.0, 17.0), 6),
    ("cadillac", (12.0, 17.0), 6),
    ("ferrari", (12.0, 17.0), 6),
    ("porsche", (12.0, 17.0), 6),
]


@pytest.mark.parametrize("car, wing_range, n_wings", CARS_AND_BOUNDS)
def test_load_aero_maps_per_car_smoke(
    aero_dir: Path, car: str, wing_range: tuple[float, float], n_wings: int
) -> None:
    surf = load_aero_maps(car, aero_dir=aero_dir)
    assert isinstance(surf, AeroSurface)
    assert isinstance(surf.bounds, AeroBounds)
    assert surf.car == car
    assert surf.bounds.front_rh_mm == (25.0, 75.0)
    assert surf.bounds.rear_rh_mm == (5.0, 50.0)
    assert surf.bounds.wing_deg == wing_range
    assert len(surf.bounds.wing_angles) == n_wings


def test_porsche_full_envelope_interpolates(aero_dir: Path) -> None:
    """Spec §9 / master-plan e2e: porsche at (42.5, 22.5, 14.5°, 1.225) is
    finite, balance in [0, 100], ld > 0, and matches a hand-precomputed
    reference from the eight bracketing JSON corners within 1e-9."""
    surf = load_aero_maps("porsche", aero_dir=aero_dir)
    raw = load_aero_map_data("porsche", aero_dir=aero_dir)

    front, rear, wing = 42.5, 22.5, 14.5
    bal, ld = surf.interpolate(front, rear, wing, BASELINE_AIR_DENSITY)

    assert np.isfinite(bal) and np.isfinite(ld)
    assert 0.0 <= bal <= 100.0
    assert ld > 0.0

    fi = int(np.searchsorted(raw.front_rh_mm, front, side="right")) - 1
    ri = int(np.searchsorted(raw.rear_rh_mm, rear, side="right")) - 1
    wi = int(np.searchsorted(np.asarray(raw.wing_angles), wing, side="right")) - 1

    f0, f1 = raw.front_rh_mm[fi], raw.front_rh_mm[fi + 1]
    r0, r1 = raw.rear_rh_mm[ri], raw.rear_rh_mm[ri + 1]
    w0, w1 = raw.wing_angles[wi], raw.wing_angles[wi + 1]

    tf = (front - f0) / (f1 - f0)
    tr = (rear - r0) / (r1 - r0)
    tw = (wing - w0) / (w1 - w0)

    def trilinear(arr: np.ndarray) -> float:
        c000 = arr[wi,     fi,     ri]
        c100 = arr[wi + 1, fi,     ri]
        c010 = arr[wi,     fi + 1, ri]
        c110 = arr[wi + 1, fi + 1, ri]
        c001 = arr[wi,     fi,     ri + 1]
        c101 = arr[wi + 1, fi,     ri + 1]
        c011 = arr[wi,     fi + 1, ri + 1]
        c111 = arr[wi + 1, fi + 1, ri + 1]
        c00 = c000 * (1 - tw) + c100 * tw
        c10 = c010 * (1 - tw) + c110 * tw
        c01 = c001 * (1 - tw) + c101 * tw
        c11 = c011 * (1 - tw) + c111 * tw
        c0 = c00 * (1 - tf) + c10 * tf
        c1 = c01 * (1 - tf) + c11 * tf
        return c0 * (1 - tr) + c1 * tr

    expected_bal = trilinear(raw.balance_pct)
    expected_ld = trilinear(raw.ld_ratio)

    assert bal == pytest.approx(expected_bal, abs=1e-9)
    assert ld == pytest.approx(expected_ld, abs=1e-9)


def test_negative_query_does_not_raise(aero_dir: Path) -> None:
    """Out-of-envelope on every axis still returns finite numbers."""
    surf = load_aero_maps("porsche", aero_dir=aero_dir)
    bal, ld = surf.interpolate(200.0, 200.0, 0.0, BASELINE_AIR_DENSITY)
    assert np.isfinite(bal) and np.isfinite(ld)


def test_default_aero_dir_resolves_to_repo_root() -> None:
    """Calling load_aero_maps with no aero_dir should hit the repo's
    aero-maps/ directory."""
    surf = load_aero_maps("porsche")
    assert surf.car == "porsche"
    assert surf.bounds.front_rh_mm == (25.0, 75.0)
