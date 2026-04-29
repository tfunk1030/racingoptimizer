"""Aero-map loader and interpolator: ride-height + wing -> (balance, l/d).

Public API:
    load_aero_maps(car, *, aero_dir=None) -> AeroSurface
    AeroSurface.interpolate(front_rh_mm, rear_rh_mm, wing_deg, air_density)
    AeroSurface.bounds -> AeroBounds
"""
from __future__ import annotations

from pathlib import Path

from racingoptimizer.aero.interpolator import (
    BASELINE_AIR_DENSITY,
    AeroBounds,
    AeroSurface,
)
from racingoptimizer.aero.loader import (
    AeroLoadError,
    AeroMapData,
    load_aero_map_data,
)

__all__ = [
    "AeroBounds",
    "AeroLoadError",
    "AeroMapData",
    "AeroSurface",
    "BASELINE_AIR_DENSITY",
    "load_aero_maps",
]


def _default_aero_dir() -> Path:
    """Repo-relative aero-maps/. __init__.py lives at
    .../src/racingoptimizer/aero/__init__.py — repo root is four parents up."""
    return Path(__file__).resolve().parents[3] / "aero-maps"


def load_aero_maps(car: str, *, aero_dir: Path | None = None) -> AeroSurface:
    """Load every aero-maps/<car>_wing_*.json and wrap in an AeroSurface."""
    root = Path(aero_dir) if aero_dir is not None else _default_aero_dir()
    return AeroSurface(load_aero_map_data(car, aero_dir=root))
