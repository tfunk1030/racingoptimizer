"""AeroMapData -> AeroSurface. Per-wing 2D RegularGridInterpolator + linear
blend on the wing axis + per-call air-density correction.

Out-of-envelope inputs clamp to the nearest grid edge; one warning per axis
that clamps. Calls never raise on geometry — only on physically-invalid
air density.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from racingoptimizer.aero.loader import AeroMapData

logger = logging.getLogger("racingoptimizer.aero")

# ISA sea-level standard atmosphere. Spec §6: per-car overrides land when a
# corpus-mean reference is available from slice A.
BASELINE_AIR_DENSITY: float = 1.225


@dataclass(frozen=True)
class AeroBounds:
    front_rh_mm: tuple[float, float]
    rear_rh_mm: tuple[float, float]
    wing_deg: tuple[float, float]
    wing_angles: tuple[float, ...]


def _clamp(value: float, lo: float, hi: float) -> tuple[float, bool]:
    """Clamp value to [lo, hi]; second return is True if clamping happened."""
    if value < lo:
        return lo, True
    if value > hi:
        return hi, True
    return value, False


class AeroSurface:
    """Queryable aero surface for one car across all loaded wing angles.

    Caches one (balance, ld_ratio) RegularGridInterpolator pair per wing slice;
    a query bracket-searches the wing axis, evaluates the two bracketing 2D
    interpolators, and linearly blends.
    """

    def __init__(
        self,
        data: AeroMapData,
        *,
        baseline_air_density: float = BASELINE_AIR_DENSITY,
    ) -> None:
        self._data = data
        self._baseline_air_density = baseline_air_density
        self._wing_axis = np.asarray(data.wing_angles, dtype=float)

        self._balance_interps: list[RegularGridInterpolator] = []
        self._ld_interps: list[RegularGridInterpolator] = []
        for wi in range(len(data.wing_angles)):
            self._balance_interps.append(
                RegularGridInterpolator(
                    (data.front_rh_mm, data.rear_rh_mm),
                    data.balance_pct[wi],
                    method="linear",
                    bounds_error=False,
                    fill_value=None,
                )
            )
            self._ld_interps.append(
                RegularGridInterpolator(
                    (data.front_rh_mm, data.rear_rh_mm),
                    data.ld_ratio[wi],
                    method="linear",
                    bounds_error=False,
                    fill_value=None,
                )
            )

    @property
    def car(self) -> str:
        return self._data.car

    @property
    def baseline_air_density(self) -> float:
        return self._baseline_air_density

    @property
    def bounds(self) -> AeroBounds:
        d = self._data
        return AeroBounds(
            front_rh_mm=(float(d.front_rh_mm[0]), float(d.front_rh_mm[-1])),
            rear_rh_mm=(float(d.rear_rh_mm[0]), float(d.rear_rh_mm[-1])),
            wing_deg=(float(d.wing_angles[0]), float(d.wing_angles[-1])),
            wing_angles=tuple(float(w) for w in d.wing_angles),
        )

    def interpolate(
        self,
        front_rh_mm: float,
        rear_rh_mm: float,
        wing_deg: float,
        air_density: float,
    ) -> tuple[float, float]:
        """Return (balance_pct, ld_ratio_corrected) at the queried point."""
        if air_density <= 0:
            raise ValueError(f"air_density must be > 0, got {air_density!r}")

        d = self._data

        front_clamped, front_was_clamped = _clamp(
            float(front_rh_mm), float(d.front_rh_mm[0]), float(d.front_rh_mm[-1])
        )
        rear_clamped, rear_was_clamped = _clamp(
            float(rear_rh_mm), float(d.rear_rh_mm[0]), float(d.rear_rh_mm[-1])
        )
        wing_clamped, wing_was_clamped = _clamp(
            float(wing_deg), float(self._wing_axis[0]), float(self._wing_axis[-1])
        )

        if front_was_clamped:
            logger.warning(
                "front_rh_mm=%s out of envelope %s for car %s; clamped to %s",
                front_rh_mm,
                (float(d.front_rh_mm[0]), float(d.front_rh_mm[-1])),
                d.car,
                front_clamped,
            )
        if rear_was_clamped:
            logger.warning(
                "rear_rh_mm=%s out of envelope %s for car %s; clamped to %s",
                rear_rh_mm,
                (float(d.rear_rh_mm[0]), float(d.rear_rh_mm[-1])),
                d.car,
                rear_clamped,
            )
        if wing_was_clamped:
            logger.warning(
                "wing_deg=%s out of envelope %s for car %s; clamped to %s",
                wing_deg,
                (float(self._wing_axis[0]), float(self._wing_axis[-1])),
                d.car,
                wing_clamped,
            )

        # Bracket wing axis. searchsorted with 'right' returns the first index
        # strictly greater; subtract 1 for the lower bracket. clamp index to
        # [0, n-2] so idx+1 is always valid.
        idx = int(np.searchsorted(self._wing_axis, wing_clamped, side="right")) - 1
        idx = max(0, min(idx, len(self._wing_axis) - 2))
        w_lo = float(self._wing_axis[idx])
        w_hi = float(self._wing_axis[idx + 1])
        t = 0.0 if w_hi == w_lo else (wing_clamped - w_lo) / (w_hi - w_lo)

        rh_query = np.array([[front_clamped, rear_clamped]])
        bal_lo = float(self._balance_interps[idx](rh_query)[0])
        bal_hi = float(self._balance_interps[idx + 1](rh_query)[0])
        ld_lo = float(self._ld_interps[idx](rh_query)[0])
        ld_hi = float(self._ld_interps[idx + 1](rh_query)[0])

        balance = (1.0 - t) * bal_lo + t * bal_hi
        ld_raw = (1.0 - t) * ld_lo + t * ld_hi

        ld_corrected = ld_raw * (air_density / self._baseline_air_density)
        return balance, ld_corrected
