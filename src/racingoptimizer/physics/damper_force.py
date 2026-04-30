"""Damper force estimation per VISION §2.

iRacing IBT files expose shock velocity but not damper force. We estimate
force via a per-car damper curve: F = damper_coefficient(velocity) where
the coefficient is approximately linear in low-velocity regime and flattens
at high velocity (digressive damper behaviour). Coefficients are seeded
per-car as engineering estimates pending real damper-spec data from
iRacing's garage tooltips.

Per spec: this is a Stage-4 stepping-stone. Real damper curves come from
per-car damper-spec tables (TODO: capture from iRacing UI).
"""
from __future__ import annotations

import numpy as np

# Per-car damper coefficient (N*s / mm) at low velocity.
# Seeded conservatively; real values vary 4-8 N*s/mm low-speed.
DAMPER_COEFFICIENT_NS_PER_MM: dict[str, float] = {
    "bmw": 6.0,
    "acura": 5.5,
    "cadillac": 6.5,
    "ferrari": 6.0,
    "porsche": 6.5,
}
DEFAULT_DAMPER_COEFFICIENT_NS_PER_MM: float = 6.0

# Velocity (mm/s) at which the digressive curve transitions to high-speed.
# Above this, force scales as sqrt(velocity) instead of linear.
DIGRESSIVE_KNEE_MM_S: float = 100.0


def damper_coefficient(car: str | None) -> float:
    """Return the per-car low-velocity damper coefficient (N*s/mm)."""
    if car is None:
        return DEFAULT_DAMPER_COEFFICIENT_NS_PER_MM
    return DAMPER_COEFFICIENT_NS_PER_MM.get(car.lower(), DEFAULT_DAMPER_COEFFICIENT_NS_PER_MM)


def estimate_damper_force_n(
    velocity_mm_s: np.ndarray, *, car: str | None = None
) -> np.ndarray:
    """Estimate damper force (N) from velocity (mm/s) per a digressive curve.

    F(v) = k * v                                                     for |v| < knee
    F(v) = k * knee * (1 + sqrt((|v| - knee) / knee)) * sign(v)      for |v| >= knee

    Where k is the per-car low-velocity coefficient.
    """
    velocity_mm_s = np.asarray(velocity_mm_s, dtype=np.float64)
    k = damper_coefficient(car)
    abs_v = np.abs(velocity_mm_s)
    low_speed = abs_v < DIGRESSIVE_KNEE_MM_S
    high_speed = ~low_speed
    force = np.empty_like(velocity_mm_s)
    force[low_speed] = k * velocity_mm_s[low_speed]
    force[high_speed] = (
        k
        * DIGRESSIVE_KNEE_MM_S
        * (1.0 + np.sqrt((abs_v[high_speed] - DIGRESSIVE_KNEE_MM_S) / DIGRESSIVE_KNEE_MM_S))
        * np.sign(velocity_mm_s[high_speed])
    )
    return force
