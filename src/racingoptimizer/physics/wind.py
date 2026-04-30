"""Wind directional decomposition for asymmetric aero correction.

VISION §10: headwind/tailwind on a straight scales downforce up/down;
crosswind shifts aero balance left/right. The aero map's ``air_density``
correction is symmetric — this module provides the directional component.

This module is intentionally self-contained. It is **not** wired into
``racingoptimizer.physics.score.aero_eff`` in this slice (S4.6); a future
PR (Stage 5 polish) will route the modifier into the aero-utilization
branch of the scorer. Until then, ``decompose_wind`` and
``aero_wind_modifier`` are available as pure helpers for downstream code
or notebooks that want directional wind correction.
"""
from __future__ import annotations

import math


def decompose_wind(
    wind_vel_ms: float, wind_dir_deg: float, car_heading_deg: float
) -> tuple[float, float]:
    """Decompose ambient wind into car-frame headwind / crosswind components.

    Parameters
    ----------
    wind_vel_ms:
        Wind speed magnitude (m/s). Always non-negative.
    wind_dir_deg:
        Direction the wind is blowing **from**, in compass degrees
        (0 = north, 90 = east). This matches iRacing's ``WindDir``
        convention.
    car_heading_deg:
        Direction the car's nose is pointing, same convention.

    Returns
    -------
    (headwind_ms, crosswind_ms):
        - ``headwind_ms`` > 0 when the wind blows INTO the car's nose
          (extra downforce on a straight); < 0 for tailwind.
        - ``crosswind_ms`` > 0 when wind comes FROM the car's right
          (i.e. ambient wind blows leftward across the car, yawing the
          nose left); < 0 for left-side wind.
    """
    rel_deg = (wind_dir_deg - car_heading_deg) % 360.0
    rel_rad = math.radians(rel_deg)
    # Wind comes FROM `wind_dir_deg`. Relative angle 180° = directly behind
    # the wind source = headwind. cos(180°) = -1, so negate to get +1.
    head = -wind_vel_ms * math.cos(rel_rad)
    # Relative angle 90° = wind from the car's right (e.g. heading
    # north, wind from east).
    cross = wind_vel_ms * math.sin(rel_rad)
    return head, cross


def aero_wind_modifier(
    headwind_ms: float,
    crosswind_ms: float,
    *,
    baseline_speed_ms: float = 60.0,
) -> tuple[float, float]:
    """Convert decomposed wind into aero downforce + balance modifiers.

    Downforce scales as ``(V_air / V_baseline)²`` where ``V_air = V_car +
    headwind``. Balance shift is approximately linear in ``crosswind /
    V_baseline`` for small angles.

    Parameters
    ----------
    headwind_ms:
        Headwind component in m/s (positive = into nose).
    crosswind_ms:
        Crosswind component in m/s (positive = wind from the car's
        right; matches ``decompose_wind`` sign convention).
    baseline_speed_ms:
        Reference car speed used to normalize. Defaults to 60 m/s
        (~135 mph, a representative GTP straight-line cruise).

    Returns
    -------
    (downforce_scale, balance_shift_pct):
        - ``downforce_scale``: multiplicative factor on total downforce.
          Clamped at 0.25 to avoid pathological negatives if a tailwind
          exceeds baseline speed.
        - ``balance_shift_pct``: additive shift on aero balance %.
          Positive = front balance shifts rearward (or per local
          convention).
    """
    if baseline_speed_ms > 0:
        v_air_ratio = (baseline_speed_ms + headwind_ms) / baseline_speed_ms
        balance_shift_pct = (crosswind_ms / baseline_speed_ms) * 5.0
    else:
        v_air_ratio = 1.0
        balance_shift_pct = 0.0
    # Clamp the linear ratio at 0.5 BEFORE squaring; otherwise an extreme
    # tailwind (V_air negative) would produce a large positive scale via
    # the square. 0.5² = 0.25 floor matches the documented minimum.
    v_air_ratio = max(0.5, v_air_ratio)
    downforce_scale = v_air_ratio**2
    return downforce_scale, balance_shift_pct
