"""Per-axle grip-margin model (PLAN.md Day 10, Mode 5 -- replaces Pacejka).

Per Reviewer Agent 1's veto: fitting a 4-parameter Pacejka tire model
from iRacing telemetry alone is circular (no measured tire forces; slip
angles and Fz are model-derived). The replacement is a simpler
*axle-grip-margin* model: ONE FIT PER AXLE PER CAR using observed
(axle Fz, axle Fy) extremes from the chassis-level force decomposition
that Day 8's diagnostic_state module produces.

Model:
    ratio_peak = percentile_p(Fy_axle / Fz_axle) over observed samples
    grip_margin(sample) = (|Fy| / Fz) / ratio_peak    in [0, 1+]

The variable is named `mu_peak` in the dataclass for shorthand, but
**this is NOT a tire friction coefficient.** It is an *axle utilization
ratio* derived from chassis-level force decomposition. Real tire mu on
dry tarmac peaks at 1.4-1.8; observed `mu_peak` values often sit
2.5-3.0 because the chassis-level Fz used in the denominator does NOT
include aero downforce (we set it to 0 in the static fit), so the
ratio Fy/Fz_static appears higher than the physical tire mu. The
ratio is what we want for the grip-margin question ("how close to the
user's empirical operating limit?"); calling it "mu" is shorthand,
not a claim about underlying tire physics.

`grip_margin = 1.0` means the sample is at the empirical grip limit;
`< 1.0` means underutilised; `> 1.0` means the model has been seeing
ratios it hadn't seen before (a sign that the corpus has expanded since
the ceiling was fitted, OR a transient like curb-strike that
genuinely exceeded steady-state grip).

This is intentionally simpler than Pacejka:
- ONE PARAMETER per axle (`mu_axle_peak`) instead of 4 Pacejka coefs.
- IDENTIFIABLE: directly fitted from a single ratio per sample (not
  three derived quantities all coupled).
- USEFUL: the only thing the recommender / renderer needs is "how
  close to grip limit was the user when they ran this setup at this
  corner?" -- which is exactly what grip-margin answers.

The "axle force ceiling" idea is standard race-engineering practice:
it's how engineers reason about whether one axle is the limiting axle
for cornering speed.

Implementation note: Day 8's `axle_force_split` returns a single
`AxleForceSplit` per (lat_g, long_g) sample. The grip-margin pipeline
calls it across every sample in a corner-phase to build the (Fz, Fy)
distribution per axle. Aggregating ratios to per-axle means and
percentiles gives the ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from racingoptimizer.physics.diagnostic_state import (
    axle_force_split,
    get_car_geometry,
)

# Percentile of the (Fy / Fz) ratio distribution that defines the per-
# axle grip ceiling. 99 is robust to single-sample noise (outlier curb-
# strikes don't dominate); 100 (true max) is too sensitive. A 99-th-pct
# ceiling gives margin = 1.0 on the second-fastest cornering sample,
# which is the right "this is the grip limit" semantic.
_CEILING_PERCENTILE: float = 99.0

# Physical-range bounds on per-axle peak mu. Race tires on dry tarmac
# typically peak at mu = 1.4-1.8; oval tires can hit 2.0+. Below 0.5
# is implausibly low (suggests the fit was on a wet stint or a sample
# with extreme aero downforce); above 3.0 is implausibly high (suggests
# a tiny Fz divisor noise). Outside [0.5, 3.0] -> reject.
_MU_MIN: float = 0.5
_MU_MAX: float = 3.0

# Lateral-G threshold for "loaded cornering" sample inclusion. Below
# this, the tire is essentially in straight-line and the ratio Fy/Fz
# is dominated by sensor noise. 0.5 G is the same threshold as Day 8's
# `min_lat_g` for the β-steering correlation (mid-corner samples).
_MIN_LAT_G_FOR_FIT: float = 0.5


@dataclass(frozen=True, slots=True)
class AxleGripCeiling:
    """Per-(car, axle) grip ceiling fitted from observed Fy/Fz ratios.

    `axle` is "front" or "rear". `mu_peak` is the percentile-anchored
    peak ratio. `n_samples` is the count of mid-corner samples that
    contributed; `n_above_ceiling` is the count of samples above the
    99th-percentile (sanity check: should be ~1% of n_samples).
    """
    car: str
    axle: str  # "front" | "rear"
    mu_peak: float
    n_samples: int
    n_above_ceiling: int
    percentile_used: float


def compute_axle_grip_ratios(
    lat_g_samples: np.ndarray,
    long_g_samples: np.ndarray,
    car: str,
    *,
    aero_downforce_n_per_axle: float = 0.0,
) -> dict[str, np.ndarray]:
    """Compute per-axle Fy/Fz ratio for each sample.

    For each sample with (lat_g, long_g), call Day 8's
    `axle_force_split` to decompose into per-axle (Fz, Fy). Return:
        {"front": ratios_front, "rear": ratios_rear}
    where ratios are |Fy| / Fz per sample.

    `aero_downforce_n_per_axle` is split equally front/rear; the caller
    can refine via the aero map at predict time. For the ceiling fit
    (running across many samples) the aero downforce is noise relative
    to the gravitational-static Fz, so a constant (or zero) is fine.
    """
    geom = get_car_geometry(car)
    lat = np.asarray(lat_g_samples, dtype=np.float64)
    lon = np.asarray(long_g_samples, dtype=np.float64)
    n = lat.size
    front_ratios = np.zeros(n)
    rear_ratios = np.zeros(n)
    for i in range(n):
        split = axle_force_split(
            lat_accel_g=float(lat[i]),
            long_accel_g=float(lon[i]),
            aero_downforce_n_front=aero_downforce_n_per_axle,
            aero_downforce_n_rear=aero_downforce_n_per_axle,
            geometry=geom,
        )
        if split.fz_front_n > 0:
            front_ratios[i] = abs(split.fy_front_n) / split.fz_front_n
        if split.fz_rear_n > 0:
            rear_ratios[i] = abs(split.fy_rear_n) / split.fz_rear_n
    return {"front": front_ratios, "rear": rear_ratios}


def fit_axle_grip_ceiling(
    car: str,
    axle: str,
    ratios: np.ndarray,
    *,
    percentile: float = _CEILING_PERCENTILE,
) -> AxleGripCeiling:
    """Fit the per-(car, axle) ceiling from the observed Fy/Fz ratios.

    Computes the percentile-anchored peak ratio. Rejects fits with mu
    outside [_MU_MIN, _MU_MAX] -- caller should fall back to a per-
    car default (e.g. 1.5 for dry GTP tires).
    """
    if axle not in ("front", "rear"):
        raise ValueError(f"axle must be 'front' or 'rear'; got {axle!r}")
    arr = np.asarray(ratios, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < 100:
        raise ValueError(
            f"fit_axle_grip_ceiling: too few samples ({arr.size}); need >=100"
        )
    mu = float(np.percentile(arr, percentile))
    if not (_MU_MIN <= mu <= _MU_MAX):
        raise ValueError(
            f"fit_axle_grip_ceiling: mu={mu:.3f} outside physical range "
            f"[{_MU_MIN}, {_MU_MAX}] for ({car}, {axle}); "
            f"caller should fall back to a default"
        )
    n_above = int(np.sum(arr >= mu))
    return AxleGripCeiling(
        car=car.strip().lower(),
        axle=axle,
        mu_peak=mu,
        n_samples=int(arr.size),
        n_above_ceiling=n_above,
        percentile_used=float(percentile),
    )


def axle_grip_margin(
    observed_ratio: float | np.ndarray,
    ceiling: AxleGripCeiling,
) -> float | np.ndarray:
    """Convert an observed Fy/Fz ratio to grip-margin in [0, 1+].

    margin = observed_ratio / ceiling.mu_peak
    margin = 1.0 means at the empirical grip limit
    margin > 1.0 means above the fitted ceiling (unusual; possibly
    transient or corpus expansion since fit)
    margin < 1.0 means underutilised
    """
    if ceiling.mu_peak <= 0:
        raise ValueError("ceiling.mu_peak must be > 0")
    arr = np.asarray(observed_ratio, dtype=np.float64)
    return arr / ceiling.mu_peak if arr.ndim > 0 else float(arr) / ceiling.mu_peak


def fit_axles_from_lap(
    lat_g_samples: np.ndarray,
    long_g_samples: np.ndarray,
    car: str,
    *,
    min_lat_g: float = _MIN_LAT_G_FOR_FIT,
) -> dict[str, AxleGripCeiling]:
    """Convenience: filter to mid-corner samples then fit both axles.

    Returns dict with keys "front" and "rear" (each an AxleGripCeiling).
    Raises ValueError if either axle fit fails -- caller may need to
    pool samples across multiple laps.
    """
    lat = np.asarray(lat_g_samples, dtype=np.float64)
    lon = np.asarray(long_g_samples, dtype=np.float64)
    mid_mask = np.abs(lat) >= min_lat_g
    if int(np.sum(mid_mask)) < 100:
        raise ValueError(
            f"fit_axles_from_lap: only {int(np.sum(mid_mask))} mid-corner "
            f"samples (|lat_g|>={min_lat_g}); need >=100"
        )
    ratios = compute_axle_grip_ratios(lat[mid_mask], lon[mid_mask], car)
    return {
        "front": fit_axle_grip_ceiling(car, "front", ratios["front"]),
        "rear": fit_axle_grip_ceiling(car, "rear", ratios["rear"]),
    }


def predict_corner_at_limit(
    lat_g_corner: float,
    long_g_corner: float,
    car: str,
    front_ceiling: AxleGripCeiling,
    rear_ceiling: AxleGripCeiling,
    *,
    threshold: float = 0.90,
) -> dict[str, bool | float]:
    """Predict whether a single corner exceeded `threshold`% of either
    axle's grip ceiling.

    Returns dict with:
        front_margin: float
        rear_margin: float
        at_limit: bool (true if either margin >= threshold)
        limiting_axle: "front" | "rear" | None
    """
    geom = get_car_geometry(car)
    split = axle_force_split(
        lat_accel_g=lat_g_corner, long_accel_g=long_g_corner,
        aero_downforce_n_front=0.0, aero_downforce_n_rear=0.0,
        geometry=geom,
    )
    front_ratio = (
        abs(split.fy_front_n) / split.fz_front_n if split.fz_front_n > 0 else 0.0
    )
    rear_ratio = (
        abs(split.fy_rear_n) / split.fz_rear_n if split.fz_rear_n > 0 else 0.0
    )
    front_margin = front_ratio / front_ceiling.mu_peak
    rear_margin = rear_ratio / rear_ceiling.mu_peak
    at_limit = (front_margin >= threshold) or (rear_margin >= threshold)
    if at_limit:
        limiting = "front" if front_margin >= rear_margin else "rear"
    else:
        limiting = None
    return {
        "front_margin": float(front_margin),
        "rear_margin": float(rear_margin),
        "at_limit": at_limit,
        "limiting_axle": limiting,
    }
