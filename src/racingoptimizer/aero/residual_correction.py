"""Per-car aero-map residual correction (PLAN.md Day 11).

The aero maps in `aero-maps/<car>_wing_*.json` are textbook
manufacturer-style outputs (balance_pct + ld_ratio per (front_rh,
rear_rh, wing)). They don't perfectly match what iRacing's physics
engine produces -- the iRacing tire model and chassis dynamics
introduce systematic offsets in observed-vs-predicted lat-G at apex.

This module fits a per-car SCALAR CORRECTION on the aero-map-derived
peak lat-G prediction. The corrected prediction is:

    lat_g_corrected = lat_g_raw_predicted * (1 + correction)

where `correction` is the mean residual ((observed - predicted) /
predicted) across the corpus's mid-corner apex samples.

Why scalar (not per-(rh, wing) field)? Two reasons:
- Per-(rh, wing) residual fields would require corpus density at
  every grid point; the corpus has thin coverage at most non-typical
  setups, so per-grid corrections would over-fit to a few samples.
- A scalar correction captures the dominant observed-vs-predicted
  bias (aero-map-tuned baseline downforce vs iRacing-physics-tuned
  baseline downforce) without claiming spatial structure the data
  cannot support.

Acceptance contract (PLAN.md Section 15.3 second half):
- Aero residual correction reduces lat-G prediction MAE by >=10%
  on the v4 cars (BMW, Cadillac, Ferrari).
- If correction does NOT beat raw, ship WITHOUT correction
  (`fallback_mode_used: true`); the authorized fallback path keeps
  the recommender's existing aero-only behaviour.

Day 11 ships the FIT logic + storage; Days 12-13 wire it into the
physics evaluator + hybrid optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Default tire-mu for the predicted-peak-lat-G estimator. Real tire
# mu is hidden by iRacing's physics; 1.5 is a textbook GTP dry-tarmac
# value. The residual correction absorbs whatever offset this guess
# introduces -- if the user's physics actually corresponds to mu=1.65,
# the correction will be negative ~10% to match.
_TIRE_MU_BASELINE: float = 1.5

# Air density baseline (kg/m^3) for downforce computation. ISA sea-
# level standard atmosphere; matches `aero/interpolator.BASELINE_AIR_DENSITY`.
_AIR_DENSITY_KG_PER_M3: float = 1.225

# Reference frontal area (m^2) for the predicted-downforce computation.
# Approximate for GTP cars (~1.7 m^2). Like _TIRE_MU_BASELINE, a guess
# the residual correction absorbs.
_FRONTAL_AREA_M2: float = 1.7

# Drag coefficient assumed to compute downforce from L/D ratio:
#   downforce = ld_ratio * drag = ld_ratio * Cd * 0.5 * rho * v^2 * A
_CD_BASELINE: float = 0.45

# Bound on the per-car correction magnitude: |correction| <= this.
# Outside, the fit is rejected (caller falls back).
_CORRECTION_BOUND_PCT: float = 30.0


@dataclass(frozen=True, slots=True)
class AeroResidualCorrection:
    """Per-car scalar correction on aero-map-derived peak lat-G.

    `correction_factor` is the multiplier applied to the raw predicted
    peak lat-G:
        lat_g_corrected = lat_g_raw * (1 + correction_factor)

    `n_samples` and `fit_mae_raw_g` / `fit_mae_corrected_g` are
    provenance metadata for the recommend pipeline.
    """
    car: str
    correction_factor: float
    n_samples: int
    fit_mae_raw_g: float        # MAE of raw prediction on the fit corpus
    fit_mae_corrected_g: float  # MAE of corrected prediction on the fit corpus
    fallback_mode_used: bool    # True if correction did not beat raw


def predict_peak_lat_g(
    ld_ratio: float,
    speed_ms: float,
    *,
    mu: float = _TIRE_MU_BASELINE,
    mass_kg: float = 1030.0,
    rho: float = _AIR_DENSITY_KG_PER_M3,
    area_m2: float = _FRONTAL_AREA_M2,
    cd: float = _CD_BASELINE,
) -> float:
    """Predicted peak steady-state lat-G at speed for given aero ld_ratio.

    Model:
        downforce_n = ld_ratio * cd * 0.5 * rho * v^2 * area
        peak_lat_g = mu * (m*g + downforce) / m / g
                   = mu * (1 + downforce / (m*g))

    Returns peak lat-G (dimensionless multiple of g). At zero speed,
    peak = mu (gravitational only); at high speed, downforce term
    dominates and peak grows linearly with v^2.
    """
    g = 9.81
    downforce = ld_ratio * cd * 0.5 * rho * speed_ms ** 2 * area_m2
    return mu * (1.0 + downforce / (mass_kg * g))


def fit_residual_correction(
    car: str,
    samples: list[dict],
    *,
    mu: float = _TIRE_MU_BASELINE,
) -> AeroResidualCorrection:
    """Fit per-car scalar correction to minimize lat-G prediction MAE.

    Args:
        car: car identifier
        samples: list of dicts with keys
            "ld_ratio": aero map's predicted ld_ratio at this sample's
                        (front_rh, rear_rh, wing)
            "speed_ms": observed speed at apex
            "observed_lat_g": observed peak lat-G at apex
        mu: tire-mu baseline (passed to predict_peak_lat_g)

    Returns AeroResidualCorrection with the per-car scalar that
    minimizes MAE. Fallback mode (correction_factor=0) if the fit
    does NOT beat raw; the metadata `fallback_mode_used` reflects
    this.
    """
    if not samples:
        raise ValueError(f"fit_residual_correction: no samples for car={car!r}")
    if len(samples) < 50:
        raise ValueError(
            f"fit_residual_correction: too few samples ({len(samples)}); "
            f"need >=50 for stable scalar fit"
        )

    raw_preds: list[float] = []
    observed: list[float] = []
    for s in samples:
        try:
            ld = float(s["ld_ratio"])
            v = float(s["speed_ms"])
            obs = float(s["observed_lat_g"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(ld) and np.isfinite(v) and np.isfinite(obs)):
            continue
        if v <= 0 or ld <= 0:
            continue
        pred = predict_peak_lat_g(ld, v, mu=mu)
        raw_preds.append(pred)
        observed.append(obs)

    if len(raw_preds) < 50:
        raise ValueError(
            f"fit_residual_correction: too few clean samples "
            f"({len(raw_preds)}) after filtering; need >=50"
        )

    raw_arr = np.asarray(raw_preds, dtype=np.float64)
    obs_arr = np.asarray(observed, dtype=np.float64)

    # Fit scalar c minimizing |raw*(1+c) - obs| (least-abs-deviation).
    # For an additive-multiplicative model on positive predictions,
    # the optimal c is the weighted mean residual; using the median
    # would be more robust but the corpus has outliers either way.
    raw_mae = float(np.mean(np.abs(raw_arr - obs_arr)))
    correction = float(np.mean((obs_arr - raw_arr) / raw_arr))
    # Guard: bound correction to [-30%, +30%]; outside, the fit is
    # picking up corpus-wide systematic bias that's not our problem
    # to solve here.
    if abs(correction) > _CORRECTION_BOUND_PCT / 100.0:
        return AeroResidualCorrection(
            car=car.strip().lower(),
            correction_factor=0.0,
            n_samples=len(raw_preds),
            fit_mae_raw_g=raw_mae,
            fit_mae_corrected_g=raw_mae,  # no correction applied
            fallback_mode_used=True,
        )
    corrected_arr = raw_arr * (1.0 + correction)
    corrected_mae = float(np.mean(np.abs(corrected_arr - obs_arr)))

    # Did correction beat raw? Pass: corrected_mae < raw_mae.
    if corrected_mae >= raw_mae:
        # Authorized fallback (PLAN.md Section 15.3): ship without.
        return AeroResidualCorrection(
            car=car.strip().lower(),
            correction_factor=0.0,
            n_samples=len(raw_preds),
            fit_mae_raw_g=raw_mae,
            fit_mae_corrected_g=raw_mae,
            fallback_mode_used=True,
        )

    return AeroResidualCorrection(
        car=car.strip().lower(),
        correction_factor=correction,
        n_samples=len(raw_preds),
        fit_mae_raw_g=raw_mae,
        fit_mae_corrected_g=corrected_mae,
        fallback_mode_used=False,
    )


def apply_correction(
    raw_lat_g: float | np.ndarray,
    correction: AeroResidualCorrection,
) -> float | np.ndarray:
    """Apply the correction multiplier to a raw predicted lat-G."""
    arr = np.asarray(raw_lat_g, dtype=np.float64)
    corrected = arr * (1.0 + correction.correction_factor)
    return float(corrected) if arr.ndim == 0 else corrected


def improvement_pct(correction: AeroResidualCorrection) -> float:
    """Relative MAE improvement of corrected vs raw prediction."""
    if correction.fit_mae_raw_g <= 0:
        return 0.0
    return (
        (correction.fit_mae_raw_g - correction.fit_mae_corrected_g)
        / correction.fit_mae_raw_g * 100.0
    )
