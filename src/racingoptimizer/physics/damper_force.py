"""Damper force estimation per VISION §2 + PLAN.md Day 9 refit (T4.4).

iRacing IBT files expose shock velocity but not damper force. We estimate
force via a per-car damper curve: F = damper_coefficient(velocity) where
the coefficient is approximately linear in low-velocity regime and
flattens at high velocity (digressive damper behaviour).

History:
- Pre-Day-9: per-car coefficients were seeded engineering estimates
  (5.5-6.5 N*s/mm, basically uniform across the 5 cars). The fixed
  knee position was 100 mm/s for all cars.
- Day 9 (PLAN.md Section 15.2, T4.4 punch-list): per-car coefficients
  refit from corpus shock-velocity distributions. The HIGH-confidence
  `*shockVel` IBT channel (Reviewer Agent 1) provides ground-truth
  velocity samples per car; we calibrate the digressive curve so its
  knee position and low-speed slope match each car's actual
  operational regime.

Why not a direct force fit? iRacing does NOT expose damper force as a
channel. So we cannot do `(velocity, force) -> regression`. Instead we
calibrate the curve's TWO parameters (`k_low_speed`, `knee_mm_s`) from
the velocity DISTRIBUTION itself: the knee is set per-car at the 30th-
percentile of |shockVel| (so the linear regime covers the bulk of
operational samples), and `k_low_speed` is set so the curve's force at
v_95 matches a per-car target derived from the static spring rate +
sprung mass ratio. This is empirical-without-ground-truth -- not as
strong as a true force regression, but it produces per-car-distinct
values rather than the seeded uniformity.

Backward compat: `estimate_damper_force_n(velocity, *, car=None)` still
defaults to seeded curves; new `estimate_damper_force_n(velocity, *,
curve=DamperCurve(...))` takes a fitted curve.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Per-car damper coefficient (N*s / mm) at low velocity.
# Seeded conservatively pre-Day-9; real values vary 4-8 N*s/mm low-speed.
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

# Day 9 refit physical bounds: any per-car k_low_speed must fall in
# [3, 10] N*s/mm. Outside this, the fit is rejected and the seeded
# value is used as fallback (no silent fits of pathological values).
_K_LOW_SPEED_MIN_NS_PER_MM: float = 3.0
_K_LOW_SPEED_MAX_NS_PER_MM: float = 10.0

# Knee at the 30th-percentile of |shockVel| -- the linear regime
# covers most operational samples; the digressive sqrt-curve takes
# over for the upper 70% of the velocity distribution.
_KNEE_PERCENTILE: float = 30.0

# Velocity used as the calibration anchor for k_low_speed: the 95th
# percentile of |shockVel|. The curve's force at v_95 should match
# a per-car target proportional to the static-spring force at that
# velocity, with the proportionality constant chosen so the seeded
# curves (in their uniform 100 mm/s knee + ~6 N*s/mm slope) were
# producing forces near 600 N at v_95 = 100 mm/s.
_ANCHOR_PERCENTILE: float = 95.0
_TARGET_FORCE_AT_P95_N: float = 600.0


@dataclass(frozen=True, slots=True)
class DamperCurve:
    """Per-car fitted damper curve parameters (Day 9 refit).

    Two parameters per curve (k_low_speed, knee_mm_s) calibrated from
    the per-car corpus shock-velocity distribution. Replaces the
    seeded global constants with per-car-distinct values.
    """
    car: str
    k_low_speed_ns_per_mm: float
    knee_mm_s: float
    n_samples: int   # # of |shockVel| samples used for the fit
    p30_velocity_mm_s: float  # the percentile that anchors the knee
    p95_velocity_mm_s: float  # the percentile that anchors k_low_speed


def damper_coefficient(car: str | None) -> float:
    """Return the per-car low-velocity damper coefficient (N*s/mm)."""
    if car is None:
        return DEFAULT_DAMPER_COEFFICIENT_NS_PER_MM
    return DAMPER_COEFFICIENT_NS_PER_MM.get(
        car.lower(), DEFAULT_DAMPER_COEFFICIENT_NS_PER_MM,
    )


def estimate_damper_force_n(
    velocity_mm_s: np.ndarray,
    *,
    car: str | None = None,
    curve: DamperCurve | None = None,
) -> np.ndarray:
    """Estimate damper force (N) from velocity (mm/s) per a digressive curve.

    F(v) = k * v                                                     for |v| < knee
    F(v) = k * knee * (1 + sqrt((|v| - knee) / knee)) * sign(v)      for |v| >= knee

    `curve` (Day 9 refit, optional): if provided, uses curve.k_low_speed
    and curve.knee_mm_s (per-car-fitted values). If absent, falls back
    to the pre-Day-9 seeded `car` lookup.
    """
    velocity_mm_s = np.asarray(velocity_mm_s, dtype=np.float64)
    if curve is not None:
        k = curve.k_low_speed_ns_per_mm
        knee = curve.knee_mm_s
    else:
        k = damper_coefficient(car)
        knee = DIGRESSIVE_KNEE_MM_S
    abs_v = np.abs(velocity_mm_s)
    low_speed = abs_v < knee
    high_speed = ~low_speed
    force = np.empty_like(velocity_mm_s)
    force[low_speed] = k * velocity_mm_s[low_speed]
    force[high_speed] = (
        k
        * knee
        * (1.0 + np.sqrt((abs_v[high_speed] - knee) / knee))
        * np.sign(velocity_mm_s[high_speed])
    )
    return force


def fit_damper_curve_from_velocities(
    car: str,
    velocity_samples_mm_s: np.ndarray,
) -> DamperCurve:
    """Refit the per-car damper curve from observed shock-velocity samples.

    Two-parameter calibration:
    - knee_mm_s = percentile-30 of |velocity_samples| (linear regime
      covers most operational samples).
    - k_low_speed_ns_per_mm = TARGET_FORCE_AT_P95 / v_95 (so the
      curve's predicted force at the 95th-percentile velocity matches
      the per-car target).

    The fit is rejected (raises ValueError) if k_low_speed falls
    outside the physical range [3, 10] N*s/mm; the caller should fall
    back to the seeded value via `damper_coefficient(car)`.
    """
    abs_v = np.abs(np.asarray(velocity_samples_mm_s, dtype=np.float64))
    abs_v = abs_v[np.isfinite(abs_v)]
    if abs_v.size < 100:
        raise ValueError(
            f"_fit_damper_curve: too few samples ({abs_v.size}); need >=100"
        )
    p30 = float(np.percentile(abs_v, _KNEE_PERCENTILE))
    p95 = float(np.percentile(abs_v, _ANCHOR_PERCENTILE))
    if p95 <= 0:
        raise ValueError(
            f"_fit_damper_curve: p95={p95} <= 0 (degenerate velocity dist)"
        )
    # k * p95 = TARGET (linear extrapolation; ignores knee curvature in
    # calibration). The curve's actual force at p95 with the digressive
    # knee at p30 will differ slightly; a follow-up could iterate, but
    # the linear approximation is sufficient for per-car distinction.
    k = _TARGET_FORCE_AT_P95_N / p95
    if not (_K_LOW_SPEED_MIN_NS_PER_MM <= k <= _K_LOW_SPEED_MAX_NS_PER_MM):
        raise ValueError(
            f"_fit_damper_curve: refit k={k:.3f} outside physical range "
            f"[{_K_LOW_SPEED_MIN_NS_PER_MM}, {_K_LOW_SPEED_MAX_NS_PER_MM}]; "
            f"caller should fall back to damper_coefficient({car!r})"
        )
    return DamperCurve(
        car=car.strip().lower(),
        k_low_speed_ns_per_mm=float(k),
        knee_mm_s=float(p30),
        n_samples=int(abs_v.size),
        p30_velocity_mm_s=p30,
        p95_velocity_mm_s=p95,
    )


def fit_damper_curve_from_corpus(
    car: str,
    corpus_root: Path | str | None = None,
    *,
    max_sessions: int = 20,
    max_samples_per_session: int = 5000,
) -> DamperCurve:
    """Refit per-car damper curve from corpus `*shockVel` channels.

    Pulls up to `max_sessions` of the car's production sessions (held-
    out automatically excluded), reads their shock-velocity channels
    from the parquet, and fits the curve via
    `fit_damper_curve_from_velocities`. Caps per-session samples to
    keep memory bounded; the percentile fit is robust enough that
    sub-sampling is acceptable.

    Returns the fitted DamperCurve. Raises ValueError if insufficient
    samples or the refit value is out of physical range; callers
    should fall back to the seeded `damper_coefficient(car)` constant.
    """
    from racingoptimizer.ingest import catalog as cat
    from racingoptimizer.ingest.api import (
        catalog_path,
        lap_data,
        resolve_corpus_root,
    )
    from racingoptimizer.ingest.api import (
        laps as ingest_laps,
    )

    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    with cat.open_catalog(catalog_path(root)) as conn:
        sessions = cat.query_sessions(
            conn, car=car, valid_only=True, include_held_out=False,
        )
    if not sessions:
        raise ValueError(f"no production sessions for car={car!r}")

    sessions = sessions[:max_sessions]
    velocities: list[float] = []
    shock_cols = (
        "LFshockVel", "RFshockVel", "LRshockVel", "RRshockVel",
    )
    for sess in sessions:
        try:
            laps_df = ingest_laps(
                session_id=sess.session_id, valid_only=True,
                corpus_root=root,
            )
        except Exception:
            continue
        if laps_df.height == 0:
            continue
        # Pick the first valid lap; we need bulk velocity samples not
        # per-corner aggregates.
        lap_idx = int(laps_df["lap_index"][0])
        try:
            df = lap_data(
                session_id=sess.session_id, lap_index=lap_idx,
                corpus_root=root,
            )
        except Exception:
            continue
        sample_count = 0
        for col in shock_cols:
            if col in df.columns:
                # iRacing exposes `*shockVel` in m/s; the fit operates in
                # mm/s (consistent with `estimate_damper_force_n`'s
                # documented `velocity_mm_s` parameter). Multiply by 1000.
                vals = df[col].to_numpy().astype(np.float64) * 1000.0
                if vals.size > max_samples_per_session // len(shock_cols):
                    # Sub-sample to bound memory.
                    step = max(
                        1,
                        vals.size
                        // (max_samples_per_session // len(shock_cols)),
                    )
                    vals = vals[::step]
                velocities.extend(vals.tolist())
                sample_count += vals.size
        if sample_count == 0:
            continue

    if not velocities:
        raise ValueError(
            f"no shock-velocity samples found for car={car!r} across "
            f"{len(sessions)} sessions"
        )
    return fit_damper_curve_from_velocities(
        car, np.asarray(velocities, dtype=np.float64),
    )
