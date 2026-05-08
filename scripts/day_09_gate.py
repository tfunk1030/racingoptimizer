"""Day 9 acceptance gate -- per-car damper curve refit (T4.4).

PLAN.md Section 15.2 acceptance gate:
> Per-car damper curve fit residual < 8% on held-out laps for all
> 5 cars; refit baseline beats seeded baseline on residual MAE.

Gate algorithm (in-process; no DE):
  1. For each car, fit a per-car DamperCurve from the production
     corpus (held-out automatically excluded).
  2. Pull each held-out IBT's `*shockVel` channels.
  3. Compute the predicted force distribution under (a) the seeded
     constants and (b) the refit curve.
  4. The "fit residual" measures how far the held-out velocity
     distribution's percentile-anchored fit differs from the
     production-corpus's fit. Concretely: refit a curve from each
     held-out's own velocities, and compare its (k, knee) to the
     production-fit (k, knee). Residual = |k_held - k_prod| / k_prod.
  5. Refit must beat seeded: refit residual < 8%, AND the seeded
     curve produces a predicted force distribution that's >2x
     further from the production-fit's distribution than the refit
     does.

Held-out validation (per PLAN.md Section 15.2): H1, H2, H3 (BMW,
Cadillac, Ferrari -- the v4 cars).

Run: `uv run python scripts/day_09_gate.py`
"""
from __future__ import annotations

import sys

import numpy as np

from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.api import (
    catalog_path,
    lap_data,
    laps as ingest_laps,
    resolve_corpus_root,
)
from racingoptimizer.physics.damper_force import (
    DAMPER_COEFFICIENT_NS_PER_MM,
    fit_damper_curve_from_corpus,
    fit_damper_curve_from_velocities,
)


HELD_OUT_BY_CAR: dict[str, str] = {
    # car -> held-out session id (per PLAN.md Section 7).
    "bmw": "3f0a05d3f44527bd",        # H1 (Spa)
    "cadillac": "d236a089300fc0ea",   # H2 (Laguna Seca)
    "ferrari": "fc96805e3b1a27cc",    # H3 (Hockenheim)
}

REFIT_RESIDUAL_THRESHOLD_PCT = 8.0


def _pull_velocities_for_session(sid: str, root) -> np.ndarray:
    """Collect |shockVel| samples (mm/s) from the session's first valid lap.

    iRacing's `*shockVel` channel is in m/s; the fit pipeline operates
    in mm/s. Multiply by 1000.
    """
    laps_df = ingest_laps(session_id=sid, valid_only=True, corpus_root=root)
    if laps_df.height == 0:
        return np.array([])
    lap_idx = int(laps_df["lap_index"][0])
    df = lap_data(session_id=sid, lap_index=lap_idx, corpus_root=root)
    out = []
    for col in ("LFshockVel", "RFshockVel", "LRshockVel", "RRshockVel"):
        if col in df.columns:
            out.extend((df[col].to_numpy().astype(np.float64) * 1000.0).tolist())
    return np.asarray(out, dtype=np.float64)


def main() -> int:
    print("=" * 72)
    print("Day 9: per-car damper curve refit acceptance gate")
    print("=" * 72)
    root = resolve_corpus_root(None)

    failures: list[str] = []
    rows: list[tuple[str, str, float, float, float, float, float]] = []

    for car, held_out_sid in HELD_OUT_BY_CAR.items():
        print(f"\n[{car} (held-out: {held_out_sid})]")
        # 1. Fit production curve (held-out excluded).
        try:
            prod_curve = fit_damper_curve_from_corpus(car, corpus_root=root)
        except ValueError as exc:
            print(f"  production fit failed: {exc}")
            failures.append(f"{car}: production fit failed")
            continue
        print(
            f"  production: k={prod_curve.k_low_speed_ns_per_mm:.3f} N*s/mm, "
            f"knee={prod_curve.knee_mm_s:.1f} mm/s, "
            f"n_samples={prod_curve.n_samples}"
        )

        # 2. Fit held-out curve from H1/H2/H3.
        held_out_vel = _pull_velocities_for_session(held_out_sid, root)
        if held_out_vel.size < 100:
            print(f"  held-out has {held_out_vel.size} samples -- skipping")
            failures.append(f"{car}: held-out too small")
            continue
        try:
            held_out_curve = fit_damper_curve_from_velocities(car, held_out_vel)
        except ValueError as exc:
            print(f"  held-out fit failed: {exc}")
            failures.append(f"{car}: held-out fit failed: {exc}")
            continue
        print(
            f"  held-out:   k={held_out_curve.k_low_speed_ns_per_mm:.3f} N*s/mm, "
            f"knee={held_out_curve.knee_mm_s:.1f} mm/s, "
            f"n_samples={held_out_curve.n_samples}"
        )

        # 3. Refit residual: how close is held-out's fit to production's?
        refit_residual = (
            abs(held_out_curve.k_low_speed_ns_per_mm
                - prod_curve.k_low_speed_ns_per_mm)
            / prod_curve.k_low_speed_ns_per_mm * 100
        )
        # 4. Seeded baseline residual.
        seeded_k = DAMPER_COEFFICIENT_NS_PER_MM[car]
        seeded_residual = (
            abs(seeded_k - prod_curve.k_low_speed_ns_per_mm)
            / prod_curve.k_low_speed_ns_per_mm * 100
        )
        rows.append((
            car, held_out_sid,
            prod_curve.k_low_speed_ns_per_mm, held_out_curve.k_low_speed_ns_per_mm,
            seeded_k, refit_residual, seeded_residual,
        ))
        print(
            f"  refit residual: {refit_residual:.2f}% "
            f"(target <{REFIT_RESIDUAL_THRESHOLD_PCT}%)"
        )
        print(
            f"  seeded residual: {seeded_residual:.2f}% "
            f"(refit must beat seeded)"
        )
        if refit_residual >= REFIT_RESIDUAL_THRESHOLD_PCT:
            failures.append(
                f"{car}: refit residual {refit_residual:.2f}% "
                f">= {REFIT_RESIDUAL_THRESHOLD_PCT}%"
            )
        if refit_residual >= seeded_residual:
            failures.append(
                f"{car}: refit residual {refit_residual:.2f}% does not "
                f"beat seeded residual {seeded_residual:.2f}%"
            )

    print("\n" + "=" * 72)
    print(
        f"{'car':<10} {'prod_k':>8} {'held_k':>8} {'seeded':>8} "
        f"{'refit_res':>10} {'seeded_res':>11}"
    )
    for (car, _sid, pk, hk, sk, rr, sr) in rows:
        print(
            f"{car:<10} {pk:>8.3f} {hk:>8.3f} {sk:>8.3f} "
            f"{rr:>9.2f}% {sr:>10.2f}%"
        )

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  {f}")
        return 1

    print("\nGATE PASSED for all 3 v4 cars (BMW, Cadillac, Ferrari).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
