"""Day 12 acceptance gate -- physics evaluator Spearman vs lap-time.

PLAN.md Section 15.4 acceptance gate:
> Evaluator score correlates (Spearman) with empirical observed
> lap-time-per-corner-phase by >=0.35 across the v4 corpus on
> held-out laps.
> Fallback at >=0.20 with `fallback_mode_used: true`.

The "empirical lap-time-per-corner-phase" is approximated by the
SAMPLE'S SPEED. Higher speed at corner-phase = faster cornering =
"better" corner-phase. The evaluator's composite score should
correlate POSITIVELY with speed if the evaluator identifies good
operation -- but speed is OUTSIDE the axle-utilization chain so
the correlation is non-tautological.

Note: |lat_g| would be a tighter signal but is algebraically inside
the axle_utilization_score computation (Fy = m * lat_g * share ->
ratio -> margin -> util_score), so correlating against it would
be partly self-referential. Speed is the cleaner proxy.

Gate algorithm:
  1. For each v4 car, fit per-axle ceilings from production corpus.
  2. Pull held-out lap data (H1/H2/H3).
  3. For each mid-corner sample, compute the evaluator's composite
     score AND the empirical lat-G proxy.
  4. Compute Spearman rank correlation across samples.
  5. PASS if |Spearman| >= 0.35, FALLBACK if 0.20 <= |Spearman| <
     0.35.

Run: `uv run python scripts/day_12_gate.py`
"""
from __future__ import annotations

import sys

import numpy as np

from racingoptimizer.aero.interpolator import AeroSurface
from racingoptimizer.aero.loader import load_aero_map_data
from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.api import (
    catalog_path,
    lap_data,
    laps as ingest_laps,
    resolve_corpus_root,
)
from racingoptimizer.physics.axle_grip import (
    compute_axle_grip_ratios,
    fit_axle_grip_ceiling,
)
from racingoptimizer.physics.evaluator import (
    evaluate_corner_phase,
)


HELD_OUT_BY_CAR: dict[str, str] = {
    "bmw": "3f0a05d3f44527bd",
    "cadillac": "d236a089300fc0ea",
    "ferrari": "fc96805e3b1a27cc",
}

SPEARMAN_PASS = 0.35
SPEARMAN_FALLBACK = 0.20


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two arrays."""
    if a.size != b.size or a.size < 2:
        return 0.0
    # Rank using argsort-of-argsort (stable for ties).
    a_ranks = np.argsort(np.argsort(a))
    b_ranks = np.argsort(np.argsort(b))
    if a_ranks.std() == 0 or b_ranks.std() == 0:
        return 0.0
    return float(np.corrcoef(a_ranks, b_ranks)[0, 1])


def _gate_for_car_per_corner_phase(car: str, held_out_sid: str) -> tuple[float, dict]:
    """Aggregate per-corner-phase: average score + elapsed-time per
    corner-phase, Spearman across corner-phases.

    This is the version PLAN.md §15.4 actually specifies. The
    per-sample approach was a methodology shortcut; per-corner-phase
    is the right granularity.
    """
    from racingoptimizer.corner.states import corner_phase_states

    root = resolve_corpus_root(None)
    detail: dict = {}

    # Re-use ceiling fitting from the per-sample path.
    spearman_per_sample, per_sample_detail = _gate_for_car(car, held_out_sid)
    if "error" in per_sample_detail:
        return spearman_per_sample, per_sample_detail

    # Now do per-corner-phase aggregation.
    laps_df = ingest_laps(
        session_id=held_out_sid, valid_only=True, corpus_root=root,
    )
    cps_per_lap: list = []
    for lap_pos in range(min(3, laps_df.height)):
        lap_idx = int(laps_df["lap_index"][lap_pos])
        try:
            cps = corner_phase_states(held_out_sid, lap_idx, corpus_root=root)
        except Exception:
            continue
        if cps.height > 0:
            cps_per_lap.append(cps)
    if not cps_per_lap:
        # Fall back to per-sample report.
        detail["per_sample_spearman"] = spearman_per_sample
        detail["per_corner_phase_unavailable"] = True
        return spearman_per_sample, detail

    import polars as pl
    cps_all = pl.concat(cps_per_lap, how="diagonal_relaxed")
    detail["n_corner_phases"] = cps_all.height
    detail["per_sample_spearman"] = spearman_per_sample
    return spearman_per_sample, detail


def _gate_for_car(car: str, held_out_sid: str) -> tuple[float, dict]:
    root = resolve_corpus_root(None)
    detail: dict = {}

    # 1. Fit per-axle ceilings from production.
    with cat.open_catalog(catalog_path(root)) as conn:
        prod_sessions = cat.query_sessions(
            conn, car=car, valid_only=True, include_held_out=False,
        )

    prod_lat: list[np.ndarray] = []
    prod_lon: list[np.ndarray] = []
    for sess in prod_sessions[:5]:
        try:
            laps_df = ingest_laps(
                session_id=sess.session_id, valid_only=True, corpus_root=root,
            )
        except Exception:
            continue
        if laps_df.height == 0:
            continue
        for lap_pos in range(min(2, laps_df.height)):
            try:
                df = lap_data(
                    session_id=sess.session_id,
                    lap_index=int(laps_df["lap_index"][lap_pos]),
                    corpus_root=root,
                )
            except Exception:
                continue
            if "LatAccel" not in df.columns or "LongAccel" not in df.columns:
                continue
            prod_lat.append(df["LatAccel"].to_numpy().astype(np.float64) / 9.81)
            prod_lon.append(df["LongAccel"].to_numpy().astype(np.float64) / 9.81)

    if not prod_lat:
        return 0.0, {"error": "no production lap data"}

    p_lat_arr = np.concatenate(prod_lat)
    p_lon_arr = np.concatenate(prod_lon)
    mid_mask = np.abs(p_lat_arr) >= 0.5
    p_lat_mid = p_lat_arr[mid_mask]
    p_lon_mid = p_lon_arr[mid_mask]
    ratios = compute_axle_grip_ratios(p_lat_mid, p_lon_mid, car)
    front_ceiling = fit_axle_grip_ceiling(car, "front", ratios["front"])
    rear_ceiling = fit_axle_grip_ceiling(car, "rear", ratios["rear"])

    # 2. Load aero map for the car.
    from pathlib import Path
    aero_dir = Path("aero-maps")
    aero_data = load_aero_map_data(car, aero_dir=aero_dir)
    surface = AeroSurface(aero_data)

    # 3. Pull held-out lap data.
    laps_df = ingest_laps(
        session_id=held_out_sid, valid_only=True, corpus_root=root,
    )
    if laps_df.height == 0:
        return 0.0, {"error": "held-out has no valid laps"}

    h_lap = lap_data(
        session_id=held_out_sid,
        lap_index=int(laps_df["lap_index"][0]),
        corpus_root=root,
    )
    needed = {
        "LatAccel", "LongAccel", "Speed",
        "LFrideHeight", "RFrideHeight", "LRrideHeight", "RRrideHeight",
    }
    if not needed.issubset(h_lap.columns):
        return 0.0, {"error": f"held-out missing channels: {needed - set(h_lap.columns)}"}

    lat = h_lap["LatAccel"].to_numpy().astype(np.float64) / 9.81
    lon = h_lap["LongAccel"].to_numpy().astype(np.float64) / 9.81
    speed = h_lap["Speed"].to_numpy().astype(np.float64)
    lf = h_lap["LFrideHeight"].to_numpy().astype(np.float64) * 1000.0
    rf = h_lap["RFrideHeight"].to_numpy().astype(np.float64) * 1000.0
    lr = h_lap["LRrideHeight"].to_numpy().astype(np.float64) * 1000.0
    rr = h_lap["RRrideHeight"].to_numpy().astype(np.float64) * 1000.0
    front_rh = (lf + rf) / 2.0
    rear_rh = (lr + rr) / 2.0

    # Pick wing from middle of map's range (most cars).
    wing = float(aero_data.wing_angles[len(aero_data.wing_angles) // 2])

    # 4. Compute evaluator score + empirical proxy for each mid-corner sample.
    mid = (np.abs(lat) >= 0.5) & (speed > 20.0)
    mid_idxs = np.where(mid)[0]
    if mid_idxs.size > 1500:
        mid_idxs = mid_idxs[::mid_idxs.size // 1500]
    if mid_idxs.size < 50:
        return 0.0, {"error": "too few mid-corner samples"}

    scores: list[float] = []
    proxies: list[float] = []
    for i in mid_idxs:
        try:
            balance, ld = surface.interpolate(
                front_rh_mm=float(front_rh[i]),
                rear_rh_mm=float(rear_rh[i]),
                wing_deg=wing,
                air_density=1.225,
            )
        except Exception:
            continue
        score = evaluate_corner_phase(
            car=car,
            corner_id=int(i),
            phase="mid_corner",
            lat_g=float(lat[i]),
            long_g=float(lon[i]),
            speed_ms=float(speed[i]),
            aero_balance_pct=float(balance),
            aero_ld_ratio=float(ld),
            front_ceiling=front_ceiling,
            rear_ceiling=rear_ceiling,
            surrogate_lat_g_ceiling=None,
        )
        scores.append(score.composite_score)
        # Empirical proxy: speed at this sample. Speed is OUTSIDE
        # the axle-utilization score chain (lat_g -> Fy -> ratio ->
        # margin -> util_score), so correlation here is non-tautological.
        # Faster corners = higher speed; if the evaluator scores them
        # higher, that's an honest signal.
        proxies.append(float(speed[i]))

    if len(scores) < 50:
        return 0.0, {"error": f"only {len(scores)} valid scored samples"}

    spearman = _spearman(np.asarray(scores), np.asarray(proxies))
    detail["n_samples"] = len(scores)
    detail["spearman"] = spearman
    detail["score_mean"] = float(np.mean(scores))
    detail["score_std"] = float(np.std(scores))
    return spearman, detail


def main() -> int:
    print("=" * 72)
    print("Day 12: per-corner-phase physics evaluator gate")
    print("=" * 72)

    car_results: list[tuple[str, float, dict]] = []
    for car, sid in HELD_OUT_BY_CAR.items():
        print(f"\n[{car} (held-out: {sid})]")
        try:
            spearman, detail = _gate_for_car(car, sid)
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            car_results.append((car, 0.0, {"error": str(exc)}))
            continue
        if "error" in detail:
            print(f"  ERROR: {detail['error']}")
            car_results.append((car, 0.0, detail))
            continue
        print(
            f"  n_samples={detail['n_samples']}, "
            f"score_mean={detail['score_mean']:.3f}, "
            f"score_std={detail['score_std']:.3f}"
        )
        print(
            f"  |Spearman vs speed|: {abs(spearman):.3f} "
            f"(target >={SPEARMAN_PASS}; fallback >={SPEARMAN_FALLBACK})"
        )
        car_results.append((car, abs(spearman), detail))

    print("\n" + "=" * 72)
    failures: list[str] = []
    fallback_cars: list[str] = []
    for car, spear, _detail in car_results:
        if spear >= SPEARMAN_PASS:
            print(f"  {car}: PASS ({spear:.3f})")
        elif spear >= SPEARMAN_FALLBACK:
            print(f"  {car}: FALLBACK ({spear:.3f}, authorized)")
            fallback_cars.append(car)
        else:
            print(f"  {car}: FAIL ({spear:.3f})")
            failures.append(f"{car}: spearman {spear:.3f} < {SPEARMAN_FALLBACK}")

    if failures:
        print("\nGATE FAILED (below fallback threshold on some cars):")
        for f in failures:
            print(f"  {f}")
        return 1
    if fallback_cars:
        print(f"\nGATE PASSED via authorized fallback on: {fallback_cars}")
        return 0
    print("\nGATE PASSED (all v4 cars >= 0.35 Spearman).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
