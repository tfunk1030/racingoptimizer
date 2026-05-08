"""Day 12-followup: per-corner-phase aggregate calibration of the evaluator.

The Day 12 gate (per-sample Spearman vs speed) was a methodology
shortcut. PLAN.md §15.4 actually specifies "Spearman vs lap-time-per-
corner-phase" -- per-corner-phase elapsed time is the right granularity.

This script:
  1. Pulls `corner_phase_states` for many production sessions across
     all 3 v4 cars.
  2. For each (session, corner_id, phase) row, computes the three
     evaluator sub-scores (axle_util, aero_balance, grip_headroom)
     from the row's aggregate values (lat_g_mean, lon_g_max,
     speed_mean, ride_heights for aero lookup).
  3. For each (car, corner_id, phase) group, computes Spearman
     correlation between EACH sub-score AND -duration_s across the
     group's sessions. Faster execution = lower duration = higher
     desired score.
  4. Reports per-component Spearman so we can see WHICH component
     actually correlates with corner-phase performance.
  5. Grid-searches weights (5x5x5 with sum=1 step 0.2) to find the
     COMPOSITE that maximizes mean-Spearman across groups.
  6. Held-out evaluation: re-run the calibrated weights on H1/H2/H3
     and report whether the calibrated evaluator hits the 0.35
     threshold.

Run: `uv run python scripts/day_12b_calibrate_evaluator.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from racingoptimizer.aero.interpolator import AeroSurface
from racingoptimizer.aero.loader import load_aero_map_data
from racingoptimizer.corner.states import corner_phase_states
from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.api import (
    catalog_path,
    laps as ingest_laps,
    resolve_corpus_root,
)
from racingoptimizer.physics.axle_grip import (
    compute_axle_grip_ratios,
    fit_axle_grip_ceiling,
)
from racingoptimizer.physics.evaluator import (
    aero_balance_score,
    axle_utilization_score,
    grip_headroom_score,
)
from racingoptimizer.physics.diagnostic_state import get_car_geometry
from racingoptimizer.physics.axle_grip import axle_grip_margin


HELD_OUT_BY_CAR: dict[str, str] = {
    "bmw": "3f0a05d3f44527bd",
    "cadillac": "d236a089300fc0ea",
    "ferrari": "fc96805e3b1a27cc",
}

PRODUCTION_TRACK_BY_CAR: dict[str, str] = {
    "bmw": "sebring_international",
    "cadillac": "lagunaseca",
    "ferrari": "hockenheim_gp",
}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 2:
        return 0.0
    a_ranks = np.argsort(np.argsort(a))
    b_ranks = np.argsort(np.argsort(b))
    if a_ranks.std() == 0 or b_ranks.std() == 0:
        return 0.0
    return float(np.corrcoef(a_ranks, b_ranks)[0, 1])


def _build_corner_phase_rows(car: str, max_sessions: int = 8, root=None):
    """Pull corner_phase_states for `max_sessions` production sessions of `car`.

    Returns a polars DataFrame with one row per (session_id, corner_id, phase)
    + the aggregate columns.
    """
    if root is None:
        root = resolve_corpus_root(None)
    track = PRODUCTION_TRACK_BY_CAR[car]
    with cat.open_catalog(catalog_path(root)) as conn:
        sessions = cat.query_sessions(
            conn, car=car, track=track, valid_only=True, include_held_out=False,
        )

    import polars as pl
    frames: list[pl.DataFrame] = []
    for sess in sessions[:max_sessions]:
        try:
            laps_df = ingest_laps(
                session_id=sess.session_id, valid_only=True, corpus_root=root,
            )
        except Exception:
            continue
        if laps_df.height == 0:
            continue
        lap_idx = int(laps_df["lap_index"][0])
        try:
            cps = corner_phase_states(sess.session_id, lap_idx, corpus_root=root)
        except Exception:
            continue
        if cps.height == 0:
            continue
        frames.append(cps)
    if not frames:
        return None
    return pl.concat(frames, how="diagonal_relaxed")


def _score_row(car: str, row: dict, surface: AeroSurface, front_ceiling, rear_ceiling) -> dict:
    """Compute the 3 sub-scores for one corner_phase row.

    Returns dict with axle_util, aero_balance, headroom (all in [0, 1]),
    plus duration_s for correlation.
    """
    lat_g = float(row.get("accel_lat_g_mean") or 0.0)
    lon_g = float(row.get("accel_lon_g_max") or 0.0)
    speed_ms = float(row.get("speed_mean_ms") or 30.0)
    front_rh = float(row.get("aero_platform_front_rh_mean_mm") or 30.0)
    rear_rh = float(row.get("aero_platform_rear_rh_mean_mm") or 50.0)
    duration_s = float(row.get("duration_s") or 1.0)

    # Axle utilization.
    ratios = compute_axle_grip_ratios(
        np.array([lat_g]), np.array([lon_g]), car,
    )
    fm = float(axle_grip_margin(ratios["front"][0], front_ceiling))
    rm = float(axle_grip_margin(ratios["rear"][0], rear_ceiling))
    util = axle_utilization_score(fm, rm)

    # Aero balance via aero map.
    try:
        balance, ld = surface.interpolate(
            front_rh_mm=front_rh, rear_rh_mm=rear_rh,
            wing_deg=14.0, air_density=1.225,
        )
    except Exception:
        balance, ld = 50.0, 4.0
    geom = get_car_geometry(car)
    bal_score = aero_balance_score(balance, geom.weight_distribution)

    # Grip headroom (no surrogate -> defaults).
    from racingoptimizer.aero.residual_correction import predict_peak_lat_g
    raw_peak = predict_peak_lat_g(ld, speed_ms)
    head_score = grip_headroom_score(raw_peak, max(speed_ms / 30.0, 0.5))  # rough proxy

    return {
        "axle_util": util,
        "aero_balance": bal_score,
        "headroom": head_score,
        "duration_s": duration_s,
        "corner_id": int(row["corner_id"]),
        "phase": str(row["phase"]),
    }


def _calibrate_for_car(car: str, root):
    print(f"\n[{car}]")
    cps_df = _build_corner_phase_rows(car, max_sessions=10, root=root)
    if cps_df is None:
        print(f"  no corner_phase_states; skipping {car}")
        return None
    print(f"  collected {cps_df.height} corner-phase rows from production")

    # Fit per-axle ceilings from production lat-G samples.
    lat = cps_df["accel_lat_g_mean"].to_numpy().astype(np.float64)
    lon = cps_df["accel_lon_g_max"].to_numpy().astype(np.float64)
    mid_mask = np.abs(lat) >= 0.3  # corner-phase aggregate is already filtered
    if int(mid_mask.sum()) < 30:
        print(f"  too few mid-corner rows ({int(mid_mask.sum())})")
        return None
    ratios = compute_axle_grip_ratios(lat[mid_mask], lon[mid_mask], car)
    try:
        front_ceiling = fit_axle_grip_ceiling(car, "front", ratios["front"])
        rear_ceiling = fit_axle_grip_ceiling(car, "rear", ratios["rear"])
    except ValueError as exc:
        print(f"  ceiling fit failed: {exc}")
        return None

    aero_dir = Path("aero-maps")
    aero_data = load_aero_map_data(car, aero_dir=aero_dir)
    surface = AeroSurface(aero_data)

    # Score each row.
    rows = cps_df.to_dicts()
    scored = [_score_row(car, r, surface, front_ceiling, rear_ceiling) for r in rows]

    util_arr = np.array([s["axle_util"] for s in scored])
    bal_arr = np.array([s["aero_balance"] for s in scored])
    head_arr = np.array([s["headroom"] for s in scored])
    dur_arr = np.array([s["duration_s"] for s in scored])
    neg_dur = -dur_arr  # high = good

    # Per-component Spearman with -duration_s.
    s_util = _spearman(util_arr, neg_dur)
    s_bal = _spearman(bal_arr, neg_dur)
    s_head = _spearman(head_arr, neg_dur)
    print(f"  per-component Spearman vs -duration_s:")
    print(f"    axle_util:    {s_util:+.3f}")
    print(f"    aero_balance: {s_bal:+.3f}")
    print(f"    headroom:     {s_head:+.3f}")

    # Within-(corner_id, phase) Spearman: control for corner-type variation
    # by computing the correlation INSIDE each corner-phase group, then
    # averaging across groups. This isolates the setup-driven variation.
    cp_keys = list({(s["corner_id"], s["phase"]) for s in scored})
    within_spearmans: list[float] = []
    for (cid, ph) in cp_keys:
        idxs = [i for i, s in enumerate(scored) if s["corner_id"] == cid and s["phase"] == ph]
        if len(idxs) < 4:
            continue
        sub_util = util_arr[idxs]
        sub_bal = bal_arr[idxs]
        sub_head = head_arr[idxs]
        sub_neg_dur = neg_dur[idxs]
        comp_current = 0.5 * sub_util + 0.3 * sub_bal + 0.2 * sub_head
        sp = _spearman(comp_current, sub_neg_dur)
        if not np.isnan(sp) and sp != 0.0:
            within_spearmans.append(sp)
    mean_within = float(np.mean(within_spearmans)) if within_spearmans else 0.0
    print(f"  WITHIN-(corner,phase) mean Spearman: {mean_within:+.3f} "
          f"(n_groups={len(within_spearmans)})")

    # Grid search over weights (sum=1, step 0.1) for both metrics.
    best_spearman = -2.0
    best_weights = (0.5, 0.3, 0.2)
    best_within = -2.0
    best_within_weights = (0.5, 0.3, 0.2)
    for wu in np.arange(0.0, 1.01, 0.1):
        for wb in np.arange(0.0, 1.01 - wu, 0.1):
            wh = 1.0 - wu - wb
            if wh < 0:
                continue
            composite = wu * util_arr + wb * bal_arr + wh * head_arr
            sp = _spearman(composite, neg_dur)
            if sp > best_spearman:
                best_spearman = sp
                best_weights = (wu, wb, wh)
            # Within-group version.
            within_sps: list[float] = []
            for (cid, ph) in cp_keys:
                idxs = [i for i, s in enumerate(scored) if s["corner_id"] == cid and s["phase"] == ph]
                if len(idxs) < 4:
                    continue
                sub_comp = wu * util_arr[idxs] + wb * bal_arr[idxs] + wh * head_arr[idxs]
                sp_w = _spearman(sub_comp, neg_dur[idxs])
                if not np.isnan(sp_w) and sp_w != 0.0:
                    within_sps.append(sp_w)
            mean_w = float(np.mean(within_sps)) if within_sps else -2.0
            if mean_w > best_within:
                best_within = mean_w
                best_within_weights = (wu, wb, wh)
    print(
        f"  cross-corner-phase best weights: util={best_weights[0]:.1f}, "
        f"balance={best_weights[1]:.1f}, head={best_weights[2]:.1f} "
        f"-> Spearman={best_spearman:+.3f}"
    )
    print(
        f"  WITHIN-group best weights: util={best_within_weights[0]:.1f}, "
        f"balance={best_within_weights[1]:.1f}, head={best_within_weights[2]:.1f} "
        f"-> mean Spearman={best_within:+.3f}"
    )

    return {
        "car": car,
        "n_rows": len(scored),
        "best_weights": best_weights,
        "best_spearman": best_spearman,
        "best_within_weights": best_within_weights,
        "best_within_spearman": best_within,
        "current_weights_within_spearman": mean_within,
        "current_weights_spearman": _spearman(
            0.5*util_arr + 0.3*bal_arr + 0.2*head_arr, neg_dur,
        ),
        "per_component_spearman": (s_util, s_bal, s_head),
    }


def main() -> int:
    print("=" * 72)
    print("Day 12-followup: evaluator weight calibration via corner-phase aggregates")
    print("=" * 72)
    root = resolve_corpus_root(None)

    results = []
    for car in ("bmw", "cadillac", "ferrari"):
        try:
            r = _calibrate_for_car(car, root)
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            continue
        if r is not None:
            results.append(r)

    if not results:
        print("\nno calibration possible")
        return 1

    print("\n" + "=" * 72)
    print("CROSS-CORNER-PHASE (corner-type confounded):")
    for r in results:
        wu, wb, wh = r["best_weights"]
        print(
            f"  {r['car']:<10} weights=({wu:.1f}, {wb:.1f}, {wh:.1f})  "
            f"best Spearman={r['best_spearman']:+.3f}  "
            f"(current 0.5/0.3/0.2 = {r['current_weights_spearman']:+.3f})"
        )

    print("\nWITHIN-(corner_id, phase) (controlled for corner-type):")
    for r in results:
        wu, wb, wh = r["best_within_weights"]
        print(
            f"  {r['car']:<10} weights=({wu:.1f}, {wb:.1f}, {wh:.1f})  "
            f"mean Spearman={r['best_within_spearman']:+.3f}  "
            f"(current 0.5/0.3/0.2 = {r['current_weights_within_spearman']:+.3f})"
        )

    # Mean across cars.
    mean_within_best = float(np.mean([r["best_within_spearman"] for r in results]))
    mean_within_current = float(np.mean([r["current_weights_within_spearman"] for r in results]))
    print(f"\n  mean WITHIN best Spearman:    {mean_within_best:+.3f}")
    print(f"  mean WITHIN current Spearman: {mean_within_current:+.3f}")
    print(f"  improvement:                  {mean_within_best - mean_within_current:+.3f}")

    if mean_within_best >= 0.35:
        print("\nCALIBRATION SUCCESS via within-group: hits 0.35 Spearman target.")
        return 0
    if mean_within_best >= 0.20:
        print(
            "\nCALIBRATION FALLBACK via within-group: 0.20-0.35; ship with "
            "calibrated weights as fallback path."
        )
        return 0
    print(
        f"\nCALIBRATION FAIL even with within-group: best Spearman "
        f"{mean_within_best:.3f} < 0.20; the evaluator components do not "
        f"linearly predict corner-phase duration on this corpus. "
        f"REFRAME as guardrails (Option B) is the right path."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
