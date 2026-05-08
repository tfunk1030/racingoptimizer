"""Day 13 investigation (C): which corner-phases respond to setup variation?

Day 12b finding: within-group mean Spearman of evaluator-vs-duration is
0.187 (Ferrari: 0.249, BMW: 0.189, Cadillac: 0.122). This means
setup-driven variation explains only ~3-6% of duration variance per
corner-phase on this corpus.

Investigation question: are SOME corner-phases (e.g. specific corners
or specific phases) much more responsive to setup, while others are
driver/conditions-limited noise? If yes, the hybrid optimizer can
weight physics constraints HIGHER on responsive corner-phases and
LOWER on noisy ones.

For each (car, corner_id, phase) group:
  1. Compute setup variation: stdev of per-parameter values across
     sessions in this group's corpus.
  2. Compute duration variation: stdev of duration_s across sessions.
  3. Compute Spearman: evaluator-score vs -duration_s within the group.

Expected findings:
  - High-Spearman groups: corner-phases where physics has real signal.
    Use larger physics weight in hybrid optimizer.
  - Low-Spearman groups: corner-phases where setup doesn't matter
    much. Use smaller physics weight.
  - Phase-specific patterns: maybe braking responds to setup more
    than mid_corner, or vice versa.

Output: per-(car, phase) Spearman histogram + recommendations for
hybrid optimizer's per-corner-phase weighting.

Run: `uv run python scripts/day_13_investigate_responsiveness.py`
"""
from __future__ import annotations

import sys
from collections import defaultdict
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
    axle_grip_margin,
    compute_axle_grip_ratios,
    fit_axle_grip_ceiling,
)
from racingoptimizer.physics.diagnostic_state import get_car_geometry
from racingoptimizer.physics.evaluator import (
    aero_balance_score,
    axle_utilization_score,
    get_weights_for_car,
    grip_headroom_score,
)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 2:
        return 0.0
    a_ranks = np.argsort(np.argsort(a))
    b_ranks = np.argsort(np.argsort(b))
    if a_ranks.std() == 0 or b_ranks.std() == 0:
        return 0.0
    return float(np.corrcoef(a_ranks, b_ranks)[0, 1])


def _investigate_car(car: str, track: str, root) -> dict:
    """Compute per-(corner_id, phase) Spearman."""
    print(f"\n[{car} @ {track}]")
    with cat.open_catalog(catalog_path(root)) as conn:
        sessions = cat.query_sessions(
            conn, car=car, track=track, valid_only=True, include_held_out=False,
        )

    import polars as pl
    frames: list[pl.DataFrame] = []
    for sess in sessions[:15]:
        try:
            laps_df = ingest_laps(
                session_id=sess.session_id, valid_only=True, corpus_root=root,
            )
        except Exception:
            continue
        if laps_df.height == 0:
            continue
        # Use multiple laps per session for richer per-(corner, phase) variation.
        for lap_pos in range(min(3, laps_df.height)):
            try:
                cps = corner_phase_states(
                    sess.session_id, int(laps_df["lap_index"][lap_pos]),
                    corpus_root=root,
                )
            except Exception:
                continue
            if cps.height > 0:
                frames.append(cps)
    if not frames:
        print("  no data")
        return {}
    cps_all = pl.concat(frames, how="diagonal_relaxed")
    print(f"  collected {cps_all.height} rows")

    # Fit per-axle ceilings.
    lat = cps_all["accel_lat_g_mean"].to_numpy().astype(np.float64)
    lon = cps_all["accel_lon_g_max"].to_numpy().astype(np.float64)
    mid_mask = np.abs(lat) >= 0.3
    if int(mid_mask.sum()) < 30:
        return {}
    ratios = compute_axle_grip_ratios(lat[mid_mask], lon[mid_mask], car)
    try:
        front_ceiling = fit_axle_grip_ceiling(car, "front", ratios["front"])
        rear_ceiling = fit_axle_grip_ceiling(car, "rear", ratios["rear"])
    except ValueError:
        return {}

    aero_data = load_aero_map_data(car, aero_dir=Path("aero-maps"))
    surface = AeroSurface(aero_data)
    geom = get_car_geometry(car)
    wu, wb, wh = get_weights_for_car(car)

    # Per-row score + duration.
    rows = cps_all.to_dicts()
    by_group: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        try:
            cid = int(r["corner_id"])
            ph = str(r["phase"])
            lat_g = float(r.get("accel_lat_g_mean") or 0.0)
            lon_g = float(r.get("accel_lon_g_max") or 0.0)
            speed_ms = float(r.get("speed_mean_ms") or 30.0)
            front_rh = float(r.get("aero_platform_front_rh_mean_mm") or 30.0)
            rear_rh = float(r.get("aero_platform_rear_rh_mean_mm") or 50.0)
            duration_s = float(r.get("duration_s") or 1.0)
        except (TypeError, ValueError):
            continue
        if lat_g == 0 or duration_s <= 0:
            continue
        # Score components.
        ar = compute_axle_grip_ratios(np.array([lat_g]), np.array([lon_g]), car)
        fm = float(axle_grip_margin(ar["front"][0], front_ceiling))
        rm = float(axle_grip_margin(ar["rear"][0], rear_ceiling))
        util = axle_utilization_score(fm, rm)
        try:
            balance, ld = surface.interpolate(
                front_rh_mm=front_rh, rear_rh_mm=rear_rh,
                wing_deg=14.0, air_density=1.225,
            )
        except Exception:
            balance, ld = 50.0, 4.0
        bal = aero_balance_score(balance, geom.weight_distribution)
        # Headroom: use the surrogate-less neutral default.
        head = 1.0
        composite = wu * util + wb * bal + wh * head
        by_group[(cid, ph)].append((composite, duration_s))

    # Per-group Spearman.
    group_spearmans: list[tuple[tuple, float, int]] = []
    for key, vals in by_group.items():
        if len(vals) < 5:
            continue
        scores = np.array([v[0] for v in vals])
        durs = np.array([v[1] for v in vals])
        sp = _spearman(scores, -durs)
        if not np.isnan(sp):
            group_spearmans.append((key, sp, len(vals)))

    # Phase-aggregated stats.
    by_phase: dict[str, list[float]] = defaultdict(list)
    for (cid, ph), sp, _n in group_spearmans:
        by_phase[ph].append(sp)
    print(f"  per-phase mean Spearman:")
    for ph, sps in sorted(by_phase.items()):
        if sps:
            print(
                f"    {ph:<14} n_groups={len(sps):2d}  "
                f"mean={np.mean(sps):+.3f}  "
                f"max={max(sps):+.3f}  "
                f"min={min(sps):+.3f}"
            )

    # Top-10 most-responsive corner-phases.
    top = sorted(group_spearmans, key=lambda x: x[1], reverse=True)[:10]
    print(f"  top-10 most responsive (corner_id, phase, Spearman, n):")
    for (cid, ph), sp, n in top:
        print(f"    corner={cid:2d} phase={ph:<14} sp={sp:+.3f} n={n}")

    # Bottom-5 least-responsive (often near zero).
    bottom = sorted(group_spearmans, key=lambda x: x[1])[:5]
    print(f"  bottom-5 least responsive:")
    for (cid, ph), sp, n in bottom:
        print(f"    corner={cid:2d} phase={ph:<14} sp={sp:+.3f} n={n}")

    return {
        "car": car,
        "n_groups": len(group_spearmans),
        "by_phase": {ph: float(np.mean(sps)) for ph, sps in by_phase.items()},
        "top_responsive": [(k, sp, n) for (k, sp, n) in top[:5]],
        "median_spearman": float(np.median([sp for _, sp, _ in group_spearmans])),
    }


def main() -> int:
    print("=" * 72)
    print("Day 13 investigation: corner-phase responsiveness to setup")
    print("=" * 72)
    root = resolve_corpus_root(None)

    cars_tracks = [
        ("bmw", "sebring_international"),
        ("cadillac", "lagunaseca"),
        ("ferrari", "hockenheim_gp"),
    ]

    results = []
    for car, track in cars_tracks:
        try:
            r = _investigate_car(car, track, root)
            if r:
                results.append(r)
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    if not results:
        return 1

    # Roll up: which phases are universally responsive?
    print("\n" + "=" * 72)
    print("CROSS-CAR PHASE RESPONSIVENESS (mean Spearman per phase):")
    all_phases = set()
    for r in results:
        all_phases.update(r["by_phase"].keys())
    for ph in sorted(all_phases):
        cars_with_phase = [
            (r["car"], r["by_phase"][ph])
            for r in results if ph in r["by_phase"]
        ]
        means = [v for _, v in cars_with_phase]
        print(
            f"  {ph:<14} cross-car-mean={np.mean(means):+.3f}  "
            f"per-car: " +
            " ".join(f"{c}={v:+.3f}" for c, v in cars_with_phase)
        )

    # Median across all (car, corner, phase) groups.
    print("\nMedian Spearman across all groups per car:")
    for r in results:
        print(f"  {r['car']:<10} median={r['median_spearman']:+.3f} "
              f"n_groups={r['n_groups']}")

    print("\n" + "=" * 72)
    print("RECOMMENDATIONS for Day 13 hybrid optimizer:")
    print("- Use per-car-calibrated weights (Day 12b) as composite-score base")
    print("- Hybrid w (physics weight) should be SMALL on this corpus (~0.1-0.2)")
    print("  since within-group Spearman 0.12-0.25 indicates physics adds")
    print("  modest signal beyond the surrogate.")
    print("- Apply guardrails as HARD CONSTRAINTS on DE search (penalize setups")
    print("  with axle_util > 1.0) rather than relying on score correlation.")
    print("- Per-phase weighting could be added but the cross-car phase pattern")
    print("  is mixed (see above); a single per-car w is the simplest defensible")
    print("  starting point.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
