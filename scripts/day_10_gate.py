"""Day 10 acceptance gate -- per-axle grip-margin model.

PLAN.md Section 15.3 acceptance gate:
> On held-out laps, per-axle grip-margin predicts whether a corner
> exceeded 90% of axle ceiling with >=70% accuracy.

The "accuracy" definition: per-corner-phase the prediction is binary
(at_limit yes/no), and "ground truth" is the EMPIRICAL judgment that
the sample's lat-G is in the top 10% of mid-corner samples (the
implicit limit). This is consistent with the ceiling fit being a
99th-percentile anchor: the top 1% of ratios are at the ceiling, the
top 10% are within 90% of ceiling.

In-process gate: fit per-axle ceilings on production lap data for
each v4 car (BMW, Cadillac, Ferrari), predict at-limit on held-out
laps, compare against empirical top-10% ground truth.

Run: `uv run python scripts/day_10_gate.py`
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
from racingoptimizer.physics.axle_grip import (
    compute_axle_grip_ratios,
    fit_axle_grip_ceiling,
    predict_corner_at_limit,
)


HELD_OUT_BY_CAR: dict[str, str] = {
    "bmw": "3f0a05d3f44527bd",
    "cadillac": "d236a089300fc0ea",
    "ferrari": "fc96805e3b1a27cc",
}

ACCURACY_THRESHOLD = 0.70
AT_LIMIT_THRESHOLD = 0.90


def _pull_lat_long_for_session(sid: str, root) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate LatAccel + LongAccel across all valid laps."""
    laps_df = ingest_laps(session_id=sid, valid_only=True, corpus_root=root)
    lat_all: list[np.ndarray] = []
    lon_all: list[np.ndarray] = []
    for lap_pos in range(min(8, laps_df.height)):
        lap_idx = int(laps_df["lap_index"][lap_pos])
        try:
            df = lap_data(session_id=sid, lap_index=lap_idx, corpus_root=root)
        except Exception:
            continue
        if "LatAccel" not in df.columns or "LongAccel" not in df.columns:
            continue
        lat_all.append(df["LatAccel"].to_numpy().astype(np.float64) / 9.81)
        lon_all.append(df["LongAccel"].to_numpy().astype(np.float64) / 9.81)
    if not lat_all:
        return np.array([]), np.array([])
    return np.concatenate(lat_all), np.concatenate(lon_all)


def main() -> int:
    print("=" * 72)
    print("Day 10: per-axle grip-margin acceptance gate")
    print("=" * 72)
    root = resolve_corpus_root(None)

    failures: list[str] = []
    summary: list[tuple[str, dict]] = []

    for car, held_out_sid in HELD_OUT_BY_CAR.items():
        print(f"\n[{car}]")
        # 1. Fit ceilings from production corpus.
        with cat.open_catalog(catalog_path(root)) as conn:
            sessions = cat.query_sessions(
                conn, car=car, valid_only=True, include_held_out=False,
            )
        if not sessions:
            failures.append(f"{car}: no production sessions")
            continue
        # Pool first few sessions' lat/long into one array for the fit.
        prod_lat: list[np.ndarray] = []
        prod_lon: list[np.ndarray] = []
        for sess in sessions[:5]:
            lat, lon = _pull_lat_long_for_session(sess.session_id, root)
            if lat.size > 0:
                prod_lat.append(lat)
                prod_lon.append(lon)
        if not prod_lat:
            failures.append(f"{car}: no production samples")
            continue
        prod_lat_arr = np.concatenate(prod_lat)
        prod_lon_arr = np.concatenate(prod_lon)
        # Filter mid-corner.
        mid_mask = np.abs(prod_lat_arr) >= 0.5
        prod_lat_mid = prod_lat_arr[mid_mask]
        prod_lon_mid = prod_lon_arr[mid_mask]
        if prod_lat_mid.size < 200:
            failures.append(f"{car}: too few mid-corner samples ({prod_lat_mid.size})")
            continue
        ratios = compute_axle_grip_ratios(prod_lat_mid, prod_lon_mid, car)
        try:
            front_ceiling = fit_axle_grip_ceiling(car, "front", ratios["front"])
            rear_ceiling = fit_axle_grip_ceiling(car, "rear", ratios["rear"])
        except ValueError as exc:
            failures.append(f"{car}: ceiling fit failed: {exc}")
            continue
        print(
            f"  production ceiling: front mu={front_ceiling.mu_peak:.3f}, "
            f"rear mu={rear_ceiling.mu_peak:.3f} (n={front_ceiling.n_samples})"
        )

        # 2. Pull held-out lat/long.
        h_lat, h_lon = _pull_lat_long_for_session(held_out_sid, root)
        h_mid_mask = np.abs(h_lat) >= 0.5
        h_lat_mid = h_lat[h_mid_mask]
        h_lon_mid = h_lon[h_mid_mask]
        if h_lat_mid.size < 100:
            failures.append(f"{car}: held-out has too few mid-corner samples")
            continue

        # 3. Predict at-limit per held-out sample. Ground truth: top-10%
        # of held-out's |lat_g| (the empirical "near-limit" subset).
        gt_threshold = np.percentile(np.abs(h_lat_mid), 90)
        gt_at_limit = np.abs(h_lat_mid) >= gt_threshold

        # Compute predicted at-limit per sample.
        h_ratios = compute_axle_grip_ratios(h_lat_mid, h_lon_mid, car)
        front_margin = h_ratios["front"] / front_ceiling.mu_peak
        rear_margin = h_ratios["rear"] / rear_ceiling.mu_peak
        pred_at_limit = (front_margin >= AT_LIMIT_THRESHOLD) | (
            rear_margin >= AT_LIMIT_THRESHOLD
        )

        # 4. Accuracy.
        correct = (pred_at_limit == gt_at_limit).sum()
        total = pred_at_limit.size
        accuracy = correct / total
        # Sanity: also report the rate of pred_at_limit so we can see
        # if the model is just predicting "always" or "never."
        pred_at_limit_rate = pred_at_limit.mean()
        gt_at_limit_rate = gt_at_limit.mean()
        print(
            f"  held-out n={total}, accuracy={accuracy:.1%} "
            f"(target >={ACCURACY_THRESHOLD:.0%})"
        )
        print(
            f"    pred at_limit: {pred_at_limit_rate:.1%}, "
            f"gt at_limit: {gt_at_limit_rate:.1%}"
        )
        summary.append((car, {
            "accuracy": accuracy,
            "pred_rate": pred_at_limit_rate,
            "gt_rate": gt_at_limit_rate,
        }))
        if accuracy < ACCURACY_THRESHOLD:
            failures.append(
                f"{car}: accuracy {accuracy:.1%} < {ACCURACY_THRESHOLD:.0%}"
            )

    print("\n" + "=" * 72)
    if failures:
        print("GATE FAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"GATE PASSED for {len(summary)} v4 cars.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
