"""Day 11 acceptance gate -- aero residual correction.

PLAN.md Section 15.3 (second half) acceptance gate:
> Aero residual correction reduces lat-G prediction MAE by >=10% on
> the v4 cars.
> Fallback mode AUTHORIZED: if correction doesn't beat raw, ship
> without (mark fallback).

Gate algorithm (in-process; no DE):
  1. For each v4 car (BMW, Cadillac, Ferrari), pull production lap
     samples (held-out automatically excluded).
  2. For each mid-corner sample (|lat_g| >= 0.5), look up the aero
     map's ld_ratio at the sample's (front_rh, rear_rh, wing).
  3. Build (ld_ratio, speed_ms, observed_lat_g) tuples.
  4. Fit per-car residual correction.
  5. Per-car PASS criterion: improvement_pct >= 10% OR fallback
     authorized (correction_factor=0, fallback_mode_used=True).

The fallback path is itself a valid PASS per PLAN.md: it means
"correction doesn't help on this corpus; ship without."

Run: `uv run python scripts/day_11_gate.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from racingoptimizer.aero.loader import load_aero_map_data
from racingoptimizer.aero.interpolator import AeroSurface
from racingoptimizer.aero.residual_correction import (
    fit_residual_correction,
    improvement_pct,
)
from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.api import (
    catalog_path,
    lap_data,
    laps as ingest_laps,
    resolve_corpus_root,
)


V4_CARS = ("bmw", "cadillac", "ferrari")
IMPROVEMENT_THRESHOLD_PCT = 10.0
MIN_LAT_G = 0.5


def _build_corpus_samples(car: str, root, max_sessions: int = 5) -> list[dict]:
    """Build (ld_ratio, speed_ms, observed_lat_g) samples across the
    car's production sessions.

    For each lap sample with |lat_g| >= MIN_LAT_G, query the aero map
    at the sample's (front_rh, rear_rh, wing) and record the lat_g.
    Wing comes from the session's setup blob.
    """
    aero_dir = Path(root).parent / "aero-maps" if isinstance(root, Path) else None
    if aero_dir is None or not aero_dir.is_dir():
        # Try repo-relative path.
        aero_dir = Path("aero-maps")
    if not aero_dir.is_dir():
        raise FileNotFoundError(f"aero-maps directory not found at {aero_dir}")
    aero_data = load_aero_map_data(car, aero_dir=aero_dir)
    surface = AeroSurface(aero_data)

    with cat.open_catalog(catalog_path(root)) as conn:
        sessions = cat.query_sessions(
            conn, car=car, valid_only=True, include_held_out=False,
        )
    samples: list[dict] = []
    for sess in sessions[:max_sessions]:
        try:
            laps_df = ingest_laps(
                session_id=sess.session_id, valid_only=True, corpus_root=root,
            )
        except Exception:
            continue
        if laps_df.height == 0:
            continue
        # Pick the first valid lap.
        lap_idx = int(laps_df["lap_index"][0])
        try:
            df = lap_data(
                session_id=sess.session_id, lap_index=lap_idx, corpus_root=root,
            )
        except Exception:
            continue
        # Required channels for this gate.
        needed = {"LatAccel", "Speed"}
        ride_height_cols = {
            "LFrideHeight", "RFrideHeight", "LRrideHeight", "RRrideHeight",
        }
        if not needed.issubset(df.columns):
            continue
        if not ride_height_cols.issubset(df.columns):
            continue

        lat = df["LatAccel"].to_numpy().astype(np.float64) / 9.81
        speed = df["Speed"].to_numpy().astype(np.float64)
        # Approximate front/rear ride height as average of left+right.
        # iRacing's `*rideHeight` is in m; convert to mm.
        lf = df["LFrideHeight"].to_numpy().astype(np.float64) * 1000.0
        rf = df["RFrideHeight"].to_numpy().astype(np.float64) * 1000.0
        lr = df["LRrideHeight"].to_numpy().astype(np.float64) * 1000.0
        rr = df["RRrideHeight"].to_numpy().astype(np.float64) * 1000.0
        front_rh_mm = (lf + rf) / 2.0
        rear_rh_mm = (lr + rr) / 2.0

        # Wing angle from setup blob; fallback to mid of map's range.
        try:
            import json
            setup = json.loads(sess.setup) if sess.setup else {}
            wing_deg = float(
                setup.get("Chassis", {})
                .get("RearWing", {})
                .get("RearWingAngle", "")
                .replace(" deg", "")
            ) if isinstance(setup, dict) else (
                aero_data.wing_angles[len(aero_data.wing_angles) // 2]
            )
        except (ValueError, AttributeError, TypeError, KeyError):
            wing_deg = aero_data.wing_angles[len(aero_data.wing_angles) // 2]

        # Filter mid-corner.
        mid_mask = (np.abs(lat) >= MIN_LAT_G) & (speed > 20.0)
        if int(np.sum(mid_mask)) < 50:
            continue
        # Sub-sample to bound memory.
        idxs = np.where(mid_mask)[0]
        if idxs.size > 2000:
            step = idxs.size // 2000
            idxs = idxs[::step]
        for i in idxs:
            f_rh = float(front_rh_mm[i])
            r_rh = float(rear_rh_mm[i])
            v = float(speed[i])
            obs = float(abs(lat[i]))
            try:
                _balance, ld = surface.interpolate(
                    front_rh_mm=f_rh, rear_rh_mm=r_rh,
                    wing_deg=wing_deg, air_density=1.225,
                )
            except Exception:
                continue
            if not np.isfinite(ld) or ld <= 0:
                continue
            samples.append({
                "ld_ratio": float(ld),
                "speed_ms": v,
                "observed_lat_g": obs,
            })
    return samples


def main() -> int:
    print("=" * 72)
    print("Day 11: aero residual correction acceptance gate")
    print("=" * 72)
    root = resolve_corpus_root(None)

    failures: list[str] = []
    for car in V4_CARS:
        print(f"\n[{car}]")
        try:
            samples = _build_corpus_samples(car, root)
        except FileNotFoundError as exc:
            print(f"  ERROR: {exc}")
            failures.append(f"{car}: aero maps not loadable")
            continue
        if len(samples) < 50:
            print(f"  too few samples ({len(samples)}); cannot fit")
            failures.append(f"{car}: too few samples")
            continue
        try:
            correction = fit_residual_correction(car, samples)
        except ValueError as exc:
            print(f"  fit failed: {exc}")
            failures.append(f"{car}: fit failed")
            continue
        improvement = improvement_pct(correction)
        print(
            f"  n_samples={correction.n_samples}, "
            f"correction_factor={correction.correction_factor:+.3f}, "
            f"raw_mae={correction.fit_mae_raw_g:.4f}g, "
            f"corrected_mae={correction.fit_mae_corrected_g:.4f}g"
        )
        print(
            f"  improvement vs raw: {improvement:+.1f}% "
            f"(target >=10% OR fallback authorized)"
        )
        if correction.fallback_mode_used:
            print(
                "  FALLBACK MODE: correction did not beat raw; ship without "
                "(authorized per PLAN.md Section 15.3)."
            )
            # Per PLAN.md, fallback is a valid pass.
            continue
        if improvement < IMPROVEMENT_THRESHOLD_PCT:
            failures.append(
                f"{car}: improvement {improvement:.1f}% < "
                f"{IMPROVEMENT_THRESHOLD_PCT}% AND fallback not triggered "
                f"(this should be unreachable but is a defensive guard)"
            )
        else:
            print(f"  PASS: corrected MAE beats raw by >={IMPROVEMENT_THRESHOLD_PCT}%")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nGATE PASSED for all v4 cars (correction or authorized fallback).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
