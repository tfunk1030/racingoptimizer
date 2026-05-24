"""Held-out per-car accuracy gate (H1-H5).

For each of the 5 held-out IBTs in `docs/physics-rebuild/holdout.sha256`:

  1. Re-fit the per-car physics model from PRODUCTION sessions only
     (held_out=0). This is the train-set the recommender will ever see
     for that car.
  2. Load the held-out IBT's corner_phase_states + observed garage setup.
  3. For each (corner_id, phase) row, ask the model to PREDICT the
     channel values it would expect at the held-out setup + that corner's
     archetype + the row's averaged environment.
  4. Compare predicted .value to the row's actual measured value, and
     check whether the actual sits inside the predicted [lo, hi] CI.

Outputs per (car, channel):
  - n           : sample count
  - mean_abs    : mean |predicted - actual|
  - normed      : mean_abs / max(|actual_std|, 1e-6)
  - coverage    : fraction of actuals inside the predicted CI

Per-car pass criteria:
  - Median coverage across trained channels >= 0.50 (loose; covers the
    sparse-regime channels honestly)
  - Mean coverage on the DENSE-REGIME slice >= 0.85
  - normed_residual median <= 2.0 (predictions within 2 channel-stds)

Run: `uv run python scripts/holdout_accuracy_gate.py`
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from racingoptimizer.constraints import load_constraints
from racingoptimizer.corner.states import corner_phase_states
from racingoptimizer.corner.phase import CornerPhaseKey, Phase
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.api import catalog_path, laps as ingest_laps, resolve_corpus_root
from racingoptimizer.physics.fitter import (
    TARGET_OUTPUT_CHANNELS, fit_per_car,
)
from racingoptimizer.physics.ontology import setup_value


HELDOUT: dict[str, tuple[str, str]] = {
    # car -> (session_id, expected_track)
    "bmw":      ("3f0a05d3f44527bd", "spa_2024_up"),
    "cadillac": ("d236a089300fc0ea", "lagunaseca"),
    "ferrari":  ("fc96805e3b1a27cc", "hockenheim_gp"),
    "acura":    ("72f43fa4527c4260", "daytona_2011_road"),
    "porsche":  ("a3d43056a952ff99", "algarve_gp"),
}


def _env_from_row(row: dict) -> EnvironmentFrame:
    """Build an EnvironmentFrame from a corner_phase_states row.

    Fills missing channels with neutral defaults so the predict pipeline
    has a complete feature vector. The held-out IBT often won't have
    all 12 weather channels populated at corner-phase grain.
    """
    def _f(*keys: str, default: float = 0.0) -> float:
        for k in keys:
            v = row.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                return fv
        return default

    return EnvironmentFrame(
        air_temp_c=_f("air_temp_c_mean", default=25.0),
        air_density=_f("air_density_kg_m3_mean", "air_density_mean", default=1.18),
        air_pressure_mbar=_f(
            "air_pressure_mbar_mean", "air_pressure_pa_mean", default=1013.25,
        ),
        relative_humidity=_f("relative_humidity_mean", default=0.5),
        wind_vel_ms=_f("wind_vel_ms_mean", default=0.0),
        wind_dir_deg=_f("wind_dir_deg_mean", "wind_dir_rad_mean", default=0.0),
        fog_level=_f("fog_level_mean", default=0.0),
        track_temp_c=_f(
            "track_temp_crew_c_mean", "track_temp_c_mean", default=30.0,
        ),
        track_wetness=_f("track_wetness_mean", default=0.0),
        weather_declared_wet=bool(
            _f("weather_declared_wet_mean", default=0.0) > 0.5
        ),
        precip_type=int(_f("precip_type_mean", default=-1.0)),
        skies=int(_f("skies_mean", default=-1.0)),
    )


def _archetype_from_row(row: dict) -> dict[str, float]:
    def _f(*keys: str, default: float = 0.0) -> float:
        for k in keys:
            v = row.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                return fv
        return default

    return {
        "apex_speed_ms": _f("speed_at_min_lat_g_ms", "speed_mean_ms", default=30.0),
        "peak_lat_g": _f("accel_lat_g_max", default=1.0),
        "corner_min_speed_ms": _f("speed_min_ms", "speed_mean_ms", default=20.0),
        "corner_max_speed_ms": _f("speed_max_ms", "speed_mean_ms", default=50.0),
        "corner_duration_s": _f("duration_s", default=1.0),
    }


def _decode_setup(blob: str | None) -> dict:
    if not blob:
        return {}
    try:
        return json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return {}


def _setup_dict_for(car: str, setup_blob: dict, model) -> dict[str, float]:
    """Extract the bounded-setup vector that the model expects, from the
    held-out IBT's YAML blob. Falls back to model.baseline_setup for any
    parameter missing in the YAML."""
    out = dict(model.baseline_setup)
    for name in model.baseline_setup:
        v = setup_value(car, name, setup_blob)
        if v is None:
            continue
        try:
            out[name] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _row_actual(row: dict, channel: str) -> float | None:
    v = row.get(channel)
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


def _channel_signal_std(rows: list[dict], channel: str) -> float:
    vals: list[float] = []
    for r in rows:
        v = _row_actual(r, channel)
        if v is not None:
            vals.append(v)
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1))


def _gate_one_car(car: str, session_id: str, track: str, root: Path) -> dict:
    print(f"\n[{car}] held-out session={session_id} track={track}")
    constraints = load_constraints()

    with cat.open_catalog(catalog_path(root)) as conn:
        held = cat.get_session(conn, session_id)
        if held is None:
            print(f"  SKIP: session {session_id} not in catalog")
            return {"car": car, "skipped": "not_in_catalog"}
        if held.held_out != 1:
            print(f"  SKIP: session not flagged held_out=1")
            return {"car": car, "skipped": "not_held_out"}
        prod_sessions = cat.query_sessions(
            conn, car=car, valid_only=True, include_held_out=False,
        )

    n_prod = len(prod_sessions)
    if n_prod < 1:
        print(f"  SKIP: no production sessions for {car}")
        return {"car": car, "skipped": "no_production"}
    print(f"  production corpus: {n_prod} sessions")

    # Fit per-car (v4) model from production-only.
    session_ids = sorted([s.session_id for s in prod_sessions])
    k_folds = 5 if len(session_ids) >= 3 else 2
    try:
        model = fit_per_car(
            car=car, session_ids=session_ids, corpus_root=root, k_folds=k_folds,
        )
    except Exception as exc:
        print(f"  FAIL: production fit raised {type(exc).__name__}: {exc}")
        return {"car": car, "skipped": "fit_failed", "error": str(exc)}
    print(f"  fit complete: schema=v{model.feature_schema_version}, "
          f"baseline_setup keys={len(model.baseline_setup)}")

    # Load held-out's corner_phase_states (over all valid laps).
    try:
        laps_df = ingest_laps(
            session_id=session_id, valid_only=True, corpus_root=root,
        )
    except Exception as exc:
        print(f"  FAIL: cannot load laps for held-out: {exc}")
        return {"car": car, "skipped": "laps_failed", "error": str(exc)}
    if laps_df.height == 0:
        print(f"  FAIL: no valid laps in held-out IBT")
        return {"car": car, "skipped": "no_laps"}

    all_rows: list[dict] = []
    for lap_idx in laps_df["lap_index"].to_list():
        try:
            cps = corner_phase_states(
                session_id, int(lap_idx), corpus_root=root,
            )
        except Exception:
            continue
        if cps.height == 0:
            continue
        all_rows.extend(cps.to_dicts())

    if not all_rows:
        print(f"  FAIL: no corner-phase rows in held-out laps")
        return {"car": car, "skipped": "no_rows"}
    print(f"  held-out corner-phase rows: {len(all_rows)}")

    # Get observed setup from held-out's YAML (catalog field name = `setup`).
    setup_blob = _decode_setup(getattr(held, "setup", None))
    if not setup_blob:
        # Fall back to model.baseline_setup which is the per-session median
        # — for v4 it's globally aggregated, so this is approximate.
        observed_setup = dict(model.baseline_setup)
        print("  WARN: no setup blob in held-out catalog row; using model baseline")
    else:
        observed_setup = _setup_dict_for(car, setup_blob, model)

    # Per-channel accumulator
    residuals: dict[str, list[float]] = defaultdict(list)
    coverage: dict[str, list[int]] = defaultdict(list)
    sample_counts: dict[str, int] = defaultdict(int)
    regimes: dict[str, list[str]] = defaultdict(list)
    actual_std_cache: dict[str, float] = {
        ch: _channel_signal_std(all_rows, ch) for ch in TARGET_OUTPUT_CHANNELS
    }

    for row in all_rows:
        cid = row.get("corner_id")
        phase_str = row.get("phase")
        if cid is None or phase_str is None:
            continue
        try:
            phase = Phase(str(phase_str))
        except ValueError:
            continue
        # session_id/lap_index are stamped onto the held-out's CPS rows
        # by corner_phase_states; fall back gracefully if missing so the
        # gate degrades to channel-only comparison.
        key = CornerPhaseKey(
            session_id=str(row.get("session_id", session_id)),
            lap_index=int(row.get("lap_index", 0) or 0),
            corner_id=int(cid),
            phase=phase,
        )
        env = _env_from_row(row)
        archetype = _archetype_from_row(row)
        try:
            pred = model.predict(observed_setup, env, key, corner_archetype=archetype)
        except Exception:
            continue
        for channel, conf in pred.states.items():
            actual = _row_actual(row, channel)
            if actual is None:
                continue
            residuals[channel].append(float(conf.value) - actual)
            inside = (conf.lo <= actual <= conf.hi) if conf.lo != conf.hi else False
            coverage[channel].append(1 if inside else 0)
            sample_counts[channel] += 1
            regimes[channel].append(conf.regime)

    # Build summary rows
    summary_rows: list[dict] = []
    for channel in sorted(residuals.keys()):
        res = residuals[channel]
        cov = coverage[channel]
        if not res:
            continue
        sig = actual_std_cache.get(channel, 0.0)
        abs_res = [abs(r) for r in res]
        mean_abs = float(np.mean(abs_res))
        normed = mean_abs / sig if sig > 1e-9 else float("nan")
        cov_frac = float(np.mean(cov)) if cov else 0.0
        # Modal regime label
        regime_mode = max(set(regimes[channel]), key=regimes[channel].count)
        summary_rows.append({
            "channel": channel,
            "n": len(res),
            "mean_abs": mean_abs,
            "actual_std": sig,
            "normed_residual": normed,
            "coverage": cov_frac,
            "regime": regime_mode,
        })

    print(f"  scored {sum(r['n'] for r in summary_rows)} (channel, sample) pairs "
          f"across {len(summary_rows)} channels")

    return {
        "car": car,
        "session_id": session_id,
        "track": track,
        "n_prod_sessions": n_prod,
        "n_holdout_rows": len(all_rows),
        "channels": summary_rows,
    }


def _print_per_car(result: dict) -> None:
    if result.get("skipped"):
        print(f"  -> {result['car']}: SKIPPED ({result['skipped']})")
        return
    rows = result["channels"]
    if not rows:
        print(f"  -> {result['car']}: NO PREDICTIONS")
        return
    # Sort by normed_residual descending (worst channels first)
    rows_sorted = sorted(
        rows, key=lambda r: (
            float("inf") if math.isnan(r["normed_residual"]) else r["normed_residual"]
        ), reverse=True,
    )
    print(f"\n  -- {result['car']} per-channel accuracy (held-out @ {result['track']}) --")
    print(f"    {'channel':<35} {'n':>5} {'mean_abs':>10} {'sig_std':>10} "
          f"{'normed':>8} {'cov':>6} regime")
    for r in rows_sorted[:25]:
        ns = "nan" if math.isnan(r["normed_residual"]) else f"{r['normed_residual']:.2f}"
        print(f"    {r['channel']:<35} {r['n']:>5d} {r['mean_abs']:>10.4f} "
              f"{r['actual_std']:>10.4f} {ns:>8} {r['coverage']:>6.2f} {r['regime']}")
    # Aggregate medians
    cov = [r["coverage"] for r in rows]
    normed = [r["normed_residual"] for r in rows if not math.isnan(r["normed_residual"])]
    dense = [r["coverage"] for r in rows if r["regime"] in ("dense", "confident")]
    print(f"    summary: median_cov={statistics.median(cov):.2f} "
          f"median_normed={(statistics.median(normed) if normed else float('nan')):.2f} "
          f"dense_mean_cov={(statistics.mean(dense) if dense else float('nan')):.2f}")


def _gate_pass(result: dict) -> tuple[bool, str]:
    if result.get("skipped"):
        return False, f"skipped:{result['skipped']}"
    rows = result["channels"]
    if not rows:
        return False, "no_predictions"
    cov = [r["coverage"] for r in rows]
    normed = [r["normed_residual"] for r in rows if not math.isnan(r["normed_residual"])]
    dense_cov = [r["coverage"] for r in rows if r["regime"] in ("dense", "confident")]
    med_cov = statistics.median(cov)
    med_normed = statistics.median(normed) if normed else float("inf")
    dense_mean = statistics.mean(dense_cov) if dense_cov else 0.0
    crit = []
    if med_cov < 0.50:
        crit.append(f"median_cov={med_cov:.2f}<0.50")
    if med_normed > 2.0:
        crit.append(f"median_normed={med_normed:.2f}>2.0")
    if dense_cov and dense_mean < 0.85:
        crit.append(f"dense_mean_cov={dense_mean:.2f}<0.85")
    return (len(crit) == 0, ";".join(crit) if crit else "ok")


def main() -> int:
    print("=" * 72)
    print("Held-out per-car accuracy gate (H1-H5)")
    print("=" * 72)
    root = Path(resolve_corpus_root(None))

    results: list[dict] = []
    for car, (session_id, track) in HELDOUT.items():
        try:
            res = _gate_one_car(car, session_id, track, root)
        except Exception as exc:
            print(f"  ERROR: {car} raised {type(exc).__name__}: {exc}")
            res = {"car": car, "skipped": "exception", "error": str(exc)}
        results.append(res)

    print("\n" + "=" * 72)
    print("PER-CAR RESULTS")
    print("=" * 72)
    pass_count = 0
    fail_lines: list[str] = []
    for r in results:
        _print_per_car(r)
        ok, why = _gate_pass(r)
        if ok:
            pass_count += 1
            print(f"    GATE PASS: {r['car']}")
        else:
            fail_lines.append(f"    GATE FAIL: {r['car']} -- {why}")
            print(f"    GATE FAIL: {r['car']} -- {why}")

    print("\n" + "=" * 72)
    print(f"OVERALL: {pass_count}/{len(results)} cars pass")
    if fail_lines:
        print("\n".join(fail_lines))
    print("=" * 72)

    # Dump JSON for downstream consumers
    out_path = Path("docs/physics-rebuild/holdout_accuracy_latest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[saved JSON: {out_path}]")
    return 0 if pass_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
