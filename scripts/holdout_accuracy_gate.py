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

Two gate criteria run side-by-side:

* Per-channel gate (P1.1 of ``docs/accuracy-rebuild-2026-05-24/PLAN.md``;
  the **hard fail** criterion). Each scoring channel that drives
  recommendations has a ``mean_abs`` AND ``normed_residual`` target
  (see ``_PER_CHANNEL_THRESHOLDS``). The gate fails when ANY car fails
  ANY non-driver-input row.

* Aggregate gate (the original loose criterion, kept for trend
  tracking only). Median coverage >= 0.50, dense-regime mean coverage
  >= 0.85, median normed_residual <= 2.0.

Run: `uv run python scripts/holdout_accuracy_gate.py`

Fast iteration (one car): `uv run python scripts/holdout_accuracy_gate.py --car acura`

Target-track validation (score only that track's sessions):

  Cross-track (strict): train *without* the track, score on-track —
  ``--car acura --eval-track belleisle``

  In-sample (matches ``optimize recommend``): train on full corpus *including*
  the track, score on-track only —
  ``--car acura --eval-track belleisle --in-sample``
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner.phase import CornerPhaseKey, Phase
from racingoptimizer.corner.states import corner_phase_states
from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.api import catalog_path, resolve_corpus_root
from racingoptimizer.ingest.api import laps as ingest_laps
from racingoptimizer.physics.fitter import (
    DRIVER_CONTROL_COLUMNS,
    TARGET_OUTPUT_CHANNELS,
    fit_per_car,
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


# Per-channel thresholds (PLAN.md §3 P1.1). Map ``channel ->
# (mean_abs_target, normed_residual_target)``. ``mean_abs_target`` of
# ``None`` triggers the "30% of channel signal_std" rule (used by
# damper force where channel scale varies per car/track).
_PER_CHANNEL_THRESHOLDS: dict[str, tuple[float | None, float | None]] = {
    # Grip-balance channels
    "accel_lat_g_max": (0.30, 0.5),
    "accel_lon_g_min": (0.30, 0.5),
    "accel_lon_g_max": (0.30, 0.5),
    "understeer_angle_mean_rad": (0.10, 0.5),
    # Wheel dynamic ride heights (telemetry-driven)
    "lf_ride_height_mean_mm": (3.0, 0.5),
    "rf_ride_height_mean_mm": (3.0, 0.5),
    "lr_ride_height_mean_mm": (3.0, 0.5),
    "rr_ride_height_mean_mm": (3.0, 0.5),
    # Static garage RH (deterministic per P0.2; tighter)
    "setup_static_lf_ride_height_mm": (1.0, 0.2),
    "setup_static_rf_ride_height_mm": (1.0, 0.2),
    "setup_static_lr_ride_height_mm": (1.0, 0.2),
    "setup_static_rr_ride_height_mm": (1.0, 0.2),
    # Damper force p99 -- mean_abs_target=None => 30% of channel std
    "damper_force_p99_n": (None, 0.5),
}

# Fraction of ``actual_std`` used for the dynamic ``mean_abs`` target
# when the threshold tuple's first element is ``None``.
_DYNAMIC_MEAN_ABS_FRACTION: float = 0.30

# Channels that most directly drive ``[OPT]`` on mid-corner-heavy GTP tracks.
# Printed first in the report; informational only (same P1.1 thresholds).
_PRIORITY_CHANNELS: tuple[str, ...] = (
    "lf_ride_height_mean_mm",
    "rf_ride_height_mean_mm",
    "lr_ride_height_mean_mm",
    "rr_ride_height_mean_mm",
    "understeer_angle_mean_rad",
    "accel_lat_g_max",
    "setup_static_lf_ride_height_mm",
    "setup_static_rf_ride_height_mm",
    "setup_static_lr_ride_height_mm",
    "setup_static_rr_ride_height_mm",
)


def _track_needle_matches(needle: str, track_slug: str) -> bool:
    n = needle.lower().replace("-", "").replace("_", "")
    t = track_slug.lower().replace("-", "").replace("_", "")
    return bool(n) and (n in t or t in n)


def _per_channel_pass(rows: list[dict]) -> tuple[bool, list[str]]:
    """Evaluate each row in ``rows`` against ``_PER_CHANNEL_THRESHOLDS``.

    ``rows`` is a list of dicts shaped like ``_gate_one_car``'s
    ``channels`` output (``channel``, ``mean_abs``, ``normed_residual``,
    ``actual_std`` per row).

    Returns ``(ok, failed)``: ``ok`` is True iff every gated channel
    that appears in ``rows`` passes both its mean_abs and normed
    targets. ``failed`` lists human-readable reasons of the form
    ``"channel: mean_abs=X.X > target | normed=Y.Y > target"``.
    Channels not present in ``_PER_CHANNEL_THRESHOLDS`` are skipped (no
    gating). Channels in the threshold dict but missing from ``rows``
    are also skipped (not a failure -- the held-out IBT may not have
    those channels available).
    """
    failed: list[str] = []
    for row in rows:
        channel = row.get("channel")
        if channel not in _PER_CHANNEL_THRESHOLDS:
            continue
        mean_abs_target, normed_target = _PER_CHANNEL_THRESHOLDS[channel]
        try:
            mean_abs = float(row.get("mean_abs", float("nan")))
            normed = float(row.get("normed_residual", float("nan")))
            actual_std = float(row.get("actual_std", 0.0))
        except (TypeError, ValueError):
            failed.append(f"{channel}: malformed row")
            continue

        # Resolve the dynamic mean_abs target ("None" => fraction of std)
        if mean_abs_target is None:
            if actual_std > 0.0 and math.isfinite(actual_std):
                effective_mean_abs_target: float | None = (
                    _DYNAMIC_MEAN_ABS_FRACTION * actual_std
                )
            else:
                effective_mean_abs_target = None
        else:
            effective_mean_abs_target = float(mean_abs_target)

        reasons: list[str] = []
        if (
            effective_mean_abs_target is not None
            and math.isfinite(mean_abs)
            and mean_abs > effective_mean_abs_target
        ):
            reasons.append(
                f"mean_abs={mean_abs:.3f} > {effective_mean_abs_target:.3f}"
            )
        if (
            normed_target is not None
            and math.isfinite(normed)
            and normed > float(normed_target)
        ):
            reasons.append(f"normed={normed:.2f} > {normed_target:.2f}")
        if reasons:
            failed.append(f"{channel}: " + " | ".join(reasons))
    return (len(failed) == 0, failed)


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
    """Map a corner-phase row to v4 archetype feature names.

    Prefer ``ARCHETYPE_KEYS`` already on the row (after
    ``_attach_corner_archetypes``); fall back to an empty dict so
    predict uses baseline/zero rather than legacy mis-named keys.
    """
    from racingoptimizer.physics.corner_schedule import archetype_dict_from_row

    return archetype_dict_from_row(row)


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


def _corner_phase_key_from_row(row: dict, fallback_session_id: str) -> CornerPhaseKey | None:
    """Build a CornerPhaseKey from a corner-phase row.

    The held-out gate reads persisted corner-phase rows that have evolved over
    time; older artifacts may carry different lap/session column names. Keep
    this parser tolerant so the gate can still score historical corpora.
    """
    cid = row.get("corner_id")
    phase_raw = row.get("phase")
    if cid is None or phase_raw is None:
        return None
    try:
        phase = Phase(str(phase_raw))
    except ValueError:
        return None

    sid = str(
        row.get("session_id")
        or row.get("source_session_id")
        or fallback_session_id
    )
    lap_raw = (
        row.get("lap_index")
        if row.get("lap_index") is not None
        else row.get("lap")
    )
    if lap_raw is None:
        lap_raw = row.get("source_lap_index")
    try:
        lap_index = int(lap_raw if lap_raw is not None else 0)
        corner_id = int(cid)
    except (TypeError, ValueError):
        return None

    return CornerPhaseKey(
        session_id=sid,
        lap_index=lap_index,
        corner_id=corner_id,
        phase=phase,
    )


def _load_corner_phase_rows(
    session_ids: list[str], root: Path,
) -> list[dict]:
    """Aggregate corner-phase rows across all valid laps for ``session_ids``."""
    all_rows: list[dict] = []
    for session_id in session_ids:
        try:
            laps_df = ingest_laps(
                session_id=session_id, valid_only=True, corpus_root=root,
            )
        except Exception:
            continue
        if laps_df.height == 0:
            continue
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
        return []
    from racingoptimizer.physics.fitter import _attach_corner_archetypes

    return _attach_corner_archetypes(pl.DataFrame(all_rows)).to_dicts()


def _score_corner_phase_rows(
    model,
    car: str,
    rows: list[dict],
    *,
    setup_by_session: dict[str, dict[str, float]],
    default_setup: dict[str, float],
    track: str,
    fallback_session_id: str,
) -> tuple[list[dict], bool]:
    """Predict corner-phase channels; return summary rows + predict-fail flag."""
    residuals: dict[str, list[float]] = defaultdict(list)
    coverage: dict[str, list[int]] = defaultdict(list)
    regimes: dict[str, list[str]] = defaultdict(list)
    actual_std_cache: dict[str, float] = {
        ch: _channel_signal_std(rows, ch) for ch in TARGET_OUTPUT_CHANNELS
    }
    predict_fail_logged = False
    for row in rows:
        key = _corner_phase_key_from_row(row, fallback_session_id)
        if key is None:
            continue
        sid = str(
            row.get("session_id")
            or row.get("source_session_id")
            or fallback_session_id
        )
        observed_setup = setup_by_session.get(sid, default_setup)
        env = _env_from_row(row)
        archetype = _archetype_from_row(row)
        context_features: dict[str, float] = {}
        for col in DRIVER_CONTROL_COLUMNS:
            v = _row_actual(row, col)
            if v is not None:
                context_features[col] = v
        try:
            pred = model.predict(
                observed_setup,
                env,
                key,
                corner_archetype=archetype,
                track=track,
                context_features=context_features or None,
            )
        except Exception as exc:
            if not predict_fail_logged:
                print(f"  WARN: predict failed: {type(exc).__name__}: {exc}")
                predict_fail_logged = True
            continue
        for channel, conf in pred.states.items():
            actual = _row_actual(row, channel)
            if actual is None:
                continue
            residuals[channel].append(float(conf.value) - actual)
            inside = (conf.lo <= actual <= conf.hi) if conf.lo != conf.hi else False
            coverage[channel].append(1 if inside else 0)
            regimes[channel].append(conf.regime)

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
    return summary_rows, predict_fail_logged


def _score_static_rh_sessions(
    model,
    car: str,
    sessions: list,
) -> list[dict]:
    """In-garage static RH: ``predict_setup_readouts`` vs YAML readouts."""
    from racingoptimizer.physics.fitter import _extract_setup_readouts

    static_channels = (
        "setup_static_lf_ride_height_mm",
        "setup_static_rf_ride_height_mm",
        "setup_static_lr_ride_height_mm",
        "setup_static_rr_ride_height_mm",
    )
    residuals: dict[str, list[float]] = {ch: [] for ch in static_channels}
    env = EnvironmentFrame(
        air_temp_c=25.0, track_temp_c=30.0,
    )
    for sess in sessions:
        setup_blob = _decode_setup(getattr(sess, "setup", None))
        if not setup_blob:
            continue
        observed = _setup_dict_for(car, setup_blob, model)
        actuals = _extract_setup_readouts(setup_blob)
        try:
            predicted = model.predict_setup_readouts(observed, env)
        except Exception:
            continue
        for ch in static_channels:
            actual = actuals.get(ch)
            pred = predicted.get(ch)
            if actual is None or pred is None:
                continue
            residuals[ch].append(float(pred) - float(actual))

    rows: list[dict] = []
    for channel in static_channels:
        res = residuals[channel]
        if not res:
            continue
        abs_res = [abs(r) for r in res]
        mean_abs = float(np.mean(abs_res))
        actual_std = float(np.std([abs(r) for r in res], ddof=1)) if len(res) > 1 else 0.0
        normed = mean_abs / actual_std if actual_std > 1e-9 else float("nan")
        rows.append({
            "channel": channel,
            "n": len(res),
            "mean_abs": mean_abs,
            "actual_std": actual_std,
            "normed_residual": normed,
            "coverage": float("nan"),
            "regime": "static_rh",
        })
    return rows


def _gate_track_eval(
    car: str,
    track_input: str,
    root: Path,
    *,
    in_sample: bool = False,
) -> dict:
    """Score on-track sessions; training set depends on ``in_sample``."""
    if in_sample:
        print(
            f"\n[{car}] track-eval: {track_input!r} "
            "(in-sample: train includes this track — same pool as recommend)"
        )
    else:
        print(
            f"\n[{car}] track-eval: {track_input!r} "
            "(cross-track: train excludes this track)"
        )

    with cat.open_catalog(catalog_path(root)) as conn:
        prod_sessions = cat.query_sessions(
            conn, car=car, valid_only=True, include_held_out=False,
        )
        catalog_tracks = sorted({s.track for s in prod_sessions})
        from racingoptimizer.cli.recommend import _match_track_slug

        matched, ambiguous = _match_track_slug(track_input, catalog_tracks)
        if matched is None:
            if ambiguous:
                print(f"  SKIP: ambiguous track {track_input!r}: {ambiguous}")
                return {"car": car, "skipped": "ambiguous_track", "candidates": ambiguous}
            print(f"  SKIP: no catalog track matches {track_input!r}")
            return {"car": car, "skipped": "track_not_found"}
        eval_sessions = [s for s in prod_sessions if s.track == matched]
        if in_sample:
            train_sessions = list(prod_sessions)
        else:
            train_sessions = [s for s in prod_sessions if s.track != matched]

    n_eval = len(eval_sessions)
    n_train = len(train_sessions)
    if n_eval < 1:
        print(f"  SKIP: no production sessions on {matched}")
        return {"car": car, "skipped": "no_eval_sessions", "track": matched}
    if not in_sample and n_train < 1:
        print(f"  SKIP: no training sessions outside {matched}")
        return {"car": car, "skipped": "no_train_sessions", "track": matched}

    print(f"  matched track slug: {matched}")
    if in_sample:
        print(
            f"  train: {n_train} sessions (full corpus) | "
            f"eval: {n_eval} on-track only"
        )
    else:
        print(
            f"  train: {n_train} sessions (excl. {matched}) | "
            f"eval: {n_eval} on-track"
        )

    train_ids = sorted(s.session_id for s in train_sessions)
    eval_ids = sorted(s.session_id for s in eval_sessions)
    k_folds = 5 if len(train_ids) >= 3 else 2
    try:
        model = fit_per_car(
            car=car, session_ids=train_ids, corpus_root=root, k_folds=k_folds,
        )
    except Exception as exc:
        print(f"  FAIL: train fit raised {type(exc).__name__}: {exc}")
        return {"car": car, "skipped": "fit_failed", "error": str(exc)}
    print(f"  fit complete: schema=v{model.feature_schema_version}")

    all_rows = _load_corner_phase_rows(eval_ids, root)
    if not all_rows:
        print("  FAIL: no corner-phase rows on eval track")
        return {"car": car, "skipped": "no_rows", "track": matched}
    print(f"  eval corner-phase rows: {len(all_rows)}")

    setup_by_session: dict[str, dict[str, float]] = {}
    for sess in eval_sessions:
        blob = _decode_setup(getattr(sess, "setup", None))
        if blob:
            setup_by_session[sess.session_id] = _setup_dict_for(car, blob, model)

    summary_rows, _ = _score_corner_phase_rows(
        model,
        car,
        all_rows,
        setup_by_session=setup_by_session,
        default_setup=dict(model.baseline_setup),
        track=matched,
        fallback_session_id=eval_ids[0],
    )
    static_rows = _score_static_rh_sessions(model, car, eval_sessions)
    if static_rows:
        print(f"  static garage RH (in-garage): {len(static_rows)} channels scored")
        summary_rows = summary_rows + static_rows

    print(f"  scored {sum(r['n'] for r in summary_rows)} (channel, sample) pairs "
          f"across {len(summary_rows)} channels")

    per_channel_ok, per_channel_failed = _per_channel_pass(summary_rows)
    gated_channels = [
        r["channel"] for r in summary_rows
        if r["channel"] in _PER_CHANNEL_THRESHOLDS
    ]
    return {
        "car": car,
        "eval_mode": "track_in_sample" if in_sample else "track_cross",
        "track": matched,
        "eval_track_input": track_input,
        "in_sample": in_sample,
        "n_train_sessions": n_train,
        "n_eval_sessions": n_eval,
        "n_holdout_rows": len(all_rows),
        "channels": summary_rows,
        "per_channel_pass": gated_channels if per_channel_ok else [
            ch for ch in gated_channels
            if not any(line.startswith(f"{ch}:") for line in per_channel_failed)
        ],
        "per_channel_failed": per_channel_failed,
    }


def _gate_one_car(car: str, session_id: str, track: str, root: Path) -> dict:
    print(f"\n[{car}] held-out session={session_id} track={track}")

    with cat.open_catalog(catalog_path(root)) as conn:
        held = cat.get_session(conn, session_id)
        if held is None:
            print(f"  SKIP: session {session_id} not in catalog")
            return {"car": car, "skipped": "not_in_catalog"}
        if held.held_out != 1:
            print("  SKIP: session not flagged held_out=1")
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

    all_rows = _load_corner_phase_rows([session_id], root)
    if not all_rows:
        print("  FAIL: no corner-phase rows in held-out laps")
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

    summary_rows, _ = _score_corner_phase_rows(
        model,
        car,
        all_rows,
        setup_by_session={session_id: observed_setup},
        default_setup=observed_setup,
        track=track,
        fallback_session_id=session_id,
    )

    print(f"  scored {sum(r['n'] for r in summary_rows)} (channel, sample) pairs "
          f"across {len(summary_rows)} channels")

    per_channel_ok, per_channel_failed = _per_channel_pass(summary_rows)
    gated_channels = [
        r["channel"] for r in summary_rows
        if r["channel"] in _PER_CHANNEL_THRESHOLDS
    ]
    return {
        "car": car,
        "eval_mode": "heldout",
        "session_id": session_id,
        "track": track,
        "n_prod_sessions": n_prod,
        "n_holdout_rows": len(all_rows),
        "channels": summary_rows,
        "per_channel_pass": gated_channels if per_channel_ok else [
            ch for ch in gated_channels
            if not any(line.startswith(f"{ch}:") for line in per_channel_failed)
        ],
        "per_channel_failed": per_channel_failed,
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
    mode = result.get("eval_mode", "heldout")
    if mode == "track_in_sample":
        title = (
            f"track-eval @ {result['track']} in-sample "
            f"({result.get('n_train_sessions', '?')} train incl. track; "
            f"{result.get('n_eval_sessions', '?')} eval sessions)"
        )
    elif mode in ("track", "track_cross"):
        title = (
            f"track-eval @ {result['track']} cross-track "
            f"(train excl. track; {result.get('n_eval_sessions', '?')} eval sessions)"
        )
    else:
        title = f"held-out @ {result['track']}"
    print(f"\n  -- {result['car']} per-channel accuracy ({title}) --")
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

    failed = result.get("per_channel_failed") or []
    passed = result.get("per_channel_pass") or []
    priority_failed = [
        line for line in failed
        if any(line.startswith(f"{ch}:") for ch in _PRIORITY_CHANNELS)
    ]
    if priority_failed:
        print("    priority channels (wheel RH / understeer / lat G / static RH):")
        for line in priority_failed:
            print(f"      - {line}")
    if failed:
        print("    per-channel FAIL:")
        for line in failed:
            print(f"      - {line}")
    if passed:
        print(f"    per-channel PASS: {', '.join(passed)}")


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--car",
        action="append",
        dest="cars",
        metavar="CAR",
        help=(
            "Run gate for this car only (repeatable). "
            f"Choices: {', '.join(sorted(HELDOUT))}. Default: all five."
        ),
    )
    parser.add_argument(
        "--eval-track",
        metavar="TRACK",
        default=None,
        help=(
            "Score on all production sessions at TRACK only. Default: cross-track "
            "(train excludes TRACK). With --in-sample, train on the full corpus "
            "(same as optimize recommend). Ignores H1-H5 held-out IBTs."
        ),
    )
    parser.add_argument(
        "--in-sample",
        action="store_true",
        help=(
            "With --eval-track: train including the target track (full per-car "
            "corpus), then score only that track's sessions. Use this to check "
            "Belle Isle accuracy the way recommend actually fits."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cars_filter: list[str] | None = None
    if args.cars:
        unknown = [c for c in args.cars if c not in HELDOUT]
        if unknown:
            print(
                f"Unknown --car value(s): {unknown}. "
                f"Valid: {sorted(HELDOUT)}",
                file=sys.stderr,
            )
            return 2
        cars_filter = list(dict.fromkeys(args.cars))  # preserve order, dedupe

    print("=" * 72)
    if args.eval_track:
        if args.in_sample:
            print("Track-eval accuracy gate (in-sample: full train, on-track score)")
        else:
            print("Track-eval accuracy gate (cross-track: train excludes target)")
        cars_for_track = cars_filter or ["acura"]
        flags = f"eval-track: {args.eval_track}"
        if args.in_sample:
            flags += " | in-sample"
        print(f"Cars: {', '.join(cars_for_track)} | {flags}")
    else:
        print("Held-out per-car accuracy gate (H1-H5)")
        if cars_filter is not None:
            print(f"Cars: {', '.join(cars_filter)}")
    print("=" * 72)
    root = Path(resolve_corpus_root(None))

    results: list[dict] = []
    if args.eval_track:
        for car in (cars_filter or ["acura"]):
            if cars_filter is None and car not in HELDOUT:
                continue
            try:
                res = _gate_track_eval(
                    car, args.eval_track, root, in_sample=args.in_sample,
                )
            except Exception as exc:
                print(f"  ERROR: {car} raised {type(exc).__name__}: {exc}")
                res = {"car": car, "skipped": "exception", "error": str(exc)}
            results.append(res)
    else:
        heldout_items = (
            [(c, HELDOUT[c]) for c in cars_filter]
            if cars_filter is not None
            else list(HELDOUT.items())
        )
        for car, (session_id, track) in heldout_items:
            try:
                res = _gate_one_car(car, session_id, track, root)
            except Exception as exc:
                print(f"  ERROR: {car} raised {type(exc).__name__}: {exc}")
                res = {"car": car, "skipped": "exception", "error": str(exc)}
            results.append(res)

    print("\n" + "=" * 72)
    print("PER-CAR RESULTS")
    print("=" * 72)
    aggregate_pass = 0
    per_channel_pass = 0
    aggregate_fail_lines: list[str] = []
    per_channel_fail_lines: list[str] = []
    for r in results:
        _print_per_car(r)
        ok, why = _gate_pass(r)
        if ok:
            aggregate_pass += 1
            print(f"    AGGREGATE GATE (informational) PASS: {r['car']}")
        else:
            aggregate_fail_lines.append(
                f"    AGGREGATE GATE (informational) FAIL: {r['car']} -- {why}"
            )
            print(
                f"    AGGREGATE GATE (informational) FAIL: {r['car']} -- {why}"
            )
        if r.get("skipped"):
            per_channel_fail_lines.append(
                f"    PER-CHANNEL GATE FAIL: {r['car']} -- skipped:{r['skipped']}"
            )
            continue
        failed = r.get("per_channel_failed") or []
        if not failed:
            per_channel_pass += 1
            print(f"    PER-CHANNEL GATE PASS: {r['car']}")
        else:
            per_channel_fail_lines.append(
                f"    PER-CHANNEL GATE FAIL: {r['car']} -- "
                + "; ".join(failed[:3])
                + (f" (+{len(failed) - 3} more)" if len(failed) > 3 else "")
            )
            print(
                f"    PER-CHANNEL GATE FAIL: {r['car']} -- {len(failed)} channel(s)"
            )

    print("\n" + "=" * 72)
    print(
        f"AGGREGATE (informational): {aggregate_pass}/{len(results)} cars pass"
    )
    if aggregate_fail_lines:
        print("\n".join(aggregate_fail_lines))
    print(
        f"PER-CHANNEL (gating): {per_channel_pass}/{len(results)} cars pass"
    )
    if per_channel_fail_lines:
        print("\n".join(per_channel_fail_lines))
    print("=" * 72)

    # Dump JSON for downstream consumers
    out_path = Path("docs/physics-rebuild/holdout_accuracy_latest.json")
    if cars_filter is not None and out_path.exists():
        try:
            prior = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(prior, list):
                by_car = {r["car"]: r for r in prior if isinstance(r, dict) and "car" in r}
                for r in results:
                    by_car[r["car"]] = r
                results_to_write = list(by_car.values())
            else:
                results_to_write = results
        except (json.JSONDecodeError, OSError):
            results_to_write = results
    else:
        results_to_write = results
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results_to_write, indent=2, default=str))
    print(f"\n[saved JSON: {out_path}]")
    return 0 if per_channel_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
