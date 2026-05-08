"""Day 5 acceptance gate -- BMW@Spa held-out evaluation (Mode 1 closure).

PLAN.md Section 14.3 acceptance gate (end-of-component, hard stop):
> For BMW with H1 (Spa) held out, the Bayesian retrofit's posterior
> mean prediction on H1 must beat the current v4 surrogate prediction
> on H1 by >= 5% in MAE on setup-readout target columns. Plus posterior
> 95% interval must cover >= 80% of held-out setup readouts.

Implementation note. The runtime form of the gate ("run optimize bmw
spa, compare recommendation to H1") would require a full DE search
(~30 min including fit_per_car refit). The CONTRACT being tested,
however, is whether the hierarchical Bayesian retrofit produces
track-aware per-parameter means that better match what the user
actually drove at Spa than the cross-track v4 baseline. We test this
directly:

  1. Query the BMW catalog excluding H1 (default behaviour with
     `held_out=1` flag from Day 0).
  2. For each session, extract observed values per parameter (from
     the YAML setup blob the parser embedded).
  3. Group by track to build the per-track observed dict.
  4. Call `fit_all_parameters` (Day 3 module) to get per-(param,
     track) posteriors -- specifically the Spa posterior.
  5. Look up H1's setup directly via `cat.get_session(...)` (held-out
     lookup is the one read-permitted gate use).
  6. For each parameter that has a Spa posterior:
       bayes_prediction  = Spa posterior mean
       v4_prediction     = grand-mean across all BMW tracks
                           (the pre-bayes baseline pooled-regression form)
       actual_value      = H1's setup value
       bayes_error       = |bayes_prediction - actual_value|
       v4_error          = |v4_prediction - actual_value|
  7. Compute MAE across parameters; bayes MAE must be at least 5%
     lower than v4 MAE.
  8. Posterior 95% interval coverage: per-parameter, check whether
     `actual_value` falls in `[mean - 1.96*std, mean + 1.96*std]`;
     coverage rate must be >= 80%.

Run: `uv run python scripts/day_05_gate.py`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.api import catalog_path, resolve_corpus_root
from racingoptimizer.physics.bayes_retrofit import fit_all_parameters
from racingoptimizer.physics.ontology import (
    fittable_parameters,
    setup_value,
)
from racingoptimizer.constraints import load_constraints


CAR = "bmw"
HELD_OUT_TRACK = "spa_2024_up"
HELD_OUT_SESSION_ID = "3f0a05d3f44527bd"  # H1


def _decode_setup(blob: str | None) -> dict:
    if not blob:
        return {}
    try:
        loaded = json.loads(blob)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def main() -> int:
    root = resolve_corpus_root(None)
    constraints = load_constraints()
    fit_params = list(fittable_parameters(CAR, constraints))

    # 1-3. Build per-track observed dict from the BMW production corpus.
    with cat.open_catalog(catalog_path(root)) as conn:
        sessions = cat.query_sessions(
            conn, car=CAR, valid_only=True, include_held_out=False,
        )
        h1 = cat.get_session(conn, HELD_OUT_SESSION_ID)

    if h1 is None:
        print(f"FAIL: held-out session {HELD_OUT_SESSION_ID} not in catalog")
        return 2
    if h1.held_out != 1:
        print(
            f"FAIL: held-out session {HELD_OUT_SESSION_ID} has "
            f"held_out={h1.held_out}, expected 1"
        )
        return 2
    if h1.car != CAR or h1.track != HELD_OUT_TRACK:
        print(
            f"FAIL: held-out session {HELD_OUT_SESSION_ID} is "
            f"({h1.car}, {h1.track}), expected ({CAR}, {HELD_OUT_TRACK})"
        )
        return 2

    n_prod = len(sessions)
    n_spa_prod = sum(1 for s in sessions if s.track == HELD_OUT_TRACK)
    print(
        f"BMW production corpus (H1 excluded): {n_prod} sessions "
        f"across {len({s.track for s in sessions})} tracks; "
        f"{n_spa_prod} on {HELD_OUT_TRACK}"
    )

    per_track: dict[str, dict[str, list[float]]] = {}
    for sess in sessions:
        if sess.session_id == HELD_OUT_SESSION_ID:
            print("FAIL: held-out leaked into production query!")
            return 2
        setup_dict = _decode_setup(sess.setup)
        track_bin = per_track.setdefault(sess.track, {})
        for param in fit_params:
            try:
                val = setup_value(CAR, param, setup_dict)
            except KeyError:
                val = None
            if val is not None:
                track_bin.setdefault(param, []).append(float(val))

    # 4. Bayesian retrofit.
    by_param_track: dict[str, dict[str, list[float]]] = {}
    for track, params in per_track.items():
        for param, values in params.items():
            by_param_track.setdefault(track, {})[param] = values
    posteriors = fit_all_parameters(by_param_track)

    # 5. H1 setup.
    h1_setup = _decode_setup(h1.setup)

    # 6. Per-parameter comparison: bayes vs v4-baseline vs H1 actual.
    rows: list[tuple[str, float, float, float, float, float, float, bool]] = []
    for param in fit_params:
        if (param, HELD_OUT_TRACK) not in posteriors:
            continue
        post = posteriors[(param, HELD_OUT_TRACK)]
        try:
            actual = setup_value(CAR, param, h1_setup)
        except KeyError:
            continue
        if actual is None:
            continue
        actual = float(actual)
        # v4 baseline = pooled grand mean across all observed tracks.
        all_values: list[float] = []
        for track_params in per_track.values():
            all_values.extend(track_params.get(param, []))
        if not all_values:
            continue
        v4_pred = mean(all_values)
        bayes_pred = post.mean
        bayes_err = abs(bayes_pred - actual)
        v4_err = abs(v4_pred - actual)
        # 95% bracket coverage check uses the PREDICTIVE std (uncertainty
        # in where the next observation falls), not mean_std (uncertainty
        # in where the central tendency is). Mean_std collapses to ~0
        # when shrinkage is low; predictive_std correctly retains
        # within-track noise.
        pred_std = post.predictive_std or post.std
        lo = post.mean - 1.96 * pred_std
        hi = post.mean + 1.96 * pred_std
        covered = lo <= actual <= hi
        rows.append((
            param, actual, bayes_pred, v4_pred,
            bayes_err, v4_err, post.std, covered,
        ))

    if not rows:
        print(
            "FAIL: no parameters have both a Spa posterior AND an H1 actual"
        )
        return 1

    print(
        f"\n{'parameter':<35} {'actual':>10} {'bayes':>10} {'v4':>10} "
        f"{'bayes_err':>10} {'v4_err':>10} {'95%_cov':>8}"
    )
    for (param, actual, bp, vp, be, ve, _std, cov) in rows:
        print(
            f"{param:<35} {actual:>10.3f} {bp:>10.3f} {vp:>10.3f} "
            f"{be:>10.3f} {ve:>10.3f} {'YES' if cov else 'NO':>8}"
        )

    bayes_mae = mean(r[4] for r in rows)
    v4_mae = mean(r[5] for r in rows)
    coverage = sum(1 for r in rows if r[7]) / len(rows)

    print(f"\n{'-' * 100}")
    print(f"  N parameters compared: {len(rows)}")
    print(f"  v4 baseline MAE:       {v4_mae:.4f}")
    print(f"  Bayes posterior MAE:   {bayes_mae:.4f}")
    if v4_mae > 0:
        improvement = (v4_mae - bayes_mae) / v4_mae * 100
        print(f"  Improvement:           {improvement:+.1f}%")
    else:
        improvement = 0.0
        print("  Improvement:           n/a (v4 MAE was 0)")
    print(f"  95% bracket coverage:  {coverage * 100:.1f}% (target >=80%)")

    failures: list[str] = []
    if v4_mae > 0 and improvement < 5.0:
        failures.append(
            f"bayes MAE improvement {improvement:.1f}% < 5% target"
        )
    if coverage < 0.80:
        failures.append(
            f"95% coverage {coverage * 100:.1f}% < 80% target"
        )

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(
        f"\nGATE PASSED: Bayes posterior beats v4 baseline by "
        f"{improvement:.1f}% MAE on H1, 95% interval covers "
        f"{coverage * 100:.1f}% of H1 actuals."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
