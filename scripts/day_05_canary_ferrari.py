"""Day 5 canary -- Ferrari Hockenheim held-out (H3) cross-track confounding.

PLAN.md Section 14.3 designates H3 (Ferrari Hockenheim) as the canary
day held-out: Ferrari Hockenheim is the documented MAJORITY track that
in CLAUDE.md lines 112-118 is said to drag Spa recommendations.

The asymmetric Mode 1 hypothesis: bayes retrofit should shrink LESS
when applied to a high-sample-count track (Hockenheim) and MORE when
applied to a low-sample-count track (Spa, where Ferrari has no data
at all so cross-car borrow kicks in via TrackModel; the retrofit
itself only sees Ferrari Hockenheim/Algarve).

This canary asks: with H3 held out, does the bayes retrofit's posterior
mean for Ferrari Hockenheim match H3's actual setup better than a
naive cross-track grand mean?

If H3 represents a typical converged Hockenheim setup (high-sample-
count, stable corpus median), bayes wins. If H3 is itself an outlier
(unlikely given Hockenheim is the dominant Ferrari corpus), bayes
might lose.

Run: `uv run python scripts/day_05_canary_ferrari.py`
"""
from __future__ import annotations

import json
import sys
from statistics import mean

from racingoptimizer.constraints import load_constraints
from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.api import catalog_path, resolve_corpus_root
from racingoptimizer.physics.bayes_retrofit import fit_all_parameters
from racingoptimizer.physics.ontology import (
    fittable_parameters,
    setup_value,
)


CAR = "ferrari"
HELD_OUT_TRACK = "hockenheim_gp"
HELD_OUT_SESSION_ID = "fc96805e3b1a27cc"  # H3


def _decode_setup(blob):
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

    with cat.open_catalog(catalog_path(root)) as conn:
        sessions = cat.query_sessions(
            conn, car=CAR, valid_only=True, include_held_out=False,
        )
        h3 = cat.get_session(conn, HELD_OUT_SESSION_ID)

    if h3 is None or h3.held_out != 1:
        print(f"FAIL: held-out session {HELD_OUT_SESSION_ID} not properly flagged")
        return 2

    n_prod = len(sessions)
    n_hock_prod = sum(1 for s in sessions if s.track == HELD_OUT_TRACK)
    print(
        f"Ferrari production corpus (H3 excluded): {n_prod} sessions; "
        f"{n_hock_prod} on {HELD_OUT_TRACK}"
    )

    per_track = {}
    for sess in sessions:
        if sess.session_id == HELD_OUT_SESSION_ID:
            print("FAIL: H3 leaked into production query!")
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

    by_param_track = {
        track: dict(params) for track, params in per_track.items()
    }
    posteriors = fit_all_parameters(by_param_track)
    h3_setup = _decode_setup(h3.setup)

    rows = []
    for param in fit_params:
        if (param, HELD_OUT_TRACK) not in posteriors:
            continue
        post = posteriors[(param, HELD_OUT_TRACK)]
        try:
            actual = setup_value(CAR, param, h3_setup)
        except KeyError:
            continue
        if actual is None:
            continue
        actual = float(actual)
        all_values = []
        for track_params in per_track.values():
            all_values.extend(track_params.get(param, []))
        if not all_values:
            continue
        v4_pred = mean(all_values)
        bayes_pred = post.mean
        bayes_err = abs(bayes_pred - actual)
        v4_err = abs(v4_pred - actual)
        pred_std = post.predictive_std or post.std
        lo = post.mean - 1.96 * pred_std
        hi = post.mean + 1.96 * pred_std
        covered = lo <= actual <= hi
        rows.append((param, actual, bayes_pred, v4_pred, bayes_err, v4_err, covered))

    if not rows:
        print("FAIL: no parameters with both Hockenheim posterior + H3 actual")
        return 1

    print(
        f"\n{'parameter':<35} {'actual':>10} {'bayes':>10} {'v4':>10} "
        f"{'bayes_err':>10} {'v4_err':>10} {'95%':>5}"
    )
    for r in rows:
        print(
            f"{r[0]:<35} {r[1]:>10.3f} {r[2]:>10.3f} {r[3]:>10.3f} "
            f"{r[4]:>10.3f} {r[5]:>10.3f} {'YES' if r[6] else 'NO':>5}"
        )

    bayes_mae = mean(r[4] for r in rows)
    v4_mae = mean(r[5] for r in rows)
    coverage = sum(1 for r in rows if r[6]) / len(rows)

    print("\n" + "-" * 100)
    print(f"  N parameters compared: {len(rows)}")
    print(f"  v4 baseline MAE:       {v4_mae:.4f}")
    print(f"  Bayes posterior MAE:   {bayes_mae:.4f}")
    if v4_mae > 0:
        improvement = (v4_mae - bayes_mae) / v4_mae * 100
        print(f"  Improvement:           {improvement:+.1f}%")
    print(f"  95% bracket coverage:  {coverage * 100:.1f}%")

    # Per-parameter win/loss tally.
    bayes_wins = sum(1 for r in rows if r[4] < r[5])
    v4_wins = sum(1 for r in rows if r[5] < r[4])
    ties = len(rows) - bayes_wins - v4_wins
    print(
        f"  Per-parameter tally:   bayes wins {bayes_wins}, "
        f"v4 wins {v4_wins}, ties {ties}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
