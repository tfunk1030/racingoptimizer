"""Day 6 acceptance gate -- lap-time-weighted samples on BMW Sebring (Mode 3).

PLAN.md Section 14.4 acceptance gate:
> For the BMW Sebring corpus (37 sessions), the fitter's per-parameter
> `baseline_setup` shifts toward the values used in the user's known-fast
> laps (top quartile by lap time) by >= 0.3 *step* units on at least 5
> fittable parameters.

In-process gate: pulls all BMW Sebring sessions from the catalog (held-
out automatically excluded), computes:
  - `unweighted` baseline = plain median (the pre-Day-6 baseline)
  - `weighted` baseline = lap-time-weighted median (the new baseline)
  - `top_quartile_median` = median across the fastest 25% of sessions

For each fittable parameter, count it as "shifted toward fast laps" if:
  abs(weighted - top_quartile_median) < abs(unweighted - top_quartile_median)
  AND the shift magnitude (abs(weighted - unweighted)) is >= 0.3 step.

Expected: >= 5 such parameters.

Canary form (broken model): if all session weights = 1.0 (equivalent to
disabling Day 6's weighting), `weighted` collapses to `unweighted` and
no parameter shifts. Tested in
`tests/physics/test_lap_weighted.py::test_mode_3_canary_uniform_weights_no_shift`.

Run: `uv run python scripts/day_06_gate.py`
"""
from __future__ import annotations

import json
import sys
from statistics import median

from racingoptimizer.constraints import load_constraints
from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.api import (
    catalog_path,
    laps as ingest_laps,
    resolve_corpus_root,
)
from racingoptimizer.physics.fitter import (
    _LAP_TIME_MIN_VALID_S,
    _LAP_WEIGHT_EPSILON_S,
    _weighted_median,
)
from racingoptimizer.physics.ontology import (
    fittable_parameters,
    setup_value,
)


CAR = "bmw"
TARGET_TRACK = "sebring_international"
SHIFT_THRESHOLD_STEPS = 0.3
MIN_PARAMS_SHIFTED = 5


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
            conn, car=CAR, track=TARGET_TRACK, valid_only=True,
            include_held_out=False,
        )

    n_sessions = len(sessions)
    if n_sessions < 8:
        print(
            f"FAIL: only {n_sessions} BMW Sebring sessions, need >=8 to "
            f"compute meaningful top-quartile baseline"
        )
        return 1

    print(f"BMW Sebring corpus (held-out excluded): {n_sessions} sessions")

    # Per-session best lap (filter sessions without valid laps).
    sessions_with_laps: list[tuple[str, float]] = []
    for sess in sessions:
        laps_df = ingest_laps(
            session_id=sess.session_id, valid_only=True, corpus_root=root,
        )
        if laps_df.height == 0:
            continue
        times = [
            float(t) for t in laps_df["lap_time_s"].to_list()
            if t is not None and t >= _LAP_TIME_MIN_VALID_S
        ]
        if not times:
            continue
        sessions_with_laps.append((sess.session_id, float(median(times))))

    n_with_laps = len(sessions_with_laps)
    if n_with_laps < 8:
        print(
            f"FAIL: only {n_with_laps} BMW Sebring sessions have valid "
            f"lap times; cannot compute meaningful gate"
        )
        return 1

    track_min = min(t for _, t in sessions_with_laps)
    weights = {
        sid: 1.0 / (best - track_min + _LAP_WEIGHT_EPSILON_S)
        for sid, best in sessions_with_laps
    }

    # Top quartile: sessions with the fastest 25% of best laps.
    sessions_sorted = sorted(sessions_with_laps, key=lambda x: x[1])
    quartile_n = max(2, n_with_laps // 4)
    top_quartile_sids = {sid for sid, _ in sessions_sorted[:quartile_n]}
    print(
        f"Top-quartile sessions ({quartile_n} of {n_with_laps}): "
        f"best laps {[f'{t:.2f}' for _, t in sessions_sorted[:quartile_n]]}"
    )

    # Build per-session setup snapshots: parameter -> value, extracted via
    # the per-car ontology's setup_value() resolver (handles nested JSON
    # paths the raw .get() does not).
    setup_snapshots: dict[str, dict[str, float]] = {}
    for sess in sessions:
        setup_dict = _decode_setup(sess.setup)
        snap: dict[str, float] = {}
        for param in fit_params:
            try:
                v = setup_value(CAR, param, setup_dict)
            except KeyError:
                v = None
            if v is not None:
                snap[param] = float(v)
        setup_snapshots[sess.session_id] = snap

    shifted_correctly: list[tuple[str, float, float, float, float]] = []
    shifted_wrong_direction: list[tuple[str, float, float, float, float]] = []
    flat: list[tuple[str, float, float, float]] = []
    no_shift_threshold: list[tuple[str, float, float, float, float]] = []
    for param in fit_params:
        spec_step = None
        # Pull step from ontology via fittable_parameters' fall-through.
        # We'll just normalise by 1.0 if step is unknown.
        # `constraints` carries bounds; ontology carries step.
        from racingoptimizer.physics.ontology import ontology_for
        onto = ontology_for(CAR)
        spec = onto.get(param)
        spec_step = float(spec.step) if spec and spec.step else 1.0

        # Per-session observed values + weights.
        all_pairs = [
            (setup_snapshots.get(sid, {}).get(param), weights.get(sid, 1.0))
            for sid, _ in sessions_with_laps
        ]
        all_pairs = [(v, w) for v, w in all_pairs if v is not None]
        if len(all_pairs) < 4:
            continue
        values = [float(v) for v, _ in all_pairs]
        ws = [float(w) for _, w in all_pairs]
        unweighted = float(median(values))
        weighted = _weighted_median(values, ws)
        # Top-quartile median (over the fastest 25% of sessions).
        tq_values = [
            setup_snapshots.get(sid, {}).get(param)
            for sid, _ in sessions_sorted[:quartile_n]
        ]
        tq_values = [float(v) for v in tq_values if v is not None]
        if len(tq_values) < 2:
            continue
        tq_median = float(median(tq_values))
        # Did weighted shift TOWARD top-quartile median compared to unweighted?
        unw_err = abs(unweighted - tq_median)
        w_err = abs(weighted - tq_median)
        shift_magnitude = abs(weighted - unweighted)
        shift_in_steps = shift_magnitude / spec_step
        if w_err < unw_err and shift_in_steps >= SHIFT_THRESHOLD_STEPS:
            shifted_correctly.append(
                (param, unweighted, weighted, tq_median, shift_in_steps)
            )
        elif w_err > unw_err and shift_magnitude > 0:
            shifted_wrong_direction.append(
                (param, unweighted, weighted, tq_median, shift_in_steps)
            )
        elif shift_magnitude == 0:
            flat.append((param, unweighted, weighted, tq_median))
        else:
            no_shift_threshold.append(
                (param, unweighted, weighted, tq_median, shift_in_steps)
            )

    print(
        f"\nSHIFTED TOWARD FAST-LAP (>= {SHIFT_THRESHOLD_STEPS} step):"
    )
    print(
        f"  {'parameter':<35} {'unweighted':>11} {'weighted':>11} "
        f"{'tq_median':>11} {'shift_steps':>11}"
    )
    for (param, unw, w, tq, shift) in shifted_correctly:
        print(
            f"  {param:<35} {unw:>11.3f} {w:>11.3f} {tq:>11.3f} {shift:>11.2f}"
        )

    if shifted_wrong_direction:
        print(
            f"\nSHIFTED AWAY FROM FAST-LAP "
            f"(weighted is FURTHER from tq median than unweighted):"
        )
        for (param, unw, w, tq, shift) in shifted_wrong_direction[:8]:
            print(
                f"  {param:<35} {unw:>11.3f} {w:>11.3f} {tq:>11.3f} {shift:>11.2f}"
            )
        print(f"  ... ({len(shifted_wrong_direction)} total)")

    if no_shift_threshold:
        print(
            f"\nSHIFTED CORRECTLY BUT BELOW {SHIFT_THRESHOLD_STEPS} STEP "
            f"THRESHOLD (informational):"
        )
        for (param, unw, w, tq, shift) in no_shift_threshold[:8]:
            print(
                f"  {param:<35} {unw:>11.3f} {w:>11.3f} {tq:>11.3f} {shift:>11.2f}"
            )
        print(f"  ... ({len(no_shift_threshold)} total)")

    print(f"\n  {len(shifted_correctly)} parameters shifted toward fast-lap (>= threshold)")
    print(f"  {len(shifted_wrong_direction)} parameters shifted AWAY from fast-lap")
    print(f"  {len(no_shift_threshold)} parameters shifted correctly but below threshold")
    print(f"  {len(flat)} parameters had no shift (held constant in corpus)")

    if len(shifted_correctly) < MIN_PARAMS_SHIFTED:
        print(
            f"\nGATE FAILED: only {len(shifted_correctly)} parameters shifted "
            f">= {SHIFT_THRESHOLD_STEPS} steps toward top-quartile median; "
            f"need >= {MIN_PARAMS_SHIFTED}"
        )
        return 1
    print(
        f"\nGATE PASSED: {len(shifted_correctly)} parameters shifted "
        f">= {SHIFT_THRESHOLD_STEPS} steps toward fast-lap setups (target "
        f">= {MIN_PARAMS_SHIFTED})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
