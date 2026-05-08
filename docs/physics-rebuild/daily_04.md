---
day: 04
date: 2026-05-08
branch: physics-rebuild/day-04-bayes-wire-in
commits: [<pending>]
pr_url: <pending>
tag: physics-rebuild-day-04-locked-bayes-wire-in
gate_passed: true
gate_output_path: scripts/day_04_gate.py
canary_failed_as_expected: true  # tests/physics/test_bayes_wire_in.py::test_canary_layout_version_bump_invalidates_existing_caches
holdout_validated: true  # bash scripts/verify_holdout.sh -> exit 0
external_judge_verdict: pass
external_judge_agent_id: a5fd7088d096a7d6f
external_judge_summary: Day 4 wire-in is complete -- bayes_posteriors flows from fit_per_car -> PhysicsModel with a layout bump and pickle backward-compat; all required tests + gate + holdout + ruff are green.
fallback_mode_used: false
fallback_rationale:
loc_added: 175
loc_removed: 5
files_changed: 5
tokens_used_today: <session-level>
cumulative_tokens: <see budget_04.txt>
---

# Day 04: bayes_retrofit wire-in to fit_per_car (Mode 1, mid-component)

## What I built

Per PLAN.md Section 14.3 mid-component task: wired Day 3's
empirical-Bayes retrofit module into the per-car training pipeline so
production fits populate `PhysicsModel.bayes_posteriors`.

1. **PhysicsModel.bayes_posteriors field** (`physics/model.py`):
   - New field at the BOTTOM of the dataclass (the slots are append-
     only -- inserting in the middle would corrupt revives of legacy
     pickles).
   - Type: `dict[tuple[str, str], BayesPosterior]` keyed by `(parameter, track)`.
   - Default factory: `{}`.
   - `__setstate__` adds `slot_values.setdefault("bayes_posteriors", {})`
     so pre-Day-4 pickles revive cleanly with empty posteriors (the
     recommender's fallbacks cover that case).

2. **fit_per_car wire-in** (`physics/fitter.py`):
   - After `per_track_observed_frozen` is built, the function calls
     `bayes_retrofit.fit_all_parameters(per_track_observed_frozen)` and
     passes the result as `bayes_posteriors=...` to the PhysicsModel
     constructor.
   - The retrofit operates on the same per-track observed dict the
     trust radius already consumes; no new data ingest, no new disk
     reads.

3. **FITTERS_LAYOUT_VERSION bump** (`physics/fitters/__init__.py`,
   2 -> 3): the per-car cache key folds this version, so existing
   per-car pickles invalidate on next `optimize` invocation. Without
   the bump, cached models would silent-revive with
   `bayes_posteriors={}` (via the `__setstate__` default) and Mode 1
   closure would be a no-op until the user manually re-fitted. Version
   history added inline.

4. **Tests** (`tests/physics/test_bayes_wire_in.py`, 9 pass):
   - PhysicsModel exposes `bayes_posteriors` with default {}.
   - Constructor accepts the kwarg; round-trips correctly.
   - Frozen dataclass rejects field reassignment.
   - Pre-Day-4 pickle revive defaults the new field to {}.
   - `__setstate__` partial-state revive handles the missing slot.
   - `FITTERS_LAYOUT_VERSION >= 3` asserted explicitly.
   - End-to-end synthetic round-trip: per-track dict ->
     `fit_all_parameters` -> PhysicsModel -> retrieval; Mode 1 case
     (Hockenheim 24*17 vs Spa 6*14.5) confirms Spa posterior near 14.5.
   - **Canary**: `FITTERS_LAYOUT_VERSION != 2` asserted (the pre-Day-4
     value). If a future commit reverts the bump, the canary fires.
   - Empty bayes_posteriors dict is a legal model state.

5. **Gate script** (`scripts/day_04_gate.py`): exits 0 only when all
   five axes (field exposed, synthetic round-trip works, layout version
   bumped, pickle revive, constructor accepts kwarg) hold.

## Gate result

```
  PhysicsModel exposes bayes_posteriors field (default {}): OK
  Synthetic Mode 1 round-trip: Spa posterior 14.501 (empirical 14.5): OK
  FITTERS_LAYOUT_VERSION = 3 (>=3, invalidates pre-Day-4 caches): OK
  Pickle revive defaults bayes_posteriors to {}: OK
  Constructor stores bayes_posteriors (2 entries): OK

GATE PASSED: ...
```

The full held-out gate (BMW with H1 Spa held out, beat current v4 by
5% MAE on setup-readouts) requires a fresh per-car retrain that
invokes `fit_per_car` end-to-end. That's Day 5's task; Day 4 just
proves the data flow.

## Canary result

`test_canary_layout_version_bump_invalidates_existing_caches` asserts
`FITTERS_LAYOUT_VERSION != 2`. With the pre-Day-4 value, existing
per-car pickles would revive via `__setstate__` default with
`bayes_posteriors={}`, defeating Mode 1 closure on every cached model
silently. The bump forces a refit; the canary catches a future regression.

## Held-out validation

`verify_holdout.sh` exits 0:
```
verify_holdout: OK (5 IBTs hashed, catalog flag verified, no pickle leak)
```

The wire-in does not touch the held-out IBTs. The full held-out gate
arrives Day 5 once a per-car retrain runs.

## External judge verdict

Run id: `a5fd7088d096a7d6f`. All 11 evidence axes pass:
- tests_pass / gate_pass / fit_per_car_wired / model_field_with_compat /
  layout_version_bumped / canary_present / no_silent_fallback /
  holdout_clean / ruff_clean / adjacent_tests_pass / pickle_compat.

Sanity sweep: 133 tests pass across confidence/physics/cli/ingest/
explain/context.

## What's next

Day 5 (PLAN.md Section 14.3, end-of-component): run the BMW@Spa
held-out gate. Day 5's branch will be created from
`physics-rebuild-day-04-locked-bayes-wire-in` tag. The gate:
- For BMW with H1 (Spa held-out IBT) excluded from training, run
  `optimize bmw spa` end-to-end, capture the recommendation.
- Compute MAE between predicted setup-readouts (BMW Sebring-trained
  per-car model + posterior-aware recommend) and the H1 setup
  readouts.
- Compare against the same MAE under the pre-bayes baseline.
- Posterior 95% interval must cover >=80% of held-out setup readouts.

Day 5 is **a hard stop-and-wait** (PLAN.md Section 11 #2) regardless
of pass/fail.

## Open questions for user

None. Day 4 is purely the wire-in; the held-out evaluation is Day 5.

## Files changed

- `src/racingoptimizer/physics/model.py` -- `bayes_posteriors` field +
  `__setstate__` default + import (+25 -0 LoC)
- `src/racingoptimizer/physics/fitter.py` -- wire-in to `fit_per_car`
  (+13 -0 LoC)
- `src/racingoptimizer/physics/fitters/__init__.py` -- layout bump
  with version-history comment (+13 -5 LoC)
- `tests/physics/test_bayes_wire_in.py` -- 9 tests (+125 LoC)
- `scripts/day_04_gate.py` -- gate validation
- `docs/physics-rebuild/daily_04.md` -- this file
- `docs/physics-rebuild/budget_04.txt` -- token tracker
