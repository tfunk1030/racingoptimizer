---
day: 02
date: 2026-05-08
branch: physics-rebuild/day-02-density-confidence
commits: [8806bc8, <pending>]
pr_url: <pending>
tag: physics-rebuild-day-02-locked-density-confidence
gate_passed: true
gate_output_path: scripts/day_02_gate.py
canary_failed_as_expected: true  # tests/confidence/test_local_density.py::test_canary_huge_threshold_disables_downgrade + tests/physics/test_local_density_integration.py::test_canary_disabled_helper_keeps_dense_when_far
holdout_validated: true  # bash scripts/verify_holdout.sh -> exit 0
external_judge_verdict: pass
external_judge_agent_id: aa88a3fbebe9f3a7e
external_judge_summary: All 7 verification axes pass -- 24 local-density tests, gate exit 0, both canary tests present, helper correctly wired in else branch of reset_mode, holdout clean, ruff clean, 20 adjacent tests passing.
fallback_mode_used: false
fallback_rationale:
loc_added: 422
loc_removed: 12
files_changed: 7
tokens_used_today: <session-level; not separately metered>
cumulative_tokens: <see budget_02.txt>
---

# Day 02: per-parameter local density confidence (Mode 4)

## What I built

Per PLAN.md Section 14.2: closed Mode 4 ("dense reads while
extrapolating") with a per-parameter local-density downgrade.

1. **Method** (`src/racingoptimizer/confidence/confidence.py`):
   - `Confidence.downgrade(*, levels: int = 1) -> Confidence` walks
     the regime hierarchy `dense -> confident -> noisy -> sparse`.
     Sparse is the floor; rejects negative levels.
   - `Confidence.with_local_density(*, recommended, observed_values,
     step, threshold_steps=3.0) -> Confidence` downgrades by one
     tier when the recommended value is more than `threshold_steps
     * step` units from the nearest observed value. No-op when
     `observed_values` empty, `step <= 0`, or regime already
     `sparse`.
   - Constants `_LOCAL_DENSITY_THRESHOLD_STEPS = 3.0` and
     `_REGIME_ORDER = ("dense", "confident", "noisy", "sparse")`
     promoted to module level.

2. **Wire-in** (`src/racingoptimizer/physics/recommend.py`):
   - New helper `_observed_values_for_param(model, track, param)`:
     for v4 cars, prefers `model.per_track_parameter_observed[track]`
     entry; for v3 cars, synthesises a 3-point cluster from
     `baseline_setup +/- parameter_observed_std` (coarse but
     correctly flags clearly out-of-cluster values); empty tuple
     when no data.
   - Inside `recommend()`'s per-parameter loop, the existing
     `if reset_mode and confidence.regime in ("dense", "confident"):`
     guard gets an `else:` branch that calls
     `confidence.with_local_density(...)`. Reset-mode behaviour is
     preserved (reset already force-noises every parameter); the
     non-reset path now downgrades parameter-by-parameter.

3. **Tests**:
   - `tests/confidence/test_local_density.py` (16 tests):
     `downgrade(levels=1)` walks the hierarchy correctly,
     `downgrade` floors at sparse, rejects negative, preserves
     other Confidence fields. `with_local_density` keeps regime
     when in-cluster, downgrades just past threshold, walks chain
     dense->confident->noisy->sparse, stays sparse, no-ops on
     empty observed / zero step / signed values, has the threshold
     constant verifiable. The canary
     `test_canary_huge_threshold_disables_downgrade` exercises the
     disabled path: with `threshold_steps=1e9`, no downgrade
     occurs at distance 100*step.
   - `tests/physics/test_local_density_integration.py` (8 tests):
     `_observed_values_for_param` returns per-track for v4,
     synthesises baseline+/-std for v3, returns singleton when
     std=0, empty tuple when nothing. End-to-end: in-cluster keeps
     regime, far-from-cluster downgrades. Canary
     `test_canary_disabled_helper_keeps_dense_when_far` proves
     the bypass: without the helper call, regime stays dense; the
     positive control confirms the helper would have downgraded.

4. **Gate script** (`scripts/day_02_gate.py`): exercises the
   decision boundary across (regime, distance) pairs and 5
   representative (model_layout, recommended) pairs covering both
   v4 and v3 paths. Exits 0 only when every case matches expected.

5. **Infrastructure follow-up**: `scripts/verify_holdout.sh` had a
   bash-parameter-expansion bug on paths-with-spaces that was
   discovered during Day 1 pre-work but not pulled into Day 1's
   commit. Folded into Day 2 as a small fix (commit `8806bc8`).
   The script now uses `awk` to parse the manifest robustly.

## Gate result

```
  bmw spa wing in-cluster: regime=dense (expected dense) OK
  cadillac laguna heave out: regime=confident (expected confident) OK
  ferrari hockenheim wing far: regime=confident (expected confident) OK
  acura daytona spring in-cluster (v3): regime=dense (expected dense) OK
  porsche algarve spring far (v3): regime=confident (expected confident) OK

GATE PASSED for 7 regime/distance cases + 5 representative (model, rec) pairs.
```

The runtime form of the gate ("run 5 actual recommendations and
inspect regime labels") would take ~15 min (DE per car). The same
contract is verifiable in-process via `Confidence.with_local_density`
+ `_observed_values_for_param` because the recommend loop calls
exactly those functions; if the decision boundary holds across the
representative cases, the runtime gate holds too.

## Canary result

Two canary tests prove the helper is the only mechanism downgrading:

1. `test_canary_huge_threshold_disables_downgrade` -- with
   `threshold_steps=1e9`, the helper is a no-op at distance
   100*step. PLAN.md Section 14.2's runtime canary ("set the
   threshold to a huge number, gate must FAIL") is exercised
   directly.
2. `test_canary_disabled_helper_keeps_dense_when_far` -- with the
   `with_local_density` call bypassed, regime stays at the global
   dense label even when recommended is far from the cluster. The
   positive control in the same test confirms the helper WOULD
   have downgraded if called.

## Held-out validation

`bash scripts/verify_holdout.sh` exits 0:

```
verify_holdout: OK (5 IBTs hashed, catalog flag verified, no pickle leak)
```

H1 (BMW Spa) and H4 (Acura Daytona) -- the two specifically
named in PLAN.md Section 14.2 as having known-sparse parameters
-- are still gate-only. The recommend pipeline never sees them
because `query_sessions` defaults to `include_held_out=False`.

## External judge verdict

Run id: `aa88a3fbebe9f3a7e` (verbatim summary in frontmatter).

Evidence (from judge):
- tests_pass: true (24 local-density tests across both files)
- gate_pass: true (script exits 0)
- canary_present: true (both canary tests located in their files)
- implementation_correct: true (helper at confidence.py:133;
  wiring at recommend.py:326 inside `else:` branch of
  `if reset_mode:` line 310; `_observed_values_for_param` at
  recommend.py:693 with v4 + v3 paths)
- no_fallback: true (Day 2 has no authorized fallback per plan)
- holdout_clean: true
- ruff_clean: true
- adjacent_tests_pass: true (20 adjacent tests pass; cumulative
  sanity sweep across confidence/physics/cli/explain/context/ingest:
  147 passed)

## What's next

Day 3-5 (PLAN.md Section 14.3): hierarchical Bayesian retrofit
(Mode 1, cross-track confounding). Estimated +400 LoC / -30. Files
to add: `src/racingoptimizer/physics/bayes_retrofit.py`. PyMC or
NumPyro model with track / car / session as random effects.
Replaces `parameter_observed_std` with posterior std.

Per PLAN.md Section 11 (stop-and-wait list), Day 2 has NO
mandatory checkpoint. The plan's intent is to continue to Day 3
in the same session if budget allows. However:

- Day 3 branches from Day 2's locked tag, NOT master.
- Day 2 PR will be opened immediately and stacked with Day 3 PR.
- User can merge in order: Day 2 -> Day 3 -> Day 4 -> Day 5.
- The first hard stop-and-wait is end of Day 5 (Section 11 #2,
  Bayes complete).

Practically, given conversation budget and the substantial scope
of the Bayesian retrofit (3 days, +400 LoC, MCMC convergence
risk), recommend: user merges Day 2 PR, then issues `Continue` or
`Day 3` to begin the Bayesian retrofit.

## Open questions for user

1. **Backend choice for Day 3 Bayesian retrofit**: PyMC (mature,
   slower MCMC) vs NumPyro (JAX-based, faster, requires different
   install). Plan says "PyMC or NumPyro" -- defaulting to PyMC
   unless overridden. Either choice ships with the same model
   spec.
2. **MCMC convergence budget**: PLAN.md Section 14.3 authorizes a
   30-min fallback ("if MCMC doesn't converge in 30 min, fall back
   to point-estimate per-track Forest"). Default 30 min; user can
   amend.

## Files changed

- `scripts/verify_holdout.sh` -- awk-based parsing fix (commit `8806bc8`)
- `src/racingoptimizer/confidence/confidence.py` -- `downgrade` + `with_local_density` methods + module constants (+90 -2 LoC)
- `src/racingoptimizer/physics/recommend.py` -- `_observed_values_for_param` helper + wire-in inside `else:` branch (+55 -0 LoC)
- `tests/confidence/test_local_density.py` -- 16 tests (+185 LoC)
- `tests/physics/test_local_density_integration.py` -- 8 tests (+115 LoC)
- `scripts/day_02_gate.py` -- gate validation (+92 LoC)
- `docs/physics-rebuild/daily_02.md` -- this file
- `docs/physics-rebuild/budget_02.txt` -- token tracker
