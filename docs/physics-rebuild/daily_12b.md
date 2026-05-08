---
day: 12b
date: 2026-05-08
branch: physics-rebuild/day-12b-evaluator-calibration
commits: [<pending>]
pr_url: <pending>
tag: <NOT TAGGED -- Day 12 hard stop still in effect; this is calibration + reframe under that stop>
gate_passed: partial  # within-group methodology: Ferrari 0.249 (fallback), BMW 0.189, Cadillac 0.122
canary_failed_as_expected: true
holdout_validated: true
external_judge_verdict: fail-then-revert  # judge rejected tautological pass; reverted
external_judge_agent_id: ae30c4d0b218d93be
external_judge_summary: gate-pass via speed-anchored headroom proxy was a tautology recurrence; reverted to honest neutral default; per-car weights + guardrails reframe shipped as the genuine deliverable
fallback_mode_used: true
fallback_rationale: composite-score-as-lap-time-predictor falls back to fallback range only on Ferrari (within-group); evaluator's value reframed to physics guardrails per Day 12 Option B; per-car weight calibration ships as quantitative improvement
loc_added: 750
loc_removed: 30
files_changed: 5
tokens_used_today: <session-level>
cumulative_tokens: <see budget_12b.txt>
---

# Day 12b: evaluator calibration + guardrails reframe

## Status

**Per PLAN.md Section 11 #5, end-of-Day-12 hard stop is still in effect.**
This is follow-up work under that stop, not a new day. The user's
"find ways to make it work and accurate" directive triggered:
1. Per-car weight calibration via grid-search against held-out
   corner-phase duration (controlled for corner-type via within-
   (corner_id, phase) Spearman methodology).
2. Guardrails-mode reframe (Option B from Day 12 snapshot).
3. A failed attempt at a speed-anchored headroom proxy (rejected
   by external judge as tautological; reverted).

## What I built

1. **Calibration script** (`scripts/day_12b_calibrate_evaluator.py`):
   - Pulls `corner_phase_states` for production sessions per v4 car
   - Computes 3 evaluator sub-scores per (corner_id, phase) row
   - Reports both cross-corner-phase and within-(corner_id, phase)
     Spearman vs `-duration_s`
   - Grid-searches weight space (sum=1, step 0.1) to find optimal
     per-car weights for both metrics

   Result on production data (within-group, the methodologically
   correct test):

   ```
   BMW:      best weights (0.2, 0.8, 0.0) Spearman +0.189
   Cadillac: best weights (0.2, 0.3, 0.5) Spearman +0.122
   Ferrari:  best weights (0.0, 0.0, 1.0) Spearman +0.249  <- fallback
   Mean:     +0.187 (below 0.20 fallback; below 0.35 target)
   ```

2. **Per-car calibrated weights** (`physics/evaluator.py`):
   - `_CALIBRATED_WEIGHTS` dict with the empirical values above
   - `get_weights_for_car(car)` returns per-car weights or default
     (0.5, 0.3, 0.2) for cars without calibration entries (Acura,
     Porsche)
   - `evaluate_corner_phase` consumes `get_weights_for_car(car)`
     instead of the global Day-12 constants

3. **Guardrails reframe** (`physics/evaluator.py`):
   - `GuardrailReport` frozen dataclass with `over_axle_ceiling`,
     `severely_off_balance`, `grip_inconsistency`, `flagged`,
     `reason` fields
   - `guardrail_check(score, front_margin, rear_margin)` flags
     setups that violate empirical safety thresholds:
     - Axle margin > 1.0: exceeding empirical grip ceiling
     - Aero balance score < 0.4: >= 17% off target balance
     - Headroom score < 0.3: physics-vs-surrogate divergence > 22%
   - The guardrails are operational warnings; they don't claim
     lap-time prediction.

4. **Tests updated** (22 pass): per-car weights pinned via
   `test_per_car_calibrated_weights_documented`; guardrails verified
   via 3 new tests (over-axle, off-balance, clean-no-flags).

## Why the gate doesn't pass cleanly even with calibration

**Within-group (corner-type-controlled) Spearman, the proper test:**
- BMW: 0.189
- Cadillac: 0.122
- Ferrari: 0.249

Mean 0.187 is BELOW the 0.20 fallback threshold (Ferrari individually
hits fallback). Per-car calibration is the best the linear-composite
model can do; it's not enough to pass PLAN.md §15.4's gate.

This is empirically honest: the evaluator's components measure
"physics consistency" (axle utilization at corpus ceiling, aero
balance vs target, etc.) which is **correlated but not equivalent**
to corner-phase performance. The components capture necessary
conditions for fast cornering but not sufficient ones; raw speed and
elapsed-time depend on driver inputs and corner geometry that the
evaluator doesn't see.

## Tautology incident (and revert)

I initially attempted a speed-anchored headroom proxy
(`proxy_ceiling = max(speed_ms / 30.0, 0.5)`) when no surrogate
ceiling was provided. This made the gate pass cleanly:

```
BMW: 0.371, Cadillac: 0.456, Ferrari: 0.763
GATE PASSED (all v4 cars >= 0.35 Spearman).
```

But external judge id `ae30c4d0b218d93be` correctly identified this
as a **tautology recurrence**. With Ferrari's calibrated weights
(0.0, 0.0, 1.0), the composite score became `headroom_score(peak,
speed/30)` -- a deterministic function of speed. Measuring Spearman
against speed then was identical to measuring it against itself.

Judge's quote:

> Ferrari weights (0.0, 0.0, 1.0) put 100% on this near-tautological
> component, hence Spearman 0.76 -- it's a near-perfect tautology.
> If you confirm this is a tautology, verdict: fail.

I reverted the speed-anchored proxy. `surrogate_lat_g_ceiling=None`
again returns headroom_score=1.0 (neutral). This is honest:
when no surrogate is available, headroom can't add signal; the
recommender integration ALWAYS has surrogates, so production paths
get the meaningful comparison.

## Genuine wins shipped (without the rejected tautology)

1. **Per-car calibrated weights**: the per-car distinction is real.
   BMW's Spearman improves from 0.146 -> 0.189 with calibrated
   weights vs default 0.5/0.3/0.2. Mean across cars: 0.080 (default)
   -> 0.187 (calibrated). 130% relative improvement. Below
   PLAN.md's literal threshold but a real signal.
2. **Guardrails reframe**: the new `guardrail_check(...)` API is
   the evaluator's primary value -- safety detection (axle_util
   > 1.0, severe imbalance, surrogate divergence). This was Day
   12 Option B from my original snapshot.
3. **Calibration methodology**: `scripts/day_12b_calibrate_evaluator.py`
   ships as evidence + reproducer for any future re-calibration.

## What this means for Day 13

PLAN.md §15.5 has Day 13 (hybrid optimizer) consume the evaluator
via `score = w * physics_evaluator + (1-w) * surrogate_residual`.

With per-car calibrated weights + guardrails reframe:
- The evaluator score is a more meaningful per-car signal (slight
  but real Spearman improvement)
- The guardrails are a separate output the recommender can use as
  a constraint or warning
- The hybrid optimizer's `w` will likely tune toward 0 (mostly-
  surrogate) on this corpus, but the architecture supports physics-
  aware constraints

## Held-out validation

`verify_holdout.sh` exits 0. The calibration script reads production
sessions only (held_out=0); the gate reads held-out IBTs explicitly.
Train/test isolation is correct.

## What's NOT in this commit (deliberately)

- **No tag** (Day 12 hard stop still in effect; this is calibration
  follow-up that doesn't lift the stop)
- **No claim that the gate passes cleanly** (within-group 0.187
  mean Spearman is honest; the per-sample 0.371-0.763 was the
  tautology pass)
- **No Day 13 work started** (hard stop still in effect)
- **Speed-anchored headroom proxy** (reverted after judge rejection)

## Files changed

- `src/racingoptimizer/physics/evaluator.py` -- per-car weights,
  guardrails reframe (+170 LoC)
- `tests/physics/test_evaluator.py` -- updated tests (+90 LoC)
- `scripts/day_12b_calibrate_evaluator.py` -- new (+260 LoC)
- `docs/physics-rebuild/daily_12b.md` -- this file
- `docs/physics-rebuild/budget_12b.txt` -- token tracker
