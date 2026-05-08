---
day: 12
date: 2026-05-08
branch: physics-rebuild/day-12-physics-evaluator
commits: [<pending>]
pr_url: <pending>
tag: <NOT TAGGED -- gate failed on speed-proxy; PLAN.md Section 11 #5 hard stop>
gate_passed: false
gate_output_path: scripts/day_12_gate.py
canary_failed_as_expected: true  # tests/physics/test_evaluator.py::test_canary_constant_inputs_yield_same_score
holdout_validated: true  # bash scripts/verify_holdout.sh -> exit 0
external_judge_verdict: <not run -- gate failure structural; user adjudication>
external_judge_agent_id:
external_judge_summary:
fallback_mode_used: false
fallback_rationale: see "Methodology finding" below
loc_added: 660
loc_removed: 0
files_changed: 4
tokens_used_today: <session-level>
cumulative_tokens: <see budget_12.txt>
---

# Day 12: per-corner-phase physics evaluator -- HARD STOP, methodology finding

## Status

**GATE PARTIAL.** Per PLAN.md Section 11 #5, end of Day 12 is the
hard stop-and-wait checkpoint regardless of pass/fail. Honest
report below; user adjudication required for Day 13.

| car | Spearman vs speed | Spearman vs |lat_g| (tautological) |
|---|---|---|
| BMW | 0.029 (FAIL) | 0.982 (PASS by construction) |
| Cadillac | 0.304 (FALLBACK, authorized at >=0.20) | 0.951 (PASS) |
| Ferrari | 0.141 (FAIL) | 0.868 (PASS) |

## Methodology finding (the meaningful result)

PLAN.md §15.4's gate is "Spearman vs lap-time-per-corner-phase >=0.35
on held-out laps." Day 12's evaluator score depends most heavily on
`axle_utilization_score` (weight 0.5), which derives from
`compute_axle_grip_ratios` -> `axle_force_split` -> `Fy = m * lat_g
* weight_share` -> ratio -> margin -> score. So the score is
ALGEBRAICALLY LINKED to |lat_g|.

Using |lat_g| as the empirical proxy yields Spearman 0.87-0.98 --
which is essentially **tautological**: I'd be correlating the score
with one of its own input quantities.

Using SPEED at corner-phase as a non-tautological proxy (speed is
NOT in the axle_util chain), the gate result drops to:
- BMW: 0.029 (essentially zero correlation)
- Cadillac: 0.304 (just above fallback)
- Ferrari: 0.141

This is the HONEST signal: the evaluator's score does not strongly
predict per-sample corner-phase speed on this corpus. Possible
reasons:

1. **The proxy is still imperfect.** Per-sample speed isn't
   "lap-time-per-corner-phase" -- a slow tight corner can be
   well-driven (high score) at low speed, while a fast sweeper can
   be sub-optimal (low score) at high speed. PLAN.md's specified
   metric was per-CORNER-PHASE-AGGREGATE elapsed time, not
   per-sample speed.
2. **The composite weights may be wrong for this corpus.** Axle-
   utilization at 50% weight makes the score sensitive to lat_g but
   not to overall corner-phase character.
3. **The evaluator's design rewards "near-grip-limit operation"
   rather than "fast corner-phase."** Those are correlated in
   theory; on real telemetry they diverge.

## What I built (regardless of gate outcome)

1. **Module** (`src/racingoptimizer/physics/evaluator.py`, +320 LoC):
   - `CornerPhaseScore` frozen dataclass with axle_utilization,
     aero_balance_score, grip_headroom_score, composite_score.
   - `axle_utilization_score(front_margin, rear_margin)`: ideal
     band [0.85, 1.0]; under-utilization linear penalty; over-
     limit tolerance 0.05; beyond tolerance penalty. Pair score =
     MIN of axles.
   - `aero_balance_score(balance_pct, weight_distribution)`: ideal
     ~ (1 - distribution_front) * 100; full score within 5%; zero
     beyond 25% deviation.
   - `grip_headroom_score(physics_peak, surrogate_ceiling)`:
     consistency check between physics and surrogate predictions.
   - `evaluate_corner_phase(...)`: composite scoring per (corner,
     phase).
   - `evaluate_lap(samples)`: bulk wrapper.

2. **Tests** (`tests/physics/test_evaluator.py`, 18 pass): each
   sub-score function in isolation, composite weighting, lap-bulk
   wrapper, broken-model canary (varied inputs produce varied
   scores).

3. **Gate script** (`scripts/day_12_gate.py`): per-car evaluator
   Spearman against held-out IBTs. Includes both proxies (speed +
   lat_g) so the methodology choice is transparent.

## What this means for Day 13

PLAN.md §15.5 has Day 13 (hybrid optimizer) consume the evaluator
output via `score = w * physics_evaluator + (1-w) * surrogate_residual`.
With the evaluator score not strongly predictive of per-sample speed,
two options:

**(A)** Day 13 proceeds anyway; the hybrid optimizer's `w` parameter
defaults to 0.6 but if physics-only doesn't help, `w` can be tuned
down. The empirical evidence (poor Spearman on this corpus) might
mean physics adds noise rather than signal; the hybrid would
naturally settle to mostly-surrogate behavior. The optimizer still
ships as infrastructure.

**(B)** Day 13 reframes the evaluator's purpose. Rather than "score
that should beat surrogate on lap-time prediction," position it as
"physics-guardrails that supplement the surrogate" (e.g. flag setups
where axle_util > 1.0 as risky). This drops the Spearman
requirement entirely.

**(C)** Halt physics-rebuild. Days 1-11 were strong wins; Day 12's
evaluator might not add measurable value. Ship what we have; revisit
the evaluator with a richer per-corner-phase aggregate gate later.

## Held-out validation

`verify_holdout.sh` exits 0. Day 12 reads held-out IBTs explicitly
in the gate (the only authorized read).

## What's next (or NOT next, pending user decision)

Day 13 (PLAN.md Section 15.5): hybrid optimizer `score = w *
physics_evaluator + (1-w) * surrogate_residual`. Estimated +200
LoC. Acceptance gate: "BMW Spa held-out, hybrid recommends within 1
click of validated fastest setup on >=5 of 8 high-impact parameters."

PLAN.md Section 11 #5: end of Day 12 is HARD STOP. User must
adjudicate before Day 13 begins.

## Files changed

- `src/racingoptimizer/physics/evaluator.py` -- new module (+320 LoC)
- `tests/physics/test_evaluator.py` -- 18 tests (+260 LoC)
- `scripts/day_12_gate.py` -- gate (+260 LoC)
- `docs/physics-rebuild/daily_12.md` -- this file
- `docs/physics-rebuild/budget_12.txt` -- token tracker

## NOT done (intentional)

- **No locked tag** (gate did not pass cleanly).
- **No external-judge invocation** (gate-failure pattern is
  methodology, not implementation).
- **No Day 13 work started** (HARD stop per Section 11 #5).
