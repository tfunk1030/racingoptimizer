---
day: 07
date: 2026-05-08
branch: physics-rebuild/day-07-week1-cumulative-gate
commits: [<pending>]
pr_url: <pending>
tag: <NOT TAGGED -- gate failed; PLAN.md Section 11 #3 hard stop>
gate_passed: false
gate_output_path: scripts/day_07_gate.py
canary_failed_as_expected: see prose
holdout_validated: true  # bash scripts/verify_holdout.sh -> exit 0
external_judge_verdict: <not run -- gate failure unambiguous; user adjudication structural>
external_judge_agent_id:
external_judge_summary:
fallback_mode_used: false
fallback_rationale: gate failure is empirical, not a fallback path invocation
loc_added: 580
loc_removed: 0
files_changed: 4
tokens_used_today: <session-level>
cumulative_tokens: <see budget_07.txt>
---

# Day 07: Week-1 cumulative gate -- HARD STOP for user adjudication

## Status

**GATE FAILED** on both original and substitute criteria. Per PLAN.md
Section 11 #3, end of Day 7 is the **hard stop-and-wait at the
Week-1 -> Week-2 transition** regardless of pass/fail. Week 2
(scoped physics) does NOT begin until the user adjudicates.

## Gate result

| # | Criterion | Result |
|---|---|---|
| 1 | Mode 2 closed -- tyre pressure floor on all 5 cars | **PASS** |
| 2 | Mode 4 closed -- per-parameter density downgrade | **PASS** |
| 3 | Mode 1 closed (original) -- BMW H1 MAE improves >=5% | **FAIL** -7.3% |
| 3a | Mode 1 closed (substitute coverage) -- 95% coverage >=80% | PASS (85.1%/100%) |
| 3b | Mode 1 closed (substitute win-rate) -- Mode-1-sensitive >=60% | **FAIL** 51.7% |
| 4 | Mode 3 closed -- BMW baseline shifts >=5 params | **PASS** (11 shifted) |
| 5 | No regressions -- representative test slice | **PASS** |
| 6 | Composite metric beat (original) -- aggregate MAE >=7% | **FAIL** -7.3% |
| 6 | Composite metric beat (substitute) -- 0.5*cov + 0.5*win >=7% | **FAIL** -1.6% |

**3 modes cleanly closed** (Mode 2, Mode 3, Mode 4). **Mode 1
empirically not closed** on this corpus by either the original
criterion or the proposed substitutes.

## What I built

1. **DEVIATION document** (`docs/physics-rebuild/DEVIATION_day_07_gate_amendment.md`):
   proposed amending PLAN.md Section 14.5 criterion #3 to replace
   the aggregate-MAE form with two substitute forms (95% coverage
   on H1+H3 plus Mode-1-sensitive win rate). Proposal documented
   with rationale and risk analysis; signed by the agent for user
   adjudication via PR review.

2. **Cumulative gate script** (`scripts/day_07_gate.py`): runs all
   six original criteria against the actual corpus, plus both
   substitute forms of #3 and #6. Reports each binary pass/fail
   plus the underlying numerics. Exit codes:
   - 0 = original passes (no deviation needed)
   - 1 = substitutes rescue the gate
   - 2 = non-Mode-1 criterion failed (real Week-1 regression)
   - 3 = both original AND substitutes fail (this turn's outcome)

3. **No new production code.** Day 7 is a gate-only day; the
   evidence determines the user-decision path.

## Mode-1-sensitive analysis

Day 7's substitute #3b targeted "Mode-1-sensitive parameters --
defined as the subset where per-track std differs by >= 1.5x
across tracks." 29 parameters out of 47 met this threshold on the
BMW corpus. Bayes wins on 15 of 29 (51.7%) -- essentially a tie.

This is the key empirical finding: even on the parameters where
cross-track confounding has the largest possible footprint, the
Bayes posterior provides no measurable predictive advantage over
the cross-track grand mean for predicting H1's setup. The
hierarchical math is correct (Day 3 canonical case) and the
posterior is calibrated correctly (95% coverage 85.1% / 100%),
but the held-out IBT does not behave like a converged-fast
session, which is the workload Bayes was designed to help with.

Day 5's per-parameter detail showed bayes WINS specifically on:
- `rear_wing_angle_deg` (1.24 vs 1.53 v4 error)
- `tyre_cold_pressure_kpa` (0.6 vs 0.625 v4 error)
- `damper_lsc_*` (4 corners, each ~6 vs ~7.8)
- `pushrod_length_offset_rear_mm` (4.96 vs 7.29)

These are the parameters where physics intuition predicts Mode 1
should bite. But the Mode-1-sensitive selector picked up too many
false positives (parameters where per-track std differs by chance,
not by Mode 1 mechanism). Tightening to "wing + tyre + damper_lsc
only" would give bayes ~70-80% wins, but that's circular: hand-
picking the parameters where bayes is known to win is not a fair
gate criterion.

## Interpretation

Three modes cleanly closed:
- **Mode 2** (tyre pressure floor, Day 1, PR #71): all 5 cars pin
  to the 152 kPa floor; recommendations no longer drift to 160+.
- **Mode 3** (lap-time-weighted samples, Day 6, PR #76): 11 BMW
  Sebring parameters shift toward fast-lap-quartile median; cleaner
  recommendations on stints where the user iterated.
- **Mode 4** (per-parameter density confidence, Day 2, PR #72):
  regime labels downgrade when recommendations land outside corpus
  density; honest "this is extrapolation" warnings.

One mode mathematically correct but empirically inconclusive on
this corpus:
- **Mode 1** (Bayes hierarchical retrofit, Days 3-4-5, PRs #73, #74,
  #75): math passed canonical Mode 1 case (100% improvement on
  Hockenheim 24*17 vs Spa 6*14.5 synthetic), 95% coverage passed
  (85.1% / 100%), per-parameter wins real on physics-relevant
  subset. But aggregate MAE on H1/H3 is 7-13% WORSE than v4
  baseline because H1/H3 are themselves outlier setups within
  their tracks (not converged-fast targets the model should
  predict).

One mode not addressed in Week 1 (by design):
- **Mode 5** (new car/track day-zero) is Week 2 scoped-physics
  territory.

## What this means for Week 2

**Mode 5 is the only mode Week 2 was designed to address.** It is
**not** dependent on Mode 1 closure. The scoped-physics work
(diagnostic state, damper refit, axle-grip-margin, hybrid
optimizer) gives day-zero capability for new cars and tracks,
which is independent of whether the Bayes wire-in helps predict
held-out setups.

The Bayes wire-in (Day 4) can be:
- **Kept** (current state in master): the math is correct, coverage
  is calibrated, and on parameters where Mode 1 actually bites
  bayes wins. The aggregate-MAE failure is a corpus-property
  finding. Week 2 proceeds; Bayes is part of the production stack
  but doesn't dominate.
- **Rolled back** (revert PR #74 + the Day 4 wire-in commit): if the
  empirical wash means the user shouldn't trust Bayes posteriors
  for production recommendations, the wire-in becomes
  "experimental" -- math stays in `bayes_retrofit.py` but the
  PhysicsModel doesn't use it. This is conservative.
- **Constrained** (new option): wire-in only used for the
  parameters where bayes empirically wins (`rear_wing_angle_deg`,
  `tyre_cold_pressure_kpa`, `damper_lsc_*`). This is a
  hand-tuned middle ground; argued against in the deviation
  document but listed here for completeness.

## User decision needed

PLAN.md Section 11 #3 mandates a hard stop here. Day 8 (start of
Week 2) does not begin until you adjudicate.

**Three plausible paths**:

**(A) Accept partial closure and proceed to Week 2.** Modes 2, 3,
4 are unambiguously closed. Mode 1 is "math correct, empirics
inconclusive" -- ship as-is and let Week 2 add value through
Mode 5 closure. The Bayes wire-in stays in production. *My
recommendation.*

**(B) Roll back Day 4's Bayes wire-in and proceed to Week 2.**
Conservative. Day 3's math stays as a tested module; the
production PhysicsModel reverts to pre-Day-4 behaviour for
`bayes_posteriors`. Week 2 starts on a smaller surface area. PR
#77 (this snapshot) would include the revert.

**(C) Halt entirely.** If the gate failure means we shouldn't
proceed at all, the agent halts and you decide future direction
manually. Days 8-14 not started.

The DEVIATION document (committed in this PR) is the procedurally-
correct paper trail; whichever path you choose, the deviation
document records the analysis.

## Held-out validation

`bash scripts/verify_holdout.sh` exits 0:
```
verify_holdout: OK (5 IBTs hashed, catalog flag verified, no pickle leak)
```

Day 7 read-only inspected H1, H3 via the gate. No held-out
modification.

## Files changed

- `scripts/day_07_gate.py` -- cumulative gate (+390 LoC)
- `docs/physics-rebuild/DEVIATION_day_07_gate_amendment.md` --
  amendment proposal for criterion #3 (+150 LoC)
- `docs/physics-rebuild/daily_07.md` -- this file
- `docs/physics-rebuild/budget_07.txt` -- token tracker

## NOT done (intentional)

- **No locked tag** (gate failed; PLAN.md Section 8.3 forbids).
- **No external-judge invocation** (gate failure is unambiguous;
  user's call is structural -- accept partial / roll back / halt).
- **No PLAN.md Section 14.5 modification** (PLAN.md immutable per
  Section 4.1 F9; the DEVIATION document is the procedurally-
  correct way to surface a plan amendment for user adjudication).
- **No Week 2 (Day 8+) work started** (HARD stop per Section 11 #3).
- **No revert of Day 4 PR #74** (that's option B; awaiting user
  decision).
