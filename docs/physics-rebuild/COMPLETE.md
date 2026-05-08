# Physics Rebuild — completion summary (Day 14)

Date: 2026-05-08. The 14-day physics-rebuild is complete.

## What shipped

### Week 1 — empirical fixes (Modes 1-4)

| Day | Mode | What | Result |
|---|---|---|---|
| 0 | (prep) | Held-out manifest + catalog flag + verify script | clean |
| 1 | 2 | Tyre pressure floor pin (`_apply_tyre_pressure_floor_pin`) | clean |
| 2 | 4 | Per-parameter local density confidence (`with_local_density`) | clean |
| 3-5 | 1 | Hierarchical Bayes retrofit (closed-form empirical-Bayes) | math correct, held-out outlier-target |
| 6 | 3 | Lap-time-weighted samples (`_compute_lap_time_weights` + weighted median) | clean |
| 7 | (Week 1 cumulative) | All-mode validation gate | partial (3/4 modes clean; Mode 1 corpus-property finding) |

**Modes cleanly closed**: 2, 3, 4. **Mode 1**: math correct, empirical
evidence inconclusive on this corpus (held-out IBTs are themselves
outlier setups, not converged targets — same pattern across Days 5,
7, 9).

### Week 2 — scoped physics (Mode 5 + infrastructure)

| Day | What | Result |
|---|---|---|
| 8 | Telemetry-derived diagnostic state (β, axle slip, axle force decomposition) | clean |
| 9 | Damper curve refit (per-car k from corpus shock-velocity p30/p95 percentiles) | partial (Cadillac 0.34% MAE; held-out outliers) |
| 10 | Per-axle grip-margin model (Pacejka replacement per Reviewer Agent 1's veto) | clean (94-96% accuracy) |
| 11 | Aero-map residual correction module | shipped via authorized fallback (correction couldn't beat raw) |
| 12 | Per-corner-phase physics evaluator | partial (methodology finding) |
| 12b | Per-car calibrated weights + guardrails reframe | shipped (130% relative improvement on within-group) |
| 13 | Hybrid optimizer with phase-aware weighting | shipped (mid_corner=0.40, others 0.05-0.10) |
| 14 | `--physics` CLI flag + final docs | shipped (this file) |

**Major Week 2 finding**: physics signal is concentrated in `mid_corner`
phase (Spearman +0.23 cross-car, vs +0.01-0.09 for braking/exit/etc).
Setup determines steady-state cornering; driver inputs dominate other
phases. Phase-aware weighting in the hybrid optimizer exploits this.

## Final test surface

- 22 tests in `tests/physics/test_evaluator.py`
- 16 tests in `tests/physics/test_hybrid_optimizer.py`
- 17 tests in `tests/physics/test_axle_grip.py`
- 24 tests in `tests/physics/test_diagnostic_state.py`
- 14 tests in `tests/physics/test_lap_weighted.py`
- 16 tests in `tests/physics/test_bayes_retrofit.py`
- 9 tests in `tests/physics/test_bayes_wire_in.py`
- 8 tests in `tests/physics/test_local_density_integration.py`
- 14 tests in `tests/aero/test_residual_correction.py`
- 16 tests in `tests/confidence/test_local_density.py`
- 9 tests in `tests/cli/test_tyre_pressure_floor.py`
- 6 tests in `tests/cli/test_physics_flag.py`
- 9 tests in `tests/ingest/test_held_out.py`
- 15 tests in `tests/physics/test_damper_refit.py`

**~195 new tests across 14 days. All pass on master.**

## What's in production now

User-visible:
- `optimize <car> <track>` recommendations now pin tyre pressure to
  the constraint floor by default (Mode 2).
- Confidence labels per parameter reflect local density (Mode 4)
  rather than a single global "dense/noisy" label.
- Recommendations are biased toward fast-lap setups via lap-time-
  weighted samples (Mode 3).
- Per-(parameter, track) Bayesian posteriors stored in PhysicsModel
  (Mode 1 math; recommend integration limited).
- New `--physics` flag adds an informational banner above the
  briefing showing per-car evaluator weights, geometry, and tyre
  floor pin status. Recommendation values unchanged.

Infrastructure:
- `physics/diagnostic_state.py` — β, axle slip, axle force
  decomposition from chassis-G channels.
- `physics/axle_grip.py` — per-axle grip-margin model with
  empirical mu_peak ceiling per car.
- `physics/damper_force.py` — refit damper curves per car
  (DamperCurve dataclass) with backward-compat seeded path.
- `physics/bayes_retrofit.py` — empirical-Bayes hierarchical
  retrofit (closed-form, no MCMC).
- `physics/evaluator.py` — per-corner-phase composite physics score
  with per-car calibrated weights and guardrails reframe.
- `physics/hybrid_optimizer.py` — phase-aware physics+surrogate
  combination with guardrail penalty.
- `aero/residual_correction.py` — per-car aero correction (fallback
  on this corpus).

## Honest assessment

**What worked**: Modes 2, 3, 4 closed cleanly. Days 8 and 10
delivered solid Mode 5 prep. The phase-aware hybrid (Day 13)
exploits a real corpus-derived structural finding (physics has
signal in mid_corner specifically).

**What didn't fully work**: Mode 1 (Bayesian retrofit) and Day 9
(damper refit) and Day 12 (evaluator as lap-time predictor) all
hit the same empirical limit -- the held-out IBTs are not
converged-fast targets, so gates that compare model output to
held-out setups generate corpus-property findings rather than
algorithm-quality findings.

**The honest framing**: physics-based scoring of corner-phase
performance from telemetry alone has fundamental limits on this
corpus. The components capture necessary conditions for fast
cornering but not sufficient ones; raw lap-time depends on driver
inputs the physics evaluator cannot see. The reframe (guardrails)
is the genuine value.

## Recommended next steps (not in this 14-day scope)

1. **Per-corner-phase weighted lap-time integration**: the existing
   per-corner-phase scoring infrastructure could be aggregated into a
   lap-time prediction with per-corner duration weights. Day 13's
   investigation showed this is the right granularity.

2. **Wire hybrid optimizer into recommend.py**: Day 14 ships the
   `--physics` flag as informational. A future iteration could
   actually use the hybrid score in DE search, with the per-phase
   weights as the integration. The infrastructure is in place; the
   wiring is ~200 LoC of `physics/recommend.py` modifications.

3. **Refine damper curve fit**: Day 9 shipped a 2-parameter
   percentile-anchored fit. A 3-parameter fit (separate low-speed,
   high-speed, transition) could produce per-car distinctions richer
   than the current "single k per car" approach.

4. **Active-learning DOE**: Reviewer Agent 3 ranked active-learning
   probes (`optimize calibrate`) #2 alternative. Closing thin-corpus
   parameters via driver-directed setup probes is complementary to
   the surrogate-side improvements this rebuild made.

## Plan-of-record adherence

PLAN.md was authored Day 0 and treated as immutable. One amendment
attempt (DEVIATION_day_07_gate_amendment.md) for criterion #3
substitutes was documented and the substitute criteria were
empirically rejected too. The plan's stop-and-wait discipline at
Days 1, 5, 7, 9, 12, 14 surfaced gate failures honestly and let the
user adjudicate.

External-judge subagents fired on Days 1, 2, 3, 4, 6, 8, 10, 12b
and twice rejected work I had nearly shipped (Day 12b speed-anchored
proxy, identified as tautology recurrence). The judge process
provided real friction, not theatre.

## Acknowledgments

User directives that shaped the build:
- "Build it for 2 weeks straight" -- the original scope
- "Find ways to make it work and accurate" (Day 12b) -- triggered
  the per-car calibration + guardrails reframe
- "A and C" (Day 13) -- triggered the phase-responsiveness
  investigation that yielded the mid_corner finding

The 14-day plan from PLAN.md was not blindly followed; the user's
directives at Day-12 and Day-13 hard stops materially improved the
final design.
