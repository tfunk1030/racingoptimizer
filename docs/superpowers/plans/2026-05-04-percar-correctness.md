# Per-Car (v4) Correctness Fixes for Cadillac@Spa

## Context

The v4 per-car path runs end-to-end and produces an exit-0 Cadillac@Spa
recommendation, but independent telemetry analysis (raw IBT read +
lap-time-anchored truth) found three concrete defects:

1. **Per-parameter sensitivity in the briefing reads `+0.000 score`** for
   every parameter. `explain.justification.build_justifications` calls
   `score_breakdown` / `score_setup` without the new `schedule=` kwarg, so
   the v4 path raises `ValueError("per-car model (v4) requires schedule")`
   and the `_safe_*` wrappers swallow it as `{}` / `0.0`.

2. **The recommendation extrapolates outside the observed range.** The
   driver has only ever run heave spring at 30 or 50 N/mm across 18
   sessions; the model recommended 25 N/mm. With `regime=noisy` and a 50%
   trust fraction, the trust bound centered on the per-session-median
   baseline (~50) does cover [27.5, 72.5] for an 80 N/mm constraint span,
   but DE still ended up at 23.71 N/mm — which means either the trust
   bound is wider than computed (`baseline` drifted low after pinning) or
   the post-DE clamp pushed against a constraint floor that's below the
   trust floor. Either way, **trust bounds should never extend beyond the
   union of values the driver has actually run on the TARGET track**.

3. **No bottoming penalty in `score.platform()`.** The Spa telemetry
   shows LF ride height bottoming to -0.2 mm somewhere on the lap; Eau
   Rouge alone produces 134 mm/s front shock vel and drops LF to 8.7 mm.
   `platform()` penalizes ride-height *variance* and shock-deflection p99
   but has no hard floor on the predicted mean ride height. Going to a
   softer heave spring (which the model recommended) would worsen this,
   and the score function has no mechanism to push back.

The recommendation's heave/third-spring direction is the OPPOSITE of what
the empirically fastest Spa setup (123.067 s lap, `wing=15 heave=50
third=550 spring_LR=150 hperch=-19`) ran. We need to fix all three
defects before per-car output is trustworthy enough to roll out beyond
Cadillac.

## Critical files

* `src/racingoptimizer/explain/justification.py` — `_safe_score_breakdown`
  and `_safe_score_total` need the schedule passed through; `_split_impacts`
  and `_sensitivity` are the call sites; `build_justifications` needs a
  new `schedule=` parameter.
* `src/racingoptimizer/cli/recommend.py` — `recommend_cmd` already has
  the schedule in scope (per-car branch); it must thread it into
  `build_justifications(...)`.
* `src/racingoptimizer/physics/recommend.py` — trust-bound construction
  (`_pin_or_trust_bounds` + `_trust_bounds`). New plumbing: target-track
  observed values must reach this code, capping the trust bound to the
  union [observed_min, observed_max] expanded by ±1 click.
* `src/racingoptimizer/physics/fitter.py:fit_per_car` — collect per-track
  observed value sets, store them on the new `target_track_observed`
  field of `PhysicsModel`.
* `src/racingoptimizer/physics/model.py:PhysicsModel` — append-only field
  `per_track_parameter_observed` (dict `track → {param → tuple[float, ...]}`).
* `src/racingoptimizer/physics/score.py:platform` — bottoming penalty
  driven by predicted mean ride height across the four corners.

## Change set

### 1. Schedule plumbing through `build_justifications`

* Add `schedule: list | None = None` to `build_justifications`.
* Add `schedule=...` to `_safe_score_breakdown` and `_safe_score_total`
  signatures; forward to `score_breakdown` / `score_setup`.
* In `cli/recommend.py:recommend_cmd`, pass the per-car branch's
  `schedule` into `build_justifications`. The non-per-car branch passes
  `schedule=None`, which is what the v3 path already wants.

Acceptance: Cadillac@Spa briefing shows non-zero `+1 click` /
`-1 click` deltas for at least the parameters whose Confidence regime is
not `sparse`.

### 2. Per-track-observed trust-bound cap

The fix has two parts:

(a) **Collect target-track observed values** at fit time:

* In `fit_per_car`, build `per_track_parameter_observed: dict[str,
  dict[str, tuple[float, ...]]]` keyed by `track → param → unique
  observed values across that track's sessions`.
* Add an append-only field on `PhysicsModel` named
  `per_track_parameter_observed: dict = field(default_factory=dict)`
  with the matching `__setstate__` default.

(b) **Cap the trust bound** at recommend time:

* Extend `recommend()` to take an optional `target_track_observed:
  dict[str, set[float]]` so the caller can override (the per-car CLI
  pipeline will pass `model.per_track_parameter_observed.get(track,
  {})`).
* Inside `_pin_or_trust_bounds`, when target-track observed values exist
  for the parameter, cap the returned `(lo, hi)` to
  `(min(observed) - one_click, max(observed) + one_click)` clipped to
  the constraint bound. `one_click` comes from
  `ParameterSpec.step` (already in the ontology) or `1% of bound span`
  fallback. This stops the optimizer from extrapolating outside the
  driver's empirical envelope on the target track.
* Pre-existing pin-vs-trust logic still applies on top.

Acceptance: With Cadillac@Spa where `heave_spring` was observed at
{30, 50} on Spa, the recommendation must come back in the range
`[30 - 5, 50 + 5] = [25, 55]` (5 N/mm = one click for heave spring per
the ontology step). Today it's at 23.71; after the fix it should be ≥ 25
or get pinned to one of the observed values.

### 3. Bottoming penalty in `score.platform()`

Add a hard penalty that fires when predicted ride height crosses a
safety floor:

```python
RIDE_HEIGHT_SAFETY_FLOOR_MM = 5.0       # splitter/floor scrape risk
RIDE_HEIGHT_PENALTY_DEPTH_MM = 10.0     # smooth ramp above the floor

bottoming_penalty = 0.0
for v in rh_vals:
    headroom = v - RIDE_HEIGHT_SAFETY_FLOOR_MM
    if headroom < RIDE_HEIGHT_PENALTY_DEPTH_MM:
        bottoming_penalty = max(
            bottoming_penalty,
            _clip01(1.0 - headroom / RIDE_HEIGHT_PENALTY_DEPTH_MM),
        )
penalty = max(rh_penalty, shock_penalty, bottoming_penalty)
```

This is additive to the existing `rh_penalty` (variance) and
`shock_penalty` (defl p99). It catches setups that the predictor expects
to drop predicted MEAN ride height into the scrape zone, which is
exactly what going softer on heave/third spring at a track that's
already bottoming would do.

Acceptance: The `score.platform` unit test continues to pass for the
existing cases (predicted RH well above floor → bottoming_penalty = 0).
Add a new test where rh_vals include a 6 mm value → bottoming_penalty ≈
0.9, total `util` drops below 0.2.

## Order of execution

1. Schedule plumbing through justifications (smallest, unblocks the UX).
2. Bottoming penalty in `platform()` + new unit test.
3. Per-track observed bounds (`fit_per_car` collection +
   `PhysicsModel` field + `_pin_or_trust_bounds` cap).
4. Re-run Cadillac@Spa and verify:
   * Heave spring recommendation in `[25, 55]`.
   * Briefing shows non-zero per-parameter sensitivity for non-pinned
     parameters.
   * Compare new recommendation against driver's empirical fast lap
     (`wing=15 heave=50 third=550 spring_LR=150 hperch=-19`); deltas
     should be small.

## What this plan does NOT fix (deferred follow-ups)

* Adding compression-demand archetype features (e.g., descending shock
  velocity per corner) so the per-car model can learn that high-elevation
  corners want stiffer heave. Worth doing once the bottoming penalty is
  in.
* Schedule-aware per-corner time-sensitivity weighting (today every Spa
  corner is uniformly weighted; longer, slower corners should weigh
  more).
* Rolling per-car v4 out to bmw / acura / ferrari / porsche. Wait until
  Cadillac validates clean.

## Verification

```bash
# Smoke check: existing tests still pass.
.venv/Scripts/optimize.exe status cadillac
.venv/Scripts/python.exe -m pytest tests/physics/test_score.py -x -q
.venv/Scripts/python.exe -m pytest tests/explain/ -x -q

# Per-car re-run with no cache.
.venv/Scripts/optimize.exe cadillac spa --no-cache

# The recommendation block must show:
# - heave_spring in [25, 55]
# - non-zero "+1 click / -1 click" sensitivity for non-pinned params
# - no top-level Warnings about extrapolated baselines
```
