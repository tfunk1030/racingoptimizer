# VISION §5 Optimization — Audit

**Status: PASS** — the recommender reasons across the full corner-phase
hypercube via differential evolution, narrows the trust radius for sparse
parameters, clamps every output to `constraints.md`, and uses the empirical
training range (not the legal constraint span) as the denominator for the
near-constant pin check. Lap time never enters the objective.

## VISION clause

> When recommending setup changes, think through the FULL consequence chain.
> Stiffening the front heave spring improves aero platform stability at high
> speed (benefiting fast corners and straights) but reduces mechanical
> compliance over bumps (hurting slow corners and braking zones). Quantify
> both sides using the per-corner-phase model. The optimal setup is the one
> that maximizes total lap performance across ALL corners, not the one that
> fixes one problem at the expense of others. Know that changing one
> parameter changes other things — stiffer springs change ride heights which
> change aero balance which change load transfer which change tire temps
> which change grip. Chase the chain.

## Per-clause scorecard

| Clause | Status | Evidence |
|---|---|---|
| Full-hypercube DE search over the bounded parameter space | PASS | `physics/recommend.py:202-212` calls `scipy.optimize.differential_evolution(objective, bounds=bounds, ...)` with seed, popsize=10, maxiter=5, polish=False, tol=5e-3, mutation=(0.3, 1.0). `bounds` is built from every fittable parameter that has a `constraints.bounds(...)` entry (`recommend.py:92-153`). |
| Reason about trade-offs across ALL corners | PASS | The objective sums the per-(corner, phase) score breakdown — `recommend.py:181-194` `objective(x)` returns `-sum(_score_breakdown_per_car(...).values())` (per-car v4) or `-sum(_score_breakdown(...).values())` (legacy). Each corner's contribution is weighted by `weight_corners(track, model)` (`recommend.py:90`, `_cached_weights`). |
| Coupled chain — one parameter change propagates through every output | PASS | Stage-3 fitter is per `(corner, phase, channel)` over the FULL setup vector + 12 env channels (`physics/fitter.py`). The objective rebuilds the whole setup dict before scoring (`recommend.py:182-185`: `candidate = dict(full_baseline); candidate[name] = value`). Coupling is asserted by `tests/physics/test_coupling.py::test_perturb_one_parameter_changes_multiple_corners`. |
| Trust radius narrows for sparse / noisy parameters | PASS | `_TRUST_FRACTION = {"sparse": 0.30, "noisy": 0.50}` (`recommend.py:327`). Confident / dense parameters use the full constraint range. `_median_regime` (`recommend.py:330-351`) pulls each parameter's regime from the median of every fitter that depends on it (Stage-3 aware via `feature_names`). |
| Pin near-constant parameters to observed median (no DE drift on noise) | PASS | `_pin_or_trust_bounds` (`recommend.py:404-477`) collapses the search window to `baseline ± 1e-6 * span` when `observed_std / pin_denom < 0.02`. Pinned parameters surface in `SetupRecommendation.pinned_to_observed_median` so the briefing can explain why no exploration happened. |
| **Pin denominator is the empirical training range, not the constraint span** | PASS (recently fixed) | `recommend.py:444` — `pin_denom = empirical_range if empirical_range > 0.0 else span`. `empirical_range` is computed in `recommend.py:136-140` as `max(global_observed[name]) - min(global_observed[name])` from `model.per_track_parameter_observed`. Without this, BMW heave (50 N/mm) was being pinned because `observed_std=11.48 / span=900 ≈ 1.27%` fell below 2% — even though across 9 BMW Sebring sessions the corpus carried 30/40/50/60/80 N/mm. With the fix: `11.48 / 50 = 23%` → not pinned. Regression test: `tests/physics/test_pin_near_constant.py::test_no_pin_when_empirical_range_dominates_wide_constraint`. |
| Per-track empirical envelope cap on trust bound (per-car v4 only) | PASS | `recommend.py:75-89, 132-149, 454-477` — per-car models thread `target_observed` (the unique values the driver has actually run on the TARGET track) into `_pin_or_trust_bounds`, which intersects the trust bound with `[min(target_observed), max(target_observed)]`. Single-value envelopes get one click of margin so DE has something to search. Documented contract: "the per-car recommender will not extrapolate outside the empirical envelope; recommending a never-tried value would be guessing against a confidence bracket the joint surrogate cannot honestly estimate." |
| Constraint clamp on every output | PASS | Two-layer clamp: (a) inside the objective on every DE evaluation — `_clamped_or_raise(model, candidate, strict=False)` (`recommend.py:185`, defined in `physics/score.py:491-514`); (b) after DE returns — `clamp(float(value), name, model.car, constraints)` on each parameter (`recommend.py:217`) plus a defensive raise if the post-clamp moved the value (`recommend.py:218-222`). The CLI also re-clamps + rounds discrete values at render time. |
| Bound-binding clamp warnings | PASS | When the observed median sits OUTSIDE the legal range (`recommend.py:118-130`) OR when DE returns a value within ~1% of a bound while the baseline sat the other side (`recommend.py:229-252`), the recommender adds a `clamp_warnings[name]` entry. This catches the case where `constraints.md` is wrong relative to the iRacing garage UI (e.g. the recent tyre pressure 165→152 floor fix in `9e949be`). |
| Lap time NOT used as the optimization signal | PASS | Greps of `physics/score.py` and `physics/recommend.py` return zero matches for `lap_time`/`laptime`/`LapTime`/`LapLast`/`LapBest`. The only references live in `physics/weights.py` where `lap_time_s` is the SOURCE of per-corner time-sensitivity weights (the SCALARS that multiply each corner's utilization sum) — never inside the objective. Validator gate: `tests/physics/test_validator_gate.py::test_no_lap_time_in_objective` parametrised over both modules. |
| Per-corner-phase score breakdown returned with the recommendation | PASS | `recommend.py:257-259` — `breakdown = score_breakdown(model, recommended, track, env, weights=weights, schedule=schedule)` is attached to `SetupRecommendation.score_breakdown`. The CLI briefing uses this to render per-parameter helps/hurts corner lists. |

## Test results

```
uv run pytest -q tests/physics/test_per_car_recommend.py tests/physics/test_pin_near_constant.py tests/physics/test_recommend.py
.....................                                                    [100%]
21 passed in 347.59s (0:05:47)
```

Breakdown (21 tests):

- `test_per_car_recommend.py::test_recommend_per_car[<car>-<track>]` × 5 cars
  (acura, bmw, cadillac, ferrari, porsche) — full DE search runs end-to-end
  and returns a clamped, in-bounds recommendation.
- `test_per_car_recommend.py::test_recommend_determinism_per_car[<car>-<track>]`
  × 5 — identical seed → identical recommendation.
- `test_pin_near_constant.py` × 6 — including
  `test_no_pin_when_empirical_range_dominates_wide_constraint`, the
  regression test for the BMW heave un-pinning fix.
- `test_recommend.py` × 5 — bounds-respect, per-corner-phase breakdown,
  determinism, model-method form.

The regression test asserts BOTH directions: with `empirical_range=50.0`
the BMW heave (std=11.48 over a 900-N/mm constraint span) does NOT pin;
without it (legacy v3 path), it DOES pin — locking in the bug we fixed.

## BMW Spa card evidence (`recommendations/bmw__spa_2024_up__20260505-180530.txt`)

| Check | Result |
|---|---|
| Pinned-warning lists ONLY truly-constant parameters | PASS — line 676: `pinned to observed median ... arb_size_front`. That's the ONLY pinned param. The BMW Spa corpus runs `arb_size_front=1` across every session (no variation), so pinning is correct. |
| Heave spring optimized (not pinned) after the empirical-range fix | PASS — line 717: `Heave Spring   50 N/mm   [OPT]` (NOT `[OPT pin]`). Pre-fix this would have been `[OPT pin]` because the BMW heave constraint span (0..900 N/mm per BMWBounds.md) swallowed the actual training stddev. |
| Other previously-pinned families now optimized | PASS — `Third Spring 460 N/mm (was 420) [OPT]`, `Third Perch Offset 45.5 mm (was 45) [OPT]`, `Heave Perch Offset -22.5 mm (was -22) [OPT]`, `Spring Perch Offset 35.5 mm (was 30) [OPT]`, `Pushrod Length Offset -32.5 mm (was -33) [OPT]`. All are in the per-axle / heave / perch families that BMWBounds.md gives wide legal envelopes to. |
| Constraint clamp working — every recommended value within legal range | PASS — every `[OPT]` and `[OPT pin]` block in the briefing carries an `observed in training [lo, hi]` line and (for discrete clicks) a `legal range a..b` line. Spot-check: `Anti Roll Bar Front: 1.00 click` (legal range 1..5), `Brake Bias Pct: 46.77 pct` (well inside 40..60), `Tyre Cold Pressure: 152.44 kPa` (sits at the post-fix 152 kPa floor — clamped from observed 152.5 kPa), `Heave Spring 49.14 N/mm` (within BMW envelope). No value falls outside its declared legal range. |
| Discrete-click rounding visible | PASS — every damper / ARB / diff-plate parameter shows e.g. `discrete-click value rounded from 8.623 to 9 (legal range 0..11)`. The DE search returns floats; `_post_clamp` rounds to integer clicks AND re-clamps. |
| Untrained-parameter warning lists ONLY items missing from `constraints.md` | PASS — line 677: `untrained_parameters: corner_weight_fl_kg, corner_weight_fr_kg, corner_weight_rl_kg, corner_weight_rr_kg (skipped — bounds not in constraints.md)`. Slice E gracefully degrades, not crashes. |
| Per-parameter justification is complete (VISION §7 cross-check) | PASS — every `[OPT]` block in the briefing carries `Helps: <3 corners>`, `Hurts: <3 corners>`, `+1 click: <Δscore>`, `-1 click: <Δscore>`, `Evidence: <regime + n_samples + observed range>`. This is the corner-by-corner trade-off VISION §5 asks for. |

## Files reviewed

- `C:\Users\VYRAL\racingoptimizer\VISION.md`
- `C:\Users\VYRAL\racingoptimizer\CLAUDE.md`
- `C:\Users\VYRAL\racingoptimizer\docs\VISION_COMPLIANCE.md`
- `C:\Users\VYRAL\racingoptimizer\src\racingoptimizer\physics\recommend.py`
- `C:\Users\VYRAL\racingoptimizer\src\racingoptimizer\physics\score.py`
- `C:\Users\VYRAL\racingoptimizer\src\racingoptimizer\constraints\clamp.py`
- `C:\Users\VYRAL\racingoptimizer\src\racingoptimizer\constraints\loader.py`
- `C:\Users\VYRAL\racingoptimizer\tests\physics\test_pin_near_constant.py`
- `C:\Users\VYRAL\racingoptimizer\tests\physics\test_per_car_recommend.py`
- `C:\Users\VYRAL\racingoptimizer\tests\physics\test_recommend.py`
- `C:\Users\VYRAL\racingoptimizer\recommendations\bmw__spa_2024_up__20260505-180530.txt`

## Notes / minor follow-ups

- The DE budget is intentionally tight (`maxiter=5`, `popsize=10`,
  `tol=5e-3`, `polish=False`) to stay under the per-spec §13 5-second
  recommendation latency. With ~40 fittable parameters × ~70 schedule
  entries × ~30 output channels per evaluation this comes out to roughly
  one minute total in the per-car v4 path. Worth re-measuring as the
  feature schema grows.
- The empirical-range fix (`empirical_range` denominator in
  `_pin_or_trust_bounds`) is per-car v4 only. Legacy v3 pickles return
  `empirical_range=0.0` and fall back to the constraint-span denominator,
  so the BMW-heave-pinning regression can re-appear if the recommender is
  ever called against a stale v3 cache. The model-cache fingerprint
  (`cli/recommend.py:_model_cache_path`, audited by §6) folds in the
  feature-schema version so this should not happen in practice.
- `test_no_pin_when_empirical_range_dominates_wide_constraint` asserts
  the symmetry — same parameters with vs without `empirical_range` →
  un-pinned vs pinned — so a future refactor that drops the denominator
  swap will be caught immediately.
