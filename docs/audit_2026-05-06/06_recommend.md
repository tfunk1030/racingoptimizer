# Audit -- Slice E2: Recommend / DE (2026-05-06)

## Summary
- Grade: PARTIAL
- The DE search, pin/trust math, ontology routing, and CLI wiring are coherent and the production code is clean -- but the recently added `--reset` and `--explore` knobs ship with **zero test coverage**, and the only per-car recommend smoke is `pytest.mark.slow` (so the merge gate never exercises the full call path).

## Implementation quality

**`src/racingoptimizer/physics/recommend.py`**
- Public surface is a single `recommend(model, track, env, constraints, *, schedule, quali, explore_pct, reset_mode)` (recommend.py:45-55) with the `PhysicsModel.recommend` thin wrapper at `physics/model.py:211-227`. Per-car v4 vs v3 routing is by `is_per_car = int(model.feature_schema_version) >= 4` (recommend.py:80) -- used consistently in `_cached_weights` (recommend.py:353), schedule guard (recommend.py:81-86), global-corpus aggregation (recommend.py:106-111), and inner objective branch (recommend.py:239-248). Clean.
- Trust-envelope policy is well-documented (recommend.py:87-104, 484-514). The intermediate "per-track strict -> cross-track fallback" history is correctly purged: `target_observed` is built solely from `model.per_track_parameter_observed` aggregated across every track (recommend.py:106-111). **No surviving per-target-track variables/dead branches** (verified via grep for `target_track` / `cross_track` / `per_track_observed`). Comment-history at recommend.py:87-104 and 500-504 is appropriate documentation.
- `_pin_or_trust_bounds` (recommend.py:472-615) reads cleanly:
  - Reset short-circuit (recommend.py:529-551) anchors on corpus envelope clipped to constraint bounds, widens by `_RESET_WIDEN_PCT = 30.0` of constraint span each side, defensive fallback when widening still produces lo>=hi.
  - Pin denominator: `empirical_range if empirical_range > 0.0 else span` (recommend.py:552) -- the BMWBounds 0..900 N/mm masking-fix.
  - Trust radius from `_TRUST_FRACTION = {"sparse": 0.30, "noisy": 0.50}` (recommend.py:388); confident/dense get full envelope.
  - Defensive guard: `target_observed` clipped to `[lo, hi]` BEFORE empirical-window math (recommend.py:569). Backed by `test_user_pin_outside_target_observed_does_not_invert`.
  - Final `if trust_hi < trust_lo: trust_lo, trust_hi = lo, hi` (recommend.py:613-614) -- belt-and-suspenders fallback.
- DE budget (recommend.py:257-267): `maxiter=15, popsize=20, polish=False, init=init_population, tol=5e-3, mutation=(0.3, 1.0)`, seed `model.seed & 0x7FFFFFFF`. All magic numbers documented inline (recommend.py:251-256). Reset uses constraint midpoints for candidate 0 (recommend.py:208-210), composing correctly with open-envelope `sub_bounds`. Confidence-downgrade for reset at result-collection time (recommend.py:310-314).
- `_seed_population` (recommend.py:639-659) sizes via `n_pop = max(min(pop_size * max(n_params, 1), 30), 5)` -- caps at 30; **docstring at recommend.py:650 says "capped at 50"** -- stale.
- Clamp warnings (recommend.py:129, 147-152, 284-307) populated at two distinct moments (baseline-clamp time + result-at-bound time); both branches share warning shape.
- `_cached_weights` (recommend.py:341-379) keys on `(id(model), track)`; `id()`-key is documented as model-lifetime-stable; falls back to corner-uniform weights when `weight_corners` raises (recommend.py:366-377). Bare `except Exception` is `# pragma: no cover` -- defensible but worth narrowing.
- `_record_depends_on` (recommend.py:415-423) is single source of truth for "does this fitter touch parameter X" -- used by both `_median_regime` and `_parameter_confidence`. No duplication.
- Minor smell: `from racingoptimizer.physics.score import _conditions_adjusted_baselines` is imported inside the function (recommend.py:225) while every other `physics.score` import is at module top (recommend.py:34-41). Could be hoisted.

**`src/racingoptimizer/physics/ontology.py`**
- `ParameterSpec` is `@dataclass(frozen=True, slots=True)` (ontology.py:43-100) with the explicit "append-only beyond this point" invariant (ontology.py:69-77) -- load-bearing because instances are pickled inside `PhysicsModel.ontology`. Three later fields (`step`, `discrete_values`, `choices`) all carry defaults so old revives still construct. Good discipline.
- Per-car ontologies built via `_build(damper_paths, **overrides)` (ontology.py:481-485, 774-778). `_common_bounded()` / `_common_ce_gated()` return fresh dicts per call (ontology.py:175, 300) so per-car overrides mutate locally. Damper-path generation via `_damper_paths` (ontology.py:423-441) cleanly factors per-corner × 5 modes for inline (BMW/Cadillac), split (Acura/Porsche), and Ferrari per-corner damper layouts.
- `_blocked_like(base)` (ontology.py:398-408) marks `fittable=False` for cars lacking corresponding YAML leaves so the optimizer never searches non-existent garage knobs.
- `setup_value` (ontology.py:833-868) handles JSON-string vs dict, NaN-tolerant via `_walk` (ontology.py:124-130), categorical -> ordinal-index mapping for `spec.choices` (ontology.py:862-867).
- `fittable_parameters(car, table)` (ontology.py:804-830) is the three-gate filter (`fittable`, `user_settable`, has `constraints.bounds`) -- exactly what `recommend` calls at recommend.py:114-117.

## Wiring

- `cli.recommend.recommend_cmd` (`cli/recommend.py:163`) -> `model.recommend(...)` (`cli/recommend.py:278` v4 per-car / `:299` v3) -> `PhysicsModel.recommend` (`physics/model.py:211-227`) -> `physics.recommend.recommend` (recommend.py:45). Both CLI sites pass `quali, explore_pct, reset_mode`; v4 additionally passes `schedule`.
- `physics.recommend.recommend` consumes:
  - `model.feature_schema_version` (recommend.py:80) -- v4-vs-legacy routing.
  - `model.per_track_parameter_observed` (recommend.py:107) -- built in `physics/fitter.py:845`. Source of the global corpus envelope.
  - `model.parameter_observed_std` (recommend.py:118-120) -- built in `physics/fitter.py:391, :841`.
  - `model.{baseline_setup, fitters, untrained_parameters, aero_correction_available, ontology, seed}`.
  - `constraints.bounds(car, name)` from `racingoptimizer.constraints.load_constraints()`.
  - `physics.ontology.fittable_parameters(model.car, constraints)` (recommend.py:114).
  - `physics.score._conditions_adjusted_baselines` (recommend.py:225) for wet/quali overlay.
  - `physics.score.{_score_breakdown, _score_breakdown_per_car, score_breakdown, _aero_surface_or_none, _clamped_or_raise, _corner_phase_keys}` (recommend.py:34-41).
  - `physics.weights.weight_corners` (recommend.py:42, 365) -- v3 path; v4 uses uniform per-schedule-corner weights (recommend.py:354-362).
- `_pin_or_trust_bounds` called from one site (recommend.py:170-180); `_trust_bounds`, `_click_step_for`, `_record_depends_on`, `_median_regime`, `_parameter_confidence`, `_seed_population`, `_fill_untrained_baselines`, `_baseline_recommendation` all internal-only.
- `reset_mode` thread: `cli/recommend.py:112` (Click flag) -> `cli/recommend.py:171, 248-253, 281, 302, 411-412` (banner, mode tag, file-name suffix) -> `physics/model.py:220, 226` -> `physics/recommend.py:54, 179, 208-210, 310-314, 482, 529-551`.
- `explore_pct` thread: `cli/recommend.py:92, 169, 280, 301` -> `physics/model.py:219, 225` -> `physics/recommend.py:53, 178, 481, 585-588`.

Test reach:
- `tests/physics/test_per_car_recommend.py` is `pytestmark = pytest.mark.slow` (line 21) -- excluded from `-m "not slow"`. Two cases (correctness + determinism), parametrised over per-car fixtures.
- `tests/physics/test_pin_near_constant.py` covers eight `_pin_or_trust_bounds` paths plus one end-to-end `recommend` smoke (`test_full_recommend_pins_near_constant_param`). **None exercise `reset_mode=True` or `explore_pct>0`.**
- `tests/physics/test_score.py` -- only `test_score_setup_no_lap_time_reference` (line 256-266) reaches into `recommend.py` (asserts `lap_time` does not appear).
- `tests/physics/test_recommend.py` -- non-slow recommend coverage (return-type, in-bounds, breakdown, determinism, `model.recommend` delegation) -- adjacent file, not in the assigned set.

## Gaps

1. **MAJOR -- no test coverage for `--reset`** (recommend.py:529-551). Reset has its own widening, fallback, seed-population path, and result-side confidence downgrade (recommend.py:310-314). A regression that re-collapsed the envelope or skipped the downgrade would not be caught.
2. **MAJOR -- no test coverage for `--explore N`** (recommend.py:585-588). Explore-widening intersects with empirical-envelope clipping; off-by-one or clipping-order regressions would silently ship.
3. **MAJOR -- `tests/physics/test_per_car_recommend.py` is `pytest.mark.slow`** (line 21), so the per-car recommend smoke does not run under the documented `-m "not slow"` merge gate. This is the only file looping every car through `recommend`. Per CLAUDE.md verification convention ("slice must have a `tests/<slice>/test_per_car_smoke.py` ... that loops the five canonical car fixtures") the slow gate lets a recommend regression land on master without local feedback.
4. **MINOR -- stale docstring**: `_seed_population` says "capped at 50" (recommend.py:650) but caps at 30 (recommend.py:652).
5. **MINOR -- inconsistent import location**: `_conditions_adjusted_baselines` imported inside function body (recommend.py:225) while sibling `physics.score` symbols import at module top (recommend.py:34-41). Either there's a cycle (worth a comment) or hoist.
6. **MINOR -- `_DIFF_CLUTCH_PLATES_VALUES`** (ontology.py:577) declared module-scope but only used by BMW (ontology.py:632); reads as one-shot constant.
7. **MINOR -- `_WEIGHTS_CACHE` is unbounded** (recommend.py:341). Comment acknowledges `WeakValueDictionary` is unavailable; no eviction. Practical impact nil today (one-model-per-process via CLI) but worth flagging.
8. **MINOR -- bare `except Exception`** in `_cached_weights` (recommend.py:366) gated by `# pragma: no cover`. Sensible fallback but masks programming errors. Narrow once failure modes are enumerated.
9. **NONE -- dead code from per-track -> cross-track -> global trust-envelope refactor**: verified absent. Only reference to per-track data is the aggregation loop at recommend.py:106-111. Comment-history at recommend.py:87-104 and 500-504 is documentation.

## Evidence
- Test suite: NOT RUN -- sandbox blocked Bash and PowerShell. Static reading of `test_pin_near_constant.py`, `test_per_car_recommend.py`, `test_recommend.py`, `test_score.py` shows test bodies and assertions are well-formed and consistent with the production code paths reviewed.
- Lint: NOT RUN -- same reason; visual review found no obvious ruff violations (no unused imports, no late-bind shadowing, no escape issues).
- Latest artefacts:
  - `recommendations/bmw-spa-reset-0506-1029.txt` -- RESET MODE evidence; banner header confirms `Confidence: noisy (median n=2674)` regime downgrade fired; narrative shows 36 of 47 parameters moved with diverse directional changes (`Front heave perch: -21.5 -> -52.0 mm`, `Rear pushrod offset: -29.5 -> -32.0 mm`). The 30% widening let DE genuinely escape the past-setup neighbourhood.
  - `recommendations/ferrari-spa-race-0506-1026.txt` -- cross-track envelope evidence; `ferrari @ spa_2024_up -- race (57.8 L fuel)` with `Confidence: dense (median n=902)` despite Ferrari having no Spa IBT files (training is global Ferrari corpus, schedule borrowed via `_maybe_borrow_cross_car_track`). 38 of 46 parameters moved using global-corpus envelope.
  - `recommendations/bmw-spa-cal-0506-1015.txt` and `bmw-spa-cal-status-0506-1018.txt` -- calibrate sister-command evidence; confirm the sub-`recommend` machinery exposes per-parameter coverage / pin counts.

## Recommended next actions
- Add `tests/physics/test_recommend_reset_mode.py` covering: (a) `reset_mode=True` produces a sub-bound *wider* than `reset_mode=False`; (b) returned `Confidence.regime` for fittable parameters is in `{"sparse","noisy"}`; (c) parameter with no `target_observed` falls back to empirical-range or baseline-only window without crashing; (d) widening past constraint bounds gets clipped.
- Add `tests/physics/test_recommend_explore_pct.py` covering: (a) `explore_pct=0` matches strict empirical envelope baseline; (b) `explore_pct=10` widens by 10% of constraint span on each side, clipped to bounds; (c) widening one-sided when empirical window touches a constraint edge.
- Promote at least one case from `test_per_car_recommend.py` to a fast smoke (e.g. BMW Sebring with a tiny session subset) so `-m "not slow"` exercises the full DE path end-to-end.
- Fix stale `_seed_population` docstring (recommend.py:650) and either hoist `_conditions_adjusted_baselines` import (recommend.py:225) or annotate the cycle.
- Optional: narrow bare `except Exception` in `_cached_weights` (recommend.py:366); add eviction or `weakref.WeakKeyDictionary` to `_WEIGHTS_CACHE` if CLI ever moves to a long-lived process.
