# Audit -- Slice E3: Score + conditions (2026-05-06)

## Summary
- Grade: **PARTIAL**
- Scoring core, wet/quali/wind branching, and the lap-time-isolation contract are all wired correctly and well tested at the unit level, but the directional half of the wind model is stranded (only magnitude is consumed), `weight_corners` does not run for any v4/per-car car (BMW, Cadillac, Ferrari), and several behaviour-critical paths (bottoming penalty, objective-confidence multiplier, wet/quali integration through `score_setup`) lack tests.

## Implementation quality

`src/racingoptimizer/physics/score.py`
- L17-24: imports cleanly bracket the lazy ones (`wind`, `wet_mode`, `quali_mode` deferred to function scope at L197, L492). Avoids the `model -> score -> wet_mode -> baselines` import cycle, but is asymmetric -- `baselines` and `phase_weights` are eager while `wet_mode` / `quali_mode` aren't.
- L37-46: `_REGIME_RANK`, `_RANK_TO_REGIME`, `_OBJECTIVE_CONFIDENCE_MULTIPLIER` are duplicated literal tables for the same regime names. Multipliers (`0.60 / 0.80 / 0.95 / 1.00`) are magic numbers with no spec citation in the file.
- L55-56: `_RIDE_HEIGHT_SAFETY_FLOOR_MM = 5.0` and `_BOTTOMING_PENALTY_DEPTH_MM = 10.0` global across all five cars; per-car splitter geometry differs.
- L99: `max_g = 0.5 * ld * density_factor + baseline`. Magic `0.5` converting L/D to a lateral-G scale -- VISION §3 forbids "hardcoded engineering formulas as the primary model".
- L165-166: `aero_eff` divides L/D by hardcoded `4.0` to land in [0,1]. Same VISION §3 concern.
- L256-290: bottoming-penalty branch is duplicated almost verbatim between the static-RH check and the at-speed-RH check.
- L304-311: `_SUB_FUNCS` keys must align with `SUB_UTILIZATIONS` (in `phase_weights.py`); no test enforces the alignment.
- L370-461: `score_setup` and `score_breakdown` duplicate the conditioning + dispatch logic; could be factored.
- L559-573: `_AERO_CACHE` module-global is never invalidated.
- L594-617: `_clamped_or_raise` swallows the clamp silently in non-strict mode -- no log line records it.

`src/racingoptimizer/physics/baselines.py`
- L56-81: `_COMMON_DEFAULTS` is identical for all five cars -- cold-start is car-agnostic. The literal numbers have no provenance trail.
- L131, L136: `wheelspin_scale_ms` and `aero_grip_baseline_g` are NEVER replaced by `derive_baselines` even though `wheel_speed_max_diff_ms` is in the corpus. Wired-but-not-derived gap.

`src/racingoptimizer/physics/weights.py`
- L46: `if laps_df.height < 3` -- three-lap threshold is unjustified by spec.
- L106-113: clipping negative sensitivity to zero may mask real anti-pattern corners.
- L121, L141: hardcoded 4-bucket default; numpy-bool-to-list polars filter is awkward.

`src/racingoptimizer/physics/phase_weights.py`
- Pinned per-spec; row-sum invariant asserted by `tests/physics/test_phase_weights.py`.

`src/racingoptimizer/physics/wet_mode.py`
- L27-30: `0.7 / 0.3 / 0.05` thresholds hardcoded global; cars cross the wet-tyre cliff at different wetness levels.
- L36-46: `_REGIME_SCALE` and `_AERO_SHIFT` asymmetry -- `damp` is in scale but missing from aero_shift, undocumented.
- L67/L74: **likely bug** -- `wet_baselines` calls `baselines_for(car)` -> `derive_baselines(car, frame=None)` which always returns COLD-START defaults, never corpus-derived. Wet mode silently throws away the per-car corpus baselines that took an entire `fit_per_car` run to derive.

`src/racingoptimizer/physics/quali_mode.py`
- L28-35: `_QUALI_SCALE` matches CLAUDE.md contract (1.15 / 1.20 / 0.55 / 1.0).
- Symmetry note: `wet_phase_weights("dry")` returns the SHARED `PHASE_WEIGHTS` reference; `quali_phase_weights()` always returns a fresh dict. Inconsistent contract / footgun.

`src/racingoptimizer/physics/wind.py`
- **L8-13 docstring is STALE** -- claims wind is "**not** wired into `score.aero_eff`"; in reality `score._wind_aero_scale` (L197) DOES call `aero_wind_modifier`. Magnitude path IS wired; directional half (`decompose_wind`, `crosswind` / `balance_shift_pct`) is NOT.
- `decompose_wind` and the `balance_shift_pct` element of `aero_wind_modifier` are dead in `src/`; consumed only by `tests/physics/test_wind.py`.

## Wiring

```
PhysicsModel.score_setup (model.py:198)
  -> score_setup (score.py:370)
       -> _clamped_or_raise (score.py:594)
       -> _resolve_weights (score.py:620)
       -> _aero_surface_or_none (score.py:562)
       -> _conditions_adjusted_baselines (score.py:467)
            -> classify_conditions (wet_mode.py:49)
            -> wet_baselines (wet_mode.py:67)         [non-dry only]
            -> wet_phase_weights (wet_mode.py:87)     [non-dry, race only]
            -> quali_phase_weights (quali_mode.py:38) [quali only -- overrides wet pick]
       -> if v4: _score_breakdown_per_car (score.py:675)
                  -> model.predict(... archetype=...) per schedule entry
                  -> aggregate_utilization (score.py:314)
                       -> grip / balance / stability / traction / aero_eff / platform
                       -> aero_eff -> _wind_aero_scale -> aero_wind_modifier (wind.py:55)
                  -> _confidence_adjusted_utilization (score.py:669)
          else:    _score_breakdown (score.py:637)
       -> sum(breakdown.values())
```

DE objective in `recommend.py:226-249` calls `_conditions_adjusted_baselines` once before the search loop, then per-evaluation calls `_score_breakdown_per_car` / `_score_breakdown` directly (bypasses `score_setup` to avoid re-conditioning).

CLI quali wiring (`cli/recommend.py`): `--quali` defined L83-88; requires `--fuel` L196-202; race-mode auto-pin skipped L205-213; threaded into `recommend()` at L280, L301, L377, L385; mode tag on artefacts at L411-418.

`weight_corners` is called only via `_cached_weights` (`recommend.py:344`); v4 path forces uniform weights when `feature_schema_version >= 4 and schedule` (`recommend.py:353-362`).

## Gaps

1. **MAJOR -- `physics/wind.py` docstring is stale** (L8-13). Magnitude path IS wired; only directional half remains deferred. Update docstring.
2. **MAJOR -- directional wind decomposition deferred.** `decompose_wind` + `balance_shift_pct` have no consumer in `src/`. VISION §10 calls for asymmetric headwind/tailwind/crosswind correction.
3. **MAJOR -- `weight_corners` is dead for v4 cars.** `recommend._cached_weights` (recommend.py:344-377) hardcodes uniform weights for BMW/Cadillac/Ferrari (3/5 fleet). Spec §6 ("weighted by each corner's TIME SENSITIVITY") satisfied only for Acura + Porsche today.
4. **MAJOR -- `wet_baselines` ignores per-car corpus baselines.** `wet_mode.py:74` calls `baselines_for(car)` with no frame, always returns cold-start defaults. The corpus-derived baselines (`model.resolved_baselines`) are never the input to `wet_baselines`.
5. **MAJOR -- `_score_breakdown_per_car` writes 0.0 for empty `state.states`** (score.py:709-711). Same value as a maximally-bad real prediction; optimizer cannot distinguish "no model coverage" from "actively predicted disaster".
6. **MINOR -- `_OBJECTIVE_CONFIDENCE_MULTIPLIER` magic numbers untested** (score.py:41-46). Sparse-vs-dense penalty test only asserts inequality, not exact multipliers.
7. **MINOR -- bottoming-penalty path lacks tests** (score.py:256-290). Both static + at-speed branches and the safety-floor / depth constants are uncovered.
8. **MINOR -- quali / wet integration through `score_setup` not tested end-to-end.** No test runs `score_setup(... quali=True)` and asserts divergence; no test exercises wet-quali precedence.
9. **MINOR -- `_AERO_CACHE` module-global never invalidated** (score.py:559-573).
10. **MINOR -- `_score_breakdown` and `score_setup` duplicate dispatch** (score.py:370-461).
11. **MINOR -- `wet_phase_weights("dry")` returns shared reference** (wet_mode.py:95); inconsistent with `quali_phase_weights()` always-fresh contract.
12. **MINOR -- magic `0.5` (grip L99) and `4.0` (aero_eff L166) violate VISION §3.**
13. **MINOR -- `wheelspin_scale_ms` and `aero_grip_baseline_g` never derived** (baselines.py:131, 136).
14. **MINOR -- wet thresholds (`0.7 / 0.3 / 0.05`) lack per-car overrides** (wet_mode.py:27-29).
15. **NONE -- VISION §6 lap-time isolation upheld.** `score.py:7-10` docstring contract; `tests/physics/test_score.py::test_score_setup_no_lap_time_reference` greps both `score.py` and `recommend.py`.

## Evidence

- **Test suite**: COULD NOT EXECUTE (Bash + PowerShell denied). Static review of relevant tests:
  - `test_score.py`: 17 unit tests covering all six sub-utilizations, aggregator phase weighting, sparse-marking invariant, sparse-vs-dense penalty, lap-time grep test. Does NOT cover bottoming penalty, exact multipliers, quali/wet via `score_setup`.
  - `test_wet_mode.py`: 7 tests covering classify thresholds, baseline scaling direction, wet phase-weight shift direction.
  - `test_quali_mode.py`: 6 tests covering normalisation invariant, scaling direction, fresh-dict-per-call.
  - `test_wind.py`: 18 tests across `decompose_wind` and `aero_wind_modifier`. Comprehensive at unit level; but `decompose_wind` is dead in `src/`.
- **Lint**: COULD NOT EXECUTE; static reading shows no obvious ruff violations.
- **Latest artefact**: `recommendations/bmw__spa_2024_up__20260505-200605.txt` -- header line 2 reads `Stint: quali (3-lap stint) (8.0 L fuel)`, confirming the `--quali` path renders end-to-end. Header line 3 reads `Conditions: AirTemp 19.3 C  TrackTemp 20.6 C  AirDensity 1.149 kg/m^3  Wind 3.0 m/s  Wetness 0.00`, confirming env (incl. wind, wetness) is captured. No wet-mode artefact exists in `recommendations/`.

## Recommended next actions

- Update `physics/wind.py` module docstring (L8-13) to reflect partial wiring.
- Fix `wet_baselines(car, regime)` (Gap #4) to accept a `CarBaselines` argument; thread `model.resolved_baselines` from `_conditions_adjusted_baselines`.
- Add integration test in `tests/physics/test_score.py` that calls `score_setup(... quali=True/False)` and asserts divergence; assert wet-quali precedence.
- Add tests for the bottoming-penalty branch (static + at-speed) and pin `_OBJECTIVE_CONFIDENCE_MULTIPLIER` exact values.
- Wire a v4-aware `weight_corners` (consuming `CornerScheduleEntry.archetype`) into `_cached_weights` for v4 cars.
- Either delete `decompose_wind` + `balance_shift_pct` OR wire them via per-corner heading captured in the corner schedule.
- Make `wet_phase_weights("dry")` return a fresh dict to match the `quali_phase_weights()` contract.
- Replace magic `0.5` (grip) and `4.0` (aero_eff) with per-car corpus-derived constants.
- Drop `wheelspin_scale_ms` / `aero_grip_baseline_g` from `CarBaselines` OR wire `derive_baselines` to populate them.
