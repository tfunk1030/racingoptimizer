# Audit -- Slice E1: Fitter / Model (2026-05-06)

## Summary
- Grade: **PARTIAL.**
- Code structure, pickle compat, and recent-fix wiring (`per_track_parameter_observed`, `parameter_observed_std`, `predict_setup_readouts`, `__setstate__` backfills, bounded `fit_quality`) are all solid. The blocker: `fit_per_car()` (v4) -- the path that ships to bmw / cadillac / ferrari users -- has zero direct test coverage. `tests/physics/test_per_car_fit_predict.py` is named for the v4 path but its `tests/physics/conftest.py:160` factory calls `fit()` (v3).

## Implementation quality

- Per-car v4 (`fit_per_car`) vs per-(car,track) v3 paths: dispatch lives in `cli/recommend.py:1057,1061` (per-car) and `cli/recommend.py:1093` (v3); `PER_CAR_MODEL_CARS` hardcoded at `cli/recommend.py:57`.
- `PhysicsModel.per_track_parameter_observed` field (added for trust radius): populated at `physics/fitter.py:845`. Consumed by `physics/recommend.py:107` and `cli/calibrate.py:244`.
- `PhysicsModel.parameter_observed_std` field (added for pin denominator): populated at `physics/fitter.py:391, :841`. Consumed by `physics/recommend.py:118-120`.
- `__setstate__` backfill defaults for legacy pickles: `physics/model.py:167-196` -- handles list / tuple / dict states; defaults `feature_schema_version`, `car_baselines`, `parameter_observed_std`, `per_track_parameter_observed`. Documented well.
- `predict_setup_readouts` (used by full_setup_card for `[predicted]` static RH tags): only caller `cli/recommend.py:365`.
- `FITTERS_LAYOUT_VERSION` and `ENV_FEATURE_SCHEMA_VERSION[_PER_CAR]` cache versioning: read by `cli/recommend.py:1179,1191`, folded into both v3 + v4 cache keys.
- `io_log` fit_quality scoring (signal/(signal+cv_residual), bounded [0,1]): clean implementation, NaN check at `io_log.py:196` uses `noise_ratio == noise_ratio` idiom.
- ~120-line copy-paste between `fit` (`fitter.py:214-422`) and `fit_per_car` (`fitter.py:626-868`) -- only three real differences (group-by key, `family_kind="rf"` force, per-track observed build) but full body duplicated. Maintenance ceiling.
- `_assemble_feature_row` / `_v4` near-duplicates: same structural duplication.
- `FitRecord.__setstate__` / `PhysicsModel.__setstate__` near-duplicates of the same defensive idiom.
- Unused `n_features` kwargs in GP kernels.
- Lazy in-function imports of `compute_dynamic_at_speed_rh` (no real cycle).
- Magic `30 m/s` floor in `dynamic_at_speed.py:166`.
- Sentinel `corner_id=-1` and track `"<per-car>"` not promoted to named constants.

## Wiring

- `fit_per_car` callers: `cli/recommend.py:1057,1061` only.
- `per_track_parameter_observed` readers: `physics/recommend.py:107`, `cli/calibrate.py:244`.
- `parameter_observed_std` readers: `physics/recommend.py:118-120`.
- `predict_setup_readouts` callers: `cli/recommend.py:365` only.
- `FITTERS_LAYOUT_VERSION` readers: `cli/recommend.py:1179,1191` (folded into both v3 + v4 cache keys).

Archetype-feature provenance asymmetry: `_attach_corner_archetypes` (training, per-session window) vs `build_corner_schedule` (predict, per-track group) compute the same five archetype fields with different semantics -- predict-time row distribution may diverge from training.

## Gaps

1. **MAJOR -- `fit_per_car` untested** (only end-to-end CLI smoke covers it; conftest factory calls `fit()` instead of `fit_per_car()`).
2. **MAJOR -- ~120-line copy-paste between `fit` (`fitter.py:214-422`) and `fit_per_car` (`fitter.py:626-868`)** -- only three real differences but full body duplicated.
3. **MAJOR -- `corner_schedule.build_corner_schedule` has no tests** despite being load-bearing for every per-car prediction.
4. **MAJOR -- Archetype-feature provenance asymmetry**: `_attach_corner_archetypes` (training, per-session window) vs `build_corner_schedule` (predict, per-track group) compute the same five archetype fields with different semantics -- predict-time row distribution may diverge from training.
5. **MINOR -- unused `n_features` kwargs in GP kernels.**
6. **MINOR -- lazy in-function imports of `compute_dynamic_at_speed_rh`** (no real cycle).
7. **MINOR -- magic `30 m/s` floor in `dynamic_at_speed.py:166`.**
8. **MINOR -- NaN check `noise_ratio == noise_ratio` in `io_log.py:196`** -- works but `math.isnan` is more readable.
9. **MINOR -- sentinel `corner_id=-1` and track `"<per-car>"` not promoted to named constants.**
10. **MINOR -- `_assemble_feature_row` / `_v4` near-duplicates.**
11. **MINOR -- `FitRecord.__setstate__` / `PhysicsModel.__setstate__` near-duplicates.**
12. **MINOR -- `predict_setup_readouts` and `compute_dynamic_at_speed_rh` untested.**

## Evidence

- Test suite: NOT RUN (sandbox blocked Bash/PowerShell). Static review confirms `test_per_car_fit_predict.py` calls `fit()` via conftest factory, not `fit_per_car()` -- v4 path direct unit coverage is zero.
- Lint: NOT RUN.
- Latest artefact: `recommendations/bmw__spa_2024_up__20260506-053645.txt` lines 204, 217, 231 show `[predicted]` tags on static ride-height rows -- confirms `predict_setup_readouts` flows to renderer.

## Recommended next actions

**P0:**
- Add a v4-targeted fixture to conftest that exercises `fit_per_car()` end-to-end on a small subset.
- Extract `_train_joint(...)` to dedupe `fit` / `fit_per_car`.

**P1:**
- Test `build_corner_schedule` against synthetic + real per-car corpora.
- Promote `_PER_CAR_LOG_*` constants and the sentinel `corner_id=-1` / track `"<per-car>"` to named constants.
- Test `predict_setup_readouts`.

**P2/P3:**
- Clean up the minor items: unused kwargs, magic numbers, NaN idioms, near-duplicates.
