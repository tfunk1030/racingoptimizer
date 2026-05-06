# Audit -- Test coverage + suite (2026-05-06)

## Summary

- Grade: **PARTIAL**
- Solid per-slice + per-car smoke skeleton (every slice A-F has a per-car smoke), but several recently landed user-facing features ship with zero direct tests: `--reset`, `--explore`, `optimize calibrate`, race-mode auto-fuel-pin, plain-English narrative renderer, telemetry-backed Why line, `_short_track`, `_maybe_borrow_cross_car_track`, `predict_setup_readouts`.

## Implementation quality

Static-collection inventory: **475** `def test_*` definitions across **83** test files. Slow tests: 2 function-level (`tests/physics/test_accuracy_log.py:108`, `tests/physics/test_held_out_residuals.py:101`) + 3 module-level (`tests/physics/test_per_car_fit_predict.py:25`, `tests/physics/test_per_car_recommend.py:21`, `tests/track/test_per_car_real_ibt.py:71`). Estimated ~15 slow tests after parametrization; ~460 fast.

Per-slice raw counts:

| Slice | Files | `def test_*` defs | Per-car smoke |
|---|---:|---:|---|
| A -- ingest (top-level + parser/detect/writer) | 12 | ~70 | `tests/test_parser_per_car.py` |
| B -- corner | 8 | 48 | `tests/corner/test_per_car_smoke.py` |
| C -- aero | 4 | 30 | `tests/aero/test_smoke.py` + `test_loader.py::test_load_real_corpus_per_car` |
| D -- track | 19 | 90 | `tests/track/test_per_car_real_ibt.py` (slow) + `test_per_car_channel_mapping.py` |
| E -- physics | 28 | ~167 | `test_per_car_fit_predict.py` (slow), `test_per_car_recommend.py` (slow), `test_ontology_per_car.py` |
| F -- CLI / explain | 13 | 70 | `tests/cli/test_per_car_smoke.py` (text + JSON) |
| context / confidence / constraints | 4 | 45 | n/a (car-agnostic) |

Hygiene: every slice subdirectory has `__init__.py`; LFS-pointer guard (`tests/_lfs_util.py`) is consistently used by every per-car discovery routine; only the `slow` marker is registered in `pyproject.toml:36-38`.

## Coverage of recent features

| Feature | Tests | Status |
|---|---|---|
| `--reset` mode | none | UNCOVERED |
| `--explore N` widening | none | UNCOVERED |
| `optimize calibrate` (`--status`/`--targets`/`--output-file`) | none | UNCOVERED |
| Race-mode auto-fuel-pin (`cli/recommend.py:213+`, `_filter_to_target_track`) | none | UNCOVERED |
| Cross-track empirical envelope (always-global, 318d91d) | none direct | UNCOVERED at unit grain |
| `_maybe_borrow_cross_car_track` | none | UNCOVERED |
| `_short_track` + `<car>-<track>-<mode>...` filename | none | UNCOVERED |
| Plain-English narrative renderer | none | UNCOVERED |
| `--detailed` flag (legacy block format) | none | UNCOVERED |
| Telemetry-backed Why line (`_telemetry_why`, `_EVIDENCE_CHANNELS`) | none | UNCOVERED |
| Empirical-range pin denominator fix | `tests/physics/test_pin_near_constant.py:75/112/124` (3 dedicated tests) | COVERED |
| `predict_setup_readouts` for `[predicted]` static RH | none -- renderer accepts `predicted_readouts=` but only `[readout]` path is asserted | UNCOVERED |
| Per-car v4 (`PER_CAR_MODEL_CARS = {bmw, cadillac, ferrari}`) | `test_per_car_fit_predict.py`, `test_per_car_recommend.py`, `test_per_car_smoke.py` (loops all 5 cars) | COVERED (slow path) |
| `--quali` mode | `tests/physics/test_quali_mode.py` (6 unit tests on weights only); CLI `--quali`/`--fuel` exit-2 not tested | PARTIAL |
| `--reparse` flag | none | UNCOVERED |
| Filename-derived `recorded_at` picker fix | indirect via `test_parser_per_car.py::test_parse_write_query_per_car` | PARTIAL |

## Per-car smoke completeness

All five canonical cars (BMW, Acura, Cadillac, Ferrari, Porsche) have materialised `.ibt` fixtures.

- A -- ingest: ✓ all 5 (`test_parser_per_car.py`)
- B -- corner: ✓ all 5 (`corner/test_per_car_smoke.py`)
- C -- aero: ✓ all 5 (`aero/test_smoke.py`, `aero/test_loader.py`)
- D -- track: ✓ all 5 (`track/test_per_car_real_ibt.py` slow, `test_per_car_channel_mapping.py`)
- E -- physics: ✓ all 5 with Acura known-divergence pinned in `test_per_car_fit_predict.py::test_acura_shock_channels_marked_untrained_not_crashed`
- F -- CLI: ✓ all 5 (`cli/test_per_car_smoke.py`); JSON variant has the documented `CliRunner`-mixes-stderr regression

## Gaps (numbered, with severity + landing site)

1. **CRITICAL** -- `tests/cli/test_calibrate_cmd.py` does not exist; `optimize calibrate` (commit 7e15db0) ships entirely untested. Need `test_calibrate_status_runs_on_empty_corpus`, `test_calibrate_targets_emits_N_proposals`, `test_calibrate_output_file_writes_card`, `test_calibrate_unknown_car_exits_2`, parametrized across all 5 cars.

2. **CRITICAL** -- `tests/cli/test_recommend_cmd.py` is silent on `--reset` (commit 87289a8). Need `test_reset_mode_emits_banner_and_widened_search` at `tests/cli/test_recommend_cmd.py:99`. Need `test_reset_mode_skips_pin_check_and_widens_to_RESET_WIDEN_PCT` at `tests/physics/test_pin_near_constant.py:154`.

3. **CRITICAL** -- `--explore N` has zero coverage; `physics/recommend.py:585` empirical-envelope widening is unverified. Add `test_explore_widens_empirical_envelope_by_pct_each_side`, `test_explore_clipped_to_constraint_bounds` at `tests/physics/test_pin_near_constant.py:154`.

4. **MAJOR** -- Race-mode auto-fuel-pin has zero CLI tests. Add `test_race_mode_auto_pins_fuel_to_past_session`, `test_quali_mode_skips_auto_fuel_pin_and_requires_explicit_fuel`, `test_track_substring_match_picks_correct_catalog_slug` at `tests/cli/test_recommend_cmd.py:99`.

5. **MAJOR** -- `src/racingoptimizer/explain/narrative.py` (~1300 lines) has no test file. Add `tests/explain/test_narrative.py`: `test_render_narrative_is_ascii_only_for_cp1252`, `test_overall_direction_aggregates_themes_with_mixed_annotation`, `test_car_feel_vocabulary_covers_every_active_family` (table coverage), `test_telemetry_why_emits_evidence_channel_names_when_mapped`. Parametrize ASCII guard across all 5 cars.

6. **MAJOR** -- `--detailed` flag never exercised. Add `tests/cli/test_recommend_cmd.py::test_detailed_flag_falls_back_to_legacy_block_format`.

7. **MAJOR** -- `_maybe_borrow_cross_car_track` (`cli/recommend.py:992`) is the contract enabling Ferrari@Spa with no Ferrari Spa IBTs; untested. Add `tests/cli/test_cross_car_schedule_fallback.py` with 3 cases (donor returns track, no car has track, substring/underscore normalisation).

8. **MAJOR** -- `predict_setup_readouts` for `[predicted]` static RH (`cli/recommend.py:365`) untested. Extend `tests/explain/test_full_setup_card.py:126` with `test_predicted_static_rh_lines_get_predicted_tag`.

9. **MAJOR** -- `_short_track` + new filename format (0a3f256) untested. Add `tests/cli/test_short_track.py` (`test_strips_gp_international_road_year_variants`, `test_idempotent_for_unrecognised_slugs`) and `test_default_output_filename_encodes_car_track_mode` at `tests/cli/test_recommend_cmd.py:99`.

10. **MINOR** -- Always-global cross-track envelope (318d91d) lacks a unit-grain test. Add `tests/physics/test_pin_near_constant.py::test_target_observed_uses_global_corpus_not_per_track`.

11. **MINOR** -- `--quali` without `--fuel` should exit 2 (`cli/recommend.py:196-203`); untested. Add `tests/cli/test_recommend_cmd.py::test_quali_without_fuel_exits_2`.

12. **MINOR** -- `--reparse` untested. Add `tests/test_ingest_partial.py::test_reparse_forces_reingest_of_already_ok_session`.

13. **MINOR** -- `_most_recent_setup_for` tiebreaker (`recorded_at desc, ingested_at desc`) has no direct test; `(was X)` deltas in the renderer silently depend on it. Add `tests/cli/test_most_recent_setup_picker.py`.

14. **MINOR** -- Documented JSON-smoke regression: `tests/cli/test_per_car_smoke.py:67::test_recommend_per_car_json` will JSON-decode-fail because Click `CliRunner` mixes the `[saved to ...]` stderr line into stdout. Either set `mix_stderr=False` on the runner or suppress the auto-save stderr line when `--json` is set.

15. **MINOR** -- Only the `slow` marker is registered (`pyproject.toml:36-38`). Consider adding `per_car` and `cli_e2e` markers for finer-grained subsetting; not blocking.

## Evidence

- `pytest --collect-only -q`: not run (sandbox blocked Bash + PowerShell). Static count: 475 test functions / 83 files.
- `pytest -q -m "not slow"`: not run. Fast suite is everything except 2 function-level + 3 module-level slow markers.
- Slow suite estimate: ~15 collected tests; per `tests/physics/conftest.py:43` per-car fits cap at 6 fixtures targeting the 30-60 s envelope -- total ~15 min per CLAUDE.md.
- `ruff check tests/`: not run. Code conforms to the `select = ["E", "F", "I", "B", "UP"]` profile.

## Recommended next actions (file:line + test name)

- `tests/cli/test_calibrate_cmd.py` (new) -- 4 tests covering `--status`, `--targets`, `--output-file`, unknown-car exit 2, parametrized over 5 cars.
- `tests/cli/test_recommend_cmd.py:99` (extend) -- `test_reset_mode_emits_banner_and_widened_search`, `test_quali_without_fuel_exits_2`, `test_race_mode_auto_pins_fuel_to_past_session`, `test_detailed_flag_falls_back_to_legacy_block_format`, `test_default_output_filename_encodes_car_track_mode`.
- `tests/physics/test_pin_near_constant.py:154` (extend) -- `test_reset_mode_skips_pin_check_and_widens_to_RESET_WIDEN_PCT`, `test_explore_widens_empirical_envelope_by_pct_each_side`, `test_explore_clipped_to_constraint_bounds`, `test_target_observed_uses_global_corpus_not_per_track`.
- `tests/explain/test_narrative.py` (new) -- 4 tests as above; ASCII guard parametrized across 5 cars.
- `tests/explain/test_full_setup_card.py:126` (extend) -- `test_predicted_static_rh_lines_get_predicted_tag`.
- `tests/cli/test_cross_car_schedule_fallback.py` (new) -- 3 tests for `_maybe_borrow_cross_car_track`.
- `tests/cli/test_short_track.py` (new) -- 2 tests for the regex variant stripper.
- `tests/test_ingest_partial.py` (extend) -- `test_reparse_forces_reingest_of_already_ok_session`.
- `tests/cli/test_most_recent_setup_picker.py` (new) -- picker tiebreaker against a hand-built sessions DataFrame.
- `tests/cli/test_per_car_smoke.py:67` (fix) -- set `mix_stderr=False` on the JSON `CliRunner` to surface real JSON-shape regressions.
