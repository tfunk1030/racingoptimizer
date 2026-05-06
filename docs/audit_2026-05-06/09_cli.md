# Audit -- Slice F2: CLI surface (2026-05-06)

## Summary
- Grade: PARTIAL
- The CLI surface works end-to-end and routes positional shorthands cleanly into a single `recommend_cmd`, but `optimize calibrate` ships with zero tests, the `--json` stderr-mixing bug from CLAUDE.md is still latent in production code (auto-save banner unconditionally writes to stderr even under `--json`), and the new `<car>-<track>-<mode>-<MMDD>-<HHMM>` filename format is hard-coded in two separate places (recommend.py:407-424 and calibrate.py:346-380) with stale docstrings.

## Implementation quality

**`src/racingoptimizer/cli/__init__.py`**
- `_OptimizeGroup.parse_args` (lines 32-45) prepends `"recommend"` when the first arg is a canonical car key OR a path ending in `.ibt` that exists. Uses `Path.exists()` so a typo like `optimize foo.ibt` falls through to Click's normal command lookup.
- `main` is `invoke_without_command=True` and prints `--help` when no subcommand is supplied (lines 50-53).
- Consistency nit: car-membership check uses `first.strip().lower()` (line 38) but the path branch uses raw `first` (line 42). Acceptable on Windows because `Path.exists()` is case-insensitive there.

**`src/racingoptimizer/cli/recommend.py`** (1629 lines)
- File carries `recommend_cmd`, `compare_cmd`, `status_cmd`, AND every helper that `calibrate.py` imports -- it's the de-facto helpers module. Splitting into `cli/_helpers.py` would let calibrate stop reaching into a sibling command's private API.
- `recommend_cmd` (lines 65-434) is ~370 lines for one Click callback: race-mode auto-fuel-pin (lines 213-242), per-car vs v3 fork (lines 255-304), and briefing assembly (lines 339-397) are all inline and would each clean up as helpers.
- `CANONICAL_CARS` (line 47) is duplicated by `tests/cli/conftest.py:PER_CAR_FIXTURES` (line 15) and by the hardcoded loop order in `_maybe_borrow_cross_car_track` (line 1011) -- three sources of truth for the car list.
- `PER_CAR_MODEL_CARS = frozenset({"cadillac", "bmw", "ferrari"})` (line 57) -- the v3/v4 routing key. Adding a car requires constraints.md edit + ontology check + cache invalidation, but the constant has no inline cross-reference back to the procedure.
- `_short_track` regex (line 624) `_(gp|international|road|2\d{3}.*)$` won't strip standalone `_oval` / `_short` / `_legends`. Best-effort pretty-printer.
- Slug-resolution duplicated 4x: `_filter_to_target_track` (lines 685-712), `_resolve_track_or_extrapolate` (lines 762-813), inline block in `_build_per_car_pipeline` (lines 924-948), and `_maybe_borrow_cross_car_track` (lines 1018-1029). One sorts `available` alphabetically before substring scanning, others don't -- drift already.
- File-output formatting duplicated:
  - `recommend.py:407-424` writes `<car>-<short_track>-<mode>[-<fuel>L]-<MMDD>-<HHMM>.<ext>`
  - `calibrate.py:346-380` writes `<car>-<short_track>-<cal[-status]>-<MMDD>-<HHMM>.txt`
- Stale docstrings:
  - `recommend.py:401` claims `<car>_<track>_<mode>[_<fuel>L]_<YYYY-MM-DD>_<HHMM>` (old format) but code at line 422 uses the new dash+MMDD format.
  - `calibrate.py:167` (user-visible `--output-file` help) claims the same old format.
- `_approximate_clean_corner_phases` multiplies by `30` (line 1572) -- undocumented heuristic, acknowledged in docstring.
- `_status_notes` (lines 1622-1626) hardcodes the missing-constraint families list. Stale: damper bounds DO exist now per constraints.md. Test `test_status_cmd.py:60::test_status_notes_only_list_unbounded_families` pins `"dampers" in note`, locking in the bug.
- `_safe_lap_data` (lines 1356-1381) catches broad `Exception` twice with no logging -- silent failure path.
- `_apply_pins_to_constraints` (line 864) reaches into `table._by_car` directly with `# noqa: SLF001` (line 875). Acknowledged leak.
- `_constraints_fingerprint` (line 1195) imports `_default_constraints_path` from `racingoptimizer.constraints.loader` -- semi-private cross-module access.

**`src/racingoptimizer/cli/calibrate.py`** (606 lines)
- Local-import block (lines 195-204) pulls eight private symbols from `racingoptimizer.cli.recommend`. Comment at line 193 frames as startup-cost optimisation but the actual cause is layering.
- `_render_calibration_card` (line 545) builds a `SetupRecommendation` with `probe_conf = Confidence(value=0.0, lo=0.0, hi=0.0, n_samples=0, regime="noisy")` -- `regime="noisy"` with `n=0` is technically valid but `sparse` would be more honest for "no data".
- No tests at all.

**`src/racingoptimizer/ingest/cli.py`** (38 lines) -- clean, single command.

**`src/racingoptimizer/ingest/detect.py`**
- `CAR_PREFIX_MAP` line 23: `"amvantageevogt3": "bmw"` with a placeholder comment. An Aston Martin GT3 IBT will be silently routed to BMW's per-car fitter (BMW is in `PER_CAR_MODEL_CARS`), poisoning the BMW corpus. Should raise `UnknownCarError`.
- `_FILENAME_RE` (line 53) assumes single-space-before-date; weird filenames lose the date but still match car/track.

**`src/racingoptimizer/ingest/paths.py`** (38 lines) -- clean.

## Wiring

**Positional dispatch:**
```
optimize <args>
  -> _OptimizeGroup.parse_args (cli/__init__.py:35)
       if first in CANONICAL_CARS -> args = ["recommend", *args]
       elif first.suffix==".ibt" and exists -> args = ["recommend", *args]
  -> super().parse_args(ctx, args)
  -> recommend_cmd (cli/recommend.py:163)
```

**`recommend_cmd` flow:**
1. `_resolve_car_track_or_exit` (lines 641-682) -- IBT-path branch sniffs car+track via `detect_car_from_filename` / `detect_track_from_filename`; pair-form uses `_resolve_car_or_exit`.
2. `resolve_corpus_root` (paths.py:23) -> `_safe_sessions(car_key)` (line 751) -> catalog DataFrame (already filtered to this car).
3. `--quali` requires `--fuel` (lines 196-203, exit 2).
4. `_parse_pins` (lines 834-861) merges `--wing`, `--fuel`, `--pin K=V` into a flat dict.
5. **Race-mode auto fuel pin** (lines 213-242): not-quali AND no fuel pin -> `_filter_to_target_track(catalog_sessions, track)` then `_most_recent_setup_for(target_subset)` then `setup_value(car_key, "fuel_level_l", past_setup)`. Stderr banner on success.
6. `_apply_pins_to_constraints` (line 244) narrows the constraints table to `[v, v]` per pinned key.
7. **Routing fork** (line 255):
   - `car in PER_CAR_MODEL_CARS`: `_build_per_car_pipeline` returns `(track_slug, sessions_for_target, schedule, model)`. If the requested car has no sessions on the track, `_maybe_borrow_cross_car_track` walks `("bmw", "cadillac", "ferrari", "porsche", "acura")` to borrow corner geometry. Model from `_build_or_load_per_car_model` (lines 1035-1075) -> `corpus/models/<car>__per-car__<digest>.pickle`.
   - else: `_resolve_track_or_extrapolate` returns `(target_slug, donor_slug)`; v3 donor extrapolation. Model from `_build_or_load_model` (lines 1093-1136) -> `corpus/models/<car>__<track>__<digest>.pickle`.
8. `_env_from_overrides` -> `_environment_from_corpus` walks valid laps, filters on `data_quality_mask`, takes per-sample medians (circular for `WindDir`).
9. `model.recommend(...)` -> SetupRecommendation.
10. `_post_clamp` (lines 1399-1462) clamps + rounds discrete params + accumulates warnings.
11. `_force_sparse_regime` (line 309) only on donor extrapolation.
12. `build_justifications` -> `render_recommendation_text` or `render_narrative` (default; `--detailed` selects the legacy block format).
13. **File output** (lines 407-434): default `recommendations/<car>-<short_track>-<mode>[-<fuel>L]-<MMDD>-<HHMM>.<ext>`, mode in `{race, quali, reset}`. `output_file == "-"` opts out.

**`calibrate_cmd` reuse:**
- Same routing fork as recommend (calibrate.py:215-233) using imports from recommend.py.
- Status-only mode stops after the coverage table (lines 277-281).
- Default mode picks `n_targets` thin-variance parameters in `_pick_targets` (lines 441-510), constructs synthetic SetupRecommendation in `_render_calibration_card` (lines 545-605), feeds `render_full_setup_card`.
- `_maybe_save` (lines 346-380) is a parallel saver implementation; re-imports `_short_track` from recommend.

**Cache-key fingerprint (`_model_cache_parts` lines 1165-1192):**
- `"|".join(sorted(session_ids))` -- order-independent
- ontology fingerprint with `(name, family, fittable, user_settable, json_path)` per spec
- `f"constraints={sha256(constraints.md)[:16]}"`
- `f"fitters_layout={FITTERS_LAYOUT_VERSION}"`
- caller appends `f"schema={ENV_FEATURE_SCHEMA_VERSION}"` (or `_PER_CAR`)
- final sha256, take 16 hex chars.

Matches the CLAUDE.md spec; the `path=` segment is the recently-added critical fix.

**Race fuel auto-pin path (recommend.py:213-242):** filters by car (already done by `_safe_sessions`), substring-matches user's track to catalog slug, picks most-recent setup, calls `setup_value(car_key, "fuel_level_l", past_setup)`. KeyError silently drops the pin (legacy schema where `BrakesDriveUnit.Fuel.FuelLevel` doesn't resolve) -- no user-visible note.

## Gaps

1. **MAJOR -- `optimize calibrate` has zero tests.** `tests/cli/` contains no `test_calibrate*.py`. The command has flag combinations (`--status`, `--targets N`, `--output-file -`), non-trivial `_pick_targets` logic with second-largest-gap fallback (calibrate.py:486-501), synthetic SetupRecommendation construction (`_render_calibration_card` line 545), and shares the per-car/v3 fork with recommend. Any regression ships unverified. **File:** entire `src/racingoptimizer/cli/calibrate.py` (606 lines).

2. **MAJOR -- `--json` stderr-mixing bug still latent in production.** `recommend.py:407-434` unconditionally emits `\n[saved to <path>]` to stderr (line 429) even under `--json`. Real production bug too: a user piping to `jq` with stderr merged into stdout will see the banner appended to JSON. Fix: when `as_json` AND `output_file is None`, default `output_file = Path("-")` so the saver block is skipped under JSON mode.

3. **MAJOR -- Slug-resolution logic duplicated 4x.** `_filter_to_target_track` (recommend.py:685-712), `_resolve_track_or_extrapolate` (lines 762-813), inline in `_build_per_car_pipeline` (lines 924-948), and `_maybe_borrow_cross_car_track` (lines 1018-1029). One unified `_match_track_slug(needle, available)` would prevent drift.

4. **MAJOR -- File-output format duplicated and docstrings stale.** `recommend.py:407-424` writes new dash+MMDD format but docstring at line 401 describes old underscore+full-date format. Same staleness in calibrate.py:167 user-visible help text.

5. **MAJOR -- Aston Martin GT3 silently mapped to BMW.** `ingest/detect.py:23` `"amvantageevogt3": "bmw"`. AMV GT3 IBTs ingested via `optimize learn` will pollute the BMW per-car corpus.

6. **MINOR -- `_status_notes` is hardcoded and stale.** `recommend.py:1622-1626` claims dampers lack constraints; per-corner damper bounds exist now. Test `test_status_cmd.py:60` pins the stale text.

7. **MINOR -- `calibrate.py` reaches into recommend.py private symbols.** Eight underscore imports at calibrate.py:195-204; `_short_track` re-imported at line 364.

8. **MINOR -- `CANONICAL_CARS` defined three places.** `recommend.py:47`, `tests/cli/conftest.py:PER_CAR_FIXTURES`, hardcoded loop in `_maybe_borrow_cross_car_track` (line 1011).

9. **MINOR -- `_apply_pins_to_constraints` accesses `ConstraintsTable._by_car`.** recommend.py:875 with `# noqa: SLF001`. A `with_pin(...)` factory on `ConstraintsTable` would remove the noqa.

10. **MINOR -- Race fuel auto-pin silently drops on KeyError.** `recommend.py:231-236` swallows `setup_value` KeyError, no stderr trail. A "could not auto-pin fuel: legacy schema" line would help debug.

11. **MINOR -- `_OptimizeGroup.parse_args` doesn't handle `<ibt_path>` with extra positionals.** `optimize ./foo.ibt ./bar.ibt` silently routes the second IBT into `track` arg.

12. **MINOR -- `_safe_lap_data` swallows all `Exception` twice with no logging.** recommend.py:1372-1373 and 1378-1379. Loses diagnostic context for corrupted parquet files.

13. **MINOR -- `_short_track` regex is variant-list-coupled.** recommend.py:624 hardcodes `(gp|international|road|2\d{3}.*)`. New variants won't strip.

14. **NONE -- `_environment_from_corpus` correctly aggregates per-sample.** Solid; verified by `tests/cli/test_environment_from_corpus.py` (bimodal density, mask filtering, circular wind-direction medians).

## Evidence
- Test suite: NOT COLLECTED -- Bash and PowerShell tools were sandbox-denied for `uv run pytest` in this audit run. Static review covered every test under `tests/cli/` (10 files: `conftest.py`, `test_compare_cmd.py`, `test_environment_from_corpus.py`, `test_golden_files.py`, `test_model_cache_path.py`, `test_per_car_smoke.py`, `test_post_clamp_discrete.py`, `test_recommend_cmd.py`, `test_status_cmd.py`, `test_untrained_track.py`) plus `tests/test_cli.py`.
- Lint: NOT COLLECTED -- same restriction. `recommend.py:875` carries `# noqa: SLF001`; no other suppressions visible by inspection.
- Latest artefacts in `recommendations/`:
  - **NEW format** (dash + MMDD): `bmw-spa-cal-0506-1015.txt`, `bmw-spa-cal-status-0506-1018.txt`, `bmw-spa-reset-0506-1029.txt`, `ferrari-spa-race-0506-1026.txt`. Mode tags `cal`, `cal-status`, `reset`, `race` confirm the calibrate, --reset, and race-default paths use the new naming.
  - **OLD format** (double-underscore + full date): `bmw__spa_2024_up__20260506-053645.txt` (most recent old-format file, written today before the refactor took effect). Bulk of pre-2026-05-06 files: `cadillac__lagunaseca__20260505-193502.txt`, `ferrari__hockenheim_gp__20260505-193514.txt`, `porsche__algarve_gp__20260505-193538.txt`.

## Recommended next actions
- Add tests for `optimize calibrate` covering: status-only output, target proposal selection, `--targets 1`/`--targets 5`, second-largest-gap fallback, the per-car vs v3 fork, and the `output_file == "-"` opt-out.
- Fix the `--json` auto-save stderr bug: when `as_json AND output_file is None`, default `output_file = Path("-")` so the saver block is skipped. Also flip `tests/cli/test_per_car_smoke.py::test_recommend_per_car_json` runner to `mix_stderr=False` for defence in depth.
- Extract `cli/_helpers.py` containing `_short_track`, `_resolve_car_or_exit`, `_resolve_car_track_or_exit`, `_filter_to_target_track`, `_most_recent_setup_for`, `_safe_sessions`, a new unified `_match_track_slug`, and `_default_artefact_path(car, track, suffix, ext)`. Stop calibrate from importing private symbols from a sibling command.
- Update stale docstrings at `recommend.py:401` and `calibrate.py:167`.
- Fix Aston Martin mapping in `ingest/detect.py:23`.
- Refresh `_status_notes` and the corresponding test assertion.
- Fold `CANONICAL_CARS` to a single source.
