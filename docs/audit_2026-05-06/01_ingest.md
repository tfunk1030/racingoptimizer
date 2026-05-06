# Audit -- Slice A: Ingest (2026-05-06)

## Summary
- Grade: PASS
- Slice A is the most mature module in the repo: status semantics, audit-trail, sample-rate detection, and per-car parser smoke are all in place; only minor cleanups remain (one stale placeholder car mapping, a private helper imported across modules, and an open question about catalog migration discipline).

## Implementation quality

### `parser.py` (355 lines)
- VISION 1 audit trail honoured: every drop is recorded with a human reason in `dropped_channels` (`parser.py:256-272`), with two distinct categories -- pattern exclusion (`EXCLUDED_CHANNEL_PATTERNS` at `parser.py:28-33`) and multi-element scalar arrays (`parser.py:263-266`).
- `recorded_at` resolution order is correct and well-documented at `parser.py:236-251`: filename-derived datetime from `_filename_recorded_at` (`parser.py:201-218`) wins, with `WeekendOptions.Date` as documented fallback. The fix-rationale comment explicitly calls out the regression it solves (per-weekend duplicates from scheduled-event date).
- Sample-rate detection (`_detect_sample_rate`, `parser.py:153-198`) is a clean four-stage fallback chain (`header.tick_rate` -> disk-header back-compute -> YAML `SessionTickRate` -> 60 Hz default), each gated on `> 0` so a zeroed field cannot poison the time axis. `DEFAULT_SAMPLE_RATE_HZ` is module-level and documented (`parser.py:35-38`).
- `TrackWetness` enum->fraction conversion (`parser.py:75-88`) clips to `[0,7]` before lookup so a malformed IBT cannot inject NaN. Lookup table `_TRACK_WETNESS_ENUM_TO_FRACTION` is documented with the iRacing SDK source (`parser.py:41-72`); chosen breakpoints align with `physics.wet_mode.classify_conditions` thresholds -- boundary alignment called out in the comment.
- `_read_yaml` (`parser.py:108-133`) replicates pyirsdk 1.3.5's IBT YAML extraction. The regex `re.sub(r"(\w+: )(,.*)", r'\1"\2"', yaml_src)` is non-obvious -- it quotes a known iRacing YAML quirk where leading-`,` values would otherwise fail to load -- would benefit from a one-line comment.
- Required-channel guard (`parser.py:285-287`) raises if `LapDistPct` or `Lap` is missing -- propagates to `_process_one` as `failed`, which is the right outcome.
- `_summarize_weather` (`parser.py:312-351`) covers all 12 VISION 10 channels; the inconsistent naming (iRacing's per-sample channel `Precipitation` vs. the spec alias `PrecipType`) is noted in the comment at `parser.py:344-348` so the summary key stays stable for downstream consumers.

### `api.py` (288 lines)
- `_process_one` (`api.py:185-288`) is the heart of the salvage logic. Three explicit stages are commented (read bytes -> parse -> detect -> write), each fault flowing into `_record_failure` (`api.py:159-182`) with structured exception text. Stages match the docstring's status semantics.
- `--reparse` short-circuit override at `api.py:224` correctly suppresses the early return for `status="ok"` rows.
- `learn`/`sessions`/`laps`/`lap_data` all use `resolve_corpus_root(...)` (`api.py:44`, `63`, `94`, `132`) so corpus location can be overridden per-call without leaking into module state.
- `lap_data` opens the catalog only to look up the parquet path + lap span, then drops the connection before scanning parquet (`api.py:133-147`) -- clean separation.
- `_iter_ibt_paths` (`api.py:152-156`) walks recursively via `rglob("*.ibt")` per the data-asset commitment.
- `sessions(...)` exposes `ingested_at` as a column (`api.py:72`) -- the cross-slice contract that `_most_recent_setup_for` depends on for tiebreaking.

### `catalog.py` (215 lines)
- Schema is small and explicit (`catalog.py:16-49`); foreign key on `laps.session_id` is enforced (`PRAGMA foreign_keys = ON` at `catalog.py:106`).
- Additive-migration ladder via `_ADDITIVE_SESSION_COLUMNS` (`catalog.py:84-87`) plus the in-place `ALTER` loop in `init_schema` (`catalog.py:90-96`) lets older corpora keep working -- well-suited to the "add a column, don't break a 5 GB corpus" cadence.
- Explicit `_SESSION_SELECT_COLS` constant (`catalog.py:171-175`) defends against `SELECT *` ordering drift; the comment at `catalog.py:170` explicitly pins this to `SessionRow` field order.
- `query_sessions` (`catalog.py:178-200`) uses parameterised SQL throughout -- no f-string injection risk.
- `valid_only` predicate is `status IN ('ok','partial')` (`catalog.py:194-195`) -- keeping partial rows in the queryable set is correct per the lose-nothing rule.

### `detect.py` (82 lines)
- `CAR_PREFIX_MAP` (`detect.py:15-25`) uses longest-prefix match, ordered by `sorted(..., key=len, reverse=True)` (`detect.py:40`) -- correct.
- `_FILENAME_RE` (`detect.py:53-55`) makes the trailing `<date> <time>` group optional, so filenames without timestamps (older exports, manual renames) still detect car/track.
- `detect_car` (`detect.py:73-82`) prefers YAML over filename; falls back gracefully when YAML returns an unknown identifier.

### `writer.py` (146 lines)
- `_build_dataframe` (`writer.py:32-59`) builds `t_s` from the per-IBT `sample_rate_hz` (`writer.py:36`), so non-60 Hz IBTs get the correct time axis (regression-tested at `tests/test_writer.py:173-208`).
- `data_quality_mask` reservation (`writer.py:46-47`) is the documented Slice D handshake -- bool column initialised to all-True so Slice D can flip dirty samples in-place without rewriting the schema.
- `_lap_rows` (`writer.py:62-86`) computes `best=1` once, with a deterministic tiebreak on lower `lap_index` (`writer.py:84`) -- important for reproducibility of the renderer's "(was X)" deltas.

## Wiring

### Imports into ingest (call-site edges)
- `cli/__init__.py:29` -- `from racingoptimizer.ingest.cli import learn_command`
- `cli/recommend.py:37` -- `from racingoptimizer.ingest import api as ingest_api` (used in `_safe_sessions` at `recommend.py:751-759`)
- `cli/recommend.py:38-44` -- `detect_car_from_filename`, `detect_track_from_filename`, `slugify_track`, `UnknownCarError`, `normalize_car_key`
- `cli/recommend.py:45` -- `from racingoptimizer.ingest.paths import resolve_corpus_root`
- `cli/recommend.py:1467` -- `from racingoptimizer.ingest.writer import session_id_from_bytes` (lazy import inside a function)
- `cli/calibrate.py:31` -- `resolve_corpus_root` plus a re-import of `_filter_to_target_track` / `_most_recent_setup_for` from `recommend.py` at line 199
- `corner/states.py:39-44` -- catalog rows + `lap_data` + `normalize_car_key` + `parquet_path`/`resolve_corpus_root`
- `corner/config.py:12` -- `normalize_car_key`
- `track/{builder,corners,corner_loading,geometry,paths,rewrite}.py` -- multiple imports of `ingest.api`, `ingest.catalog`, `ingest.paths`
- `physics/{fitter,corner_schedule,dynamic_at_speed,weights}.py` -- every fitting + scoring path goes through the catalog + parquet read
- `constraints/clamp.py:18` -- `normalize_car_key`

### `ingest.api.sessions` -> `_filter_to_target_track` / `_most_recent_setup_for`
- `cli/recommend.py:226-229` -- race-mode auto-fuel-pin: pulls `catalog_sessions = ingest_api.sessions(car=...)`, narrows via `_filter_to_target_track` (substring matching against catalog slugs), picks via `_most_recent_setup_for` (sorts `recorded_at desc` with `ingested_at desc` tiebreaker)
- The `_most_recent_setup_for` docstring (`recommend.py:725-734`) explicitly documents the `recorded_at` parser fix -- cross-slice contract honoured.
- `cli/recommend.py:345-350` -- same picker reused inside the recommend pipeline
- `cli/calibrate.py:239-242` -- calibrate command re-uses both helpers identically

### Internal ingest edges (within the slice)
- `api.py` -> `parser.parse_ibt`, `writer.write_session`, `catalog.{open_catalog,upsert_session,get_session,query_sessions,get_laps}`, `detect.{detect_car,detect_car_from_filename,detect_track_from_filename,slugify_track,UnknownCarError}`, `paths.{catalog_path,parquet_path,resolve_corpus_root}`
- `writer.py` -> `catalog.{LapRow,SessionRow,insert_laps,upsert_session}`, `detect.{detect_car,detect_car_from_filename,slugify_track}`, `parser.ParseResult`, `paths.parquet_path`
- `parser.py` -> `segment.{LapSpan,detect_lap_boundaries}`

## Gaps

1. **MINOR -- `detect.py:23` placeholder mapping for `amvantageevogt3 -> bmw`.** Inline comment flags it as a placeholder. Today it would silently mis-attribute Aston Martin GT3 sessions to the BMW per-car model and BMW constraints. Either remove the entry (so detection raises `UnknownCarError`) or carry it through to a real per-car key.
2. **MINOR -- `_now_iso` is private but cross-imported.** Defined at `writer.py:28-29` and imported across modules by `api.py:24`. Promoting it to `paths.py` (or a tiny `_time.py`) would drop the leading-underscore-on-cross-module-import code-smell.
3. **MINOR -- `query_sessions` ORDER BY hardcoded to `recorded_at` (`catalog.py:198`)** with no secondary key, so two sessions sharing a `recorded_at` come back in undefined order. Add `, session_id` as a stable tiebreaker.
4. **MINOR -- `paths.default_corpus_root` (`paths.py:16-20`) hardcodes `parents[3]`.** If `paths.py` is ever moved, the corpus default silently moves with it. Worth a unit test that asserts the resolved default ends in `<repo>/corpus`.
5. **MINOR -- `_ADDITIVE_SESSION_COLUMNS` ladder is one-way (`catalog.py:84-96`).** No down-migration / no recorded schema-version. After several additions the implicit ordering between `SessionRow` field order and added columns becomes brittle. A `PRAGMA user_version` plus a single migration table would future-proof this.
6. **MINOR -- `EXCLUDED_CHANNEL_PATTERNS` filters by substring** (`parser.py:104-105`), so any future channel containing `"CarIdx"` would be unintentionally dropped. Anchoring to `startswith` for the `CarIdx` family would tighten the filter.
7. **NONE / informational -- JSON columns (`weather_summary`, `setup`, `dropped_channels`)** are stored as opaque `TEXT`. SQLite's JSON1 functions would let downstream queries slice by `setup->>'$.chassis.front.wing'` directly; ergonomics nicety, not a correctness gap.
8. **MINOR -- Filename regex (`detect.py:53-55`) requires a single `_` separator between car prefix and track.** Filenames work today but any future iRacing rename pattern would need a parallel update.

## Evidence

- Test suite: NOT EXECUTED (sandbox blocked Bash/PowerShell). Static review of seven test files (`test_parser.py` 275 lines, `test_parser_per_car.py` 115, `test_catalog.py` 135, `test_detect.py` 67, `test_ingest_partial.py` 191, `test_ingest_smoke.py` 67, `test_writer.py` 210; 1,060 lines total) shows comprehensive coverage including: per-car parametrised parser smoke (5 cars), partial-status salvage paths, catalog round-trip + idempotency, sample-rate fallback chain (3 unit tests), TrackWetness enum mapping (5 unit tests), and a 360 Hz time-axis regression.
- Lint: NOT EXECUTED. Source is annotated with `from __future__ import annotations`, type hints are consistent, no TODO/FIXME/XXX/HACK strings present in `src/racingoptimizer/ingest/`.
- Latest artefact: `recommendations/bmw__sebring_international__20260505-134556.txt` is a recent BMW Sebring text briefing -- exit-0, full setup card, indicates the parquet+catalog read path through `ingest.api.sessions` + `_filter_to_target_track` is healthy.

## Recommended next actions

- `detect.py:23` -- remove or properly map the `amvantageevogt3` placeholder; today it silently shadows Aston Martin GT3 onto BMW.
- `parser.py:28-33` and `parser.py:104-105` -- split `EXCLUDED_CHANNEL_PATTERNS` into prefix-anchored vs substring sets.
- `catalog.py:198` -- append `, session_id` to `ORDER BY recorded_at` so legacy duplicates have a stable order.
- `catalog.py:80-96` -- pin schema version via `PRAGMA user_version`; the additive-column ladder won't scale.
- `paths.py:16-20` -- add a unit test that asserts `default_corpus_root().name == "corpus"` and lives at the repo root.
- `writer.py:28-29` -- promote `_now_iso` to `paths.py` (or a tiny `_time.py`).
