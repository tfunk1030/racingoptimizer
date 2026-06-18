# VISION.md §1 — Data Ingestion — Audit (2026-05-05)

## Summary
- Compliance grade: **PASS**
- Every IBT is parsed, every channel is preserved at the file's true sample rate, every drop is logged with a reason, every completed lap is written individually, and the corpus is queryable by car / track / lap. The only thin spot is "query by setup configuration" — setup is persisted as JSON on the session row but has no first-class filter API.

## Section text (verbatim)
> Parse every IBT file. Extract every channel at full 60Hz resolution. Process every completed lap individually — not session averages. Each lap is an independent observation of how this car behaved with this specific setup on this specific track in these specific conditions. Store the raw time-series data in a structured format that can be queried by car, track, setup configuration, corner, and lap.

Applicable "What This Is NOT" rules:
- Not a system that summarizes laps into session averages and loses the detail.

## What the code does today

**Parse every IBT file.**
- `learn(path, ...)` walks `_iter_ibt_paths` recursively (`src/racingoptimizer/ingest/api.py:141-145`) — `p.rglob("*.ibt")` for directories, single-file path otherwise. Matches the directory layout in CLAUDE.md (mixed flat + subfolders).
- Every input gets a catalog row regardless of outcome — `_process_one` always upserts, even on disk-read or parse failure (`api.py:174-271`).

**Extract every channel at full 60 Hz (or true rate).**
- `parse_ibt` iterates `ibt._var_headers` and reads each scalar via `get_all` (`src/racingoptimizer/ingest/parser.py:222-236`).
- Sample rate is auto-detected from `ibt._header.tick_rate` → disk-header back-computation → YAML `SessionTickRate` → 60 Hz fallback (`parser.py:153-198`). The detected rate is threaded through `duration_s` (`parser.py:254`), the parquet `t_s` axis (`writer.py:34-36`), and per-lap `lap_time_s` (`writer.py:65-68`). This protects 360 Hz IBTs from a 6× time-axis distortion, which is what the §1 phrase "full 60 Hz" really means in this codebase: full *native* resolution.

**Process every completed lap individually — not session averages.**
- `detect_lap_boundaries` (`src/racingoptimizer/ingest/segment.py:33-87`) returns one `LapSpan` per `LapDistPct` rollover with a `valid` flag set by the `Lap` channel monotonicity invariant. The first span (lap_index = -1) holds pre-grid samples for completeness.
- `_lap_rows` (`writer.py:62-86`) emits one `LapRow` per span into the `laps` table, marks the lowest-time valid lap as `best=1`, and stores per-lap `lap_time_s` derived from the per-IBT sample rate.
- The parquet retains per-sample rows (no aggregation): `_build_dataframe` writes one row per IBT sample with `t_s`, `lap_index`, `lap_dist_pct`, `data_quality_mask`, and every persisted channel (`writer.py:32-59`). No mean/median collapse anywhere in the ingest path.
- `lap_data(session_id, lap_index, ...)` slices the parquet to one lap by `start_sample`/`end_sample` (`api.py:114-136`).

**Lose nothing — drop log.**
- `ParseResult.dropped_channels: dict[str, str]` records every dropped channel with a human-readable reason (`parser.py:91-101`).
- Two drop reasons: pattern exclusion (`CarIdx`, `TempCM`/`CL`/`CR` — `parser.py:25-33, 224-225`) and multi-element arrays (`parser.py:227-230`). Both are explicit and queryable.
- The drop dict round-trips through SQLite as a JSON column (`catalog.py:31, 66, 86`, `writer.py:139-141`).

**Lose nothing — three-tier salvage.**
- `ok` / `partial` / `failed` status semantics documented at `api.py:177-194`. `partial` survives no-laps-detected, unknown car, and unknown track; `failed` covers OS errors, YAML/parse blowups, and writer mid-stream errors.
- Re-ingest retries non-`ok` rows; `ok` short-circuits (`api.py:206-208`).

**Each lap is an independent observation tagged with car / track / setup / conditions.**
- The session row carries `car`, `track`, `recorded_at`, `setup` (full garage YAML as JSON), and `weather_summary` (means / maxes for the 12 VISION §10 channels — `parser.py:276-315`). The full per-sample env channels live in the parquet, so per-corner-phase consumers downstream can pull true per-sample atmospherics.
- The full setup blob is stored as JSON on the session row (`writer.py:131-133`, `catalog.py:25`), preserving the YAML setup that was used for that session.

**Queryable by car / track / corner / lap.**
- `sessions(car=, track=, valid_only=)` filters at the catalog level (`api.py:46-72`, `catalog.py:178-200`).
- `laps(session_id=, car=, track=, valid_only=)` returns one row per lap (`api.py:75-111`).
- `lap_data(session_id, lap_index, channels=)` returns the per-sample frame for one lap (`api.py:114-136`).
- Corner-grain querying lives downstream in `racingoptimizer.corner` (out of scope for §1) — slice A's job is to keep the per-lap raw frames; corner segmentation is a §2 concern.
- Setup configuration querying is implicit: `setup` is a JSON column on `sessions`, so SQL-side filtering by setup parameter requires a `json_extract` query the user must compose manually. There is no Python API like `sessions(spring_rate_n_per_mm=175)`. This is a soft gap against the §1 wording.

## Evidence from artefacts

**Test suite:** `uv run pytest -q tests/test_parser.py tests/test_parser_per_car.py tests/test_detect.py tests/test_segment.py tests/test_api.py tests/test_catalog.py tests/test_writer.py tests/test_paths.py tests/test_ingest_smoke.py tests/test_ingest_partial.py` → **80 passed in 27.70s**. Per-car parser smoke runs across all 5 cars (`tests/test_parser_per_car.py`); detect tests cover every car/track filename pattern.

**BMW Spa card** (`recommendations/bmw__spa_2024_up__20260505-180530.txt`):
- Header line 3: `Confidence: dense (n=2330 backing samples for the dominant dense parameter, 46 parameters reported)` — n=2330 is the per-(corner, phase) sample count flowing through the joint surrogate. A session-averaging architecture would surface n ≈ session count (single digits to low hundreds), not thousands.
- Per-parameter "Evidence" stanzas (e.g. lines 16, 30, 44) each cite `dense confidence backed by 2330 samples` and `observed in training [low, high]` ranges that differ per parameter — the fitter sees per-lap setup-vector variation, not a single session-mean number.
- Line 676: `pinned to observed median (no per-session variation in training corpus, no learnable response surface): arb_size_front` — when no session varied a parameter the recommender pins it. The pinning logic only makes sense if the underlying training set is per-session (and per-lap under that), not aggregated.
- Predicted ride-height blocks (lines 727, 740, 754, 767) carry a `[predicted]` tag — the model emits these from the per-(corner, phase) fit, not from a session mean.

**Status command** (`uv run optimize status bmw`):
```
bmw - coverage report
-------------------------------------------------------------------------
Track                       Sessions  Valid laps  Clean CP  Fit sigma     Regime
nurburgring_combined               1           1        30          -     sparse
roadatlanta_full                   2          13       390      0.000      noisy
sebring_international             37         356     10680      0.000      dense
spa_2024_up                        9          43      1290      0.000  confident
spielberg_gp                       2           6       180      1.000      noisy
```
Per-track session counts, per-lap counts, and per-(corner, phase) "Clean CP" counts are all reported correctly (and only the worktree's local corpus is visible — the BMW Spa card was produced from a different corpus snapshot with 34 BMW sessions / 289 valid laps / 8670 clean CP across the whole car). The shape of the table proves the per-lap and per-corner-phase grain survives end-to-end into the user-facing status surface.

## Gaps vs. VISION

1. **MINOR — Querying by setup configuration is not a first-class API surface.** §1 lists "setup configuration" alongside car / track / corner / lap as a query axis. `setup` is persisted as a JSON blob on the session row (`catalog.py:25`, `writer.py:131-133`), but no `sessions(spring_rate=...)` style filter exists in `api.py`. The fix would land in `src/racingoptimizer/ingest/api.py` (a `sessions_with_setup_filter(...)` helper that does `json_extract`-based SQL filtering) plus a SQLite virtual column or generated index for the most-common setup parameters. Today the workaround is to pull all sessions and filter in Python on `json.loads(row.setup)`.

2. **NONE — `weather_summary` collapses per-sample env into means/maxes at session grain.** Flagged only because it could be misread as a §1 violation. The per-sample env channels live in the parquet alongside every other channel; the summary is used purely as a UX rollup for `optimize status` and the catalog query. Per-sample env still flows through the fitter. Code: `parser.py:276-315`.

3. **NONE — `Aston Martin Vantage Evo GT3` mapped to `bmw`.** `CAR_PREFIX_MAP` in `detect.py:23` aliases `amvantageevogt3 → bmw` as a placeholder. This is documented in the source comment and out of the GTP-only scope of VISION; not a §1 violation, just a flag for future work.

## Diff vs. 2026-05-01 baseline

The 2026-05-01 baseline (`docs/VISION_COMPLIANCE.md` §1) scored 🟢 with the same evidence path (parser.py:151-201, detect_sample_rate, dropped_channels, three-tier status, sessions/laps/lap_data API). Nothing in the §1 contract has regressed:

- File-line citations still resolve.
- Sample-rate detection still threads through `writer.py:34-36, 65-68`.
- Drop log still round-trips into the catalog (`catalog.py:31, 86`, `writer.py:139-141`).
- Per-car parser smoke still parametrises across all 5 cars.

New since 2026-05-01: the `TrackWetness` enum-to-fraction normalisation (`parser.py:63-88, 244-247`) was added — this is a §10 concern but lives in the §1 ingest layer; it's an additive fix, not a regression. The 80-test §1 suite passes against the worktree HEAD.

No new gaps opened; the soft setup-configuration-query gap (#1 above) was implicit in the 2026-05-01 baseline too — neither audit flagged it explicitly.

## Recommended next actions

- Add a setup-config filter helper to `src/racingoptimizer/ingest/api.py` so callers can `sessions(car="bmw", track="spa_2024_up", setup_filter={"Chassis.Front.HeaveSpringRate": ">100"})` instead of round-tripping through Python — closes gap #1.
- Document in `src/racingoptimizer/ingest/__init__.py` (or a docstring on `sessions`) that `setup` is queryable today only via `json.loads(row.setup)` from Python, so users know the contract.
- Optional: a `corner_data(session_id, lap_index, corner_id, phase=None)` thin API in `racingoptimizer.ingest.api` would make §1's "queryable by corner" literal — though the current pattern (lap_data → corner.detect → filter) is functionally equivalent.
