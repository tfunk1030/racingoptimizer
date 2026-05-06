# Audit -- Slice D: Track model (2026-05-06)

## Summary
- Grade: PARTIAL
- The per-bin track model is built and exercised end-to-end (curb / off-track masks, bump / grip / speed-envelope maps, geometry, corner landmarks, corner loading, anomalies all implemented and unit-tested), but most of the slice's "load-bearing for data quality" output is orphaned: **`apply_quality_mask` is never invoked from production code**, and `physics.fit()` accepts a `TrackModel` parameter it never reads -- so curbs / off-track samples are not actually masked out of the physics fit.

## Implementation quality
- `src/racingoptimizer/track/builder.py:48-51` -- `_COLD_START_THRESHOLD = 3` and `_GRAVITY_M_S2 = 9.80665` are file-local magic numbers; the cold-start floor is duplicated in `track/predict.py:65` (`matching.height < 3`), `track/geometry.py:62` (`_COLD_START_THRESHOLD = 3`), and `track/anomaly.py:37` (`_MIN_SESSIONS_FOR_EXPECTATION = 3`) -- tied together by comments rather than a shared constant.
- Gravity constant redefined three times inside the slice -- `track/builder.py:51`, `track/anomaly.py:33`, `track/masks.py:104` -- even though `racingoptimizer/corner/config.py:14` already exports the canonical `G_MS2 = 9.80665` (used by `track/corners.py:37` and `track/geometry.py:49`). The slice's own modules disagree about which alias to import.
- `track/builder.py:519-533` -- per-session aggregation loop catches `pl.exceptions.ColumnNotFoundError` per lap but `return`s early on the FIRST raising lap (drops the entire session, not just the bad lap). Spec narrative says "skip curb/bump for the session and log", but the early-return semantics are not the same as "warn once and continue".
- `track/builder.py:702-748` -- `_lap_length_from_speed_fallback` opens `ingest_api.lap_data(... channels=["Speed"])` once per lap. Same N+1 pattern in `track/corners.py:109-117` and `track/geometry.py:228-241` (per-lap `lap_data` fetches inside the per-session loop).
- `track/anomaly.py:184-192` -- per-sample `expected` lookup is a Python `for i, b in enumerate(bins)` loop hitting two dicts per iteration; for ~1700 Spa bins this is the inner loop of `flag_anomalies`. `np.searchsorted` against a sorted bin-array would be much cheaper.
- `track/masks.py:329-336` -- wheel-speed-differential off-track detector hardcodes `LFspeed`/`RFspeed`/`LRspeed`/`RRspeed` with no per-car mapping; if any car renames or omits one of these `compute_off_track_mask` will `KeyError` rather than degrade gracefully like `shock_vel_channels(car)`.
- `track/masks.py:282-343` -- `compute_off_track_mask` does not consume `data_quality_mask` to skip already-masked samples (unlike `compute_session_shock_v_p99_per_bin` at `masks.py:142-143`).
- `track/builder.py:208-215` -- `_resolve_lap_length` falls back to `1.0` m when `lap_df` has no `Speed` column. A 1.0 m lap length silently maps every sample to bin 0 / bin 1; a hard error or zero-mask short-circuit would fail loudly.
- `track/predict.py:62-66` -- `expected_from_cache` filters via `pl.col("track_pos_m") == target_pos`, comparing two derived `float`s for exact equality. Works because the writer emits `bin * bin_size_m` consistently, but fragile.
- `track/builder.py:140-172` -- `geometry` property mutates `self.__dict__` (`object.__setattr__`) to memoize on a frozen dataclass. Idiom is documented but breaks the value-semantics guarantee implied by `@dataclass(frozen=True)`.
- `track/__init__.py:46-82` -- `__all__` exports 36 symbols, including private-prefixed `compute_session_shock_v_p99_per_bin` and several fixture-only constants (`BUMP_RANGE_*`, `OFFTRACK_*`).

## Wiring
- Builder constructor -- `cli/recommend.py:1110-1115` calls `build_track_model(track, session_ids, corpus_root=root, car=car)` per recommend invocation (the only production caller).
- Mask producers -- `track/builder.py::TrackModel.curb_mask` (line 184) and `.off_track_mask` (line 196) delegate to `track/masks.py`.
- Mask consumer -- `track/rewrite.py::_compute_clean_mask:118` ORs both masks per lap and writes via `apply_quality_mask`. `apply_quality_mask` has zero callers in `src/` (verified via grep -- only `track/__init__.py:44` re-export and tests reference it). **Masks compute, never reach disk.**
- Fitter handshake -- `physics/fitter.py:217` accepts `track_model: TrackModel` but never reads any attribute (`grep "track_model\." src/racingoptimizer/physics` returns no hits). Fit's only sensitivity to slice D is `data_quality_mask` already on the parquet (`physics/dynamic_at_speed.py:108-141`), and that column is set to all-True at slice A ingest (`ingest/writer.py:47-53`) and never overwritten.
- `sample_rate_hz` -- read by `TrackModel.off_track_mask` (`builder.py:205`) -> `compute_off_track_mask` (`masks.py:288, 310, 336, 342`); written by `_resolve_session_sample_rate` (`builder.py:416-436`) reading `ingest_api.sessions(...).sample_rate_hz`. `_lap_length_from_speed_fallback` (`builder.py:702`) takes `sample_rate_hz` directly so 360 Hz IBTs do not 6x the lap length.
- `geometry` / `corner_landmarks` / `corner_loading` / `expected` / `flag_anomalies` -- all reachable via `TrackModel` properties (`builder.py:140-308`); no production caller anywhere in `src/` (only test files import them).
- Lap-length resolution -- `track/builder.py::_lap_length_for_session` is reused by `track/corners.py:297` (lazy import to dodge cycle) and `track/geometry.py:175` (same pattern).

## Gaps
1. **HIGH -- `apply_quality_mask` is never called by production code** (`track/rewrite.py:42`; no callers in `cli/`, `ingest/`). Slice D's data-quality mask never lands on disk; downstream `data_quality_mask` reads always see slice A's all-True default. CLAUDE.md's "Track model is compounding and load-bearing for data quality" is currently a no-op in real recommend runs. Tests pass because they call `apply_quality_mask` directly.
2. **HIGH -- `fit(track_model=...)` parameter is unused** (`physics/fitter.py:217-243`). Building the TrackModel inside `cli/recommend.py:1115` and passing it to `fit()` does nothing the catalog reads don't already do; per-car curb-agreement work (S1.3 / Acura mapping) is silently wasted in production.
3. **MEDIUM -- Gravity constant duplicated three times in-slice** (`builder.py:51`, `anomaly.py:33`, `masks.py:104`) when `corner/config.py:14` exports the canonical `G_MS2`. Drift risk.
4. **MEDIUM -- Shock-velocity unit-conversion claim in test docstring may be stale** (`tests/track/test_per_car_real_ibt.py:20-25` "Bug 2 (TODO)"). The actual code DOES scale m/s × 1000 at `masks.py:61` and `builder.py:538`. Needs re-verification on a fresh real-IBT corpus to confirm bump/grip likelihoods are non-zero or remove the xfail.
5. **MEDIUM -- Wheel-speed channels hardcoded** (`masks.py:329-333`). No `wheel_speed_channels(car)` helper; any per-car wheel-speed renaming would `KeyError`.
6. **MEDIUM -- Acura curb-agreement threshold tuned against 3-session corpus only** (`masks.py:78-80`). 0.3 fraction empirically picked; no regression test covers convergence as Acura session count grows.
7. **MEDIUM -- `_lap_length_from_speed_fallback` does N parquet reads per session** (`builder.py:728-746`); a single `pl.scan_parquet(...).group_by("lap_index").agg(...)` would collapse to one read.
8. **MEDIUM -- `compute_off_track_mask` ignores `data_quality_mask`** (`masks.py:282-343`) -- would re-derive triggers from already-flagged dirty samples on a second pass.
9. **LOW -- `geometry` / `corner_landmarks` / `corner_loading` / `expected` / `flag_anomalies` fully implemented but unwired** (`builder.py:140-308`); each has unit tests but no production consumer in `cli/`, `physics/`, or `explain/`. `corner_landmarks` is the obvious VISION §9 ("precise braking points, apex positions, exit points") candidate for the renderer.
10. **LOW -- Frozen-dataclass `__dict__` mutation** (`builder.py:170-172`) for memoization; consider `cached_property` on a non-frozen class.
11. **LOW -- `_resolve_lap_length` 1.0 m fallback hides errors** (`builder.py:215`).
12. **LOW -- `__all__` exports 36 symbols, several private/fixture-only** (`track/__init__.py:46-82`).
13. **LOW -- Per-sample `expected` lookup is a Python loop** (`anomaly.py:184-192`); `np.searchsorted` would vectorise.
14. **LOW -- `expected_from_cache` uses float-equality on `track_pos_m`** (`predict.py:64`).

## Evidence
- Test suite: NOT RUN (Bash / PowerShell / `uv run` access denied throughout). Static count: 84 test functions across 18 files in `tests/track/`. Coverage spans bin assignment, cold start, compounding, curb/off-track synthetics, lap-length fallback, predict, rolling median, speed envelope, geometry, corner landmarks, corner loading, apply-mask round-trip, anomaly classification, determinism, per-car channel mapping. Per-car real-IBT smoke at `tests/track/test_per_car_real_ibt.py` (slow-marked) loops all 5 GTP cars and asserts compounding-regime maps populate.
- Lint: NOT RUN. No `noqa`, no `TODO`/`FIXME`/`XXX`/`HACK` markers in `src/racingoptimizer/track/`; only sticky marker is the test docstring at `tests/track/test_per_car_real_ibt.py:20`.
- Latest artefact: `recommendations/bmw__spa_2024_up__20260506-053645.txt` (most recent BMW Spa run). Builds a TrackModel for `bmw @ spa_2024_up` per `cli/recommend.py:1115`, but per the wiring analysis above the masks the model produces never reach the fit because `apply_quality_mask` isn't invoked and `fit()` ignores the `track_model` argument.

## Recommended next actions
- Wire `apply_quality_mask` into the production pipeline -- either at the end of `optimize learn` per session, or inside `cli/recommend.py:1110+` after the TrackModel build, before training. Without this the slice's central premise is dead in real runs.
- Decide what `fit(track_model=...)` should do. Either consume `track_model.curb_mask` / `.off_track_mask` to filter the training frame, or drop the unused parameter from `physics/fitter.py:217`.
- Consolidate gravity constant: replace the three `_GRAVITY_M_S2` definitions in `track/{builder,anomaly,masks}.py` with `from racingoptimizer.corner.config import G_MS2` (matches `track/{corners,geometry}.py`).
- Add `wheel_speed_channels(car)` next to `shock_vel_channels` in `track/masks.py`.
- Hoist the cold-start floor (`_COLD_START_THRESHOLD = 3`) to a single shared constant; consume from `predict.py`, `anomaly.py`, `geometry.py`.
- Vectorize `_expected_per_sample` with `np.searchsorted`.
- Either wire or hide `geometry` / `corner_landmarks` / `corner_loading` / `expected` / `flag_anomalies`; the renderer (`explain/`) is the natural home for `corner_landmarks` per VISION §9.
- Tighten `__all__` (`track/__init__.py:46-82`) to the spec §2 public API.
- Re-verify the docstring's "Bug 2" shock-velocity claim against a fresh real-IBT corpus.
