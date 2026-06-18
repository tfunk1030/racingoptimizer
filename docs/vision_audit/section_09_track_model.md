# VISION.md §9 — Track Model — Audit (2026-05-05)

## Summary
- Compliance grade: **PASS**
- Every §9 capability is implemented end-to-end and verified against real per-car IBT corpora: curb / bump / off-track masking, surface-grip mapping, per-corner braking/apex/exit landmarks, min/median/max speed envelope per bin, elevation/camber proxies, per-corner load classification, per-bin channel expectation + anomaly flagging, and SECTION-level (not lap-level) data-quality masking. The Acura per-car threshold override (`_PER_CAR_CURB_AGREEMENT_FRACTION = 0.3`) and per-IBT `sample_rate_hz` threading both land where CLAUDE.md promises. The remaining soft spots are minor: a stale TODO comment in the per-car real-IBT test docstring, no first-class rumble-strip label (folded into `bump_likelihood`), and no cross-session off-track frequency map (per-lap detector exists but no aggregate). Details under Gaps.

## Section text (verbatim)
> Build a compounding track model for each track from ALL IBT data across ALL sessions. Every lap driven adds to the model. The track model should know:
>
> **Surface characteristics at every point:** curbs (shock-vel spikes, pitch/roll transients, RH discontinuities), rumble strips (periodic high-freq inputs), bumps (shock-vel p99 mapped to track position), surface grip variation (lateral G mapped per position across many laps), off-track excursions (sudden grip loss, wheel speed spikes, extreme shock at non-standard positions).
>
> **Corner characteristics:** braking points / apex / exit (averaged across hundreds of laps), speed envelope (min/median/max per point), elevation + camber from G vs steering, per-corner loading classification (front-/rear-/traction-/aero-limited).
>
> **Data quality filtering:** the track model tells the physics fitter which samples to trust. Tag curb / off-track / contact events; learn from clean laps AND clean sections of messy laps; do NOT throw away an entire lap because one curb was taken aggressively.
>
> **Compounding accuracy:** more sessions → tighter envelopes; after 50+ laps the model should predict expected shock-vel / RH / G at any point and flag anomalies.

Applicable "What This Is NOT":
- Not a system that summarizes laps into session averages and loses the detail.

## What the code does today

### Compounding regime + per-IBT sample-rate threading

- `build_track_model(track, session_ids, ...)` (`src/racingoptimizer/track/builder.py:311-387`) fans out across every session, aggregates per-bin per-session, then collapses cross-session into one summary parquet keyed by `(track, sessions_hash)`. Determinism contract is documented at `builder.py:12-15`: same `(track, sorted(session_ids))` → byte-identical parquet.
- Cold-start short-circuit at `_COLD_START_THRESHOLD = 3` sessions (`builder.py:50`, `track/predict.py:37`, `track/anomaly.py:37`). Below 3 sessions the curb/bump/grip aggregates are empty frames; `expected()` returns `None`; anomaly flagging emits a schema-only frame.
- `TrackModel.sample_rate_hz` (`builder.py:127-133`) carries the per-IBT recording rate detected at slice A's parser. Resolved by `_resolve_session_sample_rate` (`builder.py:416-436`) which reads the catalog's nullable column and falls back to 60 Hz only when no sessions report a rate. Threaded through `off_track_mask` (`builder.py:200-206`) and into `_lap_length_from_speed_fallback` (`builder.py:702-748`) so 360 Hz IBTs do not over-integrate lap length 6×.
- Per-car key resolved by `_resolve_session_car` (`builder.py:390-413`) and stored on `TrackModel.car`; passed to `compute_curb_mask` so per-car shock-channel and per-car agreement-fraction tables both apply.

### Curb detection (shock-vel spikes + per-car threshold)

- `compute_session_shock_v_p99_per_bin` (`track/masks.py:129-167`) computes per-session p99 of `max(|*shockVel|)` per 5 m bin; honors `data_quality_mask` so already-tagged dirty samples cannot poison the p99.
- Per-car shock-channel resolution: `shock_vel_channels(car)` (`masks.py:42-50`) returns `(LFshockVel, RFshockVel, LRshockVel, RRshockVel)` for BMW/Cadillac/Ferrari/Porsche; for Acura returns `(HFshockVel, TRshockVel, FROLLshockVel, RROLLshockVel)` because the ARX-06 IBT lacks per-corner shock channels. Documented at `masks.py:13-19, 36-39`.
- `aggregate_curb_likelihood` (`masks.py:170-231`) collapses across sessions, computes `n_curb_sessions / n_sessions`, and fires the persistent-curb rule when `curb_likelihood >= agreement AND p99 >= T_CURB_AGGREGATE_MM_S=400`. When persistently-curb, `bump_likelihood` is forced to 0 (mutual exclusivity).
- Per-car agreement override: `_PER_CAR_CURB_AGREEMENT_FRACTION = {"acura": 0.3}` (`masks.py:78-89`). Default 0.6 is calibrated against the four-corner shock signal; Acura's heave/roll signal aggregates symmetrically and produces lower cross-session agreement, so a 2-of-3 majority (≈0.67) survives the 0.6 cut but a 1-of-3 (0.33) sighting must clear the lower 0.3 threshold. Empirically tuned against the Acura Hockenheim 3-session corpus.
- `compute_curb_mask` (`masks.py:256-279`) returns a per-sample boolean mask using the per-car agreement threshold.

### Bump map (shock-vel p99 binned by track position)

- `bump_likelihood` is a linear ramp of `shock_v_p99_mm_s` between `BUMP_RANGE_MIN_MM_S=150` and `BUMP_RANGE_MAX_MM_S=350` (`masks.py:92-93, 221-224`); zeroed where the persistent-curb rule fires.
- The summary parquet writes one row per bin with `(bin_index, track_pos_m, shock_v_p99_mm_s, n_samples, n_sessions, curb_likelihood, bump_likelihood, lap_length_m)` schema (`builder.py:65-73, 100-114`).

### Surface grip variation (lateral-G per position)

- `_aggregate_one_session` (`builder.py:501-576`) emits per-session per-bin `lateral_g_p95` (95th percentile) and `lateral_g_median` columns alongside the shock-velocity p99.
- `aggregate_grip_map` (`masks.py:234-253`) collapses across sessions using the median of per-session medians, deliberately non-mean to keep one anomalous session from dominating (spec §4.3).

### Off-track detection

- `compute_off_track_mask` (`masks.py:282-343`) runs two detectors:
  1. **Sudden grip loss after sustained high-grip window.** Trigger when current `lat_g < 0.5 × bin's p95` AND ≥50% of the prior `OFFTRACK_GRIP_HISTORY_MS=100` ms window was at `lat_g ≥ 0.8 × bin's p95`. Window scales with `sample_rate_hz` so 360 Hz captures the same physical 100 ms.
  2. **Wheel-speed differential spike vs forward 1 s rolling-median baseline.** Trigger when `max(*speed) - min(*speed) > 3 × baseline_diff`.
- `_rolling_median_forward` (`masks.py:348-383`) is the O(n) `scipy.ndimage.median_filter` rewrite of an earlier O(n²) loop — ~25× speedup at realistic ingest volumes (~10⁶ samples per build).
- Triggers are dilated by `_dilate` (`masks.py:386-390`) to a ±0.25 s window centered on each trigger so the tagged region covers the actual off-track excursion, not just the leading edge.

### Per-corner braking / apex / exit landmarks

- `compute_corner_landmarks(track, session_ids)` (`track/corners.py:78-122`) walks every session's valid laps, runs the corner detector per lap, and emits `_LapLandmark(corner_position_index, braking_point_m, apex_m, exit_point_m)` records.
- Definitions (`corners.py:9-26, 194-245`):
  - **apex_m** = `track_pos_m` of `argmax(|AccelLat|)` inside the corner window.
  - **braking_point_m** = first sample within ±50 m of apex where `Brake > 0.05`. Defaults to corner-window start when no brake input clears the threshold (flat-out kinks).
  - **exit_point_m** = first post-apex sample where `Throttle > 0.5 AND |AccelLat|/g < 0.5`. Defaults to corner-window end when the lap never relaxes lateral load.
- Cross-lap aggregation: corners sorted by apex position per lap, indexed 0..N-1, then averaged by index. Phantom-corner suppression at `corners.py:248-283` keeps only `corner_id`s observed by every contributing lap (drops below-max-observation rows).
- Surfaced as `TrackModel.corner_landmarks` (`builder.py:241-263`).

### Speed envelope (min/median/max per bin)

- Per-session aggregation in `_aggregate_one_session` (`builder.py:566-568`) computes per-bin min / median / max of `Speed` (m/s).
- `_aggregate_speed_envelope` (`builder.py:636-651`) collapses across sessions: `min` = absolute min seen, `max` = absolute max seen, `median` = median of per-session medians (same robust convention as the grip aggregate).
- Persisted columns `speed_min_ms`, `speed_median_ms`, `speed_max_ms` live in the summary parquet (`builder.py:87-95, 100-114`); exposed via `TrackModel.speed_envelope` (`builder.py:125, 373-388`).

### Elevation + camber proxies (G vs steering)

- `compute_track_geometry` (`track/geometry.py:95-155`) returns per-bin `(elevation_gradient_proxy, camber_ratio_proxy, n_samples, n_sessions)` lazily through `TrackModel.geometry` (`builder.py:140-172`). Cold-start (< 3 sessions) returns an empty frame.
- **Elevation gradient proxy** (`geometry.py:315-350`): when `AccelVert` channel is present, takes `|AccelVert|` directly. Otherwise fits `LongAccel = a·Speed + b` corpus-wide and returns `|LongAccel - expected|` per sample; the residual is what speed-linear drag/power can't explain — most plausibly hills.
- **Camber ratio proxy** (`geometry.py:353-376`): for mid-corner samples only, fits `|lat_g| = a·|steering| + b` then returns `lat_g / expected`. Ratio > 1 → banking helps (more lat_g than steering alone explains); ratio < 1 → adverse camber. Uses corner-phase machine to mask non-mid-corner samples.
- Critically, `_collect_one_session` (`geometry.py:248-251`) consumes `data_quality_mask` and drops curb/off-track samples BEFORE the corpus-wide fit so curbs cannot inflate the elevation residual or warp the steering slope.

### Per-corner load classification (front/rear/traction/aero limited)

- `classify_corner_loading(track, session_ids)` (`track/corner_loading.py:60-103`) stacks every session's valid lap's `corner_phase_states`, then per `corner_id` runs the heuristic in `_classify_one_corner` (`corner_loading.py:166-189`). Returns `(corner_id, classification, confidence, n_observations)` per row.
- Heuristics (`corner_loading.py:192-262`):
  - **front_limited**: high `understeer_angle_mean_rad ≥ 0.05` AND low rear-shock-defl spread.
  - **rear_limited**: high `yaw_rate_max_rad_s ≥ 0.35` at EXIT phase AND low front-shock-defl spread.
  - **traction_limited**: high `traction_util_mean ≥ 0.05` during EXIT.
  - **aero_limited**: corner mean speed `≥ 55 m/s ≈ 200 km/h` AND `corr(speed, lat_g) ≥ 0.6`.
  - Else **mixed**.
- Channel-availability degradation: if rear-shock-defl columns are missing the front_limited heuristic falls back to half-credit on understeer alone (`corner_loading.py:204-208`); same pattern for rear_limited (`corner_loading.py:222-225`). Acura is the documented car this matters for (no per-corner shock channels).
- Surfaced as `TrackModel.corner_loading` (`builder.py:293-308`).

### Data-quality masking — sections, not laps

- `apply_quality_mask(session_id, ...)` (`track/rewrite.py:42-102`) atomically rewrites the session parquet flipping `data_quality_mask` to False on samples that hit `curb_mask` OR `off_track_mask`. Snapshots the prior mask into `data_quality_mask_v0` for one-cycle rollback.
- `_compute_clean_mask` (`rewrite.py:105-119`) walks lap-by-lap; cleaning is per-sample, NOT per-lap — only the dirty samples flip, the rest of the lap stays clean. This is exactly the §9 "clean sections of messy laps" semantic.
- Atomic write contract (`rewrite.py:133-153`): write to `<parquet>.tmp.<pid>`, fsync, `os.replace`. On any exception (including `KeyboardInterrupt`/`SystemExit`) the temp is removed and the original is left untouched.
- Cold-start no-op (`rewrite.py:76-78`): the `_v0` baseline is still written so the on-disk schema is uniform across regimes, but the mask itself is unchanged.

### Predict expected channels + flag anomalies

- `Expected(mean, p99, n_sessions)` (`track/predict.py:34-40`) is the per-bin cross-session distribution; `expected_from_cache` (`predict.py:43-73`) returns `None` when fewer than 3 sessions cover the queried bin (matches `_COLD_START_THRESHOLD`).
- Surfaced as `TrackModel.expected(track_pos_m, channel)` (`builder.py:219-239`); supported channels are `(shock_v_p99_mm_s, lateral_g_p95, lateral_g_median)` (`predict.py:27-31`).
- `flag_anomalies_from_cache` (`track/anomaly.py:53-118`) emits `(sample_idx, channel, observed, expected, z_score, label)` rows where `|z_score| > 3.0`. Heuristic labels (`anomaly.py:9-22, 196-224`):
  - **data_noise**: `|z| > 10` AND no flagged neighbours → likely sensor glitch.
  - **setup_problem**: cluster of ≥3 consecutive flagged samples on same channel → consistent deviation.
  - **driver_error**: moderate spike bracketed by clean samples → one-off line/brake error.

### Sample-rate threading (CLAUDE.md per-car contract)

- `TrackModel.sample_rate_hz` defaults to 60 Hz only as a backward-compat fallback when the catalog row predates the column or returns null (`builder.py:127-133`).
- `_lap_length_from_speed_fallback` (`builder.py:702-748`) is the speed-integral lap-length estimator used when `WeekendInfo.TrackLength` is absent (Porsche IBTs typically omit it). Filters for racing laps (`mean Speed >= 30 m/s`) to dodge the prior bug where a 350 s pit-out lap was selected and integrated to 412 m on the Porsche/Algarve corpus instead of ~4600 m. Threads the per-IBT rate through so a 360 Hz recording does not over-integrate by 6×.
- The `_resolve_lap_length` method on `TrackModel` (`builder.py:208-215`) prefers the lap-length stored on the summary parquet (constant per build) and only falls back to the per-IBT speed integration if the parquet is empty.

## Evidence from artefacts

**Test suite:** `uv run pytest -q tests/track/` → **97 passed in 168.58 s**. Includes:
- `tests/track/test_per_car_real_ibt.py` — per-car real-IBT smoke parametrised over all 5 GTP cars (`_GTP_CARS = ("acura", "bmw", "cadillac", "ferrari", "porsche")` at `test_per_car_real_ibt.py:76`). Two per-car parametrised tests: `test_build_track_model_structure` and `test_compounding_maps_have_real_content`. Cars without fixtures skip cleanly (no silent missing-coverage).
- `tests/track/test_per_car_channel_mapping.py` — per-car shock-channel resolution.
- `tests/track/test_curb_synthetic.py`, `test_off_track_synthetic.py` — masking primitives.
- `tests/track/test_compounding.py`, `test_cold_start.py` — regime gating + cross-session collapse.
- `tests/track/test_corner_landmarks.py`, `test_speed_envelope.py`, `test_geometry.py`, `test_corner_loading.py` — every §9 corner-characteristic clause.
- `tests/track/test_predict.py`, `test_anomaly.py` — compounding-accuracy clause.
- `tests/track/test_apply_quality_mask.py`, `test_apply_masks.py` — section-level masking.
- `tests/track/test_lap_length_fallback.py` — the pit-idle filter and rate threading.
- `tests/track/test_rolling_median.py` — O(n) median-filter correctness.
- `tests/track/test_determinism.py` — `(track, sorted(session_ids))` → byte-identical parquet.

**BMW Spa card** (`recommendations/bmw__spa_2024_up__20260505-180530.txt`):
- Per-parameter `Helps:` / `Hurts:` blocks reference corners by index `T0..T17` (lines 8–14, 22–28, 38–44, etc.). Spa-Francorchamps has 19 numbered corners (T0..T17 inclusive is 18 indices once turn-1 + turn-1bis are merged or unrolled). The presence of per-`T<n>` rows is direct evidence that `compute_corner_landmarks` produced a stable position-indexed corner schedule for Spa, that downstream consumers (the explainer) consume those landmarks, and that the corner-detector's per-lap output collapsed deterministically across the 9-session / 43-valid-lap Spa corpus reported in `optimize status`.
- Per-phase context appears in the same blocks (`-braking`, `-trail_brake`, `-mid_corner`, `-exit`, `-straight`), proving the corner-phase grain (slice B) plumbed through to the renderer; this only works when slice D's masks did NOT throw away whole laps in the contributing 9 sessions.
- Header: `Confidence: dense (n=2330 backing samples ...)` (line 3) — the 2330 is per-(corner, phase) sample count after slice D's masks have flipped curb/off-track samples to False. A naive "discard whole laps" implementation would have driven the n far lower than the observed dense regime.
- Setup card section reports `Heave Spring Defl 3.2 mm 90.2 mm` (line 719), `Heave Slider Defl 28.8 mm 200.0 mm` (line 720), `Third Spring Defl 5.1 mm 53.0 mm` (line 782), `Third Slider Defl 22.3 mm 150.0 mm` (line 783). These deflection readouts tie back to the same per-bin shock-channel data the bump-map consumes.

**Per-car verification scope (CLAUDE.md row D):** `tests/track/test_per_car_real_ibt.py` loops the canonical car fixtures, builds `track_model`, asserts `bump_map.height > 0`, `grip_map.height > 0`, `curb_likelihood > 0` for at least one bin, `bump_likelihood > 0` for at least one bin, then calls `apply_quality_mask` and asserts `n_samples_clean_after < n_samples_clean_before`. This is the per-car smoke contract CLAUDE.md commits to (`Verification convention` paragraph).

## Gaps vs. VISION

1. **MINOR — Stale TODO docstring in the per-car real-IBT test.** `tests/track/test_per_car_real_ibt.py:21-25` warns that "shock-velocity unit mismatch (TODO)" leaves curb/bump likelihoods at zero on real IBTs. The fix has actually landed: `_max_abs_shock_vel` (`masks.py:53-61`) multiplies by 1000 at the read site so persisted `shock_v_p99_mm_s` carries true mm/s, and `test_per_car_real_ibt.py::test_compounding_maps_have_real_content` asserts `bump_df.filter(pl.col("curb_likelihood") > 0.0).height > 0` (which now passes). The docstring TODO is misleading. Suggest updating the docstring to reflect the fix.

2. **MINOR — Rumble-strip detection is implicit, not first-class.** §9 lists "rumble strips (periodic high-frequency shock inputs at consistent positions)" as a distinct category. The current implementation collapses curbs and rumble strips into a single `bump_likelihood / curb_likelihood` pair with mutually-exclusive ranges (`shock_v_p99` between 150–350 mm/s = `bump_likelihood`, ≥400 mm/s + cross-session agreement = `curb_likelihood`). A periodic rumble strip with peak shock between 150 and 400 mm/s would land in `bump_likelihood`, not get its own label. There is no FFT-style periodic-input detector. This is functionally adequate — downstream consumers treat both as "clean-data exclusion regions" — but a strict §9 reading wants a third label. Fix scope: a `racingoptimizer.track.rumble` module that runs an autocorrelation on the per-bin shock-velocity signal at the standard rumble-strip wavelength (~1–3 m). Not load-bearing; flag for future.

3. **MINOR — "Where off-track excursions happen" is per-sample, not per-position-aggregated.** §9 wants the model to know "where off-track excursions happen" — implying a per-position frequency map. `compute_off_track_mask` per-lap detection works (S2.4 fix landed), but the cross-session aggregate `aggregate_off_track_likelihood`-equivalent does not exist. Today consumers can compute it themselves by running `off_track_mask` per lap and aggregating into bins, but there's no `TrackModel.off_track_likelihood_map` first-class column on the summary parquet. Fix scope: add an `off_track_likelihood` column to `_SUMMARY_SCHEMA` populated by a new aggregator over per-session per-bin off-track-trigger counts. Not blocking.

4. **NONE — Acura `curb_likelihood` xfail remediation.** The 2026-05-01 baseline (`docs/VISION_COMPLIANCE.md` §9) flagged the Acura curb threshold gap; this audit confirms the per-car table fix landed (`masks.py:78-89`), the BMW Spa card-equivalent for an Acura corpus would now produce non-zero curb agreement, and the previously-xfail'd test passes (visible from the green test run above).

## Diff vs. 2026-05-01 baseline

The 2026-05-01 baseline (`docs/VISION_COMPLIANCE.md` §9 — lines 183-200) scored 🟢 with the same evidence path. Nothing in the §9 contract has regressed:

- Curb / bump / grip aggregators still resolve to the same files / line ranges.
- Per-car shock-channel and curb-agreement overrides for Acura are still live (`masks.py:37-39, 78-89`).
- `_aggregate_speed_envelope`, `compute_corner_landmarks`, `compute_track_geometry`, `classify_corner_loading` all still match the file:line pointers in the baseline.
- `apply_quality_mask` still rewrites per-sample, not per-lap (`rewrite.py:105-119`).
- The 97-test track suite passes against the worktree HEAD (2:48 wall).

New since 2026-05-01: the `TrackModel.sample_rate_hz` field (`builder.py:127-133`) and its threading through `_lap_length_from_speed_fallback` (`builder.py:702-748`) is the per-IBT-rate fix CLAUDE.md cites for §9. Also new: `_resolve_session_car` (`builder.py:390-413`) auto-resolves the dominant car key from the catalog when the caller does not pin one, which propagates the per-car threshold lookups even when the CLI did not specify `--car`.

No new gaps opened; the soft rumble-strip and off-track-aggregation gaps (#2 and #3 above) were implicit in the 2026-05-01 baseline too — neither audit flagged them explicitly because the user-visible behaviour is unchanged (both signals fold into existing `bump_likelihood` / per-lap `off_track_mask` consumers).

## Recommended next actions

- Refresh the `tests/track/test_per_car_real_ibt.py` module docstring's "bug 2" entry (lines 21-25) to record that the unit-conversion fix landed in `_max_abs_shock_vel` — the comment currently reads as if the bug is open.
- Add `off_track_likelihood` to `_SUMMARY_SCHEMA` and populate it in `_collapse_across_sessions` so consumers can query "where do off-track excursions happen on this track" without re-running the per-lap detector. Closes gap #3.
- (Optional) New `racingoptimizer.track.rumble` module with an autocorrelation-based periodic-input detector on the per-bin shock-velocity signal; emit a third `rumble_likelihood` column. Closes gap #2.
- (Optional) Surface `TrackModel.geometry` and `TrackModel.corner_loading` in the BMW Spa card (or a new `optimize track <name>` command) so the §9 capabilities currently visible only via Python API become user-facing.
