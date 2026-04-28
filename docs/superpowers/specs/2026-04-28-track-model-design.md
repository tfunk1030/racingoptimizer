# Track Model — Design Spec

**Date:** 2026-04-28
**Slice:** D — Track Model (master plan §4 slice D card)
**Module:** `racingoptimizer.track`

## 1. Context

`racingoptimizer` decomposes every lap into corner-phases and fits empirical physics off those samples (see `VISION.md`, `CLAUDE.md`). The fitter must not be poisoned by curb strikes, rumble-strip rides, off-track excursions, or contact: a 600 mm/s shock-velocity spike at the T1 turn-in apex curb is evidence of curb geometry, not of soft springs. Slice D builds the per-track classifier that tells slice E which samples to trust.

Two CLAUDE.md commitments anchor this slice:

- **Track model is compounding and load-bearing for data quality.** Every new IBT for a given track refines the model. Slice D is therefore stateless over its inputs (a sorted set of session ids) but persistent over its outputs (cached parquet on disk).
- **Do not throw away whole laps — mask the dirty sections.** D produces a per-sample boolean mask, not a per-lap valid flag. Per-lap validity stays slice A's responsibility. D only decides which **samples within a lap** are clean.

The handshake with slice A is concrete: A's writer reserves a `data_quality_mask: bool = True` column in every session's parquet (master plan §3, Task A-bis). When `apply_quality_mask(session_id)` runs, slice D rewrites that column in place.

The handshake with slice B is operational: D consumes `lap_data()` directly from slice A's parquets, but defers to B's `segment_lap` for corner-id annotation when available. D is robust to B's absence — it indexes everything by `track_pos_m` (meters along the lap), not by corner.

This is one slice. Physics fitting (E), aero correction (C), and recommendation rendering (F) are explicit non-goals here.

## 2. Public API

```python
from racingoptimizer.track import build_track_model, apply_quality_mask, TrackModel

build_track_model(track: str, session_ids: list[str]) -> TrackModel
    # Build (or load from cache) a track model from the given sessions.
    # Pure function of (track, sorted(session_ids)). Cache key is
    # (track, sha256(",".join(sorted(session_ids)))[:16]). On cache miss,
    # reads each session's parquet, computes per-bin curb / bump / grip
    # aggregates, and persists to corpus/track_models/<track>.<hash>.parquet.
    # Caller does not pass car: the track model is car-agnostic at the surface
    # layer (curbs and bumps are properties of the asphalt). Per-session rows
    # in the persisted artefact retain `car` for diagnostic filtering.

TrackModel.curb_mask(lap_df: pl.DataFrame) -> np.ndarray[bool]
    # Returns a per-sample boolean array, same length as lap_df, where True
    # marks "this sample is on a curb." Uses the model's `bump_map` and
    # the lap_df's `LFshockVel`/`RFshockVel`/`LRshockVel`/`RRshockVel`
    # columns. The mask is conservative: True only when (a) the position
    # bin's curb-likelihood crosses the model's threshold AND (b) the
    # sample's instantaneous shock-velocity p99 deviation exceeds the bin's
    # baseline.

TrackModel.bump_map -> pl.DataFrame
    # Schema: (track_pos_m: f32, shock_v_p99_mm_s: f32, n_samples: u32,
    # n_sessions: u16, curb_likelihood: f32, bump_likelihood: f32).
    # One row per track-position bin. Sorted by track_pos_m.

TrackModel.grip_map -> pl.DataFrame
    # Schema: (track_pos_m: f32, lateral_g_p95: f32, lateral_g_median: f32,
    # n_samples: u32, n_sessions: u16). One row per track-position bin.
    # `lateral_g_p95` is normalised per session by that session's overall
    # max lateral G, then aggregated across sessions — see §4.

TrackModel.off_track_mask(lap_df: pl.DataFrame) -> np.ndarray[bool]
    # Returns a per-sample boolean array, same length as lap_df, where True
    # marks "this sample is off-track or in contact." Combines (a) sudden
    # lateral-G drop > N% within M samples at a position the model believes
    # is off-line, with (b) `car_contact`-channel events when present (see §8).

apply_quality_mask(session_id: str) -> None
    # Rewrites the parquet at corpus/sessions/<car>/<track>/<session_id>.parquet
    # with an updated `data_quality_mask` column. Logic:
    #   data_quality_mask[i] = NOT(curb_mask[i] OR off_track_mask[i])
    # The track model used for the rewrite is built from ALL sessions for that
    # track currently catalogued (i.e. build_track_model(track, all_sids_for_track)).
    # Atomic: writes to <session_id>.parquet.tmp, fsyncs, then renames.
    # On rewrite, the original column is preserved as `data_quality_mask_v0`
    # for one full re-fit cycle (see §3).
```

Polars (not pandas) for return types — consistent with slice A.

## 3. On-disk layout

```
corpus/
  track_models/
    <track>.<input_hash>.parquet      # cached aggregate per (track, session set)
    <track>.latest.json               # pointer file: {"hash": "...", "session_ids": [...]}
```

Parquet (not SQLite) because the artefact is wide, append-only-per-row, and read by `pl.scan_parquet` exactly the way slice A's session parquets are. SQLite would force a schema migration whenever a new aggregate column lands; parquet just adds a column. `track_models/` is one flat directory — there are at most ~10–20 distinct tracks across the corpus, no nesting needed.

`<input_hash>` = `sha256(",".join(sorted(session_ids)))[:16]`. Same input set → same file, regardless of call ordering.

`<track>.latest.json` is a pointer the cold-start logic and `apply_quality_mask` use to find the freshest model without enumerating directory contents. Updated atomically (write `.tmp`, fsync, rename) on every successful `build_track_model` call.

### Persisted schema (`<track>.<hash>.parquet`)

One row per `(session_id, track_pos_bin)`:

| Column | Type | Notes |
|---|---|---|
| `session_id` | str | sha256-prefix from slice A's catalog |
| `car` | str | normalized key (`acura`, `bmw`, ...) |
| `track_pos_m` | f32 | left edge of the bin in meters along the lap |
| `lap_dist_pct_lo` | f32 | left edge as a fraction `[0, 1)` (paired with track_pos_m for cross-checks) |
| `n_samples` | u32 | samples from this session that fell in this bin (across all valid laps) |
| `shock_v_max_mm_s` | f32 | max shock-velocity magnitude (LF/RF/LR/RR) over the bin |
| `shock_v_p99_mm_s` | f32 | per-bin p99 shock-velocity magnitude |
| `lateral_g_p95` | f32 | per-bin p95 lateral-G magnitude, **normalized** by session max lateral-G |
| `lateral_g_median` | f32 | per-bin median lateral-G (also normalized) |
| `wheel_speed_dev` | f32 | std-dev of per-wheel speed differential — proxy for off-track wheel slip |
| `is_curb_session` | bool | True iff this bin's per-session p99 exceeds the per-track p99 threshold (§4) |

The aggregate row computed at read time (collapsing `session_id`) is what `bump_map` and `grip_map` expose. Stored aggregates: bin-level `curb_likelihood` and `bump_likelihood` are derived columns (recomputed on `build_track_model` and embedded as a sidecar `<track>.<hash>.summary.parquet` to skip the aggregation on every `curb_mask` call).

### A↔D handshake — `data_quality_mask` rewrite

Slice A reserves the column (`bool`, default `True`). Slice D rewrites it via `apply_quality_mask`. Mechanism, pinned:

1. Read the existing parquet via `pl.read_parquet(path)`.
2. If `data_quality_mask_v0` is **absent**, copy the current `data_quality_mask` column to `data_quality_mask_v0` (preserve the all-true baseline for rollback).
3. Build the track model for this session's track from all currently-catalogued sessions on it.
4. Compute `curb_mask` and `off_track_mask` per lap using the model.
5. Compose `data_quality_mask = ~(curb_mask | off_track_mask)`.
6. Write the updated frame to `<sid>.parquet.tmp` via `pyarrow.parquet.write_table` with the same compression (ZSTD).
7. `os.fsync` the tmp file's directory, then `os.replace(tmp, original)` (atomic on Windows + POSIX).
8. On any exception between steps 1 and 7, the original parquet is untouched. The `.tmp` is removed in a `finally` block.

**Rollback policy:** `data_quality_mask_v0` is retained until the **next** full re-fit cycle (next time `apply_quality_mask` runs on this session). At that point it is overwritten with the then-current `data_quality_mask` (i.e. the previous cycle's output) and dropped — only one cycle of history is kept. A rollback within that window is `data_quality_mask = data_quality_mask_v0` then re-write.

Justification for in-place vs new file: slice A's `lap_data` looks up parquet path by `session_id`; a new file would force catalog updates and break idempotency. In-place rewrite + atomic rename is the smaller blast radius.

## 4. Algorithms

### 4.1 Curb detection — pinned recommendation

**Primary:** shock-velocity p99 threshold per track-position bin.

For each session, compute per-bin p99 of `max(|LFshockVel|, |RFshockVel|, |LRshockVel|, |RRshockVel|)`. A bin is flagged `is_curb_session=True` iff its p99 exceeds `T_curb_session = 350 mm/s` (initial value — tuned against the hockenheim T1 fixture; revise as the corpus grows).

A bin is **persistently a curb** in the aggregated track model iff:
- `is_curb_session=True` in ≥ 60% of sessions covering that bin, AND
- aggregated `shock_v_p99_mm_s` (cross-session p99) ≥ `T_curb_aggregate = 400 mm/s`.

`curb_likelihood ∈ [0, 1]` is the fraction of sessions agreeing.

`TrackModel.curb_mask(lap_df)` then marks sample `i` as curb iff:
- `lap_df[i, "track_pos_m"]` falls in a persistently-curb bin, AND
- `max(|shock_vel|)[i] ≥ 0.5 × bin's aggregate p99` (excludes "we drove past the curb without touching it").

**Fallback:** wavelet decomposition on the shock-velocity time series at known periodic curb frequencies (rumble-strip teeth at ~30–80 Hz at racing speed). Activated only if the unit test (§11) shows precision < 80% on the hockenheim 5-session fixture. The fallback is implemented as `racingoptimizer.track.curb_wavelet.detect(...)` and gated by a `TrackModel.curb_method ∈ {"p99_bin", "wavelet"}` field; `p99_bin` is the default. We do not ship both; we ship the simple one and earn the wavelet by demonstrating the simple one fails.

**Argued alternatives (rejected):**
- Pitch/roll transients only: too noisy on banked corners (Daytona oval mode in particular).
- Ride-height discontinuity: aero ride-height channels are filtered upstream by the iRacing engine; transients are blunted.

### 4.2 Bump detection

A bump is a track-position bin where shock-velocity p99 is consistently elevated **but the bin is not a curb** (no driver-line discontinuity). Operationally:

- `bump_likelihood = curb_likelihood × 0` if `curb_likelihood > 0.5` (curbs and bumps are mutually exclusive in the aggregate)
- Otherwise `bump_likelihood = clip((shock_v_p99_mm_s - 150) / 200, 0, 1)` — saturates at p99 = 350 mm/s.

A bin with `bump_likelihood > 0.4` produces a softer mask: it does **not** flip `data_quality_mask` to False (the physics fitter wants bump samples for damper learning), but it stamps a per-sample `is_bump_event` column on `lap_df` consumers via `TrackModel.annotate(lap_df)` in a future iteration. For slice D, bump_map is read-only output; it does not contribute to `data_quality_mask`.

### 4.3 Grip mapping

Per session: compute the session's max lateral-G across **clean** samples (`data_quality_mask = True`). For each bin, compute lateral-G p95 and median over clean samples, normalize by the session max. Store both percentiles per session-bin in the persisted parquet.

Cross-session aggregation: `grip_map.lateral_g_p95` = median across sessions of per-session normalized p95. Median (not mean) so a single anomalous session does not dominate.

Edge case: the first `apply_quality_mask` pass uses an all-True mask (slice A's default), which produces a contaminated grip_map (curbs inflate normalization). After D writes the first real mask, the next `build_track_model` call re-derives grip_map on cleaned data. Subsequent calls converge. We do not iterate to a fixed point automatically — one round-trip is enough at the precision the slice E fitter cares about; document the slight bias.

### 4.4 Off-track detection

Two detectors OR'd together:

1. **Sudden grip loss at off-line position.** `lateral_g[i] < 0.5 × bin_lateral_g_p95` AND `lateral_g[i-M:i].mean() > 0.8 × bin_lateral_g_p95`, with `M = 6 samples (= 100 ms at 60 Hz)`. The "off-line" condition is bootstrapped: in a session's first pass, off-line = `track_pos_m` more than 2 m lateral from the median trajectory across all completed laps in the session. After the model has ≥ 3 sessions, off-line = "outside the 5th–95th percentile lateral envelope of clean trajectories from the persisted model."

2. **Wheel-speed differential spike.** `wheel_speed_dev[i] > 3 × wheel_speed_dev_baseline_for_bin`. Catches dirt off-tracks where a wheel locks against grass.

Mask window: any sample matching either detector flips `data_quality_mask` to False for `±0.5 s` (= 30 samples at 60 Hz, 15 either side, see §8 for the unified contact / off-track window).

### 4.5 Bin assignment

Each sample's `track_pos_m` is computed as:

```
track_pos_m = lap_dist_pct × lap_length_m
```

where `lap_length_m` comes from the IBT YAML header's `WeekendInfo.TrackLength` (units in meters, slice A persists the raw header; slice D extracts via `json_extract(setup_or_session_meta, ...)` — pin the path in the spec at implementation time). For tracks where the header is missing or zero, fallback to the per-track median of `Speed × dt` integrated over a clean lap from the corpus. Log a warning either way.

Bin index = `floor(track_pos_m / 5.0)`. See §6 for the 5 m default justification.

## 5. Compounding behaviour

**Stateless over inputs.** `TrackModel` is a pure function of `(track, sorted(session_ids))`. No hidden state on disk that mutates across calls.

**Cache:** the persisted `<track>.<input_hash>.parquet` (and `.summary.parquet`) is the cache. `build_track_model` checks for an existing file; if present and `mtime` is newer than each underlying session parquet, load and return. Otherwise refit and write.

**New session lands → cache key changes.** When slice A ingests a new session, `apply_quality_mask` (or any other `build_track_model` call) sees a new `session_ids` set, computes a new hash, and writes a new cache file. Old cache files are not deleted automatically — disk is cheap and rollback is easier with history. A future `track-model-gc` CLI command can prune older-than-N caches; out of scope here.

**Refit is lazy.** Nothing background-recomputes when a session lands. The model rebuilds the next time someone calls `build_track_model(track, ...)`. This matches slice A's "explicit, never silently" philosophy.

**No incremental fits.** Re-aggregation across the full session set on every call is cheap (parquet scans of session-level pre-aggregates, not raw 60 Hz data). We do not maintain a streaming aggregator. If profiling reveals a hotspot for tracks with > 50 sessions (none today), revisit.

## 6. Bin size for track-position

**Pinned default: 5 m.**

Justification:
- Smaller (1 m) bins put 5–8 samples per bin per lap at GTP cornering speeds (60 Hz × ~250 km/h). With < 50 laps per session we get < 250 samples per bin per session, p99 too noisy.
- Larger (10 m) bins blur curbs into surrounding asphalt at hairpin entries where the curb segment is < 15 m.
- At 5 m: ~30–40 samples per bin per session at racing speed; ~1 700 bins for a 8.5 km Spa-like lap; small enough to capture curb edges; large enough that p99 is a stable estimator with 20+ samples.

Configurable at `build_track_model(..., bin_size_m=5.0)`. Persisted in the parquet's metadata so consumers know what they got.

## 7. Cold-start regime

**Threshold:** `< 3 sessions per track ⇒ cold-start.`

Behaviour in cold-start:
- `bump_map` returns an empty DataFrame (zero rows) — consumers must handle empty case.
- `grip_map` returns an empty DataFrame.
- `curb_mask(lap_df)` returns `np.zeros(len(lap_df), dtype=bool)` — never flags a curb.
- `off_track_mask(lap_df)` returns `np.zeros(len(lap_df), dtype=bool)` — never flags off-track.
- `apply_quality_mask` runs the rewrite, but the resulting `data_quality_mask` is identical to the input (still all True). The rewrite is a no-op on data, but the `data_quality_mask_v0` snapshot is still established (so the schema is uniform across cold-start and warm-start sessions).

Why `< 3` and not `< 2`: a single session's per-bin p99 is too noisy to threshold meaningfully (one off-line moment dominates). Two sessions can disagree without resolution. Three sessions is the minimum for a cross-session-agreement signal (`is_curb_session=True in ≥ 60% of sessions` requires ≥ 2 of 3).

This honours the master plan's risk-table mitigation: D produces all-clean masks rather than spurious flags on single-session tracks (Daytona 2011 road, Spielberg, Nurburgring Combined per master plan §1).

The cold-start regime is recorded in the persisted summary parquet's metadata as `regime: "cold_start"` so downstream slices (E, F's `status` command) can report it.

## 8. Off-track vs contact — masking window

`car_contact`-channel events (when present in the IBT — not all builds expose it; falls back gracefully when absent) are masked at the **same severity** as off-track. Both detectors converge on a single behaviour:

> Any sample where contact OR off-track is detected flips `data_quality_mask = False` for a **0.5 s window centred on the event** (i.e. `±15 samples` at 60 Hz around the trigger sample).

Justification for 0.5 s: empirically, the post-contact / post-off-track recovery window where suspension oscillation and yaw transient bleed off lasts ~300–400 ms; 500 ms gives margin without throwing away too much. Configurable via `build_track_model(..., event_window_s=0.5)`.

Contact and off-track both contribute to the same `off_track_mask` output; the API does not distinguish them. The persisted parquet's per-session bin row records `n_contact_events` and `n_off_track_events` separately for diagnostics. The merging into one mask for downstream consumption is deliberate: slice E does not need the distinction — both produce non-clean samples.

If the IBT lacks `car_contact`, the warning is logged at ingest time (slice A's responsibility); slice D logs a once-per-session info note that contact-detection is degraded for that session and proceeds with off-track-only detection.

## 9. Failure handling

| Failure mode | `build_track_model` behaviour | `apply_quality_mask` behaviour |
|---|---|---|
| Session parquet missing | Skip session, log warning, proceed with remainder. | Raise `FileNotFoundError`. |
| Session parquet missing channel (`LFshockVel` absent) | Skip session for shock-derived maps (curb / bump). Still contribute to grip_map if lateral-G present. Log: `"track={track} session={sid}: skipped curb/bump (LFshockVel absent)"`. | Skip the curb component of the mask for that session; off-track from lateral-G alone. |
| Track has zero sessions | Raise `ValueError("no sessions for track {track}")`. | N/A — caller never reaches this if A has the session. |
| `< 3 sessions` | Cold-start regime (§7). No error. | Rewrites with all-True mask + establishes `data_quality_mask_v0` baseline. |
| `lap_length_m` unresolvable from header AND fallback fails | Log error, return cold-start TrackModel (better to be safe than wrong). | No-op rewrite. |
| Atomic rename fails (e.g. file lock on Windows) | N/A | Retry once with 100 ms backoff; on second failure, leave `.tmp` and original both present, log error, return without raising. Caller checks file integrity by re-reading. |
| Parquet schema mismatch (older slice-A parquet missing `data_quality_mask`) | Log warning; treat that session's mask as all-True for input purposes. | Add the column with default True before applying the new mask. |

`build_track_model` never raises on a recoverable per-session issue; failures are always logged at WARNING and counted in a final summary line `"track={track}: built from {n_ok}/{n_input} sessions"`.

## 10. Idempotency & determinism

- Same `(track, sorted(session_ids))` tuple, same parquet contents → identical `TrackModel` and identical persisted artefact bytes.
- Random seeds: any sampler / bootstrap (none in v1; if §4.1's wavelet fallback ever lands it must use a fixed seed).
- Determinism is verified: §11's "round-trip" test calls `build_track_model` twice on the same input and asserts byte-equal cache files.
- Cache invalidation only on: (a) input session set change (different hash), or (b) any source parquet `mtime` newer than the cache. Both are explicit; nothing time-based or stochastic.

## 11. Testing

### 11.1 Unit

- **Curb-detection on synthetic shock-velocity series.** Build a numpy series of length 600 (10 s at 60 Hz) with baseline noise (`σ = 50 mm/s`) and an injected curb signature (a 5-sample burst at 600 mm/s every 30 samples) at `track_pos_m ∈ [120, 150]`. Build a TrackModel from 3 such synthetic "sessions". Assert: `curb_mask(synthetic_lap_df)` has precision ≥ 95% and recall ≥ 90% on the injected samples. Also assert: a curb-free synthetic series produces zero curb flags (false-positive rate = 0 on noise-only).
- **Bin assignment.** Assert that a sample with `lap_dist_pct = 0.5` on a track with `lap_length_m = 4574` produces `track_pos_m = 2287.0` and `bin_index = 457` (floor(2287/5)).
- **Cold-start.** Build a TrackModel from a single synthetic session. Assert: `bump_map.is_empty()`, `grip_map.is_empty()`, `curb_mask(lap_df)` is all-False, regime metadata = `"cold_start"`.
- **Mask round-trip.** Construct a fake session parquet with `data_quality_mask = True` everywhere. Call `apply_quality_mask`. Assert: parquet now contains `data_quality_mask_v0` (all True) and `data_quality_mask` with at least one False where the synthetic curb burst lands.

### 11.2 Integration — compounding

The canonical fixture set for compounding is **3 hockenheim acura sessions**. The master plan's slice-D card mentions "3 daytona acura sessions", but the corpus has only 2 acura sessions at Daytona; we substitute hockenheim, which has 5 acura sessions available (codebase conventions in the slice-D coordinator brief — pinned here):

- `ibtfiles/acuraarx06gtp_hockenheim gp 2026-03-29 20-51-12.ibt`
- `ibtfiles/acuraarx06gtp_hockenheim gp 2026-03-30 13-24-28.ibt`
- `ibtfiles/acuraarx06gtp_hockenheim gp 2026-03-30 15-21-02.ibt`
- `ibtfiles/acuraarx06gtp_hockenheim gp 2026-03-30 15-35-40.ibt`
- `ibtfiles/acuraarx06gtp_hockenheim gp 2026-03-30 15-58-44.ibt`

Test (`tests/track/test_compounding.py::test_curb_detection_compounds_across_sessions`):

1. Ingest the first 3 hockenheim acura fixtures via slice A (`learn`). Capture session ids.
2. `model_3 = build_track_model("hockenheim_gp", session_ids[:3])`.
3. Locate the T1 turn-in bin (`track_pos_m ≈ 90–110 m` on Hockenheim — pin the exact range when the implementation lands and a clean lap is profiled). Assert: `model_3.curb_mask(any_clean_lap_df)` flags ≥ 80% of the shock-velocity p99 samples in that bin.
4. Repeat with all 5 sessions: `model_5 = build_track_model("hockenheim_gp", session_ids)`.
5. Assert: false-positive rate (curb flags outside the persistently-curb bins of `model_5`) drops compared to `model_3`. The exact numeric drop is not pinned — the test asserts `fp_rate(model_5) < fp_rate(model_3)` and is a regression guard, not a precision claim.

### 11.3 Integration — quality-mask round-trip via slice A

Test (`tests/track/test_quality_mask_roundtrip.py::test_apply_quality_mask_updates_parquet`):

1. Ingest the canonical acura hockenheim fixture (`acuraarx06gtp_hockenheim gp 2026-03-29 20-51-12.ibt`) via slice A. Assert: parquet has `data_quality_mask` column, all True.
2. Ingest 2 more hockenheim acura sessions to clear the cold-start threshold.
3. Call `apply_quality_mask(session_id_1)`.
4. Re-read the parquet. Assert: `data_quality_mask_v0` exists and is all True. `data_quality_mask` has at least one False sample (T1 curb hit).
5. Call `apply_quality_mask(session_id_1)` a second time. Assert: `data_quality_mask_v0` is now whatever the previous `data_quality_mask` was (one cycle of history, per §3 rollback policy).

### 11.4 Cold-start integration

Test (`tests/track/test_cold_start.py::test_single_session_yields_all_clean_mask`):

1. Pick a single-session track from the corpus (Spielberg, Nurburgring Combined, or Daytona 2011 road per master plan §1).
2. Ingest the session. Build the track model. Assert: regime metadata = `"cold_start"`, `apply_quality_mask` runs without error, post-rewrite `data_quality_mask` is all True (identical to baseline), `data_quality_mask_v0` is established.

### 11.5 Determinism

Test (`tests/track/test_determinism.py::test_build_is_deterministic`): Build the same track model twice from the same 3-session input set. Assert: cache file exists after first call; second call short-circuits (no-op) and returns a `TrackModel` with byte-identical persisted summary parquet (`hashlib.sha256` of file bytes).

## 12. Module layout

```
src/racingoptimizer/
  track/
    __init__.py        # public API: build_track_model, apply_quality_mask, TrackModel
    model.py           # TrackModel class (curb_mask, off_track_mask, bump_map, grip_map)
    builder.py         # build_track_model + per-session aggregation
    masks.py           # curb / bump / off-track / contact detectors (§4)
    rewrite.py         # apply_quality_mask + atomic parquet rewrite (§3)
    paths.py           # corpus/track_models/<track>.<hash>.parquet path layout
    bins.py            # track_pos_m → bin assignment (§4.5)
tests/
  track/
    test_curb_synthetic.py
    test_bin_assignment.py
    test_cold_start.py
    test_compounding.py        # uses 3+5 hockenheim acura fixtures (§11.2)
    test_quality_mask_roundtrip.py
    test_determinism.py
```

No `racingoptimizer.track.utils` module. Helpers belong inside the module that needs them (master plan §2 anti-recommendation against `utils.py`).

## 13. Out of scope

- **Physics fitting (slice E).** D produces masks and maps; it does not fit spring-rate or LLTD models.
- **Aero-map correction (slice C).** D does not consume air density or aero surfaces.
- **Recommendation rendering (slice F).** D does not produce a SetupJustification or render anything to stdout.
- **Per-lap validity decisions (slice A).** A decides which laps are `valid=1`. D only decides which **samples within a lap** are clean. A lap with 30% of its samples masked False by D is still `valid=1` from A's perspective; downstream consumers (E) decide whether to keep or drop based on remaining clean-sample density.
- **Corner-id annotation (slice B).** D indexes by `track_pos_m`. If slice B has run and `corner_id` is present in `lap_df`, D ignores it. Future iterations may use it to refine bin sizes per corner; not in this slice.
- **Cross-track transfer.** D does not transfer learning from one track to another. Each track's model is independent.
- **Real-time / streaming.** D operates on persisted parquets only.

## 14. Open questions

1. **Off-track vs contact threshold.** Both produce a 0.5 s mask window; the API does not distinguish. The open question is whether the persisted per-session bin row's `n_contact_events` is useful enough to keep, or whether we collapse to `n_event` and drop the distinction. Pinned for v1: keep both for diagnostics; revisit if no consumer reads them by slice E completion.
2. **Track-position normalization across lap-distance scales.** `LapDistPct` is `[0, 1)`. Track-position in meters is derived from `WeekendInfo.TrackLength` in the IBT YAML header. Open: are there iRacing tracks where the header value disagrees with measured `Speed × dt` integrated over a clean lap? The fallback path (§4.5) integrates speed; if both exist and disagree by > 1%, log and prefer the integration. A future audit pass over all 117 sessions should resolve the question empirically.
3. **Track variants (e.g. Daytona 2011 road vs Daytona 2011 oval).** Recommendation: separate `track` keys. The track slug from slice A's `detect.py` already produces distinct slugs (`daytona_2011_road` vs `daytona_2011_oval`), and curb / bump / grip profiles differ structurally between road and oval layouts. Pin: D treats each unique slug as an independent track, no cross-variant pooling.
4. **`car_contact` channel availability.** Survey across the 117-file corpus needed before slice D implementation. If < 50% of sessions expose it, the per-session "n_contact_events" diagnostic is mostly null and we should consider dropping it from the persisted schema. Not blocking for the spec; resolved at implementation time.
5. **Bump-map bias from cold-start round-trip.** §4.3 acknowledges that the first `apply_quality_mask` runs against an all-True input mask, so initial `lateral_g_p95` normalization is contaminated by curb samples. We do not iterate to a fixed point. Open: is one round-trip's bias tolerable for slice E, or should D run a second pass automatically? Defer to slice E's spec; if E's residuals show the bias matters, we add an optional `apply_quality_mask(..., iterate=True)` flag in a later iteration.
