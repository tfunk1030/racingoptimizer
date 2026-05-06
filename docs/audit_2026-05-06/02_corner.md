# Audit -- Slice B: Corner-phase (2026-05-06)

## Summary
- Grade: PASS
- All five GTP cars run end-to-end through `corner_phase_states`; spec §6 derived columns (S2.1 damper-velocity / S4.8 damper-force / S2.10 empirical understeer) are wired and channel-gated correctly, with a few documented deferrals.

## Implementation quality

- `src/racingoptimizer/corner/phase.py:14` -- `Phase(StrEnum)` (5 values) + `CornerPhaseKey` NamedTuple. Hashable, JSON-stable.
- `src/racingoptimizer/corner/config.py:17` -- `PhaseThresholds` frozen dataclass with universal defaults; `PER_CAR={}` empty per spec; `ms_to_samples` derives sample-count windows from any `sample_rate_hz`.
- `src/racingoptimizer/corner/detect.py:19` -- Schmitt-trigger lateral-G corner detector (entry 0.5g, exit 0.3g, hold/min-duration/min-gap post-filtering); per-IBT sample rate honoured.
- `src/racingoptimizer/corner/boundaries.py:25` -- forward-only state machine `BRAKING -> TRAIL_BRAKE -> MID_CORNER -> EXIT -> STRAIGHT`; trail-brake gate uses `0.6 * lat_g_entry` per spec §5.
- `src/racingoptimizer/corner/states.py:186` -- `segment_lap` appends `corner_id`/`phase`; raises `NotImplementedError` if `track_model` is non-None.
- `src/racingoptimizer/corner/states.py:226` -- `corner_phase_states`: parquet-schema intersect against `DEFAULT_CHANNELS` (Acura's missing shock-defl quad gracefully omitted); `LatAccel/LongAccel` renamed to `AccelLat/AccelLon` at boundary.
- `src/racingoptimizer/corner/states.py:290` -- `_thresholds_for_frame` derives `sample_rate_hz` from `t_s` deltas, honouring 360 Hz Cadillac telemetry. Regression test at `tests/corner/test_segment_lap.py:132`.
- `src/racingoptimizer/corner/states.py:359-386` -- VISION §2 damper-force: `_v_*shockDefl = abs(diff) * 1000` then digressive force estimate **inlined as Polars expression** (linear below knee, sqrt taper above). Lazy-imports `damper_coefficient` + `DIGRESSIVE_KNEE_MM_S` from `physics.damper_force` to break circular dep.
- `src/racingoptimizer/corner/states.py:64-71` -- `STEERING_GEOMETRY_COEFFICIENT` per-car table (S2.10): `bmw=0.06`, `acura=0.07`, `cadillac=0.06`, `ferrari=0.065`, `porsche=0.065`. Replaces forbidden `Speed^2` formula.
- `src/racingoptimizer/corner/states.py:599-627` -- `WindDir` circular mean via `atan2(mean(sin), mean(cos))` in a separate group_by pass.
- `src/racingoptimizer/physics/damper_force.py:19` -- `DAMPER_COEFFICIENT_NS_PER_MM` per-car (4-8 N·s/mm); `DIGRESSIVE_KNEE_MM_S=100.0` magic number documented as Stage-4 stepping-stone.
- `src/racingoptimizer/physics/damper_force.py:40` -- `estimate_damper_force_n` is the numpy reference; sign-preserving. **Not invoked in production** (Polars version reimplements the same formula).

## Wiring

**Upstream:** `ingest.api.lap_data`, `ingest.catalog`, `ingest.paths`.

**Downstream consumers of `corner_phase_states` / `Phase` / `detect_corners`:**
- `src/racingoptimizer/physics/fitter.py:447` -- per-(session, valid lap) call; pooled into per-car training frame.
- `src/racingoptimizer/physics/corner_schedule.py:84` -- per-lap, then grouped by `corner_id` for per-track archetype feeding the recommendation renderer.
- `src/racingoptimizer/physics/weights.py:59` -- time-sensitivity weighting per phase.
- `src/racingoptimizer/track/corner_loading.py:152` -- stacks across sessions for per-track corner-loading model.
- `src/racingoptimizer/track/corners.py:38`, `src/racingoptimizer/track/geometry.py:50` -- re-use `detect_corners` directly (without phase walker) for slice D's track-position model.
- `physics/score.py:20`, `physics/model.py:24`, `physics/recommendation.py:8`, `physics/phase_weights.py:14`, `physics/quali_mode.py:21`, `physics/wet_mode.py:20`, `explain/narrative.py:19`, `explain/comparison.py:6`, `explain/justification.py:20` -- all import `Phase` enum / `CornerPhaseKey`. Phase is the single vocabulary shared across the recommend pipeline.
- `physics/__init__.py:23-26` -- re-exports `damper_coefficient`, `estimate_damper_force_n`, `DIGRESSIVE_KNEE_MM_S`.

## Gaps

1. **MAJOR** -- `src/racingoptimizer/corner/states.py:205-208`: `segment_lap(track_model=...)` still raises `NotImplementedError`, but slice D is merged. `track.corners`/`track.geometry` re-use `detect_corners` directly instead of integrating via this kwarg as the spec promised.
2. **MINOR** -- `src/racingoptimizer/physics/damper_force.py:40`: `estimate_damper_force_n` exported but not used in production (`corner/states.py:380-385` reimplements the digressive curve as a Polars expression). Maintenance risk: divergence between numpy reference and Polars expression.
3. **MINOR** -- `src/racingoptimizer/corner/states.py:161-167`: `_REQUIRED_SEGMENT_COLUMNS` requires only 5 columns (`t_s, AccelLat, Brake, Throttle, SteeringWheelAngle`); spec §2 advertises 8 (`+ lap_dist_pct, Speed, YawRate`). Looser is fine in practice; spec wording is stale.
4. **MINOR** -- `src/racingoptimizer/physics/damper_force.py:19-30`: per-car damper coefficients are seeded estimates; single magic-number knee `100 mm/s`. Stage-4 deferral acknowledged in docstring + `docs/VISION_COMPLIANCE.md:340`.
5. **MINOR** -- `src/racingoptimizer/corner/states.py:54-71`: `STEERING_GEOMETRY_COEFFICIENT` per-car constant; spec/comments earmark Stage 3 to fit `k_car` per session. Not yet wired.
6. **MINOR** -- `src/racingoptimizer/corner/states.py:582-593`: `data_quality_clean_frac` averages a placeholder (all-True) mask; real curb / off-track masking from slice D doesn't yet feed through. (Same root cause as Slice D's HIGH finding -- `apply_quality_mask` is never called by production code.)
7. **MINOR** -- `src/racingoptimizer/corner/boundaries.py:33-81`: phase walker is per-corner Python loops over numpy slices -- not vectorised. No correctness impact; theoretical perf ceiling.
8. **NONE** -- Spec §11 punts ovals + cross-lap stitched corners; not gaps, just out-of-scope.

## Evidence
- Test suite: NOT RUN (sandbox blocked all `uv run pytest`). Static review of 56 in-scope tests across `tests/corner/test_boundaries.py:3`, `test_derived_columns.py:16`, `test_detect_synthetic.py:5`, `test_per_car_smoke.py:1` (parametrised across 5 cars), `test_phase_enum.py:3`, `test_segment_lap.py:7`, `test_states.py:5`, `test_steering_geometry.py:8`, and `tests/test_segment.py:4` shows: phase-walk coverage for every transition, hysteresis + min-duration + min-gap detector coverage, `_aggregate` direct arithmetic verification of all spec §6 columns, per-car smoke for all 5 cars asserting `corner_id >= 0` + channel-gated columns + understeer envelope, plus 360 Hz sample-rate regression.
- Lint: NOT RUN. No obvious style issues on read.
- Latest artefact: `recommendations/bmw-spa-reset-0506-1029.txt` (2026-05-06 10:29) renders per-corner narrative ("T9 mid-corner", "T17 mid-corner", "T1 mid-corner", "T18 under braking"), demonstrating the (corner_id, phase) keys flow end-to-end from this slice through the fitter into the recommendation renderer.

## Recommended next actions

- `src/racingoptimizer/corner/states.py:205-208` -- wire `track_model=` to `track.geometry` corner-position lookup (slice D is merged), or drop the kwarg and update spec §4. Don't keep dead `NotImplementedError`.
- `src/racingoptimizer/corner/states.py:380-385` -- collapse the duplicate digressive-curve formula. Either (a) call `estimate_damper_force_n` via `pl.Expr.map_batches`, or (b) delete the numpy version and have the test exercise the Polars expression on a single-row frame.
- `src/racingoptimizer/physics/damper_force.py:19-30` -- capture real per-car damper curves from iRacing UI and turn seeded scalars into a per-corner damper-rate table indexed by clicks.
- `src/racingoptimizer/corner/states.py:64-71` -- promote `STEERING_GEOMETRY_COEFFICIENT` to a fittable parameter inside `physics.fitter` per the Stage-3 plan in the source comment.
- Spec hygiene: update `docs/superpowers/specs/2026-04-28-corner-phase-design.md` §2 required-columns wording to match `_REQUIRED_SEGMENT_COLUMNS`.
