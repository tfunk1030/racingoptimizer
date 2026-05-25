# VISION §2 — Corner-Phase Decomposition (Audit)

**Section under audit:** VISION.md §2 "Corner-Phase Decomposition — Think like an engineer."
**Branch:** `audit/vision-section-02-corner-phase`
**Worktree:** `.claude/worktrees/agent-a2b20b2883ac12b51`
**Date:** 2026-05-05
**Test command:** `uv run pytest -q tests/corner/`
**Test result:** 72 passed in 12.88s (0 failed, 0 skipped on this run — IBT-loading parametrised tests in `test_per_car_smoke.py` ran against the materialised `ibtfiles/` corpus)

---

## VISION §2 verbatim

> For each lap, segment into individual corners using GPS/speed/lateral G. For each corner, decompose into phases: braking zone, trail-brake/entry, mid-corner (peak lateral load), exit/traction, and straight. For each phase, compute the actual physics state: understeer angle (from steering geometry vs lateral G), load transfer distribution (from shock deflection asymmetry), traction utilization (from wheel speed differentials), aero platform state (from ride height and pitch), roll angle and rate, damper velocities vs forces. This is the fundamental unit of analysis — not the lap, not the session.

## Score

**🟢 Implemented** — every clause has code + tests, phases are 5-bucket and corner-keyed end-to-end, and the per-car smoke test loops the 5 canonical fixtures.

The one substantive caveat is that two §2 derived states (understeer-angle coefficient and damper-force coefficient) are currently **seeded per-car constants** with `# stepping-stone` comments rather than empirically fit from telemetry. They are flagged in code as Stage-3 / Stage-4 follow-ups and are documented as such in `docs/VISION_COMPLIANCE.md`'s "Known follow-ups" §5.

## Per-clause scorecard

| §2 clause | Implementation | Evidence | Per-car coverage |
|---|---|---|---|
| Per-lap corner segmentation via GPS / speed / lateral G | Schmitt-trigger on `\|AccelLat\| / G_MS2` with hysteresis (entry 0.5 g, exit 0.3 g, exit_hold 200 ms, min_corner_duration 400 ms, min_gap 200 ms) | `src/racingoptimizer/corner/detect.py:19-72` (`detect_corners`); thresholds in `corner/config.py:17-34` | Sample-rate auto-derived from `t_s` per IBT in `_thresholds_for_frame` (`states.py:290-306`); per-car smoke runs all 5 canonical cars (`tests/corner/test_per_car_smoke.py`) |
| 5-phase decomposition (BRAKING, TRAIL_BRAKE, MID_CORNER, EXIT, STRAIGHT) | `Phase` StrEnum + `assign_phases` forward-only state machine | `corner/phase.py:14-19`; `corner/boundaries.py:25-81` (`assign_phases` with predicates on Brake/Throttle/Steering/AccelLat) | `tests/corner/test_phase_enum.py`, `tests/corner/test_boundaries.py`, `tests/corner/test_segment_lap.py` |
| Understeer angle from steering geometry vs lateral G | `understeer_signal = SteeringWheelAngle − k(car) · AccelLat` (textbook `Speed²` denominator was removed in S2.10) | `corner/states.py:53-95` (`STEERING_GEOMETRY_COEFFICIENT`, `steering_geometry_for`); aggregator at `states.py:454-462` emits `understeer_angle_mean_rad` | `tests/corner/test_steering_geometry.py`; per-car finite-and-bounded assertion in `test_per_car_smoke.py:110-117` |
| Load transfer distribution from shock-deflection asymmetry | `((LFshockDefl + RRshockDefl) − (RFshockDefl + LRshockDefl)) · 1000` mm, sign-positive on RF+LR diagonal | `corner/states.py:484-494` emits `load_transfer_asymmetry_mean` | `tests/corner/test_derived_columns.py:69-107` (loaded-diagonal, balanced, omitted-without-quad); per-car `test_per_car_smoke.py:77-89` |
| Traction utilization from wheel-speed differentials | `(max(*Speed) − min(*Speed)) / max(Speed, 1e-6)` clipped to [0, 1] | `corner/states.py:496-506` emits `traction_util_mean` | `tests/corner/test_derived_columns.py:116-157` (zero/known-fraction/clipped); per-car `test_per_car_smoke.py:96-100` |
| Aero platform state from ride height + pitch | front_rh = mean of LF+RF, rear_rh = mean of LR+RR, pitch = rear − front (mm; sign-positive = nose-down rake) | `corner/states.py:508-525` emits `aero_platform_front_rh_mean_mm`, `aero_platform_rear_rh_mean_mm`, `aero_platform_pitch_mean_mm` | `tests/corner/test_derived_columns.py:169-182, 310-333` (cross-row pitch consistency); per-car `test_per_car_smoke.py:91-94` |
| Roll angle and rate | Unsigned max + signed mean for `Roll`; unsigned max for `RollRate` | `corner/states.py:442-452` emits `roll_max_rad`, `roll_angle_mean_rad`, `roll_rate_max_rad_s` | `tests/corner/test_derived_columns.py:190-202` (signed mean); per-car `test_per_car_smoke.py:102-104` |
| Damper velocities vs forces | Velocity: `\|d(shockDefl)/dt\| · 1000` mm/s, p99 + mean across 4 corners. Force: per-car digressive curve `F(v) = k·v` below 100 mm/s knee, `k·knee·(1 + sqrt((v−knee)/knee))` above; coefficients seeded per-car (4–8 N·s/mm). | Velocity in `corner/states.py:359-366, 530-535` (`damper_velocity_p99_mms`, `damper_velocity_mean_mms`); force in `physics/damper_force.py:19-63` (`DAMPER_COEFFICIENT_NS_PER_MM`, `DIGRESSIVE_KNEE_MM_S`, `estimate_damper_force_n`) inlined as Polars expression at `states.py:367-386, 537-541` to keep a columnar pipeline (no per-sample Python UDF) | `tests/corner/test_derived_columns.py:213-278` (constant/ramp/sine); per-car non-negativity + p99 ≥ mean in `test_per_car_smoke.py:80-85`; force is also a fitter target so `tests/physics/test_target_output_channels.py` covers it |
| **Atomic unit is corner-phase, not lap, not session** | `corner_phase_states` group_by `(corner_id, phase)` returning one row per group, dropping `corner_id == -1` (out-of-corner samples) | `corner/states.py:226-277, 312-650` (the `_aggregate` group_by pipeline) | `tests/corner/test_per_car_smoke.py::test_corner_phase_states_runs_for_each_canonical_car` parametrised over the 5 canonical cars; downstream the physics fitter trains per `(corner, phase, channel)` (`physics/fitter.py:181-219`) and the BMW Spa briefing renders one (corner, phase) bullet per Helps/Hurts entry (e.g. `T9-straight`, `T8-braking`, `T1-trail_brake`, `T12-mid_corner`, `T7-exit`) |

## Code-tour highlights

**`src/racingoptimizer/corner/phase.py:14-19`** — the 5-bucket `Phase` StrEnum is the single source of truth for phase names. Every downstream column / test / score key cites these strings.

**`src/racingoptimizer/corner/detect.py:19-72`** — `detect_corners` is purely lateral-G-driven (Schmitt trigger with hysteresis), per-sample rate-aware via `ms_to_samples`. VISION §2 says "GPS/speed/lateral G" — only lateral G is used today; speed is consumed downstream by phase decisions inside `assign_phases` (e.g. throttle thresholds gate STRAIGHT exit). GPS / lap-distance is not used for **segmentation** but is preserved as `lap_dist_pct_start` / `lap_dist_pct_end` aggregates per phase. This is conservative-but-complete: the aero/track modules use `lap_dist_pct` for track-position mapping while corner segmentation stays car-centric (lateral G).

**`src/racingoptimizer/corner/boundaries.py:25-81`** — phase state machine. Every corner starts in BRAKING and walks forward only: BRAKING → TRAIL_BRAKE (when brake + steering + lat-g all active simultaneously, per `trail_brake_lat_g = 0.6 * lat_g_entry`) → MID_CORNER (when brake released for `brake_off_hold_ms`) → EXIT (when throttle on + lat-g decreasing over `accel_lat_decreasing_window_ms`) → STRAIGHT (when steering near-zero + throttle high). The docstring acknowledges the simplification that "phases the predicates never satisfy collapse to zero samples and are omitted by the downstream aggregator" — the group_by drops zero-sample groups naturally.

**`src/racingoptimizer/corner/states.py:53-95`** — `STEERING_GEOMETRY_COEFFICIENT` per-car table. The comment explicitly calls out the S2.10 fix: VISION §3 forbids the textbook bicycle-model `Speed²` denominator, so the placeholder was replaced with the per-car linear yaw-deficiency proxy `understeer_signal = SteeringWheelAngle − k(car) · AccelLat`. Coefficients are documented as "stepping-stones — order-of-magnitude calibration anchored to typical GTP steering racks (~0.06 rad of wheel angle per m/s² of lateral demand)" with Stage 3 follow-up to "refine `k_car` empirically per session out of the physics fitter."

**`src/racingoptimizer/corner/states.py:355-386`** — damper force is computed per-sample as a Polars expression so the four-corner velocity → force mapping never falls into a Python UDF. The digressive curve and per-car coefficient live in `physics/damper_force.py` to keep the corner module's circular-dependency surface minimal (the lazy import at line 371 is documented at lines 48-51).

**`src/racingoptimizer/physics/damper_force.py:19-63`** — `DAMPER_COEFFICIENT_NS_PER_MM` is the per-car force constant (BMW 6.0, Acura 5.5, Cadillac 6.5, Ferrari 6.0, Porsche 6.5). The module docstring labels this "a Stage-4 stepping-stone" pending real damper-spec data from iRacing's garage tooltips. The digressive knee at 100 mm/s and the sqrt taper above are an engineering estimate, not a per-car fit.

## BMW Spa recommendation card — phase evidence

Inspecting `recommendations/bmw__spa_2024_up__20260505-180530.txt`, every parameter block carries Helps / Hurts entries keyed on `T<corner>-<phase>` where `<phase>` is one of `braking`, `trail_brake`, `mid_corner`, `exit`, `straight`. Sample evidence (line refs are into the recommendation file):

- Line 8-10: `T9-straight`, `T8-braking`, `T6-braking` (Pushrod Length Offset Rear Helps)
- Line 14: `T8-trail_brake` (Pushrod Length Offset Rear Hurts)
- Line 28: `T12-mid_corner` (Third Perch Offset Rear Hurts)
- Line 50-52: `T1-trail_brake`, `T2-trail_brake`, `T4-trail_brake` (Damper Hsc Rl Helps)

All 5 phases appear under live keys with corner indices 0-17, confirming the `(corner, phase)` keying flows through training → scoring → recommendation rendering exactly as VISION §2 mandates.

## Test summary

```
$ uv run pytest -q tests/corner/
........................................................................ [100%]
72 passed in 12.88s
```

Suites covered:
- `tests/corner/test_detect_synthetic.py` — Schmitt-trigger corner segmentation against synthetic lateral-G profiles.
- `tests/corner/test_boundaries.py` — phase state-machine transitions on synthetic Brake/Throttle/Steering/AccelLat.
- `tests/corner/test_phase_enum.py` — 5-bucket enum identity / ordering.
- `tests/corner/test_segment_lap.py` — `segment_lap` end-to-end on synthetic frames.
- `tests/corner/test_states.py` — `corner_phase_states` aggregator structure.
- `tests/corner/test_derived_columns.py` — synthetic verification of every §6 derived column (load transfer asymmetry, traction util, aero platform, roll, damper velocities, data quality fraction).
- `tests/corner/test_steering_geometry.py` — per-car `STEERING_GEOMETRY_COEFFICIENT` lookups + understeer formula.
- `tests/corner/test_per_car_smoke.py` — parametrised over the 5 canonical car fixtures (BMW, Acura, Cadillac, Ferrari, Porsche): runs `corner_phase_states` end-to-end, asserts ≥1 corner detected, asserts the §6 derived columns are present (channel-conditional for Acura's missing per-corner shock channels) and finite, asserts the empirical understeer signal is bounded (max abs < 3.0 rad).

## Findings

### Strengths

1. **Phase contract is rigorously enforced.** The 5-bucket `Phase` StrEnum lives in one file (`corner/phase.py`), serialises as plain strings, and propagates unchanged through aggregation → fitting → scoring → recommendation rendering. The BMW Spa recommendation card surfaces all 5 phases under live `T<corner>-<phase>` keys.

2. **Atomic unit is genuinely (corner, phase), not session-averaged.** `_aggregate` group_by + `physics/fitter.py:181-219` train per `(corner, phase, channel)`, and the recommender attributes Helps / Hurts at that grain. No session collapsing leaks into the optimisation surface.

3. **Per-car coverage holds across the 5 canonical fixtures**, including Acura's missing per-corner shock channel set — the aggregator gracefully omits the shock-gated columns (`load_transfer_asymmetry_mean`, `damper_velocity_*`, `damper_force_*`) for Acura via the `has_shocks` gate at `states.py:335-337`, without crashing the pipeline.

4. **Textbook-formula prohibition (VISION §3) is honoured for the understeer signal** — the `Speed²` denominator was removed in S2.10 and replaced with a per-car linear empirical proxy. The replacement coefficients are explicitly labelled stepping-stones with a Stage-3 follow-up to fit them empirically per session.

5. **Damper velocity AND force are both emitted at the corner-phase grain** as VISION §2 demands. The force estimator is inlined as a Polars expression so the per-sample mapping stays columnar.

6. **Sample rate is auto-detected per IBT** in `_thresholds_for_frame` (`states.py:290-306`) by inverting the median positive `t_s` delta, so the corner detector's hysteresis windows scale correctly across 60 Hz / 360 Hz recordings.

### Gaps / follow-ups (already documented in the codebase)

1. **Per-car steering-geometry coefficient `k(car)` is a seeded constant**, not a per-session empirical fit. Documented at `corner/states.py:60-66`: "Stage 3 will refine `k_car` empirically per session out of the physics fitter; until then these constants keep the column finite and per-car-distinguishable." This is a Stage-3 deferred item — the column is finite and per-car-distinguishable today (per-car smoke asserts `abs().max() < 3.0`), but the coefficient is not yet learned from telemetry as VISION §3 mandates for the broader physics model.

2. **Per-car damper-force coefficient is a seeded constant** (4–8 N·s/mm range) pending real damper-spec data from iRacing's garage tooltips. Documented in `physics/damper_force.py:6-12` and in `docs/VISION_COMPLIANCE.md` known follow-ups §5: "Force estimation is wired into the corner aggregator (`damper_force_p99_n` / `damper_force_mean_n` columns), but absolute magnitudes are stepping-stone values." The digressive knee (100 mm/s) and the sqrt taper above the knee are also engineering estimates, not per-car fits.

3. **Corner segmentation uses lateral-G only** (Schmitt trigger), not GPS/speed jointly. VISION §2 says "GPS/speed/lateral G". Speed is consumed inside the `assign_phases` predicates downstream (throttle thresholds for STRAIGHT) and `lap_dist_pct` is preserved per-phase as start/end aggregates; the track module (slice D) then maps phases to track-position bins. So the GPS/speed signal is used in the broader pipeline but not in the `detect_corners` boundary detector itself. This is a defensible design — purely-lateral-G keeps segmentation track-agnostic — but worth flagging since the prose lists all three signals.

4. **BRAKING-only entry to the phase state machine.** `boundaries.py:9` notes "every corner starts in BRAKING. Phases the predicates never satisfy (e.g. a flat-throttle corner with no real braking) collapse to zero samples and are omitted by the downstream aggregator." For a coast-into-corner scenario, the BRAKING phase row would be absent from the output rather than the corner being labelled as starting in TRAIL_BRAKE or MID_CORNER. The downstream group_by drops the zero-sample group cleanly so this does not corrupt aggregates, but the corner inventory may show fewer phases than expected for non-braking corners.

### No regressions found

- 5-bucket phase contract is intact and uniform across `corner/`, `physics/`, `explain/`.
- Atomic-unit invariant (one row per `(corner_id, phase)`, no lap collapsing) is preserved end-to-end into the recommendation card.
- All §2 listed derived states have aggregator columns AND synthetic-input unit tests AND per-car smoke-test coverage.
- Out-of-corner samples (`corner_id == -1`) are dropped by the aggregator (`states.py:346-348`) so STRAIGHT-between-corners noise never contaminates phase aggregates.

## Files cited

Implementation (absolute paths):
- `C:\Users\VYRAL\racingoptimizer\src\racingoptimizer\corner\phase.py`
- `C:\Users\VYRAL\racingoptimizer\src\racingoptimizer\corner\config.py`
- `C:\Users\VYRAL\racingoptimizer\src\racingoptimizer\corner\detect.py`
- `C:\Users\VYRAL\racingoptimizer\src\racingoptimizer\corner\boundaries.py`
- `C:\Users\VYRAL\racingoptimizer\src\racingoptimizer\corner\states.py`
- `C:\Users\VYRAL\racingoptimizer\src\racingoptimizer\corner\__init__.py`
- `C:\Users\VYRAL\racingoptimizer\src\racingoptimizer\physics\damper_force.py`
- `C:\Users\VYRAL\racingoptimizer\src\racingoptimizer\physics\fitter.py` (lines 56-115 for `TARGET_OUTPUT_CHANNELS`)

Tests:
- `C:\Users\VYRAL\racingoptimizer\tests\corner\test_phase_enum.py`
- `C:\Users\VYRAL\racingoptimizer\tests\corner\test_detect_synthetic.py`
- `C:\Users\VYRAL\racingoptimizer\tests\corner\test_boundaries.py`
- `C:\Users\VYRAL\racingoptimizer\tests\corner\test_segment_lap.py`
- `C:\Users\VYRAL\racingoptimizer\tests\corner\test_states.py`
- `C:\Users\VYRAL\racingoptimizer\tests\corner\test_derived_columns.py`
- `C:\Users\VYRAL\racingoptimizer\tests\corner\test_steering_geometry.py`
- `C:\Users\VYRAL\racingoptimizer\tests\corner\test_per_car_smoke.py`

Spec:
- `C:\Users\VYRAL\racingoptimizer\docs\superpowers\specs\2026-04-28-corner-phase-design.md`

Live recommendation card showing 5-phase per-corner attribution end-to-end:
- `C:\Users\VYRAL\racingoptimizer\recommendations\bmw__spa_2024_up__20260505-180530.txt`

Baseline cross-reference:
- `C:\Users\VYRAL\racingoptimizer\docs\VISION_COMPLIANCE.md` §2 (lines 95-109)
