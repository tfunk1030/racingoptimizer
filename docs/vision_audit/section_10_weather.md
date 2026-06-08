# VISION §10 — Weather & Track Conditions

**Section:** §10 — *Understand the environment.*
**Date:** 2026-05-05
**Auditor scope:** `EnvironmentFrame` channel completeness, env threading
through fitter + scorer + CLI, `--air-temp/--track-temp/--wind/--wetness`
overrides, wet-mode regime branching, air-density correction on aero
lookups, BMW Spa briefing-card evidence line.

## VISION clauses → evidence

| Clause | Status | Evidence |
|---|---|---|
| 12 weather/conditions channels at 60 Hz: `AirTemp`, `AirDensity`, `AirPressure`, `RelativeHumidity`, `WindVel`, `WindDir`, `FogLevel`, `TrackTempCrew`, `TrackWetness`, `WeatherDeclaredWet`, `PrecipType`, `Skies` | 🟢 | `src/racingoptimizer/context/environment.py:28-65` defines `_FLOAT_CHANNELS` (9), `_BOOL_CHANNELS` (1), `_INT_CHANNELS` (2) — full 12 enumerated; `EnvironmentFrame` dataclass has matching fields with NaN/-1/False sentinels. `from_row` is strict; `from_partial_row` is the degraded-mode adapter. |
| Env per (corner, phase) at training time | 🟢 | `physics/fitter.py:95-114` — `_ENV_COLUMNS` lists the same 12 fields suffixed `_mean`/`_max`; `physics/fitter.py:428-448` concatenates the `param_block` with the `env_block` into the joint feature matrix `X`; `ENV_FEATURE_SCHEMA_VERSION = 3` (line 127) is folded into the on-disk pickle key. |
| Env at predict time | 🟢 | `physics/score.py:283` — `score_setup(model, setup, track, env, ...)` takes an `EnvironmentFrame` and threads it through `aggregate_utilization` (line 232-276) into every sub-utilization (`grip`, `balance`, `stability`, `traction`, `aero_eff`, `platform`); `_aero_ld_for_state` (line 352-368) passes `env.air_density` to `AeroSurface.interpolate`. |
| CLI overrides (`--air-temp / --track-temp / --wind / --wetness`) wire to env frame | 🟢 | `cli/recommend.py:61-76` — four Click options; `cli/recommend.py:657-681` `_env_from_overrides` builds the frame, defaulting each non-None override into the corresponding field, otherwise falling back to per-corpus medians. Verified flow: `recommend_cmd` → `_env_from_overrides` → `model.recommend(fit_track, env, ...)`. |
| Air-density correction on aero lookup | 🟡 | `physics/score.py:88` applies `density_factor = env.air_density / BASELINE_AIR_DENSITY` to the absolute downforce contribution inside `grip()`. Per `aero/interpolator.py:8-19` audit comment, `ld_ratio` is intentionally density-invariant (it's a dimensionless ratio); the density correction only lives in `grip`, not in `aero_eff`. This is a deliberate design choice but the `air_density` argument to `AeroSurface.interpolate` is then a no-op except for input validation — caller must remember to apply the rho factor at the call site. |
| Wet-mode special case (dry/damp/wet/full_rain) | 🔴 | `physics/wet_mode.py:49-105` defines `classify_conditions`, `wet_baselines`, `wet_phase_weights` and is exported from `physics/__init__.py:48-50, 70, 82-83`. **Module is dead code in the production path.** No call to any of the three functions exists in `score.py`, `recommend.py`, `model.py`, or `cli/recommend.py` (verified by grep). The score/recommend pipeline always uses the dry-baseline `CarBaselines` and the unmodified `PHASE_WEIGHTS` table regardless of the env's `track_wetness` / `precip_type` / `weather_declared_wet`. Wet-mode tests pass (5/5) but only exercise the helpers, not the integration. |
| Wind directional asymmetry (headwind/crosswind) | 🔴 | `physics/wind.py:1-99` defines `decompose_wind` and `aero_wind_modifier`. Module docstring (lines 7-12) explicitly admits *"It is **not** wired into `racingoptimizer.physics.score.aero_eff` in this slice (S4.6); a future PR (Stage 5 polish) will route the modifier into the aero-utilization branch of the scorer."* Confirmed by grep — no production call site. `WindVel` and `WindDir` reach the joint feature vector as raw scalars (so the GP/RF surrogate sees them as features) but the directional decomposition into headwind/crosswind that VISION §10 calls out specifically is not applied to the aero correction. |
| BMW Spa briefing card "Conditions" line reports all 4 (AirTemp, AirDensity, Wind, Wetness) | 🔴 | `recommendations/bmw__spa_2024_up__20260505-180530.txt:2` reads: `Conditions: AirTemp 20.6 C  AirDensity 1.148 kg/m^3  Wind 3.5 m/s  Wetness 0.00`. All 4 fields present and formatted correctly. **However:** `src/racingoptimizer/explain/render_text.py:144` renders `f"AirTemp {env.track_temp_c:.1f} C"` — i.e. the field labelled "AirTemp" is actually the **track surface temperature**, not the air temperature (`env.air_temp_c` is never referenced from the explain module). The user thinks they're seeing 20.6 °C ambient air; they're actually seeing 20.6 °C track surface temp. |

## Findings detail

### F1 (🔴) — Wet-mode branching never executes at recommend time

`classify_conditions(env)` is defined (`wet_mode.py:49-64`) and tested
(`tests/physics/test_wet_mode.py`, 5 passing) but **no production code
calls it**. The recommender always invokes `score_setup` with the
dry-baseline `model.resolved_baselines` and the unmodified
`PHASE_WEIGHTS`. So:

- A session ingested at `track_wetness=0.85` (full rain) is scored with
  the same per-corner weights and the same `aero_grip_baseline_g` as a
  bone-dry session.
- The "VISION §10: wet conditions fundamentally change everything"
  clause is implemented at the helper layer but bypassed at the
  optimization layer.

Fix outline: in `score.score_setup` / `score_breakdown`, branch on
`classify_conditions(env)`; if regime != `"dry"`, swap
`baselines = wet_baselines(model.car, regime)` and override the per-phase
weight lookup with `wet_phase_weights(regime)`. This is one or two new
function calls + a ~5-line wiring change inside `score._score_breakdown`.

### F2 (🔴) — Wind directional decomposition not wired into aero

`decompose_wind` and `aero_wind_modifier` exist as pure helpers
(`physics/wind.py`) and the module docstring acknowledges they are not
called from `aero_eff`. Raw `WindVel` and `WindDir` reach the joint
feature vector as scalars — so the surrogate has access to them — but
the explicit physics correction VISION §10 prescribes (headwind on a
straight scales downforce up via `(V_air / V_baseline)²`; crosswind
shifts aero balance) is never applied. The fix is the work the
docstring's "Stage 5 polish" comment defers.

### F3 (🔴) — "AirTemp" briefing label shows track temp, not air temp

`explain/render_text.py:144` reads `f"AirTemp {env.track_temp_c:.1f} C"`.
The label `AirTemp` is literally the field name `track_temp_c`. The BMW
Spa card line `Conditions: AirTemp 20.6 C  AirDensity 1.148 kg/m^3
Wind 3.5 m/s  Wetness 0.00` is therefore mislabelled — that 20.6 °C is
the track surface, not ambient air. Two likely fixes:

1. Change the label to `TrackTemp` (cheapest, but loses the actual air
   temperature from the briefing).
2. Change the field to `env.air_temp_c` (matches the label) and add a
   second `TrackTemp` field to keep the surface temperature visible.

Either way, this is a real user-facing bug — VISION §10 lists `AirTemp`
and `TrackTempCrew` as separate channels with distinct effects (air
density vs tire grip / degradation), so conflating them in the briefing
defeats the whole point of carrying both.

### F4 (🟡) — `air_density` argument to `AeroSurface.interpolate` is a no-op

`aero/interpolator.py:113-128` accepts `air_density` and validates it's
positive, but the documented behaviour (lines 8-19) is that `ld_ratio`
stays density-invariant and callers must apply the `rho/rho_baseline`
factor themselves at their own use-site. Only `score.grip` (line 88)
does this; `score.aero_eff` (line 155-156) takes the raw `ld` straight
from the interpolator without a density factor. Architecturally
defensible (aero_eff measures L/D efficiency, which is genuinely
density-invariant) but the API surface is easy to misuse — a future
caller adding a new aero-utilization branch will probably forget to
apply the rho factor and lose the correction.

## Per-car verification scope

| Car | Env channels populated in joint feature vector | Wet-mode reachability |
|---|---|---|
| bmw | ✅ (12/12 via `_ENV_COLUMNS`) | ❌ (helper unused) |
| acura | ✅ | ❌ |
| cadillac | ✅ | ❌ |
| ferrari | ✅ | ❌ |
| porsche | ✅ | ❌ |

Per-car coverage of the env feature vector itself is implicit in the
`tests/physics/test_per_car_fit_predict.py` parametrisation (the joint
fitter trains over `param_block ⨁ env_block` for each car). No
per-car wet-mode integration test exists because the integration
itself is missing.

## Test results

`uv run pytest -q tests/context/ tests/physics/test_wet_mode.py
tests/physics/test_wind.py` (in worktree, dev install):

```
....................................                                     [100%]
36 passed in 0.08s
```

Breakdown:

- `tests/context/test_environment.py` — covers `EnvironmentFrame` field
  set, strict `from_row` raising on missing channels, sentinel defaults
  in `from_partial_row`.
- `tests/physics/test_wet_mode.py` — 5 tests covering
  `classify_conditions` thresholds, `wet_baselines` scale per regime,
  `wet_phase_weights` aero shift.
- `tests/physics/test_wind.py` — covers `decompose_wind` headwind /
  crosswind sign convention and `aero_wind_modifier` downforce-scale
  clamp.

All tests exercise the helpers in isolation — none exercise integration
into the score/recommend pipeline (because there is none).

## BMW Spa briefing-card evidence

`recommendations/bmw__spa_2024_up__20260505-180530.txt:1-3`:

```
bmw @ spa_2024_up - recommended setup
Conditions: AirTemp 20.6 C  AirDensity 1.148 kg/m^3  Wind 3.5 m/s  Wetness 0.00
Confidence: dense (n=2330 backing samples for the dominant dense parameter, 46 parameters reported)
```

All 4 conditions fields are reported — but per F3 the `AirTemp` value is
actually `track_temp_c`. `AirDensity 1.148 kg/m³` is plausible for a
warm Spa session (below the 1.225 ISA baseline → less downforce, less
drag, the density-correction in `grip` will scale the absolute downforce
component down by ~6%). `Wind 3.5 m/s` and `Wetness 0.00` are sane.

## Score: 🟠 — partial

The 12-channel ingest contract is solid. Env reaches the fitter and
the scorer. CLI overrides wire correctly. The numeric flow works.
**But three integration gaps undercut the spirit of §10:**

1. Wet-mode regime branching exists as helpers and is exported, but
   nothing in the production path calls them — wet sessions are scored
   with dry baselines + dry phase weights.
2. Wind directional decomposition exists as helpers and is exported,
   but nothing in the production path calls them — the surrogate sees
   raw `WindVel` / `WindDir` scalars but the explicit
   headwind/crosswind aero correction is missing.
3. The user-visible `Conditions: AirTemp X.X C` line in the briefing
   actually renders `track_temp_c` — the label is wrong.

The first two are deferred-by-design (the wind module docstring
acknowledges the deferral; wet_mode has no equivalent acknowledgement).
The third is a one-line bug.

## Recommended fixes

1. **Wire wet_mode into score_setup.** In
   `physics/score._score_breakdown` (or its caller) compute
   `regime = classify_conditions(env)`, then for non-dry regimes use
   `wet_baselines(model.car, regime)` and `wet_phase_weights(regime)`
   instead of the model's resolved baselines / `PHASE_WEIGHTS`. Add a
   per-car integration test asserting that the same setup at
   `track_wetness=0.8` scores differently from `track_wetness=0.0`.

2. **Wire wind into aero_eff** (or a dedicated aero-correction step in
   the scoring pipeline). Use `decompose_wind(env.wind_vel_ms,
   env.wind_dir_deg, car_heading_at_phase)` then
   `aero_wind_modifier(...)` to get a `downforce_scale` and
   `balance_shift_pct`. Apply both to the aero query result.
   `car_heading_at_phase` is currently absent from `EnvironmentFrame`
   and `CornerPhaseStateWithConfidence` — needs to be derived from the
   per-corner geometry in the track model. Until that derivation lands,
   a corner-bearing-agnostic scalar headwind correction
   (`headwind = -env.wind_vel_ms`, conservative) is at least better
   than zero correction.

3. **Fix the AirTemp label bug** in `explain/render_text.py:141-148`.
   Either rename the label to `TrackTemp` (one-line change, but loses
   the actual air-temperature signal from the briefing) or render both
   air and track temperatures explicitly:

   ```python
   f"AirTemp {env.air_temp_c:.1f} C  TrackTemp {env.track_temp_c:.1f} C  "
   f"AirDensity {env.air_density:.3f} kg/m^3  "
   f"Wind {env.wind_vel_ms:.1f} m/s  Wetness {env.track_wetness:.2f}"
   ```
