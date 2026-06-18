# Cross-cutting consistency — `EnvironmentFrame`, `Confidence`, `constraints` — Audit (2026-05-05)

## Summary
- Compliance grade: **PASS with caveats**
- All three cross-cutting modules have a single canonical implementation, are imported (not re-implemented) by every consumer that needs them, and the per-car v3 fitter / recommender / score path threads them end-to-end. Every recommended parameter carries a `Confidence` (no bare floats), and the constraint clamp runs both inside the DE objective AND a second time at briefing-render time before the user sees a value.
- Caveats:
  1. **Conditions header label is wrong.** `_conditions_line` (`explain/render_text.py:144`) prints `f"AirTemp {env.track_temp_c:.1f} C"` — pulling the **track** temperature but labeling it **air** temperature.
  2. **Model cache key folds ontology but NOT constraints.** `_model_cache_path` (`cli/recommend.py:632-654`) digests `session_ids + ontology + feature-schema-version` only. Edits to `constraints.md` do not invalidate the cached pickle, so a stale `model.constraints` (frozen on the pickle) is reused; the search runs against the old bounds. The CLI's `_post_clamp` re-clamps using `load_constraints()` at render time, so the user never sees an out-of-bound value, but the DE search itself is silently behind.
  3. **`wet_mode` and `wind` are dead code in the recommend/score path.** `classify_conditions`, `wet_baselines`, `wet_phase_weights`, `aero_wind_modifier`, `decompose_wind` are exported and unit-tested but never called outside their own modules. The recommender always uses dry baselines + dry phase weights regardless of `env.track_wetness` / `env.wind_*`.
  4. **JSON renderer truncates env to 5 of 12 channels.** `_env_to_json` (`explain/render_json.py:105-112`) emits only `air_density / track_temp_c / wind_vel_ms / wind_dir_deg / wetness`. The other seven (`air_temp_c, air_pressure_mbar, relative_humidity, fog_level, weather_declared_wet, precip_type, skies`) survive in the model but are dropped at the JSON boundary.
  5. **CLI re-implements its own env-channel map.** `_ENV_FLOAT_CHANNELS` / `_ENV_DISCRETE_CHANNELS` in `cli/recommend.py:689-704` parallel `_FLOAT_CHANNELS` / `_BOOL_CHANNELS` / `_INT_CHANNELS` in `context/environment.py:28-45`. Today they agree; nothing prevents drift.

## Section text (verbatim, distilled from VISION.md / CLAUDE.md)

VISION does not have a single "section 11"; the three modules cover several VISION clauses:

- **`EnvironmentFrame`** realises VISION §10 ("every data point should carry its environmental context"). 12 channels at 60 Hz: `AirTemp`, `AirDensity`, `AirPressure`, `RelativeHumidity`, `WindVel`, `WindDir`, `FogLevel`, `TrackTempCrew`, `TrackWetness`, `WeatherDeclaredWet`, `PrecipType` (`Precipitation`), `Skies`.
- **`Confidence`** realises VISION §6 / §7 ("when the model is uncertain, be conservative ... when it's confident, be aggressive ... report confidence alongside every recommendation").
- **`constraints` (load_constraints / ConstraintsTable / clamp)** realises VISION §1 / §3 / §5 / §7 / §8 (CLAUDE.md: "the optimizer MUST clamp every recommended value to these bounds before output").

Cross-cutting commitment from CLAUDE.md:

> The five cars have different suspension architectures, IBT YAML setup-blob shapes, and aero-map step sizes. Do not assume a unified setup schema across cars.

## What the code does today

### EnvironmentFrame (canonical: `src/racingoptimizer/context/environment.py`)

**Single source of truth.**
- Frozen, slotted dataclass with the 12 fields (lines 48-64). Sentinel defaults: `NaN` for floats, `False` for the bool, `-1` for ints — so `EnvironmentFrame()` is a valid "all-missing" frame and consumers can detect missingness without exceptions.
- IBT-channel ↔ field-name map lives once at module top: `_FLOAT_CHANNELS`, `_BOOL_CHANNELS`, `_INT_CHANNELS` (lines 28-45). `from_row` (strict) and `from_partial_row` (degraded mode) iterate the same tuples — they cannot drift from each other.
- Strict `from_row` raises `KeyError` naming the missing channel (line 76, tested at `tests/context/test_environment.py::test_from_row_missing_*`). Slice E "needs to know when env was unavailable" — silent zero-filling is explicitly forbidden in the docstring.

**Consumers.**
- `src/racingoptimizer/physics/model.py:23,188,197,206,218,286,371,392` — `PhysicsModel.predict / score_setup / recommend` all take `env: EnvironmentFrame` as a positional arg. `_env_to_array` (line 392) flattens the 12 fields into a NumPy row in `_ENV_COLUMNS` order. NaN-coerces to 0.0 because sklearn rejects NaN at predict time (line 421).
- `src/racingoptimizer/physics/score.py:19,75,98,112,134,149,171,235,283,311,354,445` — every sub-utilization (`grip / balance / stability / traction / aero_eff / platform`) takes `env: EnvironmentFrame`. `grip` uses `env.air_density` for the aero-map air-density correction (line 88) per VISION §10's "the aero maps need an air density correction factor".
- `src/racingoptimizer/physics/recommend.py:29,46,417` — recommend signature is `recommend(model, track, env, constraints)`. The DE objective passes `env` through every score evaluation.
- `src/racingoptimizer/physics/recommendation.py:7,15` — the `SetupRecommendation` dataclass carries the `env: EnvironmentFrame` it was generated under, so the briefing knows the conditions the recommendation is tuned for.
- `src/racingoptimizer/physics/wet_mode.py:19,49` — `classify_conditions(env)` reads `env.track_wetness`, `env.weather_declared_wet`, `env.precip_type`. **NOT WIRED — see Gap 3.**
- `src/racingoptimizer/cli/recommend.py:22,666,668` — `_env_from_overrides` builds the live `EnvironmentFrame` for the recommend pipeline by taking per-channel medians across every clean sample in the (car, track) corpus (`_environment_from_corpus`, lines 726-793). Discrete channels are aggregated by `.max()` ("any-wet wins"); `WindDir` uses a circular median (line 824) to avoid the [350°, 10°] → 180° trap.

**Per-car v3 fitter env wiring (the user-asked check).**
- `src/racingoptimizer/physics/fitter.py:96-114` — `_ENV_COLUMNS` lists the 12 corner-phase env aggregates (`air_temp_c_mean`, `air_density_mean`, …, `precip_type_max`, `skies_max`) in the same order as `_env_to_array` in `model.py`. The fitter pulls these columns out of the corner-phase frame for every quadruple at training time and pads with zeros if a column is missing (lines 428-446). The same 12-element env vector is appended to the setup-parameter feature row in every fit AND every predict.
- The wide schema enumerated in `src/racingoptimizer/corner/states.py:132-148` materialises all 12 IBT env channels per (corner, phase) row, so the joint surrogate sees every per-corner-phase env aggregate — no channel is silently dropped between corner aggregation and the fitter.
- The per-car corner aggregator pulls the 12 channels for every car (Acura corner-phase output suffers no env loss even when shock-deflection channels are missing — env survives independently).

### Confidence (canonical: `src/racingoptimizer/confidence/confidence.py`)

**Single source of truth.**
- Frozen 5-tuple `(value, lo, hi, n_samples, regime)` (lines 31-50). `__post_init__` enforces `lo <= value <= hi`, `n_samples >= 0`, and `regime in {"sparse", "noisy", "confident", "dense"}`.
- `Confidence.derive(value, n_samples, cv_residual_std, signal_std)` (lines 52-84) classifies regime: `n_samples < 30` short-circuits to `sparse`; otherwise a noise-ratio of `cv_residual_std / signal_std` partitions into `noisy > 0.5`, `confident > 0.2`, `dense ≤ 0.2`. Bracket is `value ± 1.96 * cv_residual_std` (the Gaussian 95% bracket) per the physics-fitter spec §3.

**Consumers.**
- `src/racingoptimizer/physics/model.py:21,57,243,264,280,315,349,355,386,425,431,439` — every `PhysicsModel.predict` output channel returns a `Confidence` (`CornerPhaseStateWithConfidence.states: dict[str, Confidence]`). `_maybe_downgrade_aero` (line 425) drops the regime one tier for ride-height channels when slice C is unavailable, satisfying spec §9.
- `src/racingoptimizer/physics/score.py:16,17,32,68,93,107,120,127,165,182,202,213,219,243,251,260-275,328,339` — every sub-utilization returns `(float, Confidence)`. `aggregate_utilization` rolls up per-sub `Confidence` into a phase-aggregate `Confidence` using the worst-regime rule (line 245) and a weighted bracket sum (line 268).
- `src/racingoptimizer/physics/recommend.py:27,149,188,260,264,374,386,391,400,411,421` — `recommend` returns `parameters: dict[str, tuple[float, Confidence]]`. `_parameter_confidence` (line 374) collapses every fitter that depends on a parameter into a single `Confidence` via medians of `n_samples / cv_residual_std / signal_std`. Untrained-parameter baselines also get a (zero-sample, sparse) `Confidence` so the renderer iterates a single dict.
- `src/racingoptimizer/physics/recommendation.py:6,16` — `SetupRecommendation.parameters: dict[str, tuple[float, Confidence]]`. **NO bare-float parameter exists.**
- `src/racingoptimizer/explain/justification.py:18,44,58,304,310,316` — `SetupJustification.confidence` is required (`__post_init__` raises `IncompleteJustificationError` if missing). Evidence text reads `f"{regime} confidence backed by {n_samples} samples"`.
- `src/racingoptimizer/explain/render_text.py:154-166` — the briefing's roll-up confidence line is `min` over the per-parameter `Confidence.regime`, so a single sparse parameter masks an otherwise dense fit (spec §14).
- `src/racingoptimizer/cli/recommend.py:16,515-530` — `_force_sparse_regime` overrides every parameter's `Confidence.regime` to `sparse` in the untrained-track extrapolation flow, using `dataclasses.replace` (frozen-dataclass-safe).
- `src/racingoptimizer/physics/io_log.py:37,72` — historical fit-quality log uses `Confidence.derive` to reconstruct a per-fit regime tag.

### constraints (canonical: `src/racingoptimizer/constraints/{__init__.py,loader.py,clamp.py}`)

**Single source of truth.**
- `loader.py` parses `constraints.md` into a `ConstraintsTable(_by_car: dict[str, dict[str, (lo, hi) | None]])`. `default` is the universe of known parameters; per-car keys *shadow* defaults and never introduce new parameters (line 33).
- TODO placeholders parse to `None` (line 56), so `bounds(...) is None` is the canonical "known parameter, unbounded" signal — distinct from "unknown parameter".
- `clamp(value, parameter, car, table=None)` (`clamp.py:41-64`) returns a `ClampResult(value, was_clamped, bound, status)` with four explicit statuses: `ok / clamped / unbounded / unknown_parameter`. `_to_canonical(car)` (lines 23-30) routes raw filename prefixes (e.g. `acuraarx06gtp`) through `normalize_car_key`, so the clamp survives both canonical and detected car keys.

**Consumers.**
- `src/racingoptimizer/physics/fitter.py:32,160` — `fit` calls `load_constraints()` once and passes to `fittable_parameters(car, constraints)`. Parameters with `bounds(car, name) is None` join `untrained_parameters` (line 165). The `constraints` table is stashed on the `PhysicsModel` (line 284).
- `src/racingoptimizer/physics/score.py:18,408-431` — `_clamped_or_raise` clamps every constrained parameter on every objective evaluation, using `model.constraints` (preferred) and the global `clamp(...)` (fallback path). `strict=True` raises on drift; the recommend post-clamp uses this to detect optimizer bugs.
- `src/racingoptimizer/physics/recommend.py:28,151-156` — after the DE search emits a candidate, every value goes through `clamp(...)` again with the live (pin-narrowed) `constraints` table. If the clamp moved the value > 1e-9, `recommend` raises `ValueError("recommend produced out-of-bounds value")` — defensive contract that DE's `bounds=…` argument actually held.
- `src/racingoptimizer/cli/recommend.py:17-20,124,144,839-902` — `_post_clamp` runs a SECOND clamp at render time, using the FRESH `load_constraints()` table (not the model's pickled copy). This is what saves the user from the cache-key-stale-constraints bug (Gap 2): even if the search ran against an old `model.constraints`, the rendered value is re-clamped against the on-disk file before the briefing prints. Discrete-click parameters (ARB blade index, damper clicks) are also rounded to int after clamp here (lines 882-895).
- `src/racingoptimizer/explain/justification.py:19,213,275-285` — `_step_for` and `_clamp_value` use `model.constraints` to compute the ±1-click sensitivity bounds and to clamp shifted values back into legal range.
- `src/racingoptimizer/physics/garage_inventory.py:12,51` — uses `ConstraintsTable` for inventory checks against the parsed setup blob.
- `src/racingoptimizer/physics/ontology.py:20,527` — `fittable_parameters(car, table)` lists every parameter that is both `spec.fittable` AND has `bounds(car, name) is not None`; this gates the recommender's search space and the `untrained_parameters` list.

**Briefing-time clamp ordering (the user-asked check).**
- The clamp runs **THREE times** before the briefing is rendered:
  1. Inside the DE objective (`_score_breakdown` → `_clamped_or_raise` in `score.py:297,316,408`).
  2. Defensively after DE finishes (`recommend.py:151`, `_clamped_or_raise` with `strict=False`; raises if a value would move > 1e-9).
  3. AGAIN at CLI render time (`cli/recommend.py:144` `_post_clamp` using the freshly-loaded global table).
- Discrete-click rounding happens inside step 3, so the user-facing value (e.g. `Anti Roll Bar Rear: 1.00 click`) is always integer-valued for discrete parameters.

## Evidence from artefacts

**Test suite.** `uv run pytest -q tests/context/ tests/confidence/ tests/constraints/` → **48 passed in 1.10s**. Coverage includes:
- `tests/context/test_environment.py` (10 tests): full 12-channel round-trip, NaN sentinels, `FrozenInstanceError` on mutation, strict-from-row missing-key behaviour for both old and new (S2.2) channels, partial-from-row degraded mode.
- `tests/confidence/test_regime_derivation.py` (12 tests): construct round-trip, frozen-on-mutate, `__post_init__` validation for the four invariants, parametrised regime table (`sparse / noisy / confident / dense`), 95% Gaussian bracket assertion, zero-signal-std fallback to `noisy`.
- `tests/constraints/test_clamp.py` (9 tests): per-car overrides (Acura wing 6-10), default fallback (BMW wing 12-17), tyre pressure floor 152, unbounded vs unknown-parameter status semantics, raw-filename-prefix normalisation, boundary equality is `ok`, explicit table acceptance.
- `tests/constraints/test_loader.py` (17 tests): every parameter the recommender depends on (wing, tyre pressure, heave spring/slider, static ride heights), TODO placeholders return `None`, malformed file raises `ConstraintsParseError`, missing file raises `FileNotFoundError`, explicit-path load.

**BMW Spa briefing card** (`recommendations/bmw__spa_2024_up__20260505-181936.txt`):
- Line 2: `Conditions: AirTemp 20.6 C  AirDensity 1.148 kg/m^3  Wind 3.6 m/s  Wetness 0.00` — the env channels are surfaced (but see Gap 1: `AirTemp` is mislabelled — value comes from `env.track_temp_c`).
- Line 3: `Confidence: dense (n=2357 backing samples for the dominant dense parameter, 46 parameters reported)` — the roll-up confidence is the worst regime across all 46 parameters.
- Every parameter line carries `[confidence: ...]`. Spot-check:
  - L5 `Pushrod Length Offset Rear Mm: -30.09 mm   [confidence: dense]`
  - L19 `Heave Perch Offset Front Mm: -22.33 mm   [confidence: dense]`
  - L471 `Rear Wing Angle: 15.87 deg   [confidence: dense]` — within BMW default bounds (12.0, 17.0) ✓
  - L573 `Heave Spring Rate N Per Mm: 48.47 N/mm   [confidence: dense]` — within BMW override (30, 100) ✓
  - L587 `Tyre Cold Pressure: 152.40 kPa   [confidence: dense]` — within default (152, 220) ✓
- Each parameter's "Evidence:" stanza cites both `dense confidence backed by 2357 samples` AND `observed in training [low, high]` — Confidence carries through to the user-facing text.
- Line 59 `discrete-click value rounded from 1.395 to 1 (legal range 1..5)` — the discrete-click rounding step (post-clamp #3) is observable.

**Per-car v3 fitter env contract.** `src/racingoptimizer/physics/fitter.py:441-446` rotates `env_block` columns into `_ENV_COLUMNS` order before concatenating with the setup-parameter block, so the trained X matrix has a stable column ordering. `model.py:443-471` (`_assemble_feature_row`) reverses this at predict time, indexing each fitter's `feature_names` against `_ENV_COLUMNS`. The two ends are kept in sync by a single import (`from racingoptimizer.physics.fitter import _ENV_COLUMNS` at `model.py:456`) — there is no risk of order drift.

## Gaps vs. VISION

1. **MINOR (cosmetic, user-visible) — Conditions header label is wrong.** `src/racingoptimizer/explain/render_text.py:144`:
   ```python
   f"Conditions: AirTemp {env.track_temp_c:.1f} C  "
   ```
   The value comes from `env.track_temp_c` (track surface temperature) but is labeled "AirTemp". The user reading the briefing reads the wrong number for the wrong concept. Fix: relabel to `TrackTemp`, OR also surface `env.air_temp_c` alongside, OR render both `AirTemp {env.air_temp_c}` and `TrackTemp {env.track_temp_c}`. VISION §10 explicitly lists `AirTemp` and `TrackTempCrew` as separate channels with different physical effects ("higher air density → more downforce", "higher track temp → more tire grip initially but faster degradation"). One-line fix.

2. **MINOR (correctness, latent) — Model cache key does NOT fold constraints fingerprint.** `src/racingoptimizer/cli/recommend.py:632-654` (`_model_cache_path`):
   ```python
   parts = ["|".join(sorted(session_ids))]
   parts.append(f"onto={onto_fingerprint}")
   parts.append(f"schema={int(ENV_FEATURE_SCHEMA_VERSION)}")
   ```
   Editing `constraints.md` (e.g. tightening BMW wing range) does NOT invalidate the cached pickle. The cached `PhysicsModel.constraints` is the OLD table; the DE search runs against OLD bounds; the CLI's `_post_clamp` re-clamps with the NEW table at render time. Net effect: the user never sees an out-of-bound value (saved by the third clamp pass), but the search itself explored stale bounds — the recommendation may be suboptimal vs a fresh fit. The user explicitly asked for this check; it's a documented gap.
   - Fix: hash a `constraints.md`-content fingerprint into the cache parts:
     ```python
     parts.append(f"constraints={hashlib.sha256(constraints_path.read_bytes()).hexdigest()[:8]}")
     ```
   - Alternative cheaper fix: hash the resolved `(parameter, car) -> bound` view that the constraint table actually exposes, so unrelated edits to `constraints.md` (annotations, comments, blank lines) don't invalidate caches.

3. **MEDIUM (latent feature gap) — `wet_mode` and `wind` are dead code in the live recommend path.** `src/racingoptimizer/physics/wet_mode.py` exports `classify_conditions / wet_baselines / wet_phase_weights`, all unit-tested at `tests/physics/test_wet_mode.py`, none called from `score.py` / `recommend.py` / any CLI entry point. Same for `physics/wind.py` (`aero_wind_modifier`, `decompose_wind`). VISION §10 explicitly says: *"The physics model should understand that the SAME setup behaves differently in different conditions ... wet conditions fundamentally change everything ... wind affects aero balance asymmetrically."* The infrastructure exists (`env.track_wetness`, `env.wind_vel_ms`, `env.wind_dir_deg` survive end-to-end) but the regime-dependent baselines / phase weights / aero-balance shifts are unwired. The recommender always uses dry baselines + dry phase weights regardless of `env`.
   - Fix: in `physics/score.py::score_setup` (or `recommend.py::recommend`), branch on `classify_conditions(env)` and route the baselines / phase weights through `wet_baselines(car, regime)` / `wet_phase_weights(regime)`.
   - Slice E spec compliance is technically satisfied because there is a wet-mode story merged; live behaviour is not.

4. **MINOR (UX) — JSON renderer drops 7 of the 12 env channels.** `src/racingoptimizer/explain/render_json.py:105-112` (`_env_to_json`) emits `{air_density_kg_m3, track_temp_c, wind_vel_ms, wind_dir_deg, wetness}`. Missing: `air_temp_c, air_pressure_mbar, relative_humidity, fog_level, weather_declared_wet, precip_type, skies`. A JSON-consuming caller cannot reconstruct the `EnvironmentFrame` the recommendation was tuned for. Internal code uses every channel; only the JSON surface is lossy. Trivial fix: emit all 12.

5. **MINOR (drift hazard) — CLI re-implements the env-channel name map.** `src/racingoptimizer/cli/recommend.py:689-704` declares its own `_ENV_FLOAT_CHANNELS` / `_ENV_DISCRETE_CHANNELS` dicts that mirror `context/environment.py:28-45` (`_FLOAT_CHANNELS` / `_BOOL_CHANNELS` / `_INT_CHANNELS`). They agree today (CLI explicitly excludes `WindDir` because it needs circular-median aggregation). Refactor: expose the IBT-name → field-name mapping from `racingoptimizer.context.environment` as a public module-level `IBT_CHANNEL_TO_FIELD` constant, and have the CLI subset+filter from it. Avoids future drift.

6. **NONE — Confidence is propagated everywhere it must be.** Every recommended parameter is `(float, Confidence)`. Every score sub-utilization returns `(float, Confidence)`. Every prediction channel is a `Confidence`. The justification renderer requires `Confidence` (raises `IncompleteJustificationError` otherwise). No bare-float regression possible without breaking `SetupJustification.__post_init__`. Spec §14 (worst-regime roll-up) is realised at `render_text.py:154-166`.

7. **NONE — Constraint clamp runs before the briefing renders.** Triple-clamp pattern (objective + post-DE defensive + render-time fresh-table) means the user is structurally protected from out-of-bounds values. The BMW Spa card evidence above confirms it (every bounded value is inside its constraint). Discrete-click rounding is also pre-render.

## Diff vs. 2026-05-01 baseline

`docs/VISION_COMPLIANCE.md` cross-cutting concerns:
- §3 cites `Confidence.derive` as "the only place fit quality talks about residuals, and it does so empirically (CV-folds), not from closed-form physics" — still true; no shadow regime-derivation has appeared.
- §5 cites `_clamped_or_raise (physics/score.py:402-425)` and `_post_clamp (cli/recommend.py:131)` as the dual-clamp; still true. The triple-clamp (search-objective + post-DE defensive + render-time) is unchanged.
- §6 cites `Confidence.derive` regime classification as the data-density signal — still true.
- §10 cites `EnvironmentFrame` carries all 12 channels — still true; `_FLOAT_CHANNELS` / `_BOOL_CHANNELS` / `_INT_CHANNELS` are the single-source-of-truth tuples.

New since 2026-05-01: the `TrackWetness` enum-to-fraction normalisation in `parser.py:63-88` lands at the IBT boundary, so every consumer sees the canonical 0..1 fraction. This makes `wet_mode.classify_conditions` thresholds (0.05 / 0.3 / 0.7) actually meaningful — once it is wired (Gap 3).

The cache-key constraints fingerprint gap (#2) and the wet/wind dead-code gap (#3) were not flagged in the 2026-05-01 baseline; both surface only when you trace the cross-cutting modules end-to-end across slice boundaries (which is the point of this audit).

## Recommended next actions

- **Fix the conditions-header label** in `explain/render_text.py:144` — one-line change. Either rename to `TrackTemp` (current value), add a separate `AirTemp` field reading `env.air_temp_c`, or render both. Tests live at `tests/explain/test_renderers.py`.
- **Fold a constraints fingerprint into the model cache key** in `cli/recommend.py:_model_cache_path`. Hash either the file content or the resolved `(parameter, car) -> bound` table. Update the docstring.
- **Wire `wet_mode` into the live recommend path.** In `physics/score.py::score_setup` (or `physics/recommend.py::recommend`), branch on `classify_conditions(env)` and route baselines/phase-weights through `wet_baselines / wet_phase_weights`. Add a regression test in `tests/physics/test_recommend.py` that asserts a wet-env recommendation differs from the dry baseline.
- **Surface all 12 env channels in `render_json._env_to_json`** so the JSON consumer has the full conditions context.
- **Centralise the IBT-channel ↔ field-name map** in `racingoptimizer.context.environment` as a public constant, and have `cli/recommend.py` consume it instead of redeclaring.
