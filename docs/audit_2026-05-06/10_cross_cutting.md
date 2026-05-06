# Audit -- Cross-cutting (2026-05-06)

## Summary
- Grade: **PARTIAL**
- The three cross-cutting modules are small, well-typed, and broadly consumed across slices B/E/F, but `constraints.md` still ships **5 unbounded parameter families** the recommender silently drops, `Confidence.derive` regime thresholds are hardcoded magic numbers, and `EnvironmentFrame.from_partial_row` has zero non-test consumers in `src/`.

## Implementation quality

### `racingoptimizer.context.environment`
- `src/racingoptimizer/context/environment.py:48-100` -- Frozen, slots dataclass with NaN/`False`/`-1` sentinels. Field order is **load-bearing**:
  - `physics/model.py:567-597` (`_env_to_array`) reads fields by attribute into a positional 12-element array consumed by `physics/fitter.py:162-178` (`_ENV_COLUMNS`). Reordering or renaming a field without bumping `ENV_FEATURE_SCHEMA_VERSION_PER_CAR` silently scrambles training inputs at predict time. No comment in `environment.py` warns of this.
  - `physics/model.py:546-564` (`_env_to_array_v1`) freezes pre-S2.2 5-field order for legacy pickle compatibility.
- `src/racingoptimizer/context/environment.py:28-45` -- IBT-channel <-> field-name mapping is **duplicated** at three other sites: `cli/recommend.py:1249-1264`, `physics/fitter.py:162-178`, `explain/render_json.py:112-128` (latter renames `air_density` -> `air_density_kg_m3`, `track_wetness` -> `wetness`).
- `from_row` (strict) and `from_partial_row` (lenient) have **no `src/` consumers** -- see Wiring.

### `racingoptimizer.confidence.confidence`
- `src/racingoptimizer/confidence/confidence.py:23-28` -- `Regime = Literal[...]` plus parallel `_VALID_REGIMES = frozenset({...})` duplicate the same string set.
- `src/racingoptimizer/confidence/confidence.py:65-75` -- Regime thresholds are hardcoded magic numbers (`30`, `0.5`, `0.2`) with no per-car override path. Should be module-level named constants.
- `src/racingoptimizer/confidence/confidence.py:69` -- `signal_std == 0` defensively masked via `max(signal_std, 1e-12)`; tested but silent.
- `src/racingoptimizer/confidence/confidence.py:31-50` -- Frozen+slots dataclass field order matters for any pickle that embeds a `Confidence` (per-car model caches). The `FITTERS_LAYOUT_VERSION` cache key does NOT cover cross-cutting types -- adding a `Confidence` field would break old pickle revival silently.

### `racingoptimizer.constraints.loader`
- `src/racingoptimizer/constraints/loader.py:23-33` -- `_by_car` reached into via `noqa: SLF001` from `cli/recommend.py:875`. Frozen dataclass needs an explicit `with_pins(...)` mutator.
- `src/racingoptimizer/constraints/loader.py:85-201` -- `_section_to_param_base` is a 100-line if/elif cascade. Refactor to a `_HEADING_MAP` dict plus a small set of regex/prefix matchers.
- `src/racingoptimizer/constraints/loader.py:130-134` -- Em-dash split + `rsplit` fallback duplicated 5+ times for `damper`, `brake duct`, `corner weight`, `front diff`, `arb size`.
- `src/racingoptimizer/constraints/loader.py:283-285` -- `_CORNER_SUFFIXES` order matters (FL before front so longer match wins); enforced by comment, not code.
- `src/racingoptimizer/constraints/loader.py:140-146` -- `damper` body parsing has a defensive double-strip for per-car overrides that omit the em-dash; grammar in `constraints.md:330-332` doesn't say which is canonical.

### `racingoptimizer.constraints.clamp`
- `src/racingoptimizer/constraints/clamp.py:23-30` -- `_to_canonical` swallows `UnknownCarError`, falling back to raw input which then matches no per-car override and silently uses defaults. Inconsistent with `loader.py:CANONICAL_CARS` being authoritative.
- `src/racingoptimizer/constraints/clamp.py:48` -- `clamp(table=None)` re-parses `constraints.md` on every call. `explain/justification.py:309` invokes this per parameter line.
- `ClampStatus` is a `Literal[...]` but the four statuses are not exposed as constants -- comparisons are stringly-typed.

### `constraints.md` -- `<TODO: from iRacing UI>` rows still in tree
| Section | Lines | Status |
|---|---|---|
| `Corner weight (target)` | 211-217 | All 4 corners TODO |
| `Differential` (coast/power %) | 230-234 | preload bounded; coast/power TODO |
| `Brake duct opening -- front` | 312-315 | Both bounds TODO |
| `Brake duct opening -- rear` | 317-320 | Both bounds TODO |
| `Throttle / brake mapping` | 322-326 | Parameter name + bounds TODO |

These match CLAUDE.md's "still TODO" list. Land in `untrained_parameters` per `physics/fitter.py:248`.

## Wiring

### `EnvironmentFrame` (15 sites)
**Constructed:** `cli/recommend.py:1228` (`_env_from_overrides`, only production constructor); `cli/calibrate.py:595` (all-sentinel placeholder).
**Threaded:** `physics/recommendation.py:15` (field of `SetupRecommendation`); `physics/model.py:202-461` (predict/recommend); `physics/score.py:85-678` (every score component); `physics/recommend.py:48,705`; `physics/wet_mode.py:49` (`classify_conditions`); `explain/render_json.py:106-128`; `corner/states.py:132+` (comment); `ingest/parser.py:54` (comment).
**Adapter dead code:** `from_row` / `from_partial_row` -- exercised only by `tests/context/test_environment.py` and `tests/physics/test_predict.py:191-199`. Production builds use `_env_from_overrides` because it has to merge CLI overrides on top of corpus medians.

### `Confidence` (8 sites)
**`Confidence.derive`:** `physics/model.py:366,439,524`; `physics/recommend.py:403,679`; `physics/io_log.py:77`.
**Direct `Confidence(...)`:** `cli/calibrate.py:568-573` (placeholders); `physics/recommend.py:674-677` (no-fitter fallback).

### `ConstraintsTable` / `clamp` (13 sites)
**Loaded:** `cli/recommend.py:243`, `cli/calibrate.py:235`, `physics/fitter.py:244,671` (baked into pickle; cache invalidated on `constraints.md` content hash per `cli/recommend.py:1207`).
**Threaded:** `physics/model.py:109` (field of `PhysicsModel`); `physics/garage_inventory.py:51`; `physics/ontology.py:804` (`fittable_parameters`).
**`clamp(...)`:** `physics/recommend.py:272`; `physics/score.py:609`; `explain/justification.py:309`; `cli/recommend.py:1435` (`_post_clamp`).
**Mutator (private surface leak):** `cli/recommend.py:864-891` (`_apply_pins_to_constraints` reaches into `table._by_car` with `# noqa: SLF001`).

## Gaps
1. **MAJOR -- 5 unbounded `constraints.md` families** (`constraints.md:211-326`). Capture from iRacing UI is the unblocker.
2. **MAJOR -- `EnvironmentFrame.from_partial_row` is dead production code** (`context/environment.py:82-100`). CLI uses `_env_from_overrides` instead.
3. **MAJOR -- Implicit field-order coupling** between `EnvironmentFrame` and `_env_to_array`/`_ENV_COLUMNS` (`context/environment.py:48-65` <-> `physics/model.py:567-597` <-> `physics/fitter.py:162-178`). No warning comment, no round-trip regression test.
4. **MAJOR -- Three duplicate IBT-channel <-> field mappings** (`context/environment.py:28-45`, `cli/recommend.py:1249-1264`, `physics/fitter.py:162-178`, `explain/render_json.py:112-128`).
5. **MAJOR -- `Confidence` regime thresholds are hardcoded magic numbers** (`confidence/confidence.py:66-75` -- `30`, `0.5`, `0.2`).
6. **MAJOR -- `Regime` literal duplicated** (`confidence/confidence.py:23-25`).
7. **MINOR -- `ConstraintsTable._by_car` leak** (`cli/recommend.py:875`).
8. **MINOR -- `_section_to_param_base` 100-line cascade** (`loader.py:85-201`).
9. **MINOR -- `clamp(table=None)` re-parses on every call** (`constraints/clamp.py:48`, hot in `explain/justification.py:309`).
10. **MINOR -- `_to_canonical` swallows `UnknownCarError`** (`constraints/clamp.py:23-30`).
11. **MINOR -- `EnvironmentFrame()` placeholder NaN flows** through `_env_to_array` and gets coerced to 0.0 (`cli/calibrate.py:595` -> `physics/model.py:596`).
12. **MINOR -- `from_row` raises bare `KeyError`** (`context/environment.py:73-79`).
13. **NONE -- `Confidence` 95% bracket math.** Spec-correct, test-covered.

## Evidence
- **Test suite:** sandbox denied `uv run pytest -q tests/context tests/confidence tests/constraints -m "not slow"`. Static review: 11 context tests, 11 confidence tests, 18 constraints tests; coverage thorough.
- **Lint:** sandbox denied `uv run ruff check`. Static review shows no obvious violations; only `# noqa: SLF001` is at `cli/recommend.py:875`, outside cross-cutting code.
- **Latest artefact** (`recommendations/bmw__sebring_international__20260506-075403.txt`):
  - Header `Conditions: 24 C ambient / 35 C track | Confidence: noisy (median n=51)` -- `EnvironmentFrame` round-trips into briefing; `Confidence.regime` aggregates correctly.
  - Line 59 `Untrained (constraints.md TODO): corner_weight_fl_kg, ...` -- TODO rows surface as expected.
  - Line 60 lists 47 `[OPT pin]` parameters.
  - Tyre pressure 152.0 kPa = `bounds("default", "tyre_cold_pressure_kpa")[0]` -- `_post_clamp` working.

## Recommended next actions
- Fill the 5 outstanding `<TODO: from iRacing UI>` rows in `constraints.md`.
- Delete or wire up `EnvironmentFrame.from_partial_row`.
- Promote regime thresholds (30, 0.5, 0.2) to named module constants.
- Unify the IBT-channel <-> field-name mapping into one source of truth.
- Add a "field order is part of the v4 fitter contract" comment + round-trip regression test to `EnvironmentFrame`.
- Expose `ConstraintsTable.with_pin(car, parameter, value)` to remove the `_by_car` leak in `cli/recommend.py:875`.
- `functools.lru_cache(maxsize=1)` on `load_constraints()` keyed off mtime.
- Refactor `_section_to_param_base` to a dict + small regex set; collapse 5 duplicate em-dash splits.
