# Audit -- Slice F1: Explain / Render (2026-05-06)

## Summary
- Grade: PARTIAL
- Renderers wire end-to-end and produce useful output for all five GTP cars, but `render_narrative` (the DEFAULT renderer since 2026-05-06) has zero unit-test coverage, multiple `_CAR_FEEL` / `_DIRECTION_VERB` keys are dead because they reference families that no `ParameterSpec` declares, the telemetry-backed `Why:` line never fires from the CLI (caller drops `schedule=`), and rendered text leaks `--` / `|` Unicode in violation of the documented Windows-cp1252 ASCII rule.

## Implementation quality

### `src/racingoptimizer/explain/__init__.py`
Clean re-export hub. No issues.

### `src/racingoptimizer/explain/comparison.py`, `status.py`
Two pure-dataclass files (32 lines each). Fine.

### `src/racingoptimizer/explain/render_text.py`
- `_REGIME_RANK` (line 17) is duplicated verbatim in `render_json.py:21` and `narrative.py:652` (three copies). Should live in `racingoptimizer.confidence`.
- `_HUMAN_LABEL` (lines 21-28) covers only 6 parameters. Per-car dampers / torsion bars / perches / pushrods / ARBs all fall through to the title-cased de-snaked fallback (`_humanize_slug`). For the engineering `--detailed` view this is acceptable; for parity with `narrative._PARAM_LABEL` (29 entries) it's a gap.
- `_render_block` line 199 emits `[confidence: ...]`. This is the tag the test gate `tests/cli/test_per_car_smoke.py:61` looks for as proof of a parameter block -- but the default code path is now `render_narrative` and emits no such tag. See "Wiring" + Gaps §4.
- `_humanize_slug` (line 229) duplicates similar helpers in `narrative._humanize` (line 1290) and `full_setup_card._humanize_leaf` (line 145).
- ASCII safety violated at line 73: `f"(skipped -- bounds...)"` U+2014 leak (only fires when `untrained_parameters` is non-empty).

### `src/racingoptimizer/explain/render_json.py`
- `_REGIME_RANK` duplication (above).
- `_env_to_json` (line 105) hand-codes the 12 `EnvironmentFrame` fields. Drift risk; recommend a regression assertion `set(_env_to_json(env).keys()) == {EnvironmentFrame field names}`.
- Otherwise direct dataclass -> dict serialisation.

### `src/racingoptimizer/explain/justification.py`
- `_CLICK_STEP` (lines 77-84) is a separate hardcoded step table parallel to `ParameterSpec.step`. Two-source-of-truth. The renderer (`narrative._snap`, `full_setup_card._round_to_step`) reads the ontology, this file reads the constant. Drift surface.
- `_safe_score_breakdown` and `_safe_score_total` (lines 241-253, 282-293) catch ALL exceptions. The synthetic-neutral-baseline note at lines 132-144 then masks the silence with a benign "model held value at training baseline" message -- hides legitimate fitter bugs. Bare `except Exception` smell.
- Synthetic neutral impact carries `corner_id=0, phase=Phase.STRAIGHT` even when no T0 corner exists; renderer prints `T0` literally. Cosmetic.

### `src/racingoptimizer/explain/full_setup_card.py`
- ASCII safety violated repeatedly. Legend line emits middle-dots and the header / no-value placeholder emit em-dashes (lines 167, 326, 333, 337, 345, 351). Artefact `bmw-spa-cal-0506-1015.txt:72` shows the corruption: `FULL SETUP CARD ? bmw @ spa_2024_up`.
- `_PANELS` (lines 79-85) -- five-panel hardcoded list, fine.
- `_MIRRORED_LEAVES` (lines 103-128) -- five entries, all under `("Chassis", "Right...", ...)`. Acura/Porsche cars without these YAML paths silently fall through to `[past]`.
- `_PREDICTED_READOUT_PATHS` (lines 137-142) -- only static ride heights. Documented in CLAUDE.md.
- `_format_value` (line 164): integer detection `value == int(value)` raises ValueError on NaN. Probably never hit but fragile.
- `_format_opt_value` (line 205) has three rendering modes (categorical / discrete-list / uniform-step). `test_full_setup_card.py` only exercises one numeric case directly.
- `_render_panel` (line 370) -- 100-line conditional cascade. The rounding/comparison logic is repeated across the `opt_match` and `mirror_source` branches; a `_resolve_displayed(...)` helper would eliminate duplication.
- `_walk_block` (line 495) depends on `pyyaml` preserving dict insertion order. Worth a docstring assertion.

### `src/racingoptimizer/explain/narrative.py`
1300 lines, by far the largest file in the slice.

- **Family-name mismatch with the ontology -- major dead-code issue:**
  - `_DIRECTION_VERB["heave_spring"]` (line 64) is dead. No user-settable parameter has `family="heave_spring"`; `heave_spring_rate_n_per_mm` has `family="spring_rate"` (`physics/ontology.py:238`). Same for `_DIRECTION_VERB["heave_slider"]` (line 65).
  - All four `_CAR_FEEL[("heave_spring", "front", +/-)]` entries (lines 115-126) are unreachable for the same reason. Verified in artefact `bmw-spa-reset-0506-1029.txt:48` -- `Front heave spring: 60 -> 70 N/mm  (stiffens)` followed by phase-themed `Helps:`/`Watch:` (the `_render_change` fallback path), not the rich Effect/Trade table. Same in `ferrari-spa-race-0506-1026.txt:46`.
  - `_FAMILY_TO_THEME["heave_spring"]` and `["heave_slider"]` (lines 577-578) likewise dead.
  - `_CAR_FEEL[("spring_rate", "front", +/-)]` is missing -- this is what the heave-spring parameter actually looks up after family resolves to `"spring_rate"`. Lookup misses every level (1-4), falls through to phase-themed.
  - `_CAR_FEEL[("torsion_bar", "rear", +/-)]` is also missing. Ferrari has rear torsion bars; `ferrari-spa-race-0506-1026.txt:37-39` shows the fallthrough.

- **Hardcoded car-specific landmarks in generic vocabulary:** `_CAR_FEEL` strings name "Eau Rouge / Pouhon" / "Kemmel compression" / "T9, T13" / "T8, T15" / "T1, T11, T18" -- all Spa-specific (and a few Sebring). These print verbatim for every car at every track. Reading "Bottoming risk on Kemmel" inside a Sebring or Daytona briefing is misleading. Should be generic phrasing or driven by the actual recommendation's `_dominant_impact_corner`.

- **Magic-number defaults:** `step = 0.5` fallback in `_format_value_delta` (line 1044) and `_moved` (line 1277) when `spec.step` is missing. Should derive from constraints span or be `0.0` (no rounding).

- **`_telemetry_why` (line 918)** -- Builds a counterfactual and queries `model.predict(..., corner_archetype=archetype)`. The v4 archetype branch (`feature_schema_version >= 4`) requires the `schedule` kwarg, but `cli/recommend.py:381` doesn't pass it. Result: every v4 prediction inside `_telemetry_why` raises and is silently swallowed by `narrative.py:977 except Exception: return ""`. **The documented "Telemetry-backed Why line" feature is dead in production.** Confirmed by absence of `Why:` lines in any artefact.

- **`_EVIDENCE_CHANNELS` (lines 823-915):** ASCII-safe; covers 19 family/axis combos. Missing `corner_weight` and `front_wing` (probably inert today).

- **ASCII rule leak inside `_CAR_FEEL`:** U+2014 em-dashes at lines 119, 266, 280, 308, 313, 324. Visible in `ferrari-spa-race-0506-1026.txt:72` ("Slower front rebound after pitch -- keeps weight...").

- **`_resolve_past_values` (line 1207)** does a private import of `_scalar_from_yaml` from `full_setup_card`. Reaching into another module's underscore-prefixed helper. Should be promoted to a shared `explain.utils`.

- **`_PARAM_LABEL` (lines 597-628):** 29 entries. No entries for damper-corner parameters (`damper_lsc_fl` etc.) -- they fall through to "Damper Lsc Fl" via `_humanize`. Acceptable given the per-corner explosion.

- **`_notes_block` (lines 1167-1204):** the "params held at past" loop at lines 1184-1191 has no body that appends anything (`continue`-only). The comment says "covered above" -- appears to be dead code.

## Wiring

### Call graph
- **CLI text path** (`cli/recommend.py:380-388`): `optimize <car> <track>` -> `build_justifications(rec, model, pinned, clamp_warnings, schedule)` -> `render_narrative(rec, model, justifications, most_recent_setup, track_display, quali, pinned, warnings)` -> concatenated with `render_full_setup_card(rec, car, most_recent_setup, predicted_readouts)` -> `click.echo`.
  - **BUG**: `render_narrative` is called WITHOUT `schedule=schedule`, even though it accepts the kwarg and threads it down to `_telemetry_why`. The `build_justifications` call four lines up (line 322) DOES pass `schedule=schedule` correctly. Pure caller-side oversight. See Gaps §2.
- **CLI detailed path** (`cli/recommend.py:370-378`, only when `--detailed`): `render_recommendation_text` instead of `render_narrative`. Used by the `setup-justifier` validator agent for engineering drill-down.
- **CLI JSON path** (`cli/recommend.py:325`): `render_recommendation_json(rec, model, justifications, pinned, warnings, track_display)`. Auto-saves to `recommendations/*.json` and prints `[saved ...]` to stderr -- known to break `tests/cli/test_per_car_smoke.py::test_recommend_per_car_json` (Click's `CliRunner` mixes stderr into `.output` so `json.loads` fails).
- **Calibrate path** (`cli/calibrate.py:602`): calls `render_full_setup_card(rec, car, most_recent_setup, predicted_readouts={})` -- explicitly empty dict so all calculated leaves render `[readout]` from past YAML.
- **Test suites**:
  - `tests/explain/test_renderers.py` exercises `render_status_*` and `render_comparison_*` only.
  - `tests/explain/test_full_setup_card.py` covers the setup card across the four tag paths and parametrises over 5 cars.
  - `tests/explain/test_justification.py` covers `SetupJustification` validation.
  - **No tests for `render_narrative`, `render_recommendation_text`, `render_recommendation_json`, or `build_justifications`'s actual output.**

### `build_justifications` flow
Walks `rec.parameters`. Per parameter: `_split_impacts` runs counterfactual `score_breakdown` at the parameter's training baseline (or one-step-shifted if rec equals baseline) -> splits into helps/hurts; `_sensitivity` runs +/-1-click `score_setup` deltas; `_evidence` builds string list; constructs frozen `SetupJustification`. Sorts by impact magnitude desc. Schedule kwarg is correctly threaded.

### `render_full_setup_card` predicted-readout consumption
Receives `predicted_readouts: dict[str, float]` (model channel name -> value at OPT setup). `_PREDICTED_READOUT_PATHS` maps 4 YAML paths (LF/RF/LR/RR `RideHeight`) to channels (`setup_static_*_ride_height_mm`). When prediction is present -> `[predicted]`; otherwise `[readout]` echoing past YAML. All other calculated leaves (deflections, corner weights, AeroCalculator block, hot pressures, gear speeds) always `[readout]`.

### `--detailed` branch
Single boolean flag at `cli/recommend.py:103`. `if detailed:` -> `render_recommendation_text`; else -> `render_narrative`. Both paths receive identical `justifications` / `warnings`. The setup card is appended to BOTH paths; only the briefing block is swapped.

## Gaps

1. **CRITICAL** -- `narrative.py:115-126` + `64-65` + `577-578` -- `_CAR_FEEL` / `_DIRECTION_VERB` / `_FAMILY_TO_THEME` keys for `"heave_spring"` and `"heave_slider"` are UNREACHABLE because no user-settable `ParameterSpec` declares these families (only user-insettable readouts at `physics/ontology.py:224, 228`). Front heave-spring rendering falls through to phase-themed `Helps:`/`Watch:` instead of Effect/Trade. Verified in `bmw-spa-reset-0506-1029.txt:48` and `ferrari-spa-race-0506-1026.txt:46`. Fix: add `("spring_rate", "front", +/-)` keys, or have `_car_feel` rewrite the family for `heave_*` parameter names.

2. **CRITICAL** -- `cli/recommend.py:381` -- `render_narrative(...)` called without `schedule=schedule`. `_telemetry_why` requires it (and the v4 archetype dict it carries) to call `model.predict(..., corner_archetype=archetype)` for v4 cars (BMW, Cadillac, Ferrari). Without the kwarg, every v4 prediction inside `_telemetry_why` raises and is silently swallowed. Net effect: zero `Why:` lines in any v4-car briefing -- the documented "Telemetry-backed Why line" feature is dead in production. Fix: pass `schedule=schedule` (already in scope from line 322's `build_justifications` call).

3. **MAJOR** -- Zero coverage of `render_narrative` (the DEFAULT renderer since 2026-05-06). No `tests/explain/test_narrative.py`. CLAUDE.md "Verification convention" requires a `tests/<slice>/test_per_car_smoke.py` looping the five canonical car fixtures -- `tests/explain/test_per_car_smoke.py` does not exist. The only end-to-end coverage is `tests/cli/test_per_car_smoke.py`, which (4) shows is a suspect gate.

4. **MAJOR** -- `tests/cli/test_per_car_smoke.py:61` asserts `any("[confidence:" in line for line in out.splitlines())` as the parameter-block check. `[confidence:` is emitted ONLY by `render_text._render_block` (i.e. `--detailed`). The default narrative emits `Confidence: <regime> (median n=N)` once at the top with no per-parameter `[confidence:` tag. Either the smoke test is silently failing (CLAUDE.md is stale) or it passes via unrelated stdout content. Fix: assert on `"OVERALL DIRECTION"` or `"CHANGES ("`.

5. **MAJOR** -- `full_setup_card.py:345, 348-352, 167, 326, 333, 337` and `narrative.py:119, 266, 280, 308, 313, 324` -- Unicode em-dashes (`U+2014`), middle-dots (`U+00B7`), and special chars in legend / car-feel strings. CLAUDE.md states "Output is ASCII-only (Windows cp1252 console can't encode Unicode arrows / em-dashes)". `bmw-spa-cal-0506-1015.txt:72,74` shows `?` corruption in the FULL SETUP CARD header. Replace with `--`, `|`, etc.

6. **MAJOR** -- `narrative.py:115-169` -- `_CAR_FEEL` strings hardcode Spa landmarks ("Eau Rouge / Pouhon", "Kemmel compression", "T9, T13"). Emitted verbatim for every car at every track. Should be generic or driven by `_dominant_impact_corner`.

7. **MAJOR** -- `_CAR_FEEL` missing per-axle entries:
   - `("torsion_bar", "rear", +/-)` -- Ferrari has them; falls through (`ferrari-spa-race-0506-1026.txt:37-39`).
   - `("spring_rate", "front", +/-)` -- every front-spring parameter misses (heave + non-heave).
   - `("ride_height", *, +/-)` -- only `perch_offset` and `pushrod` covered.
   - `("corner_weight", *, +/-)` -- entirely missing (also no narrative-table entries).

8. **MAJOR** -- `justification.py:251, 292` -- bare `except Exception: return {}`/`return 0.0` hides legitimate scoring failures as "neutral baseline" via the synthetic impact at lines 132-144. Narrow the catch.

9. **MAJOR** -- `justification.py:77-84` -- `_CLICK_STEP` constant table parallel to `ParameterSpec.step`. Drift risk. Read ontology first, fall back to constant.

10. **MAJOR** -- `cli/recommend.py:325` -- JSON path emits `[saved ...]` to stderr after JSON to stdout. Documented regression breaking `test_recommend_per_car_json`. Either suppress under `--json` or set `mix_stderr=False`.

11. **MINOR** -- `_REGIME_RANK` defined in three files (`render_text.py:17`, `render_json.py:21`, `narrative.py:652`). Move to `racingoptimizer.confidence`.

12. **MINOR** -- Three hand-rolled humanizers (`render_text._humanize_slug`, `narrative._humanize`, `full_setup_card._humanize_leaf`). Consolidate.

13. **MINOR** -- `narrative._resolve_past_values` (line 1218) cross-imports `_scalar_from_yaml` from `full_setup_card`. Promote to `explain.utils`.

14. **MINOR** -- `render_json._env_to_json` (line 105) drift risk. Add a regression assertion against `EnvironmentFrame` fields.

15. **MINOR** -- `full_setup_card._format_value` (line 164) raises on NaN floats via `value == int(value)`. Wrap.

16. **MINOR** -- `render_text._HUMAN_LABEL` covers only 6 parameters; should share `narrative._PARAM_LABEL`.

17. **MINOR** -- `narrative._render_change:1044` magic-number `step = 0.5` default.

18. **MINOR** -- `narrative._notes_block:1184-1191` -- "held at past" loop appears dead code (`continue`-only body, comment says "covered above").

19. **MINOR** -- `comparison.py` and `status.py` are 32 lines of dataclasses each. Could fold into one `explain/dataclasses.py`.

20. **NONE** -- `tests/explain/test_full_setup_card.py::test_render_works_for_every_canonical_car` covers all five GTP cars (no-crash smoke). Justification + comparison + status are car-agnostic.

## Evidence

- **Test suite**: NOT RUN -- `Bash` and `PowerShell` tools (and `dangerouslyDisableSandbox=true`) all denied by sandbox. Static count: `tests/explain/` has 4 files (`__init__.py`, `test_renderers.py`, `test_full_setup_card.py`, `test_justification.py`), ~17 test cases. No `test_narrative.py` and no `test_per_car_smoke.py`.
- **Lint**: NOT RUN -- same denial for `uv run ruff check src/racingoptimizer/explain`. Manual review found no obvious E/F/I issues; the most likely cleanup target is the dead `_DIRECTION_VERB["heave_spring"]` / `_CAR_FEEL[("heave_spring", ...)]` data entries (ruff can't flag dead dict keys).
- **Latest artefacts**:
  - `recommendations/bmw-spa-reset-0506-1029.txt` -- full BMW@Spa narrative; OVERALL DIRECTION + 36 changes + setup card. Confirms heave_spring fallthrough at line 48 and zero `Why:` lines anywhere in 200+ lines.
  - `recommendations/bmw-spa-cal-0506-1015.txt` -- calibration probe + setup card; line 72 shows `FULL SETUP CARD ? bmw @ spa_2024_up` (em-dash -> `?` corruption on cp1252).
  - `recommendations/ferrari-spa-race-0506-1026.txt` -- Ferrari narrative; line 37 confirms rear-torsion-bar fallthrough, line 46 confirms heave-spring fallthrough, line 72 shows leaked U+2014 em-dash inside a Trade line.

## Recommended next actions

1. (CRITICAL) Add `("spring_rate", "front", +/-)` keys to `narrative._CAR_FEEL`, remove the dead `("heave_spring", ...)` entries -- or have `_car_feel` rewrite the family when name contains "heave". Add a unit test asserting every fittable user-settable parameter in the BMW + Ferrari ontology resolves to a `_CAR_FEEL` hit.
2. (CRITICAL) Pass `schedule=schedule` from `cli/recommend.py:381` into `render_narrative`. Add an integration test asserting at least one `Why:` line in a per-car BMW briefing.
3. (MAJOR) Create `tests/explain/test_narrative.py` and `tests/explain/test_per_car_smoke.py` with: (a) per-car parametrised smoke; (b) fallthrough-detection test failing on any moved parameter rendering without an Effect/Trade pair; (c) ASCII-only assertion `assert briefing.encode("ascii", "strict")`.
4. (MAJOR) Sweep `narrative.py` and `full_setup_card.py` for em-dashes and middle-dots and replace with ASCII (`--`, `|`).
5. (MAJOR) Fix the smoke-test gate marker in `tests/cli/test_per_car_smoke.py:61` -- assert on `"OVERALL DIRECTION"` or `"CHANGES ("` (or run under `--detailed`).
6. (MAJOR) Genericise hardcoded Spa landmarks inside `_CAR_FEEL`.
7. (MINOR) Promote `_REGIME_RANK` to `racingoptimizer.confidence`; consolidate the three humanizers; promote `_scalar_from_yaml` to `explain.utils`.
8. (MINOR) Replace `justification._CLICK_STEP` with an ontology lookup; narrow the bare `except Exception` catches.
