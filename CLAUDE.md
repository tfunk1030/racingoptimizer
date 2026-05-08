# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

`racingoptimizer` is the `optimize` Python CLI for iRacing GTP setup recommendations (VISION.md §8). All six VISION slices (A–F) plus three cross-cutting modules are merged. End-user walkthrough is in `GETTING_STARTED.md`; design spec is `VISION.md` (read it first); per-clause audit is `docs/VISION_COMPLIANCE.md`.

## Commands

```bash
uv venv && uv pip install -e ".[dev]"

uv run optimize learn ./ibtfiles            # ingest a directory of .ibt files
uv run optimize learn ./ibtfiles --reparse  # force re-parse already-ok sessions
uv run optimize bmw sebring                 # recommend by (car, track) — race default
uv run optimize bmw spa --quali --fuel 8    # quali stint, 8 L pinned
uv run optimize bmw spa --explore 10        # widen empirical envelope by 10% per side
uv run optimize bmw spa --reset             # full-envelope search; "fundamentally different" setup
uv run optimize bmw spa --detailed          # legacy per-param block format (vs default narrative)
uv run optimize ./my_session.ibt            # recommend by IBT path (auto-detect)
uv run optimize compare a.ibt b.ibt         # diff two setups per (corner, phase)
uv run optimize status bmw                  # coverage + fit-quality trend
uv run optimize calibrate bmw spa           # active-learning probes for thin-variance params
uv run optimize calibrate bmw spa --status  # just the per-parameter coverage table
uv run optimize bmw spa --physics           # Day 14: prepend physics-view banner (per-car evaluator weights, geometry, tyre floor)

uv run pytest -q                            # full suite (~15 min)
uv run pytest -q -m "not slow"              # fast suite (~2 min)
uv run ruff check src tests                 # lint (must stay clean)
```

## Tooling quirks

- Click 8.3+: `CliRunner(mix_stderr=False)` is gone — stderr always separated, access via `result.stderr`. Don't use `%%` in option help text either; Click no longer printf-formats it.
- `from racingoptimizer.physics import recommend` returns the FUNCTION (re-exported via `physics/__init__.py`), not the module. For `inspect.getsource` / module attrs, use `import racingoptimizer.physics.recommend as rec_mod` or `importlib.import_module("racingoptimizer.physics.recommend")`.
- Multi-line Python on Windows: use `uv run python << 'PY' ... PY` heredoc; `python -c "..."` truncates at first newline. `grep -P` (Perl regex) is also unavailable on the bundled grep — use the `Grep` tool's regex mode or a small Python snippet.
- Background `optimize` runs buffer Python stdout until exit; the output file stays at 0 bytes the entire run. Don't poll for progress — wait for the completion notification.

## Active build

VISION.md is decomposed into six slices plus three cross-cutting modules. Status reflects what is **merged** AND what has been **verified across all 5 GTP cars** (BMW M Hybrid V8, Porsche 963, Cadillac V-Series.R, Acura ARX-06, Ferrari 499P) versus only single-car (BMW Sebring fixture) smoke. Per VISION.md "do not assume a unified setup schema across cars" — the five cars have different suspension architectures, IBT YAML setup-blob shapes, and aero-map step sizes. **A green BMW test is not a "works" claim.**

**Two recommend code paths.** `optimize <car> <track>` routes by `PER_CAR_MODEL_CARS` (top of `src/racingoptimizer/cli/recommend.py`):

- **v4 (per-car, track-agnostic)** — currently `{"cadillac", "bmw", "ferrari"}`. `fit_per_car()` pools every session for the car across every track into one fitter; cache file `corpus/models/<car>__per-car__<digest>.pickle`. Adding a car requires (a) per-car constraint overrides in `constraints.md` (BMWBounds.md / Cadillacbounds.md / Ferraribounds.md as the source of truth), (b) car key added to `PER_CAR_MODEL_CARS`, (c) for tracks the car has never been driven on, the CLI's cross-car schedule fallback (`_maybe_borrow_cross_car_track`) borrows corner geometry from any other car's sessions on that track — so Ferrari@Spa works even though Ferrari has no Spa IBTs.
- **v3 (per-(car, track))** — Acura + Porsche. Trains per pair, uses donor-track extrapolation when target is unseen.

| Slice | Module | Code merged | Per-car verification scope |
|---|---|---|---|
| **A — IBT ingestion** | `racingoptimizer.ingest` | ✅ | Detect/normalize: ✓ all 5 (`tests/test_detect.py`). Per-car parser end-to-end: ✓ all 5 (`tests/test_parser_per_car.py`). |
| **B — Corner-phase decomposition** | `racingoptimizer.corner` | ✅ | ✓ all 5 (`tests/corner/test_per_car_smoke.py` loops the canonical car fixtures and asserts ≥1 corner detected). VISION §2 damper-velocities-vs-forces fully wired: `damper_force_p99_n` / `damper_force_mean_n` columns derived per-car via `physics.damper_force.estimate_damper_force_n` (digressive curve, inlined as a Polars expression to keep the pipeline columnar). |
| **C — Aero-map loader & interpolator** | `racingoptimizer.aero` | ✅ | ✓ all 5 (`tests/aero/test_loader.py::test_load_real_corpus_per_car` and `tests/aero/test_smoke.py::test_load_aero_maps_per_car_smoke`). |
| **D — Track model** | `racingoptimizer.track` | ✅ | ✓ all 5 (`tests/track/test_per_car_real_ibt.py`). Per-car curb-agreement threshold lives in `_PER_CAR_CURB_AGREEMENT_FRACTION` (Acura uses 0.3 instead of the 0.6 four-corner default — its heave/roll-shock signal aggregates more symmetrically and produces lower cross-session agreement). Per-IBT sample rate is threaded through `TrackModel.sample_rate_hz` and `_lap_length_from_speed_fallback` so high-rate (e.g. 360 Hz) recordings don't 6× the lap length. |
| **E — Physics fitter** | `racingoptimizer.physics` | ✅ — both halves merged (U9 train + U10 score/recommend). | ✓ all 5 (`tests/physics/test_per_car_fit_predict.py`). Acura gracefully degrades when shock-deflection channels are missing — flagged in `untrained_parameters`, fit still runs. |
| **F — CLI / recommendation rendering** | `racingoptimizer.cli`, `racingoptimizer.explain` | ✅ — `optimize <car> <track>`, `optimize <ibt_path>` (auto-detect), `optimize compare`, `optimize status` (U11). `optimize learn` (slice A) preserved. | ✓ all 5 (`tests/cli/test_per_car_smoke.py` loops the canonical car fixtures, runs `optimize <car> <track>` text + JSON, and asserts an exit-0 briefing with confidence + parameter blocks for every car). |

**Verification convention:** before claiming a slice "works" or marking it ✅, the slice must have a `tests/<slice>/test_per_car_smoke.py` (or equivalent) that loops the five canonical car fixtures (skipping when missing) and asserts the slice's contract holds. Single-car smoke tests are gaps to fill before the slice is called done.

**Repo layout:**

- `src/racingoptimizer/<slice>/` — implementation (`ingest`, `corner`, `aero`, `track`, `physics`, `cli`, `explain`, plus cross-cutting `context`, `confidence`, `constraints`).
- `tests/<slice>/` — pytest suites mirroring each module; per-car smoke tests live as `test_per_car_*.py`.
- `docs/superpowers/specs/` — the design spec for each slice (read before touching the code).
- `docs/VISION_COMPLIANCE.md` — per-clause file:line audit of how the code satisfies VISION.
- `GETTING_STARTED.md` / `README.md` — end-user docs (don't duplicate them here).

Cross-cutting modules (master-plan §2) — all merged:

- `racingoptimizer.context.EnvironmentFrame` — per-corner-phase atmospheric snapshot (12 channels: `AirTemp`, `AirDensity`, `AirPressure`, `RelativeHumidity`, `WindVel`, `WindDir`, `FogLevel`, `TrackTempCrew`, `TrackWetness`, `WeatherDeclaredWet`, `PrecipType`, `Skies`).
- `racingoptimizer.confidence.Confidence` — frozen `(value, lo, hi, n_samples, regime)` with `Confidence.derive(...)` regime-derivation classmethod (sparse short-circuits noisy at `n_samples < 30`).
- `racingoptimizer.constraints.{ConstraintsTable, load_constraints, clamp}` — markdown parser for `constraints.md` plus per-car-shadowing `clamp(value, parameter, car)`.

## Stint mode + conditions branching (`physics/wet_mode.py`, `physics/quali_mode.py`)

Two orthogonal axes feed `physics.score._conditions_adjusted_baselines`:

- **Dry vs wet** (VISION §10): `physics.wet_mode.classify_conditions(env)` returns `dry / damp / wet / full_rain` from `track_wetness` + `weather_declared_wet` + `precip_type`. Non-dry regimes swap baselines (lower max grip + aero baseline, higher wheelspin tolerance) **and** phase weights (less aero_eff, more platform + grip).
- **Race vs quali** (VISION §4 / §5): `--quali` swaps to `physics.quali_mode.quali_phase_weights` — `grip` x1.15, `aero_eff` x1.20, `platform` x0.55 (re-normalised so each phase still sums to 1.0). Quali takes precedence over the wet phase-weight pick (a wet quali still wants outright pace on a wet-adjusted baseline). Quali requires `--fuel N` (no auto fuel; quali fuel is per-track).

**Race-mode fuel auto-pin** (`cli/recommend.py::recommend_cmd` race-fuel block, search `Race fuel auto-pinned`): without `--quali` AND without `--fuel`, the CLI anchors `fuel_level_l` to the most-recent past-session value for the target track (typically ~58 L on the BMW M Hybrid V8). Substring-matches the user-typed track to the catalog slug (so `optimize bmw spa` finds `spa_2024_up`); falls back to all-car sessions only when the target track has none. Without this, the optimizer would treat fuel as a freely-fittable input and minimize mass for one-lap pace -- recommending fuel loads that won't cover a race.

Wind enters `physics.score.aero_eff` via `physics.wind.aero_wind_modifier` as a *magnitude* downforce penalty (treats `wind_vel_ms` as a tailwind worst case). Directional decomposition is deferred per `physics/wind.py` docstring — needs per-corner heading data the corner schedule doesn't carry yet.

## Trust radius + `--explore N` + `--reset` (`physics/recommend.py::_pin_or_trust_bounds`)

Per-car DE search is clipped to the **global corpus envelope** -- the union of every value the user has ever run this car at, on any track. Per VISION §3 the search will not extrapolate outside what the surrogate has been trained near, but per-target-track strictness was relaxed in `318d91d` because it left good setups off the table at every track the driver hadn't fully swept. The model is trained on the global corpus too, so the trust envelope matches training density.

`--explore N` widens the envelope by N% of each parameter's constraint span on each side, clipped to legal bounds. `0` (default) is strict empirical; 5-10 is modest exploration; 20-30 is aggressive. Recommendations landing in the widened territory carry weaker confidence by design.

`--reset` opens the search to `[corpus_min - 30%, corpus_max + 30%]` of constraint span on each side, skips the corpus-density pin check (so observed-constants can move), downgrades every parameter's confidence regime to `noisy`, and prints a `RESET MODE` banner to stderr. Use when the current setup feels fundamentally broken and small +/-1-click tweaks aren't moving the car. The optimizer is intentionally extrapolating beyond corpus density; verify on track before pushing.

Pin denominator uses the **empirical training range** (max−min observed across all pooled sessions), not the constraint span. Wide legal envelopes per BMWBounds.md (e.g. heave 0..900 N/mm) would otherwise mask real corpus variation as "near constant" and pin everything (regression test in `tests/physics/test_pin_near_constant.py`). Defensive guard: `target_observed` is clipped to the constraint envelope BEFORE the empirical-window math, so a user constraint pin (e.g. `--fuel 8` collapses to `(8, 8)`) outside the in-corpus values doesn't produce an inverted bound that crashes DE.

DE budget: `maxiter=15, popsize=20` (bumped from 5/10 on 2026-05-06). ~15 min per run on the 47-param BMW per-car search.

## Per-car model cache key (`cli/recommend.py::_model_cache_parts`)

Cache files at `corpus/models/<car>__per-car__<digest>.pickle` (or `<car>__<track>__<digest>.pickle` for v3). Digest folds:

1. `session_ids` — adding a session = new training data
2. `ontology` fingerprint — `(name, family, fittable, user_settable, json_path)` per spec. **The `json_path` is critical** — without it, a leaf-path correction (e.g. moving `fuel_level_l` from `Chassis.Fuel` to `BrakesDriveUnit.Fuel`) silently reuses the OLD pickle that trained against the wrong YAML field, masking the fix.
3. `constraints.md` content hash — bounds are baked into the pickle at fit time; editing them must invalidate the cache so DE doesn't search against stale bounds.
4. `FITTERS_LAYOUT_VERSION` (in `physics.fitters.__init__`) — bump when class names / module paths under `physics.fitters` change so old pickles don't fail to revive (`ModuleNotFoundError`).
5. `ENV_FEATURE_SCHEMA_VERSION[_PER_CAR]` — pre-S2.2 (v1), S2.2 env-12 (v2), Stage-3 coupled (v3), per-car (v4).

Editing `constraints.md` invalidates EVERY per-car cache (constraint content hash is a cache-key ingredient). Next recommend per car triggers a ~15-min refit. Plan constraint edits in batches.

## Confidence is track-wide, NOT parameter-specific

The `dense` / `noisy` / `sparse` label in the briefing header reflects overall corpus density at the target track, not specific evidence at the recommended value. A polluted corpus (e.g. GT3 sessions misrouted to BMW pre-1a8c9a3) will report "dense" while recommending values that have zero physics anchor. When a recommendation feels off, cross-check `model.per_track_parameter_observed[track][param]` for actual driven values before trusting it.

## Per-car v4 cross-track confounding

The joint surrogate pools every track for the car. If a parameter was held CONSTANT within a high-sample-count track (e.g. wing=17 across 24 Hockenheim Ferrari sessions) and varied only at a low-sample-count track (wing=14-15 across 6 Spa sessions), the regressor is dragged by sample weight regardless of corner-archetype features. Recommendations at the under-sampled track silently inherit the over-sampled track's setup philosophy. Mitigation: check whether the parameter has WITHIN-TRACK variance at the target before trusting recommendations; if not, the surrogate has no signal to learn track-specific interactions.

## IBT picker fields (`ingest/parser.py::_filename_recorded_at`)

`recorded_at` is parsed from the IBT filename (`<car>_<track> YYYY-MM-DD HH-MM-SS.ibt`), with the YAML's `WeekendInfo.WeekendOptions.Date` as fallback. The YAML field is iRacing's SCHEDULED race date for series events — identical across an entire weekend — so using it as a per-session timestamp made the most-recent-setup picker pick arbitrary winners. Filename-derived datetimes are unique per session.

Already-ingested sessions short-circuit on `status="ok"`. Use `optimize learn --reparse` to force re-processing after a parser change to refresh stale `recorded_at` / setup fields without manual catalog surgery.

The `_most_recent_setup_for(sessions_df)` picker sorts by `recorded_at desc`, with `ingested_at desc` as tiebreaker. Picker also drives the `(was X)` deltas in the renderer — wrong picker = misleading deltas.

## Pin denominator (`physics/recommend.py::_pin_or_trust_bounds`)

Decides whether to pin a parameter to its observed median or let DE search a trust-radius around it. The denominator for `observed_std / x < _NEAR_CONSTANT_FRACTION` is the **empirical training range** (max−min observed across all pooled sessions), not the constraint span. Wide legal envelopes per BMWBounds.md (e.g. heave 0..900 N/mm) would otherwise mask real corpus variation as "near constant" and pin everything. Falls back to the constraint span only when no per-track observed values exist (truly-constant params still pin correctly).

Defensive guard: `target_observed` is clipped to the constraint envelope BEFORE the empirical-window math runs, so a user constraint pin (e.g. `--fuel 8` collapses to `(8, 8)`) outside the in-corpus values (corpus only has 58 L) doesn't produce an inverted `(57, 8)` bound that crashes DE's seed_population.

## Setup-card renderer contract (`explain/full_setup_card.py`)

Mirror precedence: `_MIRRORED_LEAVES` is checked BEFORE `opt_match` in `_render_panel` (commit 06f0c23). Adding a mirror entry overrides any per-corner DE recommendation for that path. Required so right-side dampers (which have their own per-corner ParameterSpec) can mirror from the left rather than render an independent DE result.

Watch-most + Why-line corner picker filter impacts by `_FAMILY_PREFERRED_PHASES` before max-by-score-delta (commits 7be3017, 5f305b4). Without this, the corner-duration weighting from e90e8fd makes the longest corner dominate every parameter's "Watch most" line.

Every garage line carries one tag. The set is closed:

| Tag | Meaning |
|---|---|
| `[OPT]` | Optimizer recommendation, post-clamp, rounded to iRacing UI step. |
| `[OPT pin]` | Optimizer pinned to observed median (no per-session variance to learn from). |
| `[OPT mirror]` | Per-axle parameter mirrored onto the symmetric corner. iRacing UI requires LR=RR for: rear coil spring rate, front + rear torsion bar turns, front + rear torsion bar OD, rear toe-in. |
| `[past]` | Most-recent session value; no constraint bounds yet to optimize against. |
| `[readout]` | iRacing-calculated, past-session value. Driver cannot type these. |
| `[predicted]` | Same readout but evaluated by `PhysicsModel.predict_setup_readouts()` at the new setup vector — what iRacing will display after the user enters the `[OPT]` values. |

Static ride heights, corner weights, deflections, and the `AeroCalculator` block are all readouts; the `[predicted]` path covers `setup_static_*_ride_height_mm` and falls back to `[readout]` for the rest.

Per-axle mirroring lives in `_MIRRORED_LEAVES` (5 entries today); predicted-readout path mapping in `_PREDICTED_READOUT_PATHS`. Add to either dict to extend coverage. The renderer also snaps `[OPT]` numeric values to either the parameter's uniform `step` OR the non-uniform `discrete_values` list (torsion bar OD's 14 diameters; clutch plates {2, 4, 6}).

## Plain-English narrative (`explain/narrative.py`)

DEFAULT briefing renderer since 2026-05-06. Replaces the legacy per-parameter `Helps:`/`Hurts:`/score-delta blocks with a 2–3 line summary per change in handling vocabulary (pitch / roll / understeer / oversteer / aero stall / bottoming / turn-in / throttle traction). Per change:

```
Front heave spring: 60 -> 50 N/mm  (softens)
  Effect: More front compliance over kerbs; better front grip
    retention through bumpy entries; smoother brake-release.
  Trade:  More pitch dive under heavy braking; slower turn-in;
    aero platform less stable on Kemmel compression.
  Watch most: T9 mid-corner
```

OVERALL DIRECTION at the top aggregates moves by THEME (perches + pushrods both → "ride height"; spring_rate + heave_spring → "platform stiffness"), with `(mixed direction)` annotation when a theme has both up and down moves.

`--detailed` reverts to the legacy block format for engineering drill-down + the `setup-justifier` validator agent. Translation tables: `_DIRECTION_VERB` (per-row terse verb), `_OVERALL_FRAGMENT` (fluent OVERALL phrase per theme), `_CAR_FEEL` (Effect + Trade per `(family, mode, axis, direction)`). New parameter families need entries in all three (and a label in `_PARAM_LABEL`) to render the full vocabulary; missing families fall through to a phase-themed `Helps:`/`Watch:` corner-list line.

Output is ASCII-only (Windows cp1252 console can't encode Unicode arrows / em-dashes). Use `->`, `--`, `!`, `+/-`, etc.

Specs live under `docs/superpowers/specs/`; plans under `docs/superpowers/plans/`. Read the active slice's spec before touching its code.

`constraints.md` covers most legal-range parameters:
- Wing, tyre pressure, heave spring/slider, static ride heights (shared defaults)
- BMW per-car overrides per BMWBounds.md (heave 0–900, third 0–900, coil 105–280, ±100 mm rear perches)
- Cadillac, Ferrari, Acura, Porsche per-car overrides similarly grounded in their respective bounds files
- Fuel level (1–100 L, fittable, with race auto-pin in CLI)
- ARB Size front + rear (categorical, BMW = Disconnect/Soft/Medium/Stiff, Ferrari = Disconnected/A..E)
- Diff coast/drive ramps + clutch friction plates (categorical / discrete numeric)
- Front diff preload (Ferrari only; -50..+50 Nm)
- Toe-in, mm-based (front axle scalar, rear per-corner with LR=RR mirror)
- Per-corner damper bounds with per-car-override fan-out (Ferrari overrides global 0–11 to 0–40)
- Per-corner torsion bar turns + OD (BMW/Cadillac front; Ferrari all 4)

Still `<TODO: from iRacing UI>`: corner weights (4 of them) and brake duct openings. Slice E's `fit` gracefully degrades — lists CE-gated unbounded parameters in `untrained_parameters` and does not refuse to run.

Categorical params (ARB size, diff coast/drive ramps) are encoded as ordinal indices via `ParameterSpec.choices`; the renderer maps the rounded index back to the label. Non-uniform numeric discrete sets (torsion bar OD's 14 diameters on BMW/Cadillac; clutch's `{2, 4, 6}`) use `ParameterSpec.discrete_values` and snap to the nearest legal value at render time.

## Active-learning probes (`cli/calibrate.py`)

`optimize calibrate <car> <track>` surfaces parameters with thin observed variance on the target track and proposes a value in the largest unsampled gap of each one's legal range. Drives the loop: probe -> drive a clean stint with the proposed values -> `optimize learn` -> re-fit -> the next recommend has slope/curvature where the parameter used to pin to its observed median.

* Default mode: coverage table + top-N (default 3, controlled by `--targets N`) thin-variance probes + a synthetic setup card with probes tagged `[OPT]` and everything else `[OPT pin]` from the past setup.
* `--status`: coverage table only. Use to decide whether more variance is the bottleneck on accuracy.
* `--output-file PATH` / `--output-file -`: same convention as recommend; default writes to `recommendations/<car>-<short-track>-cal[-status]-<MMDD>-<HHMM>.txt`.

The picker reads `model.per_track_parameter_observed[track]` (populated by `fit_per_car`); proposals are step-snapped to the parameter's UI step or `discrete_values` set; targets that would step-snap onto the past value get re-routed to the second-largest gap.

## Recommendation filename convention (`cli/recommend.py::recommend_cmd` output-file block; `cli/calibrate.py::_maybe_save`)

Default output path: `recommendations/<car>-<short-track>-<mode>[-<fuel>L]-<MMDD>-<HHMM>.<ext>`. `<short-track>` strips variant suffixes via `_short_track` (`spa_2024_up` -> `spa`, `hockenheim_gp` -> `hockenheim`, `sebring_international` -> `sebring`, `daytona_2011_road` -> `daytona`). Mode tag is one of `race` / `quali` / `reset` / `cal` / `cal-status`. Fuel suffix only for quali stints (race auto-pins, would clutter every name). Pass `--output-file -` to suppress file output (e.g. when piping JSON).

**Known regressions / gaps:**

- `optimize <car> <track> --json` emits valid JSON to stdout, then a trailing `\n[saved to recommendations\<...>.txt]` to stderr. Click's `CliRunner` mixes stderr into `result.output` by default, so `tests/cli/test_per_car_smoke.py::test_recommend_per_car_json` fails JSON-decoding the combined stream. The text smoke test (the merge gate) passes. Fix: either suppress the auto-save stderr line under `--json`, or set `mix_stderr=False` in the test.
- Corner weights still `<TODO>` in `constraints.md`; render as `[past]` and appear in `untrained_parameters`.
- Wind decomposition uses tailwind-worst-case magnitude only; per-corner directional headwind/crosswind correction needs heading data the corner schedule doesn't carry yet (deferred per `physics/wind.py` docstring).
- `fuel_level_l` is fittable but most cars' corpora have thin fuel variance — `predict_setup_readouts` learns whatever pattern exists, which may not be physically pure fuel→RH.
- Driver-input output channels (throttle, brake, damper velocity) plateau at fit_quality ~0.50 (signal == noise). Structural ceiling — the model can't fully resolve channels that depend more on driver input than setup.
- BMW corpus is Sebring-dominated (37/53 sessions). Spa-specific predictions (11 sessions) are weaker than Sebring's by design.

## Project automations (`.claude/`)

- **Hook — PreToolUse on Bash:** blocks destructive ops (`rm`, `mv`, `truncate`, `shred`, `dd`, output redirect) targeting `ibtfiles/` or `aero-maps/`. These dirs hold irreplaceable training data. If a hook fires against a real need, fix the hook or run the command outside Claude Code — do not work around it with `--no-verify` style hacks.
- **Hook — PostToolUse on Edit/Write:** sanity-checks `constraints.md` keeps required headings (`## Defaults`, `## Per-car overrides`).
- **Skill `ibt-inspect`:** summarizes channels / setup / weather for any `.ibt` path. Use during exploratory work on telemetry.
- **Skill `add-constraint`** (user-invocable only): appends new parameter ranges to `constraints.md` in the established schema.
- **Subagent `setup-justifier`:** gates VISION §7 — every recommended parameter must have corner trade-offs, telemetry evidence, sensitivity, and confidence.
- **Subagent `physics-fit-validator`:** verifies VISION §3/§6 — residuals on held-out laps, confidence-vs-density alignment, leaking-formula detection.
- **MCP — context7** (user-scoped): live docs for numpy/scipy/pandas/pyirsdk. Prefer it when uncertain about library APIs instead of guessing.

## Data assets (the only inputs that exist today)

- `ibtfiles/` — raw iRacing telemetry binaries (`.ibt`). 60 Hz, 100+ channels per lap, with the session's garage setup embedded in the YAML header. Filenames encode car and track, e.g. `acuraarx06gtp_daytona 2011 road 2026-04-03 20-31-55.ibt`. Filenames may contain spaces — quote paths. The directory mixes flat files with subfolders (e.g. `ibtfiles/cadillac/`); ingestion must walk recursively. Files range from ~1 MB to ~210 MB; do not load all telemetry into memory at once.
- `aero-maps/` — parsed aerodynamic surfaces, one JSON per `(car, wing_angle)` pair. Five cars (`acura`, `bmw`, `cadillac`, `ferrari`, `porsche`) × discrete wing angles. Schema (per file):
  - `car`, `wing` — identifiers
  - `front_rh_mm`, `rear_rh_mm` — 1D axis grids (front and rear ride height in mm)
  - `balance_pct` — 2D matrix indexed `[front_rh][rear_rh]`, aero balance %
  - `ld_ratio` — 2D matrix, lift/drag ratio
  - Note wing-angle granularity differs per car (e.g. acura indexes at 0.5° steps; others at 1.0°). Do not assume a uniform step.
- `constraints.md` — hard legal bounds for setup parameters (wing, tyre cold pressure, heave spring/slider deflection, static ride heights so far; more incoming). Defaults apply to all cars; per-car overrides shadow them (e.g. Acura wing 6.0–10.0°). The optimizer MUST clamp every recommended value to these bounds before output.

## Architectural commitments from VISION.md (non-obvious — read before designing)

These are deliberate design constraints, not suggestions. Diverging from them defeats the project's premise:

- **Corner-phase is the atomic unit, not the lap.** Segment every lap into corners, then into braking / trail-brake / mid-corner / exit / straight phases. All physics fitting and scoring happens at the corner-phase grain. Never collapse to session averages.
- **Empirical physics, not textbook formulas.** Spring-rate effects, LLTD, aero balance vs ride height — fit them from observed data. No hardcoded engineering equations as the primary model. Aero maps are the one exception (they're empirical lookups already).
- **Lap time is an outcome, not the optimization signal.** Score setups by per-corner-phase physics utilization weighted by each corner's time sensitivity, then sum. Never feed lap time directly into the objective.
- **Track model is compounding and load-bearing for data quality.** It identifies curbs, bumps, rumble strips, and off-track excursions per track position so the physics fitter can mask out non-clean samples. Do not throw away whole laps — mask the dirty sections.
- **Every data point carries environmental context.** Air density, track temp, wind, wetness are 60 Hz channels in the IBT and must travel with each sample through fitting and prediction. The aero maps need an air-density correction; do not treat them as condition-invariant.
- **Output explains every click.** Setup recommendations must justify each parameter with the corners that benefit, the corners that compromise, the telemetry evidence, and the ±1–2 click sensitivity. A bare list of numbers is not an acceptable output format.
- **Confidence is a first-class output.** Sparse/noisy parameter interactions → conservative recommendations; dense/consistent → aggressive. Report confidence alongside every value.

## CLI UX target (VISION.md §8)

Top-level command is `optimize`, with auto-detection of car and track from IBT filenames. Avoid long flag chains and module-path invocations. The default path is "drop in an IBT, get a setup out." Power-user flags are fine as overrides, not as the primary interface.

Two recommend invocations are supported and routed by `_OptimizeGroup.parse_args` in `cli/__init__.py`:

- `optimize <car> <track>` — explicit pair. Track input is slugified before catalog lookup so `laguna-seca`, `Laguna Seca`, and `lagunaseca` all match.
- `optimize <ibt_path>` — first positional is an existing `.ibt` file; car & track are sniffed from the filename via `detect_car_from_filename` / `detect_track_from_filename`.

Both forms route to the same `recommend_cmd`. The end-user-facing walkthrough lives in `GETTING_STARTED.md`; do not duplicate it here.

## Cars covered

BMW M Hybrid V8, Porsche 963, Cadillac V-Series.R, Acura ARX-06, Ferrari 499P. Each has its own suspension architecture and garage parameter set — do not assume a unified setup schema across cars.

## When to override the optimizer

The score function is per-corner-phase utilization, NOT lap time (VISION §6). For most parameters this correlates well with stopwatch pace. Three known disconnects:

- **Tyre pressure**: surrogate rewards platform-stability (cleaner ride-height telemetry) which higher cold P delivers, but doesn't see the peak-grip loss from smaller contact patch. Community wisdom (152 floor for GTP) wins on lap time.
- **Stiff vs soft setup philosophy**: when a user's recent corpus drifts toward conservative setups (rear toe-in, soft ARB, tight diff), the surrogate inherits that bias even when the same user has a faster validated setup in their IBT history. Compare optimizer recommendation to the user's fastest IBT before applying.
- **Pinned-outside-corpus parameters**: `--pin <param>=<value>` outside the global corpus envelope produces curve-fit extrapolation, not engineered compensation. Confidence still reads "dense" because density is track-wide. Verify on track.
