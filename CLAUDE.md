# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

`racingoptimizer` is a partially-built Python package targeting `pip install .` and the `optimize` CLI (VISION.md §8). Five of the six VISION slices are merged; the recommendation half of slice E and slice F (CLI) remain. Install with `uv venv && uv pip install -e ".[dev]"`; run `uv run optimize learn ./ibtfiles` to ingest telemetry into the corpus and `uv run pytest` to exercise the ~250-test suite. Read `VISION.md` first — it is the spec.

## Active build

VISION.md is decomposed into six slices plus three cross-cutting modules. Status as of the latest merge:

| Slice | Module | Status |
|---|---|---|
| **A — IBT ingestion** | `racingoptimizer.ingest` | ✅ merged |
| **B — Corner-phase decomposition** | `racingoptimizer.corner` | ✅ merged (`detect_corners`, `assign_phases`, `segment_lap`, `corner_phase_states`) |
| **C — Aero-map loader & interpolator** | `racingoptimizer.aero` | ✅ merged (`load_aero_maps`, `AeroSurface`, `BASELINE_AIR_DENSITY`) |
| **D — Track model** | `racingoptimizer.track` | ✅ merged (`build_track_model`, `TrackModel`, `compute_curb_mask`, `compute_off_track_mask`, `apply_quality_mask`) |
| **E — Physics fitter** | `racingoptimizer.physics` | 🟡 partial — `fit`, `PhysicsModel.predict`, ontology, GP/RF fitters merged (U9). `score_setup`, `recommend`, corner-weight derivation, `SetupRecommendation` are unmerged (U10, in flight). |
| **F — CLI / recommendation rendering** | `racingoptimizer.cli`, `racingoptimizer.explain` | ⏳ pending (U11). Slice A's `optimize learn` subcommand is wired; the `optimize <car> <track>`, `compare`, and `status` subcommands and the engineering-briefing renderer are not built yet. |

Cross-cutting modules (master-plan §2) — all merged:

- `racingoptimizer.context.EnvironmentFrame` — per-corner-phase atmospheric snapshot (`AirDensity`, `TrackTempCrew`, `WindVel`, `WindDir`, `TrackWetness`).
- `racingoptimizer.confidence.Confidence` — frozen `(value, lo, hi, n_samples, regime)` with `Confidence.derive(...)` regime-derivation classmethod (sparse short-circuits noisy at `n_samples < 30`).
- `racingoptimizer.constraints.{ConstraintsTable, load_constraints, clamp}` — markdown parser for `constraints.md` plus per-car-shadowing `clamp(value, parameter, car)`.

Specs live under `docs/superpowers/specs/`; plans under `docs/superpowers/plans/`. Read the active slice's spec before touching its code. The Wave-by-Wave decomposition that produced the eleven implementation units (U1–U11, of which 1–9 are merged) lives in `C:/Users/VYRAL/.claude/plans/ultrathink-and-read-through-wobbly-horizon.md`.

`constraints.md` covers the bounded subset (wing, tyre pressure, heave spring/slider, static ride heights). ARBs, dampers, corner weights, brake bias, differential, camber, toe, brake ducts, and throttle/brake mapping are all `<TODO: from iRacing UI>` placeholders awaiting manual UI capture. Slice E's `fit` gracefully degrades — it lists CE-gated unbounded parameters in `untrained_parameters` and does not refuse to run.

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

## Cars covered

BMW M Hybrid V8, Porsche 963, Cadillac V-Series.R, Acura ARX-06, Ferrari 499P. Each has its own suspension architecture and garage parameter set — do not assume a unified setup schema across cars.
