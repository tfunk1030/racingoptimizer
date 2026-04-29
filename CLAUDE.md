# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

This is a **greenfield project**. Only data assets and `VISION.md` exist — no source code, package manifest, build scripts, tests, or CI. The IBT-ingestion subsystem (`racingoptimizer.ingest`) is now implemented and the `optimize` CLI is wired up; install with `uv venv && uv pip install -e ".[dev]"`, then run `uv run optimize learn ./ibtfiles` to ingest telemetry into the corpus and `uv run pytest` to exercise the suite. Everything else (corner-phase decomposition, aero-map loader, fitter, optimizer, track model) is still unimplemented — when asked to build them, you are creating them from scratch. Read `VISION.md` first — it is the spec.

There is no `pyproject.toml`/`setup.py`/`package.json` yet. The vision targets a `pip install .` Python package exposing an `optimize` CLI (see VISION.md §8 for the intended UX). Match that when scaffolding.

## Active build

VISION.md is decomposed into independent slices, each with its own spec → plan → implementation cycle:

- **A — IBT ingestion** (`racingoptimizer.ingest`). Spec: `docs/superpowers/specs/2026-04-28-ibt-ingestion-design.md`. Plan: `docs/superpowers/plans/2026-04-28-ibt-ingestion.md`. **This is the current slice.** Until it lands, no other slice can be built.
- **B — Corner-phase decomposition.** Depends on A.
- **C — Aero-map loader & interpolator.** Independent of A; smallest well-bounded unit.
- **D — Track model.** Depends on A + B.
- **E — Physics fitter.** Depends on A + B (+ C).
- **F — CLI scaffolding.** Skeleton wired up incrementally as A–E land.

Specs and plans live under `docs/superpowers/`. Read the active slice's spec before touching its code.

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
