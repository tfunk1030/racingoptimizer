# Getting started

`optimize` reads your iRacing telemetry and recommends a setup. Five GTP cars supported.

## Install once

```bash
uv venv
uv pip install -e ".[dev]"
```

## Use it

```bash
uv run optimize learn ./ibtfiles            # ingest your .ibt files
uv run optimize cadillac lagunaseca         # recommend a race setup
uv run optimize bmw spa --quali --fuel 8    # quali stint, 8 L pinned
uv run optimize bmw spa --physics           # add physics-view banner above the briefing
uv run optimize ./my_session.ibt            # OR drop in any .ibt — auto-detects
```

The `--physics` flag prepends an informational banner showing per-car
evaluator weights, geometry (wheelbase, weight distribution), and tyre
floor pin status. Recommendation values are unchanged — the banner is
read-only diagnostic context.

The output has two parts:

1. **Plain-English briefing** (default) — every parameter that moved, written in handling vocabulary (pitch, roll, understeer, oversteer, turn-in, throttle traction, kerb compliance), grouped by family (PLATFORM / DAMPERS / BALANCE / BRAKES & DRIVETRAIN / AERO). Each change shows what it does (Effect), what it costs (Trade), the single corner-phase where you'll feel it most, and compact **±1-click sensitivity** score deltas. Header gives you mode (race vs quali), fuel load, conditions, and overall confidence. Use `--detailed` for the legacy per-parameter block format with full evidence tables.
2. **Full setup card** — every garage parameter for the car, ready to enter, tagged:
   * `[OPT]` — value the optimizer chose.
   * `[OPT pin]` — pinned to your observed median (the model has no signal to deviate, e.g. you ran the same value every session).
   * `[OPT mirror]` — value mirrored from the per-axle parameter on the opposite corner. iRacing UI requires LR=RR for: rear coil spring rate, rear spring perch offset, front + rear torsion bar turns, front + rear torsion bar OD, rear toe-in, and all right-side damper clicks (5 modes × 2 axles).
   * `[past]` — copied from your most recent session (no constraint bounds yet).
   * `[readout]` — calculated by iRacing using your last session (you don't enter this). Includes **corner weights** and static ride heights before you apply `[OPT]`.
   * `[predicted]` — calculated readout the optimizer projects under the new setup (e.g. predicted static ride heights at the new perch / pushrod / spring values; this is what iRacing will display *after* you enter the `[OPT]` numbers).

## Other commands

```bash
uv run optimize compare a.ibt b.ibt         # diff two setups corner-phase by corner-phase
uv run optimize status cadillac           # what does the model know about this car?
uv run optimize calibrate bmw spa           # propose probes that grow corpus coverage
uv run optimize calibrate bmw spa --status  # just print the per-parameter coverage table
```

## Active learning loop with `calibrate`

`optimize calibrate` finds the parameters where you've driven only one (or two)
distinct values on the target track and proposes a value in the largest unsampled
gap of each one's legal range. The recommender can't learn slopes from a parameter
you've held constant, so a deliberate probe-then-re-fit loop turns those `[OPT pin]`
parameters into real fittable ones.

```
optimize calibrate bmw spa            # see status + 3 probes + a setup card
# enter the probe values into the iRacing garage, drive a clean stint
optimize learn ./ibtfiles             # ingest the new IBT
optimize bmw spa                      # re-fit and recommend with fresh variance
```

Use `--status` when you want the coverage table without proposals (e.g. to decide
whether more variance is the bottleneck on accuracy). `--targets N` sets how many
probes to bundle into one stint; default 3 is a good balance between learning per
session and keeping the changes feel-able and unconfounded.

## Useful options on `optimize <car> <track>`

* `--json` — machine-readable JSON instead of text. Pin/fuel/reset notices and guardrail warnings land in the `warnings` array (not stderr). Pass `--output-file -` when piping through `jq`.
* `--detailed` — legacy per-parameter block format (Helps/Hurts with score deltas, ±1-click sensitivity, evidence) instead of the default plain-English narrative. Useful for engineering drill-downs.
* `--wing 17` — pin the rear wing angle.
* `--fuel 8` — pin race-fuel level (L). Race default is the past-session value at the target track (~58 L on BMW M Hybrid V8); quali stints are user-input depending on track length (commonly 5..15 L for 3 laps + reserve). The optimizer treats fuel as a fittable input so a low pin influences ride-height and balance predictions.
* `--quali` — quali-stint mode: phase weights tilt toward outright single-lap pace (more aero efficiency, more grip utilisation, less platform conservatism). Pair with `--fuel N` to pin the matching low fuel load — the optimizer will not auto-pick a quali fuel.
* `--explore N` — widen the empirical envelope by N% of each parameter's constraint span on each side, clipped to legal bounds. Lets the optimizer probe values outside what you've actually driven; recommendations in the widened territory carry weaker confidence by design. Try 5–10 for modest exploration, 20–30 for aggressive "what if I tried something different".
* `--reset` — open the search to `[corpus_min - 30%, corpus_max + 30%]` of constraint span on each side, skip the corpus-density pin check (so observed-constants can move), downgrade every parameter's confidence to `noisy`, and print a `RESET MODE` banner to stderr. Use when the current setup feels fundamentally wrong and small ±1-click tweaks aren't moving the car. The optimizer is intentionally extrapolating beyond corpus density; verify on track before pushing.
* `--output-file PATH` — override the default save location (`recommendations/<car>-<short-track>-<mode>[-<fuel>L]-<MMDD>-<HHMM>.<ext>`). Pass `-` to suppress file output entirely (useful when piping `--json` through `jq`).
* `--wetness 0.0` — override conditions (also `--air-temp`, `--track-temp`, `--wind`). Wet track triggers `wet_mode` baselines + phase weights automatically.
* `--physics` — prepend informational physics-view banner (evaluator weights, geometry, tyre floor pin). Recommendations unchanged.
* `--surrogate-only` — disable hybrid physics blend in DE; use surrogate score + axle guardrail penalty only. Default is hybrid (`hybrid_score()`).
* `--no-cache` — refit from scratch (use after changing `constraints.md` or recovering from a stale pickle).
* `--corpus-root ./somewhere` — non-default corpus location.

## `optimize learn` flags

* `--reparse` — force re-processing of already-ingested sessions. Use after a parser change to refresh stale catalog fields (e.g. `recorded_at` after the filename-derived parser fix). Normal `learn` skips sessions whose `status="ok"` to avoid redundant work.
* `--corpus-root ./somewhere` — same as recommend.

## Stint modes

The optimizer has two stint objectives:

* **Race** (default) — phase weights favor platform consistency, aero efficiency averaged across many laps, conservative grip utilisation. Fuel auto-pinned to your most-recent past session at the target track (typically ~58 L on the BMW); you'll see a `Race fuel auto-pinned to past-session value: X L` notice on stderr explaining the pick.
* **Quali** (`--quali --fuel N`) — phase weights tilt toward outright single-lap pace. You must pin the fuel load explicitly (per-track choice; longer tracks need more for 3 laps + reserve). Wet-mode still applies if conditions warrant it (a wet quali = wet baselines + quali phase weights).

The Stint header on the briefing tells you which mode + fuel produced the recommendation.

## Per-car model (all five GTP cars)

All five cars use the **v4 per-car pooled path**: one model trained on every
session of that car across every track, with corner-geometry archetype features
that let the same model score any track. When a car has never been driven on a
target track, the CLI borrows the corner schedule from another car's sessions
on that track (cross-car schedule fallback) — the model stays per-car; only
the geometry is borrowed.

## What lives where

| | path |
|---|---|
| Your raw IBTs | wherever you point `learn` |
| Ingested corpus | `./corpus/` (override with `--corpus-root` or `RACINGOPTIMIZER_CORPUS`) |
| Constraint bounds | `constraints.md` — edit to tighten/loosen the legal range for a parameter |

## When something looks wrong

* **`unknown car`** — accepts `acura`, `bmw`, `cadillac`, `ferrari`, `porsche`.
* **`model has no data on (car, track); run optimize learn …`** — corpus doesn't have that combination yet.
* **`out of envelope` warnings during recommend** — cosmetic; the optimizer probes ride heights briefly outside the aero map envelope, the interpolator clamps and continues.
* **Recommendation looks weird** — check the briefing **NOTES** / `Warnings:` section. Common ones:
  * `pinned to observed median` — you held that parameter constant in training (no variance to learn from).
  * `Untrained (no constraints.md bounds)` — parameter has no legal bounds captured yet.
  * `Not searched (calculated readout or blocked in ontology)` — e.g. corner weights, static ride heights; you cannot type these in the garage.
  * `Predicted … static ride height … outside observation envelope` — predicted platform readout after entering `[OPT]` may not match what iRacing shows; verify perch/pushrod/spring inputs before running.
  * `Predicted front heave slider deflection … exceeds 45 mm` — iRacing tech limit; soften front platform or raise perch.
  * Physics guardrail lines (`axle utilization > 1.0`, aero balance off-target, physics-vs-surrogate divergence).

## Product posture (honest)

The optimizer is a **guardrailed surrogate** with physics checks — not a
lap-time-correlated physics predictor. Evaluator Spearman vs corner duration
on this corpus averages ~0.19 (below the 0.35 design target). Use `[OPT]`
values as starting points; validate static ride heights and platform readouts
in the iRacing garage before pushing on track.

## Where to dig deeper

* `VISION.md` — what the optimizer is and isn't.
* `docs/VISION_COMPLIANCE.md` — per-clause file:line audit.
* `docs/audit_2026-05-23/00_findings_and_fix_plan.md` — latest audit findings + implementation status.
* `CLAUDE.md` — engineering conventions.

## Tests

```bash
uv run pytest -q                 # full suite (~15 min)
uv run pytest -q -m "not slow"   # fast suite (~2 min)
```
  