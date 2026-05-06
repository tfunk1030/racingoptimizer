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
uv run optimize ./my_session.ibt            # OR drop in any .ibt — auto-detects
```

The output has two parts:

1. **Plain-English briefing** (default) — every parameter that moved, written in handling vocabulary (pitch, roll, understeer, oversteer, turn-in, throttle traction, kerb compliance), grouped by family (PLATFORM / DAMPERS / BALANCE / BRAKES & DRIVETRAIN / AERO). Each change shows what it does (Effect), what it costs (Trade), and the single corner-phase where you'll feel it most. Header gives you mode (race vs quali), fuel load, conditions, and overall confidence. Use `--detailed` to swap to the legacy per-parameter block format with raw score deltas + ±1-click sensitivity.
2. **Full setup card** — every garage parameter for the car, ready to enter, tagged:
   * `[OPT]` — value the optimizer chose.
   * `[OPT pin]` — pinned to your observed median (the model has no signal to deviate, e.g. you ran the same value every session).
   * `[OPT mirror]` — value mirrored from the per-axle parameter on the opposite corner. iRacing UI requires LR=RR for: rear coil spring rate, front + rear torsion bar turns, front + rear torsion bar OD, rear toe-in.
   * `[past]` — copied from your most recent session (no constraint bounds yet).
   * `[readout]` — calculated by iRacing using your last session (you don't enter this).
   * `[predicted]` — calculated readout the optimizer projects under the new setup (e.g. predicted static ride heights at the new perch / pushrod / spring values; this is what iRacing will display *after* you enter the `[OPT]` numbers).

## Other commands

```bash
uv run optimize compare a.ibt b.ibt         # diff two setups corner-phase by corner-phase
uv run optimize status cadillac           # what does the model know about this car?
```

## Useful options on `optimize <car> <track>`

* `--json` — machine-readable JSON instead of text.
* `--detailed` — legacy per-parameter block format (Helps/Hurts with score deltas, ±1-click sensitivity, evidence) instead of the default plain-English narrative. Useful for engineering drill-downs.
* `--wing 17` — pin the rear wing angle.
* `--fuel 8` — pin race-fuel level (L). Race default is the past-session value at the target track (~58 L on BMW M Hybrid V8); quali stints are user-input depending on track length (commonly 5..15 L for 3 laps + reserve). The optimizer treats fuel as a fittable input so a low pin influences ride-height and balance predictions.
* `--quali` — quali-stint mode: phase weights tilt toward outright single-lap pace (more aero efficiency, more grip utilisation, less platform conservatism). Pair with `--fuel N` to pin the matching low fuel load — the optimizer will not auto-pick a quali fuel.
* `--explore N` — widen the empirical envelope by N% of each parameter's constraint span on each side, clipped to legal bounds. Lets the optimizer probe values outside what you've actually driven; recommendations in the widened territory carry weaker confidence by design. Try 5–10 for modest exploration, 20–30 for aggressive "what if I tried something different".
* `--reset` — abandon the per-track trust radius entirely and search the FULL constraint envelope from each parameter's midpoint. Use when the current setup feels fundamentally wrong and small ±1-click tweaks aren't moving the car. Recommendations diverge sharply from your past setup (typically 40+ of ~47 parameters move), every parameter's confidence is downgraded to `noisy`, and the briefing prints a `RESET MODE` banner. Verify on track before pushing — the optimizer is extrapolating outside corpus density on purpose.
* `--wetness 0.0` — override conditions (also `--air-temp`, `--track-temp`, `--wind`). Wet track triggers `wet_mode` baselines + phase weights automatically.
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

## Per-car vs cross-car

`bmw`, `cadillac`, `ferrari` use the per-car path: one model trained on every session of that car across every track, with corner-geometry archetype features that let the same model score any track. Ferrari at a track it's never been driven on (e.g. Spa) borrows the corner schedule from BMW/Cadillac data on that track — the model is still Ferrari, only the geometry is borrowed.

`acura` and `porsche` use the per-(car, track) path: one model per (car, track) pair, with donor-track extrapolation when the target is unseen.

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
* **Recommendation looks weird** — check the briefing's `Warnings:` section. Common ones: `pinned to observed median` (you held that parameter constant in training, no signal), `untrained_parameters` (bounds not in `constraints.md` yet).

## Where to dig deeper

* `VISION.md` — what the optimizer is and isn't.
* `docs/VISION_COMPLIANCE.md` — per-clause file:line audit.
* `CLAUDE.md` — engineering conventions.

## Tests

```bash
uv run pytest -q                 # full suite (~15 min)
uv run pytest -q -m "not slow"   # fast suite (~2 min)
```
  