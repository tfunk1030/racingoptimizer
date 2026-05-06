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

1. **Engineering briefing** — every parameter the optimizer changed, with the corners it helps, the corners it hurts, and ±1-click sensitivity. The header line tells you the stint mode (race vs quali), fuel load, and conditions (AirTemp, TrackTemp, AirDensity, Wind, Wetness).
2. **Full setup card** — every garage parameter for the car, ready to enter, tagged:
   * `[OPT]` — value the optimizer chose.
   * `[OPT pin]` — pinned to your observed median (the model has no signal to deviate, e.g. you ran the same value every session).
   * `[OPT mirror]` — value mirrored from the per-axle parameter on the opposite corner (currently only rear coil spring rate, since iRacing requires LR=RR).
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
* `--wing 17` — pin the rear wing angle.
* `--fuel 8` — pin race-fuel level (L). Race default is the past-session value (~58 L on BMW M Hybrid V8); quali stints are user-input depending on track length (commonly 5..15 L for 3 laps + reserve). The optimizer treats fuel as a fittable input so a low pin influences ride-height and balance predictions.
* `--quali` — quali-stint mode: phase weights tilt toward outright single-lap pace (more aero efficiency, more grip utilisation, less platform conservatism). Pair with `--fuel N` to pin the matching low fuel load — the optimizer will not auto-pick a quali fuel.
* `--wetness 0.0` — override conditions (also `--air-temp`, `--track-temp`, `--wind`). Wet track triggers `wet_mode` baselines + phase weights automatically.
* `--no-cache` — refit from scratch (use after changing `constraints.md` or recovering from a stale pickle).
* `--corpus-root ./somewhere` — non-default corpus location.

## Stint modes

The optimizer has two stint objectives:

* **Race** (default) — phase weights favor platform consistency, aero efficiency averaged across many laps, conservative grip utilisation. Fuel defaults to the past-session value (typically 58 L on the BMW).
* **Quali** (`--quali --fuel N`) — phase weights tilt toward outright single-lap pace. You must pin the fuel load explicitly (per-track choice; longer tracks need more for 3 laps + reserve). Wet-mode still applies if conditions warrant it (a wet quali = wet baselines + quali phase weights).

The Stint header on the briefing tells you which mode + fuel produced the recommendation.

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
  