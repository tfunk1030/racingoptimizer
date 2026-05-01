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
uv run optimize cadillac lagunaseca         # recommend a setup
uv run optimize ./my_session.ibt            # OR drop in any .ibt — auto-detects
```

The output has two parts:

1. **Engineering briefing** — every parameter the optimizer changed, with the corners it helps, the corners it hurts, and ±1-click sensitivity.
2. **Full setup card** — every garage parameter for the car, ready to enter, tagged:
   * `[OPT]` — value the optimizer chose.
   * `[OPT pin]` — pinned to your observed median (the model has no signal to deviate, e.g. you ran the same value every session).
   * `[past]` — copied from your most recent session (no constraint bounds yet).
   * `[readout]` — calculated by iRacing (you don't enter this; verify it after entering the inputs).

## Other commands

```bash
uv run optimize compare a.ibt b.ibt         # diff two setups corner-phase by corner-phase
uv run optimize status cadillac             # what does the model know about this car?
```

## Useful options on `optimize <car> <track>`

* `--json` — machine-readable JSON instead of text.
* `--wing 17` — pin the rear wing angle.
* `--wetness 0.0` — override conditions (also `--air-temp`, `--track-temp`, `--wind`).
* `--no-cache` — refit from scratch (use after changing `constraints.md`).
* `--corpus-root ./somewhere` — non-default corpus location.

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
