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
   * `[OPT pin]` — pinned to your observed median (**no variance in your corpus**; different from untrained parameters that lack bounds).
   * `[OPT mirror]` — value mirrored from the per-axle parameter on the opposite corner. iRacing UI requires LR=RR for: rear coil spring rate, rear spring perch offset, front + rear torsion bar turns, front + rear torsion bar OD, rear toe-in, and all right-side damper clicks (5 modes × 2 axles).
   * `[past]` — copied from your most recent session (no constraint bounds yet).
   * `[readout]` — calculated by iRacing using your last session (you don't enter this). Includes **corner weights** and static ride heights before you apply `[OPT]`.
   * `[predicted]` — calculated readout the optimizer projects under the new setup (e.g. predicted static ride heights at the new perch / pushrod / spring values; this is what iRacing will display *after* you enter the `[OPT]` numbers).

## Other commands

```bash
uv run optimize compare a.ibt b.ibt         # diff two setups corner-phase by corner-phase
uv run optimize status cadillac           # coverage + fit-quality trend per track
uv run optimize calibrate bmw spa           # propose probes that grow corpus coverage
uv run optimize calibrate bmw spa --status  # just print the per-parameter coverage table
```

`optimize status` shows session/lap counts per track and **fit quality** trend
after each refit. Fit quality = signal / (signal + cross-validation residual);
values near **0.50** on driver-input channels (brake, throttle, steering) are
expected — the model cannot fully resolve channels dominated by how you drive,
not what you set in the garage.

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

## How recommendations are built

Every recommendation is a search over your garage parameters, not a lap-time
optimizer. The pipeline has three layers:

1. **Telemetry model (surrogate)** — your past IBTs are segmented into
   corner-phases (braking, trail-brake, mid-corner, exit, straight). For each
   phase the model learns how setup + weather + corner shape predict ~30
   telemetry channels: lateral G, understeer angle, ride heights, shock
   deflection, damper velocity/force, brake/throttle traces, and static /
   at-speed ride-height readouts. This is the primary scoring engine.

2. **Physics checks (hybrid, default on)** — a second score blends in where
   setup physics is strongest: **mid-corner** gets 40% physics weight; braking,
   exit, and straight get 5–10%. Physics checks axle grip margin, aero balance
   from the aero maps, and grip headroom. Guardrails penalize exceeding an axle
   ceiling, severe aero imbalance, or physics-vs-surrogate divergence. Pass
   `--surrogate-only` to disable the blend.

3. **Search (differential evolution)** — the optimizer tries many setup
   combinations within your **corpus envelope** (values you've actually run
   for this car, across all tracks). It maximizes the sum of per-corner-phase
   scores weighted by corner duration. Lap time is never fed directly into the
   objective.

**What this means in practice:** the surrogate predicts telemetry channels well
on held-out laps (confidence intervals generally cover 85–95% of actuals). The
physics layer is better at catching bad setups than predicting which one is
fastest — evaluator correlation with corner duration on this corpus averages
~0.19. Treat `[OPT]` values as informed starting points, not guaranteed
fastest setups.

### Known accuracy gap (2026-05-24)

W1 through W4 of `docs/accuracy-rebuild-2026-05-24/PLAN.md` landed on
2026-05-24:

- `per_track_residuals` retired; setup gradient restored.
  `+1 click` / `-1 click` sensitivities are meaningful again (you'll
  see real ±0.001–±0.003 numbers on parameters with signal, instead of
  universal ±0.000).
- Moves below the ±0.005 sensitivity floor are held at the training
  baseline and surfaced under NOTES as `held at past value (model
  cannot resolve +/-1 click on this corpus -- below sensitivity floor)`.
- The previous five-line `T0 ...` guardrail spam in NOTES is one
  summary line per affected corner.
- Static garage RH is now a **deterministic per-car kinematic fit**
  (`physics/static_rh_kinematic.py`, R² ≥ 0.98). The legacy k-NN
  repair only runs when the kinematic fit refused to ship for that
  car. The `[predicted]` static RH on the setup card should be
  trustworthy on cars with the kinematic fit; still worth a sanity
  check against the iRacing garage's `Ride Height` readout before
  applying.
- The briefing header now shows the **per-channel held-out error
  budget** (peak lateral G, understeer angle, static front RH, damper
  force p99) in place of the old `Confidence: <regime>` rollup.
  Falls back to the legacy line on tracks the held-out gate hasn't
  scored.
- The briefing **refuses to recommend** when the per-car corpus is
  thin (< 20 production sessions OR no axle grip ceilings fit). It
  points at `optimize calibrate <car> <track>` instead.
- The held-out gate at `scripts/holdout_accuracy_gate.py` now hard-fails
  on **per-channel** thresholds (peak lat/long G ≤ 0.30 / 0.5; understeer
  ≤ 0.10 rad / 0.5; wheel RH ≤ 3.0 mm / 0.5; static RH ≤ 1.0 mm / 0.2;
  damper force p99 ≤ 30 % of channel std / 0.5). Run
  `uv run python scripts/holdout_accuracy_gate.py` to see today's
  baseline; the gate today fails on grip-balance channels for several
  cars -- those failures are the next round of accuracy work.

Still open / partial:

- **Lap-time correlation gate (P1.2).** Helpers + qualifying-pair
  filter ship in `scripts/lap_time_correlation_gate.py`; the per-pair
  LOSO refit is a placeholder pending an offline run.
- **Hybrid vs surrogate-only A/B in CI (P1.3).** Test gates
  non-regression invariants; per-car asymmetric "hybrid doesn't lose"
  assertion + CI YAML flip pending.
- **Curb / off-line row masking (P2.1).** Deferred -- needs
  `TrackModel.bump_map` API addition.
- Held-out accuracy across all 5 cars is still in `noisy` regime on
  grip-balance channels (peak lateral G, understeer angle). P2 raises
  the modeling floor; the closed structural bugs above unblock that
  work being measurable.

Full plan and prioritized fixes: `docs/accuracy-rebuild-2026-05-24/PLAN.md`.

## What the model learns from telemetry

Your IBTs carry 100+ channels at 60 Hz. The optimizer uses them in layers:

| Layer | Channels / inputs | Role |
|---|---|---|
| **Setup vector** | ~47 garage parameters (springs, dampers, wing, diff, toe, fuel, …) | What DE searches |
| **Environment** | Air temp/density/pressure, humidity, wind, track temp/wetness, weather | Condition-adjusted baselines and aero density correction |
| **Corner shape** | Apex speed, peak lat-G, duration, compression demand | Lets one per-car model score any track |
| **Surrogate targets (30)** | Lat-G, understeer, RH, shock defl, damper vel/force, brake/throttle, static + at-speed RH | What the model predicts and scores against |
| **Physics-only** | Raw lat/long accel at 60 Hz | Axle grip ceilings and guardrails (not surrogate targets) |
| **Aero maps** | External JSON per (car, wing) | Balance % and L/D from predicted ride heights |

Channels the model **cannot** learn well from setup alone (driver-input dominated):
brake apply, throttle modulation, steering trace, damper velocity — these
plateau at ~0.50 fit quality regardless of corpus size. Corner weights and
static ride heights in the garage are **calculated readouts** (`[readout]` /
`[predicted]`), not searchable inputs.

Weather and stint mode branch scoring automatically: `--quali` tilts toward
outright pace; wet conditions swap to lower grip/aero baselines and different
phase weights.

## Understanding confidence

Confidence appears in two places, and they measure different things:

| Where | What it reflects |
|---|---|
| **Briefing header** (`dense` / `noisy` / `sparse`) | Track-wide data density — how many sessions and laps you have at this track for this car |
| **Setup card tags** | Per-parameter optimizer behavior |

**Header confidence is track-wide, not parameter-specific.** A "dense" header
does not mean every recommended value has local evidence. A parameter you held
constant at another track can still show `[OPT pin]` here, or recommend a value
you've never tried at this track if it exists elsewhere in your global corpus.

| Tag / notice | Meaning |
|---|---|
| `[OPT]` | Optimizer chose this value within search bounds |
| `[OPT pin]` | Pinned to your observed median — **no variance in your corpus** to learn from (different from untrained) |
| Untrained (NOTES) | **No `constraints.md` bounds yet** — excluded from search entirely |
| `[readout]` / `[predicted]` | iRacing-calculated values; static ride heights are readouts — compare `[predicted]` against what iRacing shows after you enter `[OPT]` |

Use `optimize calibrate <car> <track> --status` to see which parameters lack
within-track variance. Use `optimize status <car>` to see fit-quality trend
over refits (`fit_quality` near 0.50 on driver-input channels like
brake/throttle is a structural ceiling, not a bug).

## Validating recommendations

Before pushing `[OPT]` values on track:

1. **Enter values in the iRacing garage** — compare static ride heights and
   platform readouts against `[predicted]` tags. Even a few mm mismatch on
   static RH means the optimizer's platform model missed; adjust perch /
   pushrod / spring inputs before running.
2. **Check within-track variance** — `optimize calibrate <car> <track> --status`.
   If a moved parameter shows zero coverage at this track, the recommendation
   may be inherited from another track's philosophy (see cross-track note below).
3. **Compare to your fastest IBT** — when your recent corpus drifts conservative
   (soft ARB, rear toe-in, tight diff), the optimizer inherits that bias even if
   you have a faster validated setup in history.
4. **Trust mid-corner moves most** — physics signal is strongest in steady-state
   cornering. Exit/braking recommendations are dominated by driver-input
   telemetry the model cannot fully resolve.

## When to override the optimizer

Three known disconnects between the score function and lap time:

* **Tyre cold pressure** — auto-pinned to 152 kPa floor unless you override
  with `--pin tyre_cold_pressure_kpa=N`. The surrogate rewards platform
  stability from higher cold P but doesn't see peak-grip loss from a smaller
  contact patch. Community GTP wisdom (152 floor) usually wins on lap time.
* **Conservative corpus bias** — the pooled per-car model learns whatever
  philosophy your recent sessions express. Override with `--pin` or `--reset`
  when you know a different direction is faster.
* **Pinned-outside-corpus values** — `--pin param=value` outside your global
  envelope produces extrapolation, not engineered compensation. Header may
  still read "dense".

## Per-car model (all five GTP cars)

All five cars use the **v4 per-car pooled path**: one model trained on every
session of that car across every track, with corner-geometry archetype features
that let the same model score any track. When a car has never been driven on a
target track, the CLI borrows the corner schedule from another car's sessions
on that track (cross-car schedule fallback) — the model stays per-car; only
the geometry is borrowed.

**Cross-track confounding:** because the surrogate pools all tracks, a parameter
held constant at a high-sample track (e.g. wing=17 across 24 Hockenheim sessions)
can drag recommendations at a low-sample track (e.g. wing=14–15 across 6 Spa
sessions) toward the over-sampled track's philosophy. Check within-track
variance before trusting track-specific moves.

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
lap-time-correlated physics predictor.

| Layer | Accuracy on this corpus |
|---|---|
| Surrogate channel prediction (held-out IBTs) | Strong — median CI coverage ~0.85–1.0 on most channels |
| Physics evaluator vs corner duration | Weak — Spearman ~0.19 average (design target was 0.35) |
| Mid-corner physics signal | Best phase — Spearman ~0.23; hybrid weights physics 40% here |
| Driver-input channels | Structural ~0.50 fit-quality ceiling |

Use `[OPT]` values as starting points; validate static ride heights and
platform readouts in the iRacing garage before pushing on track.

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
  