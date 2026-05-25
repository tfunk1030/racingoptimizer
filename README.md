# racingoptimizer

`optimize` — a physics-based setup recommender for iRacing GTP cars.

```bash
uv venv
uv pip install -e ".[dev]"
uv run optimize learn ./ibtfiles
uv run optimize bmw sebring                  # race setup
uv run optimize bmw spa --quali --fuel 8     # quali-stint setup, 8 L pinned
uv run optimize bmw spa --physics            # add a physics-view banner above the briefing
uv run optimize ./my_session.ibt             # auto-detect car/track from IBT path
```

## Status

All six VISION slices and three cross-cutting modules are merged. Each
slice has per-car test coverage (BMW M Hybrid V8, Porsche 963, Cadillac
V-Series.R, Acura ARX-06, Ferrari 499P).

**Per-car v4 path** (track-agnostic pooled model + cross-car schedule fallback):
all five GTP cars — BMW, Cadillac, Ferrari, Acura, Porsche. The legacy
v3 per-(car, track) branch remains in code for rollback only.

| Slice | Module | Per-car verification |
|---|---|---|
| A — IBT ingestion | `racingoptimizer.ingest` | car/track detect ✓ all 5; per-car parser/end-to-end ✓ all 5 (`tests/test_parser_per_car.py`) |
| B — Corner-phase | `racingoptimizer.corner` | ✓ all 5 (`tests/corner/test_per_car_smoke.py`) |
| C — Aero maps    | `racingoptimizer.aero` | ✓ all 5 (`tests/aero/test_loader.py::test_load_real_corpus_per_car`) |
| D — Track model  | `racingoptimizer.track` | ✓ all 5 (`tests/track/test_per_car_real_ibt.py`); per-car curb-agreement threshold (Acura uses heave/roll-shock fallback) |
| E — Physics fitter | `racingoptimizer.physics` | ✓ all 5 (`tests/physics/test_per_car_fit_predict.py`); Acura gracefully degrades on missing per-corner shock channels |
| F — CLI / briefing | `racingoptimizer.cli`, `racingoptimizer.explain` | ✓ all 5 (`tests/cli/test_per_car_smoke.py`) |

Cross-cutting modules: `racingoptimizer.context.EnvironmentFrame` (12-channel
weather snapshot), `racingoptimizer.confidence.Confidence` (frozen
`(value, lo, hi, n_samples, regime)`), and
`racingoptimizer.constraints.{ConstraintsTable, load_constraints, clamp}`
(per-car-shadowing parameter bounds parser).

Default DE uses hybrid physics+surrogate scoring (40% physics weight in
mid-corner, 5–10% elsewhere). The surrogate predicts ~30 telemetry channels
from setup + weather; physics adds axle/aero guardrails. Lap time is never the
objective — validate static ride heights in the iRacing garage after applying
`[OPT]` values (see `GETTING_STARTED.md`).

## Current accuracy posture

**This is a guardrailed surrogate optimizer with physics checks +
deterministic static-RH readouts.** The held-out gate at
`docs/physics-rebuild/holdout_accuracy_latest.json` shows every
telemetry channel in `noisy` regime across all 5 cars; `accel_lat_g_max`,
`understeer_angle_mean_rad`, and `wheel_speed_max_diff_ms` still need
work. The 2026-05-24 accuracy-rebuild pass closed the structural bugs
that were hiding any signal underneath; the work to actually move
those channels into the dense regime is the next pass.

**W1-W4 of the accuracy-rebuild plan landed 2026-05-24:**

- P0.1 — broken `per_track_residuals` retired; setup gradient restored.
  `FITTERS_LAYOUT_VERSION = 10` invalidates pre-2026-05-24 pickles.
- P0.2 — deterministic kinematic static-RH fit per car (R² ≥ 0.98);
  legacy k-NN repair bypassed when the kinematic fit ships.
- P0.3 — sensitivity floor enforced; moves below ±0.005 score delta
  are reverted to baseline and surfaced under NOTES instead.
- P0.4 — phantom corner-0 filter + per-corner dedupe.
- P1.1 — per-channel held-out pass criteria; gate hard-fails on ANY
  car failing ANY non-driver-input channel.
- P1.4 — pickle slot type-safety on revive.
- P2.3 — inverse-track-sample-count training weights so a Sebring-heavy
  corpus doesn't drown Spa rows.
- P2.4 — `phase_duration_s` injected into the joint feature matrix at
  fit time. `ENV_FEATURE_SCHEMA_VERSION_PER_CAR` 5 → 6.
- P3.1 — briefing header surfaces per-channel held-out error budget
  (peak lateral G, understeer angle, static front RH, damper force p99).
- P3.2 — Watch-most picker normalised so a long broad-impact corner
  doesn't dominate every parameter's "Watch most" line.
- P3.3 — thin-corpus refusal banner; recommend refuses race setup when
  n_prod < 20 OR `axle_grip_ceilings is None` and points the user at
  `optimize calibrate`.

**Partial / deferred:** P1.2 (lap-time Spearman gate -- helpers ship,
LOSO orchestration is a placeholder pending offline run), P1.3 (hybrid
vs surrogate-only A/B -- non-regression test gates; CI YAML flip
pending), P2.1 (curb masking -- needs `TrackModel.bump_map` API
addition, deferred to next pass), P2.2 (per-track mixed-effects --
skipped in favour of P2.3).

**Full plan, file:line evidence, and ordered roadmap:**
`docs/accuracy-rebuild-2026-05-24/PLAN.md`.

## Documentation

* **`GETTING_STARTED.md`** — install, commands, how recommendations are built,
  confidence, validation workflow, troubleshooting.
* **`VISION.md`** — design goals and the four "What This Is NOT" rules.
* **`docs/VISION_COMPLIANCE.md`** — per-clause file:line audit.
* **`docs/accuracy-rebuild-2026-05-24/PLAN.md`** — current accuracy plan (supersedes 2026-05-23 audit for post-audit working-tree items).
* **`docs/audit_2026-05-23/00_findings_and_fix_plan.md`** — prior full audit (2026-05-23).
* **`CLAUDE.md`** — engineering conventions, per-car verification scope, and known accuracy gap.
* **`docs/superpowers/specs/`** — per-slice design specs.
* **`docs/physics-rebuild/`** — 14-day physics-rebuild plan, daily snapshots, completion summary (2026-05-08).

## CI

GitHub Actions (`.github/workflows/ci.yml`): ruff + fast pytest + held-out
integrity check on push/PR; weekly Day 12b calibration gate when a corpus
is present locally.

## Tests

```bash
uv run pytest -q                 # full suite (~10 min)
uv run pytest -q -m "not slow"   # fast suite (~2 min)
```

The slow suite ingests real IBTs and exercises every car's full pipeline.
Smoke tests against `ibtfiles/` and `aero-maps/` auto-skip when those
directories are absent.
