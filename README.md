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

Default DE uses hybrid physics+surrogate scoring; validate static ride heights
in the iRacing garage after applying `[OPT]` values (see `GETTING_STARTED.md`).

## Documentation

* **`GETTING_STARTED.md`** — install, the four commands, troubleshooting.
* **`VISION.md`** — design goals and the four "What This Is NOT" rules.
* **`docs/VISION_COMPLIANCE.md`** — per-clause file:line audit.
* **`docs/audit_2026-05-23/00_findings_and_fix_plan.md`** — latest full audit + fix status (2026-05-23).
* **`CLAUDE.md`** — engineering conventions and per-car verification scope.
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
