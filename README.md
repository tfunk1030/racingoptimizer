# racingoptimizer

`optimize` — a physics-based setup recommender for iRacing GTP cars.

```bash
uv venv
uv pip install -e ".[dev]"
uv run optimize learn ./ibtfiles
uv run optimize bmw sebring          # or: uv run optimize ./my_session.ibt
```

## Status

All six VISION slices and three cross-cutting modules are merged. Each
slice has per-car test coverage (BMW M Hybrid V8, Porsche 963, Cadillac
V-Series.R, Acura ARX-06, Ferrari 499P).

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

## Documentation

* **`GETTING_STARTED.md`** — install, the four commands, troubleshooting.
* **`VISION.md`** — design goals and the four "What This Is NOT" rules.
* **`docs/VISION_COMPLIANCE.md`** — per-clause file:line audit.
* **`CLAUDE.md`** — engineering conventions and per-car verification scope.
* **`docs/superpowers/specs/`** — per-slice design specs.

## Tests

```bash
uv run pytest -q                 # full suite (~10 min)
uv run pytest -q -m "not slow"   # fast suite (~2 min)
```

The slow suite ingests real IBTs and exercises every car's full pipeline.
Smoke tests against `ibtfiles/` and `aero-maps/` auto-skip when those
directories are absent.
