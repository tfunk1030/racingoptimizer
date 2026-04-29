# racingoptimizer

Physics-based setup optimizer for iRacing GTP cars. See `VISION.md` for the full spec.

## Status

Five of the six VISION slices have code merged on master; the recommendation step (slice E second half) and the user-facing CLI (slice F) are still in flight. **Code-merged is not the same as verified-across-all-5-cars** — per-car coverage is tracked separately because the BMW M Hybrid V8, Porsche 963, Cadillac V-Series.R, Acura ARX-06, and Ferrari 499P have different suspension architectures and IBT setup-blob shapes.

| Slice | Module | Code | Verified across all 5 cars? |
|---|---|---|---|
| A — IBT ingestion | `racingoptimizer.ingest` | merged | car/track detect ✓ all 5; parser/writer/api end-to-end smoke = BMW only |
| B — Corner-phase decomposition | `racingoptimizer.corner` | merged | ✓ all 5 (`test_per_car_smoke.py`) |
| C — Aero-map loader & interpolator | `racingoptimizer.aero` | merged | ✓ all 5 |
| D — Track model | `racingoptimizer.track` | merged | synthetic only — no real per-car IBT integration test |
| E — Physics fitter (training) | `racingoptimizer.physics` | merged (U9) | BMW only; Acura known divergence (no shock-deflection channels) |
| E — Physics fitter (score + recommend) | `racingoptimizer.physics` | in flight (U10) | n/a |
| F — Recommendation CLI + briefing renderer | `racingoptimizer.cli`, `racingoptimizer.explain` | not started (U11) | n/a |

Cross-cutting modules already shipped: `racingoptimizer.context.EnvironmentFrame`, `racingoptimizer.confidence.Confidence`, `racingoptimizer.constraints.{ConstraintsTable, clamp}`.

## Quickstart

```bash
uv venv
uv pip install -e ".[dev]"

# 1. Ingest telemetry (slice A)
uv run optimize learn ./ibtfiles

# 2. Inspect what's in the corpus
uv run python -c "from racingoptimizer.ingest import sessions; print(sessions())"

# 3. Decompose a lap into corner-phases (slice B)
uv run python -c "from racingoptimizer.corner import corner_phase_states; print(corner_phase_states('<session_id>', 5))"

# 4. Load an aero surface (slice C)
uv run python -c "from racingoptimizer.aero import load_aero_maps; print(load_aero_maps('bmw').interpolate(45.0, 35.0, 14.0, 1.225))"

# 5. Build a track model (slice D)
uv run python -c "from racingoptimizer.track import build_track_model; print(build_track_model('sebring_international', ['<sid1>','<sid2>','<sid3>']))"

# 6. Fit a physics model (slice E, training half — predictions only, no recommendation yet)
uv run python -c "from racingoptimizer.physics import fit; m = fit('bmw', ['<sid>'], track_model); print(m.untrained_parameters)"

# Until U10 lands, m.score_setup(...) and m.recommend(...) are not implemented.
# Until U11 lands, `optimize bmw sebring --wing 17` is not wired.
```

The default corpus location is `./corpus/` next to this README. Override with the env var `RACINGOPTIMIZER_CORPUS` or the `--corpus-root` flag.

## Tests

```bash
uv run pytest -q
```

Currently ~250 tests across slices A–E. Smoke tests against IBT/aero fixtures auto-skip when `ibtfiles/` or `aero-maps/` are absent.
