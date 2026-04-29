# racingoptimizer

Physics-based setup optimizer for iRacing GTP cars. See `VISION.md` for the full spec.

## Status

Five of the six VISION slices are merged; the optimizer's recommendation step and the user-facing CLI are still in flight.

| Slice | Module | Status |
|---|---|---|
| A — IBT ingestion | `racingoptimizer.ingest` | ✅ |
| B — Corner-phase decomposition | `racingoptimizer.corner` | ✅ |
| C — Aero-map loader & interpolator | `racingoptimizer.aero` | ✅ |
| D — Track model (curbs, off-track, quality mask) | `racingoptimizer.track` | ✅ |
| E — Physics fitter (training half) | `racingoptimizer.physics` | ✅ training; 🟡 score/recommend in flight |
| F — Recommendation CLI + briefing renderer | `racingoptimizer.cli`, `racingoptimizer.explain` | ⏳ pending |

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
