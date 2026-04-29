# racingoptimizer

Physics-based setup optimizer for iRacing GTP cars. See `VISION.md` for the full spec.

## Status

Slice A (IBT ingestion) implemented. Other subsystems (corner-phase decomposition, aero-map loader, fitter, optimizer, track model) are planned and unimplemented.

## Quickstart

```bash
uv venv
uv pip install -e ".[dev]"
uv run optimize learn ./ibtfiles
uv run python -c "from racingoptimizer.ingest import sessions; print(sessions())"
```

The default corpus location is `./corpus/` next to this README. Override with the env var `RACINGOPTIMIZER_CORPUS` or the `--corpus-root` flag.
