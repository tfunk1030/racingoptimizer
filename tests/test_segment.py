from __future__ import annotations

import numpy as np

from racingoptimizer.ingest.segment import LapSpan, detect_lap_boundaries


def _synth(n_laps: int, samples_per_lap: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Build synthetic LapDistPct (sawtooth 0->1) and Lap (step) arrays."""
    pct = np.tile(np.linspace(0.0, 1.0, samples_per_lap, endpoint=False), n_laps)
    lap = np.repeat(np.arange(n_laps), samples_per_lap)
    return pct.astype(np.float32), lap.astype(np.int32)


def test_two_clean_laps_yield_two_valid_spans() -> None:
    pct, lap = _synth(n_laps=2, samples_per_lap=120)
    spans = detect_lap_boundaries(pct, lap)
    assert len(spans) == 2
    assert all(isinstance(s, LapSpan) for s in spans)
    assert all(s.valid == 1 for s in spans)
    assert spans[0].lap_index == 0
    assert spans[1].lap_index == 1
    assert spans[0].end_sample == spans[1].start_sample


def test_pre_grid_warmup_gets_lap_index_minus_one() -> None:
    # Insert 30 samples of "before first rollover" at the start.
    pct_clean, lap_clean = _synth(n_laps=2, samples_per_lap=100)
    pct = np.concatenate([np.linspace(0.4, 0.95, 30, dtype=np.float32), pct_clean])
    lap = np.concatenate([np.full(30, -1, dtype=np.int32), lap_clean])
    spans = detect_lap_boundaries(pct, lap)
    # First span must be the pre-grid warmup with lap_index = -1 and valid = 0.
    assert spans[0].lap_index == -1
    assert spans[0].valid == 0
    # Then two clean laps follow.
    assert [s.lap_index for s in spans[1:]] == [0, 1]
    assert all(s.valid == 1 for s in spans[1:])


def test_incomplete_trailing_lap_is_invalid() -> None:
    pct, lap = _synth(n_laps=1, samples_per_lap=100)
    pct = np.concatenate([pct, np.linspace(0.0, 0.5, 50, dtype=np.float32)])
    lap = np.concatenate([lap, np.full(50, 1, dtype=np.int32)])
    spans = detect_lap_boundaries(pct, lap)
    assert spans[0].valid == 1
    assert spans[-1].valid == 0   # trailing partial lap


def test_non_monotonic_lap_channel_marks_lap_invalid() -> None:
    pct, lap = _synth(n_laps=2, samples_per_lap=100)
    # Corrupt the Lap channel so it does not increment by exactly 1 across the boundary.
    lap[100:] = 5
    spans = detect_lap_boundaries(pct, lap)
    # Boundaries are still detected from LapDistPct, but the laps are flagged invalid.
    assert any(s.valid == 0 for s in spans)
