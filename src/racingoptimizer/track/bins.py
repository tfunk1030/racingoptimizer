"""Track-position binning helpers (slice D-1).

`track_pos_m` indexes every sample by meters along the lap. `bin_index` floors
that into a fixed-width bucket. Both are pure numpy transforms — no I/O, no
clipping. The 5 m default is justified in `docs/superpowers/specs/2026-04-28-
track-model-design.md` §6.
"""
from __future__ import annotations

import numpy as np

DEFAULT_BIN_SIZE_M: float = 5.0


def track_pos_m_from_pct(lap_dist_pct: np.ndarray, lap_length_m: float) -> np.ndarray:
    return lap_dist_pct.astype(np.float64) * lap_length_m


def bin_index(track_pos_m: np.ndarray, *, bin_size_m: float = DEFAULT_BIN_SIZE_M) -> np.ndarray:
    return np.floor(track_pos_m / bin_size_m).astype(np.int32)
