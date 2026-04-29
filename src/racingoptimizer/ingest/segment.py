"""Lap segmentation from raw IBT channels.

A lap boundary is detected wherever LapDistPct rolls back from near 1.0 to
near 0.0. A lap is `valid` iff it both starts and ends with such a rollover
and the IBT `Lap` channel monotonically increases by exactly 1 across it.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

ROLLOVER_HI = 0.9
ROLLOVER_LO = 0.1


class LapSpan(NamedTuple):
    lap_index: int     # -1 for pre-grid samples before the first rollover
    start_sample: int  # inclusive
    end_sample: int    # exclusive
    valid: int         # 0/1


def _rollover_indices(lap_dist_pct: np.ndarray) -> np.ndarray:
    """Return sample indices where LapDistPct rolls back (i.e. lap-start markers)."""
    prev = lap_dist_pct[:-1]
    curr = lap_dist_pct[1:]
    rollover = (prev > ROLLOVER_HI) & (curr < ROLLOVER_LO)
    # +1 because the rollover *ends* one sample after the spike at index i.
    return np.flatnonzero(rollover) + 1


def detect_lap_boundaries(lap_dist_pct: np.ndarray, lap_channel: np.ndarray) -> list[LapSpan]:
    """Decompose a session's samples into LapSpans.

    Parameters
    ----------
    lap_dist_pct : np.ndarray
        Per-sample fractional position around the lap, in [0, 1].
    lap_channel : np.ndarray
        Per-sample integer lap index as reported by iRacing.

    Returns
    -------
    list[LapSpan]
        Pre-grid samples (if any) appear as the first span with lap_index=-1.
        Each subsequent span covers one lap. The trailing span is marked
        invalid if the session ends mid-lap.
    """
    n = lap_dist_pct.shape[0]
    if n == 0:
        return []
    starts = _rollover_indices(lap_dist_pct).tolist()
    spans: list[LapSpan] = []

    if not starts:
        # No completed lap boundary seen at all.
        return [LapSpan(lap_index=-1, start_sample=0, end_sample=n, valid=0)]

    # If the recording begins near LapDistPct=0, treat sample 0 as an implicit
    # lap-start; otherwise the leading samples are pre-grid warmup.
    if lap_dist_pct[0] < ROLLOVER_LO:
        starts = [0] + starts
    elif starts[0] > 0:
        spans.append(LapSpan(lap_index=-1, start_sample=0, end_sample=starts[0], valid=0))

    # One span per completed boundary, plus the trailing partial.
    boundaries = starts + [n]
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        is_trailing_partial = i == len(boundaries) - 2 and e == n and lap_dist_pct[-1] < ROLLOVER_HI
        valid = 1
        if is_trailing_partial:
            valid = 0
        else:
            # The Lap channel must be constant within the span, and (if not the
            # first lap) increment by exactly 1 from the previous span's start.
            lap_at_start = int(lap_channel[s])
            lap_at_end_inclusive = int(lap_channel[e - 1])
            if lap_at_end_inclusive != lap_at_start:
                valid = 0
            elif i > 0:
                prev_lap_at_start = int(lap_channel[boundaries[i - 1]])
                if lap_at_start - prev_lap_at_start != 1:
                    valid = 0
        spans.append(LapSpan(lap_index=i, start_sample=s, end_sample=e, valid=valid))
    return spans
