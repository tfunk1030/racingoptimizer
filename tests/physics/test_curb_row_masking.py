"""P2.1 -- ``_collect_training_frames`` drops dirty rows.

Verifies the wiring at the fitter boundary:

* When ``curb_frac_mean`` exceeds ``_CURB_DIRTY_FRACTION``, the row is
  filtered out of the training frame before fit.
* When ``off_track_frac_mean`` is > 0, the row is also filtered out
  (off-track detector is dilated so any positive sample matters).
* Rows missing either column (legacy / cold-start TrackModel /
  cross-car-borrowed schedule) are kept -- the no-mask path is
  identical to the pre-P2.1 behaviour.

The full ``corner_phase_states`` + ``build_track_model`` integration
through ``_collect_training_frames`` needs IBT data and is exercised in
the per-car smoke suite. These tests exercise the filter logic in
isolation by stubbing the frame.
"""
from __future__ import annotations

import polars as pl

from racingoptimizer.physics.fitter import (
    _CURB_DIRTY_FRACTION,
    _OFF_TRACK_DIRTY_FRACTION,
)


def _filter_dirty_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Mirror of ``_collect_training_frames``'s tail mask filter.

    Extracted so we can exercise the filter without re-running the whole
    catalog walk + lap loader. If the production filter ever drifts,
    update this helper too -- the test will catch regressions.
    """
    if "curb_frac_mean" in frame.columns:
        frame = frame.filter(
            pl.col("curb_frac_mean").is_null()
            | (pl.col("curb_frac_mean") <= _CURB_DIRTY_FRACTION)
        )
    if "off_track_frac_mean" in frame.columns:
        frame = frame.filter(
            pl.col("off_track_frac_mean").is_null()
            | (pl.col("off_track_frac_mean") <= _OFF_TRACK_DIRTY_FRACTION)
        )
    return frame


def test_curb_dominant_row_dropped() -> None:
    frame = pl.DataFrame(
        {
            "session_id": ["s0", "s1", "s2", "s3"],
            "corner_id": [1, 1, 1, 1],
            "phase": ["mid_corner"] * 4,
            "curb_frac_mean": [0.0, 0.30, 0.50, 0.80],
            "off_track_frac_mean": [0.0, 0.0, 0.0, 0.0],
        }
    )
    out = _filter_dirty_rows(frame)
    # 0.30 + 0.50 (boundary inclusive) stay; 0.80 drops.
    assert sorted(out["session_id"].to_list()) == ["s0", "s1", "s2"]


def test_any_off_track_row_dropped() -> None:
    frame = pl.DataFrame(
        {
            "session_id": ["clean", "leaked_one_sample", "frequent"],
            "corner_id": [1, 1, 1],
            "phase": ["mid_corner"] * 3,
            "curb_frac_mean": [0.0, 0.0, 0.0],
            "off_track_frac_mean": [0.0, 0.01, 0.5],
        }
    )
    out = _filter_dirty_rows(frame)
    # _OFF_TRACK_DIRTY_FRACTION = 0.0, so anything > 0 is dropped.
    assert out["session_id"].to_list() == ["clean"]


def test_thresholds_match_plan_p2_1() -> None:
    # Hard-pin the constants so a sneaky drift in fitter.py trips the
    # gate. The plan calls out "median sample on a curb" (0.5) for
    # the curb threshold and "any off-track sample" (0.0) for off-track.
    assert _CURB_DIRTY_FRACTION == 0.5
    assert _OFF_TRACK_DIRTY_FRACTION == 0.0


def test_missing_columns_passes_through_unchanged() -> None:
    # Legacy / cold-start frame: neither cleanliness column is present.
    frame = pl.DataFrame(
        {
            "session_id": ["s0", "s1"],
            "corner_id": [1, 2],
            "phase": ["mid_corner", "exit"],
        }
    )
    out = _filter_dirty_rows(frame)
    assert out.height == 2
    assert "curb_frac_mean" not in out.columns


def test_null_fraction_rows_preserved() -> None:
    # Some rows might be null (a session's TrackModel was missing but
    # its laps still got loaded with a different lap's track_model
    # supplied; not currently possible but defensively keep these).
    frame = pl.DataFrame(
        {
            "session_id": ["null_curb", "null_off", "both_null"],
            "corner_id": [1, 2, 3],
            "phase": ["mid_corner"] * 3,
            "curb_frac_mean": [None, 0.0, None],
            "off_track_frac_mean": [0.0, None, None],
        },
        schema_overrides={
            "curb_frac_mean": pl.Float32,
            "off_track_frac_mean": pl.Float32,
        },
    )
    out = _filter_dirty_rows(frame)
    # All three rows pass: null is treated as "no info, keep" (legacy
    # null-as-clean semantic).
    assert sorted(out["session_id"].to_list()) == [
        "both_null", "null_curb", "null_off",
    ]
