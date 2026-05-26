"""P2.1 -- curb / off-track masking threaded through corner_phase_states.

These tests exercise the wiring at the corner-phase aggregator level
(``_attach_cleanliness_masks`` + the new ``curb_frac_mean`` /
``off_track_frac_mean`` columns in :func:`_aggregate`). The full
fitter integration (``_collect_training_frames`` building a real
``TrackModel`` from a populated catalog, masking dirty rows before fit)
needs IBT data and is exercised in slow integration tests.

The plan's primary signal is "median per-sample track position falls
inside a curb bin" -- equivalent to ``curb_frac_mean > 0.5``. We don't
hard-test that threshold here because it lives in the fitter; the
per-phase column emission contract IS testable in isolation.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from racingoptimizer.corner.states import _attach_cleanliness_masks


class _StubTrackModel:
    """Minimal stand-in for TrackModel that exposes the two mask methods.

    Lets us drive the aggregator without building a full TrackModel from
    parquet caches.
    """

    def __init__(
        self,
        curb: np.ndarray | None = None,
        off_track: np.ndarray | None = None,
        raise_on_curb: bool = False,
        raise_on_off: bool = False,
    ) -> None:
        self._curb = curb
        self._off = off_track
        self._raise_curb = raise_on_curb
        self._raise_off = raise_on_off

    def curb_mask(self, lap_df: pl.DataFrame) -> np.ndarray:
        if self._raise_curb:
            raise RuntimeError("stub curb_mask")
        if self._curb is None:
            return np.zeros(lap_df.height, dtype=bool)
        return self._curb

    def off_track_mask(self, lap_df: pl.DataFrame) -> np.ndarray:
        if self._raise_off:
            raise RuntimeError("stub off_track_mask")
        if self._off is None:
            return np.zeros(lap_df.height, dtype=bool)
        return self._off


def _minimal_lap_df(n: int = 8) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_s": list(range(n)),
            "lap_dist_pct": [i / max(1, n - 1) for i in range(n)],
            "Speed": [50.0] * n,
        }
    )


def test_attach_masks_emits_both_columns_with_correct_dtypes() -> None:
    df = _minimal_lap_df(n=10)
    tm = _StubTrackModel(
        curb=np.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 1], dtype=bool),
        off_track=np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0], dtype=bool),
    )
    out = _attach_cleanliness_masks(df, tm)
    assert "_is_curb" in out.columns
    assert "_is_off_track" in out.columns
    assert out.schema["_is_curb"] == pl.Boolean
    assert out.schema["_is_off_track"] == pl.Boolean
    assert out["_is_curb"].to_list() == [
        True, False, False, True, True, False, False, False, False, True,
    ]
    assert out["_is_off_track"].to_list() == [
        False, False, False, False, False, False, True, True, False, False,
    ]


def test_attach_masks_handles_track_model_exception() -> None:
    df = _minimal_lap_df(n=5)
    tm = _StubTrackModel(raise_on_curb=True, raise_on_off=True)
    out = _attach_cleanliness_masks(df, tm)
    # Both columns must still materialise (default False on exception)
    # so downstream `_aggregate` doesn't crash on missing columns when
    # the masks failed for one but not the other lap.
    assert "_is_curb" in out.columns
    assert "_is_off_track" in out.columns
    assert out["_is_curb"].to_list() == [False] * 5
    assert out["_is_off_track"].to_list() == [False] * 5


def test_attach_masks_pads_short_mask_with_false() -> None:
    df = _minimal_lap_df(n=8)
    # Short mask (only 4 samples but lap has 8); _normalize must pad
    # the trailing samples with False so the column length matches.
    tm = _StubTrackModel(curb=np.array([1, 1, 0, 0], dtype=bool))
    out = _attach_cleanliness_masks(df, tm)
    assert len(out["_is_curb"].to_list()) == 8
    assert out["_is_curb"].to_list() == [
        True, True, False, False, False, False, False, False,
    ]


def test_attach_masks_truncates_long_mask() -> None:
    df = _minimal_lap_df(n=4)
    # Long mask: 6 samples for a 4-sample lap.
    tm = _StubTrackModel(curb=np.array([1, 0, 1, 0, 1, 1], dtype=bool))
    out = _attach_cleanliness_masks(df, tm)
    assert len(out["_is_curb"].to_list()) == 4
    assert out["_is_curb"].to_list() == [True, False, True, False]


def test_aggregate_emits_curb_and_off_track_fractions_when_masks_present() -> None:
    """Integration smoke: hand-craft a labeled frame with the masks +
    corner_id/phase, then call ``_aggregate`` and verify the per-phase
    fractions are mean(boolean) over the in-phase samples.
    """
    from racingoptimizer.corner.states import _aggregate

    # 6 samples: 3 in corner 0 / mid_corner, 3 in corner 1 / mid_corner.
    # Curb fires on samples [0, 2] of corner 0 (frac=2/3) and [3] of
    # corner 1 (frac=1/3). Off-track: only sample 5 (frac=1/3 corner 1).
    frame = pl.DataFrame(
        {
            "t_s": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "AccelLat": [0.5, 0.6, 0.7, -0.5, -0.6, -0.7],
            "AccelLon": [0.0] * 6,
            "Brake": [0.0] * 6,
            "Throttle": [0.5] * 6,
            "SteeringWheelAngle": [0.1, 0.1, 0.1, -0.1, -0.1, -0.1],
            "corner_id": [0, 0, 0, 1, 1, 1],
            "phase": ["mid_corner"] * 6,
            "_is_curb": [True, False, True, False, False, False],
            "_is_off_track": [False, False, False, False, False, True],
        }
    )
    result = _aggregate(frame, session_id="sess1", lap_index=1, car="bmw")
    assert "curb_frac_mean" in result.columns
    assert "off_track_frac_mean" in result.columns

    by_corner = {int(r["corner_id"]): r for r in result.to_dicts()}
    # Corner 0: 2/3 samples on curb, 0/3 off-track.
    assert abs(float(by_corner[0]["curb_frac_mean"]) - 2.0 / 3.0) < 1e-5
    assert float(by_corner[0]["off_track_frac_mean"]) == 0.0
    # Corner 1: 0/3 on curb, 1/3 off-track.
    assert float(by_corner[1]["curb_frac_mean"]) == 0.0
    assert abs(float(by_corner[1]["off_track_frac_mean"]) - 1.0 / 3.0) < 1e-5


def test_aggregate_omits_fraction_columns_when_masks_absent() -> None:
    """Legacy callers (no TrackModel threaded) get the legacy column set
    -- no cleanliness fractions, no errors.
    """
    from racingoptimizer.corner.states import _aggregate

    frame = pl.DataFrame(
        {
            "t_s": [0.0, 0.1, 0.2],
            "AccelLat": [0.5, 0.6, 0.7],
            "AccelLon": [0.0] * 3,
            "Brake": [0.0] * 3,
            "Throttle": [0.5] * 3,
            "SteeringWheelAngle": [0.1, 0.1, 0.1],
            "corner_id": [0, 0, 0],
            "phase": ["mid_corner"] * 3,
        }
    )
    result = _aggregate(frame, session_id="sess1", lap_index=1, car="bmw")
    assert "curb_frac_mean" not in result.columns
    assert "off_track_frac_mean" not in result.columns
