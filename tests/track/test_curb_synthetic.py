"""aggregate_curb_likelihood — synthetic per-session p99 series (U7)."""
from __future__ import annotations

import polars as pl

from racingoptimizer.track import (
    BUMP_RANGE_MAX_MM_S,
    BUMP_RANGE_MIN_MM_S,
    aggregate_curb_likelihood,
)


def _frame(p99_by_bin: dict[int, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "bin_index": list(p99_by_bin.keys()),
            "shock_v_p99_mm_s": list(p99_by_bin.values()),
            "n_samples": [10] * len(p99_by_bin),
        },
        schema={
            "bin_index": pl.Int32,
            "shock_v_p99_mm_s": pl.Float64,
            "n_samples": pl.UInt32,
        },
    )


def test_curb_burst_in_4_of_5_sessions_flagged():
    sessions = []
    for keep_curb in [True, True, True, True, False]:
        bins = {50: 150.0, 200: 150.0}
        bins[100] = 600.0 if keep_curb else 150.0
        sessions.append(_frame(bins))

    out = aggregate_curb_likelihood(sessions)
    row100 = out.filter(pl.col("bin_index") == 100).row(0, named=True)
    assert row100["curb_likelihood"] >= 0.6
    assert row100["shock_v_p99_mm_s"] >= 400.0
    assert row100["bump_likelihood"] == 0.0

    row50 = out.filter(pl.col("bin_index") == 50).row(0, named=True)
    assert row50["curb_likelihood"] == 0.0
    assert row50["bump_likelihood"] == 0.0


def test_bump_only_bin_classified_as_bump():
    sessions = [_frame({200: 250.0}) for _ in range(5)]
    out = aggregate_curb_likelihood(sessions)
    row = out.filter(pl.col("bin_index") == 200).row(0, named=True)
    assert row["curb_likelihood"] == 0.0
    expected = (250.0 - BUMP_RANGE_MIN_MM_S) / (BUMP_RANGE_MAX_MM_S - BUMP_RANGE_MIN_MM_S)
    assert abs(row["bump_likelihood"] - expected) < 1e-9


def test_curb_and_bump_are_mutually_exclusive():
    sessions = [_frame({77: 700.0}) for _ in range(3)]
    out = aggregate_curb_likelihood(sessions)
    row = out.filter(pl.col("bin_index") == 77).row(0, named=True)
    assert row["curb_likelihood"] == 1.0
    assert row["bump_likelihood"] == 0.0


def test_empty_input_list_returns_empty_frame():
    out = aggregate_curb_likelihood([])
    assert out.height == 0
    assert "bin_index" in out.columns
    assert "curb_likelihood" in out.columns


def test_single_session_curb_likelihood_is_binary():
    sessions = [_frame({10: 200.0, 20: 500.0})]
    out = aggregate_curb_likelihood(sessions)
    assert set(out["n_sessions"].to_list()) == {1}
    likelihoods = sorted(out["curb_likelihood"].to_list())
    assert likelihoods == [0.0, 1.0]
