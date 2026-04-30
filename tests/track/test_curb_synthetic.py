"""aggregate_curb_likelihood — synthetic per-session p99 series (U7).

Also covers the m/s -> mm/s conversion in
`compute_session_shock_v_p99_per_bin` so the bug where iRacing's m/s
shock-vel channels were compared to mm/s thresholds can't silently
regress (slice D, fix S1.1).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from racingoptimizer.track import (
    BUMP_RANGE_MAX_MM_S,
    BUMP_RANGE_MIN_MM_S,
    aggregate_curb_likelihood,
)
from racingoptimizer.track.masks import (
    T_CURB_AGGREGATE_MM_S,
    T_CURB_SESSION_MM_S,
    compute_session_shock_v_p99_per_bin,
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


def test_compute_session_p99_converts_m_s_to_mm_s_and_trips_curb_threshold():
    """Real iRacing data feeds *shockVel in m/s. After the conversion in
    compute_session_shock_v_p99_per_bin, a 0.6 m/s impulse (= 600 mm/s)
    must land above the 350 mm/s session-curb threshold and a 0.05 m/s
    background (= 50 mm/s) must stay well below.

    Regression guard for the silent-no-op bug where shock channels were
    compared mm/s thresholds without unit conversion.
    """
    # 70 samples evenly walking lap_dist_pct over a 350 m synthetic lap;
    # bin_size_m=5 means each sample lands in its own 5 m bin (35 unique
    # bins by floor + ceil). One sample lives in bin 30 with a curb-strength
    # m/s impulse on the LF corner; everything else is quiet at 0.05 m/s.
    n = 70
    lap_length_m = 350.0
    bin_size_m = 5.0
    lap_dist_pct = np.linspace(0.0, 1.0 - 1e-9, n)
    track_pos_m = lap_dist_pct * lap_length_m
    bin_idx = np.floor(track_pos_m / bin_size_m).astype(np.int32)

    quiet_m_s = 0.05  # = 50 mm/s background, well below 350 mm/s threshold
    curb_m_s = 0.6    # = 600 mm/s impulse, above 400 mm/s aggregate threshold
    lf = np.where(bin_idx == 30, curb_m_s, quiet_m_s).astype(np.float64)

    lap_df = pl.DataFrame(
        {
            "lap_index": np.zeros(n, dtype=np.int32),
            "lap_dist_pct": lap_dist_pct.astype(np.float64),
            "data_quality_mask": np.ones(n, dtype=bool),
            "LFshockVel": lf,
            "RFshockVel": np.full(n, quiet_m_s, dtype=np.float64),
            "LRshockVel": np.full(n, quiet_m_s, dtype=np.float64),
            "RRshockVel": np.full(n, quiet_m_s, dtype=np.float64),
        }
    )

    p99 = compute_session_shock_v_p99_per_bin(
        lap_df, lap_length_m=lap_length_m, bin_size_m=bin_size_m
    )

    assert p99.height > 0
    bin30 = p99.filter(pl.col("bin_index") == 30).row(0, named=True)
    # 0.6 m/s -> 600 mm/s after conversion. Anything <= 0.6 (raw m/s)
    # would mean the conversion was skipped — the silent-no-op regression.
    assert bin30["shock_v_p99_mm_s"] > T_CURB_SESSION_MM_S, (
        f"expected curb-bin p99 > {T_CURB_SESSION_MM_S} mm/s, "
        f"got {bin30['shock_v_p99_mm_s']:.3f}"
    )

    quiet_rows = p99.filter(pl.col("bin_index") != 30)
    assert quiet_rows["shock_v_p99_mm_s"].max() < T_CURB_SESSION_MM_S

    # And the aggregator agrees: 3 sessions of the same data should flag
    # bin 30 as a persistent curb.
    out = aggregate_curb_likelihood([p99, p99, p99])
    curb = out.filter(pl.col("bin_index") == 30).row(0, named=True)
    assert curb["curb_likelihood"] >= 0.6
    assert curb["shock_v_p99_mm_s"] >= T_CURB_AGGREGATE_MM_S
    assert curb["bump_likelihood"] == 0.0


def test_compute_session_p99_no_conversion_would_miss_curbs():
    """Sanity / inverse: feed the same m/s data straight through the
    aggregator (skipping the read-site conversion) and confirm no curb
    fires. Documents the failure mode the conversion guards against.
    """
    p99_no_conversion = pl.DataFrame(
        {
            "bin_index": np.array([30], dtype=np.int32),
            "shock_v_p99_mm_s": np.array([0.6], dtype=np.float64),  # raw m/s, NOT converted
            "n_samples": np.array([10], dtype=np.uint32),
        },
        schema={
            "bin_index": pl.Int32,
            "shock_v_p99_mm_s": pl.Float64,
            "n_samples": pl.UInt32,
        },
    )
    out = aggregate_curb_likelihood([p99_no_conversion] * 3)
    row = out.filter(pl.col("bin_index") == 30).row(0, named=True)
    assert row["curb_likelihood"] == 0.0  # 0.6 << 350 mm/s — no curb flagged
