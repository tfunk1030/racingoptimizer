"""TrackModel.expected — per-bin (mean, p99) channel lookup (S4.1).

Synthetic compounding corpus is seeded directly into the per-session
cache parquet so the test does not have to ingest 5 IBT files. The
`build_track_model` cache-hit path then loads the seeded data and the
`expected` lookup walks it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from racingoptimizer.track import (
    Expected,
    build_track_model,
    cache_path,
    expected_from_cache,
    summary_path,
)
from racingoptimizer.track.builder import (
    _PER_SESSION_SCHEMA,
    _SUMMARY_SCHEMA,
)
from racingoptimizer.track.paths import sessions_hash

_BIN_SIZE = 5.0
_LAP_LEN = 350.0


def _seed_cache(
    corpus_root: Path,
    track: str,
    session_ids: list[str],
    *,
    bump_map: dict[int, list[float]],
    grip_map: dict[int, list[tuple[float, float]]] | None = None,
) -> None:
    """Write a synthetic per-session cache parquet matching `_PER_SESSION_SCHEMA`.

    `bump_map[bin_idx]` lists the per-session shock_v_p99_mm_s values for
    that bin (one per session). `grip_map[bin_idx]` lists per-session
    `(lateral_g_p95, lateral_g_median)` pairs.
    """
    digest = sessions_hash(sorted(session_ids))
    cache = cache_path(corpus_root, track, digest)
    summary = summary_path(corpus_root, track, digest)
    cache.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for bin_idx, p99_per_session in bump_map.items():
        assert len(p99_per_session) == len(session_ids)
        for sid, p99 in zip(sorted(session_ids), p99_per_session, strict=True):
            grip_pair = (
                grip_map.get(bin_idx, [(0.0, 0.0)] * len(session_ids))
                if grip_map
                else [(0.0, 0.0)] * len(session_ids)
            )
            session_grip = grip_pair[sorted(session_ids).index(sid)]
            rows.append(
                {
                    "session_id": sid,
                    "track_pos_m": float(bin_idx) * _BIN_SIZE,
                    "n_samples": 100,
                    "shock_v_p99_mm_s": float(p99),
                    "lateral_g_p95": float(session_grip[0]),
                    "lateral_g_median": float(session_grip[1]),
                }
            )
    pl.DataFrame(rows, schema=_PER_SESSION_SCHEMA).write_parquet(cache, compression="zstd")

    # Build a minimal summary so `build_track_model` cache-hit replays it.
    bins = sorted(bump_map.keys())
    summary_df = pl.DataFrame(
        {
            "bin_index": np.array(bins, dtype=np.int32),
            "track_pos_m": np.array([float(b) * _BIN_SIZE for b in bins], dtype=np.float64),
            "shock_v_p99_mm_s": np.array(
                [float(np.mean(bump_map[b])) for b in bins], dtype=np.float64
            ),
            "lateral_g_p95": np.zeros(len(bins), dtype=np.float64),
            "lateral_g_median": np.zeros(len(bins), dtype=np.float64),
            "n_samples": np.full(len(bins), 100, dtype=np.int64),
            "n_sessions": np.full(len(bins), len(session_ids), dtype=np.int64),
            "curb_likelihood": np.zeros(len(bins), dtype=np.float64),
            "bump_likelihood": np.zeros(len(bins), dtype=np.float64),
            "lap_length_m": np.full(len(bins), _LAP_LEN, dtype=np.float64),
        },
        schema=_SUMMARY_SCHEMA,
    )
    summary_df.write_parquet(summary, compression="zstd")


def test_expected_returns_mean_and_p99_consistent_with_bump_map(tmp_corpus: Path):
    sids = ["s0", "s1", "s2", "s3", "s4"]
    track = "fake_track"
    # Bin 30 has a clean 5-session distribution: [400, 410, 420, 430, 440] mm/s.
    _seed_cache(
        tmp_corpus,
        track,
        sids,
        bump_map={30: [400.0, 410.0, 420.0, 430.0, 440.0]},
    )

    model = build_track_model(track, sids, corpus_root=tmp_corpus)
    exp = model.expected(track_pos_m=30 * _BIN_SIZE, channel="shock_v_p99_mm_s")

    assert isinstance(exp, Expected)
    assert exp.n_sessions == 5
    assert abs(exp.mean - 420.0) < 1e-6
    # numpy 0.99-quantile on [400..440] (linear) ≈ 438.4
    assert 437.0 <= exp.p99 <= 440.0


def test_expected_cold_start_returns_none(tmp_corpus: Path):
    # 1 session → cold-start, builder writes an empty per-session frame.
    model = build_track_model("fake_track", ["s0"], corpus_root=tmp_corpus)
    assert model.regime == "cold_start"
    assert model.expected(track_pos_m=150.0, channel="shock_v_p99_mm_s") is None


def test_expected_unknown_channel_returns_none(tmp_corpus: Path):
    sids = ["s0", "s1", "s2"]
    track = "fake_track"
    _seed_cache(tmp_corpus, track, sids, bump_map={30: [400.0, 410.0, 420.0]})

    model = build_track_model(track, sids, corpus_root=tmp_corpus)
    assert model.expected(track_pos_m=30 * _BIN_SIZE, channel="not_a_channel") is None


def test_expected_bin_with_too_few_sessions_returns_none(tmp_corpus: Path):
    """A bin only seen in 2 of N sessions falls below the < 3 threshold."""
    sids = ["s0", "s1", "s2"]
    track = "fake_track"
    digest = sessions_hash(sorted(sids))
    cache = cache_path(tmp_corpus, track, digest)
    summary = summary_path(tmp_corpus, track, digest)
    cache.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "session_id": sid,
            "track_pos_m": 50.0,
            "n_samples": 100,
            "shock_v_p99_mm_s": 400.0,
            "lateral_g_p95": 0.0,
            "lateral_g_median": 0.0,
        }
        # Only 2 of 3 sessions cover bin 10 (track_pos_m = 50).
        for sid in sorted(sids)[:2]
    ]
    pl.DataFrame(rows, schema=_PER_SESSION_SCHEMA).write_parquet(cache, compression="zstd")
    pl.DataFrame(
        {
            "bin_index": np.array([10], dtype=np.int32),
            "track_pos_m": np.array([50.0], dtype=np.float64),
            "shock_v_p99_mm_s": np.array([400.0], dtype=np.float64),
            "lateral_g_p95": np.array([0.0], dtype=np.float64),
            "lateral_g_median": np.array([0.0], dtype=np.float64),
            "n_samples": np.array([200], dtype=np.int64),
            "n_sessions": np.array([2], dtype=np.int64),
            "curb_likelihood": np.array([0.0], dtype=np.float64),
            "bump_likelihood": np.array([0.0], dtype=np.float64),
            "lap_length_m": np.array([_LAP_LEN], dtype=np.float64),
        },
        schema=_SUMMARY_SCHEMA,
    ).write_parquet(summary, compression="zstd")

    model = build_track_model(track, sids, corpus_root=tmp_corpus)
    assert model.expected(track_pos_m=50.0, channel="shock_v_p99_mm_s") is None


def test_expected_from_cache_pure_function():
    """Direct call against an in-memory frame (no disk)."""
    cache_df = pl.DataFrame(
        [
            {
                "session_id": f"s{i}",
                "track_pos_m": 150.0,
                "n_samples": 100,
                "shock_v_p99_mm_s": 100.0 + 10.0 * i,
                "lateral_g_p95": 0.5,
                "lateral_g_median": 0.3,
            }
            for i in range(5)
        ],
        schema=_PER_SESSION_SCHEMA,
    )
    exp = expected_from_cache(
        cache_df,
        track_pos_m=150.0,
        channel="shock_v_p99_mm_s",
        bin_size_m=_BIN_SIZE,
    )
    assert exp is not None
    assert exp.n_sessions == 5
    assert abs(exp.mean - 120.0) < 1e-6  # mean(100, 110, 120, 130, 140)


def test_expected_query_outside_bin_grid_returns_none(tmp_corpus: Path):
    sids = ["s0", "s1", "s2"]
    track = "fake_track"
    _seed_cache(tmp_corpus, track, sids, bump_map={30: [400.0, 410.0, 420.0]})

    model = build_track_model(track, sids, corpus_root=tmp_corpus)
    # Bin 5 (track_pos_m = 25) was never observed.
    assert model.expected(track_pos_m=25.0, channel="shock_v_p99_mm_s") is None
