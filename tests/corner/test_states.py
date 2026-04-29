"""Integration tests for `corner_phase_states` against a real BMW Sebring lap."""
from __future__ import annotations

from pathlib import Path

import pytest

from racingoptimizer.corner import corner_phase_states
from racingoptimizer.ingest.api import laps, learn


def _best_valid_lap(sid: str, corpus_root: Path) -> int:
    rows = laps(session_id=sid, valid_only=True, corpus_root=corpus_root)
    if rows.height == 0:
        pytest.skip("fixture has no valid laps")
    return int(rows["lap_index"][0])


def test_lap_index_minus_one_raises_value_error() -> None:
    with pytest.raises(ValueError, match="pre-grid sentinel"):
        corner_phase_states("00000000deadbeef", -1)


def test_corner_phase_states_against_real_bmw_lap(
    small_ibt: Path, tmp_corpus: Path
) -> None:
    sids = learn(small_ibt, corpus_root=tmp_corpus)
    sid = sids[0]
    lap_idx = _best_valid_lap(sid, tmp_corpus)

    out = corner_phase_states(sid, lap_idx, corpus_root=tmp_corpus)

    assert out.height > 0, "expected at least one (corner, phase) row"

    # At least one mid_corner row.
    phases_seen = set(out["phase"].to_list())
    assert "mid_corner" in phases_seen

    # Every row has samples.
    assert (out["n_samples"] > 0).all()

    # Corners contiguous starting at 0.
    distinct_corners = sorted(set(out["corner_id"].to_list()))
    assert distinct_corners == list(range(len(distinct_corners)))
    assert distinct_corners[0] == 0

    # Air density sanity envelope.
    if "air_density_mean" in out.columns:
        ad = out["air_density_mean"]
        assert ad.min() > 1.0
        assert ad.max() < 1.4

    if "track_temp_c_mean" in out.columns:
        tt = out["track_temp_c_mean"]
        assert tt.min() > 0.0
        assert tt.max() < 60.0

    # Circular wind dir mean must land in [0, 360).
    if "wind_dir_deg_mean" in out.columns:
        wd = out["wind_dir_deg_mean"]
        assert wd.min() >= 0.0
        assert wd.max() < 360.0

    # Sort order: rows ordered by (corner_id, phase_order).
    phase_order = {
        "braking": 0,
        "trail_brake": 1,
        "mid_corner": 2,
        "exit": 3,
        "straight": 4,
    }
    keys = list(zip(out["corner_id"].to_list(), out["phase"].to_list(), strict=True))
    rank = [(c, phase_order[p]) for c, p in keys]
    assert rank == sorted(rank)


def test_corner_phase_states_is_deterministic(
    small_ibt: Path, tmp_corpus: Path
) -> None:
    sids = learn(small_ibt, corpus_root=tmp_corpus)
    sid = sids[0]
    lap_idx = _best_valid_lap(sid, tmp_corpus)

    a = corner_phase_states(sid, lap_idx, corpus_root=tmp_corpus)
    b = corner_phase_states(sid, lap_idx, corpus_root=tmp_corpus)
    assert a.equals(b)


def test_corner_phase_states_excludes_corner_id_minus_one(
    small_ibt: Path, tmp_corpus: Path
) -> None:
    sids = learn(small_ibt, corpus_root=tmp_corpus)
    sid = sids[0]
    lap_idx = _best_valid_lap(sid, tmp_corpus)
    out = corner_phase_states(sid, lap_idx, corpus_root=tmp_corpus)
    assert (out["corner_id"] >= 0).all()
