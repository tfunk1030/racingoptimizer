"""Cold-start regime: < 3 sessions → empty maps + no-op cache (U6)."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from racingoptimizer.track import (
    build_track_model,
    latest_pointer_path,
)


def test_single_session_yields_cold_start(tmp_corpus: Path):
    model = build_track_model(
        "fake_track",
        ["00000000deadbeef"],
        corpus_root=tmp_corpus,
    )

    assert model.regime == "cold_start"
    assert model.bump_map.height == 0
    assert model.grip_map.height == 0
    assert model.session_ids == ("00000000deadbeef",)

    assert model.cache_path.exists()
    assert model.summary_path.exists()
    assert latest_pointer_path(tmp_corpus, "fake_track").exists()


def test_cold_start_curb_mask_raises(tmp_corpus: Path):
    model = build_track_model(
        "fake_track",
        ["00000000deadbeef"],
        corpus_root=tmp_corpus,
    )
    with pytest.raises(NotImplementedError):
        model.curb_mask(pl.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(NotImplementedError):
        model.off_track_mask(pl.DataFrame({"x": [1, 2, 3]}))


def test_cache_hit_replays_without_session_data(tmp_corpus: Path):
    sids = ["00000000deadbeef"]
    first = build_track_model("fake_track", sids, corpus_root=tmp_corpus)
    second = build_track_model("fake_track", sids, corpus_root=tmp_corpus)

    assert first.cache_path == second.cache_path
    assert first.summary_path == second.summary_path
    assert first.regime == second.regime == "cold_start"
    assert first.bump_map.equals(second.bump_map)
    assert first.grip_map.equals(second.grip_map)
