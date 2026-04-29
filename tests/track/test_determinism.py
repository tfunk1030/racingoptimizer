"""Same inputs → bytewise-identical persisted parquet (U6 contract)."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from racingoptimizer.track import build_track_model
from racingoptimizer.track.builder import (
    _BUMP_SCHEMA,
    _GRIP_SCHEMA,
    _SUMMARY_SCHEMA,
)


def test_summary_is_bytewise_identical(tmp_corpus: Path, tmp_path: Path):
    sids = ["00000000deadbeef", "11111111cafe1234"]

    first = build_track_model("fake_track", sids, corpus_root=tmp_corpus)
    first_bytes = first.summary_path.read_bytes()
    first_cache_bytes = first.cache_path.read_bytes()

    other_corpus = tmp_path / "corpus2"
    other_corpus.mkdir()

    second = build_track_model("fake_track", sids, corpus_root=other_corpus)
    second_bytes = second.summary_path.read_bytes()
    second_cache_bytes = second.cache_path.read_bytes()

    assert first_bytes == second_bytes
    assert first_cache_bytes == second_cache_bytes


def test_summary_parquet_round_trips_with_zstd(tmp_corpus: Path):
    model = build_track_model(
        "fake_track", ["00000000deadbeef"], corpus_root=tmp_corpus
    )
    df = pl.read_parquet(model.summary_path)
    assert dict(df.schema) == _SUMMARY_SCHEMA
    assert dict(model.bump_map.schema) == _BUMP_SCHEMA
    assert dict(model.grip_map.schema) == _GRIP_SCHEMA


def test_cache_parquet_round_trips_with_zstd(tmp_corpus: Path):
    model = build_track_model(
        "fake_track", ["00000000deadbeef"], corpus_root=tmp_corpus
    )
    df = pl.read_parquet(model.cache_path)
    expected = {
        "session_id": pl.Utf8,
        "track_pos_m": pl.Float64,
        "n_samples": pl.Int64,
        "shock_v_p99_mm_s": pl.Float64,
        "lateral_g_p95": pl.Float64,
        "lateral_g_median": pl.Float64,
    }
    assert dict(df.schema) == expected


def test_sort_invariance_of_session_ids(tmp_corpus: Path, tmp_path: Path):
    a = build_track_model("fake_track", ["b", "a", "c"], corpus_root=tmp_corpus)
    other = tmp_path / "corpus2"
    other.mkdir()
    b = build_track_model("fake_track", ["c", "b", "a"], corpus_root=other)
    assert a.summary_path.name == b.summary_path.name
    assert a.summary_path.read_bytes() == b.summary_path.read_bytes()
