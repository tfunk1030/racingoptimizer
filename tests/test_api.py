from __future__ import annotations

from pathlib import Path

import polars as pl

from racingoptimizer.ingest import lap_data, laps, learn, sessions


def test_learn_then_query_then_lap_data(small_ibt: Path, tmp_corpus: Path) -> None:
    ids = learn(small_ibt, corpus_root=tmp_corpus)
    assert len(ids) == 1

    s = sessions(corpus_root=tmp_corpus)
    assert s.height == 1
    assert s["session_id"][0] == ids[0]
    assert s["car"][0]
    assert s["track"][0]

    lap_rows = laps(session_id=ids[0], corpus_root=tmp_corpus, valid_only=True)
    assert lap_rows.height >= 1

    first_lap = int(lap_rows["lap_index"].min())
    df = lap_data(ids[0], lap_index=first_lap, corpus_root=tmp_corpus)
    assert df.height > 100
    assert {"t_s", "lap_dist_pct", "Speed"}.issubset(df.columns)


def test_learn_is_idempotent(small_ibt: Path, tmp_corpus: Path) -> None:
    ids1 = learn(small_ibt, corpus_root=tmp_corpus)
    ids2 = learn(small_ibt, corpus_root=tmp_corpus)
    assert ids1 == ids2
    s = sessions(corpus_root=tmp_corpus)
    assert s.height == 1


def test_learn_handles_a_directory_recursively(tmp_path: Path, small_ibt: Path, tmp_corpus: Path) -> None:
    # Place the small IBT in a nested folder.
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    target = nested / small_ibt.name
    target.write_bytes(small_ibt.read_bytes())
    ids = learn(tmp_path, corpus_root=tmp_corpus)
    assert len(ids) == 1


def test_lap_data_can_project_columns(small_ibt: Path, tmp_corpus: Path) -> None:
    ids = learn(small_ibt, corpus_root=tmp_corpus)
    sid = ids[0]
    lap_rows = laps(session_id=sid, corpus_root=tmp_corpus, valid_only=True)
    first_lap = int(lap_rows["lap_index"].min())
    df = lap_data(sid, lap_index=first_lap, corpus_root=tmp_corpus, channels=["Speed", "Brake"])
    assert set(df.columns) == {"Speed", "Brake"}


def test_sessions_returns_empty_frame_when_corpus_is_new(tmp_corpus: Path) -> None:
    s = sessions(corpus_root=tmp_corpus)
    assert isinstance(s, pl.DataFrame)
    assert s.height == 0
