from __future__ import annotations

from pathlib import Path

from racingoptimizer.ingest import lap_data, learn, sessions
from racingoptimizer.ingest.catalog import open_catalog
from racingoptimizer.ingest.paths import catalog_path
from racingoptimizer.ingest.writer import session_id_from_bytes


def test_full_pipeline_against_fixture(small_ibt: Path, tmp_corpus: Path) -> None:
    ids = learn(small_ibt, corpus_root=tmp_corpus)
    assert len(ids) == 1
    s = sessions(corpus_root=tmp_corpus, valid_only=False)
    assert s.height == 1
    assert s["status"][0] == "ok"
    assert s["car"][0] == "bmw"
    assert s["track"][0] == "sebring_international"

    df = lap_data(ids[0], lap_index=int(s["lap_count"][0]) - 1, corpus_root=tmp_corpus)
    # Spec §11 floor.
    assert df.height > 100
    assert {"Speed", "Brake", "Throttle"}.issubset(df.columns)


def test_corrupt_file_records_failed_status(tmp_path: Path, tmp_corpus: Path) -> None:
    bad = tmp_path / "broken.ibt"
    bad.write_bytes(b"not an ibt file at all")
    ids = learn(bad, corpus_root=tmp_corpus)
    assert len(ids) == 1
    s = sessions(corpus_root=tmp_corpus, valid_only=False)
    assert s.height == 1
    assert s["status"][0] == "failed"
    assert s["error"][0]


def test_partial_session_can_be_upgraded_to_ok(small_ibt: Path, tmp_corpus: Path) -> None:
    """Spec §9: a previously partial/failed session is re-attempted."""
    sid = session_id_from_bytes(small_ibt.read_bytes())
    db = catalog_path(tmp_corpus)
    # Seed a 'partial' row with the matching session_id.
    with open_catalog(db) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, car, track, ingested_at, status, error) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, "bmw", "sebring_international", "2026-04-28T00:00:00", "partial", "seeded"),
        )
        conn.commit()

    ids = learn(small_ibt, corpus_root=tmp_corpus)
    assert ids == [sid]
    s = sessions(corpus_root=tmp_corpus, valid_only=False)
    assert s.height == 1
    assert s["status"][0] == "ok"
    assert s["error"][0] is None


def test_idempotent_ingest_writes_no_new_parquet(small_ibt: Path, tmp_corpus: Path) -> None:
    ids1 = learn(small_ibt, corpus_root=tmp_corpus)
    sid = ids1[0]
    s = sessions(corpus_root=tmp_corpus, valid_only=False)
    pq = tmp_corpus / s["parquet_path"][0]
    mtime_before = pq.stat().st_mtime_ns

    ids2 = learn(small_ibt, corpus_root=tmp_corpus)
    assert ids1 == ids2
    mtime_after = pq.stat().st_mtime_ns
    assert mtime_after == mtime_before
