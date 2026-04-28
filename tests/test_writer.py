from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import polars as pl

from racingoptimizer.ingest.catalog import get_laps, init_schema, query_sessions
from racingoptimizer.ingest.parser import ParseResult
from racingoptimizer.ingest.segment import LapSpan
from racingoptimizer.ingest.writer import session_id_from_bytes, write_session


def _fake_parse_result(n_samples: int = 600) -> ParseResult:
    pct = np.tile(np.linspace(0.0, 1.0, 200, endpoint=False), n_samples // 200).astype(np.float32)
    lap = np.repeat(np.arange(n_samples // 200), 200).astype(np.float32)
    speed = np.linspace(0, 80.0, n_samples).astype(np.float32)
    brake = np.zeros(n_samples, dtype=np.float32)
    throttle = np.ones(n_samples, dtype=np.float32)
    return ParseResult(
        yaml_car="bmwlmdh",
        yaml_track="Sebring International",
        recorded_at="2026-03-22T14:47:42",
        duration_s=n_samples / 60.0,
        channels={"LapDistPct": pct, "Lap": lap, "Speed": speed, "Brake": brake, "Throttle": throttle},
        setup={"chassis": {"front": {"wing": 16.0}}},
        weather_summary={"AirTemp_c_mean": 22.0},
        lap_spans=[LapSpan(0, 0, 200, 1), LapSpan(1, 200, 400, 1), LapSpan(2, 400, 600, 1)],
    )


def test_session_id_is_stable_over_bytes() -> None:
    a = b"hello world"
    assert session_id_from_bytes(a) == session_id_from_bytes(a)
    assert session_id_from_bytes(a) != session_id_from_bytes(b"hello world!")
    assert len(session_id_from_bytes(a)) == 16


def test_write_session_creates_parquet_and_catalog_row(tmp_corpus: Path) -> None:
    db = tmp_corpus / "catalog.sqlite"
    conn = sqlite3.connect(db)
    init_schema(conn)

    pr = _fake_parse_result()
    sid = "deadbeefcafef00d"
    parquet_p = write_session(
        conn=conn,
        corpus_root=tmp_corpus,
        session_id=sid,
        source_path="X:/fake.ibt",
        parse=pr,
    )

    assert parquet_p.exists()
    df = pl.read_parquet(parquet_p)
    assert {"t_s", "lap_index", "lap_dist_pct", "Speed", "Brake", "Throttle"}.issubset(df.columns)
    assert df.height == pr.channels["Speed"].shape[0]

    out = query_sessions(conn, valid_only=False)
    assert len(out) == 1
    s = out[0]
    assert s.session_id == sid
    assert s.car == "bmw"
    assert s.track == "sebring_international"
    assert s.status == "ok"
    assert json.loads(s.setup) == pr.setup

    laps = get_laps(conn, sid)
    assert len(laps) == 3
    assert all(lr.valid == 1 for lr in laps)
    # The fastest lap by lap_time_s would be marked best=1; with monotone Speed
    # and equal lap durations, the first lap wins on tie-break (lower lap_index).
    assert sum(lr.best for lr in laps) == 1
    conn.close()


def test_write_session_is_idempotent_on_same_id(tmp_corpus: Path) -> None:
    db = tmp_corpus / "catalog.sqlite"
    conn = sqlite3.connect(db)
    init_schema(conn)
    pr = _fake_parse_result()
    sid = "1111222233334444"
    write_session(conn=conn, corpus_root=tmp_corpus, session_id=sid, source_path="X:/a.ibt", parse=pr)
    parquet_p = write_session(conn=conn, corpus_root=tmp_corpus, session_id=sid, source_path="X:/a.ibt", parse=pr)
    rows = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
    laps = conn.execute("SELECT COUNT(*) FROM laps").fetchone()
    assert rows[0] == 1
    assert laps[0] == 3
    assert parquet_p.exists()
    conn.close()


