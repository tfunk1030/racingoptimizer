from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from racingoptimizer.ingest.catalog import (
    LapRow,
    SessionRow,
    init_schema,
    insert_laps,
    open_catalog,
    query_sessions,
    update_session_status,
    upsert_session,
)


def _now() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    init_schema(c)
    return c


def _make_session(**overrides) -> SessionRow:
    base = SessionRow(
        session_id="abc1234567890def",
        car="porsche",
        track="laguna_seca",
        recorded_at="2026-04-26T18:25:49",
        duration_s=600.0,
        lap_count=5,
        weather_summary=json.dumps({"AirTemp_c_mean": 22.4}),
        setup=json.dumps({"wing": 16.0}),
        source_path="C:/x/y.ibt",
        ingested_at=_now(),
        parquet_path="sessions/porsche/laguna_seca/abc1234567890def.parquet",
        status="ok",
        error=None,
        dropped_channels=json.dumps({}),
        sample_rate_hz=60.0,
    )
    return base._replace(**overrides) if overrides else base


def test_init_schema_creates_required_tables(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {r[0] for r in rows}
    assert {"sessions", "laps"}.issubset(names)


def test_upsert_then_query_round_trip(conn: sqlite3.Connection) -> None:
    s = _make_session()
    upsert_session(conn, s)
    out = query_sessions(conn, car="porsche")
    assert len(out) == 1
    assert out[0].session_id == s.session_id
    assert out[0].car == "porsche"
    assert out[0].track == "laguna_seca"


def test_query_filters_by_track(conn: sqlite3.Connection) -> None:
    upsert_session(conn, _make_session(session_id="a" * 16, track="laguna_seca"))
    upsert_session(conn, _make_session(session_id="b" * 16, track="sebring_international"))
    out = query_sessions(conn, track="sebring_international")
    assert len(out) == 1
    assert out[0].track == "sebring_international"


def test_query_excludes_failed_sessions_when_valid_only(conn: sqlite3.Connection) -> None:
    upsert_session(conn, _make_session(session_id="a" * 16, status="ok"))
    upsert_session(conn, _make_session(session_id="b" * 16, status="failed", error="bad header"))
    out_valid = query_sessions(conn, valid_only=True)
    out_all = query_sessions(conn, valid_only=False)
    assert {s.session_id for s in out_valid} == {"a" * 16}
    assert {s.session_id for s in out_all} == {"a" * 16, "b" * 16}


def test_upsert_is_idempotent_on_session_id(conn: sqlite3.Connection) -> None:
    s = _make_session()
    upsert_session(conn, s)
    upsert_session(conn, s)
    rows = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
    assert rows[0] == 1


def test_update_session_status_changes_partial_to_ok(conn: sqlite3.Connection) -> None:
    upsert_session(conn, _make_session(status="partial", error="parser ran out of bytes"))
    update_session_status(
        conn,
        session_id="abc1234567890def",
        status="ok",
        error=None,
        parquet_path="sessions/porsche/laguna_seca/abc1234567890def.parquet",
    )
    out = query_sessions(conn, valid_only=False)
    assert out[0].status == "ok"
    assert out[0].error is None


def test_insert_laps_round_trip(conn: sqlite3.Connection) -> None:
    s = _make_session()
    upsert_session(conn, s)
    laps = [
        LapRow(
            s.session_id, lap_index=0, lap_time_s=92.1,
            start_sample=0, end_sample=5530, valid=1, best=0,
        ),
        LapRow(
            s.session_id, lap_index=1, lap_time_s=91.4,
            start_sample=5530, end_sample=11020, valid=1, best=1,
        ),
    ]
    insert_laps(conn, laps)
    rows = conn.execute(
        "SELECT lap_index, lap_time_s, valid, best FROM laps WHERE session_id=? ORDER BY lap_index",
        (s.session_id,),
    ).fetchall()
    assert rows == [(0, 92.1, 1, 0), (1, 91.4, 1, 1)]


def test_open_catalog_creates_db_file(tmp_path: Path) -> None:
    db = tmp_path / "catalog.sqlite"
    with open_catalog(db) as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert db.exists()
    assert {"sessions", "laps"}.issubset(names)
