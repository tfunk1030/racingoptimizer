"""SQLite catalog: sessions and lap rows.

Schema lives in `SCHEMA_SQL`. The catalog is rebuildable from raw IBTs (see
`learn`), so there is no migration system — drop the file and re-ingest if the
schema ever changes.
"""
from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id      TEXT PRIMARY KEY,
  car             TEXT NOT NULL,
  track           TEXT NOT NULL,
  recorded_at     TEXT,
  duration_s      REAL,
  lap_count       INTEGER,
  weather_summary TEXT,
  setup           TEXT,
  source_path     TEXT,
  ingested_at     TEXT NOT NULL,
  parquet_path    TEXT,
  status          TEXT NOT NULL CHECK(status IN ('ok','partial','failed')),
  error           TEXT
);

CREATE TABLE IF NOT EXISTS laps (
  session_id   TEXT NOT NULL,
  lap_index    INTEGER NOT NULL,
  lap_time_s   REAL,
  start_sample INTEGER NOT NULL,
  end_sample   INTEGER NOT NULL,
  valid        INTEGER NOT NULL,
  best         INTEGER NOT NULL,
  PRIMARY KEY (session_id, lap_index),
  FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_car_track ON sessions(car, track);
CREATE INDEX IF NOT EXISTS idx_laps_session ON laps(session_id);
"""


class SessionRow(NamedTuple):
    session_id: str
    car: str
    track: str
    recorded_at: str | None
    duration_s: float | None
    lap_count: int | None
    weather_summary: str | None  # JSON
    setup: str | None            # JSON
    source_path: str | None
    ingested_at: str
    parquet_path: str | None
    status: str                  # 'ok' | 'partial' | 'failed'
    error: str | None


class LapRow(NamedTuple):
    session_id: str
    lap_index: int
    lap_time_s: float | None
    start_sample: int
    end_sample: int
    valid: int   # 0/1
    best: int    # 0/1


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


@contextlib.contextmanager
def open_catalog(path: Path | str) -> Iterator[sqlite3.Connection]:
    """Open the catalog at `path`, creating its directory and schema if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        init_schema(conn)
        yield conn
    finally:
        conn.close()


def upsert_session(conn: sqlite3.Connection, row: SessionRow) -> None:
    conn.execute(
        """
        INSERT INTO sessions (
            session_id, car, track, recorded_at, duration_s, lap_count,
            weather_summary, setup, source_path, ingested_at, parquet_path,
            status, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            car=excluded.car,
            track=excluded.track,
            recorded_at=excluded.recorded_at,
            duration_s=excluded.duration_s,
            lap_count=excluded.lap_count,
            weather_summary=excluded.weather_summary,
            setup=excluded.setup,
            source_path=excluded.source_path,
            ingested_at=excluded.ingested_at,
            parquet_path=excluded.parquet_path,
            status=excluded.status,
            error=excluded.error
        """,
        tuple(row),
    )
    conn.commit()


def update_session_status(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    status: str,
    error: str | None,
    parquet_path: str | None,
) -> None:
    conn.execute(
        "UPDATE sessions SET status=?, error=?, parquet_path=? WHERE session_id=?",
        (status, error, parquet_path, session_id),
    )
    conn.commit()


def insert_laps(conn: sqlite3.Connection, laps: Iterable[LapRow]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO laps
            (session_id, lap_index, lap_time_s, start_sample, end_sample, valid, best)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        list(laps),
    )
    conn.commit()


def query_sessions(
    conn: sqlite3.Connection,
    *,
    car: str | None = None,
    track: str | None = None,
    valid_only: bool = True,
) -> list[SessionRow]:
    sql = "SELECT * FROM sessions"
    where: list[str] = []
    params: list[object] = []
    if car is not None:
        where.append("car = ?")
        params.append(car)
    if track is not None:
        where.append("track = ?")
        params.append(track)
    if valid_only:
        where.append("status IN ('ok','partial')")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY recorded_at"
    rows = conn.execute(sql, params).fetchall()
    return [SessionRow(*r) for r in rows]


def get_session(conn: sqlite3.Connection, session_id: str) -> SessionRow | None:
    row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    return SessionRow(*row) if row else None


def get_laps(conn: sqlite3.Connection, session_id: str) -> list[LapRow]:
    rows = conn.execute(
        "SELECT * FROM laps WHERE session_id=? ORDER BY lap_index", (session_id,)
    ).fetchall()
    return [LapRow(*r) for r in rows]
