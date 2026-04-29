"""Public API for the ingest module."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.parser import parse_ibt
from racingoptimizer.ingest.paths import (
    catalog_path,
    parquet_path,
    resolve_corpus_root,
)
from racingoptimizer.ingest.writer import _now_iso, session_id_from_bytes, write_session


def learn(path: Path | str, corpus_root: Path | str | None = None) -> list[str]:
    """Ingest a .ibt file or every .ibt under a directory.

    Returns the list of session_ids for every file processed (existing or new),
    regardless of status. Caller can join against `sessions(...)` to inspect
    per-session outcomes.
    """
    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    db = catalog_path(root)

    targets = list(_iter_ibt_paths(Path(path)))
    out: list[str] = []
    with cat.open_catalog(db) as conn:
        for ibt in targets:
            sid = _process_one(conn, root, ibt)
            out.append(sid)
    return out


def sessions(
    car: str | None = None,
    track: str | None = None,
    valid_only: bool = True,
    corpus_root: Path | str | None = None,
) -> pl.DataFrame:
    """Return one row per session, ordered by recorded_at."""
    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    with cat.open_catalog(catalog_path(root)) as conn:
        rows = cat.query_sessions(conn, car=car, track=track, valid_only=valid_only)
    return pl.DataFrame(
        {
            "session_id": [r.session_id for r in rows],
            "car": [r.car for r in rows],
            "track": [r.track for r in rows],
            "recorded_at": [r.recorded_at for r in rows],
            "duration_s": [r.duration_s for r in rows],
            "lap_count": [r.lap_count for r in rows],
            "weather_summary": [r.weather_summary for r in rows],
            "setup": [r.setup for r in rows],
            "status": [r.status for r in rows],
            "error": [r.error for r in rows],
            "parquet_path": [r.parquet_path for r in rows],
        }
    )


def laps(
    session_id: str | None = None,
    car: str | None = None,
    track: str | None = None,
    valid_only: bool = True,
    corpus_root: Path | str | None = None,
) -> pl.DataFrame:
    """Return one row per lap matching the filters."""
    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    with cat.open_catalog(catalog_path(root)) as conn:
        sid_list: list[str]
        if session_id is not None:
            sid_list = [session_id]
        else:
            sid_list = [
                s.session_id
                for s in cat.query_sessions(conn, car=car, track=track, valid_only=valid_only)
            ]

        all_rows: list[cat.LapRow] = []
        for sid in sid_list:
            rows = cat.get_laps(conn, sid)
            if valid_only:
                rows = [r for r in rows if r.valid == 1]
            all_rows.extend(rows)

    return pl.DataFrame(
        {
            "session_id": [r.session_id for r in all_rows],
            "lap_index": [r.lap_index for r in all_rows],
            "lap_time_s": [r.lap_time_s for r in all_rows],
            "start_sample": [r.start_sample for r in all_rows],
            "end_sample": [r.end_sample for r in all_rows],
            "valid": [r.valid for r in all_rows],
            "best": [r.best for r in all_rows],
        }
    )


def lap_data(
    session_id: str,
    lap_index: int,
    channels: list[str] | None = None,
    corpus_root: Path | str | None = None,
) -> pl.DataFrame:
    """Read one lap's bulk 60 Hz channels from the session's parquet."""
    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    with cat.open_catalog(catalog_path(root)) as conn:
        sess = cat.get_session(conn, session_id)
        if sess is None:
            raise KeyError(f"unknown session_id: {session_id}")
        rows = [r for r in cat.get_laps(conn, session_id) if r.lap_index == lap_index]
        if not rows:
            raise KeyError(f"no lap_index={lap_index} in session {session_id}")
        lap = rows[0]
        pq = parquet_path(root, car=sess.car, track=sess.track, session_id=session_id)

    lf = pl.scan_parquet(pq)
    if channels is not None:
        lf = lf.select(channels)
    df = lf.slice(lap.start_sample, lap.end_sample - lap.start_sample).collect()
    return df


# ---- internal helpers ----

def _iter_ibt_paths(p: Path) -> Iterable[Path]:
    if p.is_file() and p.suffix.lower() == ".ibt":
        yield p
    elif p.is_dir():
        yield from sorted(p.rglob("*.ibt"))


def _process_one(conn: sqlite3.Connection, root: Path, ibt_path: Path) -> str:
    raw = ibt_path.read_bytes()
    sid = session_id_from_bytes(raw)
    existing = cat.get_session(conn, sid)
    if existing is not None and existing.status == "ok":
        return sid
    try:
        parse = parse_ibt(ibt_path)
        write_session(
            conn=conn,
            corpus_root=root,
            session_id=sid,
            source_path=str(ibt_path),
            parse=parse,
        )
    except Exception as exc:  # noqa: BLE001 — every failure must register
        cat.upsert_session(
            conn,
            cat.SessionRow(
                session_id=sid,
                car="unknown",
                track="unknown",
                recorded_at=None,
                duration_s=None,
                lap_count=None,
                weather_summary=None,
                setup=None,
                source_path=str(ibt_path),
                ingested_at=_now_iso(),
                parquet_path=None,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
    return sid
