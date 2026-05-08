"""Tests for the held_out catalog flag (physics-rebuild PLAN.md Section 7).

The flag is the production-side enforcement of the held-out gate set: any
production code path (fitter, recommend, learn) that calls
`query_sessions(...)` or `api.sessions(...)` must skip held-out rows by
default. Gate validation scripts opt in via `include_held_out=True`.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.api import sessions as api_sessions


def _seed(conn: sqlite3.Connection) -> None:
    """Two production sessions and one held-out session, all car=bmw."""
    cat.upsert_session(conn, cat.SessionRow(
        session_id="prod-a", car="bmw", track="sebring",
        recorded_at="2026-01-01T10:00:00", duration_s=60.0, lap_count=3,
        weather_summary=None, setup=None, source_path="ibtfiles/a.ibt",
        ingested_at="2026-01-02T10:00:00", parquet_path=None,
        status="ok", error=None, dropped_channels=None, sample_rate_hz=60.0,
        held_out=0,
    ))
    cat.upsert_session(conn, cat.SessionRow(
        session_id="prod-b", car="bmw", track="spa_2024_up",
        recorded_at="2026-01-02T10:00:00", duration_s=60.0, lap_count=3,
        weather_summary=None, setup=None, source_path="ibtfiles/b.ibt",
        ingested_at="2026-01-03T10:00:00", parquet_path=None,
        status="ok", error=None, dropped_channels=None, sample_rate_hz=60.0,
        held_out=0,
    ))
    cat.upsert_session(conn, cat.SessionRow(
        session_id="held-c", car="bmw", track="spa_2024_up",
        recorded_at="2026-01-03T10:00:00", duration_s=60.0, lap_count=3,
        weather_summary=None, setup=None, source_path="ibtfiles/c.ibt",
        ingested_at="2026-01-04T10:00:00", parquet_path=None,
        status="ok", error=None, dropped_channels=None, sample_rate_hz=60.0,
        held_out=1,
    ))


def test_default_query_excludes_held_out(tmp_path: Path) -> None:
    db = tmp_path / "catalog.sqlite"
    with cat.open_catalog(db) as conn:
        _seed(conn)
        rows = cat.query_sessions(conn)
        ids = sorted(r.session_id for r in rows)
    assert ids == ["prod-a", "prod-b"]


def test_explicit_include_held_out(tmp_path: Path) -> None:
    db = tmp_path / "catalog.sqlite"
    with cat.open_catalog(db) as conn:
        _seed(conn)
        rows = cat.query_sessions(conn, include_held_out=True)
        ids = sorted(r.session_id for r in rows)
    assert ids == ["held-c", "prod-a", "prod-b"]


def test_get_session_returns_held_out(tmp_path: Path) -> None:
    """Direct lookup by id always returns the row, held-out or not.

    `get_session` is used by the corner-state loader (which receives an
    explicit session_id) and by gate scripts. The held-out filter applies
    only to listing.
    """
    db = tmp_path / "catalog.sqlite"
    with cat.open_catalog(db) as conn:
        _seed(conn)
        row = cat.get_session(conn, "held-c")
    assert row is not None
    assert row.held_out == 1


def test_set_held_out_sessions(tmp_path: Path) -> None:
    db = tmp_path / "catalog.sqlite"
    with cat.open_catalog(db) as conn:
        _seed(conn)
        n = cat.set_held_out_sessions(conn, ["prod-a"], held_out=True)
        assert n == 1
        row = cat.get_session(conn, "prod-a")
        assert row is not None and row.held_out == 1
        # Default query now excludes prod-a.
        rows = cat.query_sessions(conn)
        ids = sorted(r.session_id for r in rows)
        assert ids == ["prod-b"]


def test_set_held_out_sessions_unset(tmp_path: Path) -> None:
    db = tmp_path / "catalog.sqlite"
    with cat.open_catalog(db) as conn:
        _seed(conn)
        n = cat.set_held_out_sessions(conn, ["held-c"], held_out=False)
        assert n == 1
        rows = cat.query_sessions(conn)
        ids = sorted(r.session_id for r in rows)
        assert ids == ["held-c", "prod-a", "prod-b"]


def test_upsert_does_not_clobber_held_out(tmp_path: Path) -> None:
    """Re-ingesting an IBT must not silently flip held_out back to 0.

    PLAN.md Section 7 invariant: once a session is gate-only, it stays
    gate-only until set_held_out_sessions is explicitly called with
    held_out=False. The upsert path's ON CONFLICT clause must omit
    held_out from the UPDATE columns.
    """
    db = tmp_path / "catalog.sqlite"
    with cat.open_catalog(db) as conn:
        _seed(conn)
        # Simulate re-ingesting `held-c` -- new row constructed without
        # the held_out flag (default 0).
        cat.upsert_session(conn, cat.SessionRow(
            session_id="held-c", car="bmw", track="spa_2024_up",
            recorded_at="2026-02-01T10:00:00", duration_s=120.0, lap_count=5,
            weather_summary=None, setup=None, source_path="ibtfiles/c.ibt",
            ingested_at="2026-02-02T10:00:00", parquet_path=None,
            status="ok", error=None, dropped_channels=None, sample_rate_hz=60.0,
        ))
        row = cat.get_session(conn, "held-c")
    assert row is not None
    assert row.held_out == 1, "re-ingest must not clobber held_out flag"
    assert row.duration_s == 120.0, "but other fields must be updated"


def test_api_sessions_excludes_held_out_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    """The high-level `api.sessions()` defaults to excluding held-out."""
    monkeypatch.setenv("RACINGOPTIMIZER_CORPUS_ROOT", str(tmp_path))
    (tmp_path / "corpus").mkdir(exist_ok=True)
    db = tmp_path / "corpus" / "catalog.sqlite"
    with cat.open_catalog(db) as conn:
        _seed(conn)
    df = api_sessions(corpus_root=tmp_path / "corpus")
    ids = sorted(df["session_id"].to_list())
    assert ids == ["prod-a", "prod-b"]


def test_api_sessions_include_held_out_opt_in(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RACINGOPTIMIZER_CORPUS_ROOT", str(tmp_path))
    (tmp_path / "corpus").mkdir(exist_ok=True)
    db = tmp_path / "corpus" / "catalog.sqlite"
    with cat.open_catalog(db) as conn:
        _seed(conn)
    df = api_sessions(corpus_root=tmp_path / "corpus", include_held_out=True)
    ids = sorted(df["session_id"].to_list())
    assert ids == ["held-c", "prod-a", "prod-b"]
    # Held-out rows expose the flag in the dataframe so callers can audit.
    held = dict(zip(
        df["session_id"].to_list(), df["held_out"].to_list(), strict=True,
    ))
    assert held == {"prod-a": 0, "prod-b": 0, "held-c": 1}


def test_set_held_out_sessions_empty_is_noop(tmp_path: Path) -> None:
    db = tmp_path / "catalog.sqlite"
    with cat.open_catalog(db) as conn:
        _seed(conn)
        n = cat.set_held_out_sessions(conn, [], held_out=True)
        assert n == 0
