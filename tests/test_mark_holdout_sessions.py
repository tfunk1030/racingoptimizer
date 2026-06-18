"""`scripts/mark_holdout_sessions.py` — re-apply gate-only flags after a
fresh ingest (the weekly CI corpus rebuild would otherwise train on the
held-out IBTs; see AUDIT.md H1/N2)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.mark_holdout_sessions import MANIFEST, main, manifest_session_ids

_EXPECTED_IDS = [
    "3f0a05d3f44527bd",
    "d236a089300fc0ea",
    "fc96805e3b1a27cc",
    "72f43fa4527c4260",
    "a3d43056a952ff99",
]


def test_manifest_ids_match_verify_holdout_hardcoded_set() -> None:
    """The manifest is the single source of truth; it must agree with the
    five IDs hardcoded in scripts/verify_holdout.sh."""
    assert manifest_session_ids() == _EXPECTED_IDS
    assert MANIFEST.is_file()


def test_missing_manifest_yields_empty(tmp_path: Path) -> None:
    assert manifest_session_ids(tmp_path / "nope.sha256") == []


def test_main_flags_ingested_sessions(tmp_path: Path, monkeypatch, capsys) -> None:
    """End-to-end on a synthetic catalog: a held-out session ingested fresh
    (held_out=0) gets flagged 1; sessions not in the catalog are reported."""
    from racingoptimizer.ingest import catalog as cat

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    db = corpus / "catalog.sqlite"
    with cat.open_catalog(db) as conn:
        cat.upsert_session(
            conn,
            cat.SessionRow(
                session_id=_EXPECTED_IDS[0],
                car="bmw",
                track="spa_2024_up",
                recorded_at="2026-05-07T11:59:06",
                duration_s=1000.0,
                lap_count=10,
                weather_summary="{}",
                setup="{}",
                source_path="x.ibt",
                ingested_at="2026-06-10T00:00:00",
                parquet_path="x.parquet",
                status="ok",
                error=None,
                dropped_channels=None,
                sample_rate_hz=60.0,
            ),
        )

    monkeypatch.setenv("RACINGOPTIMIZER_CORPUS", str(corpus))
    assert main() == 0
    out = capsys.readouterr().out
    assert f"{_EXPECTED_IDS[0]}: FLAGGED" in out
    assert "NOT IN CATALOG" in out  # the other four aren't ingested here

    with sqlite3.connect(db) as conn:
        flag = conn.execute(
            "SELECT held_out FROM sessions WHERE session_id=?",
            (_EXPECTED_IDS[0],),
        ).fetchone()[0]
    assert int(flag) == 1
