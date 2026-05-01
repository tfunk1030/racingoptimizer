from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from racingoptimizer.ingest.catalog import get_laps, init_schema, query_sessions
from racingoptimizer.ingest.parser import ParseResult
from racingoptimizer.ingest.segment import LapSpan
from racingoptimizer.ingest.writer import session_id_from_bytes, write_session


def _fake_parse_result(
    n_samples: int = 600,
    *,
    samples_per_lap: int = 200,
    sample_rate_hz: float = 60.0,
    dropped_channels: dict[str, str] | None = None,
) -> ParseResult:
    n_laps = n_samples // samples_per_lap
    assert n_laps * samples_per_lap == n_samples, "n_samples must be a multiple of samples_per_lap"
    pct = np.tile(
        np.linspace(0.0, 1.0, samples_per_lap, endpoint=False), n_laps
    ).astype(np.float32)
    lap = np.repeat(np.arange(n_laps), samples_per_lap).astype(np.float32)
    speed = np.linspace(0, 80.0, n_samples).astype(np.float32)
    brake = np.zeros(n_samples, dtype=np.float32)
    throttle = np.ones(n_samples, dtype=np.float32)
    spans = [
        LapSpan(i, i * samples_per_lap, (i + 1) * samples_per_lap, 1) for i in range(n_laps)
    ]
    return ParseResult(
        yaml_car="bmwlmdh",
        yaml_track="Sebring International",
        recorded_at="2026-03-22T14:47:42",
        duration_s=n_samples / sample_rate_hz,
        sample_rate_hz=sample_rate_hz,
        channels={
            "LapDistPct": pct, "Lap": lap, "Speed": speed,
            "Brake": brake, "Throttle": throttle,
        },
        setup={"chassis": {"front": {"wing": 16.0}}},
        weather_summary={"AirTemp_c_mean": 22.0},
        lap_spans=spans,
        dropped_channels=dropped_channels or {},
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


def test_data_quality_mask_column_present_and_all_true(tmp_corpus: Path) -> None:
    """Slice D handshake: writer reserves a `data_quality_mask` boolean column."""
    db = tmp_corpus / "catalog.sqlite"
    conn = sqlite3.connect(db)
    init_schema(conn)
    pr = _fake_parse_result()
    parquet_p = write_session(
        conn=conn,
        corpus_root=tmp_corpus,
        session_id="aaaabbbbccccdddd",
        source_path="X:/a.ibt",
        parse=pr,
    )
    df = pl.read_parquet(parquet_p)
    assert "data_quality_mask" in df.columns
    assert df.schema["data_quality_mask"] == pl.Boolean
    assert df["data_quality_mask"].all()
    conn.close()


def test_write_session_is_idempotent_on_same_id(tmp_corpus: Path) -> None:
    db = tmp_corpus / "catalog.sqlite"
    conn = sqlite3.connect(db)
    init_schema(conn)
    pr = _fake_parse_result()
    sid = "1111222233334444"
    write_session(
        conn=conn, corpus_root=tmp_corpus,
        session_id=sid, source_path="X:/a.ibt", parse=pr,
    )
    parquet_p = write_session(
        conn=conn, corpus_root=tmp_corpus,
        session_id=sid, source_path="X:/a.ibt", parse=pr,
    )
    rows = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
    laps = conn.execute("SELECT COUNT(*) FROM laps").fetchone()
    assert rows[0] == 1
    assert laps[0] == 3
    assert parquet_p.exists()
    conn.close()


def test_dropped_channels_persisted(tmp_corpus: Path) -> None:
    """VISION §1: ParseResult.dropped_channels must round-trip through the
    catalog as queryable JSON, so the audit trail survives ingestion."""
    db = tmp_corpus / "catalog.sqlite"
    conn = sqlite3.connect(db)
    init_schema(conn)

    drops = {
        "CarIdxPosition": "excluded by EXCLUDED_CHANNEL_PATTERNS",
        "TyreLayerTempCM": "multi-element array (count=4)",
    }
    pr = _fake_parse_result(dropped_channels=drops)
    sid = "0011223344556677"
    write_session(
        conn=conn, corpus_root=tmp_corpus, session_id=sid, source_path="X:/x.ibt", parse=pr,
    )

    raw = conn.execute(
        "SELECT dropped_channels, sample_rate_hz FROM sessions WHERE session_id=?",
        (sid,),
    ).fetchone()
    assert raw is not None
    persisted_drops = json.loads(raw[0])
    assert persisted_drops == drops
    assert raw[1] == pytest.approx(60.0)

    # The polars-shaped public API should also expose them.
    sess = query_sessions(conn)
    assert sess[0].dropped_channels == json.dumps(drops)
    assert sess[0].sample_rate_hz == pytest.approx(60.0)
    conn.close()


def test_non_60hz_sample_rate_propagates_to_time_axis(tmp_corpus: Path) -> None:
    """VISION §1 / Gap #9: a non-60 Hz IBT must produce a t_s column scaled by
    the detected rate, not the legacy 60 Hz constant."""
    db = tmp_corpus / "catalog.sqlite"
    conn = sqlite3.connect(db)
    init_schema(conn)

    # 360 Hz is iRacing's app.ini-tunable max; pick it as a stress case.
    # 720 samples = 2 "laps" of 360 samples each at 360 Hz = 2 seconds total.
    n_samples = 720
    samples_per_lap = 360
    rate = 360.0
    pr = _fake_parse_result(
        n_samples=n_samples, samples_per_lap=samples_per_lap, sample_rate_hz=rate
    )
    sid = "abcdef0011223344"
    parquet_p = write_session(
        conn=conn, corpus_root=tmp_corpus, session_id=sid, source_path="X:/x.ibt", parse=pr
    )

    df = pl.read_parquet(parquet_p)
    assert df.height == n_samples
    assert df["t_s"][0] == pytest.approx(0.0)
    assert df["t_s"][-1] == pytest.approx((n_samples - 1) / rate, abs=1e-9)

    # Lap timing must use the detected rate too.
    sess_row = query_sessions(conn)[0]
    assert sess_row.duration_s == pytest.approx(n_samples / rate)
    assert sess_row.sample_rate_hz == pytest.approx(rate)

    # Each lap is samples_per_lap samples long → 1.0 s at 360 Hz.
    lap_rows = get_laps(conn, sid)
    assert lap_rows
    for lr in lap_rows:
        assert lr.lap_time_s == pytest.approx(samples_per_lap / rate)
    conn.close()


