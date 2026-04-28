"""Persist a ParseResult: parquet file + catalog rows."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from racingoptimizer.ingest.catalog import LapRow, SessionRow, insert_laps, upsert_session
from racingoptimizer.ingest.detect import detect_car, detect_car_from_filename, slugify_track
from racingoptimizer.ingest.parser import ParseResult
from racingoptimizer.ingest.paths import parquet_path


def session_id_from_bytes(b: bytes) -> str:
    """Stable 16-hex-char session id derived from raw IBT bytes."""
    return hashlib.sha256(b).hexdigest()[:16]


def session_id_from_path(path: Path | str) -> str:
    return session_id_from_bytes(Path(path).read_bytes())


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _build_dataframe(pr: ParseResult) -> pl.DataFrame:
    n = pr.channels["LapDistPct"].shape[0]
    t_s = (np.arange(n, dtype=np.float64) / 60.0).astype(np.float64)

    # lap_index: -1 outside any lap_span, else the span's lap_index.
    lap_index = np.full(n, -1, dtype=np.int32)
    for span in pr.lap_spans:
        lap_index[span.start_sample : span.end_sample] = span.lap_index

    # Reserved column for slice D's track-model handshake: True for every
    # freshly ingested sample. Slice D will flip dirty samples (curb strikes,
    # off-track excursions, etc.) to False without rewriting the rest of the
    # schema across the entire corpus.
    data_quality_mask = np.ones(n, dtype=bool)

    data: dict[str, np.ndarray] = {
        "t_s": t_s,
        "lap_index": lap_index,
        "lap_dist_pct": pr.channels["LapDistPct"],
        "data_quality_mask": data_quality_mask,
    }
    for name, arr in pr.channels.items():
        if name == "LapDistPct":
            continue   # already aliased as lap_dist_pct
        data[name] = arr
    return pl.DataFrame(data)


def _lap_rows(session_id: str, pr: ParseResult) -> list[LapRow]:
    rows: list[LapRow] = []
    valid_durations: list[tuple[int, float]] = []
    for span in pr.lap_spans:
        duration = (span.end_sample - span.start_sample) / 60.0
        lap_time = duration if span.valid else None
        if span.valid and lap_time is not None:
            valid_durations.append((span.lap_index, lap_time))
        rows.append(
            LapRow(
                session_id=session_id,
                lap_index=span.lap_index,
                lap_time_s=lap_time,
                start_sample=span.start_sample,
                end_sample=span.end_sample,
                valid=span.valid,
                best=0,
            )
        )
    if valid_durations:
        # Tie-break by lower lap_index.
        best_idx = min(valid_durations, key=lambda t: (t[1], t[0]))[0]
        rows = [r._replace(best=1) if (r.valid and r.lap_index == best_idx) else r for r in rows]
    return rows


def write_session(
    *,
    conn: sqlite3.Connection,
    corpus_root: Path,
    session_id: str,
    source_path: str,
    parse: ParseResult,
) -> Path:
    """Write parquet + catalog rows. Returns the parquet path."""
    car = detect_car(yaml_car=parse.yaml_car, filename_car=detect_car_from_filename(source_path))
    track = slugify_track(parse.yaml_track)

    pq = parquet_path(corpus_root, car=car, track=track, session_id=session_id)
    pq.parent.mkdir(parents=True, exist_ok=True)

    df = _build_dataframe(parse)
    df.write_parquet(pq, compression="zstd")

    laps = _lap_rows(session_id, parse)

    session = SessionRow(
        session_id=session_id,
        car=car,
        track=track,
        recorded_at=parse.recorded_at,
        duration_s=parse.duration_s,
        lap_count=sum(1 for s in parse.lap_spans if s.valid),
        weather_summary=json.dumps(parse.weather_summary),
        setup=json.dumps(parse.setup),
        source_path=source_path,
        ingested_at=_now_iso(),
        parquet_path=str(pq.relative_to(corpus_root).as_posix()),
        status="ok",
        error=None,
    )
    upsert_session(conn, session)
    insert_laps(conn, laps)
    return pq
