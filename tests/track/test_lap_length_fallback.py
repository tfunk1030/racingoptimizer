"""Speed-integral lap-length fallback must skip pit-idle laps (S1.2 fix).

Regression: `_lap_length_from_speed_fallback` previously picked the
wallclock-longest lap by `argmax(end_sample - start_sample)`, then integrated
`Σ Speed × dt` to estimate track length. On Porsche/Algarve corpora — where
the IBT YAML omits `WeekendInfo.TrackLength` — this selected a 350-second
pit-out lap whose mean Speed was ~1 m/s, integrating to ~412 m instead of
the real ~4600 m. The wrong `lap_length_m` then poisoned every downstream
`track_pos_m_from_pct` call (1.0 lap_dist_pct collapsing to a 412 m range).

This test seeds a synthetic session with two laps:
  (a) a long pit-idle lap (mean Speed 1 m/s, very long wallclock duration)
  (b) a shorter racing lap (mean Speed 60 m/s)

and asserts the fallback returns the racing lap's integrated length, not
the pit lap's.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from racingoptimizer.ingest.catalog import (
    LapRow,
    SessionRow,
    insert_laps,
    open_catalog,
    upsert_session,
)
from racingoptimizer.ingest.paths import catalog_path, parquet_path
from racingoptimizer.track.builder import _lap_length_from_speed_fallback

_PIT_LAP_SAMPLES = 21_000   # 350 s of pit idle at 60 Hz
_RACE_LAP_SAMPLES = 4_500   # 75 s of clean lap at 60 Hz
_PIT_LAP_MEAN_SPEED = 1.0   # m/s — a crawl on pit lane
_RACE_LAP_MEAN_SPEED = 60.0  # m/s — typical GTP cruising


def _seed_two_lap_session(corpus_root: Path) -> str:
    """Seed a 2-lap session: one long pit-idle lap, one shorter racing lap.

    Returns the session_id. The session's setup JSON deliberately omits
    `WeekendInfo.TrackLength` so callers fall through to the speed-integral
    fallback.
    """
    sid = "sess0000pitidle1"
    car = "synthetic"
    track = "synth_pit_track"

    n_total = _PIT_LAP_SAMPLES + _RACE_LAP_SAMPLES
    speed = np.empty(n_total, dtype=np.float64)
    speed[:_PIT_LAP_SAMPLES] = _PIT_LAP_MEAN_SPEED
    speed[_PIT_LAP_SAMPLES:] = _RACE_LAP_MEAN_SPEED

    pq = parquet_path(corpus_root, car=car, track=track, session_id=sid)
    pq.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"Speed": speed}).write_parquet(pq, compression="zstd")

    # Setup omits WeekendInfo.TrackLength → exercises the speed-integral path.
    setup = {"WeekendInfo": {}}
    with open_catalog(catalog_path(corpus_root)) as conn:
        upsert_session(
            conn,
            SessionRow(
                session_id=sid,
                car=car,
                track=track,
                recorded_at="2026-04-28T00:00:00",
                duration_s=n_total / 60.0,
                lap_count=2,
                weather_summary=json.dumps({}),
                setup=json.dumps(setup),
                source_path=f"/synthetic/{sid}.ibt",
                ingested_at=datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds"),
                parquet_path=str(pq.relative_to(corpus_root).as_posix()),
                status="ok",
                error=None,
            dropped_channels=json.dumps({}),
            sample_rate_hz=60.0,
        ),
        )
        insert_laps(
            conn,
            [
                LapRow(
                    session_id=sid,
                    lap_index=0,
                    lap_time_s=_PIT_LAP_SAMPLES / 60.0,
                    start_sample=0,
                    end_sample=_PIT_LAP_SAMPLES,
                    valid=1,
                    best=0,
                ),
                LapRow(
                    session_id=sid,
                    lap_index=1,
                    lap_time_s=_RACE_LAP_SAMPLES / 60.0,
                    start_sample=_PIT_LAP_SAMPLES,
                    end_sample=n_total,
                    valid=1,
                    best=1,
                ),
            ],
        )
    return sid


def test_fallback_picks_racing_lap_not_pit_idle(tmp_corpus: Path) -> None:
    """The fallback must integrate the racing lap, not the longer pit-idle one."""
    sid = _seed_two_lap_session(tmp_corpus)

    result = _lap_length_from_speed_fallback(sid, corpus_root=tmp_corpus)

    pit_integral = _PIT_LAP_SAMPLES * _PIT_LAP_MEAN_SPEED / 60.0     # ~350 m
    race_integral = _RACE_LAP_SAMPLES * _RACE_LAP_MEAN_SPEED / 60.0  # ~4500 m
    assert result is not None
    assert result == pytest.approx(race_integral, rel=1e-9)
    # And explicitly, it must NOT be the pit-idle integral that the old
    # `argmax(end_sample - start_sample)` heuristic would have returned.
    assert abs(result - pit_integral) > 1.0


def test_fallback_returns_none_when_only_pit_laps(tmp_corpus: Path) -> None:
    """If every lap is below the racing-speed threshold, return None."""
    sid = "sess0000onlypit1"
    car = "synthetic"
    track = "synth_only_pit"

    speed = np.full(_PIT_LAP_SAMPLES, _PIT_LAP_MEAN_SPEED, dtype=np.float64)
    pq = parquet_path(corpus_root=tmp_corpus, car=car, track=track, session_id=sid)
    pq.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"Speed": speed}).write_parquet(pq, compression="zstd")

    setup = {"WeekendInfo": {}}
    with open_catalog(catalog_path(tmp_corpus)) as conn:
        upsert_session(
            conn,
            SessionRow(
                session_id=sid,
                car=car,
                track=track,
                recorded_at="2026-04-28T00:00:00",
                duration_s=_PIT_LAP_SAMPLES / 60.0,
                lap_count=1,
                weather_summary=json.dumps({}),
                setup=json.dumps(setup),
                source_path=f"/synthetic/{sid}.ibt",
                ingested_at=datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds"),
                parquet_path=str(pq.relative_to(tmp_corpus).as_posix()),
                status="ok",
                error=None,
            dropped_channels=json.dumps({}),
            sample_rate_hz=60.0,
        ),
        )
        insert_laps(
            conn,
            [
                LapRow(
                    session_id=sid,
                    lap_index=0,
                    lap_time_s=_PIT_LAP_SAMPLES / 60.0,
                    start_sample=0,
                    end_sample=_PIT_LAP_SAMPLES,
                    valid=1,
                    best=1,
                )
            ],
        )

    assert _lap_length_from_speed_fallback(sid, corpus_root=tmp_corpus) is None
