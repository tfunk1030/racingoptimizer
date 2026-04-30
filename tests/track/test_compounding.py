"""Compounding regime: 3 fake sessions → curb / bump likelihoods persist (U7)."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from racingoptimizer.ingest.catalog import (
    LapRow,
    SessionRow,
    insert_laps,
    open_catalog,
    upsert_session,
)
from racingoptimizer.ingest.paths import catalog_path, parquet_path
from racingoptimizer.track import build_track_model

_LAP_LENGTH_M = 350.0
_BIN_SIZE = 5.0
_LAP_SAMPLES = 700  # 70 bins × 10 samples per bin at this synthetic geometry


def _build_lap_samples(*, curb_bin: int | None, bump_bin: int | None) -> dict:
    n = _LAP_SAMPLES
    lap_dist_pct = np.linspace(0.0, 1.0 - 1e-9, n)
    track_pos_m = lap_dist_pct * _LAP_LENGTH_M
    bin_idx = np.floor(track_pos_m / _BIN_SIZE).astype(np.int32)

    # iRacing shockVel channels are in m/s; the slice-D pipeline scales
    # ×1000 at the read site to land in mm/s for the spec thresholds.
    shock = np.full(n, 0.05, dtype=np.float64)  # 50 mm/s baseline
    if curb_bin is not None:
        shock = np.where(bin_idx == curb_bin, 0.6, shock)  # 600 mm/s curb
    if bump_bin is not None:
        shock = np.where(bin_idx == bump_bin, 0.25, shock)  # 250 mm/s bump

    accel_lat = np.zeros(n, dtype=np.float64)
    return {
        "t_s": np.arange(n, dtype=np.float64) / 60.0,
        "lap_index": np.zeros(n, dtype=np.int32),
        "lap_dist_pct": lap_dist_pct.astype(np.float64),
        "data_quality_mask": np.ones(n, dtype=bool),
        "LFshockVel": shock,
        "RFshockVel": np.full(n, 0.05, dtype=np.float64),
        "LRshockVel": np.full(n, 0.05, dtype=np.float64),
        "RRshockVel": np.full(n, 0.05, dtype=np.float64),
        "LatAccel": accel_lat,
        "LFspeed": np.full(n, 60.0, dtype=np.float64),
        "RFspeed": np.full(n, 60.0, dtype=np.float64),
        "LRspeed": np.full(n, 60.0, dtype=np.float64),
        "RRspeed": np.full(n, 60.0, dtype=np.float64),
        "Speed": np.full(n, 60.0, dtype=np.float64),
    }


def _seed_session(
    corpus_root: Path, sid: str, *, curb_bin: int | None, bump_bin: int | None
) -> None:
    car = "synthetic"
    track = "synth_track"
    pq = parquet_path(corpus_root, car=car, track=track, session_id=sid)
    pq.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(_build_lap_samples(curb_bin=curb_bin, bump_bin=bump_bin)).write_parquet(
        pq, compression="zstd"
    )

    setup = {"WeekendInfo": {"TrackLength": f"{_LAP_LENGTH_M / 1000:.4f} km"}}
    with open_catalog(catalog_path(corpus_root)) as conn:
        upsert_session(
            conn,
            SessionRow(
                session_id=sid,
                car=car,
                track=track,
                recorded_at="2026-04-28T00:00:00",
                duration_s=_LAP_SAMPLES / 60.0,
                lap_count=1,
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
                    lap_time_s=_LAP_SAMPLES / 60.0,
                    start_sample=0,
                    end_sample=_LAP_SAMPLES,
                    valid=1,
                    best=1,
                )
            ],
        )


def test_compounding_3_sessions_propagates_curb_and_bump(tmp_corpus: Path):
    sids = ["sess0000000000aa", "sess0000000000bb", "sess0000000000cc"]
    # 2 of 3 sessions have curb at bin 30; all 3 have bump at bin 60.
    _seed_session(tmp_corpus, sids[0], curb_bin=30, bump_bin=60)
    _seed_session(tmp_corpus, sids[1], curb_bin=30, bump_bin=60)
    _seed_session(tmp_corpus, sids[2], curb_bin=None, bump_bin=60)

    model = build_track_model("synth_track", sids, corpus_root=tmp_corpus)
    assert model.regime == "compounding"

    bump_map = model.bump_map
    curb_row = bump_map.filter(pl.col("bin_index") == 30).row(0, named=True)
    assert curb_row["curb_likelihood"] >= 2 / 3 - 1e-9
    assert curb_row["bump_likelihood"] == 0.0

    bump_row = bump_map.filter(pl.col("bin_index") == 60).row(0, named=True)
    assert bump_row["curb_likelihood"] == 0.0
    assert bump_row["bump_likelihood"] > 0.0


def test_compounding_build_is_deterministic(tmp_corpus: Path, tmp_path: Path):
    sids = ["sess0000000000aa", "sess0000000000bb", "sess0000000000cc"]
    for sid in sids:
        _seed_session(tmp_corpus, sid, curb_bin=30, bump_bin=60)

    first = build_track_model("synth_track", sids, corpus_root=tmp_corpus)
    first_summary_bytes = first.summary_path.read_bytes()
    first_cache_bytes = first.cache_path.read_bytes()

    other_corpus = tmp_path / "corpus2"
    other_corpus.mkdir()
    for sid in sids:
        _seed_session(other_corpus, sid, curb_bin=30, bump_bin=60)
    second = build_track_model("synth_track", sids, corpus_root=other_corpus)

    assert second.summary_path.read_bytes() == first_summary_bytes
    assert second.cache_path.read_bytes() == first_cache_bytes
