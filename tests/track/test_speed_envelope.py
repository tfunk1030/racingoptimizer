"""Per-bin speed envelope (VISION §9 — typical speed at every point).

Synthetic 3-session corpus: each bin has a known speed; the envelope's
min / median / max columns must reproduce them after cross-session aggregation.
The test deliberately uses *different* per-session speeds in one bin to verify
the cross-session collapse (min, median-of-medians, max) — not just a single
session's per-bin reduction.
"""
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


def _build_lap_samples(*, speed_at_bin: dict[int, float], baseline_speed: float) -> dict:
    n = _LAP_SAMPLES
    lap_dist_pct = np.linspace(0.0, 1.0 - 1e-9, n)
    track_pos_m = lap_dist_pct * _LAP_LENGTH_M
    bin_idx = np.floor(track_pos_m / _BIN_SIZE).astype(np.int32)

    speed = np.full(n, baseline_speed, dtype=np.float64)
    for b, v in speed_at_bin.items():
        speed = np.where(bin_idx == b, v, speed)

    return {
        "t_s": np.arange(n, dtype=np.float64) / 60.0,
        "lap_index": np.zeros(n, dtype=np.int32),
        "lap_dist_pct": lap_dist_pct.astype(np.float64),
        "data_quality_mask": np.ones(n, dtype=bool),
        "LFshockVel": np.full(n, 0.05, dtype=np.float64),
        "RFshockVel": np.full(n, 0.05, dtype=np.float64),
        "LRshockVel": np.full(n, 0.05, dtype=np.float64),
        "RRshockVel": np.full(n, 0.05, dtype=np.float64),
        "LatAccel": np.zeros(n, dtype=np.float64),
        "LFspeed": speed,
        "RFspeed": speed,
        "LRspeed": speed,
        "RRspeed": speed,
        "Speed": speed,
    }


def _seed_session(
    corpus_root: Path,
    sid: str,
    *,
    speed_at_bin: dict[int, float],
    baseline_speed: float,
) -> None:
    car = "synthetic"
    track = "synth_track"
    pq = parquet_path(corpus_root, car=car, track=track, session_id=sid)
    pq.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        _build_lap_samples(speed_at_bin=speed_at_bin, baseline_speed=baseline_speed)
    ).write_parquet(pq, compression="zstd")

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


def test_speed_envelope_min_median_max_across_sessions(tmp_corpus: Path):
    sids = ["sess0000000000aa", "sess0000000000bb", "sess0000000000cc"]

    # Each session sets bin 30 to a different speed; baseline elsewhere is 60 m/s.
    # Bin 30: session A=40, B=60, C=80 → min=40, median(of per-session medians of
    # the bin)=60, max=80.
    #
    # Bin 50 holds a constant 70 m/s in all 3 sessions → min=median=max=70.
    _seed_session(tmp_corpus, sids[0], speed_at_bin={30: 40.0, 50: 70.0}, baseline_speed=60.0)
    _seed_session(tmp_corpus, sids[1], speed_at_bin={30: 60.0, 50: 70.0}, baseline_speed=60.0)
    _seed_session(tmp_corpus, sids[2], speed_at_bin={30: 80.0, 50: 70.0}, baseline_speed=60.0)

    model = build_track_model("synth_track", sids, corpus_root=tmp_corpus)
    assert model.regime == "compounding"

    env = model.speed_envelope
    expected_cols = {
        "bin_index",
        "track_pos_m",
        "speed_min_ms",
        "speed_median_ms",
        "speed_max_ms",
        "n_samples",
        "n_sessions",
    }
    assert expected_cols.issubset(set(env.columns))
    # One row per bin observed across the corpus (70 bins for the synthetic lap).
    assert env.height >= 70

    # Bin 30: cross-session envelope must capture all three speeds.
    bin30 = env.filter(pl.col("bin_index") == 30).row(0, named=True)
    assert bin30["speed_min_ms"] == 40.0
    assert bin30["speed_max_ms"] == 80.0
    # Median of per-session medians {40, 60, 80} == 60.
    assert bin30["speed_median_ms"] == 60.0
    # track_pos_m is the bin's left edge.
    assert bin30["track_pos_m"] == 30 * _BIN_SIZE

    # Bin 50: identical across all sessions, envelope collapses.
    bin50 = env.filter(pl.col("bin_index") == 50).row(0, named=True)
    assert bin50["speed_min_ms"] == 70.0
    assert bin50["speed_median_ms"] == 70.0
    assert bin50["speed_max_ms"] == 70.0

    # A baseline-only bin reflects the constant 60 m/s.
    bin10 = env.filter(pl.col("bin_index") == 10).row(0, named=True)
    assert bin10["speed_min_ms"] == 60.0
    assert bin10["speed_median_ms"] == 60.0
    assert bin10["speed_max_ms"] == 60.0


def test_speed_envelope_empty_in_cold_start(tmp_corpus: Path):
    """< 3 sessions → cold-start, speed_envelope is empty (consumers handle it)."""
    model = build_track_model("synth_track", ["sess0000000000aa"], corpus_root=tmp_corpus)
    assert model.regime == "cold_start"
    assert model.speed_envelope.height == 0
    # Schema is still well-formed so consumers can rely on column names.
    assert "speed_min_ms" in model.speed_envelope.columns
    assert "speed_median_ms" in model.speed_envelope.columns
    assert "speed_max_ms" in model.speed_envelope.columns


def test_speed_envelope_persists_through_cache(tmp_corpus: Path):
    """Re-building from cache must reproduce the envelope identically."""
    sids = ["sess0000000000aa", "sess0000000000bb", "sess0000000000cc"]
    for sid, speed in zip(sids, [55.0, 65.0, 75.0], strict=True):
        _seed_session(tmp_corpus, sid, speed_at_bin={30: speed}, baseline_speed=60.0)

    first = build_track_model("synth_track", sids, corpus_root=tmp_corpus)
    second = build_track_model("synth_track", sids, corpus_root=tmp_corpus)

    assert first.cache_path == second.cache_path
    assert first.speed_envelope.equals(second.speed_envelope)
