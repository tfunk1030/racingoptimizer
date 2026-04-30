"""Per-corner braking / apex / exit landmarks (S4.2, VISION §9).

Synthetic 3-session corpus with one known corner per lap:
- Braking begins at track_pos_m ≈ 50 m (Brake jumps to 1.0 there).
- Apex sits at ≈ 100 m (peak |LatAccel| ≈ 1.5 g).
- Exit completes at ≈ 150 m (Throttle returns to 1.0, |LatAccel| → 0).

Each session shifts those positions by ±5 m so the cross-lap mean is the
nominal value but no two sessions land on identical samples — exercises the
averaging codepath the way real per-lap noise would.
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
from racingoptimizer.track import build_track_model, compute_corner_landmarks

_LAP_LENGTH_M = 350.0
_LAP_SAMPLES = 700  # 60 Hz × ~11.6 s synthetic lap

_NOMINAL_BRAKING_M = 50.0
_NOMINAL_APEX_M = 100.0
_NOMINAL_EXIT_M = 150.0
_PEAK_LAT_ACCEL_MS2 = 14.7  # ≈ 1.5 g — well above the 0.5 g entry threshold


def _build_lap_samples(*, position_offset_m: float) -> dict[str, np.ndarray]:
    """Build one lap's channel dict with the corner shifted by `position_offset_m`.

    The lateral-G profile is a triangular ramp peaking at `apex_m + offset`,
    centred over a 100 m corner window. Brake is on for the first 30 m of the
    window (the braking zone); throttle is off through trail-brake / mid-corner
    and rises to 1.0 once lateral load relaxes.
    """
    n = _LAP_SAMPLES
    lap_dist_pct = np.linspace(0.0, 1.0 - 1e-9, n)
    track_pos = lap_dist_pct * _LAP_LENGTH_M

    braking_m = _NOMINAL_BRAKING_M + position_offset_m
    apex_m = _NOMINAL_APEX_M + position_offset_m
    exit_m = _NOMINAL_EXIT_M + position_offset_m

    # Triangular |LatAccel| profile centred on apex, ramping from 0 at
    # `braking_m` (corner entry) to peak at `apex_m`, back down to 0 at
    # `exit_m`. Sign chosen positive arbitrarily — the detector uses |·|.
    half_width = apex_m - braking_m  # = 50 m for the nominal geometry
    distance_from_apex = np.abs(track_pos - apex_m)
    inside_corner = distance_from_apex <= half_width
    triangle = np.where(
        inside_corner,
        _PEAK_LAT_ACCEL_MS2 * (1.0 - distance_from_apex / half_width),
        0.0,
    )
    lat_accel = triangle  # positive lobe

    brake = np.where(
        (track_pos >= braking_m) & (track_pos < braking_m + 30.0), 1.0, 0.0
    )
    throttle = np.where(track_pos >= exit_m, 1.0, 0.0)

    shock = np.full(n, 0.05, dtype=np.float64)
    speed = np.full(n, 60.0, dtype=np.float64)
    return {
        "t_s": np.arange(n, dtype=np.float64) / 60.0,
        "lap_index": np.zeros(n, dtype=np.int32),
        "lap_dist_pct": lap_dist_pct.astype(np.float64),
        "data_quality_mask": np.ones(n, dtype=bool),
        "LFshockVel": shock,
        "RFshockVel": shock,
        "LRshockVel": shock,
        "RRshockVel": shock,
        "LatAccel": lat_accel.astype(np.float64),
        "Brake": brake.astype(np.float64),
        "Throttle": throttle.astype(np.float64),
        "LFspeed": speed,
        "RFspeed": speed,
        "LRspeed": speed,
        "RRspeed": speed,
        "Speed": speed,
    }


def _seed_session(corpus_root: Path, sid: str, *, position_offset_m: float) -> None:
    car = "synthetic"
    track = "synth_track"
    pq = parquet_path(corpus_root, car=car, track=track, session_id=sid)
    pq.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(_build_lap_samples(position_offset_m=position_offset_m)).write_parquet(
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


def test_corner_landmarks_recover_known_synthetic_geometry(tmp_corpus: Path):
    sids = ["sess0000000000aa", "sess0000000000bb", "sess0000000000cc"]
    # Three sessions vary the corner position by ±5 m around the nominal
    # geometry. The cross-session mean must equal the nominal positions.
    _seed_session(tmp_corpus, sids[0], position_offset_m=-5.0)
    _seed_session(tmp_corpus, sids[1], position_offset_m=0.0)
    _seed_session(tmp_corpus, sids[2], position_offset_m=5.0)

    landmarks = compute_corner_landmarks("synth_track", sids, corpus_root=tmp_corpus)
    expected_columns = {
        "corner_id",
        "braking_point_m",
        "apex_m",
        "exit_point_m",
        "n_observations",
    }
    assert set(landmarks.columns) == expected_columns
    assert landmarks.height == 1, "synthetic lap geometry only models one corner"

    row = landmarks.row(0, named=True)
    assert row["corner_id"] == 0
    assert row["n_observations"] == 3
    assert abs(row["braking_point_m"] - _NOMINAL_BRAKING_M) <= 10.0
    assert abs(row["apex_m"] - _NOMINAL_APEX_M) <= 10.0
    assert abs(row["exit_point_m"] - _NOMINAL_EXIT_M) <= 10.0


def test_corner_landmarks_via_track_model_property(tmp_corpus: Path):
    """`TrackModel.corner_landmarks` lazy property delegates to the same fn."""
    sids = ["sess0000000000aa", "sess0000000000bb", "sess0000000000cc"]
    for sid in sids:
        _seed_session(tmp_corpus, sid, position_offset_m=0.0)

    model = build_track_model("synth_track", sids, corpus_root=tmp_corpus)
    assert model.regime == "compounding"

    landmarks = model.corner_landmarks
    direct = compute_corner_landmarks("synth_track", sids, corpus_root=tmp_corpus)
    assert landmarks.equals(direct)


def test_corner_landmarks_empty_in_cold_start(tmp_corpus: Path):
    """< 3 sessions ⇒ cold-start ⇒ TrackModel returns an empty landmarks frame."""
    _seed_session(tmp_corpus, "sess0000000000aa", position_offset_m=0.0)
    model = build_track_model(
        "synth_track", ["sess0000000000aa"], corpus_root=tmp_corpus
    )
    assert model.regime == "cold_start"
    assert model.corner_landmarks.height == 0
    # Schema is still well-formed so consumers can rely on column names.
    assert set(model.corner_landmarks.columns) == {
        "corner_id",
        "braking_point_m",
        "apex_m",
        "exit_point_m",
        "n_observations",
    }
