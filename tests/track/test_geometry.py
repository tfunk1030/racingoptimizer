"""Per-bin elevation / camber proxies (S4.4, VISION §9).

Synthetic 3-session corpus seeded directly into slice A's catalog +
parquet so we can drive realistic accelerometer / steering signals
without needing real IBT files. Each session is one lap with a
constructed hill at a known bin (elevation case) and a banking-style
lateral-G overshoot at a known bin (camber case).
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from racingoptimizer.corner.config import G_MS2
from racingoptimizer.ingest.catalog import (
    LapRow,
    SessionRow,
    insert_laps,
    open_catalog,
    upsert_session,
)
from racingoptimizer.ingest.paths import catalog_path, parquet_path
from racingoptimizer.track import build_track_model, compute_track_geometry
from racingoptimizer.track.geometry import GEOMETRY_SCHEMA

_TRACK = "synth_geo_track"
_LAP_LENGTH_M = 350.0
_BIN_SIZE = 5.0
_LAP_SAMPLES = 700  # 70 bins × 10 samples


def _build_lap(
    *,
    hill_bin: int | None,
    banking_bin: int | None,
    drag_slope: float = -0.02,
) -> dict[str, np.ndarray]:
    """Construct a synthetic lap with optional hill and banking signatures.

    - Speed climbs linearly from 30 -> 70 m/s across the lap (so the
      corpus-wide LongAccel ~ Speed fit has variance to lock onto).
    - LongAccel is the speed-linear baseline ``drag_slope * Speed``
      everywhere, plus a +6 m/s² spike at ``hill_bin`` (a steep crest).
    - Steering ramps in / out of one corner spanning bins 20-40 so
      mid-corner samples populate a contiguous range.
    - Lat-G follows ``a * |steering|`` with a *banking overshoot* at
      ``banking_bin`` (LatAccel scaled by 1.5×).
    """
    n = _LAP_SAMPLES
    lap_dist_pct = np.linspace(0.0, 1.0 - 1e-9, n)
    track_pos_m = lap_dist_pct * _LAP_LENGTH_M
    bin_idx = np.floor(track_pos_m / _BIN_SIZE).astype(np.int32)

    speed = np.linspace(30.0, 70.0, n)
    long_accel = drag_slope * speed
    if hill_bin is not None:
        long_accel = np.where(bin_idx == hill_bin, long_accel + 6.0, long_accel)

    # One sustained corner from bin 20 .. 40. Steering held at a high value
    # across the whole corner so every in-corner sample clears the
    # `lat_g_entry = 0.5g` detector threshold AND gives the corpus-wide
    # `|lat_g| ~ |steer|` fit a wide steering range to lock onto. We layer
    # in straight-line samples with steer = 0 outside the corner so the fit
    # has both ends of the line.
    corner_mask = (bin_idx >= 20) & (bin_idx <= 40)
    # Steering ramps 0.10 -> 0.30 across the corner so bins are
    # individually distinguishable in the fit (a flat steering value
    # would let the corpus regression pick any slope).
    steer_in = np.linspace(0.10, 0.30, int(corner_mask.sum()), dtype=np.float64)
    steer = np.zeros(n, dtype=np.float64)
    steer[corner_mask] = steer_in

    # Lat-G = 6 * |steer|: at steer = 0.10 -> 0.6g (just above the 0.5g
    # entry threshold), at steer = 0.30 -> 1.8g (peak corner load).
    # The banking_bin gets a 1.5× overshoot.
    lat_g_unitless = 6.0 * steer
    if banking_bin is not None:
        lat_g_unitless = np.where(
            bin_idx == banking_bin, lat_g_unitless * 1.5, lat_g_unitless
        )
    lat_accel = lat_g_unitless * G_MS2

    # Brake / Throttle synthesised so the phase machine reaches MID_CORNER
    # well inside the corner. The state walker is BRAKING -> TRAIL_BRAKE
    # (needs brake > 0 + steering active + lat-G > 0.3g) -> MID_CORNER
    # (needs brake < 0.02 for 50 ms), so we brake during bins 19-22 with
    # the corner steering already engaged, then release brake from bin 23
    # onwards. Throttle stays low through mid-corner so the EXIT
    # transition (needs throttle > 0.10 + decreasing lat-G) does not fire
    # until late.
    brake = np.where((bin_idx >= 19) & (bin_idx <= 22), 0.6, 0.0).astype(np.float64)
    throttle = np.where(bin_idx >= 38, 0.8, 0.0).astype(np.float64)

    return {
        "t_s": np.arange(n, dtype=np.float64) / 60.0,
        "lap_index": np.zeros(n, dtype=np.int32),
        "lap_dist_pct": lap_dist_pct.astype(np.float64),
        "data_quality_mask": np.ones(n, dtype=bool),
        "Speed": speed,
        "LongAccel": long_accel,
        "LatAccel": lat_accel,
        "Brake": brake,
        "Throttle": throttle,
        "SteeringWheelAngle": steer,
        # Required by other slice-D code paths the catalog might trigger.
        "LFshockVel": np.full(n, 0.05, dtype=np.float64),
        "RFshockVel": np.full(n, 0.05, dtype=np.float64),
        "LRshockVel": np.full(n, 0.05, dtype=np.float64),
        "RRshockVel": np.full(n, 0.05, dtype=np.float64),
        "LFspeed": speed,
        "RFspeed": speed,
        "LRspeed": speed,
        "RRspeed": speed,
    }


def _seed_session(
    corpus_root: Path,
    sid: str,
    *,
    hill_bin: int | None,
    banking_bin: int | None,
) -> None:
    car = "synthetic"
    pq = parquet_path(corpus_root, car=car, track=_TRACK, session_id=sid)
    pq.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(_build_lap(hill_bin=hill_bin, banking_bin=banking_bin)).write_parquet(
        pq, compression="zstd"
    )

    setup = {"WeekendInfo": {"TrackLength": f"{_LAP_LENGTH_M / 1000:.4f} km"}}
    with open_catalog(catalog_path(corpus_root)) as conn:
        upsert_session(
            conn,
            SessionRow(
                session_id=sid,
                car=car,
                track=_TRACK,
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


def test_geometry_returns_expected_schema(tmp_corpus: Path):
    sids = ["geo_sess0000aa", "geo_sess0000bb", "geo_sess0000cc"]
    for sid in sids:
        _seed_session(tmp_corpus, sid, hill_bin=50, banking_bin=30)

    geom = compute_track_geometry(_TRACK, sids, corpus_root=tmp_corpus)
    assert set(geom.columns) == set(GEOMETRY_SCHEMA.keys())
    for col, dtype in GEOMETRY_SCHEMA.items():
        assert geom.schema[col] == dtype, f"{col} got {geom.schema[col]} expected {dtype}"


def test_elevation_proxy_peaks_at_constructed_hill(tmp_corpus: Path):
    """Synthetic LongAccel = drag * Speed everywhere except a +6 m/s² spike
    at bin 50. The corpus-wide linear fit nails the drag baseline; the
    residual is huge at bin 50 and ~zero everywhere else.
    """
    sids = ["geo_sess0000aa", "geo_sess0000bb", "geo_sess0000cc"]
    hill_bin = 50
    for sid in sids:
        _seed_session(tmp_corpus, sid, hill_bin=hill_bin, banking_bin=None)

    geom = compute_track_geometry(_TRACK, sids, corpus_root=tmp_corpus)
    assert geom.height > 0

    hill_row = geom.filter(pl.col("bin_index") == hill_bin)
    assert hill_row.height == 1
    hill_value = hill_row["elevation_gradient_proxy"][0]
    assert hill_value is not None

    other_values = (
        geom.filter(pl.col("bin_index") != hill_bin)["elevation_gradient_proxy"]
        .drop_nulls()
        .to_numpy()
    )
    assert other_values.size > 0
    # Hill spike is +6 m/s² above the baseline; baseline residual is ≈0.
    assert hill_value > 4.0, f"expected hill bin to show large residual, got {hill_value}"
    assert hill_value > 5.0 * float(np.median(other_values))


def test_camber_ratio_exceeds_one_at_banking_bin(tmp_corpus: Path):
    """Synthetic lat-G is 1.5× the steering-prediction at the banking bin.
    The camber_ratio_proxy should land near 1.5 at that bin and near 1.0
    at the other mid-corner bins.
    """
    sids = ["geo_sess0000aa", "geo_sess0000bb", "geo_sess0000cc"]
    banking_bin = 30
    for sid in sids:
        _seed_session(tmp_corpus, sid, hill_bin=None, banking_bin=banking_bin)

    geom = compute_track_geometry(_TRACK, sids, corpus_root=tmp_corpus)
    banking_rows = geom.filter(pl.col("bin_index") == banking_bin)
    assert banking_rows.height == 1
    banking_ratio = banking_rows["camber_ratio_proxy"][0]
    assert banking_ratio is not None
    assert banking_ratio > 1.2, (
        f"expected banking_bin to exceed 1.0 ratio, got {banking_ratio}"
    )

    # Other mid-corner bins (20-40 minus banking_bin) should sit near 1.0.
    neutral = (
        geom.filter(
            (pl.col("bin_index").is_between(20, 40))
            & (pl.col("bin_index") != banking_bin)
        )["camber_ratio_proxy"]
        .drop_nulls()
        .to_numpy()
    )
    assert neutral.size > 0
    # Linear fit through banking + neutral samples leaves the neutral
    # samples near 1.0 (within ±0.3 — the banking sample shifts the fit
    # slope slightly but cannot flip the neutral regime above the
    # banking value).
    assert float(np.median(neutral)) < banking_ratio


def test_geometry_cold_start_returns_empty(tmp_corpus: Path):
    """Fewer than 3 sessions → empty frame, schema preserved."""
    sids = ["geo_sess0000aa", "geo_sess0000bb"]
    for sid in sids:
        _seed_session(tmp_corpus, sid, hill_bin=50, banking_bin=30)

    geom = compute_track_geometry(_TRACK, sids, corpus_root=tmp_corpus)
    assert geom.height == 0
    assert set(geom.columns) == set(GEOMETRY_SCHEMA.keys())


def test_track_model_geometry_property_is_lazy_and_cached(tmp_corpus: Path):
    sids = ["geo_sess0000aa", "geo_sess0000bb", "geo_sess0000cc"]
    for sid in sids:
        _seed_session(tmp_corpus, sid, hill_bin=50, banking_bin=30)

    model = build_track_model(_TRACK, sids, corpus_root=tmp_corpus)
    first = model.geometry
    second = model.geometry
    # Cache must return the same object (not just equal) on the second access.
    assert first is second
    assert first.height > 0


def test_track_model_geometry_cold_start_returns_empty(tmp_corpus: Path):
    model = build_track_model(_TRACK, ["only_one"], corpus_root=tmp_corpus)
    assert model.regime == "cold_start"
    geom = model.geometry
    assert geom.height == 0
    assert set(geom.columns) == set(GEOMETRY_SCHEMA.keys())
