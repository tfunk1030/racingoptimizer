"""Per-car shock-velocity channel mapping (slice D / S1.3 gap-fill).

Acura ARX-06 IBT files lack `LFshockVel`/`RFshockVel`/`LRshockVel`/`RRshockVel`;
they expose two heave channels (`HFshockVel`, `TRshockVel`) and two roll channels
(`FROLLshockVel`, `RROLLshockVel`) instead. `track.masks.shock_vel_channels(car)`
must route Acura to the heave/roll set and every other car to the four-corner
default. The bug being closed: `track.builder._aggregate_one_session` previously
hardcoded the four-corner names, so an Acura lap raised
`pl.exceptions.ColumnNotFoundError` and the bare `except Exception: continue`
silently dropped every Acura lap — producing empty bump/grip maps despite a
compounding regime.

This file holds the unit-test contract: pure function dispatch + an end-to-end
synthetic Acura compounding build that proves the heave/roll channels reach
the bump map without raising.
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
from racingoptimizer.track import build_track_model
from racingoptimizer.track.masks import (
    DEFAULT_SHOCK_VEL_CHANNELS,
    shock_vel_channels,
)

_ACURA_SHOCK_VEL_CHANNELS = (
    "HFshockVel",
    "TRshockVel",
    "FROLLshockVel",
    "RROLLshockVel",
)


def test_shock_vel_channels_acura_is_heave_roll():
    """Acura's IBT exports two heave + two roll shock channels — no per-corner."""
    assert shock_vel_channels("acura") == _ACURA_SHOCK_VEL_CHANNELS


def test_shock_vel_channels_acura_case_insensitive():
    """Filename-derived car keys may be capitalized; helper must lowercase."""
    assert shock_vel_channels("Acura") == _ACURA_SHOCK_VEL_CHANNELS
    assert shock_vel_channels("ACURA") == _ACURA_SHOCK_VEL_CHANNELS


@pytest.mark.parametrize("car", ["bmw", "cadillac", "ferrari", "porsche"])
def test_shock_vel_channels_other_cars_use_four_corner_default(car: str):
    """BMW / Cadillac / Ferrari / Porsche all expose the four-corner standard."""
    assert shock_vel_channels(car) == DEFAULT_SHOCK_VEL_CHANNELS


def test_shock_vel_channels_unknown_car_falls_back_to_default():
    """Unknown cars must not raise; default is the safest assumption."""
    assert shock_vel_channels("synthetic") == DEFAULT_SHOCK_VEL_CHANNELS
    assert shock_vel_channels(None) == DEFAULT_SHOCK_VEL_CHANNELS


# ---- end-to-end Acura compounding-regime build ----

_LAP_LENGTH_M = 350.0
_BIN_SIZE = 5.0
_LAP_SAMPLES = 700  # 70 bins × 10 samples per bin


def _build_acura_lap_samples(*, curb_bin: int, bump_bin: int) -> dict:
    """Synthetic Acura lap. Spikes the heave-front shock at the curb / bump bins.

    The four-corner LF/RF/LR/RR channels are deliberately absent so any code
    path still reaching for them surfaces as ColumnNotFoundError.
    """
    n = _LAP_SAMPLES
    lap_dist_pct = np.linspace(0.0, 1.0 - 1e-9, n)
    track_pos_m = lap_dist_pct * _LAP_LENGTH_M
    bin_idx = np.floor(track_pos_m / _BIN_SIZE).astype(np.int32)

    # iRacing shockVel channels in m/s; slice-D scales ×1000 to mm/s thresholds.
    hf_shock = np.full(n, 0.05, dtype=np.float64)  # 50 mm/s baseline
    hf_shock = np.where(bin_idx == curb_bin, 0.6, hf_shock)  # 600 mm/s curb
    hf_shock = np.where(bin_idx == bump_bin, 0.25, hf_shock)  # 250 mm/s bump

    quiet = np.full(n, 0.05, dtype=np.float64)
    return {
        "t_s": np.arange(n, dtype=np.float64) / 60.0,
        "lap_index": np.zeros(n, dtype=np.int32),
        "lap_dist_pct": lap_dist_pct.astype(np.float64),
        "data_quality_mask": np.ones(n, dtype=bool),
        # Acura per-car shock-vel channels:
        "HFshockVel": hf_shock,
        "TRshockVel": quiet.copy(),
        "FROLLshockVel": quiet.copy(),
        "RROLLshockVel": quiet.copy(),
        "LatAccel": np.zeros(n, dtype=np.float64),
        "LFspeed": np.full(n, 60.0, dtype=np.float64),
        "RFspeed": np.full(n, 60.0, dtype=np.float64),
        "LRspeed": np.full(n, 60.0, dtype=np.float64),
        "RRspeed": np.full(n, 60.0, dtype=np.float64),
        "Speed": np.full(n, 60.0, dtype=np.float64),
    }


def _seed_acura_session(corpus_root: Path, sid: str, *, curb_bin: int, bump_bin: int) -> None:
    car = "acura"
    track = "synth_track"
    pq = parquet_path(corpus_root, car=car, track=track, session_id=sid)
    pq.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(_build_acura_lap_samples(curb_bin=curb_bin, bump_bin=bump_bin)).write_parquet(
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
                recorded_at="2026-04-29T00:00:00",
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


def test_acura_compounding_regime_produces_non_empty_bump_map(tmp_corpus: Path):
    """The bug repro: 3 Acura sessions → bump map must be non-empty.

    Pre-fix the bare `except Exception` swallowed `ColumnNotFoundError` from
    the four-corner channel request; every Acura lap was dropped and the
    builder produced an empty bump map despite a compounding regime.
    """
    sids = ["sess000000acura1", "sess000000acura2", "sess000000acura3"]
    for sid in sids:
        _seed_acura_session(tmp_corpus, sid, curb_bin=30, bump_bin=60)

    model = build_track_model("synth_track", sids, corpus_root=tmp_corpus)
    assert model.regime == "compounding"
    assert model.bump_map.height > 0, "Acura bump map empty — channel mapping broken"

    curb_row = model.bump_map.filter(pl.col("bin_index") == 30).row(0, named=True)
    assert curb_row["curb_likelihood"] >= 1.0 - 1e-9
    bump_row = model.bump_map.filter(pl.col("bin_index") == 60).row(0, named=True)
    assert bump_row["bump_likelihood"] > 0.0
