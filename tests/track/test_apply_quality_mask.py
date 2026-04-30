"""apply_quality_mask atomic in-place parquet rewrite (slice D-3, U8)."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from racingoptimizer.ingest import learn
from racingoptimizer.ingest.catalog import (
    LapRow,
    SessionRow,
    insert_laps,
    open_catalog,
    upsert_session,
)
from racingoptimizer.ingest.paths import catalog_path, parquet_path
from racingoptimizer.track import (
    ApplyMaskResult,
    apply_quality_mask,
    build_track_model,
)
from racingoptimizer.track.builder import _SUMMARY_SCHEMA

_LAP_LENGTH_M = 350.0
_BIN_SIZE = 5.0
_LAP_SAMPLES = 700
_CURB_BIN = 30


def _build_lap_samples(*, curb_bin: int | None) -> dict:
    n = _LAP_SAMPLES
    lap_dist_pct = np.linspace(0.0, 1.0 - 1e-9, n)
    track_pos_m = lap_dist_pct * _LAP_LENGTH_M
    bin_idx = np.floor(track_pos_m / _BIN_SIZE).astype(np.int32)

    # iRacing shockVel channels in m/s; slice-D scales ×1000 to mm/s thresholds.
    shock = np.full(n, 0.05, dtype=np.float64)  # 50 mm/s baseline
    if curb_bin is not None:
        shock = np.where(bin_idx == curb_bin, 0.6, shock)  # 600 mm/s curb

    return {
        "t_s": np.arange(n, dtype=np.float64) / 60.0,
        "lap_index": np.zeros(n, dtype=np.int32),
        "lap_dist_pct": lap_dist_pct.astype(np.float64),
        "data_quality_mask": np.ones(n, dtype=bool),
        "LFshockVel": shock,
        "RFshockVel": np.full(n, 0.05, dtype=np.float64),
        "LRshockVel": np.full(n, 0.05, dtype=np.float64),
        "RRshockVel": np.full(n, 0.05, dtype=np.float64),
        "LatAccel": np.zeros(n, dtype=np.float64),
        "LFspeed": np.full(n, 60.0, dtype=np.float64),
        "RFspeed": np.full(n, 60.0, dtype=np.float64),
        "LRspeed": np.full(n, 60.0, dtype=np.float64),
        "RRspeed": np.full(n, 60.0, dtype=np.float64),
        "Speed": np.full(n, 60.0, dtype=np.float64),
    }


def _seed_synthetic_session(
    corpus_root: Path,
    sid: str,
    *,
    car: str = "synthetic",
    track: str = "synth_track",
    curb_bin: int | None,
) -> Path:
    pq = parquet_path(corpus_root, car=car, track=track, session_id=sid)
    pq.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(_build_lap_samples(curb_bin=curb_bin)).write_parquet(
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
    return pq


def test_cold_start_noop(small_ibt: Path, tmp_corpus: Path):
    ids = learn(small_ibt, corpus_root=tmp_corpus)
    sid = ids[0]
    result = apply_quality_mask(sid, corpus_root=tmp_corpus)

    assert isinstance(result, ApplyMaskResult)
    assert result.regime == "cold_start"
    assert result.noop is True
    assert result.session_id == sid
    assert result.parquet_path.exists()

    df = pl.read_parquet(result.parquet_path)
    assert "data_quality_mask" in df.columns
    assert "data_quality_mask_v0" in df.columns
    assert df["data_quality_mask"].to_numpy().all()
    assert df["data_quality_mask_v0"].to_numpy().all()
    assert result.n_samples_clean_before == result.n_samples_total
    assert result.n_samples_clean_after == result.n_samples_total


def test_compounding_round_trip_masks_curb_samples(tmp_corpus: Path):
    sids = ["sess0000000000aa", "sess0000000000bb", "sess0000000000cc"]
    for sid in sids:
        _seed_synthetic_session(tmp_corpus, sid, curb_bin=_CURB_BIN)

    target = sids[0]
    pq_before = pl.read_parquet(_seed_target_path(tmp_corpus, target))
    assert pq_before["data_quality_mask"].to_numpy().all()

    result = apply_quality_mask(target, corpus_root=tmp_corpus)
    assert result.regime == "compounding"
    assert result.noop is False
    assert result.n_samples_clean_after < result.n_samples_clean_before

    df = pl.read_parquet(result.parquet_path)
    assert "data_quality_mask_v0" in df.columns
    v0 = df["data_quality_mask_v0"].to_numpy()
    assert v0.all(), "v0 snapshot should be the pre-rewrite all-True mask"

    new_mask = df["data_quality_mask"].to_numpy()
    assert not new_mask.all(), "compounding rewrite must mark at least one dirty sample"

    bin_idx = np.floor(df["lap_dist_pct"].to_numpy() * _LAP_LENGTH_M / _BIN_SIZE).astype(np.int32)
    in_curb_bin = bin_idx == _CURB_BIN
    assert in_curb_bin.any()
    assert not new_mask[in_curb_bin].any(), "curb-bin samples should be masked False"


def test_one_cycle_rollback(tmp_corpus: Path):
    sids = ["sess0000000000aa", "sess0000000000bb", "sess0000000000cc"]
    for sid in sids:
        _seed_synthetic_session(tmp_corpus, sid, curb_bin=_CURB_BIN)

    target = sids[0]
    apply_quality_mask(target, corpus_root=tmp_corpus)
    after_first = pl.read_parquet(_seed_target_path(tmp_corpus, target))
    first_mask = after_first["data_quality_mask"].to_numpy().copy()
    assert not first_mask.all()

    apply_quality_mask(target, corpus_root=tmp_corpus)
    after_second = pl.read_parquet(_seed_target_path(tmp_corpus, target))
    v0_after_second = after_second["data_quality_mask_v0"].to_numpy()

    # Per spec: _v0 reflects the result of the first run, NOT the original all-True baseline.
    np.testing.assert_array_equal(v0_after_second, first_mask)


def test_atomic_write_advances_mtime(tmp_corpus: Path):
    sids = ["sess0000000000aa", "sess0000000000bb", "sess0000000000cc"]
    for sid in sids:
        _seed_synthetic_session(tmp_corpus, sid, curb_bin=_CURB_BIN)

    target = sids[0]
    pq = _seed_target_path(tmp_corpus, target)
    mtime_before = pq.stat().st_mtime_ns
    # Force a measurable mtime delta on coarse Windows clocks.
    import time

    time.sleep(0.05)

    apply_quality_mask(target, corpus_root=tmp_corpus)
    mtime_after = pq.stat().st_mtime_ns
    assert mtime_after >= mtime_before
    assert mtime_after != mtime_before or pq.stat().st_size > 0


def test_atomic_failure_leaves_original_and_no_tmp(tmp_corpus: Path, monkeypatch):
    sids = ["sess0000000000aa", "sess0000000000bb", "sess0000000000cc"]
    for sid in sids:
        _seed_synthetic_session(tmp_corpus, sid, curb_bin=_CURB_BIN)

    target = sids[0]
    pq = _seed_target_path(tmp_corpus, target)
    bytes_before = pq.read_bytes()

    original_write = pl.DataFrame.write_parquet

    def boom(self, *args, **kwargs):  # noqa: ANN001
        raise OSError("disk full simulation")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", boom)
    with pytest.raises(OSError, match="disk full simulation"):
        apply_quality_mask(target, corpus_root=tmp_corpus)
    monkeypatch.setattr(pl.DataFrame, "write_parquet", original_write)

    assert pq.read_bytes() == bytes_before, "original parquet must be untouched on failure"
    leftover = list(pq.parent.glob(f"{pq.name}.tmp.*"))
    assert leftover == [], f"tmp file should be cleaned up, found: {leftover}"


def test_track_auto_build_path(small_ibt: Path, tmp_corpus: Path):
    ids = learn(small_ibt, corpus_root=tmp_corpus)
    sid = ids[0]
    result = apply_quality_mask(sid, track_model=None, corpus_root=tmp_corpus)
    assert result.regime == "cold_start"
    assert result.noop is True


def test_bytewise_determinism(tmp_corpus: Path):
    sids = ["sess0000000000aa", "sess0000000000bb", "sess0000000000cc"]
    for sid in sids:
        _seed_synthetic_session(tmp_corpus, sid, curb_bin=_CURB_BIN)

    target = sids[0]
    model = build_track_model("synth_track", sids, corpus_root=tmp_corpus)

    # Warm-up: after the first run, the mask is at its fixed-point. The next
    # two runs should produce byte-identical results (same mask becomes _v0,
    # same mask is recomputed).
    apply_quality_mask(target, track_model=model, corpus_root=tmp_corpus)

    apply_quality_mask(target, track_model=model, corpus_root=tmp_corpus)
    first_df = pl.read_parquet(_seed_target_path(tmp_corpus, target))

    apply_quality_mask(target, track_model=model, corpus_root=tmp_corpus)
    second_df = pl.read_parquet(_seed_target_path(tmp_corpus, target))

    assert_frame_equal(first_df, second_df)


def test_schema_preservation(tmp_corpus: Path):
    sids = ["sess0000000000aa", "sess0000000000bb", "sess0000000000cc"]
    for sid in sids:
        _seed_synthetic_session(tmp_corpus, sid, curb_bin=_CURB_BIN)

    target = sids[0]
    pq = _seed_target_path(tmp_corpus, target)

    before = pl.read_parquet(pq)
    before_columns = list(before.columns)
    before_dtypes = {c: before[c].dtype for c in before.columns}

    apply_quality_mask(target, corpus_root=tmp_corpus)
    after = pl.read_parquet(pq)

    assert after.columns[: len(before_columns)] == before_columns, "original column order preserved"
    assert "data_quality_mask_v0" in after.columns
    for col in before.columns:
        if col == "data_quality_mask":
            continue
        assert after[col].dtype == before_dtypes[col], f"dtype changed for {col}"
        np.testing.assert_array_equal(after[col].to_numpy(), before[col].to_numpy())
    assert after["data_quality_mask"].dtype == pl.Boolean
    assert after["data_quality_mask_v0"].dtype == pl.Boolean


def test_unknown_session_raises(tmp_corpus: Path):
    with pytest.raises(KeyError):
        apply_quality_mask("does_not_exist", corpus_root=tmp_corpus)


def test_missing_parquet_raises(tmp_corpus: Path):
    sid = "sess0000000000aa"
    pq = _seed_synthetic_session(tmp_corpus, sid, curb_bin=_CURB_BIN)
    pq.unlink()
    with pytest.raises(FileNotFoundError):
        apply_quality_mask(sid, corpus_root=tmp_corpus)


def test_synthetic_summary_curb_bin_is_masked(tmp_corpus: Path):
    """Smoke that synthetic injected curb_likelihood path also masks."""
    track = "fake_track_curbs"
    sids = ["a", "b", "c"]
    for sid in sids:
        _seed_synthetic_session(tmp_corpus, sid, track=track, curb_bin=_CURB_BIN)

    # Override the persisted summary so curb_likelihood at bin 30 == 1.0.
    model = build_track_model(track, sids, corpus_root=tmp_corpus)
    bins = np.arange(0, int(_LAP_LENGTH_M / _BIN_SIZE), dtype=np.int32)
    n = bins.size
    summary = pl.DataFrame(
        {
            "bin_index": bins,
            "track_pos_m": bins.astype(np.float64) * _BIN_SIZE,
            "shock_v_p99_mm_s": np.where(bins == _CURB_BIN, 600.0, 100.0),
            "lateral_g_p95": np.full(n, 0.8, dtype=np.float64),
            "lateral_g_median": np.full(n, 0.4, dtype=np.float64),
            "n_samples": np.full(n, 200, dtype=np.int64),
            "n_sessions": np.full(n, 3, dtype=np.int64),
            "curb_likelihood": np.where(bins == _CURB_BIN, 1.0, 0.0),
            "bump_likelihood": np.zeros(n, dtype=np.float64),
            "lap_length_m": np.full(n, _LAP_LENGTH_M, dtype=np.float64),
        },
        schema=_SUMMARY_SCHEMA,
    )
    summary.write_parquet(model.summary_path, compression="zstd")

    target = sids[0]
    result = apply_quality_mask(target, corpus_root=tmp_corpus)
    assert result.regime == "compounding"
    assert result.n_samples_clean_after < result.n_samples_clean_before


def _seed_target_path(corpus_root: Path, sid: str) -> Path:
    return parquet_path(corpus_root, car="synthetic", track="synth_track", session_id=sid)
