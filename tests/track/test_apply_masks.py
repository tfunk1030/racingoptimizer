"""TrackModel.curb_mask / off_track_mask end-to-end (U7)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from racingoptimizer.track import build_track_model
from racingoptimizer.track.builder import _SUMMARY_SCHEMA


def _synthetic_lap(n: int = 70) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "lap_dist_pct": np.linspace(0.0, 1.0 - 1e-9, n),
            "AccelLat": np.zeros(n),
            "LFshockVel": np.zeros(n),
            "RFshockVel": np.zeros(n),
            "LRshockVel": np.zeros(n),
            "RRshockVel": np.zeros(n),
            "LFspeed": np.full(n, 50.0),
            "RFspeed": np.full(n, 50.0),
            "LRspeed": np.full(n, 50.0),
            "RRspeed": np.full(n, 50.0),
        }
    )


def test_cold_start_curb_mask_all_false(tmp_corpus: Path):
    model = build_track_model("fake_track", ["x"], corpus_root=tmp_corpus)
    assert model.regime == "cold_start"
    df = _synthetic_lap()
    out = model.curb_mask(df)
    assert out.shape == (df.height,)
    assert out.dtype == np.bool_
    assert not out.any()


def test_cold_start_off_track_mask_all_false(tmp_corpus: Path):
    model = build_track_model("fake_track", ["x"], corpus_root=tmp_corpus)
    df = _synthetic_lap()
    out = model.off_track_mask(df)
    assert out.shape == (df.height,)
    assert out.dtype == np.bool_
    assert not out.any()


def test_compounding_curb_mask_flags_injected_bin(tmp_corpus: Path):
    """Build a synthetic compounding model by writing a summary parquet directly,
    then reload via the cache hit path."""
    track = "fake_track_curbs"
    sids = ["a", "b", "c"]
    model = build_track_model(track, sids, corpus_root=tmp_corpus)
    # First call short-circuits to cold-start (no real session data); for the
    # compounding regime to load via cache we override the persisted summary.
    lap_length_m = 350.0
    bin_size_m = 5.0
    bins = np.arange(0, int(lap_length_m / bin_size_m), dtype=np.int32)
    n = bins.size
    summary = pl.DataFrame(
        {
            "bin_index": bins,
            "track_pos_m": bins.astype(np.float64) * bin_size_m,
            "shock_v_p99_mm_s": np.where(bins == 30, 600.0, 100.0),
            "lateral_g_p95": np.full(n, 0.8, dtype=np.float64),
            "lateral_g_median": np.full(n, 0.4, dtype=np.float64),
            "n_samples": np.full(n, 200, dtype=np.int64),
            "n_sessions": np.full(n, 3, dtype=np.int64),
            "curb_likelihood": np.where(bins == 30, 1.0, 0.0),
            "bump_likelihood": np.zeros(n, dtype=np.float64),
            "lap_length_m": np.full(n, lap_length_m, dtype=np.float64),
        },
        schema=_SUMMARY_SCHEMA,
    )
    summary.write_parquet(model.summary_path, compression="zstd")
    # Touch the per-session cache mtime back so cache_path stays valid.
    # Force the second build to take the cache-hit branch — but with 3 sids it
    # would be compounding regime.
    model2 = build_track_model(track, sids, corpus_root=tmp_corpus)
    assert model2.regime == "compounding"
    assert model2.bump_map.height == n

    df = _synthetic_lap(n=70)
    mask = model2.curb_mask(df)
    assert mask.shape == (df.height,)
    assert mask.dtype == np.bool_
    # Bin 30 corresponds to track_pos_m 150–155. With lap_length_m = 350 and
    # 70 samples linearly walking lap_dist_pct, sample 30 is at pct ≈ 30/69 ≈ 0.4348,
    # track_pos_m ≈ 152.2 → bin 30.
    assert mask[30]
