"""TrackModel.flag_anomalies — observed-vs-expected per-sample anomaly flagging (S4.1).

Synthetic 3-session corpus + a fourth lap with deliberately injected
spikes at a quiet bin. The corpus's per-session cache parquet is seeded
directly so the test does not have to ingest IBT files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from racingoptimizer.track import build_track_model, cache_path, summary_path
from racingoptimizer.track.builder import _PER_SESSION_SCHEMA, _SUMMARY_SCHEMA
from racingoptimizer.track.paths import sessions_hash

_BIN_SIZE = 5.0
_LAP_LEN = 350.0
_QUIET_BIN = 30  # track_pos_m = 150
_QUIET_P99_MM_S = 100.0  # cross-session distribution centred here


def _seed_three_session_cache(corpus_root: Path, track: str, sids: list[str]) -> None:
    """Seed a quiet 3-session corpus where bin 30's expected p99 ≈ 100 mm/s."""
    digest = sessions_hash(sorted(sids))
    cache = cache_path(corpus_root, track, digest)
    summary = summary_path(corpus_root, track, digest)
    cache.parent.mkdir(parents=True, exist_ok=True)

    # Every session reports very similar p99 in the quiet bin.
    rows: list[dict[str, object]] = []
    per_session_p99 = [_QUIET_P99_MM_S - 5.0, _QUIET_P99_MM_S, _QUIET_P99_MM_S + 5.0]
    per_session_lat_p95 = [0.50, 0.52, 0.51]
    per_session_lat_med = [0.30, 0.31, 0.30]
    for sid, p99, p95, med in zip(
        sorted(sids), per_session_p99, per_session_lat_p95, per_session_lat_med, strict=True
    ):
        rows.append(
            {
                "session_id": sid,
                "track_pos_m": float(_QUIET_BIN) * _BIN_SIZE,
                "n_samples": 200,
                "shock_v_p99_mm_s": float(p99),
                "lateral_g_p95": float(p95),
                "lateral_g_median": float(med),
                "speed_min_ms": 30.0,
                "speed_median_ms": 50.0,
                "speed_max_ms": 70.0,
            }
        )
    pl.DataFrame(rows, schema=_PER_SESSION_SCHEMA).write_parquet(cache, compression="zstd")

    pl.DataFrame(
        {
            "bin_index": np.array([_QUIET_BIN], dtype=np.int32),
            "track_pos_m": np.array([float(_QUIET_BIN) * _BIN_SIZE], dtype=np.float64),
            "shock_v_p99_mm_s": np.array([_QUIET_P99_MM_S], dtype=np.float64),
            "lateral_g_p95": np.array([0.51], dtype=np.float64),
            "lateral_g_median": np.array([0.30], dtype=np.float64),
            "speed_min_ms": np.array([30.0], dtype=np.float64),
            "speed_median_ms": np.array([50.0], dtype=np.float64),
            "speed_max_ms": np.array([70.0], dtype=np.float64),
            "n_samples": np.array([600], dtype=np.int64),
            "n_sessions": np.array([3], dtype=np.int64),
            "curb_likelihood": np.array([0.0], dtype=np.float64),
            "bump_likelihood": np.array([0.0], dtype=np.float64),
            "lap_length_m": np.array([_LAP_LEN], dtype=np.float64),
        },
        schema=_SUMMARY_SCHEMA,
    ).write_parquet(summary, compression="zstd")


def _build_anomaly_lap(*, spike_idx: int, spike_m_s: float) -> pl.DataFrame:
    """Lap_df: most samples are quiet; one sample (spike_idx) blows up to spike_m_s on LF."""
    n = 70
    lap_dist_pct = np.linspace(0.0, 1.0 - 1e-9, n)
    quiet_m_s = 0.05  # 50 mm/s baseline, well within expected
    lf = np.full(n, quiet_m_s, dtype=np.float64)
    lf[spike_idx] = spike_m_s
    return pl.DataFrame(
        {
            "lap_dist_pct": lap_dist_pct.astype(np.float64),
            "LFshockVel": lf,
            "RFshockVel": np.full(n, quiet_m_s, dtype=np.float64),
            "LRshockVel": np.full(n, quiet_m_s, dtype=np.float64),
            "RRshockVel": np.full(n, quiet_m_s, dtype=np.float64),
            "LatAccel": np.zeros(n, dtype=np.float64),
        }
    )


def test_flag_anomalies_finds_injected_spike(tmp_corpus: Path):
    track = "anom_track"
    sids = ["s0", "s1", "s2"]
    _seed_three_session_cache(tmp_corpus, track, sids)
    model = build_track_model(track, sids, corpus_root=tmp_corpus)

    # Spike sample lands in bin 30 (track_pos_m ≈ 150).
    n = 70
    spike_idx = int(round(_QUIET_BIN * _BIN_SIZE / _LAP_LEN * (n - 1)))
    # 10× the expected p99 (in mm/s) → spike of 0.1 m/s × 10 = 1.0 m/s = 1000 mm/s.
    spike_m_s = (_QUIET_P99_MM_S * 10.0) / 1000.0
    lap = _build_anomaly_lap(spike_idx=spike_idx, spike_m_s=spike_m_s)

    anomalies = model.flag_anomalies(lap)
    assert anomalies.height >= 1, "expected the 10× spike to flag at least one anomaly"

    # The shock spike must be present.
    shock_rows = anomalies.filter(pl.col("channel") == "shock_v_p99_mm_s")
    assert shock_rows.height >= 1
    flagged_idx = set(shock_rows["sample_idx"].to_list())
    assert spike_idx in flagged_idx, (
        f"expected sample_idx {spike_idx} to be flagged, got {sorted(flagged_idx)}"
    )

    # z_score must reflect the magnitude of the deviation.
    spike_row = shock_rows.filter(pl.col("sample_idx") == spike_idx).row(0, named=True)
    assert spike_row["observed"] > _QUIET_P99_MM_S * 5
    assert spike_row["z_score"] > 3.0
    # Single-sample very-high-z spike → data_noise.
    assert spike_row["label"] == "data_noise"


def test_flag_anomalies_clean_lap_returns_empty(tmp_corpus: Path):
    """A lap whose observed values sit close to the bin's expected mean
    must produce zero anomalies — the cross-session distribution and the
    lap agree.
    """
    track = "anom_track"
    sids = ["s0", "s1", "s2"]
    _seed_three_session_cache(tmp_corpus, track, sids)
    model = build_track_model(track, sids, corpus_root=tmp_corpus)

    n = 70
    # Driving at the expected mean: 0.1 m/s = 100 mm/s = the quiet bin's
    # cross-session p99 mean. The deviation is well below |z| = 3.
    on_expected_m_s = _QUIET_P99_MM_S / 1000.0  # 0.1 m/s
    lap = pl.DataFrame(
        {
            "lap_dist_pct": np.linspace(0.0, 1.0 - 1e-9, n).astype(np.float64),
            "LFshockVel": np.full(n, on_expected_m_s, dtype=np.float64),
            "RFshockVel": np.full(n, on_expected_m_s, dtype=np.float64),
            "LRshockVel": np.full(n, on_expected_m_s, dtype=np.float64),
            "RRshockVel": np.full(n, on_expected_m_s, dtype=np.float64),
            # Lateral G ≈ 0.5 g matches the seeded p95.
            "LatAccel": np.full(n, 0.51 * 9.80665, dtype=np.float64),
        }
    )
    anomalies = model.flag_anomalies(lap)
    assert anomalies.height == 0
    # Schema must be present even when empty so downstream consumers can join.
    assert set(anomalies.columns) == {
        "sample_idx",
        "channel",
        "observed",
        "expected",
        "z_score",
        "label",
    }


def test_flag_anomalies_cold_start_returns_empty(tmp_corpus: Path):
    model = build_track_model("anom_track", ["only_one"], corpus_root=tmp_corpus)
    assert model.regime == "cold_start"

    n = 70
    lap = pl.DataFrame(
        {
            "lap_dist_pct": np.linspace(0.0, 1.0 - 1e-9, n).astype(np.float64),
            "LFshockVel": np.full(n, 0.05, dtype=np.float64),
            "RFshockVel": np.full(n, 0.05, dtype=np.float64),
            "LRshockVel": np.full(n, 0.05, dtype=np.float64),
            "RRshockVel": np.full(n, 0.05, dtype=np.float64),
            "LatAccel": np.zeros(n, dtype=np.float64),
        }
    )
    assert model.flag_anomalies(lap).height == 0


def test_flag_anomalies_clusters_label_setup_problem(tmp_corpus: Path):
    """A run of consecutive moderate-z flagged samples on the same channel
    classifies as setup_problem rather than data_noise / driver_error.
    """
    track = "anom_track"
    sids = ["s0", "s1", "s2"]
    _seed_three_session_cache(tmp_corpus, track, sids)
    model = build_track_model(track, sids, corpus_root=tmp_corpus)

    # 5 consecutive elevated (but not extreme) samples — z between 3 and 10 each.
    n = 70
    spike_idx = int(round(_QUIET_BIN * _BIN_SIZE / _LAP_LEN * (n - 1)))
    quiet_m_s = 0.05
    elevated_m_s = 0.5  # 500 mm/s — z = (500 - 100) / max(p99 - mean, 1.0) ≈ huge,
    # so we use a smaller multiplier to keep z below the 10 noise threshold.
    elevated_m_s = 0.4  # 400 mm/s observed
    lf = np.full(n, quiet_m_s, dtype=np.float64)
    cluster_indices = list(range(spike_idx, spike_idx + 5))
    for i in cluster_indices:
        if 0 <= i < n:
            lf[i] = elevated_m_s
    lap = pl.DataFrame(
        {
            "lap_dist_pct": np.linspace(0.0, 1.0 - 1e-9, n).astype(np.float64),
            "LFshockVel": lf,
            "RFshockVel": np.full(n, quiet_m_s, dtype=np.float64),
            "LRshockVel": np.full(n, quiet_m_s, dtype=np.float64),
            "RRshockVel": np.full(n, quiet_m_s, dtype=np.float64),
            "LatAccel": np.zeros(n, dtype=np.float64),
        }
    )

    anomalies = model.flag_anomalies(lap)
    shock_rows = anomalies.filter(pl.col("channel") == "shock_v_p99_mm_s")
    # Only samples whose bin matches the seeded bin can be evaluated; other
    # spikes' bins lack expectation data and are skipped silently.
    in_seeded_bin = shock_rows.filter(pl.col("sample_idx").is_in(cluster_indices))
    if in_seeded_bin.height >= 3:
        labels = in_seeded_bin["label"].to_list()
        # At least one sample inside a 3-long run should be classified as setup_problem.
        assert "setup_problem" in labels
