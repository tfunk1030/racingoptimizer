"""S2.8 — `_environment_from_corpus` pulls per-sample medians from parquet.

VISION §10 + master-plan rule: every sample carries env context, never
collapse to session averages. We synthesize a corpus with two sessions whose
per-session means differ from the per-sample median, and assert the
recommender's env feature uses the per-sample median.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from racingoptimizer.cli.recommend import (
    _ENV_DEFAULTS,
    _circular_median_deg,
    _environment_from_corpus,
)
from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.catalog import LapRow, SessionRow
from racingoptimizer.ingest.paths import catalog_path, parquet_path


def _synthesize_session(
    *,
    corpus_root: Path,
    session_id: str,
    car: str,
    track: str,
    air_density_samples: np.ndarray,
    track_temp_samples: np.ndarray,
    wind_vel_samples: np.ndarray,
    wind_dir_samples: np.ndarray,
    track_wetness_samples: np.ndarray,
    quality_mask: np.ndarray | None = None,
) -> None:
    """Write one fake parquet + catalog row covering a single valid lap."""
    n = int(air_density_samples.size)
    if quality_mask is None:
        quality_mask = np.ones(n, dtype=bool)

    df = pl.DataFrame(
        {
            "t_s": (np.arange(n, dtype=np.float64) / 60.0),
            "lap_index": np.zeros(n, dtype=np.int32),
            "lap_dist_pct": np.linspace(0.0, 1.0, n, dtype=np.float64),
            "data_quality_mask": quality_mask,
            "AirDensity": air_density_samples.astype(np.float64),
            "TrackTempCrew": track_temp_samples.astype(np.float64),
            "WindVel": wind_vel_samples.astype(np.float64),
            "WindDir": wind_dir_samples.astype(np.float64),
            "TrackWetness": track_wetness_samples.astype(np.float64),
        }
    )
    pq = parquet_path(corpus_root, car=car, track=track, session_id=session_id)
    pq.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(pq, compression="zstd")

    with cat.open_catalog(catalog_path(corpus_root)) as conn:
        cat.upsert_session(
            conn,
            SessionRow(
                session_id=session_id,
                car=car,
                track=track,
                recorded_at=None,
                duration_s=float(n) / 60.0,
                lap_count=1,
                weather_summary=None,
                setup=None,
                source_path=f"<synthetic:{session_id}>",
                ingested_at="2026-04-29T00:00:00",
                parquet_path=str(pq.relative_to(corpus_root).as_posix()),
                status="ok",
                error=None,
                dropped_channels=None,
                sample_rate_hz=60.0,
            ),
        )
        cat.insert_laps(
            conn,
            [
                LapRow(
                    session_id=session_id,
                    lap_index=0,
                    lap_time_s=float(n) / 60.0,
                    start_sample=0,
                    end_sample=n,
                    valid=1,
                    best=1,
                ),
            ],
        )


def test_per_sample_median_not_session_mean(tmp_corpus: Path) -> None:
    """Bimodal AirDensity: session means hide what the sample median sees.

    Session 1 holds 100 samples at 1.10 kg/m³; session 2 holds 900 samples at
    1.30 kg/m³. The per-session-mean approach (1.20) is the unweighted average
    of (1.10, 1.30); the per-sample median is 1.30 because 90% of clean
    samples are at 1.30. The recommender's env must follow the latter.
    """
    car = "bmw"
    track = "synthetic_track"
    _synthesize_session(
        corpus_root=tmp_corpus,
        session_id="aaaaaaaaaaaaaaa1",
        car=car,
        track=track,
        air_density_samples=np.full(100, 1.10),
        track_temp_samples=np.full(100, 20.0),
        wind_vel_samples=np.full(100, 0.0),
        wind_dir_samples=np.full(100, 0.0),
        track_wetness_samples=np.full(100, 0.0),
    )
    _synthesize_session(
        corpus_root=tmp_corpus,
        session_id="aaaaaaaaaaaaaaa2",
        car=car,
        track=track,
        air_density_samples=np.full(900, 1.30),
        track_temp_samples=np.full(900, 30.0),
        wind_vel_samples=np.full(900, 5.0),
        wind_dir_samples=np.full(900, 0.0),
        track_wetness_samples=np.full(900, 0.0),
    )

    from racingoptimizer.ingest import api as ingest_api

    sessions = ingest_api.sessions(car=car, track=track, corpus_root=tmp_corpus)
    out = _environment_from_corpus(sessions, corpus_root=tmp_corpus)

    # Per-sample median dominates: 90% of samples are at 1.30, not 1.20.
    assert out["air_density"] == pytest.approx(1.30)
    assert out["track_temp_c"] == pytest.approx(30.0)
    assert out["wind_vel_ms"] == pytest.approx(5.0)


def test_data_quality_mask_filters_dirty_samples(tmp_corpus: Path) -> None:
    """Samples with mask=False are excluded before the median is taken."""
    car = "bmw"
    track = "synthetic_track"
    n = 1000
    air = np.full(n, 1.30)
    air[:500] = 1.10  # First half is "dirty"; median of clean is 1.30.
    mask = np.ones(n, dtype=bool)
    mask[:500] = False

    _synthesize_session(
        corpus_root=tmp_corpus,
        session_id="bbbbbbbbbbbbbbb1",
        car=car,
        track=track,
        air_density_samples=air,
        track_temp_samples=np.full(n, 25.0),
        wind_vel_samples=np.full(n, 0.0),
        wind_dir_samples=np.full(n, 0.0),
        track_wetness_samples=np.full(n, 0.0),
        quality_mask=mask,
    )

    from racingoptimizer.ingest import api as ingest_api

    sessions = ingest_api.sessions(car=car, track=track, corpus_root=tmp_corpus)
    out = _environment_from_corpus(sessions, corpus_root=tmp_corpus)
    assert out["air_density"] == pytest.approx(1.30)


def test_falls_back_to_defaults_when_no_clean_samples(tmp_corpus: Path) -> None:
    """All-dirty session yields zero clean samples; defaults apply."""
    car = "bmw"
    track = "synthetic_track"
    n = 60
    _synthesize_session(
        corpus_root=tmp_corpus,
        session_id="ccccccccccccccc1",
        car=car,
        track=track,
        air_density_samples=np.full(n, 1.50),
        track_temp_samples=np.full(n, 99.0),
        wind_vel_samples=np.full(n, 99.0),
        wind_dir_samples=np.full(n, 99.0),
        track_wetness_samples=np.full(n, 1.0),
        quality_mask=np.zeros(n, dtype=bool),
    )

    from racingoptimizer.ingest import api as ingest_api

    sessions = ingest_api.sessions(car=car, track=track, corpus_root=tmp_corpus)
    out = _environment_from_corpus(sessions, corpus_root=tmp_corpus)
    assert out == _ENV_DEFAULTS


def test_empty_sessions_dataframe_returns_defaults(tmp_corpus: Path) -> None:
    """Empty-corpus path returns the standard atmosphere fallback."""
    empty = pl.DataFrame({"session_id": [], "car": [], "track": []})
    out = _environment_from_corpus(empty, corpus_root=tmp_corpus)
    assert out == _ENV_DEFAULTS


def test_circular_median_handles_360_wrap() -> None:
    """[350, 10] should yield 0, not 180."""
    angles = np.array([350.0, 10.0, 0.0, 350.0, 10.0])
    result = _circular_median_deg(angles)
    # Both 0 and 360 are "north"; the modulo means we always return 0..360.
    delta = min(abs(result - 0.0), abs(result - 360.0))
    assert delta < 1.0


def test_circular_median_uniform_samples() -> None:
    """All-identical samples preserve the angle."""
    angles = np.full(100, 270.0)
    assert _circular_median_deg(angles) == pytest.approx(270.0)
