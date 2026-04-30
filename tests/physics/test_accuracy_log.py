"""Longitudinal accuracy log persistence (Stage 3 / gap #4).

`fit()` now writes a per-fitter calibration row to
`<corpus_root>/models/accuracy_log.parquet`. `optimize status` consumes it
via `load_latest_fit_quality(...)` to populate `TrackCoverage.fit_quality`
and the calibration-trend note.
"""
from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import pytest

from racingoptimizer.ingest.api import learn, sessions
from racingoptimizer.physics import fit
from racingoptimizer.physics.fitters import GPFitter
from racingoptimizer.physics.io_log import (
    FitQualitySnapshot,
    append_accuracy_log,
    load_latest_fit_quality,
)
from racingoptimizer.physics.model import FitRecord
from racingoptimizer.track import build_track_model

REPO_ROOT = Path(__file__).resolve().parents[2]
BMW_SEBRING_IBT = REPO_ROOT / "ibtfiles" / "bmwlmdh_sebring international 2026-03-22 14-52-24.ibt"


def _make_dummy_record(*, n: int = 50, cv: float = 0.1, sig: float = 1.0) -> FitRecord:
    fitter = GPFitter(random_state=0)
    # We don't actually train; the log writer only reads scalar fields.
    return FitRecord(
        fitter=fitter,
        n_samples=n,
        cv_residual_std=cv,
        signal_std=sig,
        feature_names=("foo", "bar"),
    )


def test_append_and_load_round_trip(tmp_path: Path) -> None:
    """Two append calls produce two snapshots; `load_latest` returns the freshest."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rec_a = _make_dummy_record(n=30, cv=0.5, sig=1.0)
    rec_b = _make_dummy_record(n=60, cv=0.1, sig=1.0)

    append_accuracy_log(
        corpus_root=corpus,
        car="bmw",
        track="sebring_international",
        session_ids=["sid_a"],
        records=[(1, "mid_corner", "accel_lat_g_max", rec_a)],
    )
    # Sleep briefly so the second timestamp lexicographically sorts later.
    time.sleep(0.01)
    append_accuracy_log(
        corpus_root=corpus,
        car="bmw",
        track="sebring_international",
        session_ids=["sid_a", "sid_b"],
        records=[(1, "mid_corner", "accel_lat_g_max", rec_b)],
    )

    log_path = corpus / "models" / "accuracy_log.parquet"
    assert log_path.exists(), "accuracy log parquet should have been written"
    frame = pl.read_parquet(log_path)
    assert frame.height == 2
    assert set(frame["car"].to_list()) == {"bmw"}
    assert set(frame["channel"].to_list()) == {"accel_lat_g_max"}

    snap = load_latest_fit_quality(
        corpus_root=corpus, car="bmw", track="sebring_international",
    )
    assert isinstance(snap, FitQualitySnapshot)
    # rec_b is the latest fit and has lower noise → higher fit_quality.
    assert snap.fit_quality == pytest.approx(1.0 - 0.1, abs=1e-9)
    assert snap.prior_fit_quality == pytest.approx(1.0 - 0.5, abs=1e-9)
    assert snap.n_fitters == 1


def test_load_returns_none_for_missing_log(tmp_path: Path) -> None:
    snap = load_latest_fit_quality(
        corpus_root=tmp_path, car="bmw", track="sebring_international",
    )
    assert snap is None


def test_load_returns_none_for_unknown_track(tmp_path: Path) -> None:
    rec = _make_dummy_record()
    append_accuracy_log(
        corpus_root=tmp_path,
        car="bmw",
        track="sebring_international",
        session_ids=["sid_a"],
        records=[(1, "mid_corner", "accel_lat_g_max", rec)],
    )
    snap = load_latest_fit_quality(
        corpus_root=tmp_path, car="bmw", track="watkins_glen",
    )
    assert snap is None


@pytest.mark.slow
def test_fit_writes_accuracy_log(tmp_path: Path) -> None:
    """A real `fit()` run writes an accuracy log entry."""
    if not BMW_SEBRING_IBT.exists():
        pytest.skip(f"missing BMW Sebring fixture at {BMW_SEBRING_IBT}")
    root = tmp_path / "corpus"
    root.mkdir()
    sids = learn(BMW_SEBRING_IBT, corpus_root=root)
    sess_df = sessions(corpus_root=root)
    car = sess_df.row(0, named=True)["car"]
    track = sess_df.row(0, named=True)["track"]
    tm = build_track_model(track, sids, corpus_root=root)
    fit(car, sids, tm, corpus_root=root, k_folds=2, seed=0xC0FFEE)

    snap = load_latest_fit_quality(
        corpus_root=root, car=car, track=track,
    )
    assert snap is not None, "fit() must persist an accuracy log entry"
    assert snap.fit_quality is not None
    assert snap.n_fitters > 0
