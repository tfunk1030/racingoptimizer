"""Longitudinal accuracy log persistence (Stage 3 / gap #4).

After every `fit()`, append one row per fitter to
`<corpus_root>/models/accuracy_log.parquet`. Schema:

    timestamp           UTC ISO-8601 string of the fit
    car                 normalized car key
    track               dominant track for the fit
    sessions_hash       short SHA-256 of sorted session_ids
    n_sessions          number of sessions in the fit
    corner_id           int corner identifier
    phase               phase string
    channel             output channel name
    cv_residual_std     CV residual std for this fitter
    signal_std          training-data std for this output channel
    noise_ratio         cv_residual_std / max(signal_std, 1e-12)
    n_samples           training rows the fitter saw
    regime              Confidence.derive regime tag

`load_latest_fit_quality(car, track, corpus_root)` reads the freshest entry
group for a (car, track) pair and returns
``fit_quality = 1 - median(noise_ratio)`` plus the previous entry's value
for the trend-line render in `optimize status`.

The log is append-only; we read it back by groupby + last to surface "the
latest fit" per (sessions_hash, fit_timestamp) tuple.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from racingoptimizer.confidence import Confidence
from racingoptimizer.physics.model import FitRecord


def _accuracy_log_path(corpus_root: Path) -> Path:
    return corpus_root / "models" / "accuracy_log.parquet"


def _sessions_hash(session_ids: list[str]) -> str:
    digest = hashlib.sha256(
        "|".join(sorted(session_ids)).encode("utf-8")
    ).hexdigest()
    return digest[:16]


def append_accuracy_log(
    *,
    corpus_root: Path,
    car: str,
    track: str,
    session_ids: list[str],
    records: list[tuple[int, str, str, FitRecord]],
) -> Path:
    """Append per-fitter calibration rows for one fit run.

    Each `records` tuple is `(corner_id, phase, channel, fit_record)`.
    """
    if not records:
        return _accuracy_log_path(corpus_root)
    timestamp = datetime.now(UTC).isoformat()
    sessions_hash = _sessions_hash(session_ids)
    rows: list[dict] = []
    for corner_id, phase, channel, rec in records:
        signal_std = float(max(rec.signal_std, 1e-12))
        noise_ratio = float(rec.cv_residual_std) / signal_std
        conf = Confidence.derive(
            value=0.0,
            n_samples=int(rec.n_samples),
            cv_residual_std=float(rec.cv_residual_std),
            signal_std=signal_std,
        )
        rows.append({
            "timestamp": timestamp,
            "car": car,
            "track": track,
            "sessions_hash": sessions_hash,
            "n_sessions": len(session_ids),
            "corner_id": int(corner_id),
            "phase": str(phase),
            "channel": str(channel),
            "cv_residual_std": float(rec.cv_residual_std),
            "signal_std": float(rec.signal_std),
            "noise_ratio": noise_ratio,
            "n_samples": int(rec.n_samples),
            "regime": conf.regime,
        })

    new_frame = pl.DataFrame(rows)
    path = _accuracy_log_path(corpus_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = pl.read_parquet(path)
            stacked = pl.concat([existing, new_frame], how="diagonal_relaxed")
        except Exception:
            stacked = new_frame
    else:
        stacked = new_frame
    stacked.write_parquet(path)
    return path


@dataclass(frozen=True)
class FitQualitySnapshot:
    car: str
    track: str
    timestamp: str
    sessions_hash: str
    fit_quality: float
    median_noise_ratio: float
    n_fitters: int
    prior_fit_quality: float | None


def load_latest_fit_quality(
    *,
    corpus_root: Path,
    car: str,
    track: str,
) -> FitQualitySnapshot | None:
    """Latest fit-quality snapshot for (car, track), with prior for trend."""
    path = _accuracy_log_path(corpus_root)
    if not path.exists():
        return None
    try:
        frame = pl.read_parquet(path)
    except Exception:
        return None
    if frame.is_empty():
        return None
    sub = frame.filter(
        (pl.col("car") == car) & (pl.col("track") == track)
    )
    if sub.is_empty():
        return None
    # Group by fit run (timestamp + sessions_hash) and take the latest by
    # timestamp string (ISO-8601 sorts lexicographically).
    runs = sub.group_by(["timestamp", "sessions_hash"]).agg(
        pl.col("noise_ratio").median().alias("median_noise_ratio"),
        pl.len().alias("n_fitters"),
    ).sort(["timestamp", "sessions_hash"], descending=[True, False])
    if runs.is_empty():
        return None
    rows = runs.to_dicts()
    latest = rows[0]
    prior = rows[1] if len(rows) > 1 else None

    fit_quality = max(0.0, 1.0 - float(latest["median_noise_ratio"]))
    prior_fit_quality = (
        max(0.0, 1.0 - float(prior["median_noise_ratio"]))
        if prior is not None
        else None
    )
    return FitQualitySnapshot(
        car=car,
        track=track,
        timestamp=str(latest["timestamp"]),
        sessions_hash=str(latest["sessions_hash"]),
        fit_quality=fit_quality,
        median_noise_ratio=float(latest["median_noise_ratio"]),
        n_fitters=int(latest["n_fitters"]),
        prior_fit_quality=prior_fit_quality,
    )


__all__ = [
    "FitQualitySnapshot",
    "append_accuracy_log",
    "load_latest_fit_quality",
]
