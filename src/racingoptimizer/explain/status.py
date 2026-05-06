"""ModelStatus + TrackCoverage dataclasses (spec §3)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackCoverage:
    track: str
    n_sessions: int
    n_valid_laps: int
    n_clean_corner_phases: int
    fit_quality: float | None
    # DATA DENSITY regime (sparse / noisy / confident / dense) computed
    # by ``cli.recommend._data_density_regime`` from the session + lap
    # counts. Distinct from ``Confidence.regime`` on per-parameter
    # recommendations, which is residual-driven. They can disagree
    # legitimately — a track with many laps but flat input variance
    # looks dense here and sparse to the fitter.
    regime: str  # 'sparse' | 'noisy' | 'confident' | 'dense'


@dataclass(frozen=True)
class ModelStatus:
    car: str
    coverage: tuple[TrackCoverage, ...]
    overall_regime: str
    notes: tuple[str, ...]


__all__ = ["ModelStatus", "TrackCoverage"]
