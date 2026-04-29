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
    regime: str  # 'sparse' | 'noisy' | 'confident' | 'dense'


@dataclass(frozen=True)
class ModelStatus:
    car: str
    coverage: tuple[TrackCoverage, ...]
    overall_regime: str
    notes: tuple[str, ...]


__all__ = ["ModelStatus", "TrackCoverage"]
