"""SetupComparison + CornerPhaseDelta dataclasses (spec §3)."""
from __future__ import annotations

from dataclasses import dataclass

from racingoptimizer.corner import Phase


@dataclass(frozen=True)
class CornerPhaseDelta:
    corner_id: int
    phase: Phase
    score_a: float
    score_b: float
    delta: float
    drivers: tuple[str, ...]


@dataclass(frozen=True)
class SetupComparison:
    car: str
    track: str
    setup_a_id: str
    setup_b_id: str
    total_score_a: float
    total_score_b: float
    per_corner_phase: tuple[CornerPhaseDelta, ...]
    notes: tuple[str, ...]


__all__ = ["CornerPhaseDelta", "SetupComparison"]
