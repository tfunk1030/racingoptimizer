"""SetupRecommendation dataclass (spec §2)."""
from __future__ import annotations

from dataclasses import dataclass

from racingoptimizer.confidence import Confidence
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner import CornerPhaseKey


@dataclass(frozen=True, slots=True)
class SetupRecommendation:
    car: str
    track: str
    env: EnvironmentFrame
    parameters: dict[str, tuple[float, Confidence]]
    score_breakdown: dict[CornerPhaseKey, float]
    untrained_parameters: tuple[str, ...]
    aero_correction_available: bool


__all__ = ["SetupRecommendation"]
