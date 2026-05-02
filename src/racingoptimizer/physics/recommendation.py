"""SetupRecommendation dataclass (spec §2)."""
from __future__ import annotations

from dataclasses import dataclass, field

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
    # Parameters that were pinned to their training-data median because the
    # observed value held effectively constant across sessions — no signal
    # for the joint surrogate to recommend deviating. The CLI surfaces
    # these in the briefing so the user understands why no exploration
    # happened on those clicks. Empty tuple means nothing was pinned.
    pinned_to_observed_median: tuple[str, ...] = field(default_factory=tuple)
    # Parameter -> human-readable warning when the recommendation came out
    # at a constraint bound AND the model's training baseline was outside
    # the bound (i.e. the constraint floor/ceiling silently dragged the
    # output away from where every observed session actually ran). A
    # populated entry means "verify the constraint in constraints.md
    # against the iRacing UI — the floor/ceiling looks wrong relative to
    # the data". Empty dict means no bound-binding mismatch was detected.
    clamp_warnings: dict[str, str] = field(default_factory=dict)


__all__ = ["SetupRecommendation"]
