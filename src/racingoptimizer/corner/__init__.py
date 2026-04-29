"""Corner detection and phase decomposition.

Spec: docs/superpowers/specs/2026-04-28-corner-phase-design.md.

This unit (U4) ships only the building blocks: the `Phase` enum, the
`CornerPhaseKey` NamedTuple, the `PhaseThresholds` config, and the two
pure detector primitives `detect_corners` and `assign_phases`. The
public `segment_lap` Polars wrapper and the `corner_phase_states`
aggregator land in U5.
"""
from __future__ import annotations

from racingoptimizer.corner.boundaries import assign_phases
from racingoptimizer.corner.config import (
    DEFAULT_THRESHOLDS,
    PhaseThresholds,
    thresholds_for,
)
from racingoptimizer.corner.detect import detect_corners
from racingoptimizer.corner.phase import CornerPhaseKey, Phase

__all__ = [
    "DEFAULT_THRESHOLDS",
    "CornerPhaseKey",
    "Phase",
    "PhaseThresholds",
    "assign_phases",
    "detect_corners",
    "thresholds_for",
]
