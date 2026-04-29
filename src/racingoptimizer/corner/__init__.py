"""Corner detection and phase decomposition.

Spec: docs/superpowers/specs/2026-04-28-corner-phase-design.md.

U4 ships the building blocks: the `Phase` enum, the `CornerPhaseKey`
NamedTuple, the `PhaseThresholds` config, and the two pure detector
primitives `detect_corners` and `assign_phases`.

U5 (this slice, B-3) layers the public Polars wrappers on top:
`segment_lap` (column-appending pure function) and
`corner_phase_states` (per-(corner_id, phase) aggregator), plus the
`DEFAULT_CHANNELS` curated lap_data pull list.
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
from racingoptimizer.corner.states import (
    DEFAULT_CHANNELS,
    corner_phase_states,
    segment_lap,
)

__all__ = [
    "DEFAULT_CHANNELS",
    "DEFAULT_THRESHOLDS",
    "CornerPhaseKey",
    "Phase",
    "PhaseThresholds",
    "assign_phases",
    "corner_phase_states",
    "detect_corners",
    "segment_lap",
    "thresholds_for",
]
