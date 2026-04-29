"""Corner-phase enum and per-phase identity key.

`Phase` is a StrEnum so its values serialise as plain strings into Polars
columns and JSON without an extra adapter. `CornerPhaseKey` is the atomic
unit downstream consumers (fitter, optimizer, recommender) cite — see
docs/superpowers/specs/2026-04-28-corner-phase-design.md §2.
"""
from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple


class Phase(StrEnum):
    BRAKING = "braking"
    TRAIL_BRAKE = "trail_brake"
    MID_CORNER = "mid_corner"
    EXIT = "exit"
    STRAIGHT = "straight"


class CornerPhaseKey(NamedTuple):
    session_id: str
    lap_index: int
    corner_id: int
    phase: Phase
