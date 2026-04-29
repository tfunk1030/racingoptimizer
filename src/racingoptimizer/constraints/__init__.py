"""Setup-legality constraints: loader + clamp.

Public API:
    load_constraints(path=None) -> ConstraintsTable
    ConstraintsTable.bounds(car, parameter) -> (lo, hi) | None
    ConstraintsTable.parameters(car=None) -> list[str]
    clamp(value, parameter, car, table=None) -> ClampResult
"""
from __future__ import annotations

from racingoptimizer.constraints.clamp import ClampResult, clamp
from racingoptimizer.constraints.loader import (
    ConstraintsParseError,
    ConstraintsTable,
    load_constraints,
)

__all__ = [
    "ClampResult",
    "ConstraintsParseError",
    "ConstraintsTable",
    "clamp",
    "load_constraints",
]
