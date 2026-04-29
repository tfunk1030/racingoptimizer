"""Clip a recommended value to its legal bounds.

Clamping is the renderer's last preflight before output (cli-design spec §5).
Slice E may emit a value outside the legal envelope (it fits empirically); we
clip and report it. Three statuses convey what happened:

    ok                 -> value was inside bounds
    clamped            -> value was outside bounds, returned at the boundary
    unbounded          -> parameter known but its bound is `<TODO>` in the file
    unknown_parameter  -> parameter not in constraints.md at all
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from racingoptimizer.constraints.loader import CANONICAL_CARS, ConstraintsTable, load_constraints
from racingoptimizer.ingest.detect import UnknownCarError, normalize_car_key

ClampStatus = Literal["ok", "clamped", "unbounded", "unknown_parameter"]


def _to_canonical(car: str) -> str:
    raw = car.strip().lower()
    if raw in CANONICAL_CARS:
        return raw
    try:
        return normalize_car_key(raw)
    except UnknownCarError:
        return raw


@dataclass(frozen=True)
class ClampResult:
    value: float
    was_clamped: bool
    bound: tuple[float, float] | None
    status: ClampStatus


def clamp(
    value: float,
    parameter: str,
    car: str,
    table: ConstraintsTable | None = None,
) -> ClampResult:
    if table is None:
        table = load_constraints()
    car_key = _to_canonical(car)

    # Defaults section is the universe of known parameters; per-car overrides
    # only shadow defaults, they never introduce new parameters.
    if parameter not in table.parameters():
        return ClampResult(value=value, was_clamped=False, bound=None, status="unknown_parameter")

    bound = table.bounds(car_key, parameter)
    if bound is None:
        return ClampResult(value=value, was_clamped=False, bound=None, status="unbounded")

    lo, hi = bound
    clipped = min(max(value, lo), hi)
    if clipped != value:
        return ClampResult(value=clipped, was_clamped=True, bound=bound, status="clamped")
    return ClampResult(value=value, was_clamped=False, bound=bound, status="ok")
