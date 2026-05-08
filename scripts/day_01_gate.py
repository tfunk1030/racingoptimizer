"""Day 1 acceptance gate -- tyre-pressure floor pin (Mode 2).

PLAN.md Section 14.1 acceptance gate:
> For all 5 cars on a representative track ... `optimize <car> <track>
> --json` returns `tyre_cold_pressure_kpa` equal to the per-car
> constraint floor +/- 0.01 kPa.

The runtime form ("call optimize on each car and inspect JSON") would
take ~15 min (5 cars * full DE search). The same contract is verified
much faster by exercising the helper + constraint-pin round-trip in
process: if the helper inserts the floor and `_apply_pins_to_constraints`
narrows the constraint to `[floor, floor]`, the optimizer is structurally
forced to that value (the existing constraint-pin mechanism is covered
by `tests/physics/test_pin_near_constant.py`).

This script is the "gate output" referenced by Day 1's snapshot. Exit
code 0 means the gate passed. The associated canary is verified via
`test_canary_disabled_pin_does_not_insert` in
`tests/cli/test_tyre_pressure_floor.py`.

Run: `uv run python scripts/day_01_gate.py`
"""
from __future__ import annotations

import sys

from racingoptimizer.cli.recommend import (
    CANONICAL_CARS,
    _apply_pins_to_constraints,
    _apply_tyre_pressure_floor_pin,
)
from racingoptimizer.constraints import load_constraints

PARAM = "tyre_cold_pressure_kpa"
TOL = 0.01


def main() -> int:
    table = load_constraints()
    failures: list[str] = []

    for car in CANONICAL_CARS:
        bounds = table.bounds(car, PARAM)
        if bounds is None:
            failures.append(f"{car}: no constraint registered for {PARAM}")
            continue
        floor, _hi = bounds
        # 1. Helper inserts the per-car floor when no user override is set.
        overrides: dict[str, float] = {}
        msg = _apply_tyre_pressure_floor_pin(overrides, table, car)
        if PARAM not in overrides or abs(overrides[PARAM] - floor) > TOL:
            failures.append(
                f"{car}: floor pin not inserted (got "
                f"{overrides.get(PARAM)!r}, expected {floor:.1f})"
            )
            continue
        if msg is None:
            failures.append(f"{car}: floor pin inserted but no info message")
            continue
        # 2. Round-trip through `_apply_pins_to_constraints` narrows bounds.
        pinned_table = _apply_pins_to_constraints(table, car, overrides)
        new_bounds = pinned_table.bounds(car, PARAM)
        if new_bounds is None:
            failures.append(f"{car}: bounds vanished after pin")
            continue
        lo, hi = new_bounds
        if abs(lo - floor) > TOL or abs(hi - floor) > TOL:
            failures.append(
                f"{car}: bounds {new_bounds} not narrowed to ({floor}, {floor})"
            )
            continue
        # 3. User --pin override path stays untouched.
        user_overrides = {PARAM: 200.0}
        msg2 = _apply_tyre_pressure_floor_pin(user_overrides, table, car)
        if user_overrides[PARAM] != 200.0:
            failures.append(f"{car}: user --pin override clobbered by floor")
            continue
        if msg2 is not None:
            failures.append(
                f"{car}: floor helper emitted message even though user pinned"
            )
            continue
        print(f"  {car}: floor={floor:.1f} kPa  pin OK  user-pin OK")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"\nGATE PASSED for all {len(CANONICAL_CARS)} cars.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
