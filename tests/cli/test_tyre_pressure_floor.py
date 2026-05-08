"""Day 1 of the physics-rebuild plan: tyre-pressure floor pin (Mode 2).

Background (PLAN.md Section 14.1):
- The surrogate rewards platform stability (cleaner ride-height telemetry
  that higher cold pressure delivers) but cannot see the peak-grip drop
  from a smaller contact patch.
- Recent BMW Spa runs recommend 154-163 kPa against the 152 floor;
  community wisdom is "stay at the floor."
- Cheapest fix: in `recommend_cmd`, before `_apply_pins_to_constraints`,
  if `tyre_cold_pressure_kpa` is in the constraint table AND the user
  did not pass `--pin tyre_cold_pressure_kpa=...`, force-pin to the
  per-car constraint floor.

Acceptance gate (per PLAN.md Section 14.1): for all 5 cars on a
representative track, `optimize <car> <track> --json` returns
`tyre_cold_pressure_kpa` equal to the per-car constraint floor
+/- 0.01 kPa, unless overridden by an explicit `--pin`.

Broken-model canary: with the pin disabled, the gate must FAIL.
The unit test `test_canary_disabled_pin_skips_insertion` exercises
the disabled path directly; an end-to-end canary on BMW Sebring is
documented in the daily snapshot but kept out of CI for runtime.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from racingoptimizer.cli.recommend import (
    CANONICAL_CARS,
    _apply_tyre_pressure_floor_pin,
)
from racingoptimizer.constraints import ConstraintsTable, load_constraints


def _table_with_floor(car: str, lo: float, hi: float) -> ConstraintsTable:
    """Build a real ConstraintsTable then narrow tyre pressure for one car."""
    table = load_constraints()
    return table.with_pin(car, "tyre_cold_pressure_kpa", lo)._replace_bounds(
        car, "tyre_cold_pressure_kpa", (lo, hi)
    ) if hasattr(table, "_replace_bounds") else table


def test_floor_pin_inserted_when_no_user_pin() -> None:
    """The helper inserts the per-car floor when no user override is set."""
    table = load_constraints()
    overrides: dict[str, float] = {}
    msg = _apply_tyre_pressure_floor_pin(overrides, table, "bmw")
    floor = table.bounds("bmw", "tyre_cold_pressure_kpa")[0]
    assert overrides["tyre_cold_pressure_kpa"] == pytest.approx(floor, abs=0.01)
    assert msg is not None and "152" in msg


def test_user_pin_overrides_floor() -> None:
    """An explicit user --pin must NOT be overwritten by the floor."""
    table = load_constraints()
    overrides = {"tyre_cold_pressure_kpa": 180.0}
    msg = _apply_tyre_pressure_floor_pin(overrides, table, "bmw")
    assert overrides["tyre_cold_pressure_kpa"] == 180.0
    assert msg is None  # no info message when user-overridden


def test_floor_pin_applied_for_all_canonical_cars() -> None:
    """Every GTP car recognised by the optimizer gets the floor by default."""
    table = load_constraints()
    for car in CANONICAL_CARS:
        overrides: dict[str, float] = {}
        _apply_tyre_pressure_floor_pin(overrides, table, car)
        bounds = table.bounds(car, "tyre_cold_pressure_kpa")
        assert bounds is not None, f"{car} missing tyre_cold_pressure_kpa bounds"
        assert overrides["tyre_cold_pressure_kpa"] == pytest.approx(
            bounds[0], abs=0.01,
        ), f"{car} pin {overrides['tyre_cold_pressure_kpa']} != floor {bounds[0]}"


def test_floor_pin_silent_when_parameter_unknown() -> None:
    """If the constraint table has no tyre_cold_pressure_kpa for a car,
    the helper is a no-op (does not raise)."""
    table = MagicMock()
    table.bounds.return_value = None
    overrides: dict[str, float] = {}
    msg = _apply_tyre_pressure_floor_pin(overrides, table, "fictional_car")
    assert "tyre_cold_pressure_kpa" not in overrides
    assert msg is None


def test_floor_pin_does_not_clobber_other_overrides() -> None:
    """The helper touches only tyre_cold_pressure_kpa; everything else
    in the overrides dict is untouched."""
    table = load_constraints()
    overrides = {
        "fuel_level_l": 58.0,
        "rear_wing_angle_deg": 14.0,
    }
    _apply_tyre_pressure_floor_pin(overrides, table, "bmw")
    assert overrides["fuel_level_l"] == 58.0
    assert overrides["rear_wing_angle_deg"] == 14.0
    assert "tyre_cold_pressure_kpa" in overrides


def test_floor_pin_emits_info_message_with_value() -> None:
    """The returned info string includes the actual pinned value (so the
    user sees what changed) and identifies the parameter."""
    table = load_constraints()
    overrides: dict[str, float] = {}
    msg = _apply_tyre_pressure_floor_pin(overrides, table, "bmw")
    assert msg is not None
    assert "tyre" in msg.lower() or "pressure" in msg.lower()
    floor = table.bounds("bmw", "tyre_cold_pressure_kpa")[0]
    assert f"{floor:.1f}" in msg


def test_floor_pin_idempotent_when_pin_already_at_floor() -> None:
    """If a previous pass already set tyre_cold_pressure_kpa to the floor
    value (e.g. via `--pin tyre_cold_pressure_kpa=152`), the helper
    treats that as user-set and does not reinsert."""
    table = load_constraints()
    floor = table.bounds("bmw", "tyre_cold_pressure_kpa")[0]
    overrides = {"tyre_cold_pressure_kpa": float(floor)}
    msg = _apply_tyre_pressure_floor_pin(overrides, table, "bmw")
    assert overrides["tyre_cold_pressure_kpa"] == pytest.approx(floor, abs=0.01)
    # User-pinned-at-floor is treated as user-set; no info message.
    assert msg is None


def test_floor_pin_round_trip_through_apply_pins_to_constraints() -> None:
    """End-to-end: floor pin -> _apply_pins_to_constraints -> bounds(c, p)
    is now [floor, floor] (the constraint-side enforcement of the pin)."""
    from racingoptimizer.cli.recommend import _apply_pins_to_constraints

    table = load_constraints()
    overrides: dict[str, float] = {}
    _apply_tyre_pressure_floor_pin(overrides, table, "bmw")
    pinned_table = _apply_pins_to_constraints(table, "bmw", overrides)
    new_bounds = pinned_table.bounds("bmw", "tyre_cold_pressure_kpa")
    floor = table.bounds("bmw", "tyre_cold_pressure_kpa")[0]
    assert new_bounds == pytest.approx((floor, floor), abs=0.01)


def test_canary_disabled_pin_does_not_insert() -> None:
    """Broken-model canary: with the pin function bypassed (caller chose
    not to call it), the overrides dict has no tyre_cold_pressure_kpa
    entry, and downstream bounds remain the wide constraint envelope.

    This is the inverse-direction proof that the helper is the ONLY
    thing inserting the pin -- if a future commit accidentally reverts
    the helper, recommendations drift off the floor exactly as the
    Mode 2 evidence (`recommendations/bmw-spa-race-0507-*.txt`)
    documented.
    """
    from racingoptimizer.cli.recommend import _apply_pins_to_constraints

    table = load_constraints()
    overrides: dict[str, float] = {}
    # Helper NOT called -- this is the canary path.
    pinned_table = _apply_pins_to_constraints(table, "bmw", overrides)
    new_bounds = pinned_table.bounds("bmw", "tyre_cold_pressure_kpa")
    # Bounds are unchanged from the loaded table (wide envelope).
    assert new_bounds == table.bounds("bmw", "tyre_cold_pressure_kpa")
    assert new_bounds[0] != new_bounds[1], (
        "with the helper bypassed, bounds must remain non-pinned (wide). "
        "If this fires, something OTHER than _apply_tyre_pressure_floor_pin "
        "is inserting the floor pin -- track it down before merging."
    )
