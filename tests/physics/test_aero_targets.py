"""Per-car aero targets registry."""
from __future__ import annotations

from racingoptimizer.physics.aero_targets import (
    ACURA_AERO_TARGETS,
    AeroTargets,
    aero_targets_for,
)


def test_aero_targets_for_acura_returns_known_spec() -> None:
    targets = aero_targets_for("acura")
    assert targets is ACURA_AERO_TARGETS
    assert targets.front_rh_target_mm == 15.0
    assert targets.rear_rh_target_mm == 40.0
    assert targets.rake_orientation == "more_rake"


def test_aero_targets_for_unknown_car_returns_none() -> None:
    """BMW/Cadillac/Ferrari/Porsche have no captured driver spec yet.

    Returning a default would silently nudge those cars away from
    their correctly-fit aero maps. The contract is "None = defer to
    the surrogate" -- callers MUST honour it.
    """
    for car in ("bmw", "cadillac", "ferrari", "porsche"):
        assert aero_targets_for(car) is None


def test_aero_targets_for_normalises_case_and_whitespace() -> None:
    assert aero_targets_for("ACURA") is ACURA_AERO_TARGETS
    assert aero_targets_for("  acura  ") is ACURA_AERO_TARGETS
    assert aero_targets_for("") is None


def test_acura_balance_lever_priority_starts_with_rear_wing() -> None:
    """Per docs/cars/acura_arx06.md the rear wing is the primary lever."""
    targets = aero_targets_for("acura")
    assert targets is not None
    assert targets.balance_levers_priority[0] == "rear_wing"
    assert targets.balance_levers_priority[1] == "rear_pushrod"


def test_aero_targets_is_frozen_dataclass() -> None:
    """Targets are read-only data -- mutating them by accident would
    silently change the optimizer's prior across the whole process.
    """
    import dataclasses

    targets = aero_targets_for("acura")
    assert targets is not None
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        targets.front_rh_target_mm = 30.0  # type: ignore[misc]


def test_aero_targets_notes_mention_rake_direction() -> None:
    """Smoke check that the notes string carries the rake → balance
    arrow direction the briefing wants to surface for Acura.
    """
    targets = aero_targets_for("acura")
    assert targets is not None
    assert "rake" in targets.notes.lower()
    assert "oversteer" in targets.notes.lower()


def test_aero_targets_type_aliases_exposed() -> None:
    """Catch accidental removal of the public API on refactor."""
    from racingoptimizer.physics.aero_targets import (
        AeroTargets as _AeroTargets,
    )
    from racingoptimizer.physics.aero_targets import (
        BalanceLever as _BalanceLever,
    )

    assert _AeroTargets is AeroTargets
    # ``BalanceLever`` is a ``Literal`` alias; just confirm import works.
    assert _BalanceLever is not None
