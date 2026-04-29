"""Frozen phase-weight table invariants (spec §6)."""
from __future__ import annotations

import pytest

from racingoptimizer.corner import Phase
from racingoptimizer.physics.phase_weights import PHASE_WEIGHTS, SUB_UTILIZATIONS


def test_phase_weights_cover_all_phases() -> None:
    assert set(PHASE_WEIGHTS) == set(Phase)


def test_phase_weights_cover_all_sub_utilizations() -> None:
    for phase, table in PHASE_WEIGHTS.items():
        assert set(table.keys()) == set(SUB_UTILIZATIONS), phase


def test_phase_weights_each_row_sums_to_one() -> None:
    for phase, table in PHASE_WEIGHTS.items():
        total = sum(table.values())
        assert total == pytest.approx(1.0, abs=1e-9), (phase, total)


def test_phase_weights_non_negative() -> None:
    for phase, table in PHASE_WEIGHTS.items():
        for sub, w in table.items():
            assert w >= 0.0, (phase, sub, w)


def test_braking_phase_emphasises_grip_stability_platform() -> None:
    # Spec §6: braking weights grip + stability + platform heavily.
    table = PHASE_WEIGHTS[Phase.BRAKING]
    assert table["platform"] >= 0.20
    assert table["grip"] >= 0.30
    assert table["stability"] >= 0.25


def test_mid_corner_emphasises_grip_balance() -> None:
    table = PHASE_WEIGHTS[Phase.MID_CORNER]
    assert table["grip"] >= 0.30
    assert table["balance"] >= 0.25


def test_exit_emphasises_traction_balance_aero() -> None:
    table = PHASE_WEIGHTS[Phase.EXIT]
    assert table["traction"] >= 0.30
    assert table["balance"] >= 0.20
    assert table["aero_eff"] >= 0.10


def test_straight_emphasises_aero_platform() -> None:
    table = PHASE_WEIGHTS[Phase.STRAIGHT]
    assert table["aero_eff"] >= 0.50
    assert table["platform"] >= 0.30
    # No grip / balance / traction weight on straights.
    assert table["grip"] == 0.0
    assert table["balance"] == 0.0
    assert table["traction"] == 0.0


def test_trail_brake_blends_braking_and_mid_corner() -> None:
    table = PHASE_WEIGHTS[Phase.TRAIL_BRAKE]
    # Both braking-side (grip/stability) and mid-corner-side (balance) carry weight.
    assert table["grip"] >= 0.20
    assert table["stability"] >= 0.10
    assert table["balance"] >= 0.20
