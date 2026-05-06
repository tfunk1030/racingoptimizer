"""Quali-stint phase-weight overlay tests."""
from __future__ import annotations

from racingoptimizer.corner import Phase
from racingoptimizer.physics.phase_weights import PHASE_WEIGHTS
from racingoptimizer.physics.quali_mode import quali_phase_weights


def test_quali_weights_normalise_per_phase() -> None:
    """Each phase row must sum to 1.0 (matches the race-mode invariant)."""
    qpw = quali_phase_weights()
    for phase, weights in qpw.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, (
            f"phase {phase} quali weights sum to {total}, expected 1.0"
        )


def test_quali_amplifies_aero_eff_on_straights() -> None:
    """Quali should weight aero_eff more on straights than race does."""
    race = PHASE_WEIGHTS[Phase.STRAIGHT]
    quali = quali_phase_weights()[Phase.STRAIGHT]
    assert quali["aero_eff"] > race["aero_eff"]


def test_quali_relaxes_platform_in_mid_corner() -> None:
    """Quali should reduce platform-conservatism weight in mid_corner."""
    race = PHASE_WEIGHTS[Phase.MID_CORNER]
    quali = quali_phase_weights()[Phase.MID_CORNER]
    assert quali["platform"] < race["platform"]


def test_quali_amplifies_grip_in_mid_corner() -> None:
    """Quali should weight grip utilization higher in mid_corner."""
    race = PHASE_WEIGHTS[Phase.MID_CORNER]
    quali = quali_phase_weights()[Phase.MID_CORNER]
    assert quali["grip"] > race["grip"]


def test_quali_returns_fresh_dict_each_call() -> None:
    """Caller mutation must not leak into subsequent calls."""
    a = quali_phase_weights()
    a[Phase.STRAIGHT]["aero_eff"] = 99.0
    b = quali_phase_weights()
    assert b[Phase.STRAIGHT]["aero_eff"] != 99.0


def test_quali_covers_every_phase() -> None:
    """All five phases (incl. STRAIGHT) must appear in the overlay."""
    qpw = quali_phase_weights()
    assert set(qpw.keys()) == set(PHASE_WEIGHTS.keys())
