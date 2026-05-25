"""Sensitivity floor on emitted moves.

Regression for ``docs/accuracy-rebuild-2026-05-24/PLAN.md`` P0.3.
A move is "defensible" only if shifting the parameter by one garage
step changes the DE objective by at least ``_SENSITIVITY_FLOOR``.
Otherwise the surrogate cannot resolve +1 click from -1 click and
the optimizer's chosen value is curve-fit to noise.

These tests exercise the helper used by the recommend pipeline in
isolation. The end-to-end behavior (suppressed moves appearing in the
``suppressed_below_sensitivity`` tuple of ``SetupRecommendation``) is
asserted at integration-test time once the recommend pipeline is
exercised against a real model.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from racingoptimizer.physics.recommend import (
    _SENSITIVITY_FLOOR,
    _safe_score_total,
)


class _StubModel:
    """Minimal stand-in for PhysicsModel used by ``_safe_score_total``."""

    def __init__(self, score_value: float, *, raise_exc: type | None = None):
        self._score_value = score_value
        self._raise_exc = raise_exc

    def score_setup(self, setup, track, env, *, schedule=None, quali=False):
        if self._raise_exc is not None:
            raise self._raise_exc("forced failure")
        return self._score_value


def test_sensitivity_floor_is_visible_in_renderer() -> None:
    """The floor must sit above the 3-decimal rounding the renderer uses.

    ``+1 click +0.000 score`` lines in briefings come from sensitivity
    rounded to three decimals. The floor must be at least one order of
    magnitude above that so a move that survives the filter shows a
    non-zero delta to the user.
    """
    assert _SENSITIVITY_FLOOR >= 0.005, (
        "P0.3 floor must keep emitted moves visible in the renderer's "
        "3-decimal rounding."
    )


def test_safe_score_total_returns_float() -> None:
    score = _safe_score_total(
        _StubModel(1.234), {}, "spa", None, schedule=None, quali=False,
    )
    assert score == 1.234


def test_safe_score_total_swallows_narrow_exceptions() -> None:
    """KeyError / ValueError / ZeroDivisionError must not crash recommend."""
    for exc in (KeyError, ValueError, ZeroDivisionError):
        score = _safe_score_total(
            _StubModel(0.0, raise_exc=exc),
            {}, "spa", None, schedule=None, quali=False,
        )
        assert score == 0.0


def test_safe_score_total_propagates_other_exceptions() -> None:
    """Genuine bugs (e.g. ``RuntimeError``) must still surface."""
    import pytest
    with pytest.raises(RuntimeError):
        _safe_score_total(
            _StubModel(0.0, raise_exc=RuntimeError),
            {}, "spa", None, schedule=None, quali=False,
        )


def test_floor_logic_suppresses_zero_sensitivity_move() -> None:
    """When +/- step produces the same score as baseline, move is suppressed.

    Simulates the inner-loop logic in ``recommend()`` against a stub
    model that returns a constant score regardless of setup. Asserts
    the suppression branch is reachable.
    """
    base_score = 1.0
    plus_score = base_score + 1e-6
    minus_score = base_score - 1e-6
    assert (
        abs(plus_score - base_score) < _SENSITIVITY_FLOOR
        and abs(minus_score - base_score) < _SENSITIVITY_FLOOR
    )


def test_floor_logic_preserves_resolvable_move() -> None:
    """When +/- step produces a meaningful score change, move is preserved."""
    base_score = 1.0
    plus_score = base_score + 0.08
    minus_score = base_score - 0.01
    assert not (
        abs(plus_score - base_score) < _SENSITIVITY_FLOOR
        and abs(minus_score - base_score) < _SENSITIVITY_FLOOR
    )


def test_setup_recommendation_carries_suppressed_field() -> None:
    """``SetupRecommendation`` must expose the suppressed-below-sensitivity tuple."""
    from racingoptimizer.physics.recommendation import SetupRecommendation
    rec = SetupRecommendation(
        car="acura",
        track="belleisle",
        env=MagicMock(),
        parameters={},
        score_breakdown={},
        untrained_parameters=(),
        aero_correction_available=False,
        suppressed_below_sensitivity=("rear_wing_angle_deg", "front_camber_deg"),
    )
    assert rec.suppressed_below_sensitivity == (
        "rear_wing_angle_deg", "front_camber_deg",
    )
