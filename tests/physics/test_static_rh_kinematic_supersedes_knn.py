"""Kinematic static-RH fit (P0.2) supersedes the legacy k-NN repair.

When a ``PhysicsModel`` carries a non-empty
``static_rh_kinematic.channels`` mapping, ``physics.recommend.recommend``
must bypass ``enforce_static_rh_feasible``: the kinematic fit IS the
deterministic readout, so blending the recommended setup toward an
older corpus session would degrade rather than improve it.

Tests here exercise the bypass predicate directly (the legacy k-NN
repair test coverage in ``test_static_rh_knn.py`` keeps gating the
fallback path).
"""
from __future__ import annotations

from racingoptimizer.physics.recommend import _kinematic_static_rh_ready
from racingoptimizer.physics.static_rh_kinematic import (
    StaticRhChannelFit,
    StaticRhKinematic,
)


class _Stub:
    """Minimal stand-in for the relevant ``PhysicsModel`` slot."""

    def __init__(self, kinematic):
        self.static_rh_kinematic = kinematic


def _ready_kinematic() -> StaticRhKinematic:
    fit = StaticRhChannelFit(
        channel="setup_static_lf_ride_height_mm",
        features=("pushrod_length_offset_front_mm",),
        coefficients=(1.0,),
        intercept=20.0,
        r2=0.99,
        n_samples=20,
    )
    return StaticRhKinematic(
        car="acura",
        channels={"setup_static_lf_ride_height_mm": fit},
        rejected_channels=(),
        n_sessions=20,
    )


def test_ready_kinematic_returns_true() -> None:
    model = _Stub(_ready_kinematic())
    assert _kinematic_static_rh_ready(model) is True


def test_missing_slot_returns_false() -> None:
    model = _Stub(None)
    assert _kinematic_static_rh_ready(model) is False


def test_empty_channels_returns_false() -> None:
    """A kinematic bundle with every channel rejected (R^2 < 0.98 across
    the board) must NOT trigger the bypass: the legacy k-NN repair is
    still the best fallback in that regime."""
    rejected = StaticRhKinematic(
        car="acura",
        channels={},
        rejected_channels=(
            "setup_static_lf_ride_height_mm",
            "setup_static_rf_ride_height_mm",
        ),
        n_sessions=20,
    )
    model = _Stub(rejected)
    assert _kinematic_static_rh_ready(model) is False


def test_attr_absent_returns_false() -> None:
    """Old pickles (pre-P0.2 layout) won't have the slot at all."""
    class _Legacy:
        pass

    assert _kinematic_static_rh_ready(_Legacy()) is False


def test_partial_kinematic_still_triggers_bypass() -> None:
    """Even one shipped channel is enough -- the per-channel
    ``predict_setup_readouts`` path falls through to the surrogate for
    rejected channels but still owns the static-RH readout for the
    accepted ones, and that's what ``enforce_static_rh_feasible``
    would otherwise corrupt."""
    one_channel = StaticRhKinematic(
        car="acura",
        channels={
            "setup_static_lf_ride_height_mm": StaticRhChannelFit(
                channel="setup_static_lf_ride_height_mm",
                features=("pushrod_length_offset_front_mm",),
                coefficients=(1.0,),
                intercept=20.0,
                r2=0.985,
                n_samples=20,
            ),
        },
        rejected_channels=(
            "setup_static_rf_ride_height_mm",
            "setup_static_lr_ride_height_mm",
            "setup_static_rr_ride_height_mm",
        ),
        n_sessions=20,
    )
    model = _Stub(one_channel)
    assert _kinematic_static_rh_ready(model) is True
