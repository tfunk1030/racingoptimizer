"""Deterministic static-RH kinematic fit (P0.2).

Regression tests for ``docs/accuracy-rebuild-2026-05-24/PLAN.md`` P0.2.
Static garage ride height is a kinematic function of perch / pushrod /
heave / TB / camber / toe / fuel given the car's installation ratios.
The previous surrogate had near-zero pushrod gradient (6 mm pushrod ->
0.1 mm predicted static RH); the per-car closed-form linear fit must
reproduce kinematic geometry within ``mm`` accuracy.
"""
from __future__ import annotations

import numpy as np
import pytest

from racingoptimizer.physics.static_rh_kinematic import (
    STATIC_RH_CHANNELS,
    STATIC_RH_FEATURES,
    StaticRhKinematic,
    fit_static_rh_kinematic,
    predict_static_rh_kinematic,
)


def _synthetic_corpus(
    n_sessions: int = 40,
    *,
    rh_pushrod_ratio_f: float = 1.0,
    rh_pushrod_ratio_r: float = 1.0,
    rh_perch_ratio_f: float = 0.5,
    rh_perch_ratio_r: float = 0.5,
    rh_fuel_ratio: float = -0.08,
    noise_mm: float = 0.0,
    seed: int = 42,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Build a synthetic corpus with known linear kinematic relationships.

    Static RH is a deterministic function of platform inputs; this
    generator produces a swept corpus where every feature varies
    independently so the linear fit is well-conditioned.
    """
    rng = np.random.default_rng(seed)
    sid_to_params: dict[str, dict[str, float]] = {}
    sid_to_readouts: dict[str, dict[str, float]] = {}
    for i in range(n_sessions):
        params: dict[str, float] = {
            "heave_spring_rate_n_per_mm": float(rng.uniform(80, 200)),
            "third_spring_rate_n_per_mm": float(rng.uniform(80, 200)),
            "heave_perch_offset_front_mm": float(rng.uniform(30, 80)),
            "spring_perch_offset_rear_mm": float(rng.uniform(30, 80)),
            "pushrod_length_offset_front_mm": float(rng.uniform(-30, 10)),
            "pushrod_length_offset_rear_mm": float(rng.uniform(-30, 10)),
            "torsion_bar_turns_fl": float(rng.uniform(-0.2, 0.2)),
            "torsion_bar_od_fl_mm": float(rng.choice([14.0, 15.0, 16.0])),
            "torsion_bar_turns_rl": float(rng.uniform(-0.2, 0.2)),
            "torsion_bar_od_rl_mm": float(rng.choice([14.0, 15.0, 16.0])),
            "fuel_level_l": float(rng.uniform(20, 90)),
            "camber_fl_deg": float(rng.uniform(-3.5, -0.5)),
            "camber_rl_deg": float(rng.uniform(-3.5, -0.5)),
            "toe_front_mm": float(rng.uniform(-1.0, 1.0)),
            "toe_rl_mm": float(rng.uniform(-1.0, 1.0)),
        }
        # Synthetic kinematic formula: static RH responds linearly to
        # pushrod (1 mm/mm), perch (~0.5 mm/mm), and fuel (~-0.08 mm/L).
        rh_lf = (
            45.0
            + rh_pushrod_ratio_f * params["pushrod_length_offset_front_mm"]
            + rh_perch_ratio_f * (params["heave_perch_offset_front_mm"] - 55.0)
            + rh_fuel_ratio * (params["fuel_level_l"] - 55.0)
        )
        rh_rr = (
            55.0
            + rh_pushrod_ratio_r * params["pushrod_length_offset_rear_mm"]
            + rh_perch_ratio_r * (params["spring_perch_offset_rear_mm"] - 55.0)
            + rh_fuel_ratio * (params["fuel_level_l"] - 55.0) * 0.5
        )
        readouts: dict[str, float] = {
            "setup_static_lf_ride_height_mm": rh_lf
            + float(rng.normal(0.0, noise_mm)),
            "setup_static_rf_ride_height_mm": rh_lf
            + float(rng.normal(0.0, noise_mm)),
            "setup_static_lr_ride_height_mm": rh_rr
            + float(rng.normal(0.0, noise_mm)),
            "setup_static_rr_ride_height_mm": rh_rr
            + float(rng.normal(0.0, noise_mm)),
        }
        sid = f"s{i:04d}"
        sid_to_params[sid] = params
        sid_to_readouts[sid] = readouts
    return sid_to_params, sid_to_readouts


def test_fit_reaches_r2_above_threshold_on_clean_corpus() -> None:
    """A 40-session sweep with no noise must achieve R^2 >= 0.98 per channel."""
    sid_to_params, sid_to_readouts = _synthetic_corpus(n_sessions=40, noise_mm=0.0)
    kinematic = fit_static_rh_kinematic("bmw", sid_to_params, sid_to_readouts)
    assert kinematic.is_ready()
    assert set(kinematic.channels.keys()) == set(STATIC_RH_CHANNELS)
    for fit in kinematic.channels.values():
        assert fit.r2 >= 0.98, (
            f"{fit.channel} R^2 = {fit.r2:.4f} below P0.2 ship threshold"
        )


def test_fit_recovers_unit_pushrod_gradient() -> None:
    """A 1 mm front-pushrod move must shift predicted LF static RH by ~1 mm.

    Specifically tests the failure mode that motivated P0.2: the prior
    surrogate produced a 0.1 mm static RH change for a 6 mm pushrod move.
    The kinematic fit must reproduce the synthetic 1:1 ratio.
    """
    sid_to_params, sid_to_readouts = _synthetic_corpus(
        n_sessions=40, rh_pushrod_ratio_f=1.0, noise_mm=0.0,
    )
    kinematic = fit_static_rh_kinematic("bmw", sid_to_params, sid_to_readouts)
    base = {f: 0.0 for f in STATIC_RH_FEATURES}
    base["heave_perch_offset_front_mm"] = 55.0
    base["fuel_level_l"] = 55.0
    base["pushrod_length_offset_front_mm"] = -10.0
    delta = dict(base)
    delta["pushrod_length_offset_front_mm"] = -9.0  # +1 mm
    rh_base = predict_static_rh_kinematic(kinematic, base)
    rh_delta = predict_static_rh_kinematic(kinematic, delta)
    diff = (
        rh_delta["setup_static_lf_ride_height_mm"]
        - rh_base["setup_static_lf_ride_height_mm"]
    )
    assert abs(diff - 1.0) < 0.05, (
        f"Front pushrod +1mm should move LF static RH by ~1mm, got {diff:.3f}mm"
    )


def test_fit_recovers_six_mm_pushrod_move() -> None:
    """The exact failure mode that motivated P0.2: 6 mm pushrod -> ~6 mm RH."""
    sid_to_params, sid_to_readouts = _synthetic_corpus(
        n_sessions=40, rh_pushrod_ratio_f=1.0, noise_mm=0.0,
    )
    kinematic = fit_static_rh_kinematic("bmw", sid_to_params, sid_to_readouts)
    base = {f: 0.0 for f in STATIC_RH_FEATURES}
    base["heave_perch_offset_front_mm"] = 55.0
    base["fuel_level_l"] = 55.0
    base["pushrod_length_offset_front_mm"] = -24.5
    delta = dict(base)
    delta["pushrod_length_offset_front_mm"] = -18.5  # +6 mm
    rh_base = predict_static_rh_kinematic(kinematic, base)
    rh_delta = predict_static_rh_kinematic(kinematic, delta)
    diff = (
        rh_delta["setup_static_lf_ride_height_mm"]
        - rh_base["setup_static_lf_ride_height_mm"]
    )
    assert 5.0 < diff < 7.0, (
        f"6 mm front pushrod move should yield ~6 mm LF static RH delta, "
        f"got {diff:.3f}mm -- the regression the prior surrogate caused "
        f"(0.1 mm) must not return."
    )


def test_fit_refuses_to_ship_below_r2_threshold() -> None:
    """Heavy noise drops R^2 below 0.98; channel must be rejected."""
    sid_to_params, sid_to_readouts = _synthetic_corpus(
        n_sessions=20, noise_mm=25.0,
    )
    kinematic = fit_static_rh_kinematic("bmw", sid_to_params, sid_to_readouts)
    # Below threshold, none of the four channels should ship.
    assert len(kinematic.channels) == 0
    assert set(kinematic.rejected_channels) == set(STATIC_RH_CHANNELS)
    assert not kinematic.is_ready()


def test_fit_refuses_too_few_sessions() -> None:
    """Less than the minimum sample count produces no fit at all."""
    sid_to_params, sid_to_readouts = _synthetic_corpus(n_sessions=5)
    kinematic = fit_static_rh_kinematic("acura", sid_to_params, sid_to_readouts)
    assert not kinematic.is_ready()


def test_predict_returns_empty_when_kinematic_is_none() -> None:
    assert predict_static_rh_kinematic(None, {}) == {}


def test_predict_only_returns_shipped_channels() -> None:
    """When some channels reject and others ship, only shipped ones come back.

    Constructs a corpus where one channel has very noisy targets (low R^2)
    while the others remain clean. The kinematic predict must only emit
    the clean channels so the surrogate fallback handles the rest.
    """
    sid_to_params, sid_to_readouts = _synthetic_corpus(n_sessions=40, noise_mm=0.0)
    # Add heavy noise to RR only.
    rng = np.random.default_rng(0)
    for sid in sid_to_readouts:
        sid_to_readouts[sid]["setup_static_rr_ride_height_mm"] += float(
            rng.normal(0.0, 12.0),
        )
    kinematic = fit_static_rh_kinematic("bmw", sid_to_params, sid_to_readouts)
    out = predict_static_rh_kinematic(kinematic, {f: 0.0 for f in STATIC_RH_FEATURES})
    assert "setup_static_rr_ride_height_mm" not in out
    assert "setup_static_lf_ride_height_mm" in out


def test_kinematic_slot_type_safety_round_trips() -> None:
    """``StaticRhKinematic`` survives the PhysicsModel pickle slot validator.

    Type-safety on revive (P1.4) must not reject a real kinematic fit
    placed in the slot; a wrong type (e.g. a plain dict) must be rejected.
    """
    from racingoptimizer.physics.model import _validate_pickle_slots
    sid_to_params, sid_to_readouts = _synthetic_corpus(n_sessions=40, noise_mm=0.0)
    kinematic = fit_static_rh_kinematic("bmw", sid_to_params, sid_to_readouts)
    assert isinstance(kinematic, StaticRhKinematic)
    _validate_pickle_slots({"static_rh_kinematic": kinematic})  # must not raise
    _validate_pickle_slots({"static_rh_kinematic": None})  # also fine
    with pytest.raises(TypeError, match="static_rh_kinematic"):
        _validate_pickle_slots({"static_rh_kinematic": {"a": 1}})
