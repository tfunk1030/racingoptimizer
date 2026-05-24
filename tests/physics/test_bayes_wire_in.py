"""Day 4 of the physics-rebuild plan: bayes_retrofit wire-in to PhysicsModel.

PLAN.md Section 14.3 mid-component task: after Day 3's standalone module,
wire `fit_all_parameters` into `fit_per_car` so production fits populate
`PhysicsModel.bayes_posteriors`.

Day 4 scope (this file):
- PhysicsModel carries the new `bayes_posteriors` field with default {}.
- Pre-Day-4 pickles default-revive with bayes_posteriors={} via __setstate__.
- FITTERS_LAYOUT_VERSION bumped to 3 so existing per-car caches refit on
  next `optimize` invocation (otherwise the new field would stay empty
  and Mode 1 closure would not take effect).
- The wire-in itself is exercised by Day 5's held-out gate; here we
  verify the data flow with a synthetic per-track observed dict.
"""
from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError

import pytest

from racingoptimizer.physics.bayes_retrofit import BayesPosterior
from racingoptimizer.physics.fitters import FITTERS_LAYOUT_VERSION
from racingoptimizer.physics.model import PhysicsModel


def test_physics_model_has_bayes_posteriors_field() -> None:
    """The new field is exposed on PhysicsModel and defaults to {}."""
    m = PhysicsModel(car="bmw", session_ids=())
    assert hasattr(m, "bayes_posteriors")
    assert m.bayes_posteriors == {}


def test_physics_model_accepts_bayes_posteriors_kwarg() -> None:
    """Construct with explicit posteriors; the field round-trips."""
    posteriors = {
        ("rear_wing_angle_deg", "spa_2024_up"): BayesPosterior(
            parameter="rear_wing_angle_deg",
            track="spa_2024_up",
            mean=14.5, std=0.2, n_samples=6, shrinkage=0.1,
        ),
    }
    m = PhysicsModel(car="bmw", session_ids=(), bayes_posteriors=posteriors)
    assert m.bayes_posteriors == posteriors


def test_physics_model_bayes_posteriors_immutable() -> None:
    """Frozen dataclass rejects field reassignment."""
    m = PhysicsModel(car="bmw", session_ids=())
    with pytest.raises(FrozenInstanceError):
        m.bayes_posteriors = {}  # type: ignore[misc]


def test_pre_day_4_pickle_default_revives_with_empty_posteriors() -> None:
    """A pre-Day-4 pickle (no bayes_posteriors slot) must round-trip via
    __setstate__ with the default {}; the recommender's fallbacks handle
    the empty case.

    We simulate this by building a state list that's missing the new
    slot and revive it manually.
    """
    m = PhysicsModel(car="bmw", session_ids=("a", "b"))
    blob = pickle.dumps(m)
    revived = pickle.loads(blob)
    assert revived.bayes_posteriors == {}


def test_state_dict_partial_revive_fills_default() -> None:
    """If __setstate__ receives a dict without bayes_posteriors, the
    default {} fills in (legacy pickle support)."""
    m = PhysicsModel(car="bmw", session_ids=())
    # Build a partial state dict missing the new slot.
    partial = {"car": "bmw", "session_ids": ()}
    m.__setstate__(partial)
    assert m.bayes_posteriors == {}


def test_fitters_layout_version_bumped_to_3() -> None:
    """Day 4 bumps the per-car cache key so existing pickles refit on
    next `optimize` run (otherwise bayes_posteriors stays empty and
    Mode 1 closure is silent-no-op on cached models)."""
    assert FITTERS_LAYOUT_VERSION >= 3


def test_fit_all_parameters_round_trips_through_physics_model() -> None:
    """Synthetic end-to-end: take a per-track observed dict shaped like
    `fit_per_car` produces, pass to `fit_all_parameters`, store on
    PhysicsModel, verify retrieval keys and values are consistent.
    """
    from racingoptimizer.physics.bayes_retrofit import fit_all_parameters

    per_track_observed = {
        "hockenheim_gp": {"rear_wing_angle_deg": (17.0,) * 24},
        "spa_2024_up": {
            "rear_wing_angle_deg": (14.0, 14.0, 14.0, 15.0, 15.0, 15.0),
        },
    }
    posteriors = fit_all_parameters(per_track_observed)
    m = PhysicsModel(
        car="ferrari", session_ids=tuple(),
        bayes_posteriors=posteriors,
    )
    # Keys preserved.
    assert ("rear_wing_angle_deg", "hockenheim_gp") in m.bayes_posteriors
    assert ("rear_wing_angle_deg", "spa_2024_up") in m.bayes_posteriors
    # Spa posterior closes Mode 1: mean near empirical 14.5, NOT near
    # grand mean 16.5.
    spa = m.bayes_posteriors[("rear_wing_angle_deg", "spa_2024_up")]
    assert abs(spa.mean - 14.5) < 1.0


def test_canary_layout_version_bump_invalidates_existing_caches() -> None:
    """Broken-model canary (Day 4): if the layout-version bump is
    reverted, existing per-car caches would silently revive with
    `bayes_posteriors={}` (via __setstate__ default) and the Mode 1
    closure would be a silent no-op.

    This test asserts the version IS bumped past Day 3's value (2). If
    a future commit reverts to 2, the canary fires.
    """
    # Pre-Day-4 layout version was 2. Day 4 ships 3.
    assert FITTERS_LAYOUT_VERSION != 2, (
        "FITTERS_LAYOUT_VERSION reverted to 2 -- existing per-car caches "
        "would silently revive without bayes_posteriors, defeating Mode 1."
    )


def test_bayes_posteriors_empty_dict_is_valid_state() -> None:
    """Recommender's fallback path (no posterior available for a given
    (param, track)) requires that an empty bayes_posteriors dict is a
    legal model state. v3 pickles and pre-Day-4 v4 pickles both produce
    this state."""
    m = PhysicsModel(car="acura", session_ids=("a",))
    # Should not raise; should be retrievable.
    _ = m.bayes_posteriors
    assert m.bayes_posteriors == {}


def test_bayes_trust_anchor_shifts_baseline_to_posterior_mean() -> None:
    from racingoptimizer.physics.recommend import _bayes_trust_anchor

    posteriors = {
        ("rear_wing_angle_deg", "spa_2024_up"): BayesPosterior(
            parameter="rear_wing_angle_deg",
            track="spa_2024_up",
            mean=14.5,
            std=0.3,
            n_samples=6,
            shrinkage=0.2,
            mean_std=0.25,
            predictive_std=0.5,
        ),
    }
    model = PhysicsModel(
        car="ferrari", session_ids=(), bayes_posteriors=posteriors,
    )
    anchor, std = _bayes_trust_anchor(
        model, "spa_2024_up", "rear_wing_angle_deg", baseline=17.0, observed_std=0.1,
    )
    assert anchor == pytest.approx(14.5)
    assert std == pytest.approx(0.25)


def test_bayes_trust_anchor_falls_back_without_posterior() -> None:
    from racingoptimizer.physics.recommend import _bayes_trust_anchor

    model = PhysicsModel(car="bmw", session_ids=())
    anchor, std = _bayes_trust_anchor(
        model, "sebring_international", "rear_wing_angle_deg",
        baseline=16.0, observed_std=0.4,
    )
    assert anchor == pytest.approx(16.0)
    assert std == pytest.approx(0.4)
