"""Tests for the post-physics-rebuild guardrail wiring into recommend.py.

Covers:
  - `PhysicsModel.axle_grip_ceilings` field default (None) and pickle
    backward compatibility with legacy v3 pickles.
  - `_axle_guardrail_penalty`: no-op when ceilings empty, fires when
    predicted lat-G yields axle ratio > ceiling, corner-time-weighted.
  - End-to-end `recommend()` is a no-op when ceilings is None (preserves
    legacy behavior).
"""
from __future__ import annotations

import pickle
from dataclasses import replace

import pytest

from racingoptimizer.confidence import Confidence
from racingoptimizer.corner import CornerPhaseKey
from racingoptimizer.physics.axle_grip import AxleGripCeiling
from racingoptimizer.physics.fitters import FITTERS_LAYOUT_VERSION
from racingoptimizer.physics.model import (
    CornerPhaseStateWithConfidence,
    PhysicsModel,
)
from racingoptimizer.physics.recommend import (
    _GUARDRAIL_PENALTY_OVER_CEILING,
    _axle_guardrail_penalty,
)


def test_fitters_layout_version_at_or_above_four() -> None:
    """Post-rebuild wiring requires >= v4 to force refit of legacy pickles
    so that PhysicsModel.axle_grip_ceilings populates. v5+ adds further
    parallel-work changes (hybrid score path, etc.)."""
    assert FITTERS_LAYOUT_VERSION >= 4


def test_physics_model_axle_ceilings_field_defaults_to_none() -> None:
    """New PhysicsModel constructions get ceilings=None unless populated."""
    model = PhysicsModel(car="bmw", session_ids=())
    assert model.axle_grip_ceilings is None


def test_physics_model_legacy_pickle_revives_with_ceilings_none() -> None:
    """v3 pickles lacking the field deserialize with ceilings=None.

    Simulates the legacy pickle shape by constructing a PhysicsModel
    and then dropping the new field from `__getstate__`, then reviving.
    The `__setstate__` setdefault must restore `axle_grip_ceilings=None`.
    """
    legacy_state: dict[str, object] = {
        "car": "bmw",
        "session_ids": (),
        "track_models_used": {},
        "fitters": {},
        "ontology": {},
        "constraints": None,
        "untrained_parameters": (),
        "aero_correction_available": False,
        "baseline_setup": {},
        "seed": 0xC0FFEE,
        "car_baselines": None,
        "feature_schema_version": 3,
        "parameter_observed_std": {},
        "per_track_parameter_observed": {},
        "bayes_posteriors": {},
        # axle_grip_ceilings deliberately omitted (legacy shape).
    }
    revived = PhysicsModel.__new__(PhysicsModel)
    revived.__setstate__(legacy_state)
    assert revived.axle_grip_ceilings is None
    assert revived.car == "bmw"


def test_physics_model_pickle_round_trip_preserves_ceilings() -> None:
    """A model with ceilings serialises through pickle and re-emerges intact."""
    ceilings = {
        "front": AxleGripCeiling(
            car="bmw", axle="front", mu_peak=1.5,
            n_samples=500, n_above_ceiling=25, percentile_used=95.0,
        ),
        "rear": AxleGripCeiling(
            car="bmw", axle="rear", mu_peak=1.4,
            n_samples=500, n_above_ceiling=25, percentile_used=95.0,
        ),
    }
    model = PhysicsModel(
        car="bmw", session_ids=("sid_1",),
        axle_grip_ceilings=ceilings,
    )
    revived = pickle.loads(pickle.dumps(model))
    assert revived.axle_grip_ceilings is not None
    assert revived.axle_grip_ceilings["front"].mu_peak == pytest.approx(1.5)
    assert revived.axle_grip_ceilings["rear"].mu_peak == pytest.approx(1.4)


# ---- _axle_guardrail_penalty -------------------------------------------


class _FakeModel:
    """Minimal model surface for `_axle_guardrail_penalty`.

    The helper only calls `.predict(setup, env, cpkey, corner_archetype=...)`
    and reads `.car`. We bypass the real PhysicsModel + fitter machinery
    to exercise the penalty logic with controlled lat-G predictions.
    """
    def __init__(self, car: str, lat_g_max_per_corner: dict[int, float]) -> None:
        self.car = car
        self._lat_g_max = lat_g_max_per_corner

    def predict(
        self,
        setup: dict[str, float],  # noqa: ARG002
        env,  # noqa: ARG002
        cpkey: CornerPhaseKey,
        *,
        corner_archetype=None,  # noqa: ARG002
    ) -> CornerPhaseStateWithConfidence:
        lat = self._lat_g_max.get(int(cpkey.corner_id), 1.0)
        return CornerPhaseStateWithConfidence(
            corner_phase_key=cpkey,
            states={
                "accel_lat_g_max": Confidence(
                    value=float(lat), lo=float(lat), hi=float(lat),
                    n_samples=10, regime="dense",
                ),
            },
            untrained_channels=(),
        )


def _make_ceilings(front_mu: float = 1.5, rear_mu: float = 1.5) -> dict[str, AxleGripCeiling]:
    return {
        "front": AxleGripCeiling(
            car="bmw", axle="front", mu_peak=front_mu,
            n_samples=500, n_above_ceiling=25, percentile_used=95.0,
        ),
        "rear": AxleGripCeiling(
            car="bmw", axle="rear", mu_peak=rear_mu,
            n_samples=500, n_above_ceiling=25, percentile_used=95.0,
        ),
    }


def _make_keys(corner_ids: list[int], phase: str = "mid_corner") -> list[tuple[int, str]]:
    return [(c, phase) for c in corner_ids]


def test_penalty_zero_when_ceilings_dict_empty() -> None:
    model = _FakeModel("bmw", {1: 1.5})
    penalty = _axle_guardrail_penalty(
        model, {}, env=None,
        weights={1: 1.0},
        ceilings={},  # missing both front and rear keys
        schedule=None,
        keys=_make_keys([1]),
    )
    assert penalty == 0.0


def test_penalty_zero_when_setup_well_within_ceiling() -> None:
    """lat_g=0.8 with mu_peak=1.5 -> ratio ~0.5; margin well under 1.0."""
    model = _FakeModel("bmw", {1: 0.8})
    penalty = _axle_guardrail_penalty(
        model, {}, env=None,
        weights={1: 1.0},
        ceilings=_make_ceilings(front_mu=1.5, rear_mu=1.5),
        schedule=None,
        keys=_make_keys([1]),
    )
    assert penalty == 0.0


def test_penalty_zero_when_phase_not_mid_corner() -> None:
    """Only mid_corner phases trigger the penalty (others are driver-dominated)."""
    model = _FakeModel("bmw", {1: 3.0})  # absurdly high lat_g
    penalty = _axle_guardrail_penalty(
        model, {}, env=None,
        weights={1: 1.0},
        ceilings=_make_ceilings(front_mu=0.5, rear_mu=0.5),  # tiny ceiling
        schedule=None,
        keys=_make_keys([1], phase="exit"),
    )
    assert penalty == 0.0


def test_penalty_fires_when_predicted_lat_g_exceeds_ceiling() -> None:
    """Tiny ceiling + normal lat_g -> margin > 1.0 -> penalty applied."""
    model = _FakeModel("bmw", {1: 2.0})
    penalty = _axle_guardrail_penalty(
        model, {}, env=None,
        weights={1: 1.0},
        ceilings=_make_ceilings(front_mu=0.3, rear_mu=0.3),
        schedule=None,
        keys=_make_keys([1]),
    )
    assert penalty == pytest.approx(_GUARDRAIL_PENALTY_OVER_CEILING)


def test_penalty_corner_time_weighted() -> None:
    """Corners with larger weight contribute proportionally more penalty."""
    model = _FakeModel("bmw", {1: 2.0, 2: 2.0})  # both over ceiling
    penalty = _axle_guardrail_penalty(
        model, {}, env=None,
        weights={1: 0.2, 2: 0.8},
        ceilings=_make_ceilings(front_mu=0.3, rear_mu=0.3),
        schedule=None,
        keys=_make_keys([1, 2]),
    )
    expected = (0.2 + 0.8) * _GUARDRAIL_PENALTY_OVER_CEILING
    assert penalty == pytest.approx(expected)


def test_penalty_skips_subthreshold_lat_g() -> None:
    """lat_g < 0.5 means the ceiling fit's mid-corner support doesn't apply."""
    model = _FakeModel("bmw", {1: 0.3})  # below mid-corner threshold
    penalty = _axle_guardrail_penalty(
        model, {}, env=None,
        weights={1: 1.0},
        ceilings=_make_ceilings(front_mu=0.1, rear_mu=0.1),  # would otherwise fire
        schedule=None,
        keys=_make_keys([1]),
    )
    assert penalty == 0.0


def test_penalty_handles_missing_lat_g_channel() -> None:
    """When the surrogate has no lat_g_max for a corner, that corner is skipped."""
    class _NoLatG:
        car = "bmw"
        def predict(self, setup, env, cpkey, *, corner_archetype=None):  # noqa: D401, ARG002
            return CornerPhaseStateWithConfidence(
                corner_phase_key=cpkey,
                states={},
                untrained_channels=(),
            )
    penalty = _axle_guardrail_penalty(
        _NoLatG(), {}, env=None,
        weights={1: 1.0},
        ceilings=_make_ceilings(front_mu=0.1, rear_mu=0.1),
        schedule=None,
        keys=_make_keys([1]),
    )
    assert penalty == 0.0


def test_bmw_sebring_fit_populates_ceilings(bmw_model_session) -> None:
    """The BMW Sebring fixture's fit must produce non-None axle ceilings.

    Confirms `_fit_axle_ceilings_for_car` runs successfully end-to-end
    on a real corpus and the result is persisted on PhysicsModel.
    Without this, the guardrail penalty is silently inactive and we
    wouldn't notice via the None-case tests above.
    """
    model, _track, _root = bmw_model_session
    assert model.axle_grip_ceilings is not None, (
        "BMW Sebring fit should produce ceilings (corpus has thousands "
        "of mid-corner samples); None indicates the helper is failing "
        "silently or wiring regressed"
    )
    assert "front" in model.axle_grip_ceilings
    assert "rear" in model.axle_grip_ceilings
    front = model.axle_grip_ceilings["front"]
    rear = model.axle_grip_ceilings["rear"]
    # Empirical chassis-level ratios for BMW GTP land in [2.0, 3.5]
    # per CLAUDE.md "observed values 2.5-3.0 are normal because
    # chassis-level Fz in the denominator excludes aero downforce".
    assert 1.5 <= front.mu_peak <= 4.0, f"front mu_peak={front.mu_peak}"
    assert 1.5 <= rear.mu_peak <= 4.0, f"rear mu_peak={rear.mu_peak}"
    assert front.n_samples >= 100
    assert rear.n_samples >= 100


def test_penalty_no_ceilings_no_op_in_recommend(bmw_model_session) -> None:
    """Models without ceilings (`axle_grip_ceilings=None`) recommend unchanged.

    Backward-compat check: legacy v3 pickles loaded onto v4 code revive
    with ceilings=None, and the DE objective must skip the penalty branch
    entirely. This test uses the session-scoped BMW Sebring fit.
    """
    model, track, root = bmw_model_session
    from racingoptimizer.constraints import load_constraints
    from racingoptimizer.context import EnvironmentFrame
    from racingoptimizer.physics.corner_schedule import build_corner_schedule
    from racingoptimizer.physics.recommend import recommend

    constraints = load_constraints()
    env = EnvironmentFrame(
        air_density=1.18, track_temp_c=24.0, wind_vel_ms=2.5,
        wind_dir_deg=120.0, track_wetness=0.0,
    )

    # Per-car v4 models require a target-track corner schedule.
    schedule = build_corner_schedule(
        list(model.session_ids), corpus_root=root,
    )

    # Force ceilings = None to simulate a legacy pickle revived on v5.
    legacy_model = replace(model, axle_grip_ceilings=None)
    rec = recommend(legacy_model, track, env, constraints, schedule=schedule)
    assert rec.parameters, "recommend should still produce parameters"
    # Score breakdown is positive (no penalty applied).
    assert sum(rec.score_breakdown.values()) > 0.0
