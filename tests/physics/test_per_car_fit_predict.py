"""Per-car fit + predict + pickle smoke for slice E (physics fitter).

Closes the verification gap left by the BMW-Sebring-only suite: fits a
PhysicsModel on real .ibt fixtures for every GTP car that has fixtures in
`ibtfiles/`, then asserts predict + pickle round-trip work.

Acura known divergence: Acura sessions drop the four `*shockDefl` channels.
The fitter should mark the corresponding output-channel fitters as untrained
rather than crash; this module pins that contract.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from racingoptimizer.confidence import Confidence
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner import CornerPhaseKey, Phase
from racingoptimizer.physics import PhysicsModel
from racingoptimizer.physics.model import CornerPhaseStateWithConfidence
from tests.physics.conftest import PER_CAR_CASE_IDS, PER_CAR_CASES

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    ("car", "track", "fixtures"), PER_CAR_CASES, ids=PER_CAR_CASE_IDS
)
def test_fit_per_car(
    car: str,
    track: str,
    fixtures: tuple[Path, ...],
    per_car_model_factory,
) -> None:
    if not fixtures:
        pytest.skip(f"no .ibt fixtures for car={car}")

    model, _ = per_car_model_factory(car, track, fixtures)

    assert isinstance(model, PhysicsModel)
    assert model.car == car
    assert isinstance(model.untrained_parameters, tuple)
    for name in model.untrained_parameters:
        assert isinstance(name, str)
    assert any(rec.fitter.is_trained for rec in model.fitters.values()), (
        f"car={car}: no fitter trained — required at least one trained record"
    )


@pytest.mark.parametrize(
    ("car", "track", "fixtures"), PER_CAR_CASES, ids=PER_CAR_CASE_IDS
)
def test_predict_per_car(
    car: str,
    track: str,
    fixtures: tuple[Path, ...],
    per_car_model_factory,
) -> None:
    if not fixtures:
        pytest.skip(f"no .ibt fixtures for car={car}")

    model, _ = per_car_model_factory(car, track, fixtures)
    keys = sorted(model.fitters.keys())
    assert keys, f"car={car}: model has no fitter keys"
    _param, corner_id, phase_str, _channel = keys[0]
    cpkey = CornerPhaseKey(
        session_id=model.session_ids[0],
        lap_index=1,
        corner_id=int(corner_id),
        phase=Phase(phase_str),
    )
    env = EnvironmentFrame(
        air_density=1.225, track_temp_c=25.0, wind_vel_ms=2.0,
        wind_dir_deg=90.0, track_wetness=0.0,
    )
    out = model.predict(dict(model.baseline_setup), env, cpkey)
    assert isinstance(out, CornerPhaseStateWithConfidence)
    assert out.corner_phase_key == cpkey
    assert len(out.states) >= 1, f"car={car}: predict produced no states for {cpkey!r}"
    for ch, conf in out.states.items():
        assert isinstance(ch, str)
        assert isinstance(conf, Confidence)


@pytest.mark.parametrize(
    ("car", "track", "fixtures"), PER_CAR_CASES, ids=PER_CAR_CASE_IDS
)
def test_pickle_round_trip_per_car(
    car: str,
    track: str,
    fixtures: tuple[Path, ...],
    per_car_model_factory,
) -> None:
    if not fixtures:
        pytest.skip(f"no .ibt fixtures for car={car}")

    model, _ = per_car_model_factory(car, track, fixtures)
    blob = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    revived = pickle.loads(blob)  # noqa: S301 — controlled fixture
    assert revived.car == model.car
    assert revived.session_ids == model.session_ids
    assert pickle.dumps(revived, protocol=pickle.HIGHEST_PROTOCOL) == blob


def test_acura_shock_channels_marked_untrained_not_crashed(
    per_car_model_factory,
) -> None:
    """Acura .ibt files lack the four `*shockDefl` channels.

    The fitter must skip those output channels gracefully (no FitRecord
    persisted for them) rather than crash. The corner-phase aggregator
    already filters absent channels out of its schema; this guards the
    contract end-to-end from the physics-fitter side.
    """
    cases = {car: (track, fixtures) for car, track, fixtures in PER_CAR_CASES}
    if "acura" not in cases or not cases["acura"][1]:
        pytest.skip("no Acura fixtures")
    track, fixtures = cases["acura"]
    model, _ = per_car_model_factory("acura", track, fixtures)
    shock_channels = {
        "lf_shock_defl_p99_mm",
        "rf_shock_defl_p99_mm",
        "lr_shock_defl_p99_mm",
        "rr_shock_defl_p99_mm",
    }
    fitted_channels = {channel for (_p, _c, _ph, channel) in model.fitters}
    assert not (fitted_channels & shock_channels), (
        "Acura should not have shock-deflection fitters (channels missing in IBT); "
        f"got fitters for {fitted_channels & shock_channels}"
    )
