"""Per-car recommend smoke for slice E (physics fitter, U10).

Closes the verification gap: runs `recommend(...)` against a fitted
PhysicsModel for every GTP car that has fixtures in `ibtfiles/`, then
asserts post-clamp correctness, deterministic output, and a non-empty
score breakdown.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from racingoptimizer.confidence import Confidence
from racingoptimizer.constraints import load_constraints
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.physics import SetupRecommendation

from tests.physics.conftest import PER_CAR_CASE_IDS, PER_CAR_CASES

pytestmark = pytest.mark.slow

_ENV = EnvironmentFrame(
    air_density=1.225, track_temp_c=25.0, wind_vel_ms=2.0,
    wind_dir_deg=90.0, track_wetness=0.0,
)


@pytest.mark.parametrize(
    ("car", "track", "fixtures"), PER_CAR_CASES, ids=PER_CAR_CASE_IDS
)
def test_recommend_per_car(
    car: str,
    track: str,
    fixtures: tuple[Path, ...],
    per_car_model_factory,
) -> None:
    if not fixtures:
        pytest.skip(f"no .ibt fixtures for car={car}")

    model, _ = per_car_model_factory(car, track, fixtures)
    constraints = load_constraints()
    rec = model.recommend(track, _ENV, constraints)

    assert isinstance(rec, SetupRecommendation)
    assert rec.car == car
    assert rec.track == track
    assert rec.env == _ENV

    assert rec.parameters, f"car={car}: recommend produced no parameters"
    for name, entry in rec.parameters.items():
        assert isinstance(name, str)
        assert isinstance(entry, tuple)
        value, conf = entry
        assert isinstance(value, float)
        assert isinstance(conf, Confidence)
        bound = constraints.bounds(car, name)
        if bound is not None:
            lo, hi = bound
            assert lo <= value <= hi, (
                f"car={car} {name}={value} outside [{lo}, {hi}] (post-clamp violated)"
            )

    assert rec.score_breakdown, (
        f"car={car}: recommend produced an empty score_breakdown"
    )
    for cpkey, value in rec.score_breakdown.items():
        assert isinstance(value, float)
        assert value >= 0.0
        assert cpkey.session_id == "<recommend-virtual>"

    assert isinstance(rec.untrained_parameters, tuple)
    for name in rec.untrained_parameters:
        assert isinstance(name, str)


@pytest.mark.parametrize(
    ("car", "track", "fixtures"), PER_CAR_CASES, ids=PER_CAR_CASE_IDS
)
def test_recommend_determinism_per_car(
    car: str,
    track: str,
    fixtures: tuple[Path, ...],
    per_car_model_factory,
) -> None:
    if not fixtures:
        pytest.skip(f"no .ibt fixtures for car={car}")

    model, _ = per_car_model_factory(car, track, fixtures)
    constraints = load_constraints()
    rec_a = model.recommend(track, _ENV, constraints)
    rec_b = model.recommend(track, _ENV, constraints)
    assert pickle.dumps(rec_a) == pickle.dumps(rec_b)
