"""SetupRecommendation determinism + clamp + breakdown coverage."""
from __future__ import annotations

import pickle

import pytest

from racingoptimizer.confidence import Confidence
from racingoptimizer.constraints import load_constraints
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.physics import SetupRecommendation
from racingoptimizer.physics.recommend import recommend


@pytest.fixture
def bmw_model_track(bmw_model_session):
    return bmw_model_session


def test_recommend_returns_setup_recommendation(bmw_model_track) -> None:
    model, track, _ = bmw_model_track
    constraints = load_constraints()
    env = EnvironmentFrame(
        air_density=1.18, track_temp_c=24.0, wind_vel_ms=2.5,
        wind_dir_deg=120.0, track_wetness=0.0,
    )
    rec = recommend(model, track, env, constraints)
    assert isinstance(rec, SetupRecommendation)
    assert rec.car == model.car
    assert rec.track == track
    assert rec.env == env
    # Every parameter has (value, Confidence).
    for name, (value, conf) in rec.parameters.items():
        assert isinstance(name, str)
        assert isinstance(value, float)
        assert isinstance(conf, Confidence)


def test_recommend_parameters_within_bounds(bmw_model_track) -> None:
    model, track, _ = bmw_model_track
    constraints = load_constraints()
    env = EnvironmentFrame(
        air_density=1.18, track_temp_c=24.0, wind_vel_ms=2.5,
        wind_dir_deg=120.0, track_wetness=0.0,
    )
    rec = recommend(model, track, env, constraints)
    for name, (value, _conf) in rec.parameters.items():
        bound = constraints.bounds(model.car, name)
        if bound is None:
            continue
        lo, hi = bound
        assert lo <= value <= hi, f"{name}={value} outside [{lo}, {hi}]"


def test_recommend_score_breakdown_per_corner_phase(bmw_model_track) -> None:
    model, track, _ = bmw_model_track
    constraints = load_constraints()
    env = EnvironmentFrame(
        air_density=1.18, track_temp_c=24.0, wind_vel_ms=2.5,
        wind_dir_deg=120.0, track_wetness=0.0,
    )
    rec = recommend(model, track, env, constraints)
    assert isinstance(rec.score_breakdown, dict)
    if not rec.score_breakdown:
        pytest.skip("model produced no per-(corner,phase) breakdown")
    for cpkey, value in rec.score_breakdown.items():
        assert isinstance(value, float)
        # Per-corner contribution is non-negative (utilizations and weights both
        # ≥ 0).
        assert value >= 0.0
        assert cpkey.session_id == "<recommend-virtual>"


def test_recommend_determinism(bmw_model_track) -> None:
    model, track, _ = bmw_model_track
    constraints = load_constraints()
    env = EnvironmentFrame(
        air_density=1.18, track_temp_c=24.0, wind_vel_ms=2.5,
        wind_dir_deg=120.0, track_wetness=0.0,
    )
    rec_a = recommend(model, track, env, constraints)
    rec_b = recommend(model, track, env, constraints)
    assert pickle.dumps(rec_a) == pickle.dumps(rec_b)


def test_recommend_via_model_method(bmw_model_track) -> None:
    """PhysicsModel.recommend / .score_setup delegate to module functions."""
    model, track, _ = bmw_model_track
    constraints = load_constraints()
    env = EnvironmentFrame(
        air_density=1.18, track_temp_c=24.0, wind_vel_ms=2.5,
        wind_dir_deg=120.0, track_wetness=0.0,
    )
    rec = model.recommend(track, env, constraints)
    assert isinstance(rec, SetupRecommendation)
    score = model.score_setup(dict(model.baseline_setup), track, env)
    assert isinstance(score, float)
