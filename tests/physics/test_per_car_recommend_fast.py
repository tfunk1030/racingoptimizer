"""Fast per-car recommend smoke (T2.5) -- one BMW Sebring case in merge gate."""
from __future__ import annotations

import pytest

from racingoptimizer.confidence import Confidence
from racingoptimizer.constraints import load_constraints
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.physics import SetupRecommendation
from tests.physics.conftest import BMW_SEBRING_IBT

_ENV = EnvironmentFrame(
    air_density=1.225, track_temp_c=25.0, wind_vel_ms=2.0,
    wind_dir_deg=90.0, track_wetness=0.0,
)


@pytest.mark.skipif(
    not BMW_SEBRING_IBT.exists(),
    reason="BMW Sebring IBT fixture missing",
)
def test_recommend_bmw_sebring_fast(per_car_model_factory) -> None:
    """Single-car DE smoke promoted out of the slow-only suite."""
    model, _ = per_car_model_factory(
        "bmw", "sebring_international", (BMW_SEBRING_IBT,),
    )
    constraints = load_constraints()
    rec = model.recommend("sebring_international", _ENV, constraints)

    assert isinstance(rec, SetupRecommendation)
    assert rec.car == "bmw"
    assert rec.parameters
    for name, (value, conf) in rec.parameters.items():
        assert isinstance(value, float)
        assert isinstance(conf, Confidence)
        bound = constraints.bounds("bmw", name)
        if bound is not None:
            lo, hi = bound
            assert lo <= value <= hi, f"{name}={value} outside [{lo}, {hi}]"
    assert rec.score_breakdown
