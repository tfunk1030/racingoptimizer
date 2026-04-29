"""Score-locality test (spec §13).

Tweak a single fitter, assert the per-(corner, phase) score_breakdown delta
matches that fitter's (corner, phase) only — every other entry is byte-identical.
"""
from __future__ import annotations

import numpy as np
import pytest

from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.physics.score import score_breakdown


@pytest.fixture
def bmw_model(bmw_model_session):
    model, track, _root = bmw_model_session
    return model, track


def test_score_breakdown_locality(bmw_model) -> None:
    model, track = bmw_model
    env = EnvironmentFrame(
        air_density=1.18, track_temp_c=24.0, wind_vel_ms=2.5,
        wind_dir_deg=120.0, track_wetness=0.0,
    )
    setup = dict(model.baseline_setup)

    base = score_breakdown(model, setup, track, env)
    if not base:
        pytest.skip("no score breakdown produced from fixture")

    # Pick a key whose phase has fitters; pick one fitter under it.
    target_corner_id, target_phase = next(iter(base.keys()))[2:4]
    fitter_keys = [
        k for k in model.fitters
        if k[1] == target_corner_id and k[2] == str(target_phase)
    ]
    if not fitter_keys:
        pytest.skip("no fitter matches the chosen (corner, phase)")
    target_key = fitter_keys[0]
    fitter = model.fitters[target_key].fitter

    # Monkey-patch the chosen fitter's predict to return a constant offset.
    original_predict = fitter.predict
    def constant_predict(X):
        mean = np.full((np.asarray(X).shape[0],), 999.0, dtype=np.float64)
        std = np.zeros_like(mean)
        return mean, std
    fitter.predict = constant_predict  # type: ignore[assignment]

    try:
        perturbed = score_breakdown(model, setup, track, env)
    finally:
        fitter.predict = original_predict  # type: ignore[assignment]

    target_cpkey = next(
        k for k in base
        if k.corner_id == target_corner_id and str(k.phase) == str(target_phase)
    )
    # Other (corner, phase) entries must be byte-identical floats.
    for cpkey, base_val in base.items():
        if cpkey == target_cpkey:
            continue
        assert perturbed[cpkey] == base_val, (
            f"unrelated key {cpkey} drifted from {base_val} to {perturbed[cpkey]}"
        )
