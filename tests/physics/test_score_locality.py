"""Per-output-channel locality test (Stage-3 weakened version).

Pre-Stage-3 the model fit one fitter per (parameter, corner, phase, channel)
and `predict` summed per-parameter contributions per channel — so monkey-
patching one fitter changed exactly one (corner, phase) cell. That property
*affirmed by inversion* the architecture was non-coupled, in violation of
VISION §3.

Stage-3 keys fitters by (corner, phase, channel) and feeds the FULL setup
vector. Perturbing one fitter should still affect only its own (corner,
phase) cell — that fitter is local to its (corner, phase) cell — but
perturbing one *setup parameter* now propagates to every cell because
every cell's fitter depends on the joint vector. The propagation property
is asserted by `tests/physics/test_coupling.py`; this file keeps the
weaker locality property: monkey-patching ONE fitter's `predict` does not
leak into other (corner, phase) cells.
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


def test_score_breakdown_per_fitter_locality(bmw_model) -> None:
    """Patching ONE fitter's predict only changes that fitter's (corner, phase) cell.

    This is the per-fitter locality invariant — every other (corner, phase)
    cell still queries its own fitter, which is independent. The cross-
    parameter "chase the chain" propagation lives in test_coupling.py.
    """
    model, track = bmw_model
    env = EnvironmentFrame(
        air_density=1.18, track_temp_c=24.0, wind_vel_ms=2.5,
        wind_dir_deg=120.0, track_wetness=0.0,
    )
    setup = dict(model.baseline_setup)

    base = score_breakdown(model, setup, track, env)
    if not base:
        pytest.skip("no score breakdown produced from fixture")

    # Stage-3 keys are (corner_id, phase, channel); pick one and patch it.
    target_corner_id, target_phase = next(iter(base.keys()))[2:4]
    fitter_keys = [
        k for k in model.fitters
        if len(k) == 3 and k[0] == target_corner_id and k[1] == str(target_phase)
    ]
    if not fitter_keys:
        pytest.skip("no fitter matches the chosen (corner, phase)")
    target_key = fitter_keys[0]
    fitter = model.fitters[target_key].fitter

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
    # Other (corner, phase) entries must be byte-identical floats — each
    # has its own independent fitter family.
    for cpkey, base_val in base.items():
        if cpkey == target_cpkey:
            continue
        assert perturbed[cpkey] == base_val, (
            f"unrelated key {cpkey} drifted from {base_val} to {perturbed[cpkey]}"
        )
