"""Empirical physics fitter (slice E-2 / U9).

Public API:
    fit(car, session_ids, track_model, *, seed=..., k_folds=..., corpus_root=...) -> PhysicsModel
    PhysicsModel.predict(setup, env, corner_phase_key) -> CornerPhaseStateWithConfidence
    ontology_for(car), setup_value(car, parameter, setup_json), fittable_parameters(car, table)
    ParameterSpec, UntrainedError, InsufficientDataError

`score_setup` / `recommend` / `SetupRecommendation` are deferred to U10.
"""
from __future__ import annotations

from racingoptimizer.physics.exceptions import InsufficientDataError, UntrainedError
from racingoptimizer.physics.fitter import fit
from racingoptimizer.physics.model import (
    CornerPhaseStateWithConfidence,
    FitRecord,
    PhysicsModel,
)
from racingoptimizer.physics.ontology import (
    ParameterSpec,
    fittable_parameters,
    ontology_for,
    setup_value,
)

__all__ = [
    "CornerPhaseStateWithConfidence",
    "FitRecord",
    "InsufficientDataError",
    "ParameterSpec",
    "PhysicsModel",
    "UntrainedError",
    "fit",
    "fittable_parameters",
    "ontology_for",
    "setup_value",
]
