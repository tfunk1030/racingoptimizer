"""Empirical physics fitter (slice E).

Public API:
    fit(car, session_ids, track_model, *, seed=..., k_folds=..., corpus_root=...) -> PhysicsModel
    PhysicsModel.predict(setup, env, corner_phase_key) -> CornerPhaseStateWithConfidence
    PhysicsModel.score_setup(setup, track, env) -> float
    PhysicsModel.recommend(track, env, constraints) -> SetupRecommendation
    score_setup(model, setup, track, env) -> float
    recommend(model, track, env, constraints) -> SetupRecommendation
    weight_corners(track, model, ...) -> dict[corner_id, weight]
    ontology_for(car), setup_value(car, parameter, setup_json), fittable_parameters(car, table)
    PHASE_WEIGHTS, SUB_UTILIZATIONS
    SetupRecommendation, ParameterSpec, UntrainedError, InsufficientDataError
"""
from __future__ import annotations

from racingoptimizer.physics.baselines import (
    DEFAULT_BASELINES,
    CarBaselines,
    baselines_for,
    derive_baselines,
)
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
from racingoptimizer.physics.phase_weights import PHASE_WEIGHTS, SUB_UTILIZATIONS
from racingoptimizer.physics.recommend import recommend
from racingoptimizer.physics.recommendation import SetupRecommendation
from racingoptimizer.physics.score import score_setup
from racingoptimizer.physics.weights import weight_corners
from racingoptimizer.physics.wind import aero_wind_modifier, decompose_wind

__all__ = [
    "DEFAULT_BASELINES",
    "CarBaselines",
    "CornerPhaseStateWithConfidence",
    "FitRecord",
    "InsufficientDataError",
    "PHASE_WEIGHTS",
    "ParameterSpec",
    "PhysicsModel",
    "SUB_UTILIZATIONS",
    "SetupRecommendation",
    "UntrainedError",
    "aero_wind_modifier",
    "baselines_for",
    "decompose_wind",
    "derive_baselines",
    "fit",
    "fittable_parameters",
    "ontology_for",
    "recommend",
    "score_setup",
    "setup_value",
    "weight_corners",
]
