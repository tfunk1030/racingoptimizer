"""Constraint-clamped, gradient-free setup search (spec §10).

`recommend(model, track, env, constraints)` runs scipy.optimize.
differential_evolution over the bounded parameter space, post-clamps
defensively, recomputes the per-(corner, phase) score breakdown, and
returns a SetupRecommendation with per-parameter Confidence pulled from the
fitter records.

Determinism: the seed comes straight from `model.seed` and is forwarded to
differential_evolution.

Trust radius: parameters whose median fitter regime is `sparse` get a
narrowed [baseline ± 30% of constraint range] sub-bound; `dense`
parameters get the full constraint range.

The optimisation objective NEVER references the lap-duration channel —
that is the spec §6 / VISION §5 non-negotiable. The grep test in
tests/physics/test_score.py asserts this module contains no such reference.
"""
from __future__ import annotations

from statistics import median

import numpy as np
from scipy.optimize import differential_evolution

from racingoptimizer.confidence import Confidence
from racingoptimizer.constraints import ConstraintsTable, clamp
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.physics.model import PhysicsModel
from racingoptimizer.physics.ontology import fittable_parameters
from racingoptimizer.physics.recommendation import SetupRecommendation
from racingoptimizer.physics.score import (
    _aero_surface_or_none,
    _clamped_or_raise,
    _corner_phase_keys,
    _score_breakdown,
    score_breakdown,
)
from racingoptimizer.physics.weights import weight_corners


def recommend(
    model: PhysicsModel,
    track: str,
    env: EnvironmentFrame,
    constraints: ConstraintsTable,
) -> SetupRecommendation:
    weights = _cached_weights(model, track)

    fittable = [
        p for p in fittable_parameters(model.car, constraints)
        if constraints.bounds(model.car, p) is not None
    ]
    observed_std_table: dict[str, float] = getattr(
        model, "parameter_observed_std", {}
    ) or {}
    param_names: list[str] = []
    bounds: list[tuple[float, float]] = []
    init_values: list[float] = []
    pinned_constant: set[str] = set()
    for name in fittable:
        bound = constraints.bounds(model.car, name)
        if bound is None:
            continue
        baseline = float(model.baseline_setup.get(
            name, 0.5 * (bound[0] + bound[1]),
        ))
        baseline = min(max(baseline, bound[0]), bound[1])
        regime = _median_regime(model, name)
        observed_std = float(observed_std_table.get(name, 0.0))
        sub_bounds, was_pinned = _pin_or_trust_bounds(
            bound=bound,
            baseline=baseline,
            regime=regime,
            observed_std=observed_std,
        )
        if was_pinned:
            pinned_constant.add(name)
        param_names.append(name)
        bounds.append(sub_bounds)
        init_values.append(baseline)

    full_baseline = dict(model.baseline_setup)
    if not param_names:
        return _baseline_recommendation(
            model, track, env, full_baseline, weights,
        )

    # Budget per spec §13: < 5 s recommendation latency. With ~50 (corner,
    # phase) tuples per fit and a few hundred objective evaluations the
    # search stays within budget. maxiter / pop are tunable via env in the
    # plan; keep modest defaults here.
    pop_size = 10
    init_population = _seed_population(
        bounds=bounds,
        baseline=init_values,
        seed=model.seed,
        pop_size=pop_size,
    )

    aero = _aero_surface_or_none(model)
    keys = _corner_phase_keys(model)
    baselines = model.resolved_baselines

    def objective(x: np.ndarray) -> float:
        candidate = dict(full_baseline)
        for name, value in zip(param_names, x.tolist(), strict=True):
            candidate[name] = float(value)
        clamped = _clamped_or_raise(model, candidate, strict=False)
        breakdown_inner = _score_breakdown(
            model, clamped, env, aero, keys, weights, baselines,
        )
        return -float(sum(breakdown_inner.values()))

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=int(model.seed) & 0x7FFFFFFF,
        maxiter=15,
        popsize=pop_size,
        polish=False,
        init=init_population,
        tol=1e-4,
        mutation=(0.3, 1.0),
    )

    recommended = dict(full_baseline)
    parameters: dict[str, tuple[float, Confidence]] = {}
    for name, value in zip(param_names, result.x.tolist(), strict=True):
        clamped = clamp(float(value), name, model.car, constraints)
        if clamped.was_clamped and abs(clamped.value - value) > 1e-9:
            raise ValueError(
                f"recommend produced out-of-bounds value for {name!r}: "
                f"value={value!r} clamped_to={clamped.value!r}"
            )
        recommended[name] = float(clamped.value)
        parameters[name] = (float(clamped.value), _parameter_confidence(model, name))

    _fill_untrained_baselines(parameters, model, full_baseline)
    breakdown = score_breakdown(
        model, recommended, track, env, weights=weights,
    )
    return SetupRecommendation(
        car=model.car,
        track=track,
        env=env,
        parameters=parameters,
        score_breakdown=breakdown,
        untrained_parameters=tuple(model.untrained_parameters),
        aero_correction_available=bool(model.aero_correction_available),
        pinned_to_observed_median=tuple(sorted(pinned_constant)),
    )


# ---- internals -----------------------------------------------------------


# Module-level cache: PhysicsModel is frozen+slots so we cannot stash the
# cache on the instance. Keyed by `id(model)` (stable for the model's
# lifetime) plus track. Entries persist as long as the model is referenced;
# WeakValueDictionary would be ideal but PhysicsModel is not weak-referenceable.
_WEIGHTS_CACHE: dict[tuple[int, str], dict[int, float]] = {}


def _cached_weights(model: PhysicsModel, track: str) -> dict[int, float]:
    key = (id(model), track)
    if key in _WEIGHTS_CACHE:
        return _WEIGHTS_CACHE[key]
    try:
        weights = weight_corners(track, model)
    except Exception:  # pragma: no cover — corpus-shape dependent
        corners: set[int] = set()
        for key in model.fitters:
            if len(key) == 3:
                corners.add(int(key[0]))
            elif len(key) == 4:
                corners.add(int(key[1]))
        sorted_corners = sorted(corners)
        weights = (
            {c: 1.0 / len(sorted_corners) for c in sorted_corners}
            if sorted_corners else {}
        )
    _WEIGHTS_CACHE[key] = weights
    return weights


_REGIME_RANK: dict[str, int] = {"sparse": 0, "noisy": 1, "confident": 2, "dense": 3}
_RANK_TO_REGIME: dict[int, str] = {v: k for k, v in _REGIME_RANK.items()}

# Trust-radius shrink fractions per spec §10. Sparse parameters explore
# narrow [baseline ± 15% of bound range]; noisy use 25%; confident/dense
# use the full constraint range.
_TRUST_FRACTION: dict[str, float] = {"sparse": 0.30, "noisy": 0.50}


def _median_regime(model: PhysicsModel, parameter: str) -> str:
    """Median regime across every fitter that depends on `parameter`.

    Stage 3: each fitter is keyed by ``(corner_id, phase, channel)`` and
    consumes the full setup vector via `record.feature_names`. We treat a
    fitter as depending on `parameter` iff its `feature_names` mentions
    the parameter. Pre-Stage-3 keys carry the parameter in slot 0.
    """
    ranks: list[int] = []
    for key, record in model.fitters.items():
        if not _record_depends_on(key, record, parameter):
            continue
        conf = Confidence.derive(
            value=0.0,
            n_samples=int(record.n_samples),
            cv_residual_std=float(record.cv_residual_std),
            signal_std=float(max(record.signal_std, 1e-12)),
        )
        ranks.append(_REGIME_RANK[conf.regime])
    if not ranks:
        return "sparse"
    return _RANK_TO_REGIME[int(median(ranks))]


def _record_depends_on(key: tuple, record, parameter: str) -> bool:
    """Stage-3-aware: does this FitRecord's input vector include `parameter`?"""
    feature_names = getattr(record, "feature_names", ())
    if feature_names:
        return parameter in feature_names
    # Legacy v1/v2 keys: (param, corner, phase, channel).
    if len(key) == 4:
        return key[0] == parameter
    return False


def _trust_bounds(
    bound: tuple[float, float],
    baseline: float,
    regime: str,
) -> tuple[float, float]:
    lo, hi = bound
    fraction = _TRUST_FRACTION.get(regime)
    if fraction is None:
        return (lo, hi)
    span = (hi - lo) * fraction / 2.0
    return (max(lo, baseline - span), min(hi, baseline + span))


# Threshold for declaring a parameter "effectively constant" in training.
# When the observed per-session standard deviation is below this fraction of
# the constraint range, the joint surrogate has no information about how the
# response depends on the parameter — every training row sat at the same
# value (modulo trivial garage-click noise). Pinning the parameter to its
# observed median is the only honest answer; otherwise the DE search drifts
# to whichever bound the noise gradient points at, producing constraint-edge
# recommendations like "tyre pressure 166 kPa" when every observed session
# ran 152 kPa.
#
# 2% of the constraint range is the empirical floor: tyre pressures span
# 138-166 kPa (28 kPa range), so the threshold is ~0.6 kPa — narrower than
# any meaningful click. Ride heights span 25-75 mm (50 mm range), so the
# threshold is ~1 mm. Both are inside the iRacing garage-UI step quantum
# for those parameters, so any session with intentional variation across
# clicks will clear the threshold.
_NEAR_CONSTANT_FRACTION: float = 0.02

# Width of the pinned bound. scipy.optimize.differential_evolution requires
# `lo < hi` for every dimension, so we can't pass a true zero-width bound.
# 1e-6 of the constraint range is small enough that DE cannot explore around
# the pin meaningfully; the `_post_clamp` step in the CLI rounds to user-
# meaningful precision so the final reported value still equals the pin.
_PIN_HALF_WIDTH_FRACTION: float = 1e-6


def _pin_or_trust_bounds(
    *,
    bound: tuple[float, float],
    baseline: float,
    regime: str,
    observed_std: float,
) -> tuple[tuple[float, float], bool]:
    """Decide search bounds for a single parameter.

    Returns ``((lo, hi), was_pinned)``. ``was_pinned`` is ``True`` when the
    parameter was held effectively constant in training and the bound has
    been collapsed to a near-degenerate window around ``baseline``; the
    recommender adds those parameters to a "pinned" warning so the briefing
    can explain why no exploration happened. Otherwise the existing trust-
    radius logic applies (sparse → 30%, noisy → 50%, confident/dense → full).
    """
    lo, hi = bound
    span = hi - lo
    if span > 0.0 and observed_std / span < _NEAR_CONSTANT_FRACTION:
        eps = max(span * _PIN_HALF_WIDTH_FRACTION, 1e-9)
        pinned_lo = max(lo, baseline - eps)
        pinned_hi = min(hi, baseline + eps)
        # Guarantee lo < hi for DE even at the constraint edge.
        if pinned_hi <= pinned_lo:
            pinned_hi = min(hi, pinned_lo + 2 * eps)
        return ((pinned_lo, pinned_hi), True)
    return (_trust_bounds(bound, baseline, regime), False)


def _seed_population(
    *,
    bounds: list[tuple[float, float]],
    baseline: list[float],
    seed: int,
    pop_size: int,
) -> np.ndarray:
    """Seeded population with baseline as candidate 0 (deterministic).

    Total population is pop_size, capped at 50 — differential_evolution's
    per-generation cost scales with this.
    """
    n_params = len(bounds)
    n_pop = max(min(pop_size * max(n_params, 1), 30), 5)
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    pop = np.empty((n_pop, n_params), dtype=np.float64)
    pop[0] = np.asarray(baseline, dtype=np.float64)
    for i in range(1, n_pop):
        for j, (lo, hi) in enumerate(bounds):
            pop[i, j] = rng.uniform(lo, hi)
    return pop


def _parameter_confidence(model: PhysicsModel, parameter: str) -> Confidence:
    n_samples_vals: list[int] = []
    cv_vals: list[float] = []
    signal_vals: list[float] = []
    for key, record in model.fitters.items():
        if not _record_depends_on(key, record, parameter):
            continue
        n_samples_vals.append(int(record.n_samples))
        cv_vals.append(float(record.cv_residual_std))
        signal_vals.append(float(record.signal_std))
    if not n_samples_vals:
        baseline = float(model.baseline_setup.get(parameter, 0.0))
        return Confidence(
            value=baseline, lo=baseline, hi=baseline,
            n_samples=0, regime="sparse",
        )
    value = float(model.baseline_setup.get(parameter, 0.0))
    return Confidence.derive(
        value=value,
        n_samples=int(median(n_samples_vals)),
        cv_residual_std=float(median(cv_vals)),
        signal_std=float(median(signal_vals)),
    )


def _fill_untrained_baselines(
    parameters: dict[str, tuple[float, Confidence]],
    model: PhysicsModel,
    baseline: dict[str, float],
) -> None:
    """Add untrained-parameter baselines so slice F's renderer iterates one dict."""
    for name in sorted(model.untrained_parameters):
        if name in parameters:
            continue
        value = baseline.get(name)
        if value is None:
            continue
        parameters[name] = (float(value), _parameter_confidence(model, name))


def _baseline_recommendation(
    model: PhysicsModel,
    track: str,
    env: EnvironmentFrame,
    baseline: dict[str, float],
    weights: dict[int, float],
) -> SetupRecommendation:
    parameters: dict[str, tuple[float, Confidence]] = {
        name: (float(value), _parameter_confidence(model, name))
        for name, value in baseline.items()
    }
    breakdown = score_breakdown(model, baseline, track, env, weights=weights)
    return SetupRecommendation(
        car=model.car,
        track=track,
        env=env,
        parameters=parameters,
        score_breakdown=breakdown,
        untrained_parameters=tuple(model.untrained_parameters),
        aero_correction_available=bool(model.aero_correction_available),
    )


__all__ = ["recommend"]
