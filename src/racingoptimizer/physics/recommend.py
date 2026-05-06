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
    _score_breakdown_per_car,
    score_breakdown,
)
from racingoptimizer.physics.weights import weight_corners


def recommend(
    model: PhysicsModel,
    track: str,
    env: EnvironmentFrame,
    constraints: ConstraintsTable,
    *,
    schedule: list | None = None,
    quali: bool = False,
    explore_pct: float = 0.0,
) -> SetupRecommendation:
    """Constraint-clamped DE search for the optimal setup.

    ``schedule`` (per-car v4 only): when ``model.feature_schema_version >= 4``
    the caller MUST pass the target track's
    ``list[CornerScheduleEntry]`` (built via
    ``physics.corner_schedule.build_corner_schedule``). The per-corner
    archetype features ride alongside each entry and are fed into
    ``PhysicsModel._predict_v4`` so the same per-car fitter can score the
    target corners.

    ``quali=True`` swaps to the quali-stint phase-weight overlay
    inside the DE objective so the search is biased toward outright
    single-lap pace (more aero_eff, more grip utilisation, less
    platform conservatism). Caller pins ``fuel_level_l`` to a low
    value (typically via the CLI's ``--fuel`` flag) to match.

    ``explore_pct`` widens the per-track empirical envelope by N% of
    each parameter's constraint span on each side (clipped to legal
    bounds). Lets DE probe values outside the user's observed
    setups; recommendations that land in the widened territory are
    flagged as ``sparse`` confidence so the user knows the prediction
    is extrapolating beyond corpus density. Default 0.0 = strict
    empirical envelope (current behavior).
    """
    is_per_car = int(model.feature_schema_version) >= 4
    if is_per_car and schedule is None:
        raise ValueError(
            "per-car PhysicsModel requires `schedule` — build it from the "
            "target TrackModel via "
            "`racingoptimizer.physics.corner_schedule.build_corner_schedule`."
        )
    # Per-car: cap the trust radius to the values the driver has actually
    # run on the TARGET track. Without this cap the joint surrogate's
    # response surface lets the optimizer drift outside the empirical
    # envelope (e.g. recommending heave_spring=25 N/mm when the driver
    # has only ever tried 30 or 50). v3 pickles return None and the cap
    # is a no-op.
    target_observed: dict[str, tuple[float, ...]] = {}
    # Global empirical range per parameter — aggregated across every track
    # the driver has run for this car, used as the denominator in the pin
    # check (see `_pin_or_trust_bounds`). The constraint span is the wrong
    # denominator when constraints reflect the legal UI envelope rather
    # than what the driver explored — wide envelopes would mask real
    # corpus variation as "near constant".
    global_observed: dict[str, set[float]] = {}
    if is_per_car:
        per_track = getattr(model, "per_track_parameter_observed", {}) or {}
        target_observed = dict(per_track.get(track, {}) or {})
        for _track_params in per_track.values():
            for _name, _vals in (_track_params or {}).items():
                if _vals:
                    global_observed.setdefault(_name, set()).update(_vals)
    weights = _cached_weights(model, track, schedule=schedule)

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
    # Parameter -> warning when the model's observed median sat outside the
    # constraint bound. We populate this here (baseline-clamp time) and
    # again after DE (result-at-bound time); both are signals that the
    # constraint may be wrong relative to what iRacing actually allows.
    clamp_warnings: dict[str, str] = {}
    raw_baselines: dict[str, float] = {}
    for name in fittable:
        bound = constraints.bounds(model.car, name)
        if bound is None:
            continue
        raw_baseline = float(model.baseline_setup.get(
            name, 0.5 * (bound[0] + bound[1]),
        ))
        raw_baselines[name] = raw_baseline
        baseline = min(max(raw_baseline, bound[0]), bound[1])
        if abs(raw_baseline - baseline) > 1e-6:
            # Observed median in the training corpus sat OUTSIDE the
            # constraint bound. The pin-to-median logic later will pin to
            # the clamped value, not the true median, so the recommendation
            # will sit at the bound rather than at observed reality. The
            # constraint is almost certainly wrong — the user should check
            # constraints.md against the iRacing garage UI for this param.
            clamp_warnings[name] = (
                f"observed median {raw_baseline:.3f} sat outside legal "
                f"[{bound[0]:.3f}, {bound[1]:.3f}] in constraints.md — "
                f"verify the constraint matches the iRacing garage UI; "
                f"recommendation pinned to bound {baseline:.3f}"
            )
        regime = _median_regime(model, name)
        observed_std = float(observed_std_table.get(name, 0.0))
        target_observed_values = tuple(target_observed.get(name, ()))
        target_step = _click_step_for(model, name, bound)
        global_vals = global_observed.get(name)
        empirical_range = (
            (max(global_vals) - min(global_vals))
            if global_vals and len(global_vals) > 1
            else 0.0
        )
        sub_bounds, was_pinned = _pin_or_trust_bounds(
            bound=bound,
            baseline=baseline,
            regime=regime,
            observed_std=observed_std,
            target_observed=target_observed_values,
            click_step=target_step,
            empirical_range=empirical_range,
            explore_pct=explore_pct,
        )
        if was_pinned:
            pinned_constant.add(name)
        param_names.append(name)
        bounds.append(sub_bounds)
        init_values.append(baseline)

    full_baseline = dict(model.baseline_setup)
    if not param_names:
        return _baseline_recommendation(
            model, track, env, full_baseline, weights, schedule=schedule,
        )

    # Budget per spec §13: < 5 s recommendation latency. With ~50 (corner,
    # phase) tuples per fit and a few hundred objective evaluations the
    # search stays within budget. maxiter / pop are tunable via env in the
    # plan; keep modest defaults here.
    # DE budget bumped from (5, 10) → (15, 20) on 2026-05-06: lets
    # the optimiser polish recommendations inside the trust radius
    # without changing physics. Cost: ~3x recommend latency (5 min
    # → ~15 min on the 47-param BMW per-car search). Empirical lift
    # is small (≤3% of total objective) but consistent.
    pop_size = 20
    init_population = _seed_population(
        bounds=bounds,
        baseline=init_values,
        seed=model.seed,
        pop_size=pop_size,
    )

    aero = _aero_surface_or_none(model)
    # Wet-mode + quali-mode overlay: pick the right baselines + phase
    # weights once before DE so the objective is consistent across
    # every evaluation (the DE search is over the setup vector, not
    # over the env or the stint mode). Mirrors the helper used by
    # `score_setup` so race / wet / quali / wet-quali all flow through
    # one decision point.
    from racingoptimizer.physics.score import _conditions_adjusted_baselines
    baselines, phase_weights_override = _conditions_adjusted_baselines(
        model, env, quali=quali,
    )
    if is_per_car:
        keys = None  # type: ignore[assignment]
    else:
        keys = _corner_phase_keys(model)

    def objective(x: np.ndarray) -> float:
        candidate = dict(full_baseline)
        for name, value in zip(param_names, x.tolist(), strict=True):
            candidate[name] = float(value)
        clamped = _clamped_or_raise(model, candidate, strict=False)
        if is_per_car:
            breakdown_inner = _score_breakdown_per_car(
                model, clamped, env, aero, schedule, weights, baselines,
                phase_weights=phase_weights_override,
            )
        else:
            breakdown_inner = _score_breakdown(
                model, clamped, env, aero, keys, weights, baselines,
                phase_weights=phase_weights_override,
            )
        return -float(sum(breakdown_inner.values()))

    # DE budget: per-car v4 with 47 fittable parameters × 70+ schedule
    # entries × 30+ output channels per objective evaluation works out
    # to ~150 ms per call. maxiter=15 + popsize=20 (bumped from 5+10
    # on 2026-05-06) lets DE polish inside the trust-radius envelope
    # at ~3x the prior latency (~15 min total). tol=5e-3 short-
    # circuits earlier once the population plateaus.
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=int(model.seed) & 0x7FFFFFFF,
        maxiter=15,
        popsize=pop_size,
        polish=False,
        init=init_population,
        tol=5e-3,
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
        # Detect bound-binding: if DE returned a value within ~1% of either
        # bound AND the training baseline was outside the bound, the
        # constraint is almost certainly suppressing exploration. Already
        # warned at baseline-clamp time, but re-check here for the case
        # where the baseline was inside the bound but DE still drifted to
        # an edge — second-order signal that the constraint is too tight.
        if name not in clamp_warnings and name in raw_baselines:
            bound = constraints.bounds(model.car, name)
            if bound is not None:
                lo, hi = bound
                span = hi - lo
                if span > 0:
                    edge_eps = max(span * 0.01, 1e-6)
                    raw_b = raw_baselines[name]
                    at_lo = (
                        abs(float(clamped.value) - lo) <= edge_eps
                        and raw_b < lo - edge_eps
                    )
                    at_hi = (
                        abs(float(clamped.value) - hi) <= edge_eps
                        and raw_b > hi + edge_eps
                    )
                    if at_lo or at_hi:
                        clamp_warnings[name] = (
                            f"recommendation pinned at constraint bound "
                            f"{float(clamped.value):.3f} but observed "
                            f"median {raw_b:.3f} sat the other side of "
                            f"the bound — verify constraints.md against "
                            f"the iRacing garage UI"
                        )
        recommended[name] = float(clamped.value)
        parameters[name] = (float(clamped.value), _parameter_confidence(model, name))

    _fill_untrained_baselines(parameters, model, full_baseline)
    breakdown = score_breakdown(
        model, recommended, track, env, weights=weights, schedule=schedule,
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
        clamp_warnings=dict(clamp_warnings),
    )


# ---- internals -----------------------------------------------------------


# Module-level cache: PhysicsModel is frozen+slots so we cannot stash the
# cache on the instance. Keyed by `id(model)` (stable for the model's
# lifetime) plus track. Entries persist as long as the model is referenced;
# WeakValueDictionary would be ideal but PhysicsModel is not weak-referenceable.
_WEIGHTS_CACHE: dict[tuple[int, str], dict[int, float]] = {}


def _cached_weights(
    model: PhysicsModel,
    track: str,
    *,
    schedule: list | None = None,
) -> dict[int, float]:
    key = (id(model), track)
    if key in _WEIGHTS_CACHE:
        return _WEIGHTS_CACHE[key]
    if int(model.feature_schema_version) >= 4 and schedule:
        # Per-car: weight every target-track corner uniformly until
        # `weight_corners` learns to consume a corner schedule. Future work
        # can derive per-corner time sensitivity from the schedule's
        # archetype values (longer corners + slower apex = higher weight).
        corners = sorted({int(entry.corner_id) for entry in schedule})
        weights = (
            {c: 1.0 / len(corners) for c in corners}
            if corners else {}
        )
    else:
        try:
            weights = weight_corners(track, model)
        except Exception:  # pragma: no cover — corpus-shape dependent
            corners_set: set[int] = set()
            for fkey in model.fitters:
                if len(fkey) == 3:
                    corners_set.add(int(fkey[0]))
                elif len(fkey) == 4:
                    corners_set.add(int(fkey[1]))
            sorted_corners = sorted(corners_set)
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
    target_observed: tuple[float, ...] = (),
    click_step: float = 0.0,
    empirical_range: float = 0.0,
    explore_pct: float = 0.0,
) -> tuple[tuple[float, float], bool]:
    """Decide search bounds for a single parameter.

    Returns ``((lo, hi), was_pinned)``. ``was_pinned`` is ``True`` when the
    parameter was held effectively constant in training and the bound has
    been collapsed to a near-degenerate window around ``baseline``; the
    recommender adds those parameters to a "pinned" warning so the briefing
    can explain why no exploration happened. Otherwise the existing trust-
    radius logic applies (sparse → 30%, noisy → 50%, confident/dense → full).

    ``target_observed`` (per-car v4): the unique values the driver has
    actually run on the TARGET track. When non-empty, the trust bound is
    additionally capped to STRICTLY ``[min(target_observed),
    max(target_observed)]`` clipped to the constraint bound — no extra
    margin. The per-car recommender will not extrapolate outside the
    empirical envelope; recommending a never-tried value would be guessing
    against a confidence bracket the joint surrogate cannot honestly
    estimate. ``click_step`` is kept in the signature for callers but is
    used only to widen a degenerate single-value envelope into a window
    DE can search (``click_step`` either side of the lone value).

    ``empirical_range`` (per-car v4): the global ``max - min`` observed
    across every pooled session for this parameter. Used as the pin
    denominator instead of the constraint span — wide legal envelopes
    (e.g. BMW heave 0..900 N/mm per BMWBounds.md) would otherwise mask
    real corpus variation as "near constant". When 0 (no empirical data,
    or truly constant), the original constraint-span logic applies so
    fully-uniform parameters still pin correctly.
    """
    lo, hi = bound
    span = hi - lo
    pin_denom = empirical_range if empirical_range > 0.0 else span
    if pin_denom > 0.0 and observed_std / pin_denom < _NEAR_CONSTANT_FRACTION:
        eps = max(span * _PIN_HALF_WIDTH_FRACTION, 1e-9)
        pinned_lo = max(lo, baseline - eps)
        pinned_hi = min(hi, baseline + eps)
        # Guarantee lo < hi for DE even at the constraint edge.
        if pinned_hi <= pinned_lo:
            pinned_hi = min(hi, pinned_lo + 2 * eps)
        return ((pinned_lo, pinned_hi), True)
    trust_lo, trust_hi = _trust_bounds(bound, baseline, regime)
    if target_observed:
        # Clip the observed values to the constraint envelope FIRST.
        # Without this, a constraint pin (e.g. ``--fuel 8`` collapses
        # the constraint to ``(8, 8)``) combined with a target-track
        # observed value far outside the pin (corpus only has 58 L on
        # Spa) would build empirical_lo=58, empirical_hi=8 — lo > hi —
        # and the downstream DE seed rejects with "high - low < 0".
        clipped = [v for v in target_observed if lo <= v <= hi]
        if clipped:
            empirical_lo = min(clipped)
            empirical_hi = max(clipped)
        else:
            # No observed value lies inside the (possibly user-pinned)
            # constraint window. Trust the constraint bound directly —
            # the user explicitly narrowed the search and we have no
            # in-window evidence to weight against it.
            empirical_lo = lo
            empirical_hi = hi
        # Optional `--explore N`: widen the empirical envelope by
        # explore_pct% of the constraint span on each side, clipped
        # to legal bounds. Lets DE probe values outside the user's
        # observed setups; the recommendation may land in the widened
        # territory, where confidence is weaker by design.
        if explore_pct > 0.0 and span > 0.0:
            margin = span * (explore_pct / 100.0)
            empirical_lo = max(lo, empirical_lo - margin)
            empirical_hi = min(hi, empirical_hi + margin)
        # Single observed value → expand by one click each side so DE has
        # SOMETHING to search; otherwise the bound is degenerate and DE
        # raises. Multi-value observations stay strict.
        if empirical_hi <= empirical_lo:
            margin = max(float(click_step), 1e-6)
            empirical_lo = max(lo, empirical_lo - margin)
            empirical_hi = min(hi, empirical_hi + margin)
        # Cap trust to the empirical envelope strictly. We INTERSECT (not
        # replace) so a tighter trust bound from a sparse-regime
        # parameter is kept.
        capped_lo = max(trust_lo, empirical_lo)
        capped_hi = min(trust_hi, empirical_hi)
        if capped_hi > capped_lo:
            trust_lo, trust_hi = capped_lo, capped_hi
        else:
            # Trust bound and empirical envelope don't overlap (e.g. the
            # per-session-median baseline drifted far from anything the
            # driver ran on the target track). Trust the empirical
            # envelope — it's the only ground truth we have.
            trust_lo, trust_hi = empirical_lo, empirical_hi
    # Final defensive clamp: DE requires lo <= hi. If anything above
    # produced an inverted range (constraint pin + out-of-pin empirical
    # data + trust radius interaction), fall back to the constraint
    # bound itself.
    if trust_hi < trust_lo:
        trust_lo, trust_hi = lo, hi
    return ((trust_lo, trust_hi), False)


def _click_step_for(
    model: PhysicsModel,
    parameter: str,
    bound: tuple[float, float],
) -> float:
    """One iRacing-garage click for `parameter`, for empirical-envelope margin.

    Reads ``ParameterSpec.step`` from the per-car ontology when available
    (every fittable parameter ships with a `step` per the setup-card
    rendering work). Falls back to 1% of the constraint span which is
    intentionally tiny so the cap stays close to the observed values.
    """
    spec = model.ontology.get(parameter) if model.ontology else None
    step = float(getattr(spec, "step", 0.0) or 0.0) if spec else 0.0
    if step > 0.0:
        return step
    lo, hi = bound
    span = hi - lo
    return max(span * 0.01, 1e-6) if span > 0 else 1e-6


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
    *,
    schedule: list | None = None,
) -> SetupRecommendation:
    parameters: dict[str, tuple[float, Confidence]] = {
        name: (float(value), _parameter_confidence(model, name))
        for name, value in baseline.items()
    }
    breakdown = score_breakdown(
        model, baseline, track, env, weights=weights, schedule=schedule,
    )
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
