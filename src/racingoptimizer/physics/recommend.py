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

from dataclasses import replace
from statistics import median

import numpy as np
from scipy.optimize import differential_evolution

from racingoptimizer.confidence import Confidence
from racingoptimizer.constraints import ConstraintsTable, clamp
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner import CornerPhaseKey, Phase
from racingoptimizer.physics.axle_grip import (
    AxleGripCeiling,
    compute_axle_grip_ratios,
)
from racingoptimizer.physics.model import PhysicsModel
from racingoptimizer.physics.ontology import (
    fittable_parameters,
    snap_to_garage_step,
)
from racingoptimizer.physics.recommendation import SetupRecommendation
from racingoptimizer.physics.score import (
    _aero_surface_or_none,
    _clamped_or_raise,
    _corner_phase_keys,
    _score_breakdown,
    _score_breakdown_per_car,
    score_breakdown,
)
from racingoptimizer.physics.setup_symmetry import (
    DE_SYMMETRY_MIRRORS,
    DE_SYMMETRY_SLAVES,
    apply_setup_symmetry,
    static_rh_balance_penalty,
    static_rh_platform_penalty,
)
from racingoptimizer.physics.static_rh_knn import (
    _HARD_REJECT_OBJECTIVE,
    TB_TURN_PARAMS,
    cooptimize_tb_for_static_rh,
    enforce_static_rh_feasible,
    has_static_rh_physics,
    physics_static_rh_readouts,
    static_rh_de_infeasible_readouts,
)
from racingoptimizer.physics.weights import weight_corners

# Match ``cli.calibrate._COVERED_THRESHOLD``: fewer than three distinct
# values on the target track is not enough local evidence to trust
# cross-track surrogate extrapolation during race recommend.
_WITHIN_TRACK_COVERED_THRESHOLD: int = 3
# P0.3 sensitivity floor: a move is "defensible" only if shifting the
# parameter by one garage step changes the DE objective by at least
# this much. Otherwise the surrogate cannot resolve +1 click from -1
# click and the optimizer's chosen value is curve-fit to noise. The
# briefing rounds sensitivity to three decimals (so 0.000 is shown for
# anything below 0.0005); the floor sits an order of magnitude above
# that. See docs/accuracy-rebuild-2026-05-24/PLAN.md P0.3.
_SENSITIVITY_FLOOR: float = 0.005


def _safe_score_total(
    model: PhysicsModel,
    setup: dict[str, float],
    track: str,
    env: EnvironmentFrame,
    *,
    schedule: list | None = None,
    quali: bool = False,
) -> float:
    """Total score with narrow-exception swallow for the P0.3 probe."""
    try:
        return float(
            model.score_setup(setup, track, env, schedule=schedule, quali=quali),
        )
    except (KeyError, ValueError, ZeroDivisionError):
        return 0.0


def _apply_within_track_bounds(
    *,
    sub_bounds: tuple[float, float],
    was_pinned: bool,
    track_observed: tuple[float, ...],
    bound: tuple[float, float],
    reset_mode: bool,
) -> tuple[tuple[float, float], bool, bool]:
    """Cap DE bounds to within-track evidence when local coverage is thin.

    Returns ``(bounds, was_pinned, thin_track_pin)`` where ``thin_track_pin``
    is True when the parameter was pinned specifically because the target
    track has fewer than ``_WITHIN_TRACK_COVERED_THRESHOLD`` distinct values.

    With exactly two locally observed values, DE searches only between them
    (the surrogate picks within the bracket). Per VISION §6 the recommended
    value is chosen by per-corner-phase physics utilization alone — outcome
    metrics never enter setup selection.
    """
    if reset_mode or not track_observed:
        return sub_bounds, was_pinned, False
    unique = sorted({float(v) for v in track_observed})
    n_distinct = len(unique)
    if n_distinct >= _WITHIN_TRACK_COVERED_THRESHOLD:
        return sub_bounds, was_pinned, False
    lo, hi = bound
    if n_distinct == 1:
        v = unique[0]
        return ((v, v), True, True)
    # n_distinct == 2: search only between the two locally observed values.
    track_lo, track_hi = unique[0], unique[1]
    capped_lo = max(sub_bounds[0], track_lo, lo)
    capped_hi = min(sub_bounds[1], track_hi, hi)
    if capped_hi <= capped_lo:
        capped_lo, capped_hi = track_lo, track_hi
    return ((capped_lo, capped_hi), was_pinned, False)


def _kinematic_static_rh_ready(model: PhysicsModel) -> bool:
    """Whether the model has a deterministic kinematic static-RH fit.

    True when ``model.static_rh_kinematic`` carries at least one shipped
    channel fit. Used to skip the legacy k-NN
    ``enforce_static_rh_feasible`` corpus-blend repair (P0.2 / W2
    cleanup): on a kinematic readout, blending toward a corpus session
    *degrades* the prediction. Falls back to the legacy k-NN repair
    only when the kinematic fit refused to ship (R^2 < 0.98 or the
    corpus is too thin).
    """
    kinematic_fit = getattr(model, "static_rh_kinematic", None)
    if kinematic_fit is None:
        return False
    channels = getattr(kinematic_fit, "channels", None)
    return bool(channels)


def recommend(
    model: PhysicsModel,
    track: str,
    env: EnvironmentFrame,
    constraints: ConstraintsTable,
    *,
    schedule: list | None = None,
    quali: bool = False,
    explore_pct: float = 0.0,
    reset_mode: bool = False,
    surrogate_only: bool = False,
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

    ``surrogate_only=True`` disables the hybrid physics+surrogate
    objective and reverts to the pre-rebuild surrogate utilization
    score (plus the legacy additive axle guardrail penalty when
    ceilings are available). Default False uses the Day-13 hybrid
    score (physics evaluator + surrogate, phase-aware weighting).
    """
    use_hybrid = not surrogate_only
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
    # Global empirical range per parameter, aggregated across every track
    # the driver has run for this car. Doubles as:
    # 1. the pin denominator (see `_pin_or_trust_bounds`) — the constraint
    #    span is the wrong denominator when constraints reflect the legal
    #    UI envelope rather than what the driver explored;
    # 2. the trust envelope passed to DE — the corpus is global, so the
    #    search envelope is global. If Ferrari ran heave perch at -40 mm
    #    on Algarve and the user is asking for a Spa setup, the optimizer
    #    is allowed to recommend -40 at Spa even though Spa-specific data
    #    has never tried it. Per-track strictness was the previous rule
    #    (VISION §3 "no extrapolation") but it leaves good setups off the
    #    table at every well-driven track that hasn't seen full sweeps.
    global_observed: dict[str, set[float]] = {}
    per_track = (
        getattr(model, "per_track_parameter_observed", {}) or {}
        if is_per_car
        else {}
    )
    if is_per_car:
        for _track_params in per_track.values():
            for _name, _vals in (_track_params or {}).items():
                if _vals:
                    global_observed.setdefault(_name, set()).update(_vals)
    weights = _cached_weights(model, track, schedule=schedule)
    static_rh_corpus = getattr(model, "static_rh_corpus", ()) or ()
    use_static_rh_physics = has_static_rh_physics(model)
    tb_coopt_deferred: set[str] = set()

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
    pinned_within_track_thin: set[str] = set()
    track_observed_by_param = per_track.get(track, {}) or {}
    # Parameter -> warning when the model's observed median sat outside the
    # constraint bound. We populate this here (baseline-clamp time) and
    # again after DE (result-at-bound time); both are signals that the
    # constraint may be wrong relative to what iRacing actually allows.
    clamp_warnings: dict[str, str] = {}
    raw_baselines: dict[str, float] = {}
    for name in fittable:
        if name in DE_SYMMETRY_SLAVES:
            continue
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
        trust_baseline, trust_observed_std = _bayes_trust_anchor(
            model, track, name, baseline, observed_std,
        )
        # Trust envelope = global corpus envelope (every value the user has
        # ever run this car at, on any track). Per-track strictness was the
        # previous rule but left good setups off the table at every track
        # that hadn't seen full sweeps. The surrogate is trained globally
        # too, so this stays inside training density.
        global_vals = global_observed.get(name)
        target_observed_values = (
            tuple(sorted(global_vals)) if global_vals else ()
        )
        target_step = _click_step_for(model, name, bound)
        empirical_range = (
            (max(global_vals) - min(global_vals))
            if global_vals and len(global_vals) > 1
            else 0.0
        )
        sub_bounds, was_pinned = _pin_or_trust_bounds(
            bound=bound,
            baseline=trust_baseline,
            regime=regime,
            observed_std=trust_observed_std,
            target_observed=target_observed_values,
            click_step=target_step,
            empirical_range=empirical_range,
            explore_pct=explore_pct,
            reset_mode=reset_mode,
        )
        track_vals = track_observed_by_param.get(name, ())
        sub_bounds, was_pinned, thin_pin = _apply_within_track_bounds(
            sub_bounds=sub_bounds,
            was_pinned=was_pinned,
            track_observed=tuple(track_vals),
            bound=bound,
            reset_mode=reset_mode,
        )
        if thin_pin:
            pinned_within_track_thin.add(name)
            pinned_constant.add(name)
        elif was_pinned:
            pinned_constant.add(name)
        if use_static_rh_physics and name in TB_TURN_PARAMS:
            tb_base = float(trust_baseline)
            sub_bounds = (tb_base, tb_base)
            was_pinned = True
            tb_coopt_deferred.add(name)
            if name in pinned_constant:
                pinned_constant.discard(name)
        param_names.append(name)
        bounds.append(sub_bounds)
        init_values.append(trust_baseline)

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
    # `--reset` swaps the DE seed: instead of anchoring candidate 0 at
    # the past-setup baseline (and letting DE polish around it), seed
    # at the constraint MIDPOINT so the search starts deliberately
    # away from the user's current setup. Composes with the open-
    # envelope sub_bounds returned by _pin_or_trust_bounds(reset_mode).
    seed_baseline = (
        [(lo + hi) / 2.0 for lo, hi in bounds] if reset_mode else init_values
    )
    init_population = _seed_population(
        bounds=bounds,
        baseline=seed_baseline,
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

    # Guardrail-penalty wiring (post-physics-rebuild). When hybrid
    # scoring is active, guardrail penalties are applied inside
    # hybrid_score(); the additive penalty below is surrogate-only
    # legacy path.
    axle_ceilings = getattr(model, "axle_grip_ceilings", None)

    def objective(x: np.ndarray) -> float:
        candidate = dict(full_baseline)
        for name, value in zip(param_names, x.tolist(), strict=True):
            candidate[name] = float(value)
        clamped = apply_setup_symmetry(
            _clamped_or_raise(model, candidate, strict=False),
        )
        predicted_rh = model.predict_setup_readouts(clamped, env)
        if use_static_rh_physics:
            rh_channels = physics_static_rh_readouts(model, clamped, env)
            if static_rh_de_infeasible_readouts(rh_channels):
                return _HARD_REJECT_OBJECTIVE
        rh_penalty = static_rh_platform_penalty(predicted_rh)
        rh_penalty += static_rh_balance_penalty(predicted_rh)
        if is_per_car:
            breakdown_inner = _score_breakdown_per_car(
                model, clamped, env, aero, schedule, weights, baselines,
                phase_weights=phase_weights_override,
                hybrid=use_hybrid,
            )
        else:
            breakdown_inner = _score_breakdown(
                model, clamped, env, aero, keys, weights, baselines,
                phase_weights=phase_weights_override,
                hybrid=use_hybrid,
            )
        score_sum = float(sum(breakdown_inner.values()))
        if surrogate_only and axle_ceilings is not None:
            penalty = _axle_guardrail_penalty(
                model, clamped, env, weights, axle_ceilings,
                schedule if is_per_car else None,
                keys if not is_per_car else None,
            )
            score_sum -= penalty
        score_sum -= rh_penalty
        return -score_sum

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
    for name, value in zip(param_names, result.x.tolist(), strict=True):
        clamped = clamp(float(value), name, model.car, constraints)
        if clamped.was_clamped and abs(clamped.value - value) > 1e-9:
            raise ValueError(
                f"recommend produced out-of-bounds value for {name!r}: "
                f"value={value!r} clamped_to={clamped.value!r}"
            )
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

    recommended = apply_setup_symmetry(recommended)
    # P0.2 (W2 cleanup): once the per-car kinematic linear fit ships,
    # ``predict_setup_readouts`` returns geometrically-correct static
    # RH for the four ``setup_static_*_ride_height_mm`` channels. The
    # legacy k-NN bisection + corpus-blend repair was built to mask
    # the surrogate's near-zero pushrod gradient -- on a kinematic
    # readout, blending the recommended setup toward a corpus session
    # would *degrade* it. Bypass ``enforce_static_rh_feasible`` when
    # the kinematic fit is present; fall back to the legacy k-NN repair
    # when the kinematic fit refused to ship (R^2 < 0.98 or the
    # corpus is too thin).
    if static_rh_corpus and not _kinematic_static_rh_ready(model):
        recommended, still_bad = enforce_static_rh_feasible(
            recommended,
            static_rh_corpus,
            car=model.car,
            constraints=constraints,
            model=model,
            env=env,
        )
        if still_bad:
            readouts = physics_static_rh_readouts(model, recommended, env)
            lf = readouts.get("setup_static_lf_ride_height_mm")
            rf = readouts.get("setup_static_rf_ride_height_mm")
            clamp_warnings["_static_rh_platform"] = (
                "Static ride height still outside legal 30-80 mm after "
                f"platform repair (physics-predicted LF={lf}, RF={rf} mm). "
                "Verify perch/pushrod/heave in garage before applying."
            )
    if use_static_rh_physics:
        recommended, tb_ok = cooptimize_tb_for_static_rh(
            recommended,
            model,
            env,
            constraints=constraints,
        )
        if tb_coopt_deferred:
            clamp_warnings.setdefault(
                "_static_rh_tb_coopt",
                "Torsion bar turns trimmed post-search to match physics "
                "static ride height for the recommended perch/pushrod/heave "
                "platform (DE holds TB fixed until this pass).",
            )
        if not tb_ok and tb_coopt_deferred:
            readouts = physics_static_rh_readouts(model, recommended, env)
            lf = readouts.get("setup_static_lf_ride_height_mm")
            lr = readouts.get("setup_static_lr_ride_height_mm")
            clamp_warnings["_static_rh_tb_coopt"] = (
                "Could not trim torsion bar turns to physics static RH "
                f"targets (predicted LF={lf}, LR={lr} mm after TB pass). "
                "Verify TB turns manually against garage Ride Height."
            )

    # P0.3 -- sensitivity floor on emitted moves.
    # DE returns a value for every fittable parameter; on a noisy
    # surrogate (today's regime per `holdout_accuracy_latest.json`)
    # many of those moves cannot be defended against the past setup.
    # Probe each moved parameter at +/- 1 garage step under the same
    # scoring path the DE objective used. If neither side moves the
    # score by ``_SENSITIVITY_FLOOR``, revert to the training baseline
    # and surface the parameter under NOTES so the user sees why no
    # move was emitted.
    #
    # Probes use a frozen ``recommended_snapshot`` -- mutating
    # ``recommended`` mid-loop would make each parameter's probe see
    # earlier suppression decisions in the baseline setup, producing
    # order-dependent results. Decisions are collected here, then
    # applied to ``recommended`` after the loop.
    suppressed_below_sensitivity: set[str] = set()
    recommended_snapshot = dict(recommended)
    base_score = _safe_score_total(
        model, recommended_snapshot, track, env, schedule=schedule, quali=quali,
    )
    onto_view = model.ontology
    for name in list(param_names):
        if name in pinned_constant or name in tb_coopt_deferred:
            continue
        baseline_val = full_baseline.get(name)
        if baseline_val is None:
            continue
        rec_val = recommended_snapshot.get(name)
        if rec_val is None:
            continue
        spec_s = onto_view.get(name)
        if spec_s is None:
            continue
        step_s = float(spec_s.step) if spec_s.step else 0.0
        if step_s <= 0.0:
            continue
        if abs(float(rec_val) - float(baseline_val)) < step_s * 0.5:
            continue
        plus_setup = dict(recommended_snapshot)
        plus_setup[name] = float(
            clamp(float(rec_val) + step_s, name, model.car, constraints).value,
        )
        minus_setup = dict(recommended_snapshot)
        minus_setup[name] = float(
            clamp(float(rec_val) - step_s, name, model.car, constraints).value,
        )
        plus_setup = apply_setup_symmetry(plus_setup)
        minus_setup = apply_setup_symmetry(minus_setup)
        plus_score = _safe_score_total(
            model, plus_setup, track, env, schedule=schedule, quali=quali,
        )
        minus_score = _safe_score_total(
            model, minus_setup, track, env, schedule=schedule, quali=quali,
        )
        if (
            abs(plus_score - base_score) < _SENSITIVITY_FLOOR
            and abs(minus_score - base_score) < _SENSITIVITY_FLOOR
        ):
            suppressed_below_sensitivity.add(name)
    for name in suppressed_below_sensitivity:
        recommended[name] = float(full_baseline[name])

    parameters: dict[str, tuple[float, Confidence]] = {}
    for name in param_names:
        clamped = clamp(recommended[name], name, model.car, constraints)
        val = float(clamped.value)
        spec = model.ontology.get(name)
        if spec is not None:
            val = snap_to_garage_step(val, spec)
            recommended[name] = val
        confidence = _parameter_confidence(model, name)
        if reset_mode and confidence.regime in ("dense", "confident"):
            # Reset mode searches outside corpus density on purpose;
            # downgrade so the briefing doesn't claim a confident
            # prediction for a value the regressor extrapolated to.
            confidence = replace(confidence, regime="noisy")
        else:
            # Per-parameter local density downgrade (PLAN.md Day 2,
            # Mode 4): even outside reset mode, a recommended value
            # that's > 3 step units from the nearest observed sample
            # has effectively no training density supporting it; the
            # global confidence label overstates trust. Downgrade by
            # one tier (no-op when regime is already sparse).
            observed = _observed_values_for_param(model, track, name)
            spec = model.ontology.get(name)
            step = float(spec.step) if (spec is not None and spec.step) else 0.0
            if observed and step > 0:
                confidence = confidence.with_local_density(
                    recommended=float(clamped.value),
                    observed_values=observed,
                    step=step,
                )
        parameters[name] = (val, confidence)

    for slave, master in DE_SYMMETRY_MIRRORS.items():
        if master in parameters and slave not in parameters:
            parameters[slave] = parameters[master]
    _fill_untrained_baselines(parameters, model, full_baseline)
    breakdown = score_breakdown(
        model, recommended, track, env, weights=weights, schedule=schedule,
        quali=quali, hybrid=use_hybrid,
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
        pinned_within_track_thin=tuple(sorted(pinned_within_track_thin)),
        suppressed_below_sensitivity=tuple(sorted(suppressed_below_sensitivity)),
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
        # Per-car v4: weight each target-track corner by its duration
        # (`corner_duration_s` archetype key). A 5% improvement in a
        # 12-second corner is worth more lap-time than the same 5% in
        # a 2-second corner; weighting proportionally satisfies VISION
        # §6 "weighted by each corner's TIME SENSITIVITY". Falls back
        # to uniform weights when no archetype entry carries a positive
        # duration (e.g. a degenerate cold-start schedule).
        corner_durations: dict[int, float] = {}
        for entry in schedule:
            cid = int(entry.corner_id)
            if cid in corner_durations:
                continue
            dur = float(entry.archetype.get("corner_duration_s", 0.0) or 0.0)
            if dur > 0.0:
                corner_durations[cid] = dur
        total_dur = sum(corner_durations.values())
        if total_dur > 0.0:
            weights = {c: d / total_dur for c, d in corner_durations.items()}
        else:
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

# `--reset` widening factor: percentage of the constraint span added to
# each side of the corpus envelope. 30% lets every parameter move
# meaningfully outside the trodden corpus territory while staying clear
# of the constraint edges. Compare to typical `--explore 5..10` for
# modest exploration; reset is the "fundamentally different setup" knob.
_RESET_WIDEN_PCT: float = 30.0


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
    reset_mode: bool = False,
) -> tuple[tuple[float, float], bool]:
    """Decide search bounds for a single parameter.

    Returns ``((lo, hi), was_pinned)``. ``was_pinned`` is ``True`` when the
    parameter was held effectively constant in training and the bound has
    been collapsed to a near-degenerate window around ``baseline``; the
    recommender adds those parameters to a "pinned" warning so the briefing
    can explain why no exploration happened. Otherwise the existing trust-
    radius logic applies (sparse → 30%, noisy → 50%, confident/dense → full).

    ``target_observed`` (per-car v4): the unique values the driver has
    ever run for this car on ANY track (the global corpus envelope). When
    non-empty, the trust bound is additionally capped to STRICTLY
    ``[min(target_observed), max(target_observed)]`` clipped to the
    constraint bound — no extra margin. The recommender will not
    extrapolate outside the corpus envelope; recommending a never-tried
    value would be guessing against a confidence bracket the joint
    surrogate cannot honestly estimate. NOTE: this used to be per-target-
    track strict, but that left good setups off the table at every track
    that hadn't seen full sweeps. ``click_step`` is kept in the signature
    for callers but is used only to widen a degenerate single-value
    envelope into a window DE can search (``click_step`` either side of
    the lone value).

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
    # Defensive: the caller passes ``baseline`` as ``trust_baseline`` from
    # ``_bayes_trust_anchor``, which returns the empirical Bayes posterior
    # mean unclamped. When constraints.md is wrong (or the driver's setup
    # has drifted outside our coded legal envelope), the anchor can sit
    # well outside ``[lo, hi]`` — and the PIN branch below would then
    # collapse to an inverted window (e.g. baseline=10.0 against bound
    # (1.0, 5.0) → pinned_lo=9.999996, pinned_hi=5.0). Clamping here
    # guarantees every downstream branch sees a baseline inside the
    # constraint envelope; the bound-mismatch is reported as a clamp
    # warning at the call site.
    if span > 0.0:
        baseline = min(max(baseline, lo), hi)
    # `--reset` short-circuits: skip the corpus-density pin check (so
    # parameters the driver held constant can still move) and skip the
    # regime-driven trust-radius narrowing. The user has signalled the
    # current setup is broken and wants a materially different setup.
    # We DO still anchor on the observed envelope (clipped to the
    # constraint bound) and widen it by `_RESET_WIDEN_PCT` of the
    # constraint span on each side, clipped to legal bounds. This keeps
    # the search grounded in territory the surrogate has been trained
    # near rather than letting DE roam to the constraint edge for every
    # parameter and produce a setup the model has no business predicting
    # (e.g. heave 0->900 N/mm, sign-flipped pushrod offsets). Confidence
    # is still downgraded at the recommendation level.
    if reset_mode:
        observed = (
            [v for v in target_observed if lo <= v <= hi]
            if target_observed
            else []
        )
        if observed:
            base_lo, base_hi = min(observed), max(observed)
        elif empirical_range > 0.0:
            base_lo = max(lo, baseline - empirical_range / 2.0)
            base_hi = min(hi, baseline + empirical_range / 2.0)
        else:
            base_lo = base_hi = baseline
        widen = span * (_RESET_WIDEN_PCT / 100.0) if span > 0.0 else 0.0
        reset_lo = max(lo, base_lo - widen)
        reset_hi = min(hi, base_hi + widen)
        if reset_hi <= reset_lo:
            margin = max(float(click_step), span * 0.01, 1e-6)
            reset_lo = max(lo, reset_lo - margin)
            reset_hi = min(hi, reset_lo + max(margin, 2e-6))
            if reset_hi <= reset_lo:
                reset_lo, reset_hi = lo, hi
        return ((reset_lo, reset_hi), False)
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


def _bayes_trust_anchor(
    model: PhysicsModel,
    track: str,
    parameter: str,
    baseline: float,
    observed_std: float,
) -> tuple[float, float]:
    """Track-aware trust anchor from empirical-Bayes posteriors when available.

    ``bayes_posteriors`` stores per-(parameter, track) shrinkage estimates
    from ``physics.bayes_retrofit``. When present, the posterior mean
    replaces the global corpus median as the trust-radius center and
    ``mean_std`` replaces the global observed std for near-constant pin
    detection.
    """
    posteriors = getattr(model, "bayes_posteriors", None) or {}
    posterior = posteriors.get((parameter, track))
    if posterior is None or posterior.n_samples <= 0:
        return baseline, observed_std
    anchor = float(posterior.mean)
    mean_std = float(posterior.mean_std or posterior.std or 0.0)
    if mean_std <= 0.0:
        mean_std = observed_std
    return anchor, mean_std


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


def _observed_values_for_param(
    model: PhysicsModel, track: str, parameter: str,
) -> tuple[float, ...]:
    """Return the observed values for `parameter` relevant to `track`.

    For per-car (v4) models, prefers the per-track observed list (the
    surrogate is track-agnostic but the trust radius and density check
    are scoped to the target track). Falls back to the cross-track
    median observed std as a coarse density proxy if per-track data is
    absent (e.g. cross-car schedule borrow case).

    For per-(car, track) (v3) models, the model's training corpus is
    already track-scoped so we approximate with the baseline value plus
    +/- 1 std as a 3-point synthetic cluster (the std bookkeeping is
    per-parameter; we don't keep the raw values).

    Returns an empty tuple when no observed data is available -- the
    caller leaves the global regime label alone (Mode 5 territory:
    untrained car/track; sparseness is set upstream).
    """
    per_track = getattr(model, "per_track_parameter_observed", None)
    if per_track is not None:
        per_param = per_track.get(track, {})
        values = per_param.get(parameter)
        if values:
            return tuple(float(v) for v in values)
    # v3 fallback: synthesise a 3-point cluster from baseline +/- std.
    # This is coarse but correctly flags clearly out-of-cluster values
    # (e.g. recommended 200 vs baseline 152 with std 1.5: distance ~32x
    # std, downgrades regime).
    baseline = model.baseline_setup.get(parameter)
    std = (
        getattr(model, "parameter_observed_std", {}) or {}
    ).get(parameter, 0.0)
    if baseline is None:
        return ()
    if std <= 0:
        return (float(baseline),)
    return (
        float(baseline) - float(std),
        float(baseline),
        float(baseline) + float(std),
    )


def _parameter_confidence(model: PhysicsModel, parameter: str) -> Confidence:
    n_samples_vals: list[int] = []
    cv_vals: list[float] = []
    signal_vals: list[float] = []
    bootstrap_vals: list[float] = []
    for key, record in model.fitters.items():
        if not _record_depends_on(key, record, parameter):
            continue
        n_samples_vals.append(int(record.n_samples))
        cv_vals.append(float(record.cv_residual_std))
        signal_vals.append(float(record.signal_std))
        bootstrap_vals.append(float(getattr(record, "bootstrap_std", 0.0) or 0.0))
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
        bootstrap_std=float(max(bootstrap_vals)) if bootstrap_vals else 0.0,
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


# --------------------------------------------------------------------------
# Guardrail penalty (physics-rebuild Day-10 ceilings wired into DE).
#
# Additive penalty applied to the DE objective when a candidate setup's
# predicted axle utilization exceeds the empirical per-(car, axle) grip
# ceiling. The penalty is *additive in score units* (the DE objective
# is the corner-time-weighted utilization sum) so a penalty of
# `_GUARDRAIL_PENALTY_OVER_CEILING * corner_weight` subtracted from the
# objective is equivalent in magnitude to a 15% utilization loss in
# that corner -- enough to push DE away from physically-anomalous
# setups without dominating the surrogate's relative ranking.
#
# Long-G: `accel_lon_g_*` are now trained (Initiative 1 quick win).
# In --surrogate-only legacy guardrail path we still approximate long_g=0
# in mid-corner for safety (rear margin slightly inflated = safe to flag).
# --------------------------------------------------------------------------

# Additive penalty subtracted per (corner, phase) entry that exceeds
# the ceiling. Matches `hybrid_optimizer._GUARDRAIL_PENALTY_OVER_CEILING`.
_GUARDRAIL_PENALTY_OVER_CEILING: float = 0.15


def _axle_guardrail_penalty(
    model: PhysicsModel,
    setup: dict[str, float],
    env: EnvironmentFrame,
    weights: dict[int, float],
    ceilings: dict[str, AxleGripCeiling],
    schedule: list | None,
    keys: list[tuple[int, str]] | None,
) -> float:
    """Total corner-time-weighted penalty for setups exceeding the axle ceiling.

    Iterates the per-(corner, phase) schedule (per-car v4) or keys (v3)
    -- restricted to mid-corner phases where the lat-G peak is most
    interpretable as steady-state cornering. For each entry, queries
    the surrogate for predicted lat-G max, decomposes onto axles via
    the per-car geometry, and compares to the empirical ceiling.

    Returns the sum over (corner, phase) of:
        weights[corner_id] * _GUARDRAIL_PENALTY_OVER_CEILING
    for every entry whose front OR rear utilization exceeds 1.0.
    """
    front_ceil = ceilings.get("front")
    rear_ceil = ceilings.get("rear")
    if front_ceil is None or rear_ceil is None:
        return 0.0
    car = model.car
    total = 0.0
    if schedule is not None:
        iterator: list[tuple[int, str, object]] = [
            (int(e.corner_id), str(e.phase), e.archetype) for e in schedule
        ]
    elif keys is not None:
        iterator = [(int(c), str(p), None) for c, p in keys]
    else:
        return 0.0
    from racingoptimizer.physics.corner_schedule import (
        is_real_corner_archetype,
    )

    for corner_id, phase_str, archetype in iterator:
        # Restrict to mid-corner: that's where the long-G ~ 0
        # approximation holds and where setup-driven peak lat-G is
        # the operative quantity. Other phases (braking, throttle
        # exit) are driver-dominated and the empirical ceiling is
        # less directly comparable.
        if phase_str.strip().lower() != "mid_corner":
            continue
        # P0.4: skip phantom slots (start/finish straight, pit-out)
        # whose archetype is not corner-like. Required for v4 path
        # where the schedule carries archetype dicts; v3 keys path
        # has no archetype to filter on, so accept everything there.
        if archetype is not None and not is_real_corner_archetype(
            archetype if isinstance(archetype, dict) else None,
        ):
            continue
        try:
            phase = Phase(phase_str)
        except ValueError:
            continue
        cpkey = CornerPhaseKey(
            session_id="<recommend-virtual>",
            lap_index=0,
            corner_id=int(corner_id),
            phase=phase,
        )
        if archetype is not None:
            state = model.predict(setup, env, cpkey, corner_archetype=archetype)
        else:
            state = model.predict(setup, env, cpkey)
        lat_conf = state.states.get("accel_lat_g_max")
        if lat_conf is None:
            continue
        lat_g = float(lat_conf.value)
        if not np.isfinite(lat_g) or abs(lat_g) < 0.5:
            # Below mid-corner threshold from the ceiling fit; the
            # ratio computation would extrapolate outside the fitted
            # support.
            continue
        try:
            ratios = compute_axle_grip_ratios(
                np.array([lat_g]), np.array([0.0]), car,
            )
        except (KeyError, ValueError):
            continue
        front_ratio = float(ratios["front"][0])
        rear_ratio = float(ratios["rear"][0])
        front_margin = front_ratio / max(front_ceil.mu_peak, 1e-9)
        rear_margin = rear_ratio / max(rear_ceil.mu_peak, 1e-9)
        if front_margin > 1.0 or rear_margin > 1.0:
            w = float(weights.get(int(corner_id), 0.0))
            total += w * _GUARDRAIL_PENALTY_OVER_CEILING
    return total


# --------------------------------------------------------------------------
# Staged DE: 4 progressive stages + 1 polish pass over everything.
#
# Mirrors engineering setup workflow: aero first (wing + ride heights set
# the platform), then mechanical balance (springs + ARBs against fixed
# aero), then dampers (dynamic balance with platform fixed), then detail
# (cambers, toes, brake bias, diff). Each stage runs DE over a small
# parameter subset with everything else pinned to either a previous
# stage's chosen value or the model's training baseline. A final polish
# pass re-opens the full vector with a small explore-style widening,
# seeded from the accumulated stage results, to recover any global
# optimum the staging cuts may have missed.
#
# Trade-off vs single-pass DE: each stage's smaller parameter set
# converges faster (~3 min instead of 15 for BMW 47-param), and the
# resulting setup tends to follow the engineer-intuitive ordering
# (wing -> springs -> dampers -> detail). Cost: 5 sequential DE calls
# instead of 1, total wall time roughly equivalent (~12-18 min).
# --------------------------------------------------------------------------

# Family -> stage mapping. ParameterSpec.family for every fittable user-
# settable parameter falls in exactly one stage. Readout-only families
# (heave_spring, heave_slider, ride_height, corner_weight) have
# `user_settable=False` so they're already excluded from
# `fittable_parameters()`. `fuel` lives in stage 1 because fuel mass
# affects ride height, but in race mode it's auto-pinned by the CLI
# before recommend_staged is even called.
_STAGE_FAMILIES: dict[str, frozenset[str]] = {
    "aero": frozenset(
        {"rear_wing", "tyre_pressure", "perch_offset", "pushrod", "fuel"}
    ),
    "mechanical": frozenset({"spring_rate", "arb", "torsion_bar"}),
    "dampers": frozenset({"damper"}),
    "detail": frozenset({"camber", "brake_bias", "diff"}),
}
_STAGE_ORDER: tuple[str, ...] = ("aero", "mechanical", "dampers", "detail")


def _partition_parameters_by_stage(
    fittable: list[str], ontology: dict,
) -> dict[str, list[str]]:
    """Return ``{stage_name: [param_names]}`` covering every fittable param.

    Every fittable parameter is assigned to exactly one stage based on
    its `ParameterSpec.family`. Parameters whose family doesn't appear
    in any stage map fall into the implicit "leftover" stage `detail`
    (last) so nothing falls through.
    """
    by_stage: dict[str, list[str]] = {s: [] for s in _STAGE_ORDER}
    family_to_stage: dict[str, str] = {}
    for stage, families in _STAGE_FAMILIES.items():
        for fam in families:
            family_to_stage[fam] = stage
    for name in sorted(fittable):
        spec = ontology.get(name)
        if spec is None:
            by_stage["detail"].append(name)
            continue
        stage = family_to_stage.get(spec.family, "detail")
        by_stage[stage].append(name)
    return by_stage


def recommend_staged(
    model: PhysicsModel,
    track: str,
    env: EnvironmentFrame,
    constraints: ConstraintsTable,
    *,
    schedule: list | None = None,
    quali: bool = False,
    explore_pct: float = 0.0,
    reset_mode: bool = False,
    surrogate_only: bool = False,
) -> SetupRecommendation:
    """Run DE in 4 progressive stages + 1 polish pass over everything.

    Returns the same SetupRecommendation shape as `recommend()` -- the
    final polish pass produces it. Per-stage intermediate results are
    not surfaced today (renderer treatment is a follow-up).
    """
    fittable = [
        p for p in fittable_parameters(model.car, constraints)
        if constraints.bounds(model.car, p) is not None
    ]
    by_stage = _partition_parameters_by_stage(fittable, model.ontology)

    accumulated: dict[str, float] = {}
    final_rec: SetupRecommendation | None = None

    for stage_name in _STAGE_ORDER:
        stage_params = by_stage.get(stage_name, [])
        if not stage_params:
            continue
        # Build a constraint table where every fittable param OUTSIDE
        # this stage is pinned -- to a previous stage's chosen value if
        # it's already been optimized, otherwise to the model's training
        # baseline.
        stage_constraints = constraints
        for p in fittable:
            if p in stage_params:
                continue
            if p in accumulated:
                pin_value = accumulated[p]
            else:
                bound = constraints.bounds(model.car, p)
                baseline = float(model.baseline_setup.get(
                    p, 0.5 * (bound[0] + bound[1]) if bound else 0.0,
                ))
                if bound is not None:
                    baseline = min(max(baseline, bound[0]), bound[1])
                pin_value = baseline
            stage_constraints = stage_constraints.with_pin(
                model.car, p, pin_value,
            )

        rec = recommend(
            model, track, env, stage_constraints,
            schedule=schedule, quali=quali, explore_pct=explore_pct,
            reset_mode=reset_mode, surrogate_only=surrogate_only,
        )

        # Carry forward this stage's chosen values for the next stage's
        # pin builder.
        for p in stage_params:
            if p in rec.parameters:
                accumulated[p] = float(rec.parameters[p][0])
        final_rec = rec

    # Stage 5 -- polish: small DE over the full vector with explore widened
    # to at least 5% on each side, seeded from the accumulated values.
    # Catches cases where the stage cuts pushed us into a local optimum
    # that's not globally best.
    if final_rec is None:
        # No stages ran (no fittable parameters). Fall back to single-pass.
        return recommend(
            model, track, env, constraints,
            schedule=schedule, quali=quali, explore_pct=explore_pct,
            reset_mode=reset_mode, surrogate_only=surrogate_only,
        )
    # Polish stage uses the user-supplied explore_pct directly.
    # Previously this was forced to `max(explore_pct, 5.0)`, silently
    # widening past the user's request -- a user passing
    # ``--staged --explore 0`` (intending strict empirical) got 5%
    # widening anyway. Honoring the user value lets strict-empirical
    # stay strict; users who want a polish widening can pass
    # ``--explore 5`` (or higher) explicitly.
    # Inject the accumulated values into model.baseline_setup so DE seeds
    # from there. PhysicsModel is frozen+slots; dataclasses.replace gives
    # a fresh model with overridden baseline_setup. (No __setstate__
    # involvement -- this stays in-process.)
    polish_baseline = dict(model.baseline_setup)
    polish_baseline.update(accumulated)
    polish_model = replace(model, baseline_setup=polish_baseline)
    polish_rec = recommend(
        polish_model, track, env, constraints,
        schedule=schedule, quali=quali, explore_pct=explore_pct,
        reset_mode=reset_mode, surrogate_only=surrogate_only,
    )
    return polish_rec


__all__ = ["recommend", "recommend_staged"]
