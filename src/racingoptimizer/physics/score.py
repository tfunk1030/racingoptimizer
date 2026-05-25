"""Sub-utilizations + per-phase aggregator + score_setup (spec §6).

Six sub-utilizations clamp to [0, 1]; aggregate is a phase-weighted blend
that also returns an aggregate Confidence. score_setup sums per-(corner,
phase) utilization weighted by per-corner time sensitivity.

The optimisation objective NEVER references the lap-duration channel —
that is the spec §6 / VISION §5 non-negotiable. The grep test in
tests/physics/test_score.py asserts this module contains no such reference.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from racingoptimizer.aero import BASELINE_AIR_DENSITY, AeroSurface
from racingoptimizer.confidence import Confidence
from racingoptimizer.confidence.confidence import Regime
from racingoptimizer.constraints import clamp
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner import CornerPhaseKey, Phase
from racingoptimizer.physics.baselines import CarBaselines
from racingoptimizer.physics.model import CornerPhaseStateWithConfidence, PhysicsModel
from racingoptimizer.physics.phase_weights import PHASE_WEIGHTS, SUB_UTILIZATIONS

_SubFn = Callable[
    [
        CornerPhaseStateWithConfidence,
        EnvironmentFrame,
        AeroSurface | None,
        CarBaselines,
    ],
    tuple[float, Confidence],
]


# Regime ordering: worst -> best. min(regime, ...) returns the worst.
_REGIME_RANK: dict[str, int] = {"sparse": 0, "noisy": 1, "confident": 2, "dense": 3}
_RANK_TO_REGIME: dict[int, Regime] = {
    0: "sparse", 1: "noisy", 2: "confident", 3: "dense",
}
_OBJECTIVE_CONFIDENCE_MULTIPLIER: dict[str, float] = {
    "sparse": 0.60,
    "noisy": 0.80,
    "confident": 0.95,
    "dense": 1.00,
}

# Hard floor on predicted mean ride height. iRacing GTPs scrape splitter /
# floor at very low ride heights; once the floor is touching, aero stalls
# and traction collapses. VISION §4 lists "bottoming risk" as part of the
# platform sub-utilization. Below the safety floor, the platform util gets
# driven to zero linearly across `_BOTTOMING_PENALTY_DEPTH_MM` so the DE
# objective sees a smooth gradient pushing AWAY from sub-floor predictions
# rather than a discontinuous cliff.
_RIDE_HEIGHT_SAFETY_FLOOR_MM: float = 5.0
_BOTTOMING_PENALTY_DEPTH_MM: float = 10.0

# Neutral fallbacks when aero maps or archetype speed are unavailable
# during hybrid scoring inside the DE hot loop.
_DEFAULT_AERO_BALANCE_PCT: float = 50.0
_DEFAULT_AERO_LD: float = 3.5
_DEFAULT_SPEED_MS: float = 50.0


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _channel_confidence(
    state: CornerPhaseStateWithConfidence,
    channel: str,
) -> tuple[float | None, Confidence | None]:
    conf = state.states.get(channel)
    if conf is None:
        return None, None
    return float(conf.value), conf


def _sparse_conf(value: float) -> Confidence:
    return Confidence(
        value=value, lo=value, hi=value, n_samples=0, regime="sparse",
    )


def grip(
    state: CornerPhaseStateWithConfidence,
    env: EnvironmentFrame,
    aero: AeroSurface | None,
    baselines: CarBaselines,
) -> tuple[float, Confidence]:
    lat_g, conf = _channel_confidence(state, "accel_lat_g_max")
    if lat_g is None:
        return 0.5, _sparse_conf(0.5)

    baseline = float(baselines.aero_grip_baseline_g)
    if aero is None:
        max_g = baseline
    else:
        ld = _aero_ld_for_state(state, env, aero)
        density_factor = float(env.air_density) / BASELINE_AIR_DENSITY
        max_g = 0.5 * ld * density_factor + baseline

    util = _clip01(lat_g / max(max_g, 1e-6))
    assert conf is not None
    return util, _confidence_with_value(conf, util)


def balance(
    state: CornerPhaseStateWithConfidence,
    env: EnvironmentFrame,
    aero: AeroSurface | None,
    baselines: CarBaselines,
) -> tuple[float, Confidence]:
    us, conf = _channel_confidence(state, "understeer_angle_mean_rad")
    if us is None:
        return 0.5, _sparse_conf(0.5)
    util = 1.0 - _clip01(abs(us) / baselines.understeer_scale_rad)
    assert conf is not None
    return util, _confidence_with_value(conf, util)


def stability(
    state: CornerPhaseStateWithConfidence,
    env: EnvironmentFrame,
    aero: AeroSurface | None,
    baselines: CarBaselines,
) -> tuple[float, Confidence]:
    yaw, yaw_conf = _channel_confidence(state, "yaw_rate_max_rad_s")
    if yaw is not None:
        util = 1.0 - _clip01(yaw / baselines.yaw_rate_scale_rad_s)
        assert yaw_conf is not None
        return util, _confidence_with_value(yaw_conf, util)

    lat_max, lat_max_conf = _channel_confidence(state, "accel_lat_g_max")
    lat_mean, lat_mean_conf = _channel_confidence(state, "accel_lat_g_mean")
    if lat_max is None or lat_mean is None:
        return 0.5, _sparse_conf(0.5)
    util = 1.0 - _clip01(abs(lat_max - lat_mean))
    assert lat_max_conf is not None and lat_mean_conf is not None
    worst = _worst_confidence([lat_max_conf, lat_mean_conf], util)
    return util, worst


def traction(
    state: CornerPhaseStateWithConfidence,
    env: EnvironmentFrame,
    aero: AeroSurface | None,
    baselines: CarBaselines,
) -> tuple[float, Confidence]:
    diff, conf = _channel_confidence(state, "wheel_speed_max_diff_ms")
    if diff is None:
        return 0.5, _sparse_conf(0.5)
    diff = max(diff, 0.0)
    util = 1.0 - _clip01(diff / baselines.wheelspin_scale_ms)
    assert conf is not None
    return util, _confidence_with_value(conf, util)


def aero_eff(
    state: CornerPhaseStateWithConfidence,
    env: EnvironmentFrame,
    aero: AeroSurface | None,
    baselines: CarBaselines,
) -> tuple[float, Confidence]:
    if aero is None:
        return 0.5, _sparse_conf(0.5)
    ld = _aero_ld_for_state(state, env, aero)
    util = _clip01(ld / 4.0) * _wind_aero_scale(env)
    contribs = [
        c for c in (
            state.states.get("lf_ride_height_mean_mm"),
            state.states.get("lr_ride_height_mean_mm"),
        )
        if c is not None
    ]
    if not contribs:
        return util, _sparse_conf(util)
    return util, _worst_confidence(contribs, util)


def _wind_aero_scale(env: EnvironmentFrame) -> float:
    """Approximate wind-induced aero penalty for the L/D utilization.

    VISION §10 calls for asymmetric headwind/tailwind/crosswind aero
    correction. A proper directional decomposition needs per-corner car
    heading (still pending — `physics/wind.py` documents this as Stage
    5 polish). As a load-bearing first integration we apply the
    *magnitude*-based downforce scale (fallback when no car heading available).
    # Initiative 3 (directional wind quick win): when car_heading_deg is
    # present on the state/env, the full decompose + modifier from wind.py
    # is used instead.
    relative to still air because tailwind reduces local airspeed and
    crosswind shears flow over the underbody. This reads ``WindVel``
    (already on every EnvironmentFrame) so the score actually reflects
    the channel rather than ignoring it.

    Treats ``wind_vel_ms`` as a tailwind worst case
    (``aero_wind_modifier(headwind=-wind_vel)``) and clamps to the
    documented 0.25 floor; returns 1.0 (no penalty) when wind is zero
    or NaN.
    """
    from racingoptimizer.physics.wind import aero_wind_modifier

    wind = float(env.wind_vel_ms)
    if not (wind == wind) or wind <= 0.0:  # NaN / no wind
        return 1.0
    downforce_scale, _balance_shift = aero_wind_modifier(
        headwind_ms=-wind, crosswind_ms=0.0,
    )
    return float(max(0.25, min(1.0, downforce_scale)))


def platform(
    state: CornerPhaseStateWithConfidence,
    env: EnvironmentFrame,
    aero: AeroSurface | None,
    baselines: CarBaselines,
) -> tuple[float, Confidence]:
    rh_channels = (
        "lf_ride_height_mean_mm",
        "rf_ride_height_mean_mm",
        "lr_ride_height_mean_mm",
        "rr_ride_height_mean_mm",
    )
    rh_vals: list[float] = []
    rh_confs: list[Confidence] = []
    for ch in rh_channels:
        c = state.states.get(ch)
        if c is None:
            continue
        rh_vals.append(float(c.value))
        rh_confs.append(c)
    if not rh_vals:
        return 0.5, _sparse_conf(0.5)
    mean_rh = sum(rh_vals) / len(rh_vals)
    variance = sum((v - mean_rh) ** 2 for v in rh_vals) / len(rh_vals)
    rh_penalty = _clip01(variance / baselines.ride_height_variance_scale_mm)

    shock_channels = (
        "lf_shock_defl_p99_mm",
        "rf_shock_defl_p99_mm",
        "lr_shock_defl_p99_mm",
        "rr_shock_defl_p99_mm",
    )
    shock_vals: list[float] = []
    shock_confs: list[Confidence] = []
    for ch in shock_channels:
        c = state.states.get(ch)
        if c is None:
            continue
        shock_vals.append(float(c.value))
        shock_confs.append(c)

    # Bottoming penalty (VISION §4: "platform control — ride height
    # consistency, bottoming risk"). When ANY of the four corners' predicted
    # mean ride height drops below the safety floor + ramp depth, the
    # platform util gets pushed toward zero. The check uses the worst
    # corner's RH so a single bottoming corner kills the platform score
    # regardless of how clean the other three are. Linear ramp keeps the
    # DE objective's gradient smooth across the cliff.
    bottoming_penalty = 0.0
    if rh_vals:
        worst_rh = min(rh_vals)
        headroom = worst_rh - _RIDE_HEIGHT_SAFETY_FLOOR_MM
        if headroom < _BOTTOMING_PENALTY_DEPTH_MM:
            bottoming_penalty = _clip01(
                1.0 - headroom / _BOTTOMING_PENALTY_DEPTH_MM
            )

    # Telemetry-derived at-speed bottoming penalty. The four
    # `dynamic_*_rh_at_speed_mm` channels are per-session medians of the
    # 60Hz `*rideHeight` telemetry, filtered to samples where the car is
    # at high speed and going straight (full-throttle straight-line
    # running). They are the GROUND TRUTH for the at-speed pose —
    # they include real damper compression dynamics, surface effects,
    # and track-specific straight-line speeds. VISION §3 directs fits to
    # observed reality, not iRacing's setup-only AeroCalculator panel
    # (which is a user-input scratchpad, not a setup readout).
    at_speed_rh_vals: list[float] = []
    for ch in (
        "dynamic_lf_rh_at_speed_mm", "dynamic_rf_rh_at_speed_mm",
        "dynamic_lr_rh_at_speed_mm", "dynamic_rr_rh_at_speed_mm",
    ):
        c = state.states.get(ch)
        if c is None:
            continue
        at_speed_rh_vals.append(float(c.value))
    if at_speed_rh_vals:
        worst_at_speed_rh = min(at_speed_rh_vals)
        headroom = worst_at_speed_rh - _RIDE_HEIGHT_SAFETY_FLOOR_MM
        if headroom < _BOTTOMING_PENALTY_DEPTH_MM:
            at_speed_bottoming = _clip01(
                1.0 - headroom / _BOTTOMING_PENALTY_DEPTH_MM
            )
            bottoming_penalty = max(bottoming_penalty, at_speed_bottoming)

    if shock_vals:
        shock_penalty = _clip01(max(shock_vals) / baselines.shock_defl_scale_mm)
        penalty = max(rh_penalty, shock_penalty, bottoming_penalty)
        contribs = rh_confs + shock_confs
    else:
        penalty = max(rh_penalty, bottoming_penalty)
        contribs = rh_confs

    util = 1.0 - penalty
    return util, _worst_confidence(contribs, util)


_SUB_FUNCS: dict[str, _SubFn] = {
    "grip": grip,
    "balance": balance,
    "stability": stability,
    "traction": traction,
    "aero_eff": aero_eff,
    "platform": platform,
}


def aggregate_utilization(
    state: CornerPhaseStateWithConfidence,
    phase: Phase,
    env: EnvironmentFrame,
    aero: AeroSurface | None,
    baselines: CarBaselines,
    *,
    phase_weights: dict[Phase, dict[str, float]] | None = None,
) -> tuple[float, Confidence]:
    """Weighted-sum sub-utilization for one (corner, phase).

    ``phase_weights`` overrides the default ``PHASE_WEIGHTS`` table —
    used by the wet-mode branch in ``score_setup`` to reweight away from
    aero_eff toward platform/grip in wet conditions per VISION §10.
    """
    table = phase_weights if phase_weights is not None else PHASE_WEIGHTS
    weights = table[phase]
    total = 0.0
    contribs: list[tuple[float, Confidence]] = []
    has_significant_sparse = False
    worst_rank = _REGIME_RANK["dense"]

    for sub in SUB_UTILIZATIONS:
        w = weights[sub]
        if w == 0.0:
            continue
        value, conf = _SUB_FUNCS[sub](state, env, aero, baselines)
        total += w * value
        contribs.append((w, conf))
        rank = _REGIME_RANK[conf.regime]
        if w > 0.1 and conf.regime == "sparse":
            has_significant_sparse = True
        if rank < worst_rank:
            worst_rank = rank

    if not contribs:
        return 0.5, _sparse_conf(0.5)

    regime: Regime
    if has_significant_sparse:
        regime = "sparse"
    else:
        regime = _RANK_TO_REGIME[worst_rank]

    n_samples = min((c.n_samples for _, c in contribs), default=0)
    spread = sum(w * (c.hi - c.lo) for w, c in contribs)
    half_spread = spread / 2.0
    # Hybrid sub-utilizations can momentarily exceed 1.0 when physics +
    # surrogate align strongly; clamp into [0, 1] so the Confidence
    # bracket invariant `lo <= value <= hi` holds. The downstream
    # objective uses `total` (the raw weighted sum) for ranking,
    # while the Confidence bracket reports the clamped value.
    clamped = _clip01(total)
    return total, Confidence(
        value=clamped,
        lo=max(0.0, clamped - half_spread),
        hi=min(1.0, clamped + half_spread),
        n_samples=int(n_samples),
        regime=regime,
    )


def score_setup(
    model: PhysicsModel,
    setup: dict[str, float],
    track: str,
    env: EnvironmentFrame,
    *,
    weights: dict[int, float] | None = None,
    strict: bool = False,
    schedule: list | None = None,
    quali: bool = False,
    hybrid: bool = True,
) -> float:
    """Sum of per-(corner, phase) utilization * per-corner weight.

    `setup` is silently clamped to constraints (defensive — the caller is
    expected to have done it but real-world telemetry baselines sometimes
    drift below the legal floor). Pass `strict=True` to raise on drift
    instead. `weights` is the per-corner time-sensitivity table from
    weight_corners; when None, a uniform fall-back across the model's
    corners is used.

    ``schedule`` (per-car v4 only): when the model is per-car
    (``feature_schema_version >= 4``), pass a list of
    ``CornerScheduleEntry`` describing the TARGET track's corners +
    archetype features. The per-corner predict path then scores the target
    track's corners instead of the trained-track corners (which a per-car
    model has none of, by design).

    ``quali=True`` swaps to the quali-stint phase-weight overlay
    (``physics.quali_mode.quali_phase_weights``) so the optimizer
    pushes for outright single-lap pace instead of race-distance
    consistency. The caller is expected to also pin a low fuel value
    via the recommendation pin mechanism.
    """
    setup = _clamped_or_raise(model, setup, strict=strict)
    if weights is None:
        weights = _resolve_weights(model, track, schedule=schedule)
    aero = _aero_surface_or_none(model)
    baselines, phase_weights = _conditions_adjusted_baselines(
        model, env, quali=quali,
    )
    if int(model.feature_schema_version) >= 4:
        if schedule is None:
            raise ValueError(
                "per-car model (v4) requires `schedule` — build it from the "
                "target TrackModel via "
                "`racingoptimizer.physics.corner_schedule.build_corner_schedule`."
            )
        breakdown = _score_breakdown_per_car(
            model, setup, env, aero, schedule, weights, baselines,
            phase_weights=phase_weights, hybrid=hybrid, track=track,
        )
    else:
        keys = _corner_phase_keys(model)
        breakdown = _score_breakdown(
            model, setup, env, aero, keys, weights, baselines,
            phase_weights=phase_weights, hybrid=hybrid,
        )
    return float(sum(breakdown.values()))


def score_breakdown(
    model: PhysicsModel,
    setup: dict[str, float],
    track: str,
    env: EnvironmentFrame,
    *,
    weights: dict[int, float] | None = None,
    strict: bool = False,
    schedule: list | None = None,
    quali: bool = False,
    hybrid: bool = True,
) -> dict[CornerPhaseKey, float]:
    setup = _clamped_or_raise(model, setup, strict=strict)
    if weights is None:
        weights = _resolve_weights(model, track, schedule=schedule)
    aero = _aero_surface_or_none(model)
    baselines, phase_weights = _conditions_adjusted_baselines(
        model, env, quali=quali,
    )
    if int(model.feature_schema_version) >= 4:
        if schedule is None:
            raise ValueError(
                "per-car model (v4) requires `schedule`"
            )
        return _score_breakdown_per_car(
            model, setup, env, aero, schedule, weights, baselines,
            phase_weights=phase_weights, hybrid=hybrid, track=track,
        )
    keys = _corner_phase_keys(model)
    return _score_breakdown(
        model, setup, env, aero, keys, weights, baselines,
        phase_weights=phase_weights, hybrid=hybrid,
    )


# ---- internals -----------------------------------------------------------


def _conditions_adjusted_baselines(
    model: PhysicsModel,
    env: EnvironmentFrame,
    *,
    quali: bool = False,
) -> tuple[CarBaselines, dict[Phase, dict[str, float]] | None]:
    """Pick baselines + phase weights for the env regime and stint mode.

    Two orthogonal axes:

    * **Wet vs dry** (VISION §10): ``physics.wet_mode.classify_conditions``
      reads ``env`` and chooses baselines / phase weights that reflect
      the regime — lower grip + aero baseline on wet, more platform +
      grip weight, less aero_eff weight.
    * **Race vs quali** (VISION §4 / §5): when ``quali=True``, swap to
      ``physics.quali_mode.quali_phase_weights`` so the optimizer
      pushes for outright single-lap pace (more aero_eff, more grip
      utilisation, less platform conservatism). Quali takes precedence
      over wet's phase-weight pick — a wet quali still wants outright
      one-lap pace, just on a wet-adjusted baseline.

    Returns ``None`` for the phase-weight override only when both axes
    are at their no-op default (race + dry); the caller then uses the
    package-default ``PHASE_WEIGHTS``.
    """
    from racingoptimizer.physics.quali_mode import quali_phase_weights
    from racingoptimizer.physics.wet_mode import (
        classify_conditions,
        wet_baselines,
        wet_phase_weights,
    )

    regime = classify_conditions(env)
    baselines = (
        model.resolved_baselines if regime == "dry"
        else wet_baselines(model.car, regime, base=model.resolved_baselines)
    )
    if quali:
        # Quali wants outright pace regardless of conditions; the wet
        # baseline still scales grip down so a wet quali setup is
        # sensibly less aggressive than a dry one.
        return baselines, quali_phase_weights()
    if regime == "dry":
        return baselines, None
    return baselines, wet_phase_weights(regime)


def _confidence_with_value(source: Confidence, value: float) -> Confidence:
    half = (source.hi - source.lo) / 2.0
    return Confidence(
        value=value,
        lo=max(0.0, value - half),
        hi=min(1.0, value + half),
        n_samples=source.n_samples,
        regime=source.regime,
    )


def _worst_confidence(contribs: list[Confidence], value: float) -> Confidence:
    rank = min(_REGIME_RANK[c.regime] for c in contribs)
    half = max((c.hi - c.lo) / 2.0 for c in contribs)
    n = min(c.n_samples for c in contribs)
    return Confidence(
        value=value,
        lo=max(0.0, value - half),
        hi=min(1.0, value + half),
        n_samples=int(n),
        regime=_RANK_TO_REGIME[rank],
    )


def _aero_ld_for_state(
    state: CornerPhaseStateWithConfidence,
    env: EnvironmentFrame,
    aero: AeroSurface,
) -> float:
    front = state.states.get("lf_ride_height_mean_mm")
    rear = state.states.get("lr_ride_height_mean_mm")
    if front is None or rear is None:
        bounds = aero.bounds
        front_v = float(bounds.front_rh_mm[0] + bounds.front_rh_mm[1]) / 2.0
        rear_v = float(bounds.rear_rh_mm[0] + bounds.rear_rh_mm[1]) / 2.0
    else:
        front_v = float(front.value)
        rear_v = float(rear.value)
    wing = float(aero.bounds.wing_angles[len(aero.bounds.wing_angles) // 2])
    _balance, ld = aero.interpolate(front_v, rear_v, wing, float(env.air_density))
    return float(ld)


# Cache one AeroSurface per (car). Loading a surface re-parses every wing
# JSON for the car; called inside the optimisation hot loop, so memoise it.
_AERO_CACHE: dict[str, AeroSurface | None] = {}


def _aero_surface_or_none(model: PhysicsModel) -> AeroSurface | None:
    if not model.aero_correction_available:
        return None
    if model.car in _AERO_CACHE:
        return _AERO_CACHE[model.car]
    try:
        from racingoptimizer.aero import load_aero_maps
        surface = load_aero_maps(model.car)
    except Exception:  # pragma: no cover — aero_correction_available is the gate
        surface = None
    _AERO_CACHE[model.car] = surface
    return surface


def _corner_phase_keys(model: PhysicsModel) -> list[tuple[int, str]]:
    """Distinct (corner_id, phase) tuples this model has any fitter for.

    Stage-3 keys are 3-tuples (corner_id, phase, channel); legacy keys are
    4-tuples (param, corner_id, phase, channel). This helper handles both.
    """
    seen: set[tuple[int, str]] = set()
    for key in model.fitters:
        if len(key) == 3:
            corner_id, phase, _channel = key
        elif len(key) == 4:
            _param, corner_id, phase, _channel = key
        else:
            continue
        seen.add((int(corner_id), str(phase)))
    return sorted(seen)


def _clamped_or_raise(
    model: PhysicsModel, setup: dict[str, float], *, strict: bool,
) -> dict[str, float]:
    """Return a copy of `setup` with every constrained parameter clamped.

    `strict=True` raises ValueError on drift instead of clamping; the
    `recommend` post-clamp uses this to detect optimizer bugs.
    """
    table = model.constraints
    if table is None:
        return dict(setup)
    out = dict(setup)
    for name, value in setup.items():
        if name not in table.parameters():
            continue
        result = clamp(float(value), name, model.car, table)
        if result.was_clamped:
            if strict:
                raise ValueError(
                    f"setup parameter {name!r}={value!r} is out of bounds "
                    f"for car={model.car!r}; expected in {result.bound!r}"
                )
            out[name] = float(result.value)
    return out


def _resolve_weights(
    model: PhysicsModel,
    track: str,
    *,
    schedule: list | None = None,
) -> dict[int, float]:
    if schedule:
        corners = sorted({int(entry.corner_id) for entry in schedule})
        if corners:
            return {c: 1.0 / len(corners) for c in corners}
    keys = _corner_phase_keys(model)
    if not keys:
        return {}
    corners = sorted({c for c, _ in keys})
    return {c: 1.0 / len(corners) for c in corners}


def _score_breakdown(
    model: PhysicsModel,
    setup: dict[str, float],
    env: EnvironmentFrame,
    aero: AeroSurface | None,
    keys: list[tuple[int, str]],
    weights: dict[int, float],
    baselines: CarBaselines,
    *,
    phase_weights: dict[Phase, dict[str, float]] | None = None,
    hybrid: bool = True,
) -> dict[CornerPhaseKey, float]:
    out: dict[CornerPhaseKey, float] = {}
    for corner_id, phase_str in keys:
        phase = Phase(phase_str)
        cpkey = CornerPhaseKey(
            session_id="<recommend-virtual>",
            lap_index=0,
            corner_id=int(corner_id),
            phase=phase,
        )
        state = model.predict(setup, env, cpkey)
        if not state.states:
            continue
        value = _corner_phase_objective_value(
            model, setup, env, aero, state, int(corner_id), phase_str,
            weights, baselines, phase_weights=phase_weights, hybrid=hybrid,
        )
        if value is None:
            continue
        out[cpkey] = value
    return out


def _confidence_adjusted_utilization(util: float, conf: Confidence) -> float:
    """Conservatively score uncertain predictions inside the objective."""
    multiplier = _OBJECTIVE_CONFIDENCE_MULTIPLIER[conf.regime]
    return float(util * multiplier)


def _score_breakdown_per_car(
    model: PhysicsModel,
    setup: dict[str, float],
    env: EnvironmentFrame,
    aero: AeroSurface | None,
    schedule: list,
    weights: dict[int, float],
    baselines: CarBaselines,
    *,
    phase_weights: dict[Phase, dict[str, float]] | None = None,
    hybrid: bool = True,
    track: str | None = None,
) -> dict[CornerPhaseKey, float]:
    """Per-car (v4) score path: iterate the TARGET track's corners.

    ``schedule`` is a list of ``CornerScheduleEntry`` (corner_id, phase,
    archetype dict). For each entry, the per-car model is queried with the
    archetype features so the same fitter scores any corner on any track.

    When ``hybrid`` is True (default), combines the surrogate utilization
    score with the physics evaluator via ``hybrid_optimizer.hybrid_score``.
    """
    out: dict[CornerPhaseKey, float] = {}
    phase_durations: list[float] = []
    for entry in schedule:
        pdur = float(entry.archetype.get("phase_duration_s", 0.0) or 0.0)
        phase_durations.append(pdur)
    total_phase_dur = sum(d for d in phase_durations if d > 0.0)
    for entry in schedule:
        corner_id = int(entry.corner_id)
        phase_str = str(entry.phase)
        try:
            phase = Phase(phase_str)
        except ValueError:
            continue
        cpkey = CornerPhaseKey(
            session_id="<recommend-virtual>",
            lap_index=0,
            corner_id=corner_id,
            phase=phase,
        )
        state = model.predict(
            setup, env, cpkey, corner_archetype=entry.archetype, track=track,
        )
        if not state.states:
            continue
        pdur = float(entry.archetype.get("phase_duration_s", 0.0) or 0.0)
        phase_weight = (
            (pdur / total_phase_dur) if total_phase_dur > 0.0 and pdur > 0.0
            else None
        )
        value = _corner_phase_objective_value(
            model, setup, env, aero, state, corner_id, phase_str,
            weights, baselines, phase_weights=phase_weights, hybrid=hybrid,
            archetype=entry.archetype,
            phase_duration_weight=phase_weight,
        )
        if value is None:
            continue
        out[cpkey] = value
    return out


def _aero_balance_ld_from_state(
    setup: dict[str, float],
    state: CornerPhaseStateWithConfidence,
    env: EnvironmentFrame,
    aero: AeroSurface | None,
) -> tuple[float, float]:
    """Query aero map balance + L/D at predicted ride heights and setup wing."""
    if aero is None:
        return _DEFAULT_AERO_BALANCE_PCT, _DEFAULT_AERO_LD
    front = state.states.get("lf_ride_height_mean_mm")
    rear = state.states.get("lr_ride_height_mean_mm")
    if front is None or rear is None:
        bounds = aero.bounds
        front_v = float(bounds.front_rh_mm[0] + bounds.front_rh_mm[1]) / 2.0
        rear_v = float(bounds.rear_rh_mm[0] + bounds.rear_rh_mm[1]) / 2.0
    else:
        front_v = float(front.value)
        rear_v = float(rear.value)
    wing = setup.get("rear_wing_angle_deg")
    if wing is None:
        wing = float(aero.bounds.wing_angles[len(aero.bounds.wing_angles) // 2])
    balance, ld = aero.interpolate(
        front_v, rear_v, float(wing), float(env.air_density),
    )
    return float(balance), float(ld)


def _speed_ms_from_archetype(
    archetype: dict | None,
    state: CornerPhaseStateWithConfidence,
) -> float:
    if archetype:
        for key in (
            "corner_apex_speed_ms",
            "apex_speed_ms",
            "corner_max_speed_ms",
        ):
            raw = archetype.get(key)
            if raw is not None:
                speed = float(raw)
                if speed > 0.0:
                    return speed
    for key in ("corner_apex_speed_ms", "speed_ms"):
        conf = state.states.get(key)
        if conf is not None:
            speed = float(conf.value)
            if speed > 0.0:
                return speed
    return _DEFAULT_SPEED_MS


def _long_g_for_phase(phase_str: str) -> float:
    phase = phase_str.strip().lower()
    if phase == "mid_corner":
        return 0.0
    if phase in ("braking", "trail_brake"):
        return -0.5
    if phase == "exit":
        return 0.3
    return 0.0


def _headroom_reference_g(baselines: CarBaselines) -> float | None:
    """Corpus-derived lateral-G reference for headroom scoring (not surrogate self-ref)."""
    ref = float(baselines.max_lateral_g)
    if not np.isfinite(ref) or ref <= 0.5:
        return None
    return ref


def _corner_phase_objective_value(
    model: PhysicsModel,
    setup: dict[str, float],
    env: EnvironmentFrame,
    aero: AeroSurface | None,
    state: CornerPhaseStateWithConfidence,
    corner_id: int,
    phase_str: str,
    weights: dict[int, float],
    baselines: CarBaselines,
    *,
    phase_weights: dict[Phase, dict[str, float]] | None = None,
    hybrid: bool = True,
    phase_duration_weight: float | None = None,
    archetype: dict | None = None,
) -> float | None:
    """Weighted per-(corner, phase) objective for DE and score_breakdown."""
    try:
        phase = Phase(phase_str)
    except ValueError:
        return None
    util, conf = aggregate_utilization(
        state, phase, env, aero, baselines, phase_weights=phase_weights,
    )
    corner_weight = float(weights.get(int(corner_id), 0.0))
    if phase_duration_weight is not None and phase_duration_weight > 0.0:
        corner_weight = float(phase_duration_weight)
    surrogate_score = float(_confidence_adjusted_utilization(util, conf))
    if not hybrid:
        return surrogate_score * corner_weight

    ceilings_raw = getattr(model, "axle_grip_ceilings", None)
    ceilings = ceilings_raw if isinstance(ceilings_raw, dict) else {}
    front_ceil = ceilings.get("front")
    rear_ceil = ceilings.get("rear")
    if front_ceil is None or rear_ceil is None:
        return surrogate_score * corner_weight

    lat_conf = state.states.get("accel_lat_g_max")
    if lat_conf is None:
        return surrogate_score * corner_weight
    lat_g = float(lat_conf.value)
    if not np.isfinite(lat_g):
        return surrogate_score * corner_weight

    long_g = _long_g_for_phase(phase_str)
    speed_ms = _speed_ms_from_archetype(archetype, state)
    balance_pct, ld_ratio = _aero_balance_ld_from_state(setup, state, env, aero)

    headroom_ref = _headroom_reference_g(baselines)

    from racingoptimizer.physics.axle_grip import (
        axle_grip_margin,
        compute_axle_grip_ratios,
    )
    from racingoptimizer.physics.evaluator import (
        evaluate_corner_phase,
        guardrail_check,
    )
    from racingoptimizer.physics.hybrid_optimizer import hybrid_score

    aero_correction = getattr(model, "aero_residual_correction", None)
    cps = evaluate_corner_phase(
        model.car,
        corner_id,
        phase_str,
        lat_g=lat_g,
        long_g=long_g,
        speed_ms=speed_ms,
        aero_balance_pct=balance_pct,
        aero_ld_ratio=ld_ratio,
        front_ceiling=front_ceil,
        rear_ceiling=rear_ceil,
        surrogate_lat_g_ceiling=headroom_ref,
        aero_correction=aero_correction,
    )
    try:
        ratios = compute_axle_grip_ratios(
            np.array([lat_g]), np.array([long_g]), model.car,
        )
    except (KeyError, ValueError):
        return surrogate_score * corner_weight
    front_margin = float(axle_grip_margin(ratios["front"][0], front_ceil))
    rear_margin = float(axle_grip_margin(ratios["rear"][0], rear_ceil))
    guard = guardrail_check(
        cps, front_margin=front_margin, rear_margin=rear_margin,
    )
    hs = hybrid_score(
        car=model.car,
        corner_id=corner_id,
        phase=phase_str,
        physics_score=cps.composite_score,
        surrogate_score=surrogate_score,
        over_axle_ceiling=guard.over_axle_ceiling,
        severely_off_balance=guard.severely_off_balance,
        grip_inconsistency=guard.grip_inconsistency,
    )
    return float(hs.hybrid_score * corner_weight)


def guardrail_warnings_for_setup(
    model: PhysicsModel,
    setup: dict[str, float],
    env: EnvironmentFrame,
    schedule: list,
    *,
    quali: bool = False,
    max_warnings: int = 5,
) -> list[str]:
    """Collect physics guardrail violations for the recommended setup."""
    if not schedule:
        return []
    ceilings_raw = getattr(model, "axle_grip_ceilings", None)
    ceilings = ceilings_raw if isinstance(ceilings_raw, dict) else {}
    if not ceilings.get("front") or not ceilings.get("rear"):
        return []

    from racingoptimizer.corner import CornerPhaseKey, Phase
    from racingoptimizer.physics.axle_grip import (
        axle_grip_margin,
        compute_axle_grip_ratios,
    )
    from racingoptimizer.physics.corner_schedule import (
        is_real_corner_archetype,
    )
    from racingoptimizer.physics.evaluator import (
        evaluate_corner_phase,
        guardrail_check,
    )

    aero = _aero_surface_or_none(model)
    aero_correction = getattr(model, "aero_residual_correction", None)
    warnings: list[str] = []
    # P0.4 -- collapse warnings per corner. The DE objective evaluates
    # each (corner, phase) but a single corner triggering across all
    # five phases produces noisy NOTES output ("T0 braking ...", "T0
    # exit ...", "T0 mid_corner ...", ...). Keep only the worst phase
    # per corner so the user sees one diagnostic line per corner.
    by_corner: dict[int, tuple[str, float, float, float]] = {}
    seen: set[str] = set()
    for entry in schedule:
        corner_id = int(entry.corner_id)
        phase_str = str(entry.phase)
        # P0.4: drop phantom corner slots (start/finish straight,
        # pit-out) whose archetype isn't corner-like. These produce
        # nonsense ``T0 ... physics-vs-surrogate divergence`` lines
        # in the NOTES block.
        if not is_real_corner_archetype(getattr(entry, "archetype", None)):
            continue
        try:
            phase = Phase(phase_str)
        except ValueError:
            continue
        cpkey = CornerPhaseKey(
            session_id="<guardrail-check>",
            lap_index=0,
            corner_id=corner_id,
            phase=phase,
        )
        try:
            state = model.predict(
                setup, env, cpkey, corner_archetype=entry.archetype,
            )
        except (ValueError, KeyError):
            continue
        lat_conf = state.states.get("accel_lat_g_max")
        if lat_conf is None:
            continue
        lat_g = float(lat_conf.value)
        if not np.isfinite(lat_g) or lat_g <= 0.5:
            continue
        long_g = _long_g_for_phase(phase_str)
        speed_ms = _speed_ms_from_archetype(entry.archetype, state)
        balance_pct, ld_ratio = _aero_balance_ld_from_state(setup, state, env, aero)
        headroom_ref = _headroom_reference_g(model.resolved_baselines)
        cps = evaluate_corner_phase(
            model.car,
            corner_id,
            phase_str,
            lat_g=lat_g,
            long_g=long_g,
            speed_ms=speed_ms,
            aero_balance_pct=balance_pct,
            aero_ld_ratio=ld_ratio,
            front_ceiling=ceilings["front"],
            rear_ceiling=ceilings["rear"],
            surrogate_lat_g_ceiling=headroom_ref,
            aero_correction=aero_correction,
        )
        try:
            ratios = compute_axle_grip_ratios(
                np.array([lat_g]), np.array([long_g]), model.car,
            )
        except (KeyError, ValueError):
            continue
        front_margin = float(axle_grip_margin(ratios["front"][0], ceilings["front"]))
        rear_margin = float(axle_grip_margin(ratios["rear"][0], ceilings["rear"]))
        guard = guardrail_check(
            cps, front_margin=front_margin, rear_margin=rear_margin,
        )
        if not guard.flagged or not guard.reason:
            continue
        severity = max(front_margin, rear_margin)
        existing = by_corner.get(corner_id)
        if existing is None or severity > existing[3]:
            by_corner[corner_id] = (
                f"T{corner_id} {phase_str}: {guard.reason} "
                f"(front margin {front_margin:.2f}, rear {rear_margin:.2f})",
                front_margin,
                rear_margin,
                severity,
            )

    for corner_id in sorted(by_corner):
        msg = by_corner[corner_id][0]
        if msg in seen:
            continue
        seen.add(msg)
        warnings.append(msg)
        if len(warnings) >= max_warnings:
            break
    return warnings


__all__ = [
    "aero_eff",
    "aggregate_utilization",
    "balance",
    "guardrail_warnings_for_setup",
    "grip",
    "platform",
    "score_breakdown",
    "score_setup",
    "stability",
    "traction",
]
