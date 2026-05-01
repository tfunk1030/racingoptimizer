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
    util = _clip01(ld / 4.0)
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

    if shock_vals:
        shock_penalty = _clip01(max(shock_vals) / baselines.shock_defl_scale_mm)
        penalty = max(rh_penalty, shock_penalty)
        contribs = rh_confs + shock_confs
    else:
        penalty = rh_penalty
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
) -> tuple[float, Confidence]:
    weights = PHASE_WEIGHTS[phase]
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
    return total, Confidence(
        value=total,
        lo=max(0.0, total - half_spread),
        hi=min(1.0, total + half_spread),
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
) -> float:
    """Sum of per-(corner, phase) utilization * per-corner weight.

    `setup` is silently clamped to constraints (defensive — the caller is
    expected to have done it but real-world telemetry baselines sometimes
    drift below the legal floor). Pass `strict=True` to raise on drift
    instead. `weights` is the per-corner time-sensitivity table from
    weight_corners; when None, a uniform fall-back across the model's
    corners is used.
    """
    setup = _clamped_or_raise(model, setup, strict=strict)
    if weights is None:
        weights = _resolve_weights(model, track)
    aero = _aero_surface_or_none(model)
    keys = _corner_phase_keys(model)
    baselines = model.resolved_baselines
    breakdown = _score_breakdown(model, setup, env, aero, keys, weights, baselines)
    return float(sum(breakdown.values()))


def score_breakdown(
    model: PhysicsModel,
    setup: dict[str, float],
    track: str,
    env: EnvironmentFrame,
    *,
    weights: dict[int, float] | None = None,
    strict: bool = False,
) -> dict[CornerPhaseKey, float]:
    setup = _clamped_or_raise(model, setup, strict=strict)
    if weights is None:
        weights = _resolve_weights(model, track)
    aero = _aero_surface_or_none(model)
    keys = _corner_phase_keys(model)
    baselines = model.resolved_baselines
    return _score_breakdown(model, setup, env, aero, keys, weights, baselines)


# ---- internals -----------------------------------------------------------


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


def _resolve_weights(model: PhysicsModel, track: str) -> dict[int, float]:
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
            out[cpkey] = 0.0
            continue
        util, conf = aggregate_utilization(state, phase, env, aero, baselines)
        w = weights.get(int(corner_id), 0.0)
        out[cpkey] = float(_confidence_adjusted_utilization(util, conf) * w)
    return out


def _confidence_adjusted_utilization(util: float, conf: Confidence) -> float:
    """Conservatively score uncertain predictions inside the objective."""
    multiplier = _OBJECTIVE_CONFIDENCE_MULTIPLIER[conf.regime]
    return float(util * multiplier)


__all__ = [
    "aero_eff",
    "aggregate_utilization",
    "balance",
    "grip",
    "platform",
    "score_breakdown",
    "score_setup",
    "stability",
    "traction",
]
