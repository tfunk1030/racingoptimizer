"""Static garage ride height: physics readout fitters + corpus k-NN repair.

Primary path: ``PhysicsModel.predict_setup_readouts()`` Ridge/Forest fitters
for DE feasibility, TB co-optimization, and setup-card ``[predicted]`` RH.

Corpus k-NN remains for post-DE platform repair (blend perch/pushrod/heave
toward nearest legal driven session) when physics alone predicts illegal RH.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from racingoptimizer.physics.setup_symmetry import (
    STATIC_RH_ENVELOPE_MM,
    STATIC_RH_READOUT_CHANNELS,
    apply_setup_symmetry,
)

if TYPE_CHECKING:
    from racingoptimizer.constraints import ConstraintsTable
    from racingoptimizer.context import EnvironmentFrame
    from racingoptimizer.physics.model import PhysicsModel

# Setup inputs that move static garage RH (k-NN feature vector).
STATIC_RH_PLATFORM_PARAMS: tuple[str, ...] = (
    "heave_spring_rate_n_per_mm",
    "third_spring_rate_n_per_mm",
    "heave_perch_offset_front_mm",
    "spring_perch_offset_rear_mm",
    "pushrod_length_offset_front_mm",
    "pushrod_length_offset_rear_mm",
    "torsion_bar_turns_fl",
    "torsion_bar_od_fl_mm",
    "torsion_bar_turns_rl",
    "torsion_bar_od_rl_mm",
    "fuel_level_l",
    "camber_fl_deg",
    "camber_rl_deg",
    "toe_front_mm",
    "toe_rl_mm",
)

# Coarse platform knobs adjusted by DE + corpus repair; TB turns trimmed after.
STATIC_RH_COARSE_PARAMS: tuple[str, ...] = tuple(
    name for name in STATIC_RH_PLATFORM_PARAMS
    if name not in ("torsion_bar_turns_fl", "torsion_bar_turns_rl")
)

TB_TURN_PARAMS: tuple[str, ...] = ("torsion_bar_turns_fl", "torsion_bar_turns_rl")

_TB_FRONT_CHANNELS: tuple[str, ...] = (
    "setup_static_lf_ride_height_mm",
    "setup_static_rf_ride_height_mm",
)
_TB_REAR_CHANNELS: tuple[str, ...] = (
    "setup_static_lr_ride_height_mm",
    "setup_static_rr_ride_height_mm",
)

_KNN_NEIGHBORS = 5
_KNN_EPS = 1e-6
# Normalized L2 distance above this → treat as extrapolation (reject in DE).
_MAX_EXTRAPOLATION_DISTANCE = 0.35
_HARD_REJECT_OBJECTIVE = 1e12


@dataclass(frozen=True, slots=True)
class StaticRhCorpusEntry:
    """One session's platform setup + observed static RH readouts."""

    params: tuple[tuple[str, float], ...]
    readouts: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class StaticRhKnnResult:
    readouts: dict[str, float]
    neighbor_distance: float
    extrapolated: bool


def build_static_rh_corpus(
    sid_to_params: dict[str, dict[str, float]],
    sid_to_readouts: dict[str, dict[str, float]],
) -> tuple[StaticRhCorpusEntry, ...]:
    """Build lookup table from per-session setup snapshots and YAML readouts."""
    entries: list[StaticRhCorpusEntry] = []
    for sid, params in sid_to_params.items():
        readouts_raw = sid_to_readouts.get(sid, {})
        readouts = {
            ch: float(v)
            for ch in STATIC_RH_READOUT_CHANNELS
            if (v := readouts_raw.get(ch)) is not None
        }
        if len(readouts) < 2:
            continue
        plat = {
            k: float(v)
            for k, v in params.items()
            if k in STATIC_RH_PLATFORM_PARAMS
        }
        if not plat:
            continue
        entries.append(StaticRhCorpusEntry(
            params=tuple(sorted(plat.items())),
            readouts=tuple(sorted(readouts.items())),
        ))
    return tuple(entries)


def static_rh_within_envelope(readouts: dict[str, float]) -> bool:
    lo, hi = STATIC_RH_ENVELOPE_MM
    for channel in STATIC_RH_READOUT_CHANNELS:
        value = readouts.get(channel)
        if value is None:
            continue
        rh = float(value)
        if rh < lo or rh > hi:
            return False
    return True


def has_static_rh_physics(model: PhysicsModel) -> bool:
    """True when trained joint fitters can predict static RH readouts."""
    if int(getattr(model, "feature_schema_version", 0)) < 3:
        return False
    trained = 0
    for key, record in model.fitters.items():
        channel = key[-1] if isinstance(key, tuple) and key else None
        if not isinstance(channel, str):
            continue
        if channel not in STATIC_RH_READOUT_CHANNELS:
            continue
        if record.fitter.is_trained:
            trained += 1
    return trained >= 2


def physics_static_rh_readouts(
    model: PhysicsModel,
    setup: dict[str, float],
    env: EnvironmentFrame,
) -> dict[str, float]:
    """Static RH channels from ``predict_setup_readouts`` (physics fitters)."""
    raw = model.predict_setup_readouts(setup, env)
    return {
        ch: float(raw[ch])
        for ch in STATIC_RH_READOUT_CHANNELS
        if ch in raw
    }


def static_rh_de_infeasible_readouts(readouts: dict[str, float]) -> bool:
    """True when DE must reject this candidate (missing or illegal static RH)."""
    if not readouts:
        return True
    return not static_rh_within_envelope(readouts)


def predict_static_rh_knn(
    corpus: tuple[StaticRhCorpusEntry, ...],
    setup: dict[str, float],
    *,
    car: str,
    constraints: ConstraintsTable | None,
    k: int = _KNN_NEIGHBORS,
) -> StaticRhKnnResult | None:
    """Inverse-distance k-NN over corpus sessions in normalized param space."""
    if not corpus:
        return None

    param_names = sorted({
        name
        for entry in corpus
        for name, _ in entry.params
    })
    if not param_names:
        return None

    query = np.array([float(setup.get(n, 0.0)) for n in param_names])
    spans = _param_spans(param_names, corpus, car, constraints)
    query_norm = query / spans

    dists: list[float] = []
    neighbor_readouts: list[dict[str, float]] = []
    for entry in corpus:
        vec = np.array([dict(entry.params).get(n, 0.0) for n in param_names])
        dist = float(np.linalg.norm(query_norm - vec / spans))
        dists.append(dist)
        neighbor_readouts.append(dict(entry.readouts))

    order = np.argsort(dists)
    k_eff = min(k, len(order))
    nearest_idx = order[:k_eff]
    min_dist = float(dists[int(nearest_idx[0])])

    weights = np.array([
        1.0 / (dists[int(i)] + _KNN_EPS) for i in nearest_idx
    ])
    weights /= weights.sum()

    out: dict[str, float] = {}
    for channel in STATIC_RH_READOUT_CHANNELS:
        weighted_sum = 0.0
        weight_sum = 0.0
        for wi, idx in zip(weights, nearest_idx, strict=False):
            value = neighbor_readouts[int(idx)].get(channel)
            if value is None:
                continue
            weighted_sum += float(wi) * float(value)
            weight_sum += float(wi)
        if weight_sum > 0.0:
            out[channel] = weighted_sum / weight_sum

    return StaticRhKnnResult(
        readouts=out,
        neighbor_distance=min_dist,
        extrapolated=min_dist > _MAX_EXTRAPOLATION_DISTANCE,
    )


def static_rh_de_infeasible(
    result: StaticRhKnnResult | None,
) -> bool:
    """True when k-NN rejects this candidate (illegal or extrapolated)."""
    if result is None or not result.readouts:
        return True
    if result.extrapolated:
        return True
    return not static_rh_within_envelope(result.readouts)


def _static_rh_readouts_for_setup(
    setup: dict[str, float],
    *,
    model: PhysicsModel | None = None,
    env: EnvironmentFrame | None = None,
    corpus: tuple[StaticRhCorpusEntry, ...] = (),
    car: str = "",
    constraints: ConstraintsTable | None = None,
) -> tuple[dict[str, float], StaticRhKnnResult | None]:
    """Physics readouts when available; optional k-NN fallback for repair."""
    readouts: dict[str, float] = {}
    knn: StaticRhKnnResult | None = None
    if model is not None and env is not None and has_static_rh_physics(model):
        readouts = physics_static_rh_readouts(model, setup, env)
    if not readouts and corpus:
        knn = predict_static_rh_knn(
            corpus, setup, car=car, constraints=constraints,
        )
        if knn is not None:
            readouts = dict(knn.readouts)
    return readouts, knn


def _static_rh_acceptable(
    readouts: dict[str, float],
    *,
    knn: StaticRhKnnResult | None,
    physics_authoritative: bool,
) -> bool:
    if not static_rh_within_envelope(readouts):
        return False
    if physics_authoritative:
        return True
    return knn is not None and not knn.extrapolated


def enforce_static_rh_feasible(
    setup: dict[str, float],
    corpus: tuple[StaticRhCorpusEntry, ...],
    *,
    car: str,
    constraints: ConstraintsTable | None,
    model: PhysicsModel | None = None,
    env: EnvironmentFrame | None = None,
) -> tuple[dict[str, float], bool]:
    """Blend platform params toward nearest legal corpus session until RH legal.

    Feasibility is checked with physics readout fitters when ``model`` and
    ``env`` are provided; otherwise falls back to k-NN on ``corpus``.
    """
    setup = apply_setup_symmetry(setup)
    physics_authoritative = (
        model is not None and env is not None and has_static_rh_physics(model)
    )
    readouts, knn = _static_rh_readouts_for_setup(
        setup,
        model=model,
        env=env,
        corpus=corpus,
        car=car,
        constraints=constraints,
    )
    if readouts and _static_rh_acceptable(
        readouts, knn=knn, physics_authoritative=physics_authoritative,
    ):
        return setup, False

    legal = [
        e for e in corpus
        if static_rh_within_envelope(dict(e.readouts))
    ]
    if not legal:
        return setup, True

    target = min(
        legal,
        key=lambda e: _setup_distance(setup, dict(e.params), car, constraints),
    )
    target_params = dict(target.params)

    lo, hi = 0.0, 1.0
    best = setup
    for _ in range(14):
        mid = (lo + hi) / 2.0
        trial = _lerp_platform(setup, target_params, mid)
        trial = apply_setup_symmetry(trial)
        trial_readouts, trial_knn = _static_rh_readouts_for_setup(
            trial,
            model=model,
            env=env,
            corpus=corpus,
            car=car,
            constraints=constraints,
        )
        if trial_readouts and _static_rh_acceptable(
            trial_readouts,
            knn=trial_knn,
            physics_authoritative=physics_authoritative,
        ):
            best = trial
            hi = mid
        else:
            lo = mid

    final_readouts, final_knn = _static_rh_readouts_for_setup(
        best,
        model=model,
        env=env,
        corpus=corpus,
        car=car,
        constraints=constraints,
    )
    still_bad = not (
        final_readouts
        and _static_rh_acceptable(
            final_readouts,
            knn=final_knn,
            physics_authoritative=physics_authoritative,
        )
    )
    return best, still_bad


def cooptimize_tb_for_static_rh(
    setup: dict[str, float],
    model: PhysicsModel,
    env: EnvironmentFrame,
    *,
    constraints: ConstraintsTable | None = None,
) -> tuple[dict[str, float], bool]:
    """Trim torsion-bar turns so physics static RH is legal at fixed platform.

    DE + ``enforce_static_rh_feasible`` set perch/pushrod/heave; this pass
    holds those fixed and grid-searches TB turns to minimize platform+balance
    penalty on ``predict_setup_readouts``.
    """
    if not has_static_rh_physics(model):
        return setup, False

    from racingoptimizer.physics.ontology import ontology_for, snap_to_garage_step

    try:
        onto = ontology_for(model.car)
    except KeyError:
        return setup, False

    if not any(name in onto for name in TB_TURN_PARAMS):
        return setup, True

    out = apply_setup_symmetry(setup)
    ok = True
    for param, channels in (
        ("torsion_bar_turns_fl", _TB_FRONT_CHANNELS),
        ("torsion_bar_turns_rl", _TB_REAR_CHANNELS),
    ):
        if param not in onto:
            continue
        spec = onto[param]
        if constraints is not None:
            bound = constraints.bounds(model.car, param)
        else:
            bound = None
        if bound is None:
            continue
        lo, hi = bound
        tuned, param_ok = _optimize_tb_param_physics(
            out,
            param,
            channels,
            model,
            env,
            car=model.car,
            constraints=constraints,
            bounds=(float(lo), float(hi)),
        )
        out[param] = snap_to_garage_step(tuned, spec)
        out = apply_setup_symmetry(out)
        ok = ok and param_ok

    final = physics_static_rh_readouts(model, out, env)
    success = (
        ok
        and final
        and static_rh_within_envelope(final)
    )
    return out, success


def _coarse_rh_targets_from_corpus(
    setup: dict[str, float],
    corpus: tuple[StaticRhCorpusEntry, ...],
    *,
    car: str,
    constraints: ConstraintsTable | None,
    k: int = _KNN_NEIGHBORS,
) -> dict[str, float] | None:
    """Weighted RH targets from legal sessions closest in coarse param space."""
    coarse_names = sorted(
        name
        for name in STATIC_RH_COARSE_PARAMS
        if any(name in dict(entry.params) for entry in corpus)
    )
    if not coarse_names:
        return None

    spans = _param_spans(coarse_names, corpus, car, constraints)
    query = np.array([float(setup.get(name, 0.0)) for name in coarse_names]) / spans

    dists: list[float] = []
    neighbor_readouts: list[dict[str, float]] = []
    for entry in corpus:
        readouts = dict(entry.readouts)
        if not static_rh_within_envelope(readouts):
            continue
        vec = np.array([
            float(dict(entry.params).get(name, 0.0)) for name in coarse_names
        ]) / spans
        dists.append(float(np.linalg.norm(query - vec)))
        neighbor_readouts.append(readouts)

    if not dists:
        return None

    order = np.argsort(dists)
    k_eff = min(k, len(order))
    nearest = order[:k_eff]
    weights = np.array([1.0 / (dists[int(i)] + _KNN_EPS) for i in nearest])
    weights /= weights.sum()

    targets: dict[str, float] = {}
    for channel in STATIC_RH_READOUT_CHANNELS:
        weighted_sum = 0.0
        weight_sum = 0.0
        for wi, idx in zip(weights, nearest, strict=False):
            value = neighbor_readouts[int(idx)].get(channel)
            if value is None:
                continue
            weighted_sum += float(wi) * float(value)
            weight_sum += float(wi)
        if weight_sum > 0.0:
            targets[channel] = weighted_sum / weight_sum
    return targets or None


def _optimize_tb_param_physics(
    setup: dict[str, float],
    param: str,
    channels: tuple[str, ...],
    model: PhysicsModel,
    env: EnvironmentFrame,
    *,
    car: str,
    constraints: ConstraintsTable | None,
    bounds: tuple[float, float],
    spec_step: float = 0.001,
) -> tuple[float, bool]:
    """1-D grid search on garage TB clicks to minimize physics static RH penalty."""
    from racingoptimizer.physics.ontology import ontology_for, snap_to_garage_step
    from racingoptimizer.physics.setup_symmetry import (
        static_rh_balance_penalty,
        static_rh_platform_penalty,
    )

    lo, hi = bounds
    if abs(hi - lo) < 1e-9:
        return float(lo), True

    spec = None
    try:
        spec = ontology_for(car)[param]
        spec_step = float(spec.step or spec_step)
    except KeyError:
        pass

    def objective(tb_val: float) -> float:
        trial = dict(setup)
        trial[param] = float(tb_val)
        trial = apply_setup_symmetry(trial)
        readouts = physics_static_rh_readouts(model, trial, env)
        if not readouts:
            return 1e6
        pen = static_rh_platform_penalty(readouts) + static_rh_balance_penalty(
            readouts,
        )
        if not static_rh_within_envelope(readouts):
            return 1e6 + pen * 10.0
        return pen

    def snap_val(raw: float) -> float:
        if spec is None:
            return float(raw)
        return snap_to_garage_step(float(raw), spec)

    start = snap_val(min(max(float(setup.get(param, 0.5 * (lo + hi))), lo), hi))
    best_val = start
    best_err = objective(start)
    tick = max(spec_step, 1e-6)
    n_steps = int(round((hi - lo) / tick))
    for i in range(n_steps + 1):
        candidate = snap_val(lo + i * tick)
        err = objective(candidate)
        if err < best_err:
            best_err = err
            best_val = candidate

    param_ok = best_err < 1.0
    return best_val, param_ok


def _optimize_tb_param(
    setup: dict[str, float],
    param: str,
    channels: tuple[str, ...],
    targets: dict[str, float],
    corpus: tuple[StaticRhCorpusEntry, ...],
    *,
    car: str,
    constraints: ConstraintsTable | None,
    bounds: tuple[float, float],
    spec_step: float = 0.001,
) -> tuple[float, bool]:
    """1-D grid search on garage TB clicks to match target static RH channels."""
    from racingoptimizer.physics.ontology import ParameterSpec, ontology_for, snap_to_garage_step

    channel_targets = [
        (ch, float(targets[ch]))
        for ch in channels
        if ch in targets
    ]
    if not channel_targets:
        return float(setup.get(param, 0.0)), True

    lo, hi = bounds
    if abs(hi - lo) < 1e-9:
        return float(lo), True

    spec: ParameterSpec | None = None
    try:
        spec = ontology_for(car)[param]
        spec_step = float(spec.step or spec_step)
    except KeyError:
        pass

    def objective(tb_val: float) -> float:
        trial = dict(setup)
        trial[param] = float(tb_val)
        trial = apply_setup_symmetry(trial)
        knn = predict_static_rh_knn(
            corpus, trial, car=car, constraints=constraints,
        )
        if knn is None or not static_rh_within_envelope(knn.readouts):
            return 1e6
        err = 0.0
        for channel, target in channel_targets:
            pred = knn.readouts.get(channel)
            if pred is None:
                return 1e6
            delta = float(pred) - target
            err += delta * delta
        return err

    def snap_val(raw: float) -> float:
        if spec is None:
            return float(raw)
        return snap_to_garage_step(float(raw), spec)

    start = snap_val(min(max(float(setup.get(param, 0.5 * (lo + hi))), lo), hi))
    best_val = start
    best_err = objective(start)
    tick = max(spec_step, 1e-6)
    n_steps = int(round((hi - lo) / tick))
    for i in range(n_steps + 1):
        candidate = snap_val(lo + i * tick)
        err = objective(candidate)
        if err < best_err:
            best_err = err
            best_val = candidate

    threshold = 0.25 * len(channel_targets)
    param_ok = best_err <= threshold
    return best_val, param_ok


def _param_spans(
    param_names: list[str],
    corpus: tuple[StaticRhCorpusEntry, ...],
    car: str,
    constraints: ConstraintsTable | None,
) -> np.ndarray:
    spans: list[float] = []
    for name in param_names:
        vals = [dict(e.params).get(name) for e in corpus]
        vals = [float(v) for v in vals if v is not None]
        if constraints is not None:
            bound = constraints.bounds(car, name)
            if bound is not None:
                lo, hi = bound
                span = max(hi - lo, 1e-6)
                spans.append(span)
                continue
        if len(vals) >= 2:
            spans.append(max(max(vals) - min(vals), 1e-6))
        else:
            spans.append(1.0)
    return np.array(spans)


def _setup_distance(
    setup: dict[str, float],
    target: dict[str, float],
    car: str,
    constraints: ConstraintsTable | None,
) -> float:
    names = sorted(set(setup) & set(target) & set(STATIC_RH_PLATFORM_PARAMS))
    if not names:
        return float("inf")
    spans_list: list[float] = []
    for name in names:
        if constraints is not None:
            bound = constraints.bounds(car, name)
            if bound is not None:
                spans_list.append(max(bound[1] - bound[0], 1e-6))
                continue
        spans_list.append(1.0)
    spans = np.array(spans_list)
    q = np.array([float(setup[n]) for n in names]) / spans
    t = np.array([float(target[n]) for n in names]) / spans
    return float(np.linalg.norm(q - t))


def _lerp_platform(
    setup: dict[str, float],
    target: dict[str, float],
    alpha: float,
    *,
    params: tuple[str, ...] = STATIC_RH_COARSE_PARAMS,
) -> dict[str, float]:
    out = dict(setup)
    for name in params:
        if name not in target:
            continue
        base = float(setup.get(name, target[name]))
        out[name] = base + alpha * (float(target[name]) - base)
    return out


__all__ = [
    "STATIC_RH_COARSE_PARAMS",
    "STATIC_RH_PLATFORM_PARAMS",
    "StaticRhCorpusEntry",
    "StaticRhKnnResult",
    "TB_TURN_PARAMS",
    "_HARD_REJECT_OBJECTIVE",
    "_MAX_EXTRAPOLATION_DISTANCE",
    "build_static_rh_corpus",
    "cooptimize_tb_for_static_rh",
    "enforce_static_rh_feasible",
    "has_static_rh_physics",
    "physics_static_rh_readouts",
    "predict_static_rh_knn",
    "static_rh_de_infeasible",
    "static_rh_de_infeasible_readouts",
    "static_rh_within_envelope",
]
