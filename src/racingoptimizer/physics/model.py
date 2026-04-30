"""`PhysicsModel` + `predict` (training-side; score/recommend deferred to U10).

Stores one fitter per (parameter, corner_id, phase, output_channel) plus
ontology, constraints, and aero-availability flag. `predict` linearly combines
the per-parameter contributions (Gaussian sum-of-vars) and wraps the result
in `Confidence.derive(...)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from racingoptimizer.confidence import Confidence
from racingoptimizer.constraints import ConstraintsTable
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner import CornerPhaseKey, Phase
from racingoptimizer.physics.baselines import (
    DEFAULT_BASELINES,
    CarBaselines,
    default_baselines_for,
)
from racingoptimizer.physics.exceptions import UntrainedError
from racingoptimizer.physics.fitters import FitterBase
from racingoptimizer.physics.ontology import ParameterSpec

# Spec §9: when slice C is unavailable, regimes for high-speed channels
# degrade one tier. These output channels are downforce-derived per spec §6
# (grip / aero_eff / platform). U9 currently fits the same set of state
# columns regardless; the downgrade applies during prediction.
AERO_DEPENDENT_CHANNELS: frozenset[str] = frozenset(
    {
        "lf_ride_height_mean_mm",
        "rf_ride_height_mean_mm",
        "lr_ride_height_mean_mm",
        "rr_ride_height_mean_mm",
    }
)

_REGIME_DOWNGRADE: dict[str, str] = {
    "dense": "confident",
    "confident": "noisy",
    "noisy": "sparse",
    "sparse": "sparse",
}


@dataclass(frozen=True, slots=True)
class CornerPhaseStateWithConfidence:
    corner_phase_key: CornerPhaseKey
    states: dict[str, Confidence]
    untrained_channels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FitRecord:
    fitter: FitterBase
    n_samples: int
    cv_residual_std: float
    signal_std: float


@dataclass(frozen=True, slots=True)
class PhysicsModel:
    car: str
    session_ids: tuple[str, ...]
    track_models_used: dict[str, str] = field(default_factory=dict)
    # Key: (parameter, corner_id, phase, output_channel). Value: FitRecord.
    fitters: dict[tuple[str, int, str, str], FitRecord] = field(default_factory=dict)
    ontology: dict[str, ParameterSpec] = field(default_factory=dict)
    constraints: ConstraintsTable | None = None
    untrained_parameters: tuple[str, ...] = ()
    aero_correction_available: bool = False
    baseline_setup: dict[str, float] = field(default_factory=dict)
    seed: int = 0xC0FFEE
    # None on PhysicsModels pickled before this field existed; read via
    # `resolved_baselines` for the cold-start fallback.
    car_baselines: CarBaselines | None = None
    # Env-feature schema the per-quadruple fitters were trained against.
    # v1 = 5-channel env (air_density, track_temp_c, wind_vel_ms,
    # wind_dir_deg, track_wetness). v2 = VISION section 10 12-channel set.
    # Pre-S2.2 pickles deserialise without this field — `__setstate__`
    # backfills v1 so the slot is always initialised on revive. New models
    # written by `fit` set this to the current version
    # (`ENV_FEATURE_SCHEMA_VERSION` in fitter.py).
    feature_schema_version: int = 2

    @property
    def resolved_baselines(self) -> CarBaselines:
        """Return `car_baselines` if set, else the per-car cold-start default.

        Backward-compat shim for pickles produced before the
        `car_baselines` field existed.
        """
        if self.car_baselines is not None:
            return self.car_baselines
        return DEFAULT_BASELINES.get(
            self.car, default_baselines_for(self.car),
        )

    def __setstate__(self, state: object) -> None:
        # Frozen+slots dataclasses pickle as a positional list ordered by
        # `__slots__` (one value per slot). Older protocols may pass a
        # ``(None, slots_dict)`` tuple or a bare slots dict — handle all
        # three. Pre-S2.2 pickles lack `feature_schema_version` and pickles
        # made before `car_baselines` existed lack that slot too; backfill
        # both so every slot is initialised on revive.
        slots_order = list(type(self).__slots__)
        slot_values: dict[str, object] = {}
        if isinstance(state, list):
            # Positional list, indexed by `__slots__`. Older shorter lists
            # leave any trailing slot unset; backfill below.
            for name, value in zip(slots_order, state, strict=False):
                slot_values[name] = value
        elif isinstance(state, tuple) and len(state) == 2:
            _instance_dict, slots = state
            if isinstance(slots, dict):
                slot_values.update(slots)
            elif isinstance(slots, list):
                for name, value in zip(slots_order, slots, strict=False):
                    slot_values[name] = value
        elif isinstance(state, dict):
            slot_values.update(state)
        slot_values.setdefault("feature_schema_version", 1)
        slot_values.setdefault("car_baselines", None)
        for name, value in slot_values.items():
            object.__setattr__(self, name, value)

    def score_setup(
        self,
        setup: dict[str, float],
        track: str,
        env: EnvironmentFrame,
    ) -> float:
        # Local import sidesteps the module-graph cycle: score imports model.
        from racingoptimizer.physics.score import score_setup as _score
        return _score(self, setup, track, env)

    def recommend(
        self,
        track: str,
        env: EnvironmentFrame,
        constraints: ConstraintsTable,
    ):
        from racingoptimizer.physics.recommend import recommend as _recommend
        return _recommend(self, track, env, constraints)

    def predict(
        self,
        setup: dict[str, float],
        env: EnvironmentFrame,
        corner_phase_key: CornerPhaseKey,
    ) -> CornerPhaseStateWithConfidence:
        corner_id = corner_phase_key.corner_id
        phase = (
            corner_phase_key.phase.value
            if isinstance(corner_phase_key.phase, Phase)
            else str(corner_phase_key.phase)
        )

        # Identify the channels with at least one trained fitter at this (corner, phase).
        channels: dict[str, list[tuple[str, FitRecord]]] = {}
        for (param, c_id, ph, channel), record in self.fitters.items():
            if c_id != corner_id or ph != phase:
                continue
            if not record.fitter.is_trained:
                continue
            channels.setdefault(channel, []).append((param, record))

        # Backward compat: v1 models were fit on a 5-feature env vector
        # (air_density, track_temp_c, wind_vel_ms, wind_dir_deg,
        # track_wetness) — a different set than the v2 12-feature prefix,
        # so build the v1 vector explicitly. v2+ models get the full
        # 12-feature vector matching `fitter._ENV_COLUMNS`.
        if int(self.feature_schema_version) < 2:
            env_features = _env_to_array_v1(env)
        else:
            env_features = _env_to_array(env)
        states: dict[str, Confidence] = {}
        untrained: list[str] = []

        for channel in sorted(channels.keys()):
            contribs = sorted(channels[channel])  # deterministic per-param order
            mean_sum = 0.0
            var_sum = 0.0
            min_n = None
            signal_std = 0.0
            for param, record in contribs:
                value = setup.get(param)
                if value is None:
                    value = self.baseline_setup.get(param)
                if value is None:
                    continue
                row = np.concatenate(
                    [np.array([float(value)], dtype=np.float64), env_features]
                ).reshape(1, -1)
                try:
                    mu, _sigma = record.fitter.predict(row)
                except (UntrainedError, ValueError):
                    # ValueError covers feature-count mismatches (e.g. a
                    # legacy v1 model whose underlying fitter expects 5
                    # env columns but the dispatch fed 12). Treat as
                    # "this fitter contributed nothing" rather than a
                    # hard failure so the rest of the channels still
                    # surface in the returned state.
                    continue
                mean_sum += float(mu[0])
                # Combine std via residual std (from CV) for the Confidence band;
                # sum-of-vars on the per-fitter `cv_residual_std` keeps the
                # bracket comparable across families.
                var_sum += float(record.cv_residual_std) ** 2
                signal_std = max(signal_std, float(record.signal_std))
                min_n = record.n_samples if min_n is None else min(min_n, record.n_samples)

            if min_n is None:
                # No fittable parameter contributed (all setup values missing).
                untrained.append(channel)
                continue

            confidence = Confidence.derive(
                value=mean_sum,
                n_samples=int(min_n),
                cv_residual_std=float(np.sqrt(var_sum)),
                signal_std=float(signal_std),
            )
            if not self.aero_correction_available and channel in AERO_DEPENDENT_CHANNELS:
                downgraded = _REGIME_DOWNGRADE[confidence.regime]
                if downgraded != confidence.regime:
                    confidence = Confidence(
                        value=confidence.value,
                        lo=confidence.lo,
                        hi=confidence.hi,
                        n_samples=confidence.n_samples,
                        regime=downgraded,  # type: ignore[arg-type]
                    )
            states[channel] = confidence

        return CornerPhaseStateWithConfidence(
            corner_phase_key=corner_phase_key,
            states=states,
            untrained_channels=tuple(sorted(untrained)),
        )


# Number of env features in the v1 schema (pre-S2.2). Mirrors
# `racingoptimizer.physics.fitter.ENV_FEATURE_COUNT_V1`; duplicated here
# to keep the import graph acyclic (fitter imports model, not the other
# way around).
_ENV_FEATURE_COUNT_V1: int = 5


def _env_to_array_v1(env: EnvironmentFrame) -> np.ndarray:
    """5-feature env vector matching the pre-S2.2 fitter._ENV_COLUMNS.

    Order: air_density, track_temp_c, wind_vel_ms, wind_dir_deg, track_wetness.
    Used by `PhysicsModel.predict` to reconstruct the input vector for
    pickled v1 models so the per-fitter X-shape stays valid after revive.
    NaN sentinels coerce to 0.0 (mirrors `_env_to_array` and the fit-side
    `fill_null(0.0)`).
    """
    arr = np.array(
        [
            float(env.air_density),
            float(env.track_temp_c),
            float(env.wind_vel_ms),
            float(env.wind_dir_deg),
            float(env.track_wetness),
        ],
        dtype=np.float64,
    )
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _env_to_array(env: EnvironmentFrame) -> np.ndarray:
    """12-feature env vector matching `racingoptimizer.physics.fitter._ENV_COLUMNS`.

    Field order MUST stay aligned with `_ENV_COLUMNS` in fitter.py — the
    fitter consumed columns in that order at training time, and predict
    feeds X back to it in the same order.

    NaN-valued floats (the `from_partial_row` sentinel for "missing
    channel") are coerced to 0.0 here because sklearn's GP / RF reject
    NaN inputs at predict time. The fit pipeline already does the same
    with ``fill_null(0.0)`` so the convention is consistent across train
    and infer. Bool / int channels are cast to float so the per-row
    vector stays numeric; ``-1`` int sentinels pass through as just
    another value the fitter can ignore.
    """
    raw = [
        # Atmospheric floats:
        float(env.air_temp_c),
        float(env.air_density),
        float(env.air_pressure_mbar),
        float(env.relative_humidity),
        float(env.wind_vel_ms),
        float(env.wind_dir_deg),
        float(env.fog_level),
        # Track surface floats:
        float(env.track_temp_c),
        float(env.track_wetness),
        # Discrete weather state, cast to float:
        float(env.weather_declared_wet),
        float(env.precip_type),
        float(env.skies),
    ]
    arr = np.array(raw, dtype=np.float64)
    # Substitute 0.0 for any NaN — matches the fit-side `fill_null(0.0)`.
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


__all__ = [
    "AERO_DEPENDENT_CHANNELS",
    "CornerPhaseStateWithConfidence",
    "FitRecord",
    "PhysicsModel",
]
