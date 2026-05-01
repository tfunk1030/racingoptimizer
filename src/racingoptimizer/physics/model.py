"""`PhysicsModel` + `predict` (training-side; score/recommend deferred to U10).

Stage-3 architecture (`feature_schema_version >= 3`): one multi-input fitter
per (corner_id, phase, output_channel). Its feature space is the full bounded
setup vector + 12 env channels. `predict` queries each fitter once with the
joint feature row, so changing any single setup parameter propagates through
every output channel via the trained joint mapping ("chase the chain", VISION
§3 / §5).

Backward compat: pre-Stage-3 models persisted one fitter per
(parameter, corner_id, phase, output_channel) with feature vector
`[single_param, env...]`. `predict` dispatches to the legacy
sum-of-per-parameter path when `feature_schema_version <= 2`.
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
# (grip / aero_eff / platform).
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
    # Stage 3: ordered names of every input feature the fitter consumes.
    # The first ``len(feature_names) - len(env_columns)`` entries are the
    # bounded setup parameters; the trailing entries are env channels in
    # `racingoptimizer.physics.fitter._ENV_COLUMNS` order. Pre-Stage-3
    # records (revived from legacy pickles) have ``feature_names == ()``
    # and `predict` dispatches to the legacy single-parameter path.
    feature_names: tuple[str, ...] = ()

    def __setstate__(self, state: object) -> None:
        # Legacy v1/v2 pickles serialised FitRecord with only the first
        # 4 slots (no `feature_names`). Backfill the new slot to its
        # default `()` so revive doesn't fail with AttributeError, and
        # `_predict_legacy` keeps the old per-parameter behaviour.
        slots_order = list(type(self).__slots__)
        slot_values: dict[str, object] = {}
        if isinstance(state, list):
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
        slot_values.setdefault("feature_names", ())
        for name, value in slot_values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class PhysicsModel:
    car: str
    session_ids: tuple[str, ...]
    track_models_used: dict[str, str] = field(default_factory=dict)
    # Stage 3 keying: (corner_id, phase, output_channel). Pre-Stage-3 keys
    # were (parameter, corner_id, phase, output_channel) — `predict`
    # decides which shape the dict carries via `feature_schema_version`.
    fitters: dict[tuple, FitRecord] = field(default_factory=dict)
    ontology: dict[str, ParameterSpec] = field(default_factory=dict)
    constraints: ConstraintsTable | None = None
    untrained_parameters: tuple[str, ...] = ()
    aero_correction_available: bool = False
    baseline_setup: dict[str, float] = field(default_factory=dict)
    seed: int = 0xC0FFEE
    # None on PhysicsModels pickled before this field existed.
    car_baselines: CarBaselines | None = None
    # Env-feature schema the per-quadruple fitters were trained against.
    # v1 = 5-channel env, per-parameter linear sum (pre-S2.2).
    # v2 = 12-channel env, per-parameter linear sum.
    # v3 = 12-channel env, joint multi-input model (Stage 3).
    feature_schema_version: int = 3
    # ----------------------------------------------------------------------
    # IMPORTANT: append-only beyond this point.
    #
    # PhysicsModel is a frozen+slots dataclass; pickle serialises instance
    # state as a positional list ordered by `__slots__`. Inserting a field
    # in the middle would shift every later slot's position and silently
    # corrupt revives of older pickles. New fields MUST go at the end so
    # __setstate__ can backfill defaults via `slot_values.setdefault(...)`
    # without mis-aligning legacy lists.
    # ----------------------------------------------------------------------

    # Per-parameter observed standard deviation across training sessions.
    # Used by `physics/recommend.py` to detect parameters the driver held
    # effectively constant in the training corpus — those have no learnable
    # response surface, so the recommender pins them to `baseline_setup`
    # rather than letting the DE search drift to a constraint bound. Empty
    # on PhysicsModels pickled before this field existed (default-pinned by
    # the recommender as long as `baseline_setup` is populated).
    parameter_observed_std: dict[str, float] = field(default_factory=dict)

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
        # `__slots__`. Older pickles may be shorter — backfill defaults so
        # every slot is populated on revive.
        slots_order = list(type(self).__slots__)
        slot_values: dict[str, object] = {}
        if isinstance(state, list):
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
        # Pre-S2.2 pickles lack `feature_schema_version` (default v1);
        # v2 pickles set it to 2. Stage-3 pickles set it to 3.
        slot_values.setdefault("feature_schema_version", 1)
        slot_values.setdefault("car_baselines", None)
        # `parameter_observed_std` was added after the wet-mode-enum fix.
        # Old pickles default to an empty dict; the recommender treats every
        # parameter as "observed std unknown" → falls through to the existing
        # trust-radius behaviour (no pinning, no regression).
        slot_values.setdefault("parameter_observed_std", {})
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
        if int(self.feature_schema_version) >= 3:
            return self._predict_v3(setup, env, corner_phase_key)
        return self._predict_legacy(setup, env, corner_phase_key)

    # ---- Stage 3 joint prediction ---------------------------------------

    def _predict_v3(
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

        # Stage-3 fitter keys are (corner_id, phase, output_channel).
        channels: dict[str, FitRecord] = {}
        for key, record in self.fitters.items():
            if len(key) != 3:
                # Defensive: a v1/v2 record slipped through. Skip rather
                # than misinterpret its position.
                continue
            c_id, ph, channel = key
            if c_id != corner_id or ph != phase:
                continue
            if not record.fitter.is_trained:
                continue
            channels[channel] = record

        env_features = _env_to_array(env)
        states: dict[str, Confidence] = {}
        untrained: list[str] = []

        for channel in sorted(channels.keys()):
            record = channels[channel]
            x_row = _assemble_feature_row(
                record.feature_names, setup, self.baseline_setup, env_features,
            )
            try:
                mu, sigma = record.fitter.predict(x_row.reshape(1, -1))
            except (UntrainedError, ValueError):
                untrained.append(channel)
                continue
            mean_value = float(mu[0])
            posterior_std = float(sigma[0]) if sigma.size else 0.0

            # Use the larger of CV-residual std and the per-row posterior
            # std so the bracket reflects both training-grain noise (CV)
            # and per-prediction extrapolation widening (GP posterior /
            # RF tree-spread). Spec §3 / §7 calibration target.
            bracket_std = max(float(record.cv_residual_std), posterior_std)
            confidence = Confidence.derive(
                value=mean_value,
                n_samples=int(record.n_samples),
                cv_residual_std=bracket_std,
                signal_std=float(max(record.signal_std, 1e-12)),
            )
            confidence = _maybe_downgrade_aero(
                confidence, channel, self.aero_correction_available,
            )
            states[channel] = confidence

        return CornerPhaseStateWithConfidence(
            corner_phase_key=corner_phase_key,
            states=states,
            untrained_channels=tuple(sorted(untrained)),
        )

    # ---- Legacy v1 / v2 per-parameter prediction ------------------------

    def _predict_legacy(
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

        # Legacy keys are (param, corner, phase, channel).
        channels: dict[str, list[tuple[str, FitRecord]]] = {}
        for key, record in self.fitters.items():
            if len(key) != 4:
                continue
            param, c_id, ph, channel = key
            if c_id != corner_id or ph != phase:
                continue
            if not record.fitter.is_trained:
                continue
            channels.setdefault(channel, []).append((param, record))

        # v1 models were fit on a 5-feature env vector; v2 on the
        # 12-feature vector (matching `fitter._ENV_COLUMNS`).
        if int(self.feature_schema_version) < 2:
            env_features = _env_to_array_v1(env)
        else:
            env_features = _env_to_array(env)

        states: dict[str, Confidence] = {}
        untrained: list[str] = []

        for channel in sorted(channels.keys()):
            contribs = sorted(channels[channel])
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
                    continue
                mean_sum += float(mu[0])
                var_sum += float(record.cv_residual_std) ** 2
                signal_std = max(signal_std, float(record.signal_std))
                min_n = (
                    record.n_samples if min_n is None
                    else min(min_n, record.n_samples)
                )

            if min_n is None:
                untrained.append(channel)
                continue

            confidence = Confidence.derive(
                value=mean_sum,
                n_samples=int(min_n),
                cv_residual_std=float(np.sqrt(var_sum)),
                signal_std=float(signal_std),
            )
            confidence = _maybe_downgrade_aero(
                confidence, channel, self.aero_correction_available,
            )
            states[channel] = confidence

        return CornerPhaseStateWithConfidence(
            corner_phase_key=corner_phase_key,
            states=states,
            untrained_channels=tuple(sorted(untrained)),
        )


# Number of env features in the v1 schema (pre-S2.2).
_ENV_FEATURE_COUNT_V1: int = 5


def _env_to_array_v1(env: EnvironmentFrame) -> np.ndarray:
    """5-feature env vector matching the pre-S2.2 fitter._ENV_COLUMNS.

    Order: air_density, track_temp_c, wind_vel_ms, wind_dir_deg, track_wetness.
    Used by `_predict_legacy` to reconstruct the input vector for pickled
    v1 models so the per-fitter X-shape stays valid after revive.
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

    NaN-valued floats are coerced to 0.0 here because sklearn's GP / RF
    reject NaN inputs at predict time. Bool / int channels are cast to
    float; -1 int sentinels pass through as just another value.
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
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _maybe_downgrade_aero(
    confidence: Confidence, channel: str, aero_available: bool,
) -> Confidence:
    """Spec §9: aero-derived channels lose one regime tier without aero maps."""
    if aero_available or channel not in AERO_DEPENDENT_CHANNELS:
        return confidence
    downgraded = _REGIME_DOWNGRADE[confidence.regime]
    if downgraded == confidence.regime:
        return confidence
    return Confidence(
        value=confidence.value,
        lo=confidence.lo,
        hi=confidence.hi,
        n_samples=confidence.n_samples,
        regime=downgraded,  # type: ignore[arg-type]
    )


def _assemble_feature_row(
    feature_names: tuple[str, ...],
    setup: dict[str, float],
    baseline: dict[str, float],
    env_features: np.ndarray,
) -> np.ndarray:
    """Build the joint feature row in the fitter's trained order.

    Setup-parameter slots fall through to the model baseline when the
    caller's setup omits a value (pinned-to-baseline parameter); env slots
    are pulled from the pre-built env vector by name. Unknown env channels
    fall back to 0.0 — matches the fit-side ``fill_null(0.0)`` convention.
    """
    from racingoptimizer.physics.fitter import _ENV_COLUMNS

    env_index = {name: idx for idx, name in enumerate(_ENV_COLUMNS)}
    row = np.empty(len(feature_names), dtype=np.float64)
    for i, name in enumerate(feature_names):
        if name in env_index:
            row[i] = float(env_features[env_index[name]])
            continue
        if name in setup and setup[name] is not None:
            row[i] = float(setup[name])
            continue
        if name in baseline and baseline[name] is not None:
            row[i] = float(baseline[name])
            continue
        row[i] = 0.0
    return row


__all__ = [
    "AERO_DEPENDENT_CHANNELS",
    "CornerPhaseStateWithConfidence",
    "FitRecord",
    "PhysicsModel",
]
