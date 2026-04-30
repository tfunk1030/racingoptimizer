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
                except UntrainedError:
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


def _env_to_array(env: EnvironmentFrame) -> np.ndarray:
    return np.array(
        [
            float(env.air_density),
            float(env.track_temp_c),
            float(env.wind_vel_ms),
            float(env.wind_dir_deg),
            float(env.track_wetness),
        ],
        dtype=np.float64,
    )


__all__ = [
    "AERO_DEPENDENT_CHANNELS",
    "CornerPhaseStateWithConfidence",
    "FitRecord",
    "PhysicsModel",
]
