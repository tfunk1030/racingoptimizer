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

from racingoptimizer.aero.residual_correction import AeroResidualCorrection
from racingoptimizer.confidence import Confidence
from racingoptimizer.constraints import ConstraintsTable
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner import CornerPhaseKey, Phase
from racingoptimizer.physics.axle_grip import AxleGripCeiling
from racingoptimizer.physics.baselines import (
    DEFAULT_BASELINES,
    CarBaselines,
    default_baselines_for,
)
from racingoptimizer.physics.bayes_retrofit import BayesPosterior
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
    bootstrap_std: float = 0.0
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
        slot_values.setdefault("bootstrap_std", 0.0)
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

    # Per-track per-parameter observed value sets. Populated only by
    # `fit_per_car` (the v4 per-car path). Keyed by ``track → parameter →
    # tuple of distinct observed values across that track's sessions``.
    # Used by ``physics.recommend._pin_or_trust_bounds`` to cap the trust
    # radius to the empirical envelope on the TARGET track so the per-car
    # recommender cannot extrapolate the heave spring (or anything else)
    # outside what the driver has actually run there. Empty on per-(car,
    # track) v3 pickles and on legacy pickles produced before this field
    # existed.
    per_track_parameter_observed: dict[str, dict[str, tuple[float, ...]]] = field(
        default_factory=dict
    )

    # Hierarchical-Bayesian per-(parameter, track) posteriors (PLAN.md
    # Day 4, Mode 1). Populated by `fit_per_car` after the per-track-
    # observed dict is built; consumed by the recommender as the
    # track-aware replacement for `parameter_observed_std`. Keyed by
    # (parameter, track) -> BayesPosterior. Empty on per-(car, track) v3
    # pickles and on legacy pickles produced before this field existed
    # (the recommender's existing fallbacks handle that case).
    bayes_posteriors: dict[tuple[str, str], BayesPosterior] = field(
        default_factory=dict
    )

    # Lightweight per-track additive residual correction (cross-track
    # de-confounding). Populated only by fit_per_car (v4). For each track,
    # we compute the mean (actual - pooled_per_car_prediction) per channel
    # across that track's corner-phase rows. During v4 prediction for a
    # known target track we add the track-specific residual to the base
    # surrogate output. This counters the case where a high-sample track
    # (e.g. Sebring) drags the pooled model's predictions for a low-sample
    # track (e.g. Spa) on parameters that have real track-specific behavior.
    # Only applied for parameters/channels that show meaningful per-track
    # structure; keeps the core per-car surrogate as the primary model.
    # Structure: track -> channel -> residual (float)
    per_track_residuals: dict[str, dict[str, float]] = field(
        default_factory=dict
    )

    # Per-(car, axle) grip ceilings fitted from the training corpus during
    # `fit_per_car` (physics-rebuild Day-10 model, wired into DE in
    # post-rebuild work). Used by `physics/recommend.py` to apply an
    # additive guardrail penalty in the DE objective when a candidate
    # setup's predicted axle utilization exceeds the empirical ceiling.
    # `None` on legacy pickles (FITTERS_LAYOUT_VERSION < 4) and on cars
    # whose corpus lacks enough mid-corner samples for a stable fit; the
    # recommender treats `None` as "guardrail inactive" and falls back to
    # pure-surrogate scoring without regression.
    # Keyed by axle name: {"front": AxleGripCeiling, "rear": AxleGripCeiling}.
    axle_grip_ceilings: dict[str, AxleGripCeiling] | None = None

    # Optional per-car scalar correction applied to aero-map-derived peak
    # lat-G before evaluator grip-headroom scoring. None on legacy pickles
    # (pre-Day-11 residual-correction wiring) and on fits where insufficient
    # clean samples prevent a stable correction fit.
    aero_residual_correction: AeroResidualCorrection | None = None

    # Per-session (platform setup → static garage RH) lookup for k-NN
    # readout prediction and DE feasibility. Populated by fit/fit_per_car;
    # empty on legacy pickles (falls back to forest readout fitters only).
    static_rh_corpus: tuple = field(default_factory=tuple)

    # Per-car deterministic linear fit for the four
    # ``setup_static_*_ride_height_mm`` channels (P0.2 of
    # ``docs/accuracy-rebuild-2026-05-24/PLAN.md``). When this fit ships
    # (R^2 >= 0.98 per channel), ``predict_setup_readouts`` returns the
    # kinematic value for those channels and bypasses the noisy Ridge/
    # Forest surrogate. ``None`` on legacy pickles or when the corpus is
    # too thin for a stable fit.
    static_rh_kinematic: object | None = None

    # Per-(channel, track) closed-form Bayes random-intercepts on the
    # surrogate's training residuals (P2.2 of accuracy-rebuild-2026-05-24).
    # Empirical-Bayes partial pooling toward zero: low-sample tracks
    # shrink to no-correction, high-sample tracks with systematic
    # divergence retain a real intercept. Applied additively at predict
    # time so the surrogate's setup gradient is preserved -- only the
    # per-track level shifts. Empty dict on legacy pickles and on cars
    # whose corpus has too few tracks for partial pooling (the predict
    # path falls back to surrogate-only via ``predict_correction``'s
    # ``None``/missing-key returning ``(0.0, 0.0)``).
    track_random_intercepts: dict = field(default_factory=dict)

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
        slot_values.setdefault("per_track_parameter_observed", {})
        slot_values.setdefault("bayes_posteriors", {})
        slot_values.setdefault("per_track_residuals", {})
        slot_values.setdefault("axle_grip_ceilings", None)
        slot_values.setdefault("aero_residual_correction", None)
        slot_values.setdefault("static_rh_corpus", ())
        slot_values.setdefault("static_rh_kinematic", None)
        slot_values.setdefault("track_random_intercepts", {})
        _repair_legacy_slot_shift(slot_values)
        _validate_pickle_slots(slot_values)
        for name, value in slot_values.items():
            object.__setattr__(self, name, value)

    def score_setup(
        self,
        setup: dict[str, float],
        track: str,
        env: EnvironmentFrame,
        *,
        schedule: list | None = None,
        quali: bool = False,
    ) -> float:
        # Local import sidesteps the module-graph cycle: score imports model.
        from racingoptimizer.physics.score import score_setup as _score
        return _score(self, setup, track, env, schedule=schedule, quali=quali)

    def recommend(
        self,
        track: str,
        env: EnvironmentFrame,
        constraints: ConstraintsTable,
        *,
        schedule: list | None = None,
        quali: bool = False,
        explore_pct: float = 0.0,
        reset_mode: bool = False,
        staged: bool = False,
        surrogate_only: bool = False,
    ):
        if staged:
            from racingoptimizer.physics.recommend import recommend_staged
            return recommend_staged(
                self, track, env, constraints,
                schedule=schedule, quali=quali, explore_pct=explore_pct,
                reset_mode=reset_mode, surrogate_only=surrogate_only,
            )
        from racingoptimizer.physics.recommend import recommend as _recommend
        return _recommend(
            self, track, env, constraints,
            schedule=schedule, quali=quali, explore_pct=explore_pct,
            reset_mode=reset_mode, surrogate_only=surrogate_only,
        )

    def predict_setup_readouts(
        self,
        setup: dict[str, float],
        env: EnvironmentFrame,
    ) -> dict[str, float]:
        """Predict deterministic setup-readout channels at ``setup``.

        These channels (``setup_static_*_ride_height_mm`` etc.) are
        functions of setup inputs only — perches, pushrods, springs,
        torsion bars. Corner geometry and env have ~0 weight in the
        trained Ridge regressor, but the signature still requires an env
        frame so the assembled feature row matches what the fitter saw at
        training time.

        Returns ``{channel_name: predicted_value}`` for every channel the
        trained model carries with a ``setup_static_`` (or other readout)
        prefix. Picks one fitter per channel (any ``corner_id`` /
        ``phase`` tuple — readouts are corner-independent so any will
        do). Channels missing from the fitter dict are omitted, and an
        unsupported feature schema (legacy v1/v2) returns an empty dict.
        """
        readout_prefixes = ("setup_static_", "setup_heave_slider_")

        by_channel: dict[str, FitRecord] = {}
        for key, record in self.fitters.items():
            channel = key[-1] if isinstance(key, tuple) and key else None
            if not isinstance(channel, str):
                continue
            if not any(channel.startswith(p) for p in readout_prefixes):
                continue
            if not record.fitter.is_trained:
                continue
            by_channel.setdefault(channel, record)

        # P0.2 -- per-car deterministic kinematic fit for the four
        # ``setup_static_*_ride_height_mm`` channels. When the fit
        # shipped (R^2 >= 0.98 per channel) we take its prediction as
        # the source of truth and skip the surrogate path for those
        # channels. The surrogate fit is still computed during training
        # so legacy paths keep functioning; we just stop reading it for
        # the channels the kinematic fit owns.
        out: dict[str, float] = {}
        kinematic = getattr(self, "static_rh_kinematic", None)
        if kinematic is not None:
            from racingoptimizer.physics.static_rh_kinematic import (
                predict_static_rh_kinematic,
            )
            kinematic_readouts = predict_static_rh_kinematic(kinematic, setup)
            out.update(kinematic_readouts)
            for channel in kinematic_readouts:
                by_channel.pop(channel, None)

        if not by_channel:
            return out

        schema = int(self.feature_schema_version)
        if schema < 3:
            # Legacy per-parameter sum models don't carry joint feature
            # rows; the readout was never trained as a single fitter.
            return out

        env_features = _env_to_array(env)
        archetype: dict[str, float] = {}
        for channel, record in by_channel.items():
            if schema >= 4:
                row = _assemble_feature_row_v4(
                    record.feature_names, setup, self.baseline_setup,
                    env_features, archetype,
                )
            else:
                row = _assemble_feature_row(
                    record.feature_names, setup, self.baseline_setup,
                    env_features,
                )
            try:
                mu, _sigma = record.fitter.predict(row.reshape(1, -1))
            except (UntrainedError, ValueError):
                continue
            out[channel] = float(mu[0])

        return out

    def predict(
        self,
        setup: dict[str, float],
        env: EnvironmentFrame,
        corner_phase_key: CornerPhaseKey,
        *,
        corner_archetype: dict[str, float] | None = None,
        track: str | None = None,
        context_features: dict[str, float] | None = None,
    ) -> CornerPhaseStateWithConfidence:
        """Predict per-channel corner-phase state under ``setup`` + ``env``.

        ``track`` (optional) selects the per-track random-intercept
        correction (P2.2). When supplied AND the model carries a fit for
        ``(channel, track)``, the surrogate's prediction is shifted by
        the partial-pooled residual mean and the CI is widened by the
        intercept's posterior std. Defaulting to ``None`` preserves the
        legacy behaviour exactly (no correction, no CI widening).
        """
        if int(self.feature_schema_version) >= 4:
            if corner_archetype is None:
                raise ValueError(
                    "per-car model requires `corner_archetype` (dict of "
                    "apex_speed_ms / peak_lat_g / corner_min/max_speed_ms / "
                    "corner_duration_s) — schedule must come from the target "
                    "TrackModel, not the trained sessions."
                )
            return self._predict_v4(
                setup,
                env,
                corner_phase_key,
                corner_archetype,
                track=track,
                context_features=context_features,
            )
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

    # ---- Stage 4 per-car (track-agnostic) prediction --------------------

    def _predict_v4(
        self,
        setup: dict[str, float],
        env: EnvironmentFrame,
        corner_phase_key: CornerPhaseKey,
        corner_archetype: dict[str, float],
        *,
        track: str | None = None,
        context_features: dict[str, float] | None = None,
    ) -> CornerPhaseStateWithConfidence:
        """Predict for a target corner using the per-car fitters keyed by (phase, channel).

        ``corner_archetype`` carries the geometric descriptors of the TARGET
        corner (apex speed, peak lat-G, max/min speed, duration). It is the
        bridge that lets a Laguna-trained Cadillac fitter score a Spa corner:
        the (phase, channel) fitter's input row is reconstructed in the
        trained order with the target corner's archetype values plugged in.

        ``track`` selects the per-track random-intercept correction
        (P2.2). When None or absent from the fit, the surrogate's
        prediction passes through unchanged.
        """
        phase = (
            corner_phase_key.phase.value
            if isinstance(corner_phase_key.phase, Phase)
            else str(corner_phase_key.phase)
        )

        channels: dict[str, FitRecord] = {}
        for key, record in self.fitters.items():
            if len(key) != 2:
                continue
            ph, channel = key
            if ph != phase:
                continue
            if not record.fitter.is_trained:
                continue
            channels[channel] = record

        env_features = _env_to_array(env)
        from racingoptimizer.physics.aero_fit_features import (
            aero_map_features_for_predict,
        )
        from racingoptimizer.physics.fitter import _load_aero_surface

        static_readouts = self.predict_setup_readouts(setup, env)
        aero_surface = _load_aero_surface(self.car)
        extra_features = aero_map_features_for_predict(
            car=self.car,
            setup=setup,
            aero_surface=aero_surface,
            air_density=float(env.air_density),
            static_readouts=static_readouts,
        )
        if context_features:
            for name, value in context_features.items():
                if value is not None:
                    extra_features[name] = float(value)
        states: dict[str, Confidence] = {}
        untrained: list[str] = []

        for channel in sorted(channels.keys()):
            record = channels[channel]
            x_row = _assemble_feature_row_v4(
                record.feature_names,
                setup,
                self.baseline_setup,
                env_features,
                corner_archetype,
                extra_features=extra_features,
            )
            try:
                mu, sigma = record.fitter.predict(x_row.reshape(1, -1))
            except (UntrainedError, ValueError):
                untrained.append(channel)
                continue
            mean_value = float(mu[0])
            posterior_std = float(sigma[0]) if sigma.size else 0.0

            # Note: ``per_track_residuals`` (added 2026-05-24) was retired
            # in P0.1 of ``docs/accuracy-rebuild-2026-05-24/PLAN.md`` -- it
            # added ``track_median(actual) - global_median(actual)``, not
            # a real residual, which double-counted track bias (the
            # surrogate is already trained on those rows) and flattened
            # the setup -> output gradient that DE needs. The slot is
            # kept on PhysicsModel for pickle compat but no longer read.
            #
            # P2.2 replaces it with a proper random-intercept correction:
            # alpha_t fit by closed-form empirical Bayes on the
            # surrogate's training residuals, partial-pooled toward zero
            # so low-sample tracks shrink to no-correction. Applied
            # additively here -- the setup gradient (d mu / d setup) is
            # unchanged because alpha_t does not depend on setup.
            from racingoptimizer.physics.track_random_intercepts import (
                predict_correction,
            )
            intercept_value, intercept_std = predict_correction(
                getattr(self, "track_random_intercepts", None),
                channel,
                track,
            )
            mean_value = mean_value + intercept_value

            # Compose CI std: surrogate uncertainty (cv_residual or
            # posterior) AND intercept uncertainty add in quadrature
            # because they are independent sources of error.
            surrogate_std = max(float(record.cv_residual_std), posterior_std)
            if intercept_std > 0.0:
                bracket_std = float(
                    (surrogate_std ** 2 + intercept_std ** 2) ** 0.5
                )
            else:
                bracket_std = surrogate_std
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


def _assemble_feature_row_v4(
    feature_names: tuple[str, ...],
    setup: dict[str, float],
    baseline: dict[str, float],
    env_features: np.ndarray,
    corner_archetype: dict[str, float],
    *,
    extra_features: dict[str, float] | None = None,
) -> np.ndarray:
    """Build the per-car (v4) joint feature row.

    Same shape rules as ``_assemble_feature_row`` for setup + env, plus a
    third source: corner archetype features keyed by name (apex_speed_ms,
    peak_lat_g, etc). When a feature_name matches an archetype key, that
    archetype value is used; setup/env/baseline fallbacks remain unchanged.
    """
    from racingoptimizer.physics.fitter import _ENV_COLUMNS

    env_index = {name: idx for idx, name in enumerate(_ENV_COLUMNS)}
    extras = extra_features or {}
    row = np.empty(len(feature_names), dtype=np.float64)
    for i, name in enumerate(feature_names):
        if name in extras and extras[name] is not None:
            row[i] = float(extras[name])
            continue
        if name in corner_archetype and corner_archetype[name] is not None:
            row[i] = float(corner_archetype[name])
            continue
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


def _repair_legacy_slot_shift(slot_values: dict[str, object]) -> None:
    """Fix pickles saved before ``per_track_residuals`` was append-only.

    Pre-v7 pickles serialised a positional slot list whose tail was one
    entry short: ``axle_grip_ceilings`` landed in ``per_track_residuals``,
    ``aero_residual_correction`` landed in ``axle_grip_ceilings``, and the
    real aero correction slot was empty. Detect the mis-typed values and
    restore them in-place so hybrid scoring and guardrails work again
    without forcing an immediate refit on every stale cache file.
    """
    ptr = slot_values.get("per_track_residuals")
    axle = slot_values.get("axle_grip_ceilings")
    aero = slot_values.get("aero_residual_correction")

    if isinstance(axle, AeroResidualCorrection) and aero is None:
        slot_values["aero_residual_correction"] = axle
        slot_values["axle_grip_ceilings"] = None
        axle = None

    if isinstance(ptr, dict) and ptr:
        sample = next(iter(ptr.values()))
        if isinstance(sample, AxleGripCeiling):
            slot_values["axle_grip_ceilings"] = ptr
            slot_values["per_track_residuals"] = {}


# P1.4 -- slot type-safety on pickle revive.
#
# Pre-2026-05-24 pickles surfaced a silent slot-shift bug: somebody
# inserted a field in the middle of ``PhysicsModel.__slots__`` and
# every later slot's positional value shifted by one, so
# ``axle_grip_ceilings`` landed inside ``per_track_residuals`` (a
# ``dict[str, AxleGripCeiling]``) and ``aero_residual_correction``
# landed inside ``axle_grip_ceilings`` (an ``AeroResidualCorrection``).
# Hybrid scoring read ``axle_grip_ceilings.get("front")`` and hit an
# ``AttributeError`` -- except a defensive ``isinstance(..., dict)``
# guard in ``physics/score.py`` swallowed it and silently disabled
# guardrails. No regression test asserted the slot types, so the
# corruption went undetected until a stale pickle was inspected.
#
# This helper runs after ``_repair_legacy_slot_shift`` and refuses to
# revive a pickle whose slot types are still wrong. The error names
# the slot and tells the user how to recover (refit, or run with
# ``--no-cache``). Repair is intentionally NOT attempted here -- if
# the shift is novel, repairing in silence would re-create the
# original "silently broken" failure mode.
_SLOT_EXPECTED_TYPES: dict[str, tuple[type, ...]] = {
    # Always dict/dict-of-dict, never an object instance:
    "parameter_observed_std": (dict,),
    "per_track_parameter_observed": (dict,),
    "bayes_posteriors": (dict,),
    "per_track_residuals": (dict,),
    "track_random_intercepts": (dict,),
    "track_models_used": (dict,),
    "fitters": (dict,),
    "ontology": (dict,),
    "baseline_setup": (dict,),
    # Always tuple:
    "session_ids": (tuple,),
    "untrained_parameters": (tuple,),
    "static_rh_corpus": (tuple,),
    # Always int/bool:
    "feature_schema_version": (int,),
    "seed": (int,),
    "aero_correction_available": (bool, int),
}


def _validate_pickle_slots(slot_values: dict[str, object]) -> None:
    """Type-check pickle slots before they are assigned to a PhysicsModel.

    Raises ``TypeError`` if any slot carries a value whose type cannot
    plausibly belong to that slot. Designed to detect future
    slot-shift accidents (see ``_repair_legacy_slot_shift``) rather
    than silently letting wrong-typed values flow into production
    scoring paths.
    """
    for slot_name, allowed_types in _SLOT_EXPECTED_TYPES.items():
        if slot_name not in slot_values:
            continue
        value = slot_values[slot_name]
        if not isinstance(value, allowed_types):
            raise TypeError(
                f"PhysicsModel pickle revive: slot {slot_name!r} has "
                f"type {type(value).__name__!r} but expected "
                f"{[t.__name__ for t in allowed_types]!r}. This is "
                "almost certainly slot-shift corruption from inserting "
                "a field in the middle of __slots__ (see comment in "
                "model.py near the slot decl). Recover with "
                "`optimize <car> <track> --no-cache` to force a refit, "
                "and bump FITTERS_LAYOUT_VERSION to invalidate other "
                "stale pickles. See docs/accuracy-rebuild-2026-05-24/"
                "PLAN.md P1.4."
            )
    # The two optional-object slots: validate explicitly because their
    # allowed types include None.
    axle = slot_values.get("axle_grip_ceilings")
    if axle is not None and not isinstance(axle, dict):
        raise TypeError(
            f"PhysicsModel pickle revive: slot 'axle_grip_ceilings' has "
            f"type {type(axle).__name__!r} but expected 'dict' or 'NoneType'. "
            "Recover with `optimize <car> <track> --no-cache`."
        )
    aero = slot_values.get("aero_residual_correction")
    if aero is not None and not isinstance(aero, AeroResidualCorrection):
        raise TypeError(
            f"PhysicsModel pickle revive: slot 'aero_residual_correction' "
            f"has type {type(aero).__name__!r} but expected "
            "'AeroResidualCorrection' or 'NoneType'. Recover with "
            "`optimize <car> <track> --no-cache`."
        )
    rh = slot_values.get("static_rh_kinematic")
    if rh is not None:
        from racingoptimizer.physics.static_rh_kinematic import (
            StaticRhKinematic,
        )
        if not isinstance(rh, StaticRhKinematic):
            raise TypeError(
                f"PhysicsModel pickle revive: slot 'static_rh_kinematic' "
                f"has type {type(rh).__name__!r} but expected "
                "'StaticRhKinematic' or 'NoneType'. Recover with "
                "`optimize <car> <track> --no-cache`."
            )


__all__ = [
    "AERO_DEPENDENT_CHANNELS",
    "CornerPhaseStateWithConfidence",
    "FitRecord",
    "PhysicsModel",
]
