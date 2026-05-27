"""Top-level orchestrator: `fit(car, session_ids, track_model)`.

Stage-3 architecture: one multi-input fitter per (car, corner_id, phase,
output_channel). The fitter's feature space is the FULL bounded setup
parameter vector + the 12 env channels. This realises VISION §3 / §5
("every parameter interacts with every other parameter — model the coupled
system, not independent channels"): perturbing any single setup parameter
propagates through every output channel via the trained joint mapping.

Fitter family selection: the joint vector mixes continuous (springs, ride
heights) and discrete (dampers when bounded, ARBs) parameters, so we route
families by what is present. Default to GP unless the joint vector contains
any RF-family parameter, in which case the whole quadruple goes to RF (which
handles continuous + discrete jointly without disastrous extrapolation).

Pickled `feature_schema_version=3` artefacts are produced; older v1 / v2
pickles still load (`PhysicsModel.__setstate__` and `_predict_legacy` keep
the old per-parameter sum-of-fitters semantics alive for them).
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from statistics import median, pstdev

import numpy as np
import polars as pl
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold

from racingoptimizer.constraints import load_constraints
from racingoptimizer.corner import corner_phase_states
from racingoptimizer.ingest import api as ingest_api
from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.paths import catalog_path, resolve_corpus_root
from racingoptimizer.physics.axle_grip import (
    AxleGripCeiling,
    compute_axle_grip_ratios,
    fit_axle_grip_ceiling,
)
from racingoptimizer.physics.baselines import derive_baselines
from racingoptimizer.physics.exceptions import InsufficientDataError
from racingoptimizer.physics.fitters import (
    FitterBase,
    ForestFitter,
    GPFitter,
    RidgeFitter,
)
from racingoptimizer.physics.io_log import append_accuracy_log
from racingoptimizer.physics.model import FitRecord, PhysicsModel
from racingoptimizer.physics.ontology import (
    Family,
    fittable_parameters,
    ontology_for,
    setup_value,
)
from racingoptimizer.physics.static_rh_kinematic import (
    fit_static_rh_kinematic,
)
from racingoptimizer.physics.static_rh_knn import build_static_rh_corpus
from racingoptimizer.track.builder import TrackModel

# Curated set of corner-phase output channels we attempt to fit. Picked from
# `corner_phase_states`'s wide schema; covers grip / brake / steer / handling
# / suspension state.
TARGET_OUTPUT_CHANNELS: tuple[str, ...] = (
    "accel_lat_g_max",
    "accel_lat_g_mean",
    "brake_max",
    "brake_mean",
    "throttle_max",
    "throttle_mean",
    "steering_max_rad",
    "understeer_angle_mean_rad",
    "lf_shock_defl_p99_mm",
    "rf_shock_defl_p99_mm",
    "lr_shock_defl_p99_mm",
    "rr_shock_defl_p99_mm",
    "lf_ride_height_mean_mm",
    "rf_ride_height_mean_mm",
    "lr_ride_height_mean_mm",
    "rr_ride_height_mean_mm",
    # VISION §3 names "shock velocities" as a thing the empirical model
    # must learn as a function of setup. The corner aggregator already
    # produces velocity (mm/s) and force (N) columns at corner-phase
    # grain (see `corner/states.py:504-511`); they belong here so the
    # joint surrogate can predict spring/damper-rate → platform-velocity
    # cause-and-effect for the recommender's setup search.
    "damper_velocity_p99_mms",
    "damper_velocity_mean_mms",
    "damper_force_p99_n",
    "damper_force_mean_n",
    # Traction / longitudinal dynamics (previously only partially surfaced).
    # wheel_speed_max_diff_ms closes the gap where score.traction() fell back
    # to 0.5. accel_lon_g_* replace the long_g=0 approximation used in hybrid
    # guardrails for mid-corner.
    "wheel_speed_max_diff_ms",
    "accel_lon_g_min",
    "accel_lon_g_max",
    # Static ride-height readouts (TRACK-INVARIANT). iRacing's setup
    # YAML stores `Chassis.LeftFront.RideHeight` etc. as the calculated
    # static ride height — a deterministic function of the garage
    # parameters (pushrod offset, spring rate, spring perch offset,
    # static weight, tyre stiffness) at rest. SESSION-INVARIANT.
    # Populated by `_attach_setup_readouts`.
    #
    # We deliberately do NOT include the `TiresAero.AeroCalculator.*`
    # fields (FrontRhAtSpeed, RearRhAtSpeed, DownforceBalance, LD).
    # That panel is a USER-INPUT scratchpad — the driver types in a
    # ride-height pair to see what aero balance / L/D the aero map
    # would produce there. The values are not derived from the setup
    # parameters, so fitting them as setup→y targets just memorises
    # whatever the driver happened to type in across sessions.
    "setup_static_lf_ride_height_mm",
    "setup_static_rf_ride_height_mm",
    "setup_static_lr_ride_height_mm",
    "setup_static_rr_ride_height_mm",
    # Telemetry-derived at-speed ride heights (PER-SESSION, values
    # come from real 60Hz telemetry filtered to high-speed straight-
    # line samples). VISION §3: "fit from what the car actually does."
    # These are the GROUND TRUTH for the at-speed pose — they include
    # damper compression dynamics, real aero loading at the speeds the
    # car achieves, and track-specific straight-line speeds — none of
    # which iRacing's setup-only AeroCalculator captures. Populated by
    # `_attach_dynamic_at_speed`.
    "dynamic_lf_rh_at_speed_mm",
    "dynamic_rf_rh_at_speed_mm",
    "dynamic_lr_rh_at_speed_mm",
    "dynamic_rr_rh_at_speed_mm",
    "dynamic_front_rh_at_speed_mm",
    "dynamic_rear_rh_at_speed_mm",
)


# Setup-readout extraction map. The four `Chassis.<Corner>.RideHeight`
# strings are the static ride heights iRacing computes from the garage
# parameters (pushrod, springs, perch, weight, tyres) — deterministic
# functions of setup. The `TiresAero.AeroCalculator.*` panel is NOT a
# setup readout (it's a user-input scratchpad for testing arbitrary
# (front_rh, rear_rh) pairs against the aero map) — those fields are
# intentionally absent from this list.
_SETUP_READOUT_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("setup_static_lf_ride_height_mm", ("Chassis", "LeftFront", "RideHeight")),
    ("setup_static_rf_ride_height_mm", ("Chassis", "RightFront", "RideHeight")),
    ("setup_static_lr_ride_height_mm", ("Chassis", "LeftRear", "RideHeight")),
    ("setup_static_rr_ride_height_mm", ("Chassis", "RightRear", "RideHeight")),
)

# Spec §5 family routing: continuous low-dim → GP. Discrete/coupled families
# (damper clicks, per-corner-weight balancing) fall through to ForestFitter.
# Brake bias % and differential preload (Nm) are continuous scalars; they
# routed to RF historically because they were CE-gated. With bounds landing
# in `constraints.md` (bf2e48b) they join the GP cohort so the joint
# surrogate stays GP for the full bounded vector.
_GP_FAMILIES: frozenset[Family] = frozenset(
    {"heave_spring", "heave_slider", "tyre_pressure",
     "front_wing", "rear_wing", "ride_height", "arb",
     "brake_bias", "diff",
     "spring_rate", "perch_offset", "pushrod", "camber",
     # Dampers are integer clicks (0..11) but the GP path handles them
     # the same way it handles ARB blade indices — DE searches a
     # continuous range, the post-clamp rounds to the nearest legal
     # integer at briefing render time. Including dampers here keeps the
     # v3 (per-car, per-track) joint vector on GP rather than forcing
     # the whole vector to RF the moment any one damper is fittable.
     "damper",
     # Torsion bars (cadillac front-only). Continuous turns + (treated-
     # as-continuous) OD diameter envelope. Future renderer work can map
     # the recommended OD to the nearest of the 14 legal diameters.
     "torsion_bar",
     "traction_control"}
)

# Per-phase columns the env feature vector pulls. Same 12 fields as
# `EnvironmentFrame` (VISION section 10), in the same order — `_env_to_array`
# in model.py mirrors this.
_ENV_COLUMNS: tuple[str, ...] = (
    # Atmospheric floats:
    "air_temp_c_mean",
    "air_density_mean",
    "air_pressure_mbar_mean",
    "relative_humidity_mean",
    "wind_vel_ms_mean",
    "wind_dir_deg_mean",
    "fog_level_mean",
    # Track surface floats:
    "track_temp_c_mean",
    "track_wetness_mean",
    # Discrete weather state (cast to numeric for the feature vector):
    "weather_declared_wet_max",
    "precip_type_max",
    "skies_max",
)

# Corner-phase rows aggregate per-sample track quality into a clean fraction.
# Dirty sections (curbs/off-track/contact-like events) must not train the same
# as clean phases per VISION §9.
MIN_TRAINING_CLEAN_FRACTION: float = 0.8

# Number of env features in the v1 schema (pre-S2.2). Kept for legacy pickles
# that lived under that schema.
ENV_FEATURE_COUNT_V1: int = 5
# Current env schema version emitted by `fit`.
# v3 = Stage-3 joint multi-input model. One fitter per (corner_id, phase,
#      channel); corner_id is track-local so the model can't transfer to a
#      new track without re-fitting. Kept for backward compat.
# v4 = Per-car model (track-agnostic). Sessions pooled across ALL tracks for
#      the car. One fitter per (phase, channel). Corner archetype features
#      (apex_speed, peak_lat_g, corner_min/max_speed, corner_duration_s) are
#      part of the input feature vector so the same fitter can score any
#      corner on any track once the target's archetypes are extracted.
ENV_FEATURE_SCHEMA_VERSION: int = 3
ENV_FEATURE_SCHEMA_VERSION_PER_CAR: int = 8

# W6 P4 -- driver-input control variables per (corner, phase). Isolates
# setup signal from driver noise on grip-balance channels. Already emitted
# by ``corner.states``; appended to the per-car Forest feature vector.
DRIVER_CONTROL_COLUMNS: tuple[str, ...] = (
    "steering_mean_rad",
    "brake_mean",
    "throttle_mean",
)

# P2.1 cleanliness thresholds. A corner-phase row is dropped from the
# training frame when its mean curb-or-off-track fraction exceeds these
# bounds. Curb threshold is 0.5 (half the phase samples on a persistent
# curb bin -- the median sample lies inside a curb, the plan's primary
# signal). Off-track threshold is stricter (>0 = ANY off-track sample
# observed during the phase) because the off-track detector is already
# dilated and a single trigger flags a meaningful excursion.
_CURB_DIRTY_FRACTION: float = 0.5
_OFF_TRACK_DIRTY_FRACTION: float = 0.0

# Per-corner archetype columns appended to every training row (and required
# at predict time). Computed via polars window aggregation over
# (session_id, corner_id) so every (corner, phase) row carries the same
# corner-level archetype values. Order MUST stay stable — pickled feature
# names embed it.
CORNER_ARCHETYPE_COLUMNS: tuple[str, ...] = (
    "corner_apex_speed_ms",       # min speed observed across the corner
    "corner_peak_lat_g",          # max |lat_g| across the corner
    "corner_max_speed_ms",        # max speed across the corner (entry/exit fastest)
    "corner_min_speed_ms",        # min speed (== apex_speed; kept for symmetry)
    "corner_duration_s",          # total time in corner (sum of phase durations)
    "corner_compression_demand_mms",  # peak damper velocity in compression phases
    # P2.4: phase-level duration. The corner-level keys above all
    # aggregate to (session, corner) so every phase row of the same
    # corner carries identical values; the surrogate cannot learn
    # within-corner phase asymmetry from those alone. Adding the
    # per-phase duration lets the fit pick up "longer braking phase
    # at this corner type means brake bias matters more" type
    # interactions. Bumps ENV_FEATURE_SCHEMA_VERSION_PER_CAR.
    "phase_duration_s",
)


def fit(
    car: str,
    session_ids: list[str],
    track_model: TrackModel,
    *,
    seed: int = 0xC0FFEE,
    k_folds: int = 5,
    corpus_root: Path | str | None = None,
) -> PhysicsModel:
    # NOTE on `track_model`: the parameter is kept for caller compat but the
    # fitter does not read any TrackModel attribute. Data-quality filtering
    # happens via the `data_quality_mask` column on the parquet, which is
    # populated by `racingoptimizer.track.rewrite.apply_quality_mask` --
    # called by `optimize learn` per-session in `ingest.api._apply_masks_for_session_ids`.
    # Pass any TrackModel here (commonly the per-(car, track) model the
    # caller already built); fit() reads the mask off-disk regardless.
    _ = track_model  # vestigial; explicitly retained for caller compat
    car_key = car.strip().lower()
    sorted_ids = sorted(session_ids)
    if not sorted_ids:
        raise InsufficientDataError(f"no session_ids supplied for car={car_key!r}")

    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)

    setups: dict[str, dict] = {}
    track_models_used: dict[str, str] = {}
    with cat.open_catalog(catalog_path(root)) as conn:
        for sid in sorted_ids:
            sess = cat.get_session(conn, sid)
            if sess is None:
                raise ValueError(f"session_id not found in catalog: {sid!r}")
            if sess.car != car_key:
                raise ValueError(
                    f"session {sid!r} car={sess.car!r} does not match requested car={car_key!r}"
                )
            setups[sid] = _decode_setup(sess.setup)
            track_models_used[sid] = sess.track

    constraints = load_constraints()
    onto = ontology_for(car_key)
    fit_params = fittable_parameters(car_key, constraints)
    untrained_params: set[str] = {
        name for name, spec in onto.items()
        if not spec.fittable or constraints.bounds(car_key, name) is None
    }

    # Per-session setup snapshot. setup_value handles missing JSON paths.
    setup_snapshots: dict[str, dict[str, float]] = {}
    for sid in sorted_ids:
        snap: dict[str, float] = {}
        for name in fit_params:
            val = setup_value(car_key, name, setups[sid])
            if val is not None:
                snap[name] = val
        setup_snapshots[sid] = snap

    training = _collect_training_frames(sorted_ids, root)
    if training.height == 0:
        raise InsufficientDataError(
            f"no clean (corner, phase) rows for car={car_key!r} across {len(sorted_ids)} session(s)"
        )

    # Stage 3: drop fittable parameters that have no observed value across
    # any session (so the joint feature vector stays well-defined). Such
    # parameters get listed as untrained.
    available_params: list[str] = []
    for name in fit_params:
        if any(setup_snapshots[sid].get(name) is not None for sid in sorted_ids):
            available_params.append(name)
        else:
            untrained_params.add(name)

    sid_to_param_value: dict[str, dict[str, float]] = {
        sid: dict(setup_snapshots[sid]) for sid in sorted_ids
    }
    sid_to_readouts: dict[str, dict[str, float]] = {
        sid: _extract_setup_readouts(setups[sid]) for sid in sorted_ids
    }
    # Telemetry-derived at-speed RH per session (track-dependent reality
    # vs iRacing's setup-only calculator). VISION §3 — fit from
    # observation. See `physics.dynamic_at_speed`.
    from racingoptimizer.physics.dynamic_at_speed import (
        compute_dynamic_at_speed_rh,
    )
    sid_to_dynamic: dict[str, dict[str, float]] = {
        sid: compute_dynamic_at_speed_rh(sid, corpus_root=root)
        for sid in sorted_ids
    }

    # Build the joint training frame: one row per (session, corner, phase)
    # with every available parameter as its own column. Sessions missing a
    # parameter contribute NaN for that column; rows with any NaN parameter
    # are dropped to keep the joint vector well-defined.
    joint = _attach_setup_columns(training, sid_to_param_value, available_params)
    # Setup readouts (static RH + aero calc) — track-invariant target
    # columns so the per-(corner, phase, readout) fitters learn the clean
    # setup→equilibrium chain without track-archetype confounding.
    joint = _attach_setup_readouts(joint, sid_to_readouts)
    joint = _attach_dynamic_at_speed(joint, sid_to_dynamic)

    # Family routing for the joint fit: any RF-family parameter forces the
    # whole vector to RF (trees handle continuous + discrete jointly);
    # otherwise we use GP.
    family_kind = _joint_family_kind(available_params, onto)

    fitters: dict[tuple, FitRecord] = {}
    rng = np.random.default_rng(seed)
    cv_seed = int(rng.integers(0, 2**31 - 1))

    fit_records_for_log: list[
        tuple[int, str, str, FitRecord]
    ] = []

    if joint.height >= 5:
        for (corner_id, phase), sub in joint.group_by(
            ["corner_id", "phase"], maintain_order=True
        ):
            corner_id_int = int(corner_id)
            phase_str = str(phase)
            for output_channel in TARGET_OUTPUT_CHANNELS:
                if output_channel not in sub.columns:
                    continue
                channel_family = _channel_family_kind(output_channel, family_kind)
                rec = _fit_one_quadruple(
                    sub=sub,
                    parameters=available_params,
                    output_channel=output_channel,
                    family_kind=channel_family,
                    seed=seed,
                    cv_seed=cv_seed,
                    k_folds=k_folds,
                )
                if rec is None:
                    continue
                fitters[(corner_id_int, phase_str, output_channel)] = rec
                fit_records_for_log.append(
                    (corner_id_int, phase_str, output_channel, rec)
                )

    if not fitters:
        raise InsufficientDataError(
            f"no fitters trained for car={car_key!r}; need more sessions on this car"
        )

    aero_surface = _load_aero_surface(car_key)
    aero_available = aero_surface is not None
    aero_residual_correction = _fit_aero_residual_correction(
        car_key, joint, aero_surface=aero_surface,
    )

    # Per-parameter observed values across the requested sessions. Both
    # `baseline_setup` (median) and `parameter_observed_std` (population std)
    # come from the same source of truth so the recommender can detect
    # parameters the driver held effectively constant in training.
    #
    # Why std matters: when a parameter has zero (or near-zero) variation in
    # the training corpus, the joint surrogate has no information about how
    # the response surface depends on it. The DE search then drifts to
    # whichever bound the noise gradient points at — producing absurd
    # constraint-edge recommendations like "tyre pressure 166 kPa" when
    # every observed session ran 152 kPa. The recommender uses this dict
    # to pin such parameters to the observed median (see
    # `physics/recommend.py::_pin_or_trust_bounds`).
    # Lap-time-weighted samples (PLAN.md Day 6, Mode 3): same weighting
    # logic as fit_per_car but track-scoped (v3 trains on one track).
    session_weights = _compute_lap_time_weights(
        sorted_ids, track_per_session=track_models_used, corpus_root=root,
    )

    baseline: dict[str, float] = {}
    parameter_observed_std: dict[str, float] = {}
    for name in fit_params:
        pairs = [
            (setup_snapshots[sid].get(name), session_weights.get(sid, 1.0))
            for sid in sorted_ids
        ]
        pairs = [(float(v), float(w)) for (v, w) in pairs if v is not None]
        if not pairs:
            continue
        values = [v for v, _ in pairs]
        weights = [w for _, w in pairs]
        baseline[name] = _weighted_median(values, weights)
        parameter_observed_std[name] = (
            _weighted_std(values, weights) if len(values) >= 2 else 0.0
        )

    car_baselines = derive_baselines(car_key, training)

    # Per-(car, axle) grip ceilings from training corpus (see
    # `_fit_axle_ceilings_for_car` docstring + `physics/recommend.py
    # _axle_guardrail_penalty`). `None` -> guardrail inactive in DE.
    axle_grip_ceilings = _fit_axle_ceilings_for_car(
        car_key, sorted_ids, root,
    )

    model = PhysicsModel(
        car=car_key,
        session_ids=tuple(sorted_ids),
        track_models_used=track_models_used,
        fitters=fitters,
        ontology=onto,
        constraints=constraints,
        untrained_parameters=tuple(sorted(untrained_params)),
        aero_correction_available=aero_available,
        aero_residual_correction=aero_residual_correction,
        baseline_setup=baseline,
        parameter_observed_std=parameter_observed_std,
        seed=int(seed),
        car_baselines=car_baselines,
        feature_schema_version=ENV_FEATURE_SCHEMA_VERSION,
        axle_grip_ceilings=axle_grip_ceilings,
        static_rh_corpus=build_static_rh_corpus(
            sid_to_param_value, sid_to_readouts,
        ),
        static_rh_kinematic=fit_static_rh_kinematic(
            car_key, sid_to_param_value, sid_to_readouts,
        ),
    )

    # Persist a row per fitter to the corpus accuracy log so `optimize
    # status` can render the calibration trend and TrackCoverage.fit_quality
    # is populated.
    if fit_records_for_log:
        track_label = _dominant_track(track_models_used)
        try:
            append_accuracy_log(
                corpus_root=root,
                car=car_key,
                track=track_label,
                session_ids=sorted_ids,
                records=fit_records_for_log,
            )
        except Exception as exc:
            # Telemetry-side persistence; never block the model build on
            # it, but surface the failure as a warning so disk-full /
            # perms / parquet schema drift can't silently break the
            # `optimize status` trend line.
            warnings.warn(
                f"append_accuracy_log failed for ({car_key}, {track_label}): "
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    return model


# ---- internals -----------------------------------------------------------


def _decode_setup(blob: str | None) -> dict:
    if not blob:
        return {}
    try:
        loaded = json.loads(blob)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _collect_training_frames(session_ids: list[str], root: Path) -> pl.DataFrame:
    """Pull every valid lap's per-(corner, phase) frame and stack them.

    P2.1 wiring: for each unique track represented in ``session_ids``,
    lazily build a ``TrackModel`` (cached) and thread it into
    ``corner_phase_states`` so the per-(corner, phase) rows carry
    ``curb_frac_mean`` and ``off_track_frac_mean``. After concatenation,
    rows where the fraction exceeds the dirty thresholds are dropped
    from the training frame -- the physics fit no longer learns off
    samples where the driver was on a kerb or off-line.

    Cold-start tracks (insufficient sessions to build a real TrackModel)
    skip the masking automatically: the cold-start ``TrackModel`` returns
    zero masks, so every sample is "clean" and no rows get dropped.
    Same fallback for any track-build failure -- the dirty-row filter is
    a quality improvement, not a correctness gate.
    """
    # Lazy TrackModel cache keyed by track slug.
    track_models: dict[str, object | None] = {}
    sid_to_track: dict[str, str | None] = {}

    # First pass: build the (session_id -> track) lookup and group by
    # track so each TrackModel is built once on its full session list.
    sessions_by_track: dict[str, list[str]] = {}
    try:
        with cat.open_catalog(catalog_path(root)) as conn:
            for sid in session_ids:
                sess = cat.get_session(conn, sid)
                if sess is None:
                    sid_to_track[sid] = None
                    continue
                track = sess.track
                sid_to_track[sid] = track
                if track:
                    sessions_by_track.setdefault(track, []).append(sid)
    except Exception:
        # If the catalog walk fails, fall back to the legacy no-mask
        # path; ``track_models`` stays empty and corner_phase_states
        # gets no track_model below.
        sid_to_track = {sid: None for sid in session_ids}

    def _get_tm(track: str | None) -> object | None:
        if not track:
            return None
        if track in track_models:
            return track_models[track]
        try:
            from racingoptimizer.track.builder import build_track_model
            tm: object | None = build_track_model(
                track, sessions_by_track.get(track, []),
                corpus_root=root,
            )
        except Exception:
            tm = None
        track_models[track] = tm
        return tm

    frames: list[pl.DataFrame] = []
    for sid in session_ids:
        laps_df = ingest_api.laps(session_id=sid, valid_only=True, corpus_root=root)
        if laps_df.height == 0:
            continue
        tm = _get_tm(sid_to_track.get(sid))
        for lap_idx in laps_df["lap_index"].to_list():
            try:
                cps = corner_phase_states(
                    sid, int(lap_idx), corpus_root=root, track_model=tm,
                )
            except (KeyError, ValueError, FileNotFoundError):
                continue
            if cps.height == 0:
                continue
            frames.append(cps)
    if not frames:
        return pl.DataFrame()

    stacked = pl.concat(frames, how="diagonal_relaxed")
    # P2.1: drop dirty rows. Columns are absent when no TrackModel was
    # threaded (cold-start corpus, catalog miss, build failure) -- the
    # frame passes through unchanged in that case.
    if "curb_frac_mean" in stacked.columns:
        stacked = stacked.filter(
            pl.col("curb_frac_mean").is_null()
            | (pl.col("curb_frac_mean") <= _CURB_DIRTY_FRACTION)
        )
    if "off_track_frac_mean" in stacked.columns:
        stacked = stacked.filter(
            pl.col("off_track_frac_mean").is_null()
            | (pl.col("off_track_frac_mean") <= _OFF_TRACK_DIRTY_FRACTION)
        )
    return stacked


_READOUT_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_setup_readout(value: object) -> float | None:
    """Parse a setup-blob readout into a float; return None on parse failure.

    iRacing stores most calculated readouts as unit-suffixed strings
    ("30.0 mm", "53.78%") but a few as bare floats ("LD: 3.675"). This helper
    normalises both. Returns ``None`` when the value is missing or
    unparseable so the training frame's null-drop catches it.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = _READOUT_NUMBER_RE.search(value)
        if m is None:
            return None
        try:
            return float(m.group())
        except ValueError:
            return None
    return None


def _extract_setup_readouts(setup: dict) -> dict[str, float]:
    """Return ``{readout_channel: float}`` for every readout we could parse.

    Walks ``_SETUP_READOUT_PATHS`` against the nested setup dict produced by
    `parse_ibt`. Missing intermediate keys or unparseable string values get
    skipped — the resulting partial dict is passed to
    `_attach_setup_readouts`, which broadcasts each value across every row
    of the corresponding session.
    """
    out: dict[str, float] = {}
    for channel, path in _SETUP_READOUT_PATHS:
        cursor: object = setup
        for part in path:
            if not isinstance(cursor, dict):
                cursor = None
                break
            cursor = cursor.get(part)
        parsed = _parse_setup_readout(cursor)
        if parsed is not None:
            out[channel] = parsed
    return out


def _attach_setup_readouts(
    training: pl.DataFrame,
    sid_to_readouts: dict[str, dict[str, float]],
) -> pl.DataFrame:
    """Broadcast per-session setup readouts onto every training row.

    Adds one column per channel in ``_SETUP_READOUT_PATHS``. Sessions
    missing a readout contribute NaN for that column; ``_fit_one_quadruple``
    drops rows with NaN in the target column at fit time.
    """
    if "session_id" not in training.columns or not sid_to_readouts:
        return training
    out = training
    for channel, _path in _SETUP_READOUT_PATHS:
        sid_to_value = {
            sid: r[channel]
            for sid, r in sid_to_readouts.items()
            if channel in r
        }
        if not sid_to_value:
            continue
        out = out.with_columns(
            pl.col("session_id").replace_strict(
                sid_to_value, default=None, return_dtype=pl.Float64
            ).alias(channel)
        )
    return out


def _attach_dynamic_at_speed(
    training: pl.DataFrame,
    sid_to_dynamic: dict[str, dict[str, float]],
) -> pl.DataFrame:
    """Broadcast per-session telemetry-derived at-speed RH onto every row.

    Same broadcast pattern as ``_attach_setup_readouts``. Sessions whose
    high-speed-straight filter yielded too few qualifying samples
    contribute NaN for the dynamic columns and the fitter drops those
    rows at training time.
    """
    from racingoptimizer.physics.dynamic_at_speed import (
        DYNAMIC_AT_SPEED_CHANNELS,
    )

    if "session_id" not in training.columns or not sid_to_dynamic:
        return training
    out = training
    for channel in DYNAMIC_AT_SPEED_CHANNELS:
        sid_to_value = {
            sid: d[channel]
            for sid, d in sid_to_dynamic.items()
            if channel in d
        }
        if not sid_to_value:
            continue
        out = out.with_columns(
            pl.col("session_id").replace_strict(
                sid_to_value, default=None, return_dtype=pl.Float64
            ).alias(channel)
        )
    return out


_TRACK_BALANCE_WEIGHT_COLUMN: str = "_track_balance_weight"


def _attach_track_balance_weights(
    frame: pl.DataFrame,
    *,
    track_per_session: dict[str, str],
) -> pl.DataFrame:
    """Add a ``_track_balance_weight`` column = ``1 / sqrt(n_track_rows)``.

    P2.3: The per-car v4 surrogate pools every session for the car
    across every track. Without re-weighting, a Sebring-heavy corpus
    (37 sessions) drowns Spa rows (11 sessions) at fit time and the
    Forest's split criteria are dominated by Sebring's archetype +
    setup distribution. Multiplying each row's contribution by
    ``1 / sqrt(n_track_rows_in_joint)`` normalises Sebring back to
    ``sqrt(11/37) ~= 0.55`` per-row weight relative to Spa, which is
    enough to let Spa rows shape the fit without flattening Sebring
    completely.

    Rows whose ``session_id`` has no entry in ``track_per_session``
    fall back to weight 1.0. Empty frames return unchanged.
    """
    if frame.height == 0 or "session_id" not in frame.columns:
        return frame
    if not track_per_session:
        return frame.with_columns(
            pl.lit(1.0, dtype=pl.Float64).alias(_TRACK_BALANCE_WEIGHT_COLUMN),
        )
    track_lookup = pl.DataFrame(
        {
            "session_id": list(track_per_session.keys()),
            "_track": list(track_per_session.values()),
        }
    )
    annotated = frame.join(track_lookup, on="session_id", how="left")
    counts = (
        annotated.filter(pl.col("_track").is_not_null())
        .group_by("_track")
        .agg(pl.len().alias("_n"))
    )
    annotated = annotated.join(counts, on="_track", how="left")
    weight_expr = (
        pl.when(pl.col("_n").is_not_null() & (pl.col("_n") > 0))
        .then(1.0 / pl.col("_n").cast(pl.Float64).sqrt())
        .otherwise(1.0)
    )
    annotated = annotated.with_columns(
        weight_expr.cast(pl.Float64).alias(_TRACK_BALANCE_WEIGHT_COLUMN),
    )
    return annotated.drop(["_track", "_n"])


def _fit_track_random_intercepts(
    joint: pl.DataFrame,
    fitters: dict[tuple, FitRecord],
    track_per_session: dict[str, str],
    *,
    min_residual_samples_per_track: int = 10,
) -> dict[tuple[str, str], object]:
    """Fit per-(channel, track) closed-form Bayes random intercepts.

    P2.2 of the accuracy-rebuild plan. For each trained (phase, channel)
    fitter, predict on the training frame, group residuals by track,
    and fit a one-way random-intercept model that partial-pools each
    track's intercept toward zero by empirical-Bayes shrinkage.

    Returns ``{(channel, track): TrackIntercept}``; applied additively
    in ``PhysicsModel._predict_v4`` per channel + target track.

    Honest-residual caveat: residuals are computed in-sample because the
    fitter is already trained. In-sample residuals underestimate true
    out-of-fold residuals, so the magnitude of fitted intercepts is
    biased toward zero -- this is *conservative*. The shrinkage in the
    Bayes posterior compounds the conservatism: when in-sample residuals
    don't separate tracks, ``shrinkage -> 1.0`` and the intercept goes
    to zero (no correction). The recommender consumes the result as an
    *optional* additive shift; a zero intercept is identical to the
    no-correction baseline.
    """
    from racingoptimizer.physics.track_random_intercepts import fit_all_channels

    if joint.height == 0 or not fitters or not track_per_session:
        return {}
    if "session_id" not in joint.columns or "phase" not in joint.columns:
        return {}

    # session_id -> track
    track_lookup = pl.DataFrame(
        {
            "session_id": list(track_per_session.keys()),
            "_track": list(track_per_session.values()),
        }
    )
    annotated = joint.join(track_lookup, on="session_id", how="left")

    per_channel_per_track: dict[str, dict[str, list[float]]] = {}

    for key, record in fitters.items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        phase_str, channel = key
        if channel not in annotated.columns:
            continue
        if not record.fitter.is_trained:
            continue
        feature_names = record.feature_names
        if not feature_names:
            continue

        sub = annotated.filter(
            (pl.col("phase") == str(phase_str))
            & pl.col(channel).is_not_null()
            & pl.col("_track").is_not_null()
        )
        if sub.height < 3:
            continue

        missing = [c for c in feature_names if c not in sub.columns]
        if missing:
            continue
        feats = sub.select(list(feature_names)).cast(pl.Float64).fill_null(0.0).to_numpy()
        try:
            mu, _sigma = record.fitter.predict(feats)
        except (ValueError, np.linalg.LinAlgError):
            continue
        actuals = sub[channel].cast(pl.Float64).to_numpy()
        residuals = (actuals - mu).astype(float)
        tracks = sub["_track"].to_list()

        bucket = per_channel_per_track.setdefault(channel, {})
        for resid, track in zip(residuals, tracks, strict=True):
            if track is None:
                continue
            if not np.isfinite(resid):
                continue
            bucket.setdefault(track, []).append(float(resid))

    # Drop tracks with too few residuals per channel; partial pooling
    # below ``min_residual_samples_per_track`` is dominated by noise and
    # the recommender is better off with no correction (intercept=0).
    pruned: dict[str, dict[str, list[float]]] = {}
    for channel, by_track in per_channel_per_track.items():
        kept = {
            track: vals for track, vals in by_track.items()
            if len(vals) >= min_residual_samples_per_track
        }
        if kept:
            pruned[channel] = kept

    if not pruned:
        return {}
    return dict(fit_all_channels(pruned))


def _attach_corner_archetypes(frame: pl.DataFrame) -> pl.DataFrame:
    """Add per-(session, corner) archetype columns to the training frame.

    Each `(corner_id, phase)` row gets the same archetype values, broadcast
    over the phases of the corner via a polars window aggregation. Archetype
    values describe the corner's geometric demands — apex speed, peak
    lateral G, max/min speed across the corner, total duration. They are
    used as input features so the per-car fitter can transfer learning
    across geometrically-similar corners on different tracks (VISION §3 / §6:
    "be able to generate optimal setups for any track from just the track
    model and aero maps").

    Required columns on `frame`: ``session_id``, ``corner_id``,
    ``speed_min_ms``, ``speed_max_ms``, ``accel_lat_g_max``, ``t_start_s``,
    ``t_end_s``. Missing columns produce a no-op (returns frame unchanged).
    """
    if frame.height == 0:
        return frame
    required = {
        "session_id", "corner_id", "speed_min_ms", "speed_max_ms",
        "accel_lat_g_max", "t_start_s", "t_end_s",
    }
    if not required.issubset(set(frame.columns)):
        return frame
    duration_per_phase = pl.col("t_end_s") - pl.col("t_start_s")
    compression_phases = {"braking", "trail_brake", "mid_corner"}
    cols = [
        # P2.4: phase-level duration carried into the joint feature
        # vector so the surrogate can learn within-corner phase
        # asymmetry (e.g. extended braking phase changes brake bias
        # sensitivity in the fit).
        duration_per_phase.cast(pl.Float64).alias("phase_duration_s"),
        pl.col("speed_min_ms")
        .min()
        .over(["session_id", "corner_id"])
        .cast(pl.Float64)
        .alias("corner_apex_speed_ms"),
        pl.col("accel_lat_g_max")
        .max()
        .over(["session_id", "corner_id"])
        .cast(pl.Float64)
        .alias("corner_peak_lat_g"),
        pl.col("speed_max_ms")
        .max()
        .over(["session_id", "corner_id"])
        .cast(pl.Float64)
        .alias("corner_max_speed_ms"),
        pl.col("speed_min_ms")
        .min()
        .over(["session_id", "corner_id"])
        .cast(pl.Float64)
        .alias("corner_min_speed_ms"),
        duration_per_phase
        .sum()
        .over(["session_id", "corner_id"])
        .cast(pl.Float64)
        .alias("corner_duration_s"),
    ]
    if "damper_velocity_p99_mms" in frame.columns:
        cols.append(
            pl.when(pl.col("phase").is_in(list(compression_phases)))
            .then(pl.col("damper_velocity_p99_mms"))
            .otherwise(None)
            .max()
            .over(["session_id", "corner_id"])
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias("corner_compression_demand_mms"),
        )
    else:
        cols.append(
            pl.lit(0.0).cast(pl.Float64).alias("corner_compression_demand_mms"),
        )
    return frame.with_columns(cols)


def _fit_axle_ceilings_for_car(
    car: str,
    session_ids: list[str],
    root: Path,
    *,
    max_laps_per_session: int = 3,
    mid_corner_threshold_g: float = 0.5,
) -> dict[str, AxleGripCeiling] | None:
    """Fit per-(car, axle) grip ceilings from the training corpus.

    Pulls up to `max_laps_per_session` valid laps per session, extracts
    `LatAccel` and `LongAccel` (60Hz channels in the IBT), pools across
    all sessions for the car, filters to mid-corner samples (|lat_g| >=
    `mid_corner_threshold_g`), then fits per-(car, axle) `mu_peak`
    via `physics.axle_grip.fit_axle_grip_ceiling`.

    Returns `None` on any failure path -- no `LatAccel`/`LongAccel`
    channels, insufficient mid-corner samples (< 100 after filter),
    or `fit_axle_grip_ceiling` raises (mu outside physical range, etc.).
    The recommender treats `None` as "guardrail inactive", so a
    failed ceiling fit silently degrades to surrogate-only scoring
    rather than crashing the fit.

    Why a separate corpus scan vs reusing `_collect_training_frames`:
    that helper aggregates 60Hz samples into per-(corner, phase) rows
    (mean/max/p99) and drops the raw time series. The axle ceiling
    needs raw samples to fit a percentile-anchored peak, so we scan
    a few laps per session directly.
    """
    lat_samples: list[np.ndarray] = []
    lon_samples: list[np.ndarray] = []
    for sid in session_ids:
        try:
            laps_df = ingest_api.laps(
                session_id=sid, valid_only=True, corpus_root=root,
            )
        except Exception:
            continue
        if laps_df.height == 0:
            continue
        for lap_pos in range(min(max_laps_per_session, laps_df.height)):
            try:
                df = ingest_api.lap_data(
                    session_id=sid,
                    lap_index=int(laps_df["lap_index"][lap_pos]),
                    corpus_root=root,
                )
            except Exception:
                continue
            if "LatAccel" not in df.columns or "LongAccel" not in df.columns:
                continue
            lat_samples.append(
                df["LatAccel"].to_numpy().astype(np.float64) / 9.81
            )
            lon_samples.append(
                df["LongAccel"].to_numpy().astype(np.float64) / 9.81
            )
    if not lat_samples:
        return None
    lat_arr = np.concatenate(lat_samples)
    lon_arr = np.concatenate(lon_samples)
    mid_mask = np.abs(lat_arr) >= mid_corner_threshold_g
    if int(np.sum(mid_mask)) < 100:
        return None
    try:
        ratios = compute_axle_grip_ratios(
            lat_arr[mid_mask], lon_arr[mid_mask], car,
        )
        front_ceiling = fit_axle_grip_ceiling(car, "front", ratios["front"])
        rear_ceiling = fit_axle_grip_ceiling(car, "rear", ratios["rear"])
    except (ValueError, KeyError):
        return None
    return {"front": front_ceiling, "rear": rear_ceiling}


def fit_per_car(
    car: str,
    session_ids: list[str],
    *,
    seed: int = 0xC0FFEE,
    k_folds: int = 5,
    corpus_root: Path | str | None = None,
) -> PhysicsModel:
    """Train a track-agnostic per-car PhysicsModel.

    Pools every session in ``session_ids`` regardless of which track they
    were recorded on, computes corner archetype features per
    ``(session, corner_id)``, and trains one joint fitter per
    ``(phase, output_channel)`` over the FULL feature vector
    ``[setup, env, archetype]``. The resulting model can score ANY track:
    the caller passes the target track's per-corner archetype values at
    predict time and the same fitter is queried for every corner.

    This realises VISION §3 ("Build an empirical physics model for each car
    from the measured data") and VISION §6 ("be able to generate optimal
    setups for any track from just the track model and aero maps") — neither
    of which are satisfied by the per-(car, track) ``fit`` API that this
    function complements.
    """
    car_key = car.strip().lower()
    sorted_ids = sorted(session_ids)
    if not sorted_ids:
        raise InsufficientDataError(f"no session_ids supplied for car={car_key!r}")

    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)

    setups: dict[str, dict] = {}
    track_models_used: dict[str, str] = {}
    with cat.open_catalog(catalog_path(root)) as conn:
        for sid in sorted_ids:
            sess = cat.get_session(conn, sid)
            if sess is None:
                raise ValueError(f"session_id not found in catalog: {sid!r}")
            if sess.car != car_key:
                raise ValueError(
                    f"session {sid!r} car={sess.car!r} does not match requested car={car_key!r}"
                )
            setups[sid] = _decode_setup(sess.setup)
            track_models_used[sid] = sess.track

    constraints = load_constraints()
    onto = ontology_for(car_key)
    fit_params = fittable_parameters(car_key, constraints)
    untrained_params: set[str] = {
        name for name, spec in onto.items()
        if not spec.fittable or constraints.bounds(car_key, name) is None
    }

    setup_snapshots: dict[str, dict[str, float]] = {}
    for sid in sorted_ids:
        snap: dict[str, float] = {}
        for name in fit_params:
            val = setup_value(car_key, name, setups[sid])
            if val is not None:
                snap[name] = val
        setup_snapshots[sid] = snap

    training = _collect_training_frames(sorted_ids, root)
    if training.height == 0:
        raise InsufficientDataError(
            f"no clean (corner, phase) rows for car={car_key!r} across {len(sorted_ids)} session(s)"
        )

    available_params: list[str] = []
    for name in fit_params:
        if any(setup_snapshots[sid].get(name) is not None for sid in sorted_ids):
            available_params.append(name)
        else:
            untrained_params.add(name)

    sid_to_param_value: dict[str, dict[str, float]] = {
        sid: dict(setup_snapshots[sid]) for sid in sorted_ids
    }
    sid_to_readouts: dict[str, dict[str, float]] = {
        sid: _extract_setup_readouts(setups[sid]) for sid in sorted_ids
    }
    # Telemetry-derived at-speed RH per session — VISION §3 directs fits
    # to observed reality, not iRacing's setup-only calculator.
    from racingoptimizer.physics.dynamic_at_speed import (
        compute_dynamic_at_speed_rh,
    )
    sid_to_dynamic: dict[str, dict[str, float]] = {
        sid: compute_dynamic_at_speed_rh(sid, corpus_root=root)
        for sid in sorted_ids
    }

    joint = _attach_setup_columns(training, sid_to_param_value, available_params)
    joint = _attach_corner_archetypes(joint)
    aero_surface = _load_aero_surface(car_key)
    from racingoptimizer.physics.aero_fit_features import attach_aero_map_features

    joint = attach_aero_map_features(
        joint, car_key, setups, aero_surface,
    )
    # Setup-readout target columns (static RH + aero calc) — track-invariant
    # so the fitters learn the clean setup→equilibrium chain. Particularly
    # important for the per-car path because cross-track pooling otherwise
    # buries the setup signal under archetype variance.
    joint = _attach_setup_readouts(joint, sid_to_readouts)
    joint = _attach_dynamic_at_speed(joint, sid_to_dynamic)
    # P2.3 -- inverse-track-sample-count training weights. Per-track row
    # counts in the joint frame can vary 5x+ across cars (e.g. BMW
    # Sebring 37 sessions vs Spa 11). Without re-weighting, the Forest's
    # split criteria are dominated by the over-represented track and
    # under-sampled tracks inherit their setup philosophy regardless of
    # archetype features. Weighting rows by 1/sqrt(n_track_rows) gently
    # rebalances without zeroing out any track's contribution. Track is
    # joined in via session_id; rows missing a track mapping default to
    # weight 1.0 (defensive).
    joint = _attach_track_balance_weights(
        joint, track_per_session=track_models_used,
    )

    # Per-car (v4) ALWAYS uses Forest. Justification: with sessions pooled
    # across tracks, the joint feature vector is ~35 dims of mixed-scale
    # (setup ranges in 0..550, env in 0..1000, archetype in 5..100, plus
    # near-constants like tyre_pressure pinned at 152). A scalar-length-scale
    # GP collapses onto whichever feature has the most variance and treats
    # everything else as noise — that's how the v4 fits ended up returning
    # CONSTANT predicted ride heights regardless of heave_spring (verified
    # via probe). Trees split per feature, so they retain "stiffer heave →
    # higher predicted RH" even when corner archetype features dominate
    # variance. Anisotropic GP (one length scale per feature) would also
    # work but the 35-dim hyperparameter optimization runs >30 min per
    # fitter, which is impractical for the 100+ (phase, channel) fitters
    # the per-car path produces.
    family_kind = "rf"

    fitters: dict[tuple, FitRecord] = {}
    rng = np.random.default_rng(seed)
    cv_seed = int(rng.integers(0, 2**31 - 1))
    fit_records_for_log: list[tuple[int, str, str, FitRecord]] = []

    if joint.height >= 5:
        for (phase,), sub in joint.group_by(["phase"], maintain_order=True):
            phase_str = str(phase)
            for output_channel in TARGET_OUTPUT_CHANNELS:
                if output_channel not in sub.columns:
                    continue
                channel_family = _channel_family_kind(output_channel, family_kind)
                # Setup-readout targets are session-invariant, so they
                # don't need archetype features in their input vector
                # (and including them would let the Forest spuriously
                # find archetype-correlated noise). Drop archetype
                # columns for ridge-routed channels.
                arch_cols = (
                    () if channel_family == "ridge"
                    else CORNER_ARCHETYPE_COLUMNS
                )
                from racingoptimizer.physics.aero_fit_features import (
                    aero_fit_column_names,
                )

                augment_cols = (
                    ()
                    if channel_family == "ridge"
                    else DRIVER_CONTROL_COLUMNS + aero_fit_column_names()
                )
                rec = _fit_one_quadruple(
                    sub=sub,
                    parameters=available_params,
                    output_channel=output_channel,
                    family_kind=channel_family,
                    seed=seed,
                    cv_seed=cv_seed,
                    k_folds=k_folds,
                    archetype_columns=arch_cols,
                    augment_columns=augment_cols,
                    weight_column=_TRACK_BALANCE_WEIGHT_COLUMN,
                )
                if rec is None:
                    continue
                # Per-car keying is (phase, channel) — no corner_id.
                fitters[(phase_str, output_channel)] = rec
                # Log under sentinel corner_id=-1 so the accuracy log can
                # distinguish per-car rows from per-(car,track) rows.
                fit_records_for_log.append((-1, phase_str, output_channel, rec))

    if not fitters:
        raise InsufficientDataError(
            f"no fitters trained for car={car_key!r}; "
            f"need more sessions on this car"
        )

    aero_available = aero_surface is not None
    aero_residual_correction = _fit_aero_residual_correction(
        car_key, joint, aero_surface=aero_surface,
    )

    # Lap-time-weighted samples (PLAN.md Day 6, Mode 3): bias the
    # per-session contribution toward fast laps so a conservative
    # recent stint doesn't drag the baseline conservative. Weight per
    # session = 1 / (session_best_lap - track_min_in_corpus + 0.5s).
    # Sessions with no valid laps fall back to weight 1.0 (defensive).
    # See `_compute_lap_time_weights` for the formula and rationale.
    session_weights = _compute_lap_time_weights(
        sorted_ids, track_per_session=track_models_used, corpus_root=root,
    )

    baseline: dict[str, float] = {}
    parameter_observed_std: dict[str, float] = {}
    for name in fit_params:
        pairs = [
            (setup_snapshots[sid].get(name), session_weights.get(sid, 1.0))
            for sid in sorted_ids
        ]
        pairs = [(float(v), float(w)) for (v, w) in pairs if v is not None]
        if not pairs:
            continue
        values = [v for v, _ in pairs]
        weights = [w for _, w in pairs]
        baseline[name] = _weighted_median(values, weights)
        parameter_observed_std[name] = (
            _weighted_std(values, weights) if len(values) >= 2 else 0.0
        )

    car_baselines = derive_baselines(car_key, training)

    # VISION §6: "be able to generate optimal setups for any track from just
    # the track model and aero maps." The per-car model pools sessions
    # across tracks, but the recommender must still respect the empirical
    # envelope the driver has explored ON THE TARGET TRACK — otherwise the
    # joint surrogate's response surface lets the optimizer drift into
    # values the driver has never tried (and the model can't honestly
    # predict). We collect per-(track, parameter) observed sets here and
    # the recommender caps its trust radius using them.
    per_track_observed: dict[str, dict[str, list[float]]] = {}
    for sid in sorted_ids:
        track = track_models_used.get(sid)
        if not track:
            continue
        snap = setup_snapshots.get(sid, {})
        if not snap:
            continue
        per_param = per_track_observed.setdefault(track, {})
        for name, value in snap.items():
            if value is None:
                continue
            per_param.setdefault(name, []).append(float(value))
    per_track_observed_frozen: dict[str, dict[str, tuple[float, ...]]] = {
        track: {
            param: tuple(sorted(set(values)))
            for param, values in by_param.items()
        }
        for track, by_param in per_track_observed.items()
    }

    # Hierarchical-Bayesian retrofit (PLAN.md Day 4, Mode 1). Per
    # parameter, fit a one-way random-intercept Gaussian model across
    # tracks and store per-(parameter, track) posteriors on the model.
    # The recommender prefers the posterior mean over baseline_setup
    # for the trust-radius anchor when a posterior is available -- this
    # is what closes Mode 1 cross-track confounding (a low-sample-count
    # track no longer inherits a high-sample-count track's median).
    # The retrofit operates on the SAME per-track observed dict the
    # surrogate's trust radius already consumes; no new data ingest.
    from racingoptimizer.physics.bayes_retrofit import fit_all_parameters
    bayes_posteriors = fit_all_parameters(per_track_observed_frozen)

    # ``per_track_residuals`` was retired in P0.1 of
    # ``docs/accuracy-rebuild-2026-05-24/PLAN.md``. The earlier
    # implementation computed ``track_median(actual) - global_median(actual)``
    # and added it to every prediction at that track. That is not a
    # residual: the surrogate is already trained on those rows, so the
    # add-on double-counts the track bias and flattens the setup -> output
    # gradient DE needs. Proximate cause of ``+1 click +/-0.000 score``
    # in recent briefings. The slot is kept empty for pickle compat.
    per_track_residuals: dict[str, dict[str, float]] = {}

    # P2.2 -- closed-form Bayes random intercepts on the surrogate's
    # training residuals, partial-pooled toward zero. Applied additively
    # in ``_predict_v4`` to the surrogate mu (setup gradient is preserved
    # because alpha_t does not depend on setup). Failure is graceful:
    # empty dict when no track has enough residual samples; the recommender
    # falls back to surrogate-only via ``predict_correction``'s (0, 0).
    try:
        track_random_intercepts = _fit_track_random_intercepts(
            joint, fitters, track_models_used,
        )
    except Exception as exc:  # noqa: BLE001 -- defensive: fit is optional
        warnings.warn(
            f"track_random_intercepts fit failed for {car_key}: "
            f"{type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        track_random_intercepts = {}

    # Per-(car, axle) grip ceilings from training corpus
    # (physics-rebuild Day-10 model, wired into DE in post-rebuild
    # work). Returns `None` on any failure path -- the recommender
    # treats `None` as "guardrail inactive" so a failed ceiling fit
    # degrades to pure-surrogate scoring rather than crashing the
    # fit. Acura/Porsche are likely to land here on this corpus
    # until they accumulate more mid-corner samples.
    axle_grip_ceilings = _fit_axle_ceilings_for_car(
        car_key, sorted_ids, root,
    )

    model = PhysicsModel(
        car=car_key,
        session_ids=tuple(sorted_ids),
        track_models_used=track_models_used,
        fitters=fitters,
        ontology=onto,
        constraints=constraints,
        untrained_parameters=tuple(sorted(untrained_params)),
        aero_correction_available=aero_available,
        aero_residual_correction=aero_residual_correction,
        baseline_setup=baseline,
        parameter_observed_std=parameter_observed_std,
        seed=int(seed),
        car_baselines=car_baselines,
        feature_schema_version=ENV_FEATURE_SCHEMA_VERSION_PER_CAR,
        per_track_parameter_observed=per_track_observed_frozen,
        bayes_posteriors=bayes_posteriors,
        per_track_residuals=per_track_residuals,
        axle_grip_ceilings=axle_grip_ceilings,
        static_rh_corpus=build_static_rh_corpus(
            sid_to_param_value, sid_to_readouts,
        ),
        static_rh_kinematic=fit_static_rh_kinematic(
            car_key, sid_to_param_value, sid_to_readouts,
        ),
        track_random_intercepts=track_random_intercepts,
    )

    if fit_records_for_log:
        # Per-car log uses a sentinel "<per-car>" track label so `optimize
        # status` can show pooled fit quality alongside per-(car, track)
        # entries without colliding.
        try:
            append_accuracy_log(
                corpus_root=root,
                car=car_key,
                track="<per-car>",
                session_ids=sorted_ids,
                records=fit_records_for_log,
            )
        except Exception as exc:
            warnings.warn(
                f"append_accuracy_log failed for per-car {car_key}: "
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    return model


def _attach_setup_columns(
    training: pl.DataFrame,
    sid_to_param_value: dict[str, dict[str, float]],
    parameters: list[str],
) -> pl.DataFrame:
    """Broadcast the per-session setup values onto every training row.

    For each parameter, build a session_id -> value map and replace into a
    new column with that name. Sessions missing the parameter contribute
    NaN; downstream `_fit_one_quadruple` drops those rows.
    """
    if "session_id" not in training.columns or not parameters:
        return training
    out = training
    for name in parameters:
        sid_to_value = {
            sid: snap[name]
            for sid, snap in sid_to_param_value.items()
            if name in snap and snap[name] is not None
        }
        if not sid_to_value:
            continue
        out = out.with_columns(
            pl.col("session_id").replace_strict(
                sid_to_value, default=None, return_dtype=pl.Float64
            ).alias(name)
        )
    return out


def _joint_family_kind(
    parameters: list[str], onto: dict
) -> str:
    """Select the joint fitter family for the (corner, phase, channel) fit.

    Trees handle a mixed continuous + discrete vector without disastrous
    extrapolation, so any RF-family parameter (damper / corner_weight /
    brake_bias / diff) routes the whole quadruple to ForestFitter. Pure
    GP families (springs / wings / ride heights / tyre pressures / ARBs)
    keep the GP path.
    """
    for name in parameters:
        spec = onto.get(name)
        if spec is None:
            continue
        if spec.family not in _GP_FAMILIES:
            return "rf"
    return "gp"


def _fit_one_quadruple(
    *,
    sub: pl.DataFrame,
    parameters: list[str],
    output_channel: str,
    family_kind: str,
    seed: int,
    cv_seed: int,
    k_folds: int,
    archetype_columns: tuple[str, ...] = (),
    augment_columns: tuple[str, ...] = (),
    weight_column: str | None = None,
) -> FitRecord | None:
    """Fit one (corner_id, phase, output_channel) quadruple over the joint vector.

    When ``archetype_columns`` is non-empty (per-car schema v4), those columns
    are appended to the input feature vector AFTER the env block, in the
    given order. The pickled ``feature_names`` records this so predict can
    rebuild the row in the same shape.

    ``weight_column`` (P2.3): when set and present in ``sub``, the per-row
    values are passed as ``sample_weight`` to the underlying fitter. Forest
    honours it natively; ridge/GP silently ignore. Used to apply
    inverse-track-sample-count weighting so a Sebring-heavy corpus doesn't
    drown Spa rows.
    """
    if "data_quality_clean_frac" in sub.columns:
        sub = sub.filter(pl.col("data_quality_clean_frac") >= MIN_TRAINING_CLEAN_FRACTION)
    drop_cols = (
        [output_channel]
        + [p for p in parameters if p in sub.columns]
        + [a for a in archetype_columns if a in sub.columns]
        + [a for a in augment_columns if a in sub.columns]
    )
    cleaned = sub.drop_nulls(drop_cols)
    if cleaned.height < 3:
        return None

    y = cleaned[output_channel].cast(pl.Float64).to_numpy()
    if y.std() == 0.0:
        return None

    # Param block: one column per available parameter, in the input order.
    param_cols = [p for p in parameters if p in cleaned.columns]
    if param_cols:
        param_block = (
            cleaned.select(param_cols).cast(pl.Float64).fill_null(0.0).to_numpy()
        )
    else:
        param_block = np.zeros((cleaned.height, 0), dtype=np.float64)

    env_cols = [c for c in _ENV_COLUMNS if c in cleaned.columns]
    env_block = (
        cleaned.select(env_cols).cast(pl.Float64).fill_null(0.0).to_numpy()
        if env_cols else np.zeros((cleaned.height, 0), dtype=np.float64)
    )
    # Pad missing env columns with zeros to keep the env vector at 12.
    if env_block.shape[1] < len(_ENV_COLUMNS):
        env_padded = np.zeros((env_block.shape[0], len(_ENV_COLUMNS)), dtype=np.float64)
        for src_idx, col in enumerate(env_cols):
            dst_idx = _ENV_COLUMNS.index(col)
            env_padded[:, dst_idx] = env_block[:, src_idx]
        env_block = env_padded
        env_cols = list(_ENV_COLUMNS)
    elif env_cols != list(_ENV_COLUMNS):
        # Reorder so the trained vector matches `_ENV_COLUMNS` order.
        index_map = {c: i for i, c in enumerate(env_cols)}
        env_block = env_block[:, [index_map[c] for c in _ENV_COLUMNS]]
        env_cols = list(_ENV_COLUMNS)

    arch_cols = [a for a in archetype_columns if a in cleaned.columns]
    if arch_cols:
        arch_block = (
            cleaned.select(arch_cols).cast(pl.Float64).fill_null(0.0).to_numpy()
        )
    else:
        arch_block = np.zeros((cleaned.height, 0), dtype=np.float64)

    aug_cols = [a for a in augment_columns if a in cleaned.columns]
    if aug_cols:
        aug_block = (
            cleaned.select(aug_cols).cast(pl.Float64).fill_null(0.0).to_numpy()
        )
    else:
        aug_block = np.zeros((cleaned.height, 0), dtype=np.float64)

    X = np.concatenate([param_block, env_block, arch_block, aug_block], axis=1)
    feature_names: tuple[str, ...] = (
        tuple(param_cols) + tuple(env_cols) + tuple(arch_cols) + tuple(aug_cols)
    )

    cv_residual_std = _kfold_residual_std(
        X=X, y=y, family_kind=family_kind, k_folds=k_folds,
        seed=seed, cv_seed=cv_seed,
    )

    sample_weight: np.ndarray | None = None
    if weight_column is not None and weight_column in cleaned.columns:
        weights_col = (
            cleaned.select(weight_column).cast(pl.Float64).fill_null(1.0).to_numpy()
        )
        sample_weight = weights_col.reshape(-1)
        if sample_weight.shape[0] != X.shape[0]:
            sample_weight = None

    fitter = _make_fitter(family_kind, seed=seed)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            fitter.fit(X, y, sample_weight=sample_weight)
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        return None
    if not fitter.is_trained:
        return None

    bootstrap_std = 0.0
    from racingoptimizer.physics.fitters.forest import ForestFitter
    if isinstance(fitter, ForestFitter):
        try:
            _, tree_std = fitter.predict(X)
            bootstrap_std = float(np.median(tree_std))
        except (ValueError, np.linalg.LinAlgError):
            bootstrap_std = 0.0

    return FitRecord(
        fitter=fitter,
        n_samples=int(cleaned.height),
        cv_residual_std=float(cv_residual_std),
        signal_std=float(y.std(ddof=0)),
        feature_names=feature_names,
        bootstrap_std=bootstrap_std,
    )


def _make_fitter(family_kind: str, *, seed: int) -> FitterBase:
    if family_kind == "gp":
        return GPFitter(random_state=seed)
    if family_kind == "ridge":
        return RidgeFitter(random_state=seed)
    return ForestFitter(random_state=seed)


# Channel name prefixes that route to RidgeFitter (deterministic setup
# readouts — see `RidgeFitter` docstring). The `dynamic_*` family is
# session-invariant (one value per session) so the same low-data linear
# fit applies even though the underlying source is telemetry rather than
# iRacing's calculator.
_RIDGE_CHANNEL_PREFIXES: tuple[str, ...] = (
    "setup_static_",
    "setup_aero_",
    "dynamic_",
)


def _channel_family_kind(channel: str, default: str) -> str:
    """Override `default` when `channel` is a setup-readout target.

    Setup readouts (static_*, aero_*) get Ridge regardless of the joint
    vector's discrete/continuous mix because the readout itself is a
    deterministic linear-ish function of the bounded setup parameters and
    Ridge in low-data regime fits the chain far better than Forest.
    """
    if any(channel.startswith(p) for p in _RIDGE_CHANNEL_PREFIXES):
        return "ridge"
    return default


def _kfold_residual_std(
    *,
    X: np.ndarray,
    y: np.ndarray,
    family_kind: str,
    k_folds: int,
    seed: int,
    cv_seed: int,
) -> float:
    n = X.shape[0]
    k = max(min(k_folds, n), 2) if n >= 2 else 1
    if k < 2:
        return float(y.std(ddof=0))
    kf = KFold(n_splits=k, shuffle=True, random_state=cv_seed)
    residuals: list[np.ndarray] = []
    for train_idx, test_idx in kf.split(X):
        if len(train_idx) < 2 or len(test_idx) < 1:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                fold_fitter = _make_fitter(family_kind, seed=seed)
                fold_fitter.fit(X[train_idx], y[train_idx])
                if not fold_fitter.is_trained:
                    continue
                pred, _ = fold_fitter.predict(X[test_idx])
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            continue
        residuals.append(np.asarray(pred) - y[test_idx])
    if not residuals:
        return float(y.std(ddof=0))
    return float(np.std(np.concatenate(residuals), ddof=0))


_AIR_DENSITY_REF: float = 1.225


def _load_aero_surface(car: str):
    """Best-effort aero-map loader for fit-time correction wiring."""
    try:
        from racingoptimizer.aero import AeroLoadError, load_aero_maps
    except ImportError:
        return None
    try:
        return load_aero_maps(car)
    except (FileNotFoundError, AeroLoadError, ImportError):
        return None
    except Exception:  # pragma: no cover — defensive against future load errors
        return None


def _fit_aero_residual_correction(
    car: str,
    frame: pl.DataFrame,
    *,
    aero_surface,
):
    """Fit per-car Day-11 residual correction from mid-corner rows."""
    if aero_surface is None:
        return None
    required = {
        "phase",
        "accel_lat_g_max",
        "aero_platform_front_rh_mean_mm",
        "aero_platform_rear_rh_mean_mm",
    }
    if not required.issubset(set(frame.columns)):
        return None
    speed_col = "corner_apex_speed_ms"
    if speed_col not in frame.columns:
        speed_col = "speed_min_ms"
    if speed_col not in frame.columns:
        return None
    cols = list(required)
    cols.append(speed_col)
    has_wing_col = "rear_wing_angle_deg" in frame.columns
    if has_wing_col:
        cols.append("rear_wing_angle_deg")
    sample_frame = (
        frame.filter(pl.col("phase") == "mid_corner")
        .select(cols)
        .drop_nulls([
            "accel_lat_g_max",
            speed_col,
            "aero_platform_front_rh_mean_mm",
            "aero_platform_rear_rh_mean_mm",
        ])
    )
    if sample_frame.height < 50:
        return None

    from racingoptimizer.aero.residual_correction import fit_residual_correction

    default_wing = float(
        aero_surface.bounds.wing_angles[len(aero_surface.bounds.wing_angles) // 2]
    )
    samples: list[dict[str, float]] = []
    for row in sample_frame.to_dicts():
        try:
            speed_ms = float(row[speed_col])
            observed_lat = float(row["accel_lat_g_max"])
            front_rh = float(row["aero_platform_front_rh_mean_mm"])
            rear_rh = float(row["aero_platform_rear_rh_mean_mm"])
            if not (
                np.isfinite(speed_ms)
                and np.isfinite(observed_lat)
                and np.isfinite(front_rh)
                and np.isfinite(rear_rh)
            ):
                continue
            if speed_ms <= 0.0:
                continue
            wing_deg = (
                float(row["rear_wing_angle_deg"])
                if has_wing_col and row.get("rear_wing_angle_deg") is not None
                else default_wing
            )
            _balance, ld_ratio = aero_surface.interpolate(
                front_rh, rear_rh, wing_deg, _AIR_DENSITY_REF,
            )
            if not np.isfinite(ld_ratio) or ld_ratio <= 0.0:
                continue
        except (TypeError, ValueError):
            continue
        samples.append(
            {
                "ld_ratio": float(ld_ratio),
                "speed_ms": speed_ms,
                "observed_lat_g": observed_lat,
            }
        )
    if len(samples) < 50:
        return None
    try:
        return fit_residual_correction(car, samples)
    except ValueError:
        return None


def _try_load_aero(car: str) -> bool:
    """Per spec §9: detect whether slice C is reachable for `car`."""
    return _load_aero_surface(car) is not None


def _dominant_track(track_models_used: dict[str, str]) -> str:
    """Return the most-frequent track string across the trained sessions."""
    counts: dict[str, int] = {}
    for track in track_models_used.values():
        counts[track] = counts.get(track, 0) + 1
    if not counts:
        return "unknown"
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


# ---- Lap-time-weighted samples (PLAN.md Day 6, Mode 3) -----------


# Floor on the lap-time gap denominator. Without this, a session whose best
# lap matches the track minimum would have weight 1/0 = inf, dominating
# every aggregate. 0.5s is the smallest gap that's still meaningful for GTP
# (typical pace deltas across stints are 0.5-2.0s).
_LAP_WEIGHT_EPSILON_S: float = 0.5

# Hard floor on what counts as a "real" lap time. Some sessions have
# `valid=1` laps in the catalog that are <30s -- partial laps, outlap
# fragments, or restart artifacts that the validity heuristic misclassed.
# 60s is below every plausible GTP race lap on every catalog track
# (Sebring ~110s, Daytona road ~95s, Algarve ~105s, etc.) and well above
# the outlier short laps we observed. Anything shorter is filtered out
# of the lap-weight calculation so the "best lap" doesn't accidentally
# point at a junk row.
_LAP_TIME_MIN_VALID_S: float = 60.0


def _compute_lap_time_weights(
    session_ids: list[str],
    *,
    track_per_session: dict[str, str],
    corpus_root: Path,
) -> dict[str, float]:
    """Per-session weight = 1 / (session_best_lap - track_min + epsilon).

    Closes Mode 3 (driver-bias inheritance, PLAN.md Section 14.4): without
    weighting, a slow recent stint pulls the baseline toward conservative
    setups regardless of historical fast laps. Weighting by inverse-gap-
    to-track-min biases the baseline toward setups associated with the
    user's fastest laps on each track, while still using the full corpus.

    Implementation note: PLAN.md §14.4 specified lap_time as the
    weighting axis. The catalog has data-quality issues (some `valid=1`
    laps are partial laps / restart artifacts shorter than any real GTP
    race lap), so we use the session's MEDIAN valid lap time instead of
    its MIN. Median is robust to outliers AND captures "typical pace"
    rather than a one-off best, which is arguably a better signal for
    "what setup were they driving when they were going fast." Hard floor
    `_LAP_TIME_MIN_VALID_S` filters out sub-60s rows (always junk on
    every catalog track for GTP).

    For each session:
    - Pull all valid laps with lap_time >= _LAP_TIME_MIN_VALID_S.
    - Compute the session's median lap time as its pace measure.
    - Compute track_min as the minimum across all input session_ids on
      that same track.
    - Sessions with no valid laps after filtering fall back to weight
      1.0 (defensive -- include at uniform weight rather than drop).

    Returns a dict keyed by session_id; missing session_ids would receive
    1.0 if looked up via .get(sid, 1.0) at the call site.
    """
    # Per-session best lap times.
    session_best: dict[str, float] = {}
    for sid in session_ids:
        try:
            laps_df = ingest_api.laps(
                session_id=sid, valid_only=True, corpus_root=corpus_root,
            )
        except Exception:
            continue
        if laps_df.height == 0:
            continue
        times = [
            float(t) for t in laps_df["lap_time_s"].to_list()
            if t is not None and t >= _LAP_TIME_MIN_VALID_S
        ]
        if not times:
            continue
        session_best[sid] = float(median(times))

    # Per-track minimum across the input batch.
    track_min: dict[str, float] = {}
    for sid in session_ids:
        if sid not in session_best:
            continue
        track = track_per_session.get(sid)
        if not track:
            continue
        prev = track_min.get(track)
        cur = session_best[sid]
        if prev is None or cur < prev:
            track_min[track] = cur

    # Compute weights.
    weights: dict[str, float] = {}
    for sid in session_ids:
        if sid not in session_best:
            weights[sid] = 1.0
            continue
        track = track_per_session.get(sid)
        if not track or track not in track_min:
            weights[sid] = 1.0
            continue
        gap = session_best[sid] - track_min[track]
        weights[sid] = 1.0 / (gap + _LAP_WEIGHT_EPSILON_S)
    return weights


def _weighted_median(values: list[float], weights: list[float]) -> float:
    """Weighted median = smallest value whose cumulative weight >= half.

    Stable across permutations of the input. `values` and `weights` MUST
    be the same length; non-positive weights are treated as 0.
    """
    if not values:
        raise ValueError("_weighted_median: empty values")
    if len(values) != len(weights):
        raise ValueError(
            f"_weighted_median: len(values)={len(values)} != "
            f"len(weights)={len(weights)}"
        )
    paired = sorted(
        ((float(v), max(0.0, float(w))) for v, w in zip(values, weights, strict=True)),
        key=lambda p: p[0],
    )
    total = sum(w for _, w in paired)
    if total <= 0:
        # All weights zero or negative: fall back to plain median.
        return float(median(values))
    half = total / 2.0
    cum = 0.0
    for v, w in paired:
        cum += w
        if cum >= half:
            return float(v)
    return float(paired[-1][0])


def _weighted_std(values: list[float], weights: list[float]) -> float:
    """Population (biased) weighted std.

    sqrt( sum_i w_i * (x_i - x_bar_w)^2 / sum_i w_i ),
    where x_bar_w is the weighted mean.

    Returns 0.0 when weights sum to <=0 (defensive); same fallback as
    statistics.pstdev when fewer than 2 samples.
    """
    if len(values) < 2:
        return 0.0
    if len(values) != len(weights):
        raise ValueError(
            f"_weighted_std: len(values)={len(values)} != "
            f"len(weights)={len(weights)}"
        )
    total = sum(max(0.0, float(w)) for w in weights)
    if total <= 0:
        return float(pstdev(values))
    wmean = sum(float(v) * max(0.0, float(w)) for v, w in zip(values, weights, strict=True)) / total
    var = sum(
        max(0.0, float(w)) * (float(v) - wmean) ** 2
        for v, w in zip(values, weights, strict=True)
    ) / total
    return float(var ** 0.5)


__all__ = [
    "CORNER_ARCHETYPE_COLUMNS",
    "ENV_FEATURE_SCHEMA_VERSION",
    "ENV_FEATURE_SCHEMA_VERSION_PER_CAR",
    "TARGET_OUTPUT_CHANNELS",
    "fit",
    "fit_per_car",
]
