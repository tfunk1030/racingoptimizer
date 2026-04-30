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
import warnings
from pathlib import Path
from statistics import median

import numpy as np
import polars as pl
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold

from racingoptimizer.constraints import load_constraints
from racingoptimizer.corner import corner_phase_states
from racingoptimizer.ingest import api as ingest_api
from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.paths import catalog_path, resolve_corpus_root
from racingoptimizer.physics.baselines import derive_baselines
from racingoptimizer.physics.exceptions import InsufficientDataError
from racingoptimizer.physics.fitters import FitterBase, ForestFitter, GPFitter
from racingoptimizer.physics.io_log import append_accuracy_log
from racingoptimizer.physics.model import FitRecord, PhysicsModel
from racingoptimizer.physics.ontology import (
    Family,
    fittable_parameters,
    ontology_for,
    setup_value,
)
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
)

# Spec §5 family routing: continuous low-dim → GP. Discrete/coupled families
# (damper, corner_weight, brake_bias, diff) fall through to ForestFitter.
_GP_FAMILIES: frozenset[Family] = frozenset(
    {"heave_spring", "heave_slider", "tyre_pressure",
     "front_wing", "rear_wing", "ride_height", "arb"}
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

# Number of env features in the v1 schema (pre-S2.2). Kept for legacy pickles
# that lived under that schema.
ENV_FEATURE_COUNT_V1: int = 5
# Current env schema version emitted by `fit`.
# v3 = Stage-3 joint multi-input model (this PR). See `model.py` for the
# v1 / v2 history.
ENV_FEATURE_SCHEMA_VERSION: int = 3


def fit(
    car: str,
    session_ids: list[str],
    track_model: TrackModel,
    *,
    seed: int = 0xC0FFEE,
    k_folds: int = 5,
    corpus_root: Path | str | None = None,
) -> PhysicsModel:
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

    # Build the joint training frame: one row per (session, corner, phase)
    # with every available parameter as its own column. Sessions missing a
    # parameter contribute NaN for that column; rows with any NaN parameter
    # are dropped to keep the joint vector well-defined.
    joint = _attach_setup_columns(training, sid_to_param_value, available_params)

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
                rec = _fit_one_quadruple(
                    sub=sub,
                    parameters=available_params,
                    output_channel=output_channel,
                    family_kind=family_kind,
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

    aero_available = _try_load_aero(car_key)

    baseline = {
        name: float(median(v for v in (
            setup_snapshots[sid].get(name) for sid in sorted_ids
        ) if v is not None))
        for name in fit_params
        if any(setup_snapshots[sid].get(name) is not None for sid in sorted_ids)
    }

    car_baselines = derive_baselines(car_key, training)

    model = PhysicsModel(
        car=car_key,
        session_ids=tuple(sorted_ids),
        track_models_used=track_models_used,
        fitters=fitters,
        ontology=onto,
        constraints=constraints,
        untrained_parameters=tuple(sorted(untrained_params)),
        aero_correction_available=aero_available,
        baseline_setup=baseline,
        seed=int(seed),
        car_baselines=car_baselines,
        feature_schema_version=ENV_FEATURE_SCHEMA_VERSION,
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
        except Exception:
            # Telemetry-side persistence; never block the model build on it.
            pass

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
    """Pull every valid lap's per-(corner, phase) frame and stack them."""
    frames: list[pl.DataFrame] = []
    for sid in session_ids:
        laps_df = ingest_api.laps(session_id=sid, valid_only=True, corpus_root=root)
        if laps_df.height == 0:
            continue
        for lap_idx in laps_df["lap_index"].to_list():
            try:
                cps = corner_phase_states(sid, int(lap_idx), corpus_root=root)
            except (KeyError, ValueError, FileNotFoundError):
                continue
            if cps.height == 0:
                continue
            frames.append(cps)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


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
) -> FitRecord | None:
    """Fit one (corner_id, phase, output_channel) quadruple over the joint vector."""
    drop_cols = [output_channel] + [p for p in parameters if p in sub.columns]
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

    X = np.concatenate([param_block, env_block], axis=1)
    feature_names: tuple[str, ...] = tuple(param_cols) + tuple(env_cols)

    cv_residual_std = _kfold_residual_std(
        X=X, y=y, family_kind=family_kind, k_folds=k_folds,
        seed=seed, cv_seed=cv_seed,
    )

    fitter = _make_fitter(family_kind, seed=seed)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            fitter.fit(X, y)
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        return None
    if not fitter.is_trained:
        return None

    return FitRecord(
        fitter=fitter,
        n_samples=int(cleaned.height),
        cv_residual_std=float(cv_residual_std),
        signal_std=float(y.std(ddof=0)),
        feature_names=feature_names,
    )


def _make_fitter(family_kind: str, *, seed: int) -> FitterBase:
    if family_kind == "gp":
        return GPFitter(random_state=seed)
    return ForestFitter(random_state=seed)


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


def _try_load_aero(car: str) -> bool:
    """Per spec §9: detect whether slice C is reachable for `car`."""
    try:
        from racingoptimizer.aero import AeroLoadError, load_aero_maps
    except ImportError:
        return False
    try:
        load_aero_maps(car)
    except (FileNotFoundError, AeroLoadError, ImportError):
        return False
    except Exception:  # pragma: no cover — defensive against future load errors
        return False
    return True


def _dominant_track(track_models_used: dict[str, str]) -> str:
    """Return the most-frequent track string across the trained sessions."""
    counts: dict[str, int] = {}
    for track in track_models_used.values():
        counts[track] = counts.get(track, 0) + 1
    if not counts:
        return "unknown"
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


__all__ = ["TARGET_OUTPUT_CHANNELS", "fit"]
