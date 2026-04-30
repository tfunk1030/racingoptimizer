"""Top-level orchestrator: `fit(car, session_ids, track_model)`.

Spec §7 procedure: pull every valid lap's per-(corner, phase) frame, join the
session-level setup blob in as a constant column per session, then for each
(parameter, corner_id, phase, output_channel) quadruple build (X, y), run
K-fold CV to estimate `cv_residual_std`, fit the family-appropriate fitter
on the full data, and stash the result.

The `score_setup` / `recommend` / `weight_corners` flow is reserved for U10
and is intentionally absent here.
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
# / suspension state. Spec §6 lists six sub-utilizations; U9 fits the raw
# state columns only and U10 will combine them into utilizations.
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
# in model.py mirrors this. Schema version 2 spans all 12 channels;
# version-1 models (pre-S2.2) only carried the first five env channels
# (air_density, track_temp_c, wind_vel_ms, wind_dir_deg, track_wetness)
# and `PhysicsModel.predict` truncates to that prefix when revived.
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

# Number of env features in the v1 schema (pre-S2.2). `PhysicsModel.predict`
# rebuilds the v1 vector explicitly when loading an old pickle.
ENV_FEATURE_COUNT_V1: int = 5
# Current env schema version emitted by `fit`.
ENV_FEATURE_SCHEMA_VERSION: int = 2


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

    fitters: dict[tuple[str, int, str, str], FitRecord] = {}
    rng = np.random.default_rng(seed)
    cv_seed = int(rng.integers(0, 2**31 - 1))

    for parameter in fit_params:
        sids_with_param = [
            sid for sid in sorted_ids if setup_snapshots[sid].get(parameter) is not None
        ]
        if not sids_with_param:
            untrained_params.add(parameter)
            continue

        # Cold start (one session) leaves the parameter constant across rows;
        # the fitter still trains on env features and Confidence reflects the
        # sparse n_samples count via `Confidence.derive`'s n_samples<30 short
        # circuit. Multiple sessions with identical parameter values are
        # caught by the y/param-std checks inside `_fit_one_quadruple`.

        spec = onto[parameter]
        family_kind = "gp" if spec.family in _GP_FAMILIES else "rf"

        # Broadcast the per-session parameter value onto every training row.
        sid_to_value = {sid: setup_snapshots[sid][parameter] for sid in sids_with_param}
        df_param = training.with_columns(
            pl.col("session_id").replace_strict(
                sid_to_value, default=None, return_dtype=pl.Float64
            ).alias("__param_value")
        ).filter(pl.col("__param_value").is_not_null())
        if df_param.height < 5:
            untrained_params.add(parameter)
            continue

        fit_any = False
        for (corner_id, phase), sub in df_param.group_by(
            ["corner_id", "phase"], maintain_order=True
        ):
            corner_id = int(corner_id)
            phase_str = str(phase)
            for output_channel in TARGET_OUTPUT_CHANNELS:
                if output_channel not in sub.columns:
                    continue
                rec = _fit_one_quadruple(
                    sub=sub,
                    output_channel=output_channel,
                    family_kind=family_kind,
                    seed=seed,
                    cv_seed=cv_seed,
                    k_folds=k_folds,
                )
                if rec is None:
                    continue
                fitters[(parameter, corner_id, phase_str, output_channel)] = rec
                fit_any = True
        if not fit_any:
            untrained_params.add(parameter)

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

    return PhysicsModel(
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


def _fit_one_quadruple(
    *,
    sub: pl.DataFrame,
    output_channel: str,
    family_kind: str,
    seed: int,
    cv_seed: int,
    k_folds: int,
) -> FitRecord | None:
    """Fit one (parameter, corner_id, phase, output_channel) quadruple."""
    cleaned = sub.drop_nulls([output_channel, "__param_value"])
    if cleaned.height < 3:
        return None

    y = cleaned[output_channel].cast(pl.Float64).to_numpy()
    if y.std() == 0.0:
        return None
    param_vals = cleaned["__param_value"].cast(pl.Float64).to_numpy()
    # When the parameter does not vary across rows (cold-start: single session
    # → constant setup) the fitter still trains on env features. Confidence
    # short-circuits to `sparse` via `n_samples<30`, mirroring the spec's
    # cold-start regime.

    env_cols = [
        c for c in _ENV_COLUMNS if c in cleaned.columns
    ]
    # Cast to Float64 covers the bool / Int32 weather columns at the tail
    # of `_ENV_COLUMNS` (weather_declared_wet_max -> 0.0/1.0; precip_type_max,
    # skies_max -> numeric). fill_null(0.0) handles columns the corner-phase
    # aggregator omitted because the IBT lacked the source channel.
    env_block = (
        cleaned.select(env_cols).cast(pl.Float64).fill_null(0.0).to_numpy()
        if env_cols else np.zeros((cleaned.height, 0), dtype=np.float64)
    )
    # Pad missing env columns with zeros to keep a fixed 12-feature env vector
    # (VISION section 10). Older corpora may only carry the v1 prefix (5 cols)
    # — the padding keeps the X-shape stable across cars / IBT versions.
    if env_block.shape[1] < len(_ENV_COLUMNS):
        n_missing = len(_ENV_COLUMNS) - env_block.shape[1]
        pad = np.zeros((env_block.shape[0], n_missing), dtype=np.float64)
        env_block = np.concatenate([env_block, pad], axis=1)

    X = np.concatenate(
        [param_vals.reshape(-1, 1), env_block], axis=1
    )

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


__all__ = ["TARGET_OUTPUT_CHANNELS", "fit"]
