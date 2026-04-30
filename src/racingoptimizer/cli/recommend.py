"""`optimize <car> <track>`, `optimize compare`, `optimize status` (slice F)."""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import click
import numpy as np
import polars as pl

from racingoptimizer.confidence import Confidence
from racingoptimizer.constraints import (
    ConstraintsTable,
    clamp,
    load_constraints,
)
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.explain import (
    CornerPhaseDelta,
    SetupComparison,
    render_comparison_json,
    render_comparison_text,
    render_recommendation_json,
    render_recommendation_text,
    render_status_json,
    render_status_text,
)
from racingoptimizer.explain.justification import build_justifications
from racingoptimizer.explain.status import ModelStatus, TrackCoverage
from racingoptimizer.ingest import api as ingest_api
from racingoptimizer.ingest.detect import (
    UnknownCarError,
    normalize_car_key,
)
from racingoptimizer.ingest.paths import resolve_corpus_root

CANONICAL_CARS = ("acura", "bmw", "cadillac", "ferrari", "porsche")


# --------------------------------------------------------------------------
# `optimize <car> <track>` recommend command
# --------------------------------------------------------------------------


@click.command(
    name="recommend",
    context_settings={"ignore_unknown_options": False},
)
@click.argument("car")
@click.argument("track")
@click.option("--wing", type=float, default=None, help="Pin rear wing angle (degrees).")
@click.option(
    "--air-temp", type=float, default=None,
    help="Override training-data median air temperature (deg C).",
)
@click.option(
    "--track-temp", type=float, default=None,
    help="Override training-data median track temperature (deg C).",
)
@click.option(
    "--wind", type=float, default=None,
    help="Override training-data median wind velocity (m/s).",
)
@click.option(
    "--wetness", type=float, default=None,
    help="Override training-data median track wetness (0..1).",
)
@click.option(
    "--pin", "pins", multiple=True, default=(),
    help="Pin a parameter: --pin KEY=VAL. Repeatable.",
)
@click.option(
    "--json", "as_json", is_flag=True, default=False,
    help="Emit JSON instead of human-text briefing.",
)
@click.option(
    "--corpus-root", type=click.Path(path_type=Path), default=None,
    help="Override corpus location (else uses RACINGOPTIMIZER_CORPUS or repo default).",
)
@click.option(
    "--no-cache", is_flag=True, default=False,
    help="Bypass on-disk PhysicsModel cache and refit.",
)
def recommend_cmd(
    car: str,
    track: str,
    wing: float | None,
    air_temp: float | None,
    track_temp: float | None,
    wind: float | None,
    wetness: float | None,
    pins: tuple[str, ...],
    as_json: bool,
    corpus_root: Path | None,
    no_cache: bool,
) -> None:
    """Recommend a setup for `<car>` at `<track>`."""
    car_key = _resolve_car_or_exit(car)
    root = resolve_corpus_root(corpus_root)
    catalog_sessions = _safe_sessions(car_key, corpus_root=root)
    track_slug, donor_track = _resolve_track_or_extrapolate(
        track, catalog_sessions, car_key,
    )

    pinned_overrides = _parse_pins(pins, wing=wing)
    constraints_table = load_constraints()
    pinned_constraints = _apply_pins_to_constraints(
        constraints_table, car_key, pinned_overrides,
    )

    fit_track = donor_track or track_slug
    sessions_for_fit = catalog_sessions.filter(pl.col("track") == fit_track)
    session_ids = sorted(sessions_for_fit["session_id"].to_list())
    model = _build_or_load_model(
        car_key, fit_track, session_ids, root, no_cache=no_cache,
    )

    env = _env_from_overrides(
        model=model, sessions=sessions_for_fit,
        air_temp=air_temp, track_temp=track_temp,
        wind=wind, wetness=wetness,
        corpus_root=root,
    )

    rec = model.recommend(fit_track, env, pinned_constraints)
    rec, clamp_warnings, top_warnings = _post_clamp(rec, model, constraints_table)

    if donor_track is not None:
        rec = _force_sparse_regime(rec, track_slug)
        top_warnings.insert(
            0,
            f"untrained track {track_slug}; extrapolated from {donor_track} "
            f"({len(session_ids)} sessions). Treat all values as starting "
            f"points; verify on track.",
        )

    justifications = build_justifications(
        rec, model,
        pinned=pinned_overrides,
        clamp_warnings=clamp_warnings,
    )

    if as_json:
        out = render_recommendation_json(
            rec, model,
            justifications=justifications,
            pinned=pinned_overrides,
            warnings=top_warnings,
            track_display=track_slug,
        )
        click.echo(json.dumps(out, indent=2, sort_keys=False))
    else:
        click.echo(render_recommendation_text(
            rec, model,
            justifications=justifications,
            pinned=pinned_overrides,
            warnings=top_warnings,
            track_display=track_slug,
        ))


# --------------------------------------------------------------------------
# `optimize compare`
# --------------------------------------------------------------------------


@click.command(name="compare")
@click.argument("ibt_a", type=click.Path(exists=True, path_type=Path))
@click.argument("ibt_b", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--no-auto-learn", is_flag=True, default=False,
    help="Refuse to ingest IBTs missing from the catalog (default: auto-learn).",
)
@click.option(
    "--json", "as_json", is_flag=True, default=False,
    help="Emit JSON instead of human-text comparison.",
)
@click.option(
    "--corpus-root", type=click.Path(path_type=Path), default=None,
    help="Override corpus location.",
)
def compare_cmd(
    ibt_a: Path,
    ibt_b: Path,
    no_auto_learn: bool,
    as_json: bool,
    corpus_root: Path | None,
) -> None:
    """Diff two setups by per-corner-phase score."""
    root = resolve_corpus_root(corpus_root)
    sid_a = _hash_or_learn(ibt_a, root, allow_learn=not no_auto_learn)
    sid_b = _hash_or_learn(ibt_b, root, allow_learn=not no_auto_learn)

    sessions_all = ingest_api.sessions(corpus_root=root)
    row_a = _session_row(sessions_all, sid_a)
    row_b = _session_row(sessions_all, sid_b)
    if row_a is None or row_b is None:
        click.echo(
            f"one or both IBTs not in catalog (a={sid_a!r}, b={sid_b!r}); "
            f"run `optimize learn` to ingest first",
            err=True,
        )
        sys.exit(2)

    if row_a["car"] != row_b["car"] or row_a["track"] != row_b["track"]:
        click.echo(
            f"cannot compare across cars/tracks: "
            f"a=({row_a['car']}, {row_a['track']})  "
            f"b=({row_b['car']}, {row_b['track']})",
            err=True,
        )
        sys.exit(2)

    car = str(row_a["car"])
    track = str(row_a["track"])
    catalog_sessions = ingest_api.sessions(car=car, track=track, corpus_root=root)
    session_ids = sorted(catalog_sessions["session_id"].to_list())
    model = _build_or_load_model(car, track, session_ids, root)

    setup_a = _decode_setup(row_a["setup"])
    setup_b = _decode_setup(row_b["setup"])
    setup_a_full = _hydrate_setup(model, setup_a)
    setup_b_full = _hydrate_setup(model, setup_b)

    env = _env_from_overrides(model=model, sessions=catalog_sessions, corpus_root=root)
    score_a = float(model.score_setup(setup_a_full, track, env))
    score_b = float(model.score_setup(setup_b_full, track, env))

    deltas = _comparison_deltas(model, setup_a_full, setup_b_full, track, env)

    cmp = SetupComparison(
        car=car,
        track=track,
        setup_a_id=sid_a,
        setup_b_id=sid_b,
        total_score_a=score_a,
        total_score_b=score_b,
        per_corner_phase=tuple(deltas),
        notes=(),
    )
    if as_json:
        click.echo(json.dumps(render_comparison_json(cmp), indent=2, sort_keys=False))
    else:
        click.echo(render_comparison_text(cmp))


# --------------------------------------------------------------------------
# `optimize status`
# --------------------------------------------------------------------------


@click.command(name="status")
@click.argument("car")
@click.option(
    "--json", "as_json", is_flag=True, default=False,
    help="Emit JSON instead of human-text status.",
)
@click.option(
    "--corpus-root", type=click.Path(path_type=Path), default=None,
    help="Override corpus location.",
)
def status_cmd(car: str, as_json: bool, corpus_root: Path | None) -> None:
    """Print what the model knows about a car."""
    car_key = _resolve_car_or_exit(car)
    root = resolve_corpus_root(corpus_root)
    sessions = _safe_sessions(car_key, corpus_root=root)
    if sessions.height == 0:
        empty = ModelStatus(
            car=car_key,
            coverage=(),
            overall_regime="sparse",
            notes=(f"no sessions ingested for {car_key}; run `optimize learn`",),
        )
        if as_json:
            click.echo(json.dumps(render_status_json(empty), indent=2, sort_keys=False))
        else:
            click.echo(render_status_text(empty))
        return

    coverage: list[TrackCoverage] = []
    for track_slug in sorted(set(sessions["track"].to_list())):
        sub = sessions.filter(pl.col("track") == track_slug)
        sids = sorted(sub["session_id"].to_list())
        valid_laps = ingest_api.laps(
            car=car_key, track=track_slug, valid_only=True, corpus_root=root,
        )
        n_clean = _approximate_clean_corner_phases(sids, root)
        regime = _coverage_regime(len(sids), valid_laps.height)
        coverage.append(
            TrackCoverage(
                track=track_slug,
                n_sessions=len(sids),
                n_valid_laps=int(valid_laps.height),
                n_clean_corner_phases=int(n_clean),
                fit_quality=None,
                regime=regime,
            )
        )

    overall = _overall_regime(coverage)
    notes = _status_notes()

    status = ModelStatus(
        car=car_key,
        coverage=tuple(coverage),
        overall_regime=overall,
        notes=tuple(notes),
    )
    if as_json:
        click.echo(json.dumps(render_status_json(status), indent=2, sort_keys=False))
    else:
        click.echo(render_status_text(status))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _resolve_car_or_exit(raw: str) -> str:
    raw_key = raw.strip().lower()
    if raw_key in CANONICAL_CARS:
        return raw_key
    try:
        return normalize_car_key(raw_key)
    except UnknownCarError:
        click.echo(
            f"unknown car {raw!r}; expected one of {', '.join(CANONICAL_CARS)}",
            err=True,
        )
        sys.exit(2)


def _safe_sessions(car: str, *, corpus_root: Path) -> pl.DataFrame:
    try:
        return ingest_api.sessions(car=car, corpus_root=corpus_root)
    except FileNotFoundError:
        click.echo(
            f"corpus not found at {corpus_root}; run `optimize learn <ibt>` first",
            err=True,
        )
        sys.exit(4)


def _resolve_track_or_extrapolate(
    track: str, sessions: pl.DataFrame, car_key: str,
) -> tuple[str, str | None]:
    """Return (target_slug, donor_slug).

    `donor_slug` is None when `track` matches a slug the car has been driven
    on. When the car has never seen `track` but has data on at least one
    other track, `target_slug` is the (normalised) requested slug and
    `donor_slug` is the trained track with the most sessions — the model
    will be fit there and recommendations extrapolated. When the car has no
    other tracks ingested, exit 2 with the standard untrained message.
    """
    needle = track.strip().lower()
    available = sorted(set(sessions["track"].to_list()))
    if needle in available:
        return needle, None
    matches = [slug for slug in available if needle in slug]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        click.echo(
            f"ambiguous track {track!r}; candidates: {', '.join(matches)}",
            err=True,
        )
        sys.exit(2)
    if not available:
        click.echo(
            f"model has no data on ({car_key}, {needle}); "
            f"run `optimize learn <ibt>` first",
            err=True,
        )
        sys.exit(2)
    counts = (
        sessions.group_by("track")
        .agg(pl.len().alias("n"))
        .sort(["n", "track"], descending=[True, False])
    )
    return needle, str(counts["track"][0])


def _force_sparse_regime(rec, target_track: str):
    """Override every parameter's confidence regime to 'sparse' and retag track.

    Used in the untrained-track extrapolation flow: the donor model produced
    `rec`, but the user asked for a different track, so every confidence
    must reflect the extrapolation gap.
    """
    new_params = {
        name: (
            value,
            confidence if confidence.regime == "sparse"
            else replace(confidence, regime="sparse"),
        )
        for name, (value, confidence) in rec.parameters.items()
    }
    return replace(rec, track=target_track, parameters=new_params)


def _parse_pins(pins: tuple[str, ...], *, wing: float | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if wing is not None:
        out["rear_wing_angle_deg"] = float(wing)
    for raw in pins:
        if "=" not in raw:
            click.echo(
                f"invalid --pin {raw!r}; expected KEY=VALUE",
                err=True,
            )
            sys.exit(2)
        key, value_str = raw.split("=", 1)
        try:
            out[key.strip()] = float(value_str.strip())
        except ValueError:
            click.echo(
                f"invalid --pin {raw!r}; value must be numeric",
                err=True,
            )
            sys.exit(2)
    return out


def _apply_pins_to_constraints(
    table: ConstraintsTable,
    car: str,
    pins: dict[str, float],
) -> ConstraintsTable:
    """Return a constraints table with pinned parameters narrowed to [v, v]."""
    if not pins:
        return table
    # ConstraintsTable doesn't expose a mutator; re-build via the dataclass
    # field. The leaky access is intentional and confined to this helper.
    by_car: dict[str, dict[str, tuple[float, float] | None]] = {
        k: dict(v) for k, v in table._by_car.items()  # noqa: SLF001
    }
    by_car.setdefault(car, {})
    for key, value in pins.items():
        if key not in table.parameters():
            click.echo(
                f"unknown --pin parameter {key!r}; "
                f"known parameters: {', '.join(table.parameters())}",
                err=True,
            )
            sys.exit(2)
        # Narrowest legal interval — recommender treats it as a fixed value.
        original = table.bounds(car, key)
        if original is not None:
            value = max(original[0], min(original[1], float(value)))
        by_car[car][key] = (float(value), float(value))
    return ConstraintsTable(_by_car=by_car)


def _build_or_load_model(
    car: str,
    track: str,
    session_ids: list[str],
    root: Path,
    *,
    no_cache: bool = False,
):
    cache_path = _model_cache_path(root, car, track, session_ids)
    if not no_cache and cache_path.exists():
        try:
            with cache_path.open("rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass

    from racingoptimizer.physics import InsufficientDataError, fit
    from racingoptimizer.track import build_track_model

    track_model = build_track_model(track, session_ids, corpus_root=root)
    # Cold-start corpora (≤2 sessions) have too few rows for k=5 CV; fall
    # back to k=2 so the recommend path produces *some* fit. Spec §10
    # untrained-recommendation handling kicks in further down if even k=2
    # fails.
    k_folds = 5 if len(session_ids) >= 3 else 2
    try:
        model = fit(car, session_ids, track_model, corpus_root=root, k_folds=k_folds)
    except InsufficientDataError as exc:
        click.echo(
            f"insufficient training data for ({car}, {track}): {exc}",
            err=True,
        )
        sys.exit(2)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with cache_path.open("wb") as fh:
            pickle.dump(model, fh)
    except Exception:
        pass
    return model


def _model_cache_path(root: Path, car: str, track: str, session_ids: list[str]) -> Path:
    digest = hashlib.sha256(
        "|".join(sorted(session_ids)).encode("utf-8")
    ).hexdigest()[:16]
    return root / "models" / f"{car}__{track}__{digest}.pickle"


def _env_from_overrides(
    *,
    model,
    sessions: pl.DataFrame,
    corpus_root: Path,
    air_temp: float | None = None,
    track_temp: float | None = None,
    wind: float | None = None,
    wetness: float | None = None,
) -> EnvironmentFrame:
    medians = _environment_from_corpus(sessions, corpus_root=corpus_root)
    return EnvironmentFrame(
        air_temp_c=air_temp if air_temp is not None else float(medians["air_temp_c"]),
        air_density=float(medians["air_density"]),
        air_pressure_mbar=float(medians["air_pressure_mbar"]),
        relative_humidity=float(medians["relative_humidity"]),
        wind_vel_ms=wind if wind is not None else float(medians["wind_vel_ms"]),
        wind_dir_deg=float(medians["wind_dir_deg"]),
        fog_level=float(medians["fog_level"]),
        track_temp_c=track_temp if track_temp is not None else float(medians["track_temp_c"]),
        track_wetness=wetness if wetness is not None else float(medians["track_wetness"]),
        weather_declared_wet=bool(medians["weather_declared_wet"]),
        precip_type=int(medians["precip_type"]),
        skies=int(medians["skies"]),
    )


# Per-sample env channel name in IBT/parquet -> EnvironmentFrame field key.
# Wind direction is handled separately because circular medians cannot be
# pooled with arithmetic medians.
# Per-sample env channels (continuous floats) → EnvironmentFrame field key.
# Aggregated via per-sample median across every clean sample in the corpus.
_ENV_FLOAT_CHANNELS: dict[str, str] = {
    "AirTemp": "air_temp_c",
    "AirDensity": "air_density",
    "AirPressure": "air_pressure_mbar",
    "RelativeHumidity": "relative_humidity",
    "WindVel": "wind_vel_ms",
    "FogLevel": "fog_level",
    "TrackTempCrew": "track_temp_c",
    "TrackWetness": "track_wetness",
}
# Discrete env channels (bool / int). Aggregated via .max() — any-wet wins.
_ENV_DISCRETE_CHANNELS: dict[str, str] = {
    "WeatherDeclaredWet": "weather_declared_wet",
    "Precipitation": "precip_type",
    "Skies": "skies",
}
# Backward-compat alias used by tests.
_ENV_CHANNELS = _ENV_FLOAT_CHANNELS
_WIND_DIR_CHANNEL = "WindDir"

# VISION §10 standard-atmosphere fallback when no clean samples exist.
_ENV_DEFAULTS: dict[str, float | bool | int] = {
    "air_temp_c": 25.0,
    "air_density": 1.225,
    "air_pressure_mbar": 1013.25,
    "relative_humidity": 0.5,
    "wind_vel_ms": 0.0,
    "wind_dir_deg": 0.0,
    "fog_level": 0.0,
    "track_temp_c": 25.0,
    "track_wetness": 0.0,
    "weather_declared_wet": False,
    "precip_type": 0,
    "skies": 0,
}


def _environment_from_corpus(
    sessions: pl.DataFrame, *, corpus_root: Path,
) -> dict[str, float]:
    """Per-channel median across every clean sample in the (car, track) corpus.

    VISION §10 + master-plan rule: every data point carries env context, and
    "do not collapse to session averages." So we walk valid laps and pull the
    per-sample weather time-series from each lap's parquet slice, filter to
    `data_quality_mask == True`, then take per-channel medians (circular
    median for `WindDir`). Falls back to standard-atmosphere defaults when
    zero clean samples are available.
    """
    if sessions.height == 0:
        return dict(_ENV_DEFAULTS)

    columns = (
        list(_ENV_FLOAT_CHANNELS)
        + list(_ENV_DISCRETE_CHANNELS)
        + [_WIND_DIR_CHANNEL, "data_quality_mask"]
    )
    accum: dict[str, list[np.ndarray]] = {c: [] for c in columns if c != "data_quality_mask"}

    for sid in sessions["session_id"].to_list():
        try:
            valid_laps = ingest_api.laps(
                session_id=sid, valid_only=True, corpus_root=corpus_root,
            )
        except Exception:
            continue
        for lap_idx in valid_laps["lap_index"].to_list():
            df = _safe_lap_data(sid, int(lap_idx), columns, corpus_root)
            if df is None or df.height == 0:
                continue
            if "data_quality_mask" in df.columns:
                df = df.filter(pl.col("data_quality_mask"))
            if df.height == 0:
                continue
            for channel in accum:
                if channel in df.columns:
                    accum[channel].append(df[channel].to_numpy())

    out: dict[str, float | bool | int] = dict(_ENV_DEFAULTS)
    for channel, field in _ENV_FLOAT_CHANNELS.items():
        chunks = accum[channel]
        if not chunks:
            continue
        stacked = np.concatenate(chunks)
        if stacked.size:
            out[field] = float(np.median(stacked))
    # Discrete channels: any-wet wins. .max() over uint/bool arrays.
    for channel, field in _ENV_DISCRETE_CHANNELS.items():
        chunks = accum.get(channel, [])
        if not chunks:
            continue
        stacked = np.concatenate(chunks)
        if stacked.size:
            value = stacked.max()
            if field == "weather_declared_wet":
                out[field] = bool(value)
            else:
                out[field] = int(value)

    wind_chunks = accum[_WIND_DIR_CHANNEL]
    if wind_chunks:
        stacked_dir = np.concatenate(wind_chunks)
        if stacked_dir.size:
            out["wind_dir_deg"] = _circular_median_deg(stacked_dir)
    return out


def _safe_lap_data(
    session_id: str, lap_index: int, channels: list[str], corpus_root: Path,
) -> pl.DataFrame | None:
    """Read lap_data, gracefully skipping channels missing from a parquet.

    Per-car channel coverage is uneven (e.g. Acura lacks shock-deflection
    columns). We retry with the intersection of requested channels and the
    parquet's actual schema rather than letting a single missing column
    poison every lap.
    """
    try:
        return ingest_api.lap_data(
            session_id, lap_index, channels=channels, corpus_root=corpus_root,
        )
    except pl.exceptions.ColumnNotFoundError:
        pass
    except Exception:
        return None
    try:
        full = ingest_api.lap_data(
            session_id, lap_index, channels=None, corpus_root=corpus_root,
        )
    except Exception:
        return None
    keep = [c for c in channels if c in full.columns]
    return full.select(keep) if keep else None


def _circular_median_deg(angles_deg: np.ndarray) -> float:
    """Circular (directional) median of compass-bearing samples in degrees.

    Wind direction is an angle on the unit circle, so the arithmetic median
    of e.g. [350, 10] is 180 — the wrong direction. Convert to unit vectors,
    take the componentwise median, and project back to a 0..360 bearing.
    """
    radians = np.deg2rad(angles_deg.astype(np.float64))
    x = float(np.median(np.cos(radians)))
    y = float(np.median(np.sin(radians)))
    if x == 0.0 and y == 0.0:
        return 0.0
    return float(math.degrees(math.atan2(y, x)) % 360.0)


def _post_clamp(rec, model, constraints_table: ConstraintsTable):
    """Clamp every parameter against the global constraints table.

    Returns (rec_with_clamped_values, clamp_warnings, top_level_warnings).
    Clamped values are appended to the parameter's evidence via the
    `clamp_warnings` map consumed by `build_justifications`.
    """
    clamp_warnings: dict[str, str] = {}
    top_warnings: list[str] = []
    new_params: dict[str, tuple[float, Confidence]] = {}
    for name, (value, confidence) in rec.parameters.items():
        if name not in constraints_table.parameters():
            new_params[name] = (value, confidence)
            top_warnings.append(
                f"{name} skipped: not in constraints.md (no legal bounds)"
            )
            continue
        result = clamp(float(value), name, model.car, constraints_table)
        if result.bound is None:
            top_warnings.append(
                f"{name} skipped: bound is TODO in constraints.md"
            )
            continue
        if result.was_clamped:
            clamp_warnings[name] = (
                f"value clamped from {value:.3f} to {result.value:.3f} "
                f"(legal bounds: {result.bound[0]:.3f} - {result.bound[1]:.3f})"
            )
            new_params[name] = (float(result.value), confidence)
        else:
            new_params[name] = (value, confidence)
    return replace(rec, parameters=new_params), clamp_warnings, top_warnings


def _hash_or_learn(path: Path, root: Path, *, allow_learn: bool) -> str:
    raw = path.read_bytes()
    from racingoptimizer.ingest.writer import session_id_from_bytes
    sid = session_id_from_bytes(raw)
    sessions_all = ingest_api.sessions(corpus_root=root)
    if sid in set(sessions_all["session_id"].to_list()):
        return sid
    if not allow_learn:
        click.echo(
            f"IBT {path.name!r} not in catalog and --no-auto-learn was set",
            err=True,
        )
        sys.exit(2)
    ingest_api.learn(path, corpus_root=root)
    return sid


def _session_row(sessions: pl.DataFrame, session_id: str) -> dict | None:
    matches = sessions.filter(pl.col("session_id") == session_id)
    if matches.height == 0:
        return None
    return matches.head(1).to_dicts()[0]


def _decode_setup(blob: str | None) -> dict:
    if not blob:
        return {}
    try:
        loaded = json.loads(blob)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _hydrate_setup(model, raw_setup: dict) -> dict[str, float]:
    """Pull every fittable parameter value out of the raw IBT setup JSON."""
    from racingoptimizer.physics.ontology import setup_value
    out = dict(model.baseline_setup)
    for name in model.ontology:
        try:
            value = setup_value(model.car, name, raw_setup)
        except KeyError:
            continue
        if value is not None:
            out[name] = float(value)
    return out


def _comparison_deltas(model, setup_a, setup_b, track: str, env):
    from racingoptimizer.physics.score import score_breakdown

    breakdown_a = score_breakdown(model, setup_a, track, env)
    breakdown_b = score_breakdown(model, setup_b, track, env)
    drivers = tuple(_driver_params(setup_a, setup_b))
    deltas: list[CornerPhaseDelta] = []
    for cpkey in sorted(set(breakdown_a) | set(breakdown_b)):
        a = float(breakdown_a.get(cpkey, 0.0))
        b = float(breakdown_b.get(cpkey, 0.0))
        delta_val = b - a
        if abs(delta_val) < 1e-6:
            continue
        deltas.append(
            CornerPhaseDelta(
                corner_id=int(cpkey.corner_id),
                phase=cpkey.phase,
                score_a=a,
                score_b=b,
                delta=delta_val,
                drivers=drivers,
            )
        )
    return deltas


def _driver_params(setup_a: dict, setup_b: dict) -> list[str]:
    """Parameters whose values differ between the two setups."""
    diffs: list[tuple[str, float]] = []
    for key, a_val in setup_a.items():
        b_val = setup_b.get(key)
        if b_val is None:
            continue
        delta = abs(float(a_val) - float(b_val))
        if delta < 1e-6:
            continue
        diffs.append((key, delta))
    diffs.sort(key=lambda kv: kv[1], reverse=True)
    return [
        f"{key}: {setup_a[key]:.2f} -> {setup_b[key]:.2f}"
        for key, _ in diffs[:5]
    ]


def _approximate_clean_corner_phases(
    session_ids: list[str], root: Path,
) -> int:
    """Best-effort estimate without full corner-phase decomposition.

    `n_valid_laps × ~30 typical corner-phase rows per lap` is a working
    estimate when slice D's mask hasn't been applied. When the corner-phase
    parquet is materialised on disk this can be replaced with a real count.
    """
    total_rows = 0
    for sid in session_ids:
        try:
            laps = ingest_api.laps(session_id=sid, valid_only=True, corpus_root=root)
        except Exception:
            continue
        total_rows += int(laps.height) * 30
    return total_rows


def _coverage_regime(n_sessions: int, n_laps: int) -> str:
    if n_laps == 0 or n_sessions == 0:
        return "sparse"
    if n_sessions >= 4 and n_laps >= 100:
        return "dense"
    if n_sessions >= 2 and n_laps >= 30:
        return "confident"
    if n_sessions >= 2:
        return "noisy"
    return "sparse"


_REGIME_RANK: dict[str, int] = {"sparse": 0, "noisy": 1, "confident": 2, "dense": 3}
_RANK_TO_REGIME: dict[int, str] = {v: k for k, v in _REGIME_RANK.items()}


def _overall_regime(coverage: list[TrackCoverage]) -> str:
    if not coverage:
        return "sparse"
    weighted_rank = 0.0
    weight_sum = 0.0
    for cov in coverage:
        weight = max(int(cov.n_clean_corner_phases), 1)
        weighted_rank += weight * _REGIME_RANK[cov.regime]
        weight_sum += weight
    if weight_sum == 0:
        return "sparse"
    return _RANK_TO_REGIME[int(round(weighted_rank / weight_sum))]


def _status_notes() -> list[str]:
    return [
        "constraints.md is missing bounds for: ARBs, dampers, corner_weights, "
        "brake_bias, differential, camber, toe, brake_ducts.",
    ]


__all__ = ["compare_cmd", "recommend_cmd", "status_cmd"]
