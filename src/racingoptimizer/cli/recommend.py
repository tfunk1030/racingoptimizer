"""`optimize <car> <track>`, `optimize compare`, `optimize status` (slice F)."""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import click
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
    track_slug = _resolve_track_or_exit(track, catalog_sessions)

    sessions_for_track = catalog_sessions.filter(pl.col("track") == track_slug)
    if sessions_for_track.height == 0:
        click.echo(
            f"model has no data on ({car_key}, {track_slug}); "
            f"run `optimize learn <ibt>` first",
            err=True,
        )
        sys.exit(2)

    pinned_overrides = _parse_pins(pins, wing=wing)
    constraints_table = load_constraints()
    pinned_constraints = _apply_pins_to_constraints(
        constraints_table, car_key, pinned_overrides,
    )

    session_ids = sorted(sessions_for_track["session_id"].to_list())
    model = _build_or_load_model(
        car_key, track_slug, session_ids, root, no_cache=no_cache,
    )

    env = _env_from_overrides(
        model=model, sessions=sessions_for_track,
        air_temp=air_temp, track_temp=track_temp,
        wind=wind, wetness=wetness,
    )

    rec = model.recommend(track_slug, env, pinned_constraints)
    rec, clamp_warnings, top_warnings = _post_clamp(rec, model, constraints_table)

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

    env = _env_from_overrides(model=model, sessions=catalog_sessions)
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


def _resolve_track_or_exit(track: str, sessions: pl.DataFrame) -> str:
    needle = track.strip().lower()
    available = sorted(set(sessions["track"].to_list()))
    if needle in available:
        return needle
    matches = [slug for slug in available if needle in slug]
    if not matches:
        click.echo(
            f"unknown track {track!r}; available tracks for this car: "
            f"{', '.join(available) if available else '(none)'}",
            err=True,
        )
        sys.exit(2)
    if len(matches) > 1:
        click.echo(
            f"ambiguous track {track!r}; candidates: {', '.join(matches)}",
            err=True,
        )
        sys.exit(2)
    return matches[0]


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
    air_temp: float | None = None,
    track_temp: float | None = None,
    wind: float | None = None,
    wetness: float | None = None,
) -> EnvironmentFrame:
    medians = _median_environment(model, sessions)
    return EnvironmentFrame(
        air_density=medians["air_density"],
        track_temp_c=track_temp if track_temp is not None else medians["track_temp_c"],
        wind_vel_ms=wind if wind is not None else medians["wind_vel_ms"],
        wind_dir_deg=medians["wind_dir_deg"],
        track_wetness=wetness if wetness is not None else medians["track_wetness"],
    )


def _median_environment(model, sessions: pl.DataFrame) -> dict[str, float]:
    """Median per-channel environment from the catalog `weather_summary` JSON.

    Falls back to standard atmospheric defaults when telemetry omits weather.
    """
    defaults = {
        "air_density": 1.225,
        "track_temp_c": 25.0,
        "wind_vel_ms": 0.0,
        "wind_dir_deg": 0.0,
        "track_wetness": 0.0,
    }
    parsed: list[dict] = []
    for raw in sessions["weather_summary"].to_list():
        if not raw:
            continue
        try:
            blob = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(blob, dict):
            parsed.append(blob)
    if not parsed:
        return defaults
    out: dict[str, float] = {}
    for key, default in defaults.items():
        values = [
            float(b[key]) for b in parsed
            if isinstance(b.get(key), (int, float))
        ]
        out[key] = float(sorted(values)[len(values) // 2]) if values else default
    return out


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
