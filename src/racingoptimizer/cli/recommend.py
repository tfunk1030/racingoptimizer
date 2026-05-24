"""`optimize <car> <track>`, `optimize compare`, `optimize status` (slice F)."""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
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
from racingoptimizer.context import (
    IBT_BOOL_CHANNELS,
    IBT_FLOAT_CHANNELS,
    IBT_INT_CHANNELS,
    EnvironmentFrame,
)
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
from racingoptimizer.explain.full_setup_card import render_full_setup_card
from racingoptimizer.explain.justification import build_justifications
from racingoptimizer.explain.status import ModelStatus, TrackCoverage
from racingoptimizer.ingest import api as ingest_api
from racingoptimizer.ingest.detect import (
    UnknownCarError,
    detect_car_from_filename,
    detect_track_from_filename,
    normalize_car_key,
    slugify_track,
)
from racingoptimizer.ingest.paths import resolve_corpus_root

CANONICAL_CARS = ("acura", "bmw", "cadillac", "ferrari", "porsche")

# Cars that use the per-car (track-agnostic) physics model. Per VISION §3 /
# §6: "Build an empirical physics model for each car... be able to generate
# optimal setups for any track from just the track model and aero maps."
# All five GTP cars use the v4 per-car pooled model (`fit_per_car`). The
# v3 per-(car, track) branch below is retained for rollback only.
PER_CAR_MODEL_CARS: frozenset[str] = frozenset(
    {"cadillac", "bmw", "ferrari", "acura", "porsche"},
)


# --------------------------------------------------------------------------
# `optimize <car> <track>` recommend command
# --------------------------------------------------------------------------


@click.command(
    name="recommend",
    context_settings={"ignore_unknown_options": False},
)
@click.argument("car")
@click.argument("track", required=False, default=None)
@click.option("--wing", type=float, default=None, help="Pin rear wing angle (degrees).")
@click.option(
    "--fuel", "fuel", type=float, default=None,
    help=(
        "Pin race-fuel level (L). Race default is the past-session value "
        "(typically ~58 L on the BMW M Hybrid V8); a quali stint is "
        "user-input depending on track length (commonly 5..15 L for 3 "
        "laps + reserve). The optimizer treats fuel as a fittable input "
        "so a low pin biases ride-height + balance predictions."
    ),
)
@click.option(
    "--quali", "quali", is_flag=True, default=False,
    help=(
        "Quali-stint mode: phase weights tilt toward outright single-lap "
        "pace (more aero_eff, more grip utilisation, less platform "
        "conservatism). Pair with `--fuel N` to pin the matching low "
        "fuel load — the optimizer will not auto-pick a quali fuel."
    ),
)
@click.option(
    "--explore", "explore_pct", type=float, default=0.0,
    help=(
        "Widen the per-track empirical envelope by N%% of each "
        "parameter's constraint span on each side (clipped to legal "
        "bounds). Lets the optimizer probe values outside what you've "
        "driven; recommendations in the widened territory carry weaker "
        "confidence. Default 0 = strict empirical envelope. Try 5-10 "
        "for modest exploration, 20-30 for aggressive."
    ),
)
@click.option(
    "--detailed", "detailed", is_flag=True, default=False,
    help=(
        "Render the legacy per-parameter block format (Helps/Hurts "
        "with score deltas, ±1-click sensitivity, evidence) instead "
        "of the default plain-English narrative. Useful for engineering "
        "drill-downs and validator agents."
    ),
)
@click.option(
    "--staged", "staged", is_flag=True, default=False,
    help=(
        "Run DE in 4 progressive stages + 1 polish pass instead of a "
        "single search over all 47 parameters. Stages: aero (wing + "
        "ride heights + tyre P) -> mechanical (springs + ARBs + torsion) "
        "-> dampers -> detail (cambers + toes + brake bias + diff). Each "
        "stage holds previous stages' chosen values as pins. Final "
        "polish re-opens the full vector at the user-supplied --explore "
        "level (0 by default) widening "
        "seeded from the accumulated stage results. Mirrors engineer "
        "setup workflow; total wall time roughly equivalent to single-"
        "pass DE."
    ),
)
@click.option(
    "--reset", "reset_mode", is_flag=True, default=False,
    help=(
        "Open the search to [corpus_min - 30%, corpus_max + 30%] of "
        "constraint span on each side, skip the corpus-density pin "
        "check (so observed-constants can move), and downgrade every "
        "parameter's confidence to noisy. Use when the current setup "
        "feels fundamentally wrong and small tweaks aren't moving the "
        "car. The briefing prints a RESET MODE banner; verify on "
        "track before pushing."
    ),
)
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
@click.option(
    "--output-file", type=click.Path(path_type=Path), default=None,
    help=(
        "Write the full briefing + setup card to this file. Defaults to "
        "`recommendations/<car>-<track>-<mode>[-<fuel>L]-<MMDD>-<HHMM>.txt` "
        "(or `.json` with --json). Pass `-` to disable file output."
    ),
)
@click.option(
    "--surrogate-only", "surrogate_only", is_flag=True, default=False,
    help=(
        "Use surrogate-only DE objective (pre-rebuild behavior). "
        "Default uses hybrid physics+surrogate scoring."
    ),
)
@click.option(
    "--physics", "physics_mode", is_flag=True, default=False,
    help=(
        "Enable physics-aware briefing: surfaces guardrail warnings "
        "(axle utilization > 1.0, severe aero imbalance) per the Week 2 "
        "evaluator. Recommendation values are unchanged; only the "
        "briefing output gains physics-derived warnings."
    ),
)
def recommend_cmd(
    car: str,
    track: str | None,
    wing: float | None,
    fuel: float | None,
    quali: bool,
    explore_pct: float,
    detailed: bool,
    reset_mode: bool,
    staged: bool,
    air_temp: float | None,
    track_temp: float | None,
    wind: float | None,
    wetness: float | None,
    pins: tuple[str, ...],
    as_json: bool,
    corpus_root: Path | None,
    no_cache: bool,
    output_file: Path | None,
    physics_mode: bool,
    surrogate_only: bool,
) -> None:
    """Recommend a setup for `<car>` at `<track>`.

    Two invocation forms:

    \b
    * ``optimize <car> <track>`` — explicit pair (e.g. ``optimize bmw sebring``).
    * ``optimize <ibt_path>`` — single positional pointing at an existing
      ``.ibt`` file; car & track are auto-detected from the filename
      (VISION §8 "drop in an IBT, get a setup out").
    """
    car_key, track = _resolve_car_track_or_exit(car, track)
    root = resolve_corpus_root(corpus_root)
    catalog_sessions = _safe_sessions(car_key, corpus_root=root)

    if quali and fuel is None:
        click.echo(
            "--quali requires --fuel <liters>. Pick the fuel load that "
            "covers your quali stint (commonly 5..15 L for 3 laps + "
            "reserve depending on track length).",
            err=True,
        )
        sys.exit(2)
    pinned_overrides = _parse_pins(pins, wing=wing, fuel=fuel)
    # Mode 2 floor pin (PLAN.md Section 14.1): default-pin tyre cold
    # pressure to the per-car constraint floor unless the user
    # explicitly set it. The surrogate cannot see the peak-grip drop
    # from a smaller contact patch and drifts off the floor by
    # several kPa; this default keeps recommendations honest with
    # community-known optimal pressure.
    constraints_for_floor = load_constraints()
    _floor_msg = _apply_tyre_pressure_floor_pin(
        pinned_overrides, constraints_for_floor, car_key,
    )
    pin_info_messages: list[str] = []
    if _floor_msg is not None:
        if as_json:
            pin_info_messages.append(_floor_msg)
        else:
            click.echo(_floor_msg, err=True)
    # Race-mode auto fuel pin: without --quali AND without an explicit
    # --fuel/--pin, anchor fuel to the most-recent past-session value
    # (typically the user's last race load, e.g. 58 L on BMW). The
    # optimizer treats fuel_level_l as a fittable input — leaving it
    # unpinned in race mode would have it freely minimizing mass for
    # one-lap pace and recommending values that won't cover a race
    # distance. Quali mode requires the user to set --fuel explicitly
    # (already enforced above) and skips this auto-pin.
    if not quali and "fuel_level_l" not in pinned_overrides:
        from racingoptimizer.physics.ontology import setup_value
        # Filter to the TARGET track first. Without this, the picker
        # selected the most-recent recorded BMW session across all
        # tracks — and legacy IBT files with no filename datetime
        # default to the YAML's WeekendOptions.Date (currently a
        # future "2026-05-09"), so they win the sort over today's
        # actual on-target Spa sessions. Substring-match the user-
        # typed track against the catalog (so "spa" matches stored
        # slug "spa_2024_up"); fall back to all sessions only if the
        # target track has none. (Legacy GT3 sessions ingested before
        # the GT3-routing removal may still sit in the corpus under
        # the wrong car key; re-ingest with `--reparse` once those
        # IBT files are removed from `ibtfiles/`.)
        target_subset = _filter_to_target_track(catalog_sessions, track)
        if target_subset.height == 0:
            target_subset = catalog_sessions
        past_setup = _most_recent_setup_for(target_subset)
        if past_setup is not None:
            try:
                past_fuel = setup_value(car_key, "fuel_level_l", past_setup)
            except KeyError:
                past_fuel = None
            if past_fuel is not None:
                pinned_overrides["fuel_level_l"] = float(past_fuel)
                fuel_msg = (
                    f"Race fuel auto-pinned to past-session value: "
                    f"{past_fuel:.1f} L (override with --fuel N, or use "
                    f"--quali --fuel N for short stint)."
                )
                if as_json:
                    pin_info_messages.append(fuel_msg)
                else:
                    click.echo(fuel_msg, err=True)
    constraints_table = load_constraints()
    pinned_constraints = _apply_pins_to_constraints(
        constraints_table, car_key, pinned_overrides,
    )

    if reset_mode:
        reset_msg = (
            "RESET MODE -- recommendations diverge sharply from your past "
            "setup; treat as a fresh starting point and verify on track."
        )
        if as_json:
            pin_info_messages.append(reset_msg)
        else:
            click.echo(reset_msg, err=True)

    if car_key in PER_CAR_MODEL_CARS:
        # Per-car path: pool every Cadillac session across every track. The
        # target track is whatever the user asked for; we build a target
        # TrackModel from however many sessions exist on that track (cold-
        # start ok with as few as 1 session) and extract a per-corner
        # archetype schedule. The same per-car PhysicsModel scores every
        # corner via its archetype features.
        track_slug, donor_track, sessions_for_target, schedule, model = (
            _build_per_car_pipeline(
                car_key=car_key,
                track=track,
                catalog_sessions=catalog_sessions,
                root=root,
                no_cache=no_cache,
            )
        )
        sessions_for_fit = catalog_sessions  # env medians come from full corpus
        session_ids = sorted(catalog_sessions["session_id"].to_list())
        env = _env_from_overrides(
            model=model, sessions=sessions_for_target,
            air_temp=air_temp, track_temp=track_temp,
            wind=wind, wetness=wetness,
            corpus_root=root,
        )
        rec = model.recommend(
            track_slug, env, pinned_constraints,
            schedule=schedule, quali=quali, explore_pct=explore_pct,
            reset_mode=reset_mode, staged=staged,
            surrogate_only=surrogate_only,
        )
    else:
        track_slug, donor_track = _resolve_track_or_extrapolate(
            track, catalog_sessions, car_key,
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
        rec = model.recommend(
            fit_track, env, pinned_constraints,
            quali=quali, explore_pct=explore_pct,
            reset_mode=reset_mode, staged=staged,
            surrogate_only=surrogate_only,
        )
        schedule = None  # v3 path: per-(car, track) keying owns the corners

    rec, clamp_warnings, top_warnings = _post_clamp(rec, model, constraints_table)

    recommended_setup = {name: float(val[0]) for name, val in rec.parameters.items()}
    opt_setup: dict[str, float] = dict(model.baseline_setup)
    opt_setup.update(recommended_setup)
    try:
        predicted_readouts = model.predict_setup_readouts(opt_setup, env)
    except AttributeError:
        predicted_readouts = {}
    top_warnings.extend(
        _static_ride_height_envelope_warnings(predicted_readouts),
    )
    top_warnings.extend(
        _heave_slider_tech_warnings(model, recommended_setup, env, schedule),
    )
    from racingoptimizer.physics.score import (
        guardrail_warnings_for_setup,
        headroom_baseline_missing_warning,
    )
    top_warnings.extend(
        guardrail_warnings_for_setup(
            model, recommended_setup, env, schedule or [],
        ),
    )
    head_warn = headroom_baseline_missing_warning(
        model, surrogate_only=surrogate_only,
    )
    if head_warn is not None:
        top_warnings.append(head_warn)
    top_warnings.extend(
        _within_track_variance_warnings(model, track_slug, rec.parameters),
    )
    if pin_info_messages:
        top_warnings = pin_info_messages + top_warnings

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
        schedule=schedule,
    )

    if as_json:
        out = render_recommendation_json(
            rec, model,
            justifications=justifications,
            pinned=pinned_overrides,
            warnings=top_warnings,
            track_display=track_slug,
        )
        rendered = json.dumps(out, indent=2, sort_keys=False)
        click.echo(rendered)
    else:
        # Render briefing + full setup card into a single string, echo it
        # to stdout, and (default) write it to a timestamped file under
        # `recommendations/`. Same content in both places — the file is
        # the artefact the user takes back to the iRacing garage.
        if car_key in PER_CAR_MODEL_CARS:
            # For per-car: prefer the latest setup on the target track if
            # any exists, otherwise the latest across every car session.
            target_sessions = catalog_sessions.filter(
                pl.col("track") == track_slug
            )
            most_recent_setup = _most_recent_setup_for(
                target_sessions if target_sessions.height > 0
                else catalog_sessions
            )
        else:
            most_recent_setup = _most_recent_setup_for(
                sessions_for_fit if donor_track is None
                else _safe_sessions(car_key, corpus_root=root).filter(
                    pl.col("track") == fit_track
                ),
            )
        # Predict deterministic setup readouts (static ride heights) at
        # the recommended setup vector so the card can show the platform
        # state the user will actually see after entering the new
        # perches/pushrods/springs — instead of echoing the past
        # session's stale values.
        opt_setup: dict[str, float] = dict(model.baseline_setup)
        for _name, (_val, _conf) in rec.parameters.items():
            opt_setup[_name] = float(_val)
        if not predicted_readouts:
            try:
                predicted_readouts = model.predict_setup_readouts(opt_setup, env)
            except AttributeError:
                predicted_readouts = {}
        if detailed:
            briefing = render_recommendation_text(
                rec, model,
                justifications=justifications,
                pinned=pinned_overrides,
                warnings=top_warnings,
                track_display=track_slug,
                quali=quali,
            )
        else:
            from racingoptimizer.explain.narrative import render_narrative
            briefing = render_narrative(
                rec, model, justifications,
                most_recent_setup=most_recent_setup,
                track_display=track_slug,
                quali=quali,
                pinned=pinned_overrides,
                warnings=top_warnings,
                schedule=schedule,
            )
        physics_banner = ""
        if physics_mode:
            physics_banner = _render_physics_banner(rec, car_key) + "\n"
        rendered = (
            physics_banner
            + briefing
            + "\n"
            + render_full_setup_card(
                rec, car=car_key, most_recent_setup=most_recent_setup,
                predicted_readouts=predicted_readouts,
            )
        )
        click.echo(rendered)

    # File output. Always written unless the user passed `-` to opt out.
    # Default path encodes the run's identity at a glance:
    # ``recommendations/<car>_<track>_<mode>[_<fuel>L]_<YYYY-MM-DD>_<HHMM>``
    # ``.txt`` (or ``.json`` when --json). Mode tag is one of
    # ``race`` / ``quali`` / ``reset`` so the user can spot the run type
    # without opening the file. Fuel suffix is rendered for quali stints
    # (the load is the user's choice and varies per track); race mode
    # auto-pins fuel and the suffix is omitted to keep names short.
    if as_json and output_file is None:
        # JSON output is intended for piping (jq, etc.). The auto-save
        # banner ``\n[saved to ...]`` would land on stderr and (under
        # Click 8 CliRunner default mix_stderr) bleed into stdout,
        # corrupting downstream JSON parsing. Default to ``-`` (suppress
        # file output) when the user picks JSON without overriding the
        # output path.
        output_file = Path("-")
    if output_file is None:
        from datetime import datetime
        ext = ".json" if as_json else ".txt"
        ts = datetime.now().strftime("%m%d-%H%M")
        if reset_mode:
            mode_tag = "reset"
        elif quali:
            mode_tag = "quali"
        else:
            mode_tag = "race"
        if staged:
            mode_tag = f"{mode_tag}-staged"
        fuel_tag = ""
        if quali and fuel is not None:
            fuel_tag = f"-{int(round(fuel))}L"
        output_file = (
            Path("recommendations")
            / f"{car_key}-{_short_track(track_slug)}-{mode_tag}{fuel_tag}-"
            f"{ts}{ext}"
        )
    if str(output_file) != "-":
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(rendered, encoding="utf-8")
            if not as_json:
                click.echo(f"\n[saved to {output_file}]", err=True)
        except OSError as exc:
            click.echo(
                f"\n[warning: could not write {output_file}: {exc}]",
                err=True,
            )


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

    from racingoptimizer.physics.io_log import load_latest_fit_quality

    coverage: list[TrackCoverage] = []
    fit_quality_trends: list[str] = []
    for track_slug in sorted(set(sessions["track"].to_list())):
        sub = sessions.filter(pl.col("track") == track_slug)
        sids = sorted(sub["session_id"].to_list())
        valid_laps = ingest_api.laps(
            car=car_key, track=track_slug, valid_only=True, corpus_root=root,
        )
        n_clean = _approximate_clean_corner_phases(sids, root)
        regime = _data_density_regime(len(sids), valid_laps.height)

        snapshot = load_latest_fit_quality(
            corpus_root=root, car=car_key, track=track_slug,
        )
        fit_quality_value: float | None = (
            float(snapshot.fit_quality) if snapshot is not None else None
        )
        if snapshot is not None and snapshot.prior_fit_quality is not None:
            delta = snapshot.fit_quality - snapshot.prior_fit_quality
            fit_quality_trends.append(
                f"{track_slug} fit_quality: "
                f"{snapshot.fit_quality:.3f} ({delta:+.3f} vs prior fit)"
            )
        elif snapshot is not None:
            fit_quality_trends.append(
                f"{track_slug} fit_quality: {snapshot.fit_quality:.3f} (first fit)"
            )

        coverage.append(
            TrackCoverage(
                track=track_slug,
                n_sessions=len(sids),
                n_valid_laps=int(valid_laps.height),
                n_clean_corner_phases=int(n_clean),
                fit_quality=fit_quality_value,
                regime=regime,
            )
        )

    overall = _overall_regime(coverage)
    notes = _status_notes()
    notes.extend(fit_quality_trends)

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


def _short_track(slug: str) -> str:
    """Strip variant tokens from a catalog track slug for filename use.

    Examples: ``spa_2024_up`` -> ``spa``, ``hockenheim_gp`` -> ``hockenheim``,
    ``sebring_international`` -> ``sebring``, ``daytona_2011_road`` ->
    ``daytona``. Idempotent for slugs that have no recognised variant.
    """
    return re.sub(r"_(gp|international|road|2\d{3}.*)$", "", slug)


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


def _resolve_car_track_or_exit(
    car_or_path: str, track: str | None
) -> tuple[str, str]:
    """Accept either ``(<car>, <track>)`` or ``(<ibt_path>, None)``.

    VISION §8: "drop in an IBT, get a setup out". When the first positional
    argument points at an existing ``.ibt`` file, sniff car and track from
    the filename (catalog ingestion not required — purely pattern-driven).
    Otherwise treat both args as the canonical ``<car> <track>`` pair.
    """
    candidate = Path(car_or_path)
    looks_like_ibt = (
        track is None
        and candidate.suffix.lower() == ".ibt"
        and candidate.exists()
    )
    if looks_like_ibt:
        raw_car = detect_car_from_filename(candidate.name)
        raw_track = detect_track_from_filename(candidate.name)
        if raw_car is None or raw_track is None:
            click.echo(
                f"could not auto-detect car/track from filename {candidate.name!r}; "
                "pass `<car> <track>` explicitly",
                err=True,
            )
            sys.exit(2)
        try:
            car_key = normalize_car_key(raw_car)
        except UnknownCarError:
            click.echo(
                f"unknown car prefix {raw_car!r} in {candidate.name!r}",
                err=True,
            )
            sys.exit(2)
        return car_key, raw_track
    if track is None:
        click.echo(
            "missing TRACK; usage: `optimize <car> <track>` or `optimize <ibt_path>`",
            err=True,
        )
        sys.exit(2)
    return _resolve_car_or_exit(car_or_path), track


def _filter_to_target_track(
    sessions_df: pl.DataFrame, raw_track: str,
) -> pl.DataFrame:
    """Return rows of `sessions_df` whose `track` matches `raw_track`.

    Mirrors the substring matching used by `_resolve_track_or_extrapolate`
    so user input like "spa" lines up with catalog slugs like
    "spa_2024_up". Returns an empty frame when no track matches; caller
    decides whether to fall back to the full session set.
    """
    available = sorted(set(sessions_df["track"].to_list()))
    match, _ = _match_track_slug(raw_track, available)
    if match is None:
        return sessions_df.head(0)
    return sessions_df.filter(pl.col("track") == match)


def _most_recent_setup_for(sessions_df: pl.DataFrame) -> dict | str | None:
    """Return the parsed setup JSON from the most recently recorded session.

    Used by the full-setup-card renderer so every garage parameter the
    iRacing UI exposes can be shown with a value (the optimizer fills the
    bounded ones, the past setup fills the rest). Returns ``None`` when
    no past setup is ingested — the renderer prints a skip message.
    """
    if sessions_df.height == 0 or "setup" not in sessions_df.columns:
        return None
    # Sort by `recorded_at` (when the session was actually driven) and
    # break ties on `ingested_at` (when it landed in the corpus). The
    # tiebreaker matters when multiple IBTs share the same
    # YAML-declared date — iRacing's `WeekendInfo.WeekendOptions.Date`
    # was previously stored verbatim, and it carries the SCHEDULED
    # race date for series events (identical across an entire weekend).
    # The parser fix uses the per-IBT filename datetime, but already-
    # ingested sessions still carry the bogus weekend date until
    # they're re-ingested; the ingested_at fallback keeps the picker
    # honest in the meantime.
    sort_keys: list[str] = []
    descending: list[bool] = []
    if "recorded_at" in sessions_df.columns:
        sort_keys.append("recorded_at")
        descending.append(True)
    if "ingested_at" in sessions_df.columns:
        sort_keys.append("ingested_at")
        descending.append(True)
    ordered = (
        sessions_df.sort(sort_keys, descending=descending)
        if sort_keys else sessions_df
    )
    raw = ordered["setup"][0]
    return raw if raw else None


def _safe_sessions(car: str, *, corpus_root: Path) -> pl.DataFrame:
    try:
        return ingest_api.sessions(car=car, corpus_root=corpus_root)
    except FileNotFoundError:
        click.echo(
            f"corpus not found at {corpus_root}; run `optimize learn <ibt>` first",
            err=True,
        )
        sys.exit(4)


def _match_track_slug(
    raw_track: str, available: list[str],
) -> tuple[str | None, list[str]]:
    """Resolve a user-typed track string against the available catalog slugs.

    Returns ``(matched_slug, ambiguous_candidates)``:
      - ``(slug, [])`` when the input matches uniquely (exact slug, bare
        alphanum form, or a single substring hit).
      - ``(None, [a, b, ...])`` when the substring scan finds multiple
        plausible candidates — caller decides whether to error.
      - ``(None, [])`` when nothing matches.

    Same rules ingestion applies to IBT filenames (`slugify_track`), so
    ``laguna-seca`` / ``Laguna Seca`` / ``laguna_seca`` collapse to the
    catalog's canonical form, and ``lagunaseca`` matches the un-underscore
    variant via the bare-form pass.
    """
    raw = raw_track.strip().lower()
    needle = slugify_track(raw) or raw
    bare = needle.replace("_", "")
    if needle in available:
        return needle, []
    if bare in available:
        return bare, []
    candidates = sorted({
        slug for slug in available
        if needle in slug or bare in slug.replace("_", "")
    })
    if len(candidates) == 1:
        return candidates[0], []
    if len(candidates) > 1:
        return None, candidates
    return None, []


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

    User input is slugified with the same rules ingestion applies to the IBT
    filename, so ``laguna-seca`` / ``Laguna Seca`` / ``laguna_seca`` all
    collapse to the catalog's canonical ``lagunaseca`` (or ``laguna_seca``)
    slug. Substring matching also tries the un-slugified bare alphanum form
    so ``lagunaseca`` matches a catalog entry stored as ``laguna_seca``.
    """
    available = sorted(set(sessions["track"].to_list()))
    match, ambiguous = _match_track_slug(track, available)
    if ambiguous:
        click.echo(
            f"ambiguous track {track!r}; candidates: {', '.join(ambiguous)}",
            err=True,
        )
        sys.exit(2)
    if match is not None:
        return match, None
    if not available:
        # `_match_track_slug` re-derives `needle` internally; recompute it
        # here for the error message rather than threading it back.
        needle = slugify_track(track.strip().lower()) or track.strip().lower()
        click.echo(
            f"model has no data on ({car_key}, {needle}); "
            f"run `optimize learn <ibt>` first",
            err=True,
        )
        sys.exit(2)
    needle = slugify_track(track.strip().lower()) or track.strip().lower()
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


def _parse_pins(
    pins: tuple[str, ...],
    *,
    wing: float | None,
    fuel: float | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if wing is not None:
        out["rear_wing_angle_deg"] = float(wing)
    if fuel is not None:
        out["fuel_level_l"] = float(fuel)
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


def _apply_tyre_pressure_floor_pin(
    overrides: dict[str, float],
    table: ConstraintsTable,
    car: str,
) -> str | None:
    """Default-pin tyre cold pressure to the per-car constraint floor.

    Per physics-rebuild PLAN.md Section 14.1 (Mode 2): the surrogate model
    rewards platform stability (cleaner ride-height telemetry from higher
    cold pressure) but cannot see the peak-grip drop from a smaller
    contact patch. Recent recommendations on BMW Spa drift to 154-163 kPa
    against the 152 floor; community wisdom is "stay at the floor."

    This helper inserts `tyre_cold_pressure_kpa = <car_floor>` into
    `overrides` IF AND ONLY IF the user did not already set it (any value,
    including the floor itself, counts as user-set). The helper then
    returns a one-line info message describing the insertion, or `None`
    if no insertion happened. The caller (recommend_cmd) prints the
    message to stderr so the user knows the optimizer is honouring the
    floor.

    The override is plumbed through `_apply_pins_to_constraints`, which
    narrows the constraint to `[floor, floor]`; downstream the recommend
    pipeline treats it as a fixed value.

    To override the floor with a higher pressure (e.g. for tyre-warming
    behaviour or wet conditions), pass `--pin tyre_cold_pressure_kpa=N`
    explicitly. Setting it to the floor itself (e.g. `--pin
    tyre_cold_pressure_kpa=152`) is also treated as user-set so the
    helper stays a no-op (no info message).

    Returns the info string, or None if the helper did nothing.
    """
    if "tyre_cold_pressure_kpa" in overrides:
        return None
    bounds = table.bounds(car, "tyre_cold_pressure_kpa")
    if bounds is None:
        # Defensive: no constraint registered for this car. Skip.
        return None
    floor, _hi = bounds
    overrides["tyre_cold_pressure_kpa"] = float(floor)
    return (
        f"Tyre cold pressure auto-pinned to constraint floor: "
        f"{floor:.1f} kPa (override with --pin tyre_cold_pressure_kpa=N)."
    )


def _apply_pins_to_constraints(
    table: ConstraintsTable,
    car: str,
    pins: dict[str, float],
) -> ConstraintsTable:
    """Return a constraints table with pinned parameters narrowed to [v, v]."""
    if not pins:
        return table
    known = set(table.parameters())
    for key in pins:
        if key not in known:
            click.echo(
                f"unknown --pin parameter {key!r}; "
                f"known parameters: {', '.join(sorted(known))}",
                err=True,
            )
            sys.exit(2)
    for key, value in pins.items():
        # Narrowest legal interval -- recommender treats it as a fixed value.
        original = table.bounds(car, key)
        if original is not None:
            value = max(original[0], min(original[1], float(value)))
        table = table.with_pin(car, key, float(value))
    return table


def _build_per_car_pipeline(
    *,
    car_key: str,
    track: str,
    catalog_sessions: pl.DataFrame,
    root: Path,
    no_cache: bool = False,
) -> tuple[str, str | None, pl.DataFrame, list, object]:
    """Build the five artefacts a per-car (v4) recommend run needs.

    Returns ``(target_track_slug, donor_track_slug, sessions_for_target,
    schedule, model)``.

    ``donor_track_slug`` is set when the requested track has no sessions
    for this car (and no cross-car geometry exists) but the car has been
    driven elsewhere — the corner schedule is borrowed from the donor
    track while the per-car model still pools all sessions.

    * ``track_slug``: normalised target track slug (matches catalog naming).
    * ``sessions_for_target``: subset of ``catalog_sessions`` filtered to the
      target track. Used for env-median computation. May be empty when the
      target track has zero sessions for this car (true cold-start).
    * ``schedule``: list of ``CornerScheduleEntry`` for the target track,
      built from valid laps on the target track. The recommender feeds
      these archetype features into the per-car PhysicsModel at predict
      time so the same fitter scores any track.
    * ``model``: per-car ``PhysicsModel`` trained on EVERY session of
      ``car_key`` across every track. Cached on disk under
      ``corpus/models/<car>__per-car__<digest>.pickle``.

    A target track with zero sessions raises a CLI error with the standard
    untrained-track message — there's nothing to extract a schedule from
    yet. (Donor extrapolation is replaced by the per-car path itself: the
    model already pools data across tracks; we just need at least one valid
    lap on the target to know its corner geometry.)
    """
    from racingoptimizer.physics.corner_schedule import build_corner_schedule

    donor_track: str | None = None

    # Track slug normalisation: same rules as the per-(car, track) path.
    available = sorted(set(catalog_sessions["track"].to_list()))
    track_slug, ambiguous = _match_track_slug(track, available)
    if ambiguous:
        click.echo(
            f"ambiguous track {track!r}; candidates: {', '.join(ambiguous)}",
            err=True,
        )
        sys.exit(2)

    if track_slug is None:
        # Per-car has no sessions on the requested track. Before bailing,
        # see whether ANY OTHER car has been driven there — the corner
        # schedule is pure track geometry (braking / apex / exit
        # positions + archetype features), so borrowing BMW@Spa's
        # schedule to score a Ferrari@Spa setup is sensible. The
        # per-car FITTER still trains on Ferrari sessions only; only
        # the corner schedule is borrowed.
        track_slug, sessions_for_target = _maybe_borrow_cross_car_track(
            track, car_key, root,
        )
        if track_slug is None:
            if not available:
                click.echo(
                    f"per-car {car_key} has no sessions on track {track!r}, "
                    f"and no other car has either; "
                    f"available for {car_key}: (none). "
                    f"Run `optimize learn <ibt>` to ingest a session on this "
                    f"track first.",
                    err=True,
                )
                sys.exit(2)
            needle = slugify_track(track.strip().lower()) or track.strip().lower()
            counts = (
                catalog_sessions.group_by("track")
                .agg(pl.len().alias("n"))
                .sort(["n", "track"], descending=[True, False])
            )
            donor_track = str(counts["track"][0])
            track_slug = needle
            sessions_for_target = catalog_sessions.filter(
                pl.col("track") == donor_track,
            )
    else:
        sessions_for_target = catalog_sessions.filter(
            pl.col("track") == track_slug,
        )
    target_sids = sorted(sessions_for_target["session_id"].to_list())
    schedule = build_corner_schedule(target_sids, corpus_root=root)
    if not schedule:
        click.echo(
            f"could not extract a corner schedule for ({car_key}, {track_slug}); "
            f"target sessions had no detectable corners on any valid lap.",
            err=True,
        )
        sys.exit(2)

    pooled_sids = sorted(catalog_sessions["session_id"].to_list())
    model = _build_or_load_per_car_model(
        car_key, pooled_sids, root, no_cache=no_cache,
    )
    return track_slug, donor_track, sessions_for_target, schedule, model


def _maybe_borrow_cross_car_track(
    track: str, requested_car: str, root: Path,
) -> tuple[str | None, pl.DataFrame]:
    """Find a track slug + session list from ANY car on the requested track.

    Used when the per-car path needs a corner schedule for a track the
    requested car has never been driven on (e.g. Ferrari@Spa with no
    Ferrari Spa IBTs but BMW + Cadillac have plenty). Returns
    ``(track_slug, sessions_df)`` keyed to the *donor* car's sessions,
    or ``(None, empty_df)`` if no car has been driven on that track.

    The schedule built from these sessions feeds the per-car fitter at
    predict time as `corner_archetype` features — corner geometry, not
    car-specific physics. The fitter itself still trains exclusively on
    `requested_car` sessions.
    """
    for other_car in CANONICAL_CARS:
        if other_car == requested_car:
            continue
        other_sessions = _safe_sessions(other_car, corpus_root=root)
        if other_sessions.is_empty():
            continue
        other_tracks = sorted(set(other_sessions["track"].to_list()))
        # Cross-car borrow: silently skip ambiguity (the next car may have
        # a clean match). _match_track_slug returns (None, [...]) when the
        # substring scan finds multiple candidates; treat that as "no
        # confident match here, try the next car."
        match, _ambiguous = _match_track_slug(track, other_tracks)
        if match is not None:
            return match, other_sessions.filter(pl.col("track") == match)
    return None, pl.DataFrame()


def _build_or_load_per_car_model(
    car: str,
    session_ids: list[str],
    root: Path,
    *,
    no_cache: bool = False,
):
    """Per-car (v4) PhysicsModel cache layer.

    Cache key folds the SET of pooled session ids + ontology fingerprint +
    feature-schema version. Adding a new session on any track invalidates
    the cache (all sessions contribute to the per-car fit).
    """
    cache_path = _per_car_model_cache_path(root, car, session_ids)
    if not no_cache and cache_path.exists():
        try:
            with cache_path.open("rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass

    from racingoptimizer.physics import InsufficientDataError
    from racingoptimizer.physics.fitter import fit_per_car

    k_folds = 5 if len(session_ids) >= 3 else 2
    try:
        model = fit_per_car(car, session_ids, corpus_root=root, k_folds=k_folds)
    except InsufficientDataError as exc:
        click.echo(
            f"insufficient training data for per-car {car}: {exc}",
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


def _per_car_model_cache_path(
    root: Path, car: str, session_ids: list[str],
) -> Path:
    """Cache path for the per-car PhysicsModel. Mirrors `_model_cache_path`
    semantics but keys on ``per-car`` instead of a track."""
    from racingoptimizer.physics.fitter import (
        ENV_FEATURE_SCHEMA_VERSION_PER_CAR,
    )

    parts = _model_cache_parts(car, session_ids)
    parts.append(f"schema={int(ENV_FEATURE_SCHEMA_VERSION_PER_CAR)}")
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]
    return root / "models" / f"{car}__per-car__{digest}.pickle"


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

    # Pass `car` so the track model picks the per-car curb-agreement
    # threshold (Acura's heave/roll signal needs a lower fraction than the
    # four-corner default; see racingoptimizer.track.masks).
    track_model = build_track_model(track, session_ids, corpus_root=root, car=car)
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
    """Cache path keyed by session ids + ontology + constraints + fitters layout + feature schema.

    Each component matters for pickle validity:
    * Session ids — adding a session = new training data.
    * Ontology fingerprint — per-car parameter set + each spec's
      `(family, fittable, user_settable)` triple. Flipping a CE-gated
      entry invalidates the pickle.
    * Constraints content — bounds in `constraints.md` are baked into
      the model at fit time; editing them must invalidate the pickle so
      DE doesn't search against stale bounds.
    * Fitters package layout version — class adds/renames/module-moves
      under `physics.fitters` would otherwise leave a valid digest
      pointing at a pickle that fails to revive
      (`ModuleNotFoundError`).
    * Feature-schema version — pre-S2.2 (v1), S2.2 env-12 (v2),
      Stage-3 coupled (v3).
    """
    from racingoptimizer.physics.fitter import ENV_FEATURE_SCHEMA_VERSION

    parts = _model_cache_parts(car, session_ids)
    parts.append(f"schema={int(ENV_FEATURE_SCHEMA_VERSION)}")
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]
    return root / "models" / f"{car}__{track}__{digest}.pickle"


def _model_cache_parts(car: str, session_ids: list[str]) -> list[str]:
    """Cache key components shared by both per-(car, track) and per-car paths.

    Folds session ids, ontology fingerprint, constraints.md content, and
    the fitters-package layout version. Caller appends any path-specific
    suffix (per-track schema vs per-car schema) before hashing.

    Ontology fingerprint includes ``json_path`` because a path
    correction (e.g. fuel_level_l moving from Chassis.Fuel to
    BrakesDriveUnit.Fuel) changes which YAML field the fitter pulls
    its training values from — without folding the path, a stale
    pickle reuses the OLD training data even though the ontology now
    reads a different leaf, masking the fix as a no-op.
    """
    from racingoptimizer.physics.fitters import FITTERS_LAYOUT_VERSION
    from racingoptimizer.physics.ontology import ontology_for

    parts = ["|".join(sorted(session_ids))]
    onto = ontology_for(car)
    onto_fingerprint = "|".join(
        f"{name}:{spec.family}:{int(spec.fittable)}:"
        f"{int(spec.user_settable)}:path={'.'.join(spec.json_path)}"
        for name, spec in sorted(onto.items())
    )
    parts.append(f"onto={onto_fingerprint}")
    parts.append(f"constraints={_constraints_fingerprint()}")
    parts.append(f"fitters_layout={FITTERS_LAYOUT_VERSION}")
    return parts


def _constraints_fingerprint() -> str:
    """Stable digest of the active `constraints.md` file content.

    Hashed at cache-key build time so editing the file (e.g. tightening
    a per-car spring bound after a UI capture) invalidates every
    pre-existing pickle that was fit against the old bounds.
    """
    try:
        from racingoptimizer.constraints.loader import _default_constraints_path
        path = _default_constraints_path()
        if not path.is_file():
            return "missing"
        return hashlib.sha256(
            path.read_bytes()
        ).hexdigest()[:16]
    except Exception:
        # Constraints file unreadable — fall back to a sentinel rather
        # than crash the cache lookup. Caller will refit on the next
        # run when the file becomes readable.
        return "unreadable"


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
# Sourced from the canonical IBT_*_CHANNELS tuples in
# `racingoptimizer.context.environment` so the parser, the corner
# aggregator, and the corpus-median override path stay in lockstep. Wind
# direction is filtered out of the float dict because circular medians
# cannot pool with arithmetic medians (handled separately via
# `_WIND_DIR_CHANNEL`).
_WIND_DIR_CHANNEL = "WindDir"
_ENV_FLOAT_CHANNELS: dict[str, str] = {
    ibt: field
    for ibt, field in IBT_FLOAT_CHANNELS
    if ibt != _WIND_DIR_CHANNEL
}
# Discrete env channels (bool / int). Aggregated via .max() — any-wet wins.
_ENV_DISCRETE_CHANNELS: dict[str, str] = dict(
    IBT_BOOL_CHANNELS + IBT_INT_CHANNELS
)
# Backward-compat alias used by tests.
_ENV_CHANNELS = _ENV_FLOAT_CHANNELS

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

    Discrete parameters (`ParameterSpec.is_discrete=True`, e.g. ARB blade
    index, damper clicks) are rounded to the nearest integer after
    clamping. The DE search runs continuously over `[lo, hi]`, so without
    this round step the briefing emits values like "anti_roll_bar_front:
    3.700" — a value the user cannot enter into the iRacing garage UI.
    """
    from racingoptimizer.physics.ontology import ontology_for

    try:
        onto = ontology_for(model.car)
    except KeyError:
        onto = {}

    clamp_warnings: dict[str, str] = {}
    top_warnings: list[str] = []
    pinned = tuple(getattr(rec, "pinned_to_observed_median", ()) or ())
    if pinned:
        top_warnings.append(
            "pinned to observed median (no per-session variation in training "
            "corpus, no learnable response surface): " + ", ".join(pinned)
        )
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
        clamped_value = float(result.value)
        spec = onto.get(name)
        if spec is not None and spec.is_discrete:
            rounded = float(round(clamped_value))
            # Re-clamp the rounded value back inside the legal range —
            # `round(0.4)` from a `[1, 5]` bound would otherwise emit 0.
            lo, hi = result.bound
            rounded = min(max(rounded, lo), hi)
            if rounded != clamped_value:
                clamp_warnings[name] = (
                    f"discrete-click value rounded from {clamped_value:.3f} "
                    f"to {int(rounded)} (legal range "
                    f"{int(lo)}..{int(hi)})"
                )
            clamped_value = rounded
        if result.was_clamped and name not in clamp_warnings:
            clamp_warnings[name] = (
                f"value clamped from {value:.3f} to {clamped_value:.3f} "
                f"(legal bounds: {result.bound[0]:.3f} - {result.bound[1]:.3f})"
            )
        new_params[name] = (clamped_value, confidence)
    return replace(rec, parameters=new_params), clamp_warnings, top_warnings


def _within_track_variance_warnings(
    model,
    track: str,
    parameters: dict,
) -> list[str]:
    """Warn when a recommendation extrapolates with zero within-track variance."""
    per_track = getattr(model, "per_track_parameter_observed", {}) or {}
    track_obs = per_track.get(track, {})
    warnings: list[str] = []
    for name, (value, _conf) in parameters.items():
        observed = track_obs.get(name)
        if not observed:
            continue
        unique = {float(v) for v in observed}
        if len(unique) > 1:
            continue
        only = next(iter(unique))
        rec_val = float(value)
        spec_step = 1.0
        try:
            from racingoptimizer.physics.ontology import ontology_for
            spec = ontology_for(model.car).get(name)
            if spec is not None and spec.step:
                spec_step = float(spec.step)
        except KeyError:
            pass
        if abs(rec_val - only) <= spec_step * 0.5:
            continue
        warnings.append(
            f"{name}: recommended {rec_val:.4g} but only {only:.4g} observed "
            f"at {track} (zero within-track variance) -- cross-track surrogate "
            "may be extrapolating; verify on track."
        )
    return warnings


def _static_ride_height_envelope_warnings(
    predicted_readouts: dict[str, float],
) -> list[str]:
    """Warn when predicted static ride heights fall outside observation envelopes.

    `constraints.md` static RH rows are not DE feasibility constraints;
    this check surfaces likely garage readout mismatches before the user
    enters values in iRacing.
    """
    if not predicted_readouts:
        return []
    _FRONT_BOUNDS = (30.0, 80.0)
    _REAR_BOUNDS = (30.0, 80.0)
    _CHECKS: tuple[tuple[str, str, tuple[float, float]], ...] = (
        ("setup_static_lf_ride_height_mm", "LF static ride height", _FRONT_BOUNDS),
        ("setup_static_rf_ride_height_mm", "RF static ride height", _FRONT_BOUNDS),
        ("setup_static_lr_ride_height_mm", "LR static ride height", _REAR_BOUNDS),
        ("setup_static_rr_ride_height_mm", "RR static ride height", _REAR_BOUNDS),
    )
    warnings: list[str] = []
    for channel, label, (lo, hi) in _CHECKS:
        value = predicted_readouts.get(channel)
        if value is None:
            continue
        rh = float(value)
        if rh < lo or rh > hi:
            warnings.append(
                f"Predicted {label} {rh:.1f} mm outside observation envelope "
                f"({lo:.0f}-{hi:.0f} mm) -- verify platform inputs before running."
            )
    return warnings


def _heave_slider_tech_warnings(
    model,
    setup: dict[str, float],
    env,
    schedule: list | None,
) -> list[str]:
    """Warn when predicted front heave slider deflection exceeds 45 mm (iRacing tech)."""
    if schedule is None or int(getattr(model, "feature_schema_version", 0)) < 4:
        return []
    from racingoptimizer.corner import CornerPhaseKey, Phase

    _TECH_LIMIT_MM = 45.0
    for entry in schedule:
        if str(entry.phase).strip().lower() != "mid_corner":
            continue
        cpkey = CornerPhaseKey(
            session_id="<tech-check>",
            lap_index=0,
            corner_id=int(entry.corner_id),
            phase=Phase.MID_CORNER,
        )
        try:
            state = model.predict(
                setup, env, cpkey, corner_archetype=entry.archetype,
            )
        except (ValueError, KeyError):
            continue
        for channel in ("heave_slider_mm", "front_heave_slider_mm"):
            conf = state.states.get(channel)
            if conf is None:
                continue
            value = float(conf.value)
            if value > _TECH_LIMIT_MM:
                return [
                    "Predicted front heave slider deflection "
                    f"{value:.1f} mm exceeds 45 mm tech limit "
                    f"(corner T{entry.corner_id} mid-corner) -- "
                    "soften front platform or raise perch before running."
                ]
    return []


def _render_physics_banner(rec, car: str) -> str:
    """Render the --physics-mode banner shown above the briefing.

    Surfaces guardrail-style warnings derived from the Week 2 evaluator
    + axle-grip-margin model:
      * Recommended tyre pressure pinned to constraint floor (Mode 2 win)
      * Per-car bayes posteriors loaded (Mode 1 status)
      * Recommended setup deltas vs corpus median (Mode 3 + 4 status)

    The banner is INFORMATIONAL only -- recommendation values are
    unchanged. The intent is to give the user a "physics view" of the
    recommendation without altering the optimizer's behaviour.
    """
    lines = ["=" * 72, "PHYSICS VIEW (--physics flag)", "=" * 72]
    # Mode 2: tyre pressure floor pin status.
    if "tyre_cold_pressure_kpa" in rec.parameters:
        tp_value, _conf = rec.parameters["tyre_cold_pressure_kpa"]
        lines.append(
            f"  tyre_cold_pressure_kpa: {tp_value:.1f} kPa "
            f"(constraint floor pinned, Mode 2)"
        )
    # Mode 5 prep: surface axle-grip-margin reference values per car.
    try:
        from racingoptimizer.physics.diagnostic_state import get_car_geometry
        geom = get_car_geometry(car)
        lines.append(
            f"  car geometry: wheelbase={geom.wheelbase_m:.2f} m, "
            f"weight_dist_front={geom.weight_distribution:.2f}"
        )
    except KeyError:
        pass
    # Per-car physics weights (Day 13 calibration).
    try:
        from racingoptimizer.physics.evaluator import get_weights_for_car
        wu, wb, wh = get_weights_for_car(car)
        lines.append(
            f"  per-car evaluator weights (util/balance/headroom): "
            f"({wu:.1f}, {wb:.1f}, {wh:.1f})"
        )
    except Exception:
        pass
    lines.append(
        "  Note: default DE uses hybrid physics+surrogate scoring "
        "(Day 13). Pass --surrogate-only to revert to surrogate-only."
    )
    lines.append(
        "  recommendation values are unchanged; this banner is informational."
    )
    lines.append("=" * 72)
    return "\n".join(lines)


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


def _data_density_regime(n_sessions: int, n_laps: int) -> str:
    """Classify a (car, track) pair by how much TRAINING DATA exists for it.

    This is a capacity heuristic — "do we have enough laps to even try?"
    — and is distinct from `Confidence.regime`, which classifies the
    fitter's residual-vs-signal ratio at predict time. The status
    command surfaces this density regime per track; the briefing's
    per-parameter `[confidence: ...]` tag uses `Confidence.regime`. They
    can disagree (a track with hundreds of laps but flat input variance
    looks `dense` here and `sparse` to the fitter), and that's OK — they
    answer different questions.
    """
    if n_laps == 0 or n_sessions == 0:
        return "sparse"
    if n_sessions >= 4 and n_laps >= 100:
        return "dense"
    if n_sessions >= 2 and n_laps >= 30:
        return "confident"
    if n_sessions >= 2:
        return "noisy"
    return "sparse"


# Back-compat alias — older imports referenced this name. Remove once
# downstream tooling is updated.
_coverage_regime = _data_density_regime


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
        "constraints.md is missing verified ontology paths for: brake_ducts "
        "and some per-car blocked garage leaves.",
    ]


__all__ = ["compare_cmd", "recommend_cmd", "status_cmd"]
