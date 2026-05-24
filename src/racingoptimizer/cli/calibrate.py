"""`optimize calibrate <car> <track>` -- propose probes that grow corpus coverage.

The recommender can only learn slopes from variance the driver has actually
produced. Many parameters get pinned because every session ran them at the
same value (e.g. front torsion bar OD never moved off 15.1 mm). This command:

* Default mode: picks the N parameters with the thinnest observed variance
  on the target track and proposes a value in unsampled territory for each.
  After the user runs a few laps with those changes, re-fitting widens the
  fitter's evidence base for those parameters.
* ``--status`` mode: prints the per-parameter coverage table without
  proposing changes.

The output is read-only against the corpus -- no model retrain, no DE search.
We just inspect ``model.per_track_parameter_observed[track]`` and the
constraint envelope.
"""
from __future__ import annotations

import math
from pathlib import Path

import click
import polars as pl

from racingoptimizer.constraints import (
    ConstraintsTable,
    load_constraints,
)
from racingoptimizer.explain.full_setup_card import render_full_setup_card
from racingoptimizer.ingest.paths import resolve_corpus_root
from racingoptimizer.physics.ontology import (
    fittable_parameters,
    ontology_for,
    setup_value,
)

# `ConstraintsTable.bounds(car, param)` returns ``tuple[float, float] | None``.
ParameterBound = tuple[float, float]


# Top-N default. Three is small enough that the user can drive a clean
# stint with the changes and feel each one, but big enough that one
# session meaningfully grows coverage.
_DEFAULT_TARGETS: int = 3

# Parameters with N >= this are considered "covered" for the headline
# completeness percentage. Two distinct values gives the fitter a slope
# but no curvature; three is the threshold where the regression is
# robust to a single noisy session.
_COVERED_THRESHOLD: int = 3


def _coverage_pct(
    observed: tuple[float, ...],
    bound: ParameterBound | None,
) -> float:
    """Fraction of the constraint envelope spanned by observed values.

    Returns 0.0 when only one (or zero) values are observed, OR when the
    constraint envelope is unbounded (no bound recorded yet). Categorical
    parameters with discrete choices report coverage as
    ``len(distinct) / len(choices)``.
    """
    if not observed:
        return 0.0
    if len(observed) == 1:
        return 0.0
    if bound is None:
        return 0.0
    span = bound[1] - bound[0]
    if span <= 0.0:
        return 0.0
    rng = max(observed) - min(observed)
    return min(1.0, rng / span)


def _largest_gap(
    observed: tuple[float, ...],
    bound: ParameterBound,
) -> tuple[float, float, float]:
    """Find the largest unsampled interval inside the constraint envelope.

    Returns ``(gap_lo, gap_hi, midpoint)`` where ``midpoint`` is the
    value to propose. If observed is empty, returns the constraint span
    centred on its midpoint. Includes the pre-min and post-max regions
    as candidate gaps so an under-explored extreme can be picked.
    """
    lo, hi = bound
    if not observed:
        return (lo, hi, (lo + hi) / 2.0)
    sorted_obs = sorted(set(observed))
    edges = [lo, *sorted_obs, hi]
    best_gap = (lo, hi, 0.0)
    best_width = -math.inf
    for left, right in zip(edges, edges[1:], strict=False):
        width = right - left
        if width > best_width:
            best_width = width
            best_gap = (left, right, (left + right) / 2.0)
    return best_gap


def _snap_to_step(
    value: float,
    *,
    step: float | None,
    discrete: tuple[float, ...] = (),
    bound: ParameterBound,
) -> float:
    """Round a proposal to the iRacing UI's user-typeable resolution."""
    if discrete:
        return min(discrete, key=lambda c: abs(c - value))
    if step is not None and step > 0.0:
        lo, hi = bound
        # Anchor to the constraint low so steps line up with the legal
        # grid (e.g. heave rates land on 60, 70, 80 not 65, 75).
        n_steps = round((value - lo) / step)
        snapped = lo + n_steps * step
        return float(min(hi, max(lo, snapped)))
    return float(value)


def _formatted(value: float, *, step: float | None, units: str) -> str:
    """Render a number at the precision implied by its UI step."""
    suffix = f" {units}" if units else ""
    s = step if step is not None else 0.0
    if s >= 1.0:
        return f"{value:.0f}{suffix}"
    if s >= 0.1:
        return f"{value:.1f}{suffix}"
    if s >= 0.01:
        return f"{value:.2f}{suffix}"
    return f"{value:.3f}{suffix}"


@click.command(name="calibrate")
@click.argument("car")
@click.argument("track")
@click.option(
    "--status", "status_only", is_flag=True, default=False,
    help=(
        "Print the per-parameter coverage table only -- no probe proposal "
        "and no setup card."
    ),
)
@click.option(
    "--targets", "n_targets", type=int, default=_DEFAULT_TARGETS,
    show_default=True,
    help=(
        "How many thin-variance parameters to perturb. Each chosen "
        "parameter gets one proposal in the largest unsampled gap of "
        "its constraint envelope; the past setup is preserved everywhere "
        "else. Higher values teach the fitter more per session at the "
        "cost of confounded changes -- keep this small."
    ),
)
@click.option(
    "--corpus-root", type=click.Path(path_type=Path), default=None,
    help="Override corpus location.",
)
@click.option(
    "--output-file", type=click.Path(path_type=Path), default=None,
    help=(
        "Write the briefing + (default mode only) the calibration "
        "setup card to this file. Defaults to "
        "`recommendations/<car>_<track>_calibrate[_status]_<YYYY-MM-DD>_"
        "<HHMM>.txt`. Pass `-` to disable file output."
    ),
)
def calibrate_cmd(
    car: str,
    track: str,
    status_only: bool,
    n_targets: int,
    corpus_root: Path | None,
    output_file: Path | None,
) -> None:
    """Probe under-explored parameters to grow the model's coverage.

    \b
    Default: proposes ``--targets`` (3) thin-variance changes on top of
    your most recent setup, prints a coverage table, prints "What this
    teaches", and renders a full setup card you can take to the garage.
    Drive a clean stint with the new values, re-ingest, then re-run
    ``optimize <car> <track>`` to feed the fresh variance into the fit.

    \b
    With ``--status``: only the coverage table is printed, useful for
    deciding whether the model is already saturated or whether more
    variance is the bottleneck on accuracy.
    """
    # Local imports defer the heavy CLI module so `optimize calibrate
    # --help` stays fast.
    from racingoptimizer.cli.recommend import (
        PER_CAR_MODEL_CARS,
        _build_or_load_model,
        _build_per_car_pipeline,
        _filter_to_target_track,
        _most_recent_setup_for,
        _resolve_car_track_or_exit,
        _resolve_track_or_extrapolate,
        _safe_sessions,
    )

    car_key, track_input = _resolve_car_track_or_exit(car, track)
    root = resolve_corpus_root(corpus_root)
    catalog_sessions = _safe_sessions(car_key, corpus_root=root)
    if catalog_sessions.height == 0:
        click.echo(
            f"no sessions for {car_key}; run `optimize learn` first.", err=True,
        )
        raise SystemExit(2)

    if car_key in PER_CAR_MODEL_CARS:
        track_slug, _donor, sessions_for_target, _schedule, model = (
            _build_per_car_pipeline(
                car_key=car_key,
                track=track_input,
                catalog_sessions=catalog_sessions,
                root=root,
                no_cache=False,
            )
        )
    else:
        track_slug, donor_track = _resolve_track_or_extrapolate(
            track_input, catalog_sessions, car_key,
        )
        fit_track = donor_track or track_slug
        sessions_for_fit = catalog_sessions.filter(pl.col("track") == fit_track)
        sessions_for_target = sessions_for_fit
        session_ids = sorted(sessions_for_fit["session_id"].to_list())
        model = _build_or_load_model(
            car_key, fit_track, session_ids, root, no_cache=False,
        )

    constraints = load_constraints()
    onto = ontology_for(car_key)
    fittables = fittable_parameters(car_key, constraints)

    target_subset = _filter_to_target_track(catalog_sessions, track_slug)
    if target_subset.height == 0:
        target_subset = sessions_for_target
    most_recent_setup = _most_recent_setup_for(target_subset)

    per_track = getattr(model, "per_track_parameter_observed", {}) or {}
    track_observed = per_track.get(track_slug, {})

    rows = _build_coverage_rows(
        fittables=fittables,
        ontology=onto,
        constraints=constraints,
        car_key=car_key,
        track_observed=track_observed,
    )
    rows.sort(key=lambda r: (r["n_distinct"], r["coverage_pct"], r["name"]))

    n_total = len(rows)
    n_pinned = sum(1 for r in rows if r["n_distinct"] <= 1)
    n_covered = sum(1 for r in rows if r["n_distinct"] >= _COVERED_THRESHOLD)
    n_sessions = int(target_subset.height)

    out_lines: list[str] = []
    out_lines.append("=" * 72)
    out_lines.append(
        f" {car_key} @ {track_slug} -- calibration probe "
        f"(N={n_sessions} sessions)"
    )
    out_lines.append(
        f" Coverage: {n_covered}/{n_total} parameters with N>={_COVERED_THRESHOLD} "
        f"distinct values  ({100.0 * n_covered / max(n_total, 1):.0f}% explored, "
        f"{n_pinned} pinned)"
    )
    out_lines.append("=" * 72)
    out_lines.append("")
    out_lines.append(_render_coverage_table(rows))
    out_lines.append("")

    if status_only:
        rendered = "\n".join(out_lines)
        click.echo(rendered)
        _maybe_save(rendered, output_file, car_key, track_slug, status=True)
        return

    targets = _pick_targets(
        rows=rows, ontology=onto, constraints=constraints, car_key=car_key,
        n_targets=n_targets, past_setup=most_recent_setup,
    )

    if not targets:
        out_lines.append(
            "No fittable parameters with thin variance found -- the model "
            "is already well-calibrated on this track.",
        )
        rendered = "\n".join(out_lines)
        click.echo(rendered)
        _maybe_save(rendered, output_file, car_key, track_slug, status=False)
        return

    out_lines.append("CALIBRATION TARGETS")
    out_lines.append("")
    for idx, t in enumerate(targets, start=1):
        out_lines.extend(_render_target_block(idx, t))
    out_lines.append("")
    out_lines.append("WHAT THIS TEACHES")
    out_lines.append(
        "  Drive a clean stint (5+ laps) with these values entered in the "
        "iRacing garage,",
    )
    out_lines.append(
        "  then run `optimize learn` and re-fit. Each probe gives the "
        "fitter a fresh distinct",
    )
    out_lines.append(
        "  value to learn from -- pinned parameters become fittable, slopes "
        "with two samples",
    )
    out_lines.append(
        "  gain a third anchor, and the recommender stops short-circuiting "
        "to the corpus median.",
    )
    out_lines.append("")
    if most_recent_setup is None:
        out_lines.append(
            "(No past setup found; setup card not rendered.)",
        )
    else:
        # Build a synthetic recommendation: most-recent setup with the
        # probe values overwritten. Reuses the full_setup_card renderer
        # via a minimal stub.
        out_lines.append(
            _render_calibration_card(
                car_key=car_key,
                track_slug=track_slug,
                model=model,
                past_setup=most_recent_setup,
                targets=targets,
                fittables=fittables,
                ontology=onto,
            ),
        )

    rendered = "\n".join(out_lines)
    click.echo(rendered)
    _maybe_save(rendered, output_file, car_key, track_slug, status=False)


def _maybe_save(
    rendered: str,
    output_file: Path | None,
    car_key: str,
    track_slug: str,
    *,
    status: bool,
) -> None:
    """Write the briefing to `recommendations/...` unless the caller opted out.

    Default filename: ``<car>-<short-track>-cal[-status]-<MMDD>-<HHMM>.txt``.
    Pass ``-`` as ``output_file`` to skip file output (e.g. when piping).
    """
    if str(output_file) == "-":
        return
    if output_file is None:
        from datetime import datetime

        from racingoptimizer.cli.recommend import _short_track

        ts = datetime.now().strftime("%m%d-%H%M")
        suffix = "cal-status" if status else "cal"
        output_file = (
            Path("recommendations")
            / f"{car_key}-{_short_track(track_slug)}-{suffix}-{ts}.txt"
        )
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(rendered, encoding="utf-8")
        click.echo(f"\n[saved to {output_file}]", err=True)
    except OSError as exc:
        click.echo(
            f"\n[warning: could not write {output_file}: {exc}]",
            err=True,
        )


def _build_coverage_rows(
    *,
    fittables: list[str],
    ontology: dict,
    constraints: ConstraintsTable,
    car_key: str,
    track_observed: dict[str, tuple[float, ...]],
) -> list[dict]:
    rows: list[dict] = []
    for name in fittables:
        spec = ontology[name]
        bound = constraints.bounds(car_key, name)
        observed = tuple(track_observed.get(name, ()))
        rows.append({
            "name": name,
            "spec": spec,
            "bound": bound,
            "observed": observed,
            "n_distinct": len(set(observed)),
            "coverage_pct": _coverage_pct(observed, bound),
        })
    return rows


def _render_coverage_table(rows: list[dict]) -> str:
    """ASCII coverage table sorted thinnest-first."""
    if not rows:
        return "(no fittable parameters with constraints recorded yet)"
    header = f"{'PARAMETER':<34} {'N':>3}  {'RANGE':<26} {'COVERAGE':>8}"
    lines = [header, "-" * len(header)]
    for r in rows:
        observed = r["observed"]
        bound = r["bound"]
        spec = r["spec"]
        units = spec.units or ""
        if observed:
            lo = min(observed)
            hi = max(observed)
            if r["n_distinct"] == 1:
                rng_text = f"{_formatted(lo, step=spec.step, units=units)} (single)"
            else:
                rng_text = (
                    f"{_formatted(lo, step=spec.step, units='')} .. "
                    f"{_formatted(hi, step=spec.step, units=units)}"
                )
        else:
            rng_text = "(none observed)"
        cov_pct = r["coverage_pct"] * 100.0
        cov_text = f"{cov_pct:>5.0f}%" if bound else "  n/a"
        name = r["name"]
        if len(name) > 34:
            name = name[:31] + "..."
        lines.append(
            f"{name:<34} {r['n_distinct']:>3}  {rng_text:<26} {cov_text:>8}"
        )
    return "\n".join(lines)


def _pick_targets(
    *,
    rows: list[dict],
    ontology: dict,
    constraints: ConstraintsTable,
    car_key: str,
    n_targets: int,
    past_setup: dict | str | None,
) -> list[dict]:
    """Choose the top-N thin-variance parameters and build a probe per row.

    Filters out:

    * parameters with no constraint envelope (untrained, can't propose);
    * parameters where the largest gap is below their UI step (already
      sampled densely enough to learn from);
    * parameters whose proposed value would equal the past value after
      step-snapping (the iRacing garage wouldn't accept it as a change).
    """
    targets: list[dict] = []
    for row in rows:
        if len(targets) >= max(0, n_targets):
            break
        if row["bound"] is None:
            continue
        spec = row["spec"]
        bound = row["bound"]
        step = spec.step if spec.step is not None else 0.0
        gap_lo, gap_hi, midpoint = _largest_gap(row["observed"], bound)
        if gap_hi - gap_lo <= max(step, 1e-9):
            continue
        proposal = _snap_to_step(
            midpoint,
            step=spec.step,
            discrete=tuple(spec.discrete_values or ()),
            bound=bound,
        )
        lo, hi = bound
        past_value: float | None = None
        if past_setup is not None:
            try:
                past_value = setup_value(car_key, row["name"], past_setup)
            except KeyError:
                past_value = None
        epsilon = max(step * 0.5, 1e-9)
        if past_value is not None and abs(proposal - past_value) < epsilon:
            # Snapping landed on the past value. Try the next-largest
            # gap by rebuilding edges with the proposal removed.
            sorted_obs = sorted({*row["observed"], proposal})
            edges = [lo, *sorted_obs, hi]
            second = max(
                ((right - left, (left + right) / 2.0)
                 for left, right in zip(edges, edges[1:], strict=False)
                 if right - left > max(step, 1e-9)),
                default=(0.0, proposal),
            )
            proposal = _snap_to_step(
                second[1], step=spec.step,
                discrete=tuple(spec.discrete_values or ()),
                bound=bound,
            )
            if past_value is not None and abs(proposal - past_value) < epsilon:
                continue
        targets.append({
            **row,
            "past_value": past_value,
            "proposal": proposal,
            "gap": (gap_lo, gap_hi),
        })
    return targets


def _render_target_block(idx: int, t: dict) -> list[str]:
    spec = t["spec"]
    bound = t["bound"]
    units = spec.units or ""
    past = t["past_value"]
    proposal = t["proposal"]
    name = t["name"].replace("_", " ")
    past_text = (
        _formatted(past, step=spec.step, units=units) if past is not None
        else "(unknown)"
    )
    new_text = _formatted(proposal, step=spec.step, units=units)
    n = t["n_distinct"]
    if n == 0:
        density_note = "no observations on this track yet"
    elif n == 1:
        density_note = "only 1 distinct value driven so far"
    elif n == 2:
        density_note = "2 distinct values -- gives a slope but no curvature"
    else:
        density_note = f"{n} distinct values"
    span = bound[1] - bound[0]
    gap = t["gap"][1] - t["gap"][0]
    gap_pct = (gap / span * 100.0) if span > 0 else 0.0
    lines = [
        f"{idx}. {name}: {past_text} -> {new_text}",
        f"   {density_note}; probing into a "
        f"{gap_pct:.0f}% unsampled gap of the legal range.",
    ]
    return lines


def _render_calibration_card(
    *,
    car_key: str,
    track_slug: str,
    model,
    past_setup: dict | str,
    targets: list[dict],
    fittables: list[str],
    ontology: dict,
) -> str:
    """Render a setup card with the past setup + probe overrides applied.

    The full_setup_card renderer expects a SetupRecommendation. We
    construct one inline with every parameter pinned to the past value
    (tag ``[OPT pin]``) except the probes (tag ``[OPT]``), which carry
    the proposed values.
    """
    from racingoptimizer.confidence import Confidence
    from racingoptimizer.context import EnvironmentFrame
    from racingoptimizer.physics.recommendation import SetupRecommendation

    parameters: dict[str, tuple[float, Confidence]] = {}
    target_names = {t["name"]: t for t in targets}
    pinned_conf = Confidence(
        value=0.0, lo=0.0, hi=0.0, n_samples=0, regime="sparse",
    )
    probe_conf = Confidence(
        value=0.0, lo=0.0, hi=0.0, n_samples=0, regime="noisy",
    )
    pinned_names: list[str] = []
    for name in fittables:
        spec = ontology[name]
        if name in target_names:
            parameters[name] = (
                float(target_names[name]["proposal"]), probe_conf,
            )
            continue
        if not spec.user_settable:
            continue
        try:
            past_val = setup_value(car_key, name, past_setup)
        except KeyError:
            past_val = None
        if past_val is None:
            continue
        parameters[name] = (float(past_val), pinned_conf)
        pinned_names.append(name)
    rec = SetupRecommendation(
        car=car_key,
        track=track_slug,
        env=EnvironmentFrame(),
        parameters=parameters,
        score_breakdown={},
        untrained_parameters=(),
        aero_correction_available=False,
        pinned_to_observed_median=tuple(pinned_names),
    )
    return render_full_setup_card(
        rec, car=car_key, most_recent_setup=past_setup,
        predicted_readouts={},
    )
