"""Full-setup card: every garage parameter, grouped by panel, ready to enter.

The per-parameter "engineering briefing" in :mod:`render_text` justifies
each click the optimizer recommends -- corners helped, hurts, +/-1-click
sensitivity, evidence. Useful for understanding WHY a value was picked,
but the user has asked for a complementary view: a flat readable list of
EVERY garage parameter for the car, in the order they appear in the iRacing
garage UI, ready to be entered.

This module produces that view by walking the most-recent ingested setup
JSON for the (car, track) combination and tagging each leaf with a source:

* ``[OPT]``        -- optimizer-recommended value, post-clamp.
* ``[OPT pin]``    -- optimizer pinned to observed median (no signal to deviate).
* ``[OPT mirror]`` -- value mirrored from the per-axle parameter (e.g.
                     iRacing requires LR=RR rear spring rate). The
                     optimizer trains the parameter once; the renderer
                     mirrors it onto the symmetric corner.
* ``[past]``       -- value from your most-recent session (no constraint
                     bounds to optimize against yet).
* ``[readout]``    -- calculated by iRacing; you don't enter this. Past
                     session value, listed for reference.
* ``[predicted]``  -- calculated readout the optimizer's setup-readout
                     fitter projects under the new inputs. This is what
                     iRacing's calculator will show after you enter the
                     ``[OPT]`` values.

VISION §7 gates this output: every USER-set parameter must surface a value
the driver can actually type. Calculated readouts must NOT be presented as
"set this" -- they're informational only.
"""
from __future__ import annotations

import json
import re
from typing import Any

from racingoptimizer.physics.ontology import ParameterSpec, ontology_for
from racingoptimizer.physics.recommendation import SetupRecommendation

# Pulls a leading signed float out of YAML strings like "30 N/mm" or "-22.5 mm".
# Used by `_scalar_from_yaml` so the renderer can compute past->opt deltas
# against numeric values regardless of whether the YAML stored them as
# strings (the iRacing setup blob format) or pre-coerced floats.
_LEADING_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Calculated readouts iRacing's UI displays as outputs. The driver cannot
# type these into the garage. Identified by the leaf field name in the
# setup JSON. Confirmed against the Cadillac V-Series.R YAML; the same leaf
# names appear on every GTP car that exposes the parameter.
_CALCULATED_LEAF_NAMES: frozenset[str] = frozenset({
    # Suspension deflections (current/max strings like "8.8 mm 97.7 mm").
    "HeaveSpringDefl", "HeaveSliderDefl", "HeaveDamperDefl",
    "ThirdSpringDefl", "ThirdSliderDefl", "ShockDefl", "SpringDefl",
    "TorsionBarDefl",
    # Ride height -- output of perch offsets / pushrod lengths / TB turns.
    "RideHeight",
    # Per-corner weight -- output of the corner-weight balancing process.
    "CornerWeight",
    # Cross weight -- derived from corner weights.
    "CrossWeight",
    # Tire feedback -- last run readouts.
    "LastHotPressure", "LastTempsOMI", "LastTempsIMO", "TreadRemaining",
    # AeroCalculator block -- entirely a calculator.
    "DownforceBalance", "FrontRhAtSpeed", "RearRhAtSpeed", "LD",
    # Gear ratios -- derived from GearStack selection.
    "SpeedInFirst", "SpeedInSecond", "SpeedInThird", "SpeedInFourth",
    "SpeedInFifth", "SpeedInSixth", "SpeedInSeventh",
})


# Top-level YAML keys -> garage panel display name. Iteration order here
# defines section order in the rendered card.
#
# `Dampers` and `Systems` are Ferrari/Acura/Porsche tops; BMW/Cadillac
# carry the equivalent fields under `BrakesDriveUnit`. Listing all of
# them here is harmless when a car doesn't have the YAML key (the
# walk-block call returns an empty list and the panel is skipped).
_PANELS: tuple[tuple[str, str], ...] = (
    ("TiresAero", "TIRES & AERO"),
    ("Chassis", "CHASSIS"),
    ("Dampers", "DAMPERS"),
    ("BrakesDriveUnit", "BRAKES / DRIVETRAIN"),
    ("Systems", "SYSTEMS"),
)


# Setup leaves whose iRacing UI value must mirror another corner.
# {target_path: (source_path, parameter_name)} -- when the renderer walks
# `target_path`, it pulls the OPT value from `source_path`'s recommendation
# and tags `[OPT mirror]`. Source param entries that don't exist for the
# car at hand silently no-op (e.g. Ferrari has no rear coil spring, so
# the SpringRate row resolves to source_match=None and falls through).
#
# Coverage:
# * Rear coil spring rate (LR -> RR): all cars that have a rear coil.
# * Front torsion bar turns + OD (LF -> RF): Cadillac, BMW, Ferrari.
# * Rear torsion bar turns + OD (LR -> RR): Ferrari (4-corner torsion).
#
# iRacing's UI keeps left/right pairs of these coupled; the optimizer
# only trains the left side and the renderer mirrors so the user sees
# matching values to enter.
_MIRRORED_LEAVES: dict[tuple[str, ...], tuple[tuple[str, ...], str]] = {
    ("Chassis", "RightRear", "SpringRate"): (
        ("Chassis", "LeftRear", "SpringRate"),
        "rear_coil_spring_rate_n_per_mm",
    ),
    ("Chassis", "RightFront", "TorsionBarTurns"): (
        ("Chassis", "LeftFront", "TorsionBarTurns"),
        "torsion_bar_turns_fl",
    ),
    ("Chassis", "RightFront", "TorsionBarOD"): (
        ("Chassis", "LeftFront", "TorsionBarOD"),
        "torsion_bar_od_fl_mm",
    ),
    ("Chassis", "RightRear", "TorsionBarTurns"): (
        ("Chassis", "LeftRear", "TorsionBarTurns"),
        "torsion_bar_turns_rl",
    ),
    ("Chassis", "RightRear", "TorsionBarOD"): (
        ("Chassis", "LeftRear", "TorsionBarOD"),
        "torsion_bar_od_rl_mm",
    ),
    ("Chassis", "RightRear", "ToeIn"): (
        ("Chassis", "LeftRear", "ToeIn"),
        "toe_rl_mm",
    ),
}


# Calculated readouts the optimizer's setup-readout fitter can predict at
# the recommended setup vector. Maps the YAML leaf path -> the model's
# output channel name. When a prediction is available, the renderer shows
# `[predicted]` with the projected value (and notes the past readout in
# parentheses if it differs); otherwise it falls back to `[readout]`
# echoing the past-session YAML value.
_PREDICTED_READOUT_PATHS: dict[tuple[str, ...], str] = {
    ("Chassis", "LeftFront", "RideHeight"): "setup_static_lf_ride_height_mm",
    ("Chassis", "RightFront", "RideHeight"): "setup_static_rf_ride_height_mm",
    ("Chassis", "LeftRear", "RideHeight"): "setup_static_lr_ride_height_mm",
    ("Chassis", "RightRear", "RideHeight"): "setup_static_rr_ride_height_mm",
}


def _humanize_leaf(name: str) -> str:
    """Turn a YAML leaf key into a friendly garage label.

    ``"HeaveSpring"`` -> ``"Heave Spring"``; ``"LFshockVel"`` stays mostly
    intact. Heuristic: insert a space before each capital letter unless
    preceded by another capital (preserves acronyms like ``LF``).
    """
    out: list[str] = []
    for i, ch in enumerate(name):
        if i > 0 and ch.isupper() and not name[i - 1].isupper():
            out.append(" ")
        out.append(ch)
    return "".join(out)


def _is_leaf_calculated(leaf_name: str) -> bool:
    return leaf_name in _CALCULATED_LEAF_NAMES


def _format_value(value: Any) -> str:
    """Render an arbitrary YAML value as a single readable string."""
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        # Integers print without decimal noise; floats with one decimal.
        if isinstance(value, int) or value == int(value):
            return f"{int(value)}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _round_to_step(value: float, step: float | None) -> float:
    """Snap ``value`` to the nearest ``step`` increment.

    The iRacing garage UI exposes parameters as discrete clicks (e.g. spring
    rates step in 5 N/mm, perch offsets in 0.5 mm). Outputting a continuous
    optimizer value like ``26.438 N/mm`` is unenterable -- the driver can
    only set ``25`` or ``30``. Rounding here keeps the recommendation in
    the legal value lattice without changing the search itself.

    ``step is None`` returns the value unchanged.
    """
    if step is None or step <= 0.0:
        return value
    return round(value / step) * step


def _snap_to_discrete(value: float, choices: tuple[float, ...]) -> float:
    """Snap ``value`` to the nearest entry in ``choices`` (non-uniform list).

    Used for parameters whose iRacing UI exposes a fixed list of values
    that don't follow a uniform step -- torsion bar OD's 14 diameters
    being the canonical example. Returns the closest legal value;
    distance ties resolve to the lower entry (Python ``min`` is stable).
    """
    return min(choices, key=lambda c: abs(c - value))


def _format_opt_value(
    value: float,
    spec_step: float | None,
    units: str,
    *,
    discrete_values: tuple[float, ...] | None = None,
    choices: tuple[str, ...] | None = None,
) -> str:
    """Render an optimizer-recommended value as a single user-enterable string.

    Three rendering modes:

    * ``choices`` set -> categorical parameter; round ``value`` to the
      nearest valid index and emit the corresponding label (no units).
    * ``discrete_values`` set -> numeric parameter with non-uniform legal
      values (e.g. torsion bar OD); snap to the closest entry and format
      with the same precision as the surrounding step would imply.
    * Otherwise -> uniform-step rounding via ``_round_to_step``.
    """
    if choices:
        idx = max(0, min(len(choices) - 1, int(round(value))))
        return choices[idx]
    if discrete_values:
        snapped = _snap_to_discrete(value, discrete_values)
        # Render integer-valued discrete sets without trailing zeros
        # (clutch plates -> "6", not "6.00") and float-valued sets with
        # 2 decimals (torsion bar OD -> "14.34", "17.94").
        if all(v == int(v) for v in discrete_values):
            body = f"{int(round(snapped))}"
        else:
            body = f"{snapped:.2f}"
        return f"{body} {units}".rstrip()
    snapped = _round_to_step(value, spec_step)
    if spec_step is None or spec_step >= 1.0:
        body = f"{snapped:.0f}"
    elif spec_step >= 0.1:
        body = f"{snapped:.1f}"
    else:
        body = f"{snapped:.2f}"
    return f"{body} {units}".rstrip()


def _scalar_from_yaml(raw: Any) -> float | None:
    """Pull a single float out of the YAML setup blob.

    YAML leaf values are usually strings like ``"30 N/mm"`` or ``"-22.5 mm"``;
    pyyaml does not auto-coerce these. We accept the leading number and drop
    the unit so the renderer can compute deltas against the optimizer's
    raw float. Returns ``None`` when no number can be parsed.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str):
        m = _LEADING_NUM_RE.search(raw)
        if m is not None:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


def _ontology_path_index(rec: SetupRecommendation, car: str) -> dict[
    tuple[str, ...], tuple[str, float, ParameterSpec]
]:
    """Build ``{json_path: (parameter_name, optimizer_value, spec)}`` for the
    parameters the optimizer reported a value for.

    The renderer uses this to splice the optimizer's recommendation into the
    ingested-setup walk: when a leaf's path matches a recommended parameter,
    we replace the past value with the optimizer's, round to the garage
    click step, append units, and emit the past->opt delta -- all of which
    require the spec.
    """
    try:
        onto = ontology_for(car)
    except KeyError:
        return {}
    index: dict[tuple[str, ...], tuple[str, float, ParameterSpec]] = {}
    for name, (value, _conf) in rec.parameters.items():
        spec = onto.get(name)
        if spec is None:
            continue
        if not spec.user_settable:
            # Optimizer should never produce values for non-settable params,
            # but if it ever did (legacy pickle, future bug), the renderer
            # must not present them as "set this".
            continue
        index[spec.json_path] = (name, float(value), spec)
    return index


def render_full_setup_card(
    rec: SetupRecommendation,
    *,
    car: str,
    most_recent_setup: dict | str | None,
    predicted_readouts: dict[str, float] | None = None,
) -> str:
    """Render every garage parameter as a single readable block.

    ``most_recent_setup`` is the parsed YAML setup blob from the user's most
    recent ingested session for this (car, track) combination. The card
    walks that structure so it covers exactly the parameters that exist in
    the iRacing UI for this car (no fictitious entries, no missing ones).

    ``predicted_readouts`` maps the model's setup-readout channel name
    (e.g. ``setup_static_lf_ride_height_mm``) to the value the trained
    fitter projects at the optimizer's recommended setup vector. Used to
    render ``[predicted]`` static ride heights in place of the stale past
    ``[readout]`` values. Optional -- when omitted (or for a channel the
    model didn't carry) the card falls back to the past YAML value with
    the ``[readout]`` tag.

    Returns a multi-line string; caller is responsible for emitting it
    (e.g. via ``click.echo``).
    """
    if most_recent_setup is None:
        return (
            "FULL SETUP CARD: skipped -- no past setup ingested for "
            f"({car}, {rec.track}). Run `optimize learn <ibt>` first.\n"
        )
    if isinstance(most_recent_setup, str):
        try:
            setup = json.loads(most_recent_setup)
        except json.JSONDecodeError:
            return "FULL SETUP CARD: skipped -- past setup blob is unparseable.\n"
    else:
        setup = most_recent_setup
    if not isinstance(setup, dict):
        return "FULL SETUP CARD: skipped -- past setup blob is not an object.\n"

    pinned = set(getattr(rec, "pinned_to_observed_median", ()) or ())
    opt_index = _ontology_path_index(rec, car)
    readouts = dict(predicted_readouts or {})

    lines: list[str] = []
    lines.append("=" * 64)
    lines.append(f"FULL SETUP CARD -- {car} @ {rec.track}")
    lines.append("=" * 64)
    lines.append(
        "Legend: [OPT] optimizer recommendation | [OPT pin] pinned to "
        "observed median | [OPT mirror] mirrored from per-axle parameter "
        "| [past] from your most recent session (no bounds yet) | "
        "[readout] calculated by iRacing -- verify, don't enter | "
        "[predicted] readout the optimizer projects under the new inputs."
    )

    for top_key, panel_label in _PANELS:
        block = setup.get(top_key)
        if not isinstance(block, dict):
            continue
        rendered = _render_panel(
            panel_label, top_key, block, opt_index, pinned, readouts,
        )
        if rendered:
            lines.append("")
            lines.extend(rendered)

    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_panel(
    panel_label: str,
    top_key: str,
    block: dict,
    opt_index: dict[tuple[str, ...], tuple[str, float, ParameterSpec]],
    pinned: set[str],
    predicted_readouts: dict[str, float],
) -> list[str]:
    """Walk one top-level garage panel and emit its rows.

    Emits one line per leaf with the format:

        Heave Spring          25 N/mm    (was 30)  [OPT]

    where the value is rounded to the iRacing garage click step and the
    parenthetical shows the past value when it differs (so the user can
    spot what actually changed). Calculated readouts pass through with
    their raw YAML string and the ``[readout]`` tag -- unless the model
    has a prediction for the new setup, in which case ``[predicted]``
    overrides with the projected value.
    """
    lines: list[str] = [f"-- {panel_label} " + "-" * (60 - len(panel_label))]
    rows = list(_walk_block(block, prefix=(top_key,)))
    if not rows:
        return []
    # Group rows by their immediate sub-section (e.g. "LeftFront", "Front")
    # for readability. Emit a sub-heading whenever the second path segment
    # changes.
    last_subkey: str | None = None
    for path, leaf_name, value in rows:
        # path looks like ('Chassis', 'LeftFront', 'RideHeight').
        subkey = path[1] if len(path) >= 2 else "(root)"
        if subkey != last_subkey:
            lines.append(f"  [{subkey}]")
            last_subkey = subkey

        opt_match = opt_index.get(path)
        mirror_source = _MIRRORED_LEAVES.get(path)
        predicted_channel = _PREDICTED_READOUT_PATHS.get(path)
        is_calc = _is_leaf_calculated(leaf_name)

        delta_note = ""
        if is_calc:
            # Try the predicted-readout fitter first -- gives the user
            # the platform state they'll see after entering the new
            # setup, instead of echoing last session's stale value.
            predicted_val: float | None = None
            if predicted_channel is not None:
                predicted_val = predicted_readouts.get(predicted_channel)
            if predicted_val is not None:
                tag = "[predicted]"
                # Static ride heights render with one decimal + mm units;
                # match the past-readout YAML format ("30.5 mm") for
                # visual consistency.
                displayed = f"{predicted_val:.1f} mm"
                past_scalar = _scalar_from_yaml(value)
                if past_scalar is not None and abs(predicted_val - past_scalar) >= 0.05:
                    delta_note = f" (was {past_scalar:.1f})"
            else:
                tag = "[readout]"
                displayed = _format_value(value)
        elif opt_match is not None:
            param_name, opt_val, spec = opt_match
            tag = "[OPT pin]" if param_name in pinned else "[OPT]"
            spec_discrete = getattr(spec, "discrete_values", None)
            spec_choices = getattr(spec, "choices", None)
            displayed = _format_opt_value(
                opt_val, spec.step, spec.units,
                discrete_values=spec_discrete, choices=spec_choices,
            )
            past_scalar = _scalar_from_yaml(value)
            if past_scalar is not None and not spec_choices:
                if spec_discrete:
                    snapped = _snap_to_discrete(opt_val, spec_discrete)
                    threshold = 1e-6
                else:
                    snapped = _round_to_step(opt_val, spec.step)
                    threshold = max(spec.step or 0.0, 1e-6) / 2
                if abs(snapped - past_scalar) >= threshold:
                    delta_note = f" (was {_format_value(past_scalar)})"
            elif spec_choices:
                # Categorical: compare label to past YAML string directly.
                idx = max(0, min(len(spec_choices) - 1, int(round(opt_val))))
                opt_label = spec_choices[idx]
                past_label = (
                    str(value).strip() if value is not None else ""
                )
                if past_label and past_label != opt_label:
                    delta_note = f" (was {past_label})"
        elif mirror_source is not None:
            source_path, _param_name = mirror_source
            source_match = opt_index.get(source_path)
            if source_match is not None:
                _src_param, src_val, src_spec = source_match
                tag = "[OPT mirror]"
                src_discrete = getattr(src_spec, "discrete_values", None)
                src_choices = getattr(src_spec, "choices", None)
                displayed = _format_opt_value(
                    src_val, src_spec.step, src_spec.units,
                    discrete_values=src_discrete, choices=src_choices,
                )
                past_scalar = _scalar_from_yaml(value)
                if past_scalar is not None:
                    snapped = _round_to_step(src_val, src_spec.step)
                    if abs(snapped - past_scalar) >= max(
                        src_spec.step or 0.0, 1e-6,
                    ) / 2:
                        delta_note = f" (was {_format_value(past_scalar)})"
            else:
                tag = "[past]"
                displayed = _format_value(value)
        else:
            tag = "[past]"
            displayed = _format_value(value)

        label = _humanize_leaf(leaf_name)
        # Pad label to 28 chars, displayed value to 18, tag at the end.
        # Delta note (if any) sits between value and tag so the eye lands
        # on the OLD value the user is currently running.
        lines.append(
            f"    {label:<28} {displayed:>18}{delta_note:<14}  {tag}"
        )
    return lines


def _walk_block(
    block: dict, *, prefix: tuple[str, ...],
) -> list[tuple[tuple[str, ...], str, Any]]:
    """Flatten a nested setup block into ``(path, leaf_name, value)`` rows.

    Order is the dict insertion order from the YAML parse -- which mirrors
    the iRacing garage panel layout closely enough for readability.
    """
    out: list[tuple[tuple[str, ...], str, Any]] = []
    for key, value in block.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            out.extend(_walk_block(value, prefix=path))
        else:
            out.append((path, key, value))
    return out


__all__ = ["render_full_setup_card"]
