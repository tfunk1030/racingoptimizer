"""Full-setup card: every garage parameter, grouped by panel, ready to enter.

The per-parameter "engineering briefing" in :mod:`render_text` justifies
each click the optimizer recommends — corners helped, hurts, ±1-click
sensitivity, evidence. Useful for understanding WHY a value was picked,
but the user has asked for a complementary view: a flat readable list of
EVERY garage parameter for the car, in the order they appear in the iRacing
garage UI, ready to be entered.

This module produces that view by walking the most-recent ingested setup
JSON for the (car, track) combination and tagging each leaf with a source:

* ``[OPT]``       — optimizer-recommended value, post-clamp.
* ``[OPT pin]``   — optimizer pinned to observed median (no signal to deviate).
* ``[past]``      — value from your most-recent session (no constraint bounds
                    to optimize against yet).
* ``[readout]``   — calculated by iRacing; you don't enter this. Listed so
                    you can verify the resulting platform state matches.

VISION §7 gates this output: every USER-set parameter must surface a value
the driver can actually type. Calculated readouts must NOT be presented as
"set this" — they're informational only.
"""
from __future__ import annotations

import json
from typing import Any

from racingoptimizer.physics.ontology import ontology_for
from racingoptimizer.physics.recommendation import SetupRecommendation

# Calculated readouts iRacing's UI displays as outputs. The driver cannot
# type these into the garage. Identified by the leaf field name in the
# setup JSON. Confirmed against the Cadillac V-Series.R YAML; the same leaf
# names appear on every GTP car that exposes the parameter.
_CALCULATED_LEAF_NAMES: frozenset[str] = frozenset({
    # Suspension deflections (current/max strings like "8.8 mm 97.7 mm").
    "HeaveSpringDefl", "HeaveSliderDefl", "HeaveDamperDefl",
    "ThirdSpringDefl", "ThirdSliderDefl", "ShockDefl", "SpringDefl",
    "TorsionBarDefl",
    # Ride height — output of perch offsets / pushrod lengths / TB turns.
    "RideHeight",
    # Per-corner weight — output of the corner-weight balancing process.
    "CornerWeight",
    # Cross weight — derived from corner weights.
    "CrossWeight",
    # Tire feedback — last run readouts.
    "LastHotPressure", "LastTempsOMI", "LastTempsIMO", "TreadRemaining",
    # AeroCalculator block — entirely a calculator.
    "DownforceBalance", "FrontRhAtSpeed", "RearRhAtSpeed", "LD",
    # Gear ratios — derived from GearStack selection.
    "SpeedInFirst", "SpeedInSecond", "SpeedInThird", "SpeedInFourth",
    "SpeedInFifth", "SpeedInSixth", "SpeedInSeventh",
})


# Top-level YAML keys → garage panel display name. Iteration order here
# defines section order in the rendered card.
_PANELS: tuple[tuple[str, str], ...] = (
    ("TiresAero", "TIRES & AERO"),
    ("Chassis", "CHASSIS"),
    ("BrakesDriveUnit", "BRAKES / DRIVETRAIN"),
)


def _humanize_leaf(name: str) -> str:
    """Turn a YAML leaf key into a friendly garage label.

    ``"HeaveSpring"`` → ``"Heave Spring"``; ``"LFshockVel"`` stays mostly
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
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        # Integers print without decimal noise; floats with one decimal.
        if isinstance(value, int) or value == int(value):
            return f"{int(value)}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _ontology_path_index(rec: SetupRecommendation, car: str) -> dict[
    tuple[str, ...], tuple[str, float]
]:
    """Build ``{json_path: (parameter_name, optimizer_value)}`` for the
    parameters the optimizer reported a value for.

    The renderer uses this to splice the optimizer's recommendation into the
    ingested-setup walk: when a leaf's path matches a recommended parameter,
    we replace the past value with the optimizer's and tag it ``[OPT]``.
    """
    try:
        onto = ontology_for(car)
    except KeyError:
        return {}
    index: dict[tuple[str, ...], tuple[str, float]] = {}
    for name, (value, _conf) in rec.parameters.items():
        spec = onto.get(name)
        if spec is None:
            continue
        if not spec.user_settable:
            # Optimizer should never produce values for non-settable params,
            # but if it ever did (legacy pickle, future bug), the renderer
            # must not present them as "set this".
            continue
        index[spec.json_path] = (name, float(value))
    return index


def render_full_setup_card(
    rec: SetupRecommendation,
    *,
    car: str,
    most_recent_setup: dict | str | None,
) -> str:
    """Render every garage parameter as a single readable block.

    ``most_recent_setup`` is the parsed YAML setup blob from the user's most
    recent ingested session for this (car, track) combination. The card
    walks that structure so it covers exactly the parameters that exist in
    the iRacing UI for this car (no fictitious entries, no missing ones).

    Returns a multi-line string; caller is responsible for emitting it
    (e.g. via ``click.echo``).
    """
    if most_recent_setup is None:
        return (
            "FULL SETUP CARD: skipped — no past setup ingested for "
            f"({car}, {rec.track}). Run `optimize learn <ibt>` first.\n"
        )
    if isinstance(most_recent_setup, str):
        try:
            setup = json.loads(most_recent_setup)
        except json.JSONDecodeError:
            return "FULL SETUP CARD: skipped — past setup blob is unparseable.\n"
    else:
        setup = most_recent_setup
    if not isinstance(setup, dict):
        return "FULL SETUP CARD: skipped — past setup blob is not an object.\n"

    pinned = set(getattr(rec, "pinned_to_observed_median", ()) or ())
    opt_index = _ontology_path_index(rec, car)

    lines: list[str] = []
    lines.append("=" * 64)
    lines.append(f"FULL SETUP CARD — {car} @ {rec.track}")
    lines.append("=" * 64)
    lines.append(
        "Legend: [OPT] optimizer recommendation · [OPT pin] pinned to "
        "observed median · [past] from your most recent session "
        "(no bounds yet) · [readout] calculated by iRacing — verify, "
        "don't enter."
    )

    for top_key, panel_label in _PANELS:
        block = setup.get(top_key)
        if not isinstance(block, dict):
            continue
        rendered = _render_panel(
            panel_label, top_key, block, opt_index, pinned,
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
    opt_index: dict[tuple[str, ...], tuple[str, float]],
    pinned: set[str],
) -> list[str]:
    """Walk one top-level garage panel and emit its rows."""
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
        is_calc = _is_leaf_calculated(leaf_name)

        if is_calc:
            tag = "[readout]"
            displayed = _format_value(value)
        elif opt_match is not None:
            param_name, opt_val = opt_match
            tag = "[OPT pin]" if param_name in pinned else "[OPT]"
            displayed = _format_value(opt_val)
        else:
            tag = "[past]"
            displayed = _format_value(value)

        label = _humanize_leaf(leaf_name)
        # Pad label to 28 chars, value to 18, tag right-aligned.
        lines.append(f"    {label:<28} {displayed:>18}  {tag}")
    return lines


def _walk_block(
    block: dict, *, prefix: tuple[str, ...],
) -> list[tuple[tuple[str, ...], str, Any]]:
    """Flatten a nested setup block into ``(path, leaf_name, value)`` rows.

    Order is the dict insertion order from the YAML parse — which mirrors
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
