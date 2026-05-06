"""Plain-English narrative renderer for setup recommendations.

Default rendering for `optimize <car> <track>` since 2026-05-06. Replaces
the per-parameter ±1-click + score-delta blocks with a 2-3 line summary
per change, plus an OVERALL DIRECTION header. Pass `--detailed` to bring
back the legacy block-per-parameter format from
``explain.render_text.render_recommendation_text``.

The narrative re-uses the existing ``SetupJustification`` dataclass
(``corners_helped`` / ``corners_hurt`` lists) so no new physics work is
needed — only translation: corner-phase + parameter family →
engineering English.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from racingoptimizer.corner import Phase
from racingoptimizer.explain.justification import (
    CornerPhaseImpact,
    SetupJustification,
)
from racingoptimizer.physics.ontology import ParameterSpec, ontology_for

if TYPE_CHECKING:
    from racingoptimizer.physics import PhysicsModel, SetupRecommendation


# ---------------------------------------------------------------------------
# Translation tables
# ---------------------------------------------------------------------------

_PHASE_LABEL: dict[Phase, str] = {
    Phase.BRAKING: "under braking",
    Phase.TRAIL_BRAKE: "on trail-braking",
    Phase.MID_CORNER: "mid-corner",
    Phase.EXIT: "on power exit",
    Phase.STRAIGHT: "on straights",
}


# Per-family theme + (verb when value INCREASES, verb when value DECREASES).
_FAMILY_THEME: dict[str, tuple[str, str, str]] = {
    "spring_rate":   ("platform stiffness",          "stiffens",        "softens"),
    "perch_offset":  ("static ride height",          "lowers car",      "raises car"),
    "pushrod":       ("static ride height",          "raises car",      "lowers car"),
    "heave_spring":  ("front heave compliance",      "stiffens",        "softens"),
    "heave_slider":  ("front heave compliance",      "stiffens",        "softens"),
    "damper":        ("damping rate",                "stiffens",        "softens"),
    "rear_wing":     ("downforce vs drag",           "more downforce",  "less drag"),
    "front_wing":    ("front downforce",             "more front",      "less front"),
    "tyre_pressure": ("tire pressure",               "raises",          "lowers"),
    "arb":           ("anti-roll stiffness",         "stiffens",        "softens"),
    "ride_height":   ("static ride height",          "raises",          "lowers"),
    "camber":        ("camber",                      "more negative",   "less negative"),
    "torsion_bar":   ("front platform stiffness",    "stiffens",        "softens"),
    "brake_bias":    ("brake bias",                  "moves forward",   "moves rearward"),
    "diff":          ("diff lockup",                 "more locked",     "freer"),
    "fuel":          ("fuel mass",                   "heavier",         "lighter"),
    "corner_weight": ("corner weight target",        "raises",          "lowers"),
}


# Per-parameter human label override.
_PARAM_LABEL: dict[str, str] = {
    "rear_wing_angle_deg":            "Rear wing",
    "tyre_cold_pressure_kpa":         "Tyre cold pressure",
    "heave_spring_rate_n_per_mm":     "Front heave spring",
    "third_spring_rate_n_per_mm":     "Rear third spring",
    "rear_coil_spring_rate_n_per_mm": "Rear coil spring",
    "heave_perch_offset_front_mm":    "Front heave perch",
    "spring_perch_offset_rear_mm":    "Rear spring perch",
    "third_perch_offset_rear_mm":     "Rear third perch",
    "pushrod_length_offset_front_mm": "Front pushrod offset",
    "pushrod_length_offset_rear_mm":  "Rear pushrod offset",
    "fuel_level_l":                   "Fuel level",
    "diff_preload_nm":                "Rear diff preload",
    "front_diff_preload_nm":          "Front diff preload",
    "diff_coast_drive_ramps":         "Diff coast/drive ramps",
    "diff_clutch_friction_plates":    "Diff clutch plates",
    "brake_bias_pct":                 "Brake bias",
    "anti_roll_bar_front":            "Front ARB blade",
    "anti_roll_bar_rear":             "Rear ARB blade",
    "arb_size_front":                 "Front ARB size",
    "arb_size_rear":                  "Rear ARB size",
    "toe_front_mm":                   "Front toe",
    "toe_rl_mm":                      "Rear toe",
    "torsion_bar_turns_fl":           "Front torsion bar turns",
    "torsion_bar_od_fl_mm":           "Front torsion bar OD",
    "torsion_bar_turns_rl":           "Rear torsion bar turns",
    "torsion_bar_od_rl_mm":           "Rear torsion bar OD",
    "camber_fl_deg":                  "FL camber",
    "camber_fr_deg":                  "FR camber",
    "camber_rl_deg":                  "RL camber",
    "camber_rr_deg":                  "RR camber",
}


# Group ordering for the briefing.
_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("AERO", ("rear_wing", "front_wing", "tyre_pressure")),
    (
        "PLATFORM (springs / perches / pushrods / torsion bars)",
        (
            "spring_rate", "heave_spring", "perch_offset", "pushrod",
            "torsion_bar", "ride_height", "fuel",
        ),
    ),
    ("DAMPERS", ("damper",)),
    ("BALANCE (ARBs / camber / toe)", ("arb", "camber")),
    ("BRAKES & DRIVETRAIN", ("brake_bias", "diff")),
)

_FAMILY_TO_GROUP: dict[str, str] = {}
for _label, _families in _GROUPS:
    for _f in _families:
        _FAMILY_TO_GROUP[_f] = _label


_REGIME_RANK = {"sparse": 0, "noisy": 1, "confident": 2, "dense": 3}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_narrative(
    rec: SetupRecommendation,
    model: PhysicsModel,
    justifications: list[SetupJustification],
    *,
    most_recent_setup: dict | str | None = None,
    track_display: str | None = None,
    quali: bool = False,
    pinned: dict[str, float] | None = None,
    warnings: list[str] | None = None,
) -> str:
    """Plain-English briefing — every parameter that moved, with helps + watch."""
    pinned = pinned or {}
    warnings = list(warnings or [])
    onto = ontology_for(model.car)

    past_value = _resolve_past_values(justifications, onto, most_recent_setup)

    lines: list[str] = []
    lines.append("=" * 72)
    track_label = track_display or rec.track
    fuel_l = rec.parameters.get("fuel_level_l")
    fuel_str = f" ({fuel_l[0]:.1f} L fuel)" if fuel_l else ""
    mode = "quali (3-lap stint)" if quali else "race"
    lines.append(f" {model.car} @ {track_label} — {mode}{fuel_str}")
    rolled = _rollup_regime(justifications)
    n_med = _median_n_samples(justifications)
    lines.append(
        f" Conditions: {rec.env.air_temp_c:.0f}°C ambient / "
        f"{rec.env.track_temp_c:.0f}°C track  |  "
        f"Confidence: {rolled} (median n={n_med})"
    )
    lines.append("=" * 72)
    lines.append("")

    # ---- OVERALL DIRECTION ----
    moved = [j for j in justifications if _moved(j, past_value, onto)]
    lines.extend(_overall_direction(moved, past_value, onto))
    lines.append("")

    # ---- CHANGES ----
    skipped = [
        j for j in justifications
        if not _moved(j, past_value, onto) and not j.pinned
    ]
    pinned_js = [j for j in justifications if j.pinned]

    lines.append(f"CHANGES ({len(moved)} of {len(justifications)} parameters moved)")
    lines.append("")

    by_group: dict[str, list[SetupJustification]] = defaultdict(list)
    for j in moved:
        spec = onto.get(j.parameter)
        family = spec.family if spec else "other"
        by_group[_FAMILY_TO_GROUP.get(family, "OTHER")].append(j)

    for group_label, _families in _GROUPS:
        if group_label not in by_group:
            continue
        lines.append(f"— {group_label} —")
        for j in sorted(by_group[group_label], key=_param_sort_key):
            lines.extend(_render_change(j, past_value.get(j.parameter), onto))
            lines.append("")

    if "OTHER" in by_group:
        lines.append("— OTHER —")
        for j in sorted(by_group["OTHER"], key=_param_sort_key):
            lines.extend(_render_change(j, past_value.get(j.parameter), onto))
            lines.append("")

    # ---- NOTES (pins / sparse-corpus warnings / untrained / clamp) ----
    note_lines = _notes_block(rec, justifications, pinned_js, skipped, past_value, warnings, onto)
    if note_lines:
        lines.append("NOTES")
        lines.extend(note_lines)

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _render_change(
    j: SetupJustification,
    past: float | None,
    onto: dict[str, ParameterSpec],
) -> list[str]:
    spec = onto.get(j.parameter)
    label = _PARAM_LABEL.get(j.parameter, _humanize(j.parameter))
    family = spec.family if spec else "other"
    direction = _direction_word(family, (j.value - past) if past is not None else 0.0)

    change_str = _format_value_delta(j, spec, past)
    helps = _phase_phrase(j.corners_helped[:5])
    hurts = _phase_phrase(j.corners_hurt[:4])

    body: list[str] = [f"{label}: {change_str}  ({direction})"]
    if helps:
        body.append(f"  Helps: {helps}")
    if hurts:
        body.append(f"  Watch: {hurts}")
    if not helps and not hurts:
        body.append("  (no per-corner trade-off — model held this at training baseline)")
    return body


def _format_value_delta(
    j: SetupJustification, spec: ParameterSpec | None, past: float | None,
) -> str:
    """`50 → 60 N/mm` or `Soft → Medium` or `60 N/mm` (no past)."""
    units = (spec.units if spec else j.unit) or ""
    if spec and spec.choices:
        idx = max(0, min(len(spec.choices) - 1, int(round(j.value))))
        new_label = spec.choices[idx]
        if past is not None:
            past_idx = max(0, min(len(spec.choices) - 1, int(round(past))))
            old_label = spec.choices[past_idx]
            if old_label == new_label:
                return new_label
            return f"{old_label} → {new_label}"
        return new_label

    step = (spec.step if spec and spec.step else 0.5)
    if step >= 1.0:
        fmt = lambda v: f"{v:.0f}"  # noqa: E731
    elif step >= 0.1:
        fmt = lambda v: f"{v:.1f}"  # noqa: E731
    else:
        fmt = lambda v: f"{v:.2f}"  # noqa: E731

    val_s = fmt(j.value).rstrip()
    unit_s = f" {units}" if units else ""
    if past is None:
        return f"{val_s}{unit_s}"
    past_s = fmt(past).rstrip()
    if past_s == val_s:
        return f"{val_s}{unit_s}"
    return f"{past_s} → {val_s}{unit_s}"


def _direction_word(family: str, delta: float) -> str:
    if abs(delta) < 1e-9:
        return "no change"
    if family in _FAMILY_THEME:
        _theme, up, down = _FAMILY_THEME[family]
        return up if delta > 0 else down
    return "increases" if delta > 0 else "decreases"


def _phase_phrase(impacts: tuple[CornerPhaseImpact, ...]) -> str:
    """`T9, T13 mid-corner; T3 under braking`"""
    if not impacts:
        return ""
    by_phase: dict[Phase, list[int]] = defaultdict(list)
    for imp in impacts:
        by_phase[imp.phase].append(imp.corner_id)
    parts: list[str] = []
    for phase in (
        Phase.BRAKING, Phase.TRAIL_BRAKE, Phase.MID_CORNER,
        Phase.EXIT, Phase.STRAIGHT,
    ):
        if phase not in by_phase:
            continue
        corners = sorted(set(by_phase[phase]))
        corner_str = ", ".join(f"T{c}" for c in corners)
        parts.append(f"{corner_str} {_PHASE_LABEL.get(phase, phase.value)}")
    return "; ".join(parts)


def _overall_direction(
    moved: list[SetupJustification],
    past_value: dict[str, float | None],
    onto: dict[str, ParameterSpec],
) -> list[str]:
    """One paragraph summarising the dominant direction per family."""
    if not moved:
        return [
            "OVERALL DIRECTION",
            "  No changes from past setup — already optimal in the model's view.",
        ]
    family_dirs: dict[str, dict[str, int]] = defaultdict(
        lambda: {"up": 0, "down": 0},
    )
    for j in moved:
        spec = onto.get(j.parameter)
        family = spec.family if spec else "other"
        past = past_value.get(j.parameter)
        if past is None:
            continue
        if j.value > past:
            family_dirs[family]["up"] += 1
        else:
            family_dirs[family]["down"] += 1

    summaries: list[str] = []
    for family, dirs in family_dirs.items():
        if family not in _FAMILY_THEME:
            continue
        theme, up, down = _FAMILY_THEME[family]
        if dirs["up"] > dirs["down"]:
            summaries.append(f"{up} {theme}")
        elif dirs["down"] > dirs["up"]:
            summaries.append(f"{down} {theme}")
        else:
            summaries.append(f"adjusts {theme}")

    if not summaries:
        sentence = f"Adjusts {len(moved)} parameter(s) without a single dominant theme."
    else:
        sentence = "; ".join(summaries) + "."
    return [
        "OVERALL DIRECTION",
        f"  {sentence[0].upper()}{sentence[1:]}",
    ]


def _notes_block(
    rec: SetupRecommendation,
    justifications: list[SetupJustification],
    pinned_js: list[SetupJustification],
    skipped: list[SetupJustification],
    past_value: dict[str, float | None],
    warnings: list[str],
    onto: dict[str, ParameterSpec],
) -> list[str]:
    out: list[str] = []
    for j in pinned_js:
        label = _PARAM_LABEL.get(j.parameter, _humanize(j.parameter))
        spec = onto.get(j.parameter)
        val_s = _format_value_delta(j, spec, None)
        out.append(f"  {label}: pinned by user at {val_s}")
    # Truly unchanged params already implicit; only mention "held at past" for params
    # the user might have expected to move
    for j in skipped:
        past = past_value.get(j.parameter)
        if past is None:
            continue
        spec = onto.get(j.parameter)
        step = (spec.step if spec and spec.step else 0.5)
        if abs(j.value - past) >= step / 2:
            continue  # actually moved by a half-step or more — covered above
    if rec.pinned_to_observed_median:
        for name in sorted(rec.pinned_to_observed_median):
            label = _PARAM_LABEL.get(name, _humanize(name))
            out.append(
                f"  {label}: corpus has only one value — vary it next session "
                f"for a recommendation"
            )
    if rec.untrained_parameters:
        params_str = ", ".join(sorted(rec.untrained_parameters))
        out.append(f"  Untrained (constraints.md TODO): {params_str}")
    for w in warnings:
        out.append(f"  ⚠ {w}")
    return out


def _resolve_past_values(
    justifications: list[SetupJustification],
    onto: dict[str, ParameterSpec],
    most_recent_setup: dict | str | None,
) -> dict[str, float | None]:
    setup = _coerce_setup(most_recent_setup)
    out: dict[str, float | None] = {}
    if setup is None:
        for j in justifications:
            out[j.parameter] = None
        return out
    from racingoptimizer.explain.full_setup_card import _scalar_from_yaml
    for j in justifications:
        spec = onto.get(j.parameter)
        if spec is None:
            out[j.parameter] = None
            continue
        cur: dict | str | float | int | None = setup
        for seg in spec.json_path:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(seg)
        if cur is None:
            out[j.parameter] = None
            continue
        if spec.choices and isinstance(cur, str):
            raw_norm = cur.strip().lower()
            for idx, label in enumerate(spec.choices):
                if label.strip().lower() == raw_norm:
                    out[j.parameter] = float(idx)
                    break
            else:
                out[j.parameter] = None
        else:
            out[j.parameter] = _scalar_from_yaml(cur)
    return out


def _coerce_setup(setup: dict | str | None) -> dict | None:
    if setup is None:
        return None
    if isinstance(setup, str):
        import json
        try:
            return json.loads(setup)
        except Exception:
            return None
    return setup


def _moved(
    j: SetupJustification,
    past_value: dict[str, float | None],
    onto: dict[str, ParameterSpec],
) -> bool:
    if j.pinned:
        return False
    past = past_value.get(j.parameter)
    if past is None:
        return False
    spec = onto.get(j.parameter)
    step = (spec.step if spec and spec.step else 0.5)
    return abs(j.value - past) >= step / 2


def _param_sort_key(j: SetupJustification) -> tuple[int, str]:
    """Sort within a group: largest absolute trade-off first, then name."""
    helped = sum(abs(i.score_delta) for i in j.corners_helped)
    hurt = sum(abs(i.score_delta) for i in j.corners_hurt)
    return (-int((helped + hurt) * 1e6), j.parameter)


def _humanize(name: str) -> str:
    return name.replace("_", " ").title()


def _rollup_regime(justifications: list[SetupJustification]) -> str:
    if not justifications:
        return "sparse"
    worst_rank = min(
        _REGIME_RANK[j.confidence.regime] for j in justifications
    )
    for k, v in _REGIME_RANK.items():
        if v == worst_rank:
            return k
    return "sparse"


def _median_n_samples(justifications: list[SetupJustification]) -> int:
    if not justifications:
        return 0
    counts = sorted(j.confidence.n_samples for j in justifications)
    return counts[len(counts) // 2]


__all__ = ["render_narrative"]
