"""Human-text briefing renderers (spec §4 layout)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from racingoptimizer.explain.comparison import SetupComparison
from racingoptimizer.explain.justification import (
    SetupJustification,
)
from racingoptimizer.explain.status import ModelStatus

if TYPE_CHECKING:
    from racingoptimizer.physics import PhysicsModel, SetupRecommendation

# Caution-ordered: a single sparse parameter masks an otherwise dense fit
# (spec §14 confidence regime roll-up).
_REGIME_RANK: dict[str, int] = {"sparse": 0, "noisy": 1, "confident": 2, "dense": 3}

# Maps machine parameter names to human briefing labels. Falls back to a
# title-cased de-snaked version when missing.
_HUMAN_LABEL: dict[str, str] = {
    "rear_wing_angle_deg": "Rear Wing Angle",
    "tyre_cold_pressure_kpa": "Tyre Cold Pressure",
    "static_ride_height_front_mm": "Front Static Ride Height",
    "static_ride_height_rear_mm": "Rear Static Ride Height",
    "heave_spring_mm": "Heave Spring",
    "heave_slider_mm": "Heave Slider",
}


def render_recommendation_text(
    rec: SetupRecommendation,
    model: PhysicsModel,
    *,
    justifications: list[SetupJustification] | None = None,
    pinned: dict[str, float] | None = None,
    warnings: list[str] | None = None,
    track_display: str | None = None,
) -> str:
    if justifications is None:
        from racingoptimizer.explain.justification import build_justifications
        justifications = build_justifications(rec, model, pinned=pinned)
    pinned = pinned or {}
    warnings = list(warnings or [])

    lines: list[str] = []
    track_label = track_display or _humanize_slug(rec.track)
    lines.append(f"{rec.car} @ {track_label} - recommended setup")
    lines.append(_conditions_line(rec))

    rolled = _roll_up_confidence(justifications)
    lines.append(rolled)

    if pinned:
        pin_str = ", ".join(f"{k} = {v:g}" for k, v in sorted(pinned.items()))
        lines.append(f"Pinned by user: {pin_str}")

    lines.append("")

    for justification in justifications:
        lines.extend(_render_block(justification))
        lines.append("")

    if rec.untrained_parameters:
        warnings.append(
            f"untrained_parameters: {', '.join(rec.untrained_parameters)} "
            f"(skipped — bounds not in constraints.md)"
        )

    if warnings:
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines).rstrip() + "\n"


def render_comparison_text(cmp: SetupComparison) -> str:
    lines: list[str] = []
    lines.append(f"{cmp.car} @ {_humanize_slug(cmp.track)} - setup comparison")
    lines.append(f"  setup A: {cmp.setup_a_id}  total score = {cmp.total_score_a:.3f}")
    lines.append(f"  setup B: {cmp.setup_b_id}  total score = {cmp.total_score_b:.3f}")
    lines.append(f"  delta (B - A) = {cmp.total_score_b - cmp.total_score_a:+.3f}")
    lines.append("")

    if cmp.per_corner_phase:
        lines.append("Per-(corner, phase) deltas (top 10 by |delta|):")
        ordered = sorted(cmp.per_corner_phase, key=lambda d: abs(d.delta), reverse=True)
        for delta in ordered[:10]:
            lines.append(
                f"  T{delta.corner_id}-{delta.phase.value}: "
                f"A={delta.score_a:.3f}  B={delta.score_b:.3f}  "
                f"delta={delta.delta:+.3f}"
            )
            for driver in delta.drivers[:3]:
                lines.append(f"    - {driver}")
    if cmp.notes:
        lines.append("")
        lines.append("Notes:")
        for note in cmp.notes:
            lines.append(f"  - {note}")
    return "\n".join(lines).rstrip() + "\n"


def render_status_text(status: ModelStatus) -> str:
    lines: list[str] = []
    lines.append(f"{status.car} - coverage report")
    lines.append("-" * 73)
    lines.append(
        f"{'Track':<27}{'Sessions':>9}{'Valid laps':>12}{'Clean CP':>10}"
        f"{'Fit sigma':>11}{'Regime':>11}"
    )
    for cov in status.coverage:
        fit = f"{cov.fit_quality:.3f}" if cov.fit_quality is not None else "-"
        lines.append(
            f"{cov.track:<27}{cov.n_sessions:>9}{cov.n_valid_laps:>12}"
            f"{cov.n_clean_corner_phases:>10}{fit:>11}{cov.regime:>11}"
        )
    lines.append("-" * 73)
    lines.append(f"Overall regime: {status.overall_regime}")
    if status.notes:
        lines.append("")
        lines.append("Notes:")
        for note in status.notes:
            lines.append(f"  - {note}")
    return "\n".join(lines).rstrip() + "\n"


# ---- internals -----------------------------------------------------------


def _conditions_line(rec: SetupRecommendation) -> str:
    env = rec.env
    return (
        f"Conditions: AirTemp {env.track_temp_c:.1f} C  "
        f"AirDensity {env.air_density:.3f} kg/m^3  "
        f"Wind {env.wind_vel_ms:.1f} m/s  "
        f"Wetness {env.track_wetness:.2f}"
    )


def _roll_up_confidence(justifications: list[SetupJustification]) -> str:
    if not justifications:
        return "Confidence: sparse (no parameters fitted)"
    worst_rank = min(_REGIME_RANK[j.confidence.regime] for j in justifications)
    rank_to_regime = {v: k for k, v in _REGIME_RANK.items()}
    regime = rank_to_regime[worst_rank]
    samples_for_regime = [
        j.confidence.n_samples for j in justifications
        if j.confidence.regime == regime
    ]
    n_samples = max(samples_for_regime) if samples_for_regime else 0
    return (
        f"Confidence: {regime} "
        f"(n={n_samples} backing samples for the dominant {regime} parameter, "
        f"{len(justifications)} parameters reported)"
    )


def _render_block(j: SetupJustification) -> list[str]:
    label = _HUMAN_LABEL.get(j.parameter) or _humanize_slug(j.parameter)
    confidence_tag = "pinned by user" if j.pinned else j.confidence.regime
    head = f"{label}: {j.value:.2f} {j.unit}   [confidence: {confidence_tag}]"
    out = [head]
    if j.pinned:
        out.append("    (no sensitivity reported - pinned)")
    else:
        out.append(
            f"    +1 click: {j.sensitivity_plus_1_click:+.3f} score    "
            f"-1 click: {j.sensitivity_minus_1_click:+.3f} score"
        )
    if j.corners_helped:
        out.append("    Helps:")
        for impact in j.corners_helped[:3]:
            out.append(_impact_line(impact))
    if j.corners_hurt:
        out.append("    Hurts:")
        for impact in j.corners_hurt[:3]:
            out.append(_impact_line(impact))
    out.append("    Evidence:")
    for evidence in j.telemetry_evidence:
        out.append(f"      - {evidence}")
    return out


def _impact_line(impact) -> str:
    return (
        f"           T{impact.corner_id}-{impact.phase.value} "
        f"({impact.note}, score {impact.score_delta:+.3f})"
    )


def _humanize_slug(slug: str) -> str:
    return " ".join(word.capitalize() for word in slug.split("_"))


__all__ = [
    "render_comparison_text",
    "render_recommendation_text",
    "render_status_text",
]
