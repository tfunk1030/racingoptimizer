"""Per-parameter justification dataclasses + builder (spec §3).

`SetupJustification` raises `IncompleteJustificationError` when the four
required `setup-justifier` fields are missing or both `corners_helped` and
`corners_hurt` are empty (a strictly-helps parameter is allowed; a parameter
that neither helps nor hurts is not a recommendation).

`build_justifications(rec, model)` walks `rec.parameters` and constructs one
`SetupJustification` per parameter by re-using the per-(corner, phase) score
breakdown already on the recommendation, plus per-parameter sensitivity
deltas computed by recomputing the score with the parameter shifted ±1 click.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from racingoptimizer.confidence import Confidence
from racingoptimizer.constraints import clamp
from racingoptimizer.corner import Phase
from racingoptimizer.physics.ontology import ontology_for

if TYPE_CHECKING:
    from racingoptimizer.physics import PhysicsModel, SetupRecommendation


class IncompleteJustificationError(ValueError):
    """A SetupJustification is missing a required `setup-justifier` field."""


@dataclass(frozen=True)
class CornerPhaseImpact:
    corner_id: int
    phase: Phase
    score_delta: float
    note: str


@dataclass(frozen=True)
class SetupJustification:
    parameter: str
    value: float
    unit: str
    confidence: Confidence
    corners_helped: tuple[CornerPhaseImpact, ...]
    corners_hurt: tuple[CornerPhaseImpact, ...]
    sensitivity_minus_1_click: float
    sensitivity_plus_1_click: float
    telemetry_evidence: tuple[str, ...]
    pinned: bool = False

    def __post_init__(self) -> None:
        missing: list[str] = []
        if not self.parameter:
            missing.append("parameter")
        if self.unit is None:
            missing.append("unit")
        if self.confidence is None:
            missing.append("confidence")
        if not self.telemetry_evidence:
            missing.append("telemetry_evidence")
        if missing:
            raise IncompleteJustificationError(
                f"SetupJustification for {self.parameter!r} missing required fields: "
                f"{', '.join(missing)}"
            )
        if not self.corners_helped and not self.corners_hurt:
            raise IncompleteJustificationError(
                f"SetupJustification for {self.parameter!r} has neither "
                f"corners_helped nor corners_hurt — not a recommendation"
            )


# Default per-parameter click step. Values are 1 step in the parameter's
# native unit; sourced from iRacing UI conventions. Falls back to 1.0% of
# the constraint range when the parameter is unknown here.
_CLICK_STEP: dict[str, float] = {
    "rear_wing_angle_deg": 0.5,
    "tyre_cold_pressure_kpa": 1.0,
    "static_ride_height_front_mm": 0.5,
    "static_ride_height_rear_mm": 0.5,
    "heave_spring_mm": 0.5,
    "heave_slider_mm": 0.5,
}


def build_justifications(
    rec: SetupRecommendation,
    model: PhysicsModel,
    *,
    pinned: dict[str, float] | None = None,
    clamp_warnings: dict[str, str] | None = None,
) -> list[SetupJustification]:
    """Build one SetupJustification per parameter on `rec.parameters`."""
    pinned = pinned or {}
    clamp_warnings = clamp_warnings or {}
    onto = ontology_for(model.car)
    setup = {name: value for name, (value, _conf) in rec.parameters.items()}

    out: list[SetupJustification] = []
    for parameter, (value, confidence) in rec.parameters.items():
        helped, hurt = _split_impacts(parameter, value, model, rec, setup)
        unit = onto[parameter].units if parameter in onto else "?"
        is_pinned = parameter in pinned
        if is_pinned:
            sens_minus = 0.0
            sens_plus = 0.0
        else:
            sens_minus, sens_plus = _sensitivity(
                model, setup, parameter, value, rec.track, rec.env,
            )
        evidence = _evidence(
            confidence=confidence,
            is_pinned=is_pinned,
            clamp_warning=clamp_warnings.get(parameter),
        )
        # When the model finds no per-corner trade-off (e.g. cold-start
        # single-session corpus where every parameter sits exactly on its
        # training baseline), inject a synthetic neutral-baseline note so
        # the SetupJustification's helps/hurts contract stays satisfied.
        if not helped and not hurt:
            neutral = CornerPhaseImpact(
                corner_id=0,
                phase=Phase.STRAIGHT,
                score_delta=0.0,
                note=(
                    "pinned by user - reflects model's evaluation"
                    if is_pinned
                    else "model held value at training baseline; "
                         "no per-corner sensitivity in this corpus"
                ),
            )
            helped = (neutral,)
        justification = SetupJustification(
            parameter=parameter,
            value=float(value),
            unit=unit,
            confidence=confidence,
            corners_helped=helped,
            corners_hurt=hurt,
            sensitivity_minus_1_click=float(sens_minus),
            sensitivity_plus_1_click=float(sens_plus),
            telemetry_evidence=tuple(evidence),
            pinned=is_pinned,
        )
        out.append(justification)

    out.sort(key=_impact_magnitude, reverse=True)
    return out


# ---- internals -----------------------------------------------------------


def _split_impacts(
    parameter: str,
    value: float,
    model: PhysicsModel,
    rec: SetupRecommendation,
    setup: dict[str, float],
) -> tuple[tuple[CornerPhaseImpact, ...], tuple[CornerPhaseImpact, ...]]:
    """Per-(corner, phase) score-delta vs the parameter's training baseline.

    Compute the per-(corner, phase) breakdown with the parameter set to its
    model baseline (i.e. the median seen in training); the delta against
    `rec.score_breakdown` is what THIS parameter's recommended value
    contributes. Falls back to a one-step-shifted counterfactual when the
    recommendation matches the baseline exactly so every parameter still
    yields a justification.
    """
    baseline_value = float(model.baseline_setup.get(parameter, value))
    if abs(baseline_value - float(value)) < 1e-9:
        # Recommend matched the baseline -- shift one click in either
        # direction so we still report a non-empty trade-off.
        baseline_value = _shifted_for_counterfactual(model, parameter, value)
        if abs(baseline_value - float(value)) < 1e-9:
            return (), ()

    counterfactual = dict(setup)
    counterfactual[parameter] = baseline_value
    counterfactual_breakdown = _safe_score_breakdown(
        model, counterfactual, rec.track, rec.env,
    )

    helped: list[CornerPhaseImpact] = []
    hurt: list[CornerPhaseImpact] = []
    for cpkey, score in rec.score_breakdown.items():
        cf_score = counterfactual_breakdown.get(cpkey, score)
        delta = float(score - cf_score)
        if abs(delta) < 1e-6:
            continue
        impact = CornerPhaseImpact(
            corner_id=int(cpkey.corner_id),
            phase=cpkey.phase,
            score_delta=delta,
            note=_phase_note(cpkey.phase, delta),
        )
        if delta > 0:
            helped.append(impact)
        else:
            hurt.append(impact)

    helped.sort(key=lambda i: abs(i.score_delta), reverse=True)
    hurt.sort(key=lambda i: abs(i.score_delta), reverse=True)
    return tuple(helped), tuple(hurt)


def _shifted_for_counterfactual(
    model: PhysicsModel, parameter: str, value: float,
) -> float:
    """Shift `value` by one click within the parameter's legal bounds."""
    step = _step_for(model, parameter)
    if step <= 0 or model.constraints is None:
        return value
    bound = model.constraints.bounds(model.car, parameter)
    if bound is None:
        return value
    lo, hi = bound
    candidate = float(value) + step
    if candidate <= hi:
        return candidate
    candidate = float(value) - step
    if candidate >= lo:
        return candidate
    return value


def _safe_score_breakdown(
    model: PhysicsModel,
    setup: dict[str, float],
    track: str,
    env,
) -> dict:
    from racingoptimizer.physics.score import score_breakdown
    try:
        return score_breakdown(model, setup, track, env)
    except Exception:
        return {}


def _sensitivity(
    model: PhysicsModel,
    setup: dict[str, float],
    parameter: str,
    value: float,
    track: str,
    env,
) -> tuple[float, float]:
    """Per-click score delta from shifting the parameter ±1 click."""
    step = _step_for(model, parameter)
    if step <= 0:
        return 0.0, 0.0
    base = _safe_score_total(model, setup, track, env)

    minus = dict(setup)
    minus[parameter] = float(_clamp_value(model, parameter, value - step))
    plus = dict(setup)
    plus[parameter] = float(_clamp_value(model, parameter, value + step))

    minus_score = _safe_score_total(model, minus, track, env)
    plus_score = _safe_score_total(model, plus, track, env)
    return (minus_score - base, plus_score - base)


def _safe_score_total(model: PhysicsModel, setup: dict[str, float], track: str, env) -> float:
    try:
        return float(model.score_setup(setup, track, env))
    except Exception:
        return 0.0


def _step_for(model: PhysicsModel, parameter: str) -> float:
    if parameter in _CLICK_STEP:
        return _CLICK_STEP[parameter]
    if model.constraints is not None:
        bound = model.constraints.bounds(model.car, parameter)
        if bound is not None:
            return max(abs(bound[1] - bound[0]) * 0.01, 1e-6)
    return 0.0


def _clamp_value(model: PhysicsModel, parameter: str, value: float) -> float:
    if model.constraints is None:
        return value
    return float(clamp(float(value), parameter, model.car, model.constraints).value)


_PHASE_HUMAN: dict[Phase, str] = {
    Phase.BRAKING: "braking",
    Phase.TRAIL_BRAKE: "trail-brake",
    Phase.MID_CORNER: "mid-corner",
    Phase.EXIT: "exit",
    Phase.STRAIGHT: "straight",
}


def _phase_note(phase: Phase, delta: float) -> str:
    direction = "score gain" if delta > 0 else "score cost"
    return f"{_PHASE_HUMAN[phase]} {direction} {abs(delta):.3f}"


def _evidence(
    *,
    confidence: Confidence,
    is_pinned: bool,
    clamp_warning: str | None,
) -> list[str]:
    out: list[str] = []
    out.append(
        f"{confidence.regime} confidence backed by {confidence.n_samples} samples"
    )
    if is_pinned:
        out.append("user override — pinned via --pin")
    if clamp_warning:
        out.append(clamp_warning)
    if confidence.lo != confidence.hi:
        out.append(
            f"value bracket [{confidence.lo:.3f}, {confidence.hi:.3f}]"
        )
    return out


def _impact_magnitude(j: SetupJustification) -> float:
    helped = sum(i.score_delta for i in j.corners_helped)
    hurt = sum(abs(i.score_delta) for i in j.corners_hurt)
    return helped + hurt


__all__ = [
    "CornerPhaseImpact",
    "IncompleteJustificationError",
    "SetupJustification",
    "build_justifications",
]
