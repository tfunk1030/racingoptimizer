"""SetupJustification field validation tests (spec §3, §12)."""
from __future__ import annotations

import pytest

from racingoptimizer.confidence import Confidence
from racingoptimizer.corner import Phase
from racingoptimizer.explain import (
    CornerPhaseImpact,
    IncompleteJustificationError,
    SetupJustification,
)


def _conf(regime: str = "dense", n: int = 50) -> Confidence:
    return Confidence(value=42.0, lo=40.0, hi=44.0, n_samples=n, regime=regime)  # type: ignore[arg-type]


def _impact(corner: int = 1, delta: float = 0.2) -> CornerPhaseImpact:
    return CornerPhaseImpact(
        corner_id=corner,
        phase=Phase.MID_CORNER,
        score_delta=delta,
        note="t1 mid score gain",
    )


def test_valid_justification_constructs() -> None:
    j = SetupJustification(
        parameter="rear_wing_angle_deg",
        value=14.0,
        unit="deg",
        confidence=_conf(),
        corners_helped=(_impact(),),
        corners_hurt=(),
        sensitivity_minus_1_click=0.05,
        sensitivity_plus_1_click=-0.18,
        telemetry_evidence=("dense fit, n=50",),
    )
    assert j.parameter == "rear_wing_angle_deg"
    assert j.confidence.regime == "dense"


def test_missing_telemetry_evidence_raises() -> None:
    with pytest.raises(IncompleteJustificationError, match="telemetry_evidence"):
        SetupJustification(
            parameter="rear_wing_angle_deg",
            value=14.0,
            unit="deg",
            confidence=_conf(),
            corners_helped=(_impact(),),
            corners_hurt=(),
            sensitivity_minus_1_click=0.05,
            sensitivity_plus_1_click=-0.18,
            telemetry_evidence=(),
        )


def test_both_helps_and_hurts_empty_raises() -> None:
    with pytest.raises(IncompleteJustificationError, match="corners_helped"):
        SetupJustification(
            parameter="rear_wing_angle_deg",
            value=14.0,
            unit="deg",
            confidence=_conf(),
            corners_helped=(),
            corners_hurt=(),
            sensitivity_minus_1_click=0.05,
            sensitivity_plus_1_click=-0.18,
            telemetry_evidence=("ok",),
        )


def test_pure_helper_is_valid() -> None:
    j = SetupJustification(
        parameter="x", value=1.0, unit="?",
        confidence=_conf(), corners_helped=(_impact(),), corners_hurt=(),
        sensitivity_minus_1_click=0.0, sensitivity_plus_1_click=0.0,
        telemetry_evidence=("e",),
    )
    assert not j.corners_hurt


def test_pure_hurter_is_valid() -> None:
    j = SetupJustification(
        parameter="x", value=1.0, unit="?",
        confidence=_conf(), corners_helped=(),
        corners_hurt=(_impact(delta=-0.1),),
        sensitivity_minus_1_click=0.0, sensitivity_plus_1_click=0.0,
        telemetry_evidence=("e",),
    )
    assert not j.corners_helped
