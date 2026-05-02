"""Golden-file render tests for the briefing renderers (CLI spec §12).

Builds deterministic synthetic instances of `SetupRecommendation`,
`SetupComparison`, and `ModelStatus`, renders them through the public
text + JSON renderers from `racingoptimizer.explain`, and compares the
output against golden files committed under `tests/cli/golden/`.

Regenerate after a deliberate render-format change with::

    uv run pytest tests/cli/test_golden_files.py --regenerate-golden -v

Then review the diff before committing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from racingoptimizer.confidence import Confidence
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner import CornerPhaseKey, Phase
from racingoptimizer.explain import (
    CornerPhaseDelta,
    CornerPhaseImpact,
    ModelStatus,
    SetupComparison,
    SetupJustification,
    TrackCoverage,
    render_comparison_json,
    render_comparison_text,
    render_recommendation_json,
    render_recommendation_text,
    render_status_json,
    render_status_text,
)
from racingoptimizer.physics import SetupRecommendation

GOLDEN_DIR = Path(__file__).parent / "golden"


# --- comparison helpers ---------------------------------------------------


def _compare_or_regen_text(actual: str, golden_name: str, regenerate: bool) -> None:
    path = GOLDEN_DIR / golden_name
    if regenerate:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8", newline="\n")
        return
    assert path.exists(), (
        f"golden file {path} missing; rerun with --regenerate-golden to create it"
    )
    expected = path.read_text(encoding="utf-8")
    assert actual == expected, (
        f"golden mismatch for {golden_name}; rerun with --regenerate-golden "
        f"after reviewing the diff"
    )


def _compare_or_regen_json(actual: dict, golden_name: str, regenerate: bool) -> None:
    path = GOLDEN_DIR / golden_name
    serialised = json.dumps(actual, indent=2, sort_keys=False) + "\n"
    if regenerate:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(serialised, encoding="utf-8", newline="\n")
        return
    assert path.exists(), (
        f"golden file {path} missing; rerun with --regenerate-golden to create it"
    )
    expected = path.read_text(encoding="utf-8")
    assert serialised == expected, (
        f"golden mismatch for {golden_name}; rerun with --regenerate-golden "
        f"after reviewing the diff"
    )


# --- synthetic builders ---------------------------------------------------


def _env() -> EnvironmentFrame:
    return EnvironmentFrame(
        air_density=1.184,
        track_temp_c=32.5,
        wind_vel_ms=2.4,
        wind_dir_deg=215.0,
        track_wetness=0.0,
    )


def _build_synthetic_recommendation() -> SetupRecommendation:
    env = _env()
    parameters: dict[str, tuple[float, Confidence]] = {
        "rear_wing_angle_deg": (
            12.5,
            Confidence(value=12.5, lo=12.0, hi=13.0, n_samples=240, regime="confident"),
        ),
        "tyre_cold_pressure_kpa": (
            165.0,
            Confidence(value=165.0, lo=163.5, hi=166.5, n_samples=480, regime="dense"),
        ),
        "static_ride_height_front_mm": (
            55.0,
            Confidence(value=55.0, lo=54.0, hi=56.0, n_samples=120, regime="confident"),
        ),
        "static_ride_height_rear_mm": (
            70.0,
            Confidence(value=70.0, lo=68.5, hi=71.5, n_samples=90, regime="noisy"),
        ),
    }
    score_breakdown: dict[CornerPhaseKey, float] = {
        CornerPhaseKey("synthetic", 1, 1, Phase.BRAKING): 0.812,
        CornerPhaseKey("synthetic", 1, 1, Phase.MID_CORNER): 0.741,
        CornerPhaseKey("synthetic", 1, 3, Phase.TRAIL_BRAKE): 0.685,
        CornerPhaseKey("synthetic", 1, 5, Phase.EXIT): 0.902,
        CornerPhaseKey("synthetic", 1, 7, Phase.STRAIGHT): 0.954,
    }
    return SetupRecommendation(
        car="bmw",
        track="sebring_international",
        env=env,
        parameters=parameters,
        score_breakdown=score_breakdown,
        untrained_parameters=("damper_lsc_fl", "anti_roll_bar_front"),
        aero_correction_available=True,
    )


def _build_synthetic_justifications() -> list[SetupJustification]:
    """Hand-built justifications matching the synthetic recommendation.

    Keeping this independent of `build_justifications(...)` keeps the golden
    output deterministic and decoupled from the physics-model code path.
    """
    rear_wing = SetupJustification(
        parameter="rear_wing_angle_deg",
        value=12.5,
        unit="deg",
        confidence=Confidence(
            value=12.5, lo=12.0, hi=13.0, n_samples=240, regime="confident",
        ),
        corners_helped=(
            CornerPhaseImpact(
                corner_id=1, phase=Phase.MID_CORNER,
                score_delta=0.045, note="mid-corner score gain 0.045",
            ),
            CornerPhaseImpact(
                corner_id=5, phase=Phase.EXIT,
                score_delta=0.022, note="exit score gain 0.022",
            ),
        ),
        corners_hurt=(
            CornerPhaseImpact(
                corner_id=7, phase=Phase.STRAIGHT,
                score_delta=-0.018, note="straight score cost 0.018",
            ),
        ),
        sensitivity_minus_1_click=-0.012,
        sensitivity_plus_1_click=0.008,
        telemetry_evidence=(
            "confident confidence backed by 240 samples",
            "observed in training [12.000, 13.000]",
        ),
    )
    tyre_pressure = SetupJustification(
        parameter="tyre_cold_pressure_kpa",
        value=165.0,
        unit="kPa",
        confidence=Confidence(
            value=165.0, lo=163.5, hi=166.5, n_samples=480, regime="dense",
        ),
        corners_helped=(
            CornerPhaseImpact(
                corner_id=1, phase=Phase.BRAKING,
                score_delta=0.031, note="braking score gain 0.031",
            ),
            CornerPhaseImpact(
                corner_id=3, phase=Phase.TRAIL_BRAKE,
                score_delta=0.019, note="trail-brake score gain 0.019",
            ),
        ),
        corners_hurt=(),
        sensitivity_minus_1_click=-0.005,
        sensitivity_plus_1_click=-0.011,
        telemetry_evidence=(
            "dense confidence backed by 480 samples",
            "observed in training [163.500, 166.500]",
        ),
    )
    front_rh = SetupJustification(
        parameter="static_ride_height_front_mm",
        value=55.0,
        unit="mm",
        confidence=Confidence(
            value=55.0, lo=54.0, hi=56.0, n_samples=120, regime="confident",
        ),
        corners_helped=(
            CornerPhaseImpact(
                corner_id=1, phase=Phase.MID_CORNER,
                score_delta=0.014, note="mid-corner score gain 0.014",
            ),
        ),
        corners_hurt=(
            CornerPhaseImpact(
                corner_id=5, phase=Phase.EXIT,
                score_delta=-0.009, note="exit score cost 0.009",
            ),
        ),
        sensitivity_minus_1_click=0.003,
        sensitivity_plus_1_click=-0.006,
        telemetry_evidence=(
            "confident confidence backed by 120 samples",
            "observed in training [54.000, 56.000]",
        ),
    )
    rear_rh = SetupJustification(
        parameter="static_ride_height_rear_mm",
        value=70.0,
        unit="mm",
        confidence=Confidence(
            value=70.0, lo=68.5, hi=71.5, n_samples=90, regime="noisy",
        ),
        corners_helped=(
            CornerPhaseImpact(
                corner_id=3, phase=Phase.TRAIL_BRAKE,
                score_delta=0.011, note="trail-brake score gain 0.011",
            ),
        ),
        corners_hurt=(
            CornerPhaseImpact(
                corner_id=1, phase=Phase.BRAKING,
                score_delta=-0.007, note="braking score cost 0.007",
            ),
        ),
        sensitivity_minus_1_click=-0.004,
        sensitivity_plus_1_click=0.002,
        telemetry_evidence=(
            "noisy confidence backed by 90 samples",
            "observed in training [68.500, 71.500]",
        ),
    )
    return [rear_wing, tyre_pressure, front_rh, rear_rh]


def _build_synthetic_comparison() -> SetupComparison:
    return SetupComparison(
        car="bmw",
        track="sebring_international",
        setup_a_id="baseline_session_a",
        setup_b_id="candidate_session_b",
        total_score_a=8.124,
        total_score_b=8.391,
        per_corner_phase=(
            CornerPhaseDelta(
                corner_id=1, phase=Phase.BRAKING,
                score_a=0.812, score_b=0.844, delta=0.032,
                drivers=(
                    "rear_wing_angle_deg shifted +0.5 deg",
                    "tyre_cold_pressure_kpa shifted -1.0 kPa",
                ),
            ),
            CornerPhaseDelta(
                corner_id=3, phase=Phase.TRAIL_BRAKE,
                score_a=0.685, score_b=0.712, delta=0.027,
                drivers=(
                    "static_ride_height_rear_mm shifted +0.5 mm",
                ),
            ),
            CornerPhaseDelta(
                corner_id=5, phase=Phase.EXIT,
                score_a=0.902, score_b=0.881, delta=-0.021,
                drivers=(
                    "static_ride_height_front_mm shifted -0.5 mm",
                ),
            ),
            CornerPhaseDelta(
                corner_id=7, phase=Phase.STRAIGHT,
                score_a=0.954, score_b=0.965, delta=0.011,
                drivers=(
                    "rear_wing_angle_deg shifted +0.5 deg",
                ),
            ),
        ),
        notes=(
            "candidate session lapped 0.18s faster on average over 7 clean laps",
            "wind delta < 1 m/s vs baseline; air density within 0.5%",
        ),
    )


def _build_synthetic_status() -> ModelStatus:
    return ModelStatus(
        car="bmw",
        coverage=(
            TrackCoverage(
                track="sebring_international",
                n_sessions=4, n_valid_laps=28,
                n_clean_corner_phases=420,
                fit_quality=0.187, regime="dense",
            ),
            TrackCoverage(
                track="daytona_road",
                n_sessions=2, n_valid_laps=11,
                n_clean_corner_phases=132,
                fit_quality=0.412, regime="confident",
            ),
            TrackCoverage(
                track="watkins_glen",
                n_sessions=1, n_valid_laps=3,
                n_clean_corner_phases=24,
                fit_quality=None, regime="sparse",
            ),
        ),
        overall_regime="confident",
        notes=(
            "watkins_glen has < 30 clean corner-phases; recommendations "
            "will short-circuit to sparse",
            "no Acura-style shock-deflection gap detected for this car",
        ),
    )


# --- the six golden-file tests --------------------------------------------


def test_synthetic_recommendation_text_golden(request: pytest.FixtureRequest) -> None:
    rec = _build_synthetic_recommendation()
    justifications = _build_synthetic_justifications()
    actual = render_recommendation_text(
        rec, model=None, justifications=justifications,
    )
    _compare_or_regen_text(
        actual,
        "synthetic_recommendation.txt",
        request.config.getoption("--regenerate-golden"),
    )


def test_synthetic_recommendation_json_golden(request: pytest.FixtureRequest) -> None:
    rec = _build_synthetic_recommendation()
    justifications = _build_synthetic_justifications()
    actual = render_recommendation_json(
        rec, model=None, justifications=justifications,
    )
    _compare_or_regen_json(
        actual,
        "synthetic_recommendation.json",
        request.config.getoption("--regenerate-golden"),
    )


def test_synthetic_comparison_text_golden(request: pytest.FixtureRequest) -> None:
    cmp = _build_synthetic_comparison()
    actual = render_comparison_text(cmp)
    _compare_or_regen_text(
        actual,
        "synthetic_comparison.txt",
        request.config.getoption("--regenerate-golden"),
    )


def test_synthetic_comparison_json_golden(request: pytest.FixtureRequest) -> None:
    cmp = _build_synthetic_comparison()
    actual = render_comparison_json(cmp)
    _compare_or_regen_json(
        actual,
        "synthetic_comparison.json",
        request.config.getoption("--regenerate-golden"),
    )


def test_synthetic_status_text_golden(request: pytest.FixtureRequest) -> None:
    status = _build_synthetic_status()
    actual = render_status_text(status)
    _compare_or_regen_text(
        actual,
        "synthetic_status.txt",
        request.config.getoption("--regenerate-golden"),
    )


def test_synthetic_status_json_golden(request: pytest.FixtureRequest) -> None:
    status = _build_synthetic_status()
    actual = render_status_json(status)
    _compare_or_regen_json(
        actual,
        "synthetic_status.json",
        request.config.getoption("--regenerate-golden"),
    )
