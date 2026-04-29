"""Renderer smoke tests over hand-built dataclasses."""
from __future__ import annotations

import json

from racingoptimizer.confidence import Confidence
from racingoptimizer.corner import Phase
from racingoptimizer.explain import (
    CornerPhaseDelta,
    ModelStatus,
    SetupComparison,
    TrackCoverage,
    render_comparison_json,
    render_comparison_text,
    render_status_json,
    render_status_text,
)


def test_status_text_includes_track_and_regime() -> None:
    s = ModelStatus(
        car="bmw",
        coverage=(
            TrackCoverage(
                track="sebring_international",
                n_sessions=4,
                n_valid_laps=126,
                n_clean_corner_phases=11_832,
                fit_quality=0.08,
                regime="dense",
            ),
        ),
        overall_regime="dense",
        notes=("constraints.md missing bounds for ARBs",),
    )
    text = render_status_text(s)
    assert "sebring_international" in text
    assert "dense" in text
    assert "ARBs" in text


def test_status_json_round_trips() -> None:
    s = ModelStatus(
        car="bmw",
        coverage=(
            TrackCoverage(
                track="sebring_international",
                n_sessions=4, n_valid_laps=126, n_clean_corner_phases=11_832,
                fit_quality=0.08, regime="dense",
            ),
        ),
        overall_regime="dense", notes=(),
    )
    out = render_status_json(s)
    payload = json.loads(json.dumps(out))
    assert payload["car"] == "bmw"
    assert payload["coverage"][0]["track"] == "sebring_international"
    assert payload["coverage"][0]["fit_quality"] == 0.08


def test_comparison_text_renders_zero_delta_self_vs_self() -> None:
    cmp = SetupComparison(
        car="bmw",
        track="sebring_international",
        setup_a_id="abc123",
        setup_b_id="abc123",
        total_score_a=4.21,
        total_score_b=4.21,
        per_corner_phase=(),
        notes=("same hash on both inputs",),
    )
    text = render_comparison_text(cmp)
    assert "+0.000" in text
    assert "abc123" in text


def test_comparison_json_carries_drivers() -> None:
    cmp = SetupComparison(
        car="bmw",
        track="sebring_international",
        setup_a_id="aaa",
        setup_b_id="bbb",
        total_score_a=4.0,
        total_score_b=4.5,
        per_corner_phase=(
            CornerPhaseDelta(
                corner_id=1,
                phase=Phase.MID_CORNER,
                score_a=0.4,
                score_b=0.5,
                delta=0.1,
                drivers=("rear_wing_angle_deg: 14.0 -> 16.0",),
            ),
        ),
        notes=(),
    )
    out = render_comparison_json(cmp)
    payload = json.loads(json.dumps(out))
    assert payload["per_corner_phase"][0]["drivers"][0].startswith("rear_wing_angle_deg")


def _conf(regime="dense", n=50):
    return Confidence(value=14.0, lo=12.0, hi=16.0, n_samples=n, regime=regime)
