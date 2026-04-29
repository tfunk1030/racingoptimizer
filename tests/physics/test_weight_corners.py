"""weight_corners derivation + lap-time isolation (spec §6, §13).

The weight derivation is allowed to reference lap_time (spec §6 calls it out
explicitly). The objective is NOT — see test_score.py for the grep test that
confirms score.py / recommend.py keep no lap_time reference.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from racingoptimizer.physics import weight_corners


@pytest.fixture
def bmw_model(bmw_model_session):
    return bmw_model_session


def test_weight_corners_returns_normalised_dict(bmw_model) -> None:
    model, track, root = bmw_model
    weights = weight_corners(track, model, corpus_root=root)
    assert isinstance(weights, dict)
    assert weights, "weight_corners produced no corners"
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-9
    for c, w in weights.items():
        assert isinstance(c, int)
        assert w >= 0.0
        assert w <= 1.0


def test_weight_corners_uniform_when_no_data(bmw_model) -> None:
    """A track with no laps in the corpus → uniform weights across the model's corners."""
    model, _track, root = bmw_model
    weights = weight_corners("nonexistent_track_xyz", model, corpus_root=root)
    if not weights:
        pytest.skip("model has no fitters → no corners to weight")
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-9
    n = len(weights)
    for w in weights.values():
        assert abs(w - 1.0 / n) < 1e-9


def test_weight_corners_lap_time_lives_only_in_weight_derivation() -> None:
    """Spec §6: lap_time is allowed in weights.py (the derivation source).

    score.py / recommend.py must NOT reference it — that's the test in
    test_score.py::test_score_setup_no_lap_time_reference.
    """
    weights_src = (
        Path(__file__).resolve().parents[2]
        / "src" / "racingoptimizer" / "physics" / "weights.py"
    )
    text = weights_src.read_text(encoding="utf-8").lower()
    # Sanity: weights.py SHOULD reference lap_time (it's the source of truth
    # for per-corner sensitivity).
    assert "lap_time" in text, "weights.py should reference lap_time"
