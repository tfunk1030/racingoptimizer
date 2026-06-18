"""Out-of-domain aero warning + confidence downgrade (AUDIT H2).

When a recommend run scored most corner-phases against clamped (envelope-
edge) aero, `_aero_out_of_domain_warnings` must say so and downgrade the
aero-driven parameter families one confidence tier. Pure-unit: a real
AeroSurface is seeded into the score-path cache for a stub model; no
corpus or LFS fixtures needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from racingoptimizer.aero.interpolator import BASELINE_AIR_DENSITY, AeroSurface
from racingoptimizer.aero.loader import load_aero_map_data
from racingoptimizer.cli.recommend import _aero_out_of_domain_warnings
from racingoptimizer.confidence import Confidence
from racingoptimizer.physics import score as score_mod


@dataclass
class _StubRec:
    parameters: dict[str, tuple[float, Confidence]]


def _conf(value: float, regime: str = "dense") -> Confidence:
    return Confidence(value=value, lo=value, hi=value, n_samples=40, regime=regime)


@pytest.fixture
def cadillac_model(aero_dir: Path):
    """Stub model whose car resolves to a real cached AeroSurface."""
    surface = AeroSurface(load_aero_map_data("cadillac", aero_dir=aero_dir))
    score_mod._AERO_CACHE["cadillac"] = surface
    yield SimpleNamespace(car="cadillac", aero_correction_available=True), surface
    score_mod._AERO_CACHE.pop("cadillac", None)


def test_systematic_clamp_warns_and_downgrades(cadillac_model) -> None:
    model, surface = cadillac_model
    # Simulate a DE run that queried below the 25 mm front floor throughout.
    for _ in range(10):
        surface.interpolate(8.43, 25.0, 14.0, BASELINE_AIR_DENSITY)

    rec = _StubRec(
        parameters={
            "rear_wing_angle_deg": (14.0, _conf(14.0)),
            "brake_bias_pct": (47.5, _conf(47.5)),
        }
    )
    warnings = _aero_out_of_domain_warnings(rec, model)

    assert any("OUT OF DOMAIN" in w for w in warnings)
    assert any("rear_wing_angle_deg" in w for w in warnings)
    # Aero-driven family downgraded one tier; unrelated family untouched.
    assert rec.parameters["rear_wing_angle_deg"][1].regime == "confident"
    assert rec.parameters["brake_bias_pct"][1].regime == "dense"
    # The bias estimate is quantified from the map's own floor gradient.
    assert any("balance bias" in w for w in warnings)


def test_in_domain_run_is_silent(cadillac_model) -> None:
    model, surface = cadillac_model
    for _ in range(10):
        surface.interpolate(50.0, 25.0, 14.0, BASELINE_AIR_DENSITY)

    rec = _StubRec(parameters={"rear_wing_angle_deg": (14.0, _conf(14.0))})
    assert _aero_out_of_domain_warnings(rec, model) == []
    assert rec.parameters["rear_wing_angle_deg"][1].regime == "dense"


def test_minority_clamp_below_threshold_is_silent(cadillac_model) -> None:
    model, surface = cadillac_model
    surface.interpolate(8.0, 25.0, 14.0, BASELINE_AIR_DENSITY)
    for _ in range(9):
        surface.interpolate(50.0, 25.0, 14.0, BASELINE_AIR_DENSITY)

    rec = _StubRec(parameters={"rear_wing_angle_deg": (14.0, _conf(14.0))})
    assert _aero_out_of_domain_warnings(rec, model) == []


def test_no_queries_is_silent(cadillac_model) -> None:
    model, _surface = cadillac_model
    rec = _StubRec(parameters={"rear_wing_angle_deg": (14.0, _conf(14.0))})
    assert _aero_out_of_domain_warnings(rec, model) == []


def test_sparse_regime_is_not_downgraded_further(cadillac_model) -> None:
    model, surface = cadillac_model
    for _ in range(10):
        surface.interpolate(8.43, 25.0, 14.0, BASELINE_AIR_DENSITY)

    rec = _StubRec(
        parameters={"rear_wing_angle_deg": (14.0, _conf(14.0, regime="sparse"))}
    )
    warnings = _aero_out_of_domain_warnings(rec, model)
    assert any("OUT OF DOMAIN" in w for w in warnings)
    assert rec.parameters["rear_wing_angle_deg"][1].regime == "sparse"
