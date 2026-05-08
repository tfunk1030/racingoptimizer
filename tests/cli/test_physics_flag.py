"""Day 14 of physics-rebuild: --physics CLI flag.

Verifies the new `--physics` flag adds an informational banner to the
briefing output without changing recommendation values. The flag is the
visible Day 14 deliverable that surfaces physics-derived metadata
(per-car evaluator weights, geometry, tyre floor pin status) to the user.
"""
from __future__ import annotations

from racingoptimizer.cli.recommend import _render_physics_banner


class _MockRec:
    """Minimal mock of SetupRecommendation for the banner test."""
    def __init__(self, parameters: dict[str, tuple[float, object]]):
        self.parameters = parameters


def test_physics_banner_includes_tyre_pressure_when_present() -> None:
    """The banner reports the tyre_cold_pressure_kpa value when set."""
    rec = _MockRec(parameters={
        "tyre_cold_pressure_kpa": (152.0, None),
    })
    out = _render_physics_banner(rec, "bmw")
    assert "PHYSICS VIEW" in out
    assert "tyre_cold_pressure_kpa: 152.0 kPa" in out
    assert "Mode 2" in out


def test_physics_banner_includes_car_geometry() -> None:
    """The banner shows per-car geometry (wheelbase, weight distribution)."""
    rec = _MockRec(parameters={})
    out = _render_physics_banner(rec, "bmw")
    assert "wheelbase=" in out
    assert "weight_dist_front=" in out


def test_physics_banner_includes_per_car_evaluator_weights() -> None:
    """Day 12b's per-car evaluator weights are surfaced."""
    rec = _MockRec(parameters={})
    out_bmw = _render_physics_banner(rec, "bmw")
    # BMW calibrated weights = (0.2, 0.8, 0.0)
    assert "(0.2, 0.8, 0.0)" in out_bmw
    out_ferrari = _render_physics_banner(rec, "ferrari")
    # Ferrari calibrated weights = (0.0, 0.0, 1.0)
    assert "(0.0, 0.0, 1.0)" in out_ferrari


def test_physics_banner_unchanged_recommendation_disclaimer() -> None:
    """The banner explicitly states recommendation values are unchanged."""
    rec = _MockRec(parameters={})
    out = _render_physics_banner(rec, "bmw")
    assert "recommendation values are unchanged" in out


def test_physics_banner_works_for_all_canonical_cars() -> None:
    """The banner doesn't crash for any of the 5 GTP cars."""
    rec = _MockRec(parameters={})
    for car in ("bmw", "cadillac", "ferrari", "acura", "porsche"):
        out = _render_physics_banner(rec, car)
        assert "PHYSICS VIEW" in out
        # Acura and Porsche fall back to default weights.


def test_physics_banner_handles_unknown_car_gracefully() -> None:
    """An unknown car key should not crash; geometry section drops."""
    rec = _MockRec(parameters={})
    out = _render_physics_banner(rec, "unknown_car")
    # The banner still emits the header and disclaimer.
    assert "PHYSICS VIEW" in out
