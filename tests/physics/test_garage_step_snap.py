"""Garage click snapping and display precision for iRacing UI steps."""

from __future__ import annotations

import pytest

from racingoptimizer.explain.full_setup_card import _format_opt_value
from racingoptimizer.physics.ontology import (
    garage_step_decimal_places,
    ontology_for,
    snap_to_garage_step,
)


def test_acura_front_heave_spring_steps_in_10_n_per_mm() -> None:
    spec = ontology_for("acura")["heave_spring_rate_n_per_mm"]
    assert spec.step == 10.0
    assert snap_to_garage_step(175.0, spec) == 180.0
    assert snap_to_garage_step(172.0, spec) == 170.0


def test_garage_step_decimal_places() -> None:
    assert garage_step_decimal_places(10.0) == 0
    assert garage_step_decimal_places(0.5) == 1
    assert garage_step_decimal_places(0.001) == 3


def test_acura_torsion_bar_turns_render_thousandths() -> None:
    spec = ontology_for("acura")["torsion_bar_turns_fl"]
    assert spec.step == 0.001
    rendered = _format_opt_value(0.0986, spec.step, spec.units)
    assert rendered == "0.099 turns"
    snapped = snap_to_garage_step(0.0986, spec)
    assert snapped == pytest.approx(0.099)


def test_pushrod_renders_one_decimal() -> None:
    spec = ontology_for("acura")["pushrod_length_offset_front_mm"]
    assert spec.step == 0.5
    rendered = _format_opt_value(-26.0, spec.step, spec.units)
    assert rendered == "-26.0 mm"
    rendered_half = _format_opt_value(-24.5, spec.step, spec.units)
    assert rendered_half == "-24.5 mm"


@pytest.mark.parametrize("car", ["acura", "bmw", "cadillac", "ferrari", "porsche"])
def test_brake_bias_snaps_to_half_percent(car: str) -> None:
    """Every GTP brake bias must snap to 0.5 % clicks (the iRacing UI step).

    Regression: a recent recommendation rendered ``Brake Pressure Bias
    47.59 pct`` -- a value the user cannot type into the garage. The
    pct field was missing ``step=`` in `_common_ce_gated()` and the
    Acura override (Ferrari had it). See ``docs/accuracy-rebuild-2026-05-24/PLAN.md``
    receipts; this test pins the fix.
    """
    onto = ontology_for(car)
    if "brake_bias_pct" not in onto:
        pytest.skip(f"{car} ontology has no brake_bias_pct")
    spec = onto["brake_bias_pct"]
    assert spec.step == 0.5, (
        f"{car} brake_bias_pct missing step=0.5 -- recommendation will "
        "render values the iRacing garage UI cannot accept."
    )
    assert snap_to_garage_step(47.59, spec) == 47.5
    assert snap_to_garage_step(47.74, spec) == 47.5
    assert snap_to_garage_step(47.76, spec) == 48.0


@pytest.mark.parametrize("car", ["acura", "bmw", "cadillac", "ferrari", "porsche"])
def test_diff_preload_snaps_to_5nm(car: str) -> None:
    """Every GTP rear-diff preload must snap to 5 Nm clicks.

    Regression: a recent recommendation rendered ``Preload 82.09 Nm``
    -- a value the user cannot type into the garage. The Nm field was
    missing ``step=`` on every car except Ferrari.
    """
    onto = ontology_for(car)
    if "diff_preload_nm" not in onto:
        pytest.skip(f"{car} ontology has no diff_preload_nm")
    spec = onto["diff_preload_nm"]
    assert spec.step == 5.0, (
        f"{car} diff_preload_nm missing step=5.0 -- recommendation will "
        "render values the iRacing garage UI cannot accept."
    )
    assert snap_to_garage_step(82.09, spec) == 80.0
    assert snap_to_garage_step(82.51, spec) == 85.0
