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
