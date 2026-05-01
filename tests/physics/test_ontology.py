from __future__ import annotations

import json

import pytest

from racingoptimizer.constraints import load_constraints
from racingoptimizer.physics.ontology import (
    ACURA,
    BMW,
    CADILLAC,
    FERRARI,
    PORSCHE,
    fittable_parameters,
    ontology_for,
    parameters,
    setup_value,
)


@pytest.mark.parametrize("car", ["acura", "bmw", "cadillac", "ferrari", "porsche"])
def test_ontology_covers_five_cars(car: str) -> None:
    onto = ontology_for(car)
    assert len(onto) > 0
    # Bounded families that constraints.md actually clamps must be present.
    for name in (
        "rear_wing_angle_deg",
        "tyre_cold_pressure_kpa",
        "static_ride_height_front_mm",
        "static_ride_height_rear_mm",
        "heave_spring_mm",
        "heave_slider_mm",
    ):
        assert name in onto, f"{name} missing for {car}"
        assert onto[name].fittable is True


@pytest.mark.parametrize("car", ["acura", "bmw", "cadillac", "ferrari", "porsche"])
def test_ce_gated_families_present_but_unfittable(car: str) -> None:
    onto = ontology_for(car)
    assert "anti_roll_bar_front" in onto
    assert onto["anti_roll_bar_front"].fittable is False
    # Damper LSC FL specifically — every car has a damper.
    assert "damper_lsc_fl" in onto
    assert onto["damper_lsc_fl"].fittable is False


def test_unknown_car_raises() -> None:
    with pytest.raises(KeyError, match="unknown car"):
        ontology_for("mclaren")


def test_setup_value_happy_path_bmw() -> None:
    setup = {
        "TiresAero": {
            "AeroSettings": {"RearWingAngle": "17 deg"},
            "LeftFront": {"StartingPressure": "152 kPa"},
        },
        "Chassis": {
            "Front": {
                "HeaveSpringDefl": "24.7 mm 97.7 mm",
                "HeaveSliderDefl": "41.8 mm 200.0 mm",
            },
            "LeftFront": {"RideHeight": "30.1 mm"},
            "LeftRear": {"RideHeight": "47.6 mm"},
        },
    }
    assert setup_value("bmw", "rear_wing_angle_deg", setup) == pytest.approx(17.0)
    assert setup_value("bmw", "tyre_cold_pressure_kpa", setup) == pytest.approx(152.0)
    assert setup_value("bmw", "heave_spring_mm", setup) == pytest.approx(24.7)
    assert setup_value("bmw", "heave_slider_mm", setup) == pytest.approx(41.8)
    assert setup_value("bmw", "static_ride_height_front_mm", setup) == pytest.approx(30.1)
    assert setup_value("bmw", "static_ride_height_rear_mm", setup) == pytest.approx(47.6)


def test_setup_value_accepts_json_string() -> None:
    setup = {"TiresAero": {"AeroSettings": {"RearWingAngle": "12.5 deg"}}}
    assert setup_value("bmw", "rear_wing_angle_deg", json.dumps(setup)) == pytest.approx(12.5)


def test_setup_value_returns_none_when_path_missing() -> None:
    setup = {"TiresAero": {}}
    assert setup_value("bmw", "rear_wing_angle_deg", setup) is None
    # Acura uses HeaveDamperDefl path → None on a BMW-shaped blob.
    assert setup_value("acura", "heave_slider_mm", setup) is None


def test_setup_value_returns_none_on_garbage_blob() -> None:
    assert setup_value("bmw", "rear_wing_angle_deg", "not-json") is None


def test_setup_value_unknown_parameter_raises() -> None:
    with pytest.raises(KeyError, match="not in ontology"):
        setup_value("bmw", "unicorn_dust_pct", {})


def test_setup_value_handles_negative_signed_string() -> None:
    setup = {
        "Chassis": {
            "Front": {"HeaveSpringDefl": "-1.5 mm 50.0 mm"},
        }
    }
    assert setup_value("bmw", "heave_spring_mm", setup) == pytest.approx(-1.5)


def test_parameters_helper_sorted_and_complete() -> None:
    names = parameters("bmw")
    assert names == sorted(names)
    assert "rear_wing_angle_deg" in names


def test_fittable_parameters_only_returns_bounded_user_settable() -> None:
    """The optimizer's search list must satisfy three gates: fittable AND
    user-settable AND bounded by `constraints.md`. Calculated readouts
    (ride heights, heave deflections — `user_settable=False`) must be
    excluded even though they're still `fittable=True` (model learns the
    correlation as targets, but the driver can't type their values into
    the garage UI). CE-gated parameters with TODO bounds must also be
    excluded.
    """
    table = load_constraints()
    result = set(fittable_parameters("bmw", table))
    # Every entry the optimizer is allowed to recommend MUST be a real
    # garage input (user_settable=True) AND must have legal bounds.
    must_include = {
        "rear_wing_angle_deg",
        "tyre_cold_pressure_kpa",
        # USER-input springs / perches / pushrods that drive the calculated
        # readouts — added once `constraints.md` gained estimated bounds.
        "heave_spring_rate_n_per_mm",
        "third_spring_rate_n_per_mm",
        "rear_coil_spring_rate_n_per_mm",
        "heave_perch_offset_front_mm",
        "spring_perch_offset_rear_mm",
        "third_perch_offset_rear_mm",
        "pushrod_length_offset_front_mm",
        "pushrod_length_offset_rear_mm",
    }
    assert must_include.issubset(result), (
        f"missing expected fittable params: {must_include - result}"
    )
    # Calculated readouts: `user_settable=False`, must be excluded even
    # though `constraints.md` still has rows for them (kept for the
    # observation-envelope record).
    assert "static_ride_height_front_mm" not in result
    assert "static_ride_height_rear_mm" not in result
    assert "heave_spring_mm" not in result
    assert "heave_slider_mm" not in result
    # CE-gated must NOT show up (constraints are still <TODO>).
    assert "damper_lsc_fl" not in result
    assert "diff_coast_ratio_pct" not in result


@pytest.mark.parametrize(
    ("car", "module_dict"),
    [
        ("acura", ACURA),
        ("bmw", BMW),
        ("cadillac", CADILLAC),
        ("ferrari", FERRARI),
        ("porsche", PORSCHE),
    ],
)
def test_module_constants_match_ontology_for(car: str, module_dict: dict) -> None:
    assert ontology_for(car) is module_dict
