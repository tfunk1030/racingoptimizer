from __future__ import annotations

from pathlib import Path

import pytest

from racingoptimizer.constraints.loader import (
    ConstraintsParseError,
    ConstraintsTable,
    load_constraints,
)


@pytest.fixture
def table() -> ConstraintsTable:
    return load_constraints()


def test_default_rear_wing_angle(table: ConstraintsTable) -> None:
    assert table.bounds("default", "rear_wing_angle_deg") == (12.0, 17.0)


def test_acura_rear_wing_override(table: ConstraintsTable) -> None:
    assert table.bounds("acura", "rear_wing_angle_deg") == (6.0, 10.0)


def test_bmw_falls_back_to_default(table: ConstraintsTable) -> None:
    assert table.bounds("bmw", "rear_wing_angle_deg") == (12.0, 17.0)


def test_other_cars_fall_back_to_default(table: ConstraintsTable) -> None:
    for car in ("cadillac", "ferrari", "porsche"):
        assert table.bounds(car, "rear_wing_angle_deg") == (12.0, 17.0)


def test_suspension_deflections_parsed(table: ConstraintsTable) -> None:
    assert table.bounds("default", "heave_spring_mm") == (0.6, 25.0)
    assert table.bounds("default", "heave_slider_mm") == (25.0, 45.0)


def test_static_ride_height_per_corner(table: ConstraintsTable) -> None:
    assert table.bounds("default", "static_ride_height_front_mm") == (30.0, 80.0)
    assert table.bounds("default", "static_ride_height_rear_mm") == (30.0, 80.0)


def test_tyre_cold_pressure(table: ConstraintsTable) -> None:
    # Floor is 152 kPa — confirmed iRacing GTP minimum cold-set pressure.
    # Higher floors (e.g. 165) silently clamp the recommender's training-
    # baseline-pinned output to an illegal-high value when every observed
    # session ran 152 kPa. See clamp-warning regression in test_recommend.py.
    assert table.bounds("default", "tyre_cold_pressure_kpa") == (152.0, 220.0)


def test_todo_placeholders_return_none(table: ConstraintsTable) -> None:
    """Parameters whose `constraints.md` row is still ``<TODO: from iRacing UI>``.

    Note: ARB blades, brake bias, diff preload, and the camber rows now have
    estimated bounds (annotated as such in `constraints.md`). When iRacing-
    UI capture invalidates an estimate, edit `constraints.md` accordingly —
    do NOT add the row back to this test list.
    """
    assert table.bounds("default", "damper_lsc_fl") is None
    assert table.bounds("default", "diff_coast_ratio_pct") is None
    assert table.bounds("default", "diff_power_ratio_pct") is None
    # Toe is in mm but the loader's "Toe" section is degree-based — kept TODO
    # until the units mismatch is resolved.
    assert table.bounds("default", "toe_rr_deg") is None
    assert table.bounds("default", "brake_duct_front") is None
    assert table.bounds("default", "corner_weight_fl_kg") is None
    assert table.bounds("default", "throttle_brake_mapping") is None


def test_unknown_parameter_returns_none(table: ConstraintsTable) -> None:
    assert table.bounds("default", "made_up_thing") is None
    assert "made_up_thing" not in table.parameters()


def test_parameters_includes_bounded_and_todo(table: ConstraintsTable) -> None:
    params = table.parameters()
    for required in (
        "rear_wing_angle_deg",
        "tyre_cold_pressure_kpa",
        "heave_spring_mm",
        "heave_slider_mm",
        "static_ride_height_front_mm",
        "static_ride_height_rear_mm",
        "damper_lsc_fl",
        "damper_hsc_rr",
        "damper_lsr_fl",
        "damper_hsr_rr",
        "anti_roll_bar_front",
        "anti_roll_bar_rear",
        "brake_bias_pct",
        "diff_preload_nm",
        "diff_coast_ratio_pct",
        "diff_power_ratio_pct",
        "camber_fl_deg",
        "toe_fl_deg",
        "brake_duct_front",
        "brake_duct_rear",
        "corner_weight_fl_kg",
        "throttle_brake_mapping",
    ):
        assert required in params, f"missing parameter: {required}"


def test_per_car_override_does_not_leak(table: ConstraintsTable) -> None:
    # Acura override is for wing only — heave spring still falls back to default.
    assert table.bounds("acura", "heave_spring_mm") == (0.6, 25.0)


def test_malformed_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "constraints.md"
    bad.write_text(
        "## Defaults\n"
        "### Rear wing angle\n"
        "| min | max |\n"
        "this is not a separator\n"
        "| 12.0 | 17.0 |\n",
        encoding="utf-8",
    )
    with pytest.raises(ConstraintsParseError):
        load_constraints(bad)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_constraints(tmp_path / "nope.md")


def test_load_from_explicit_path(tmp_path: Path) -> None:
    src = tmp_path / "constraints.md"
    src.write_text(
        "## Defaults\n"
        "### Rear wing angle\n"
        "| min | max |\n"
        "| --- | --- |\n"
        "| 5.0 | 9.0 |\n"
        "## Per-car overrides\n",
        encoding="utf-8",
    )
    t = load_constraints(src)
    assert t.bounds("default", "rear_wing_angle_deg") == (5.0, 9.0)
