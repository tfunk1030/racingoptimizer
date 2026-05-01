"""Coverage for `explain.full_setup_card`.

The renderer is called unconditionally on every text-mode `optimize <car>
<track>` invocation (`cli/recommend.py:189-191`). It walks an ingested
setup YAML and tags each leaf with one of `[OPT]` / `[OPT pin]` /
`[past]` / `[readout]`. Pre-fix this module had zero tests. These tests
pin the four tag paths and the two empty-input paths.
"""
from __future__ import annotations

import json

import pytest

from racingoptimizer.confidence import Confidence
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.explain.full_setup_card import render_full_setup_card
from racingoptimizer.physics.recommendation import SetupRecommendation


def _confidence(value: float) -> Confidence:
    return Confidence(value=value, lo=value - 0.1, hi=value + 0.1, n_samples=10, regime="confident")


def _env() -> EnvironmentFrame:
    return EnvironmentFrame(
        air_temp_c=25.0, air_density=1.2, air_pressure_mbar=1013.0,
        relative_humidity=0.5, wind_vel_ms=0.0, wind_dir_deg=0.0,
        fog_level=0.0, track_temp_c=30.0, track_wetness=0.0,
        weather_declared_wet=False, precip_type=0, skies=0,
    )


def _bmw_setup_blob() -> dict:
    """Minimal but representative GTP setup YAML with all 3 panels."""
    return {
        "TiresAero": {
            "AeroSettings": {"RearWingAngle": "13 deg"},
            "LeftFront": {"StartingPressure": "152 kPa", "LastHotPressure": "168 kPa"},
            "RightFront": {"StartingPressure": "152 kPa"},
        },
        "Chassis": {
            "Front": {
                "HeaveSpring": "55 N/mm",
                "HeaveSpringDefl": "12.4 mm 50.0 mm",
                "HeaveSliderDefl": "30.5 mm 80.0 mm",
                "HeavePerchOffset": "10.0 mm",
                "PushrodLengthOffset": "0.0 mm",
                "ArbBlades": "3 click",
            },
            "LeftFront": {"RideHeight": "45.0 mm", "CornerWeight": "320 kg"},
            "LeftRear": {
                "RideHeight": "55.0 mm",
                "SpringRate": "180 N/mm",
                "SpringPerchOffset": "32.0 mm",
            },
            "Rear": {
                "ThirdSpring": "200 N/mm",
                "ThirdPerchOffset": "30.0 mm",
                "ArbBlades": "2 click",
            },
        },
        "BrakesDriveUnit": {
            "BrakeSpec": {"BrakePressureBias": "47 %"},
            "RearDiffSpec": {"Preload": "60 Nm"},
        },
    }


def _make_rec(parameters: dict[str, tuple[float, Confidence]], **kw) -> SetupRecommendation:
    return SetupRecommendation(
        car="bmw", track="sebring", env=_env(),
        parameters=parameters,
        score_breakdown={},
        untrained_parameters=(),
        aero_correction_available=True,
        **kw,
    )


def test_render_emits_three_panel_headers_with_real_setup() -> None:
    rec = _make_rec({
        "rear_wing_angle_deg": (15.5, _confidence(15.5)),
        "tyre_cold_pressure_kpa": (160.0, _confidence(160.0)),
    })
    out = render_full_setup_card(rec, car="bmw", most_recent_setup=_bmw_setup_blob())
    assert "TIRES & AERO" in out
    assert "CHASSIS" in out
    assert "BRAKES / DRIVETRAIN" in out
    # Header banner present.
    assert "FULL SETUP CARD — bmw @ sebring" in out


def test_optimizer_recommendations_are_tagged_OPT() -> None:
    """Parameters in `rec.parameters` show as [OPT] with the optimizer value."""
    rec = _make_rec({
        "rear_wing_angle_deg": (15.5, _confidence(15.5)),
        "heave_spring_rate_n_per_mm": (60.0, _confidence(60.0)),
    })
    out = render_full_setup_card(rec, car="bmw", most_recent_setup=_bmw_setup_blob())
    # Wing angle line — display uses the optimizer value (15.5), tagged [OPT].
    wing_line = next(
        line for line in out.splitlines() if "Rear Wing Angle" in line
    )
    assert "[OPT]" in wing_line
    assert "15.5" in wing_line
    # Heave spring rate — driver-input, optimizer-recommended.
    spring_line = next(
        line for line in out.splitlines() if "Heave Spring" in line and "Defl" not in line
    )
    assert "[OPT]" in spring_line
    assert "60" in spring_line


def test_pinned_parameters_get_OPT_pin_tag() -> None:
    rec = _make_rec(
        parameters={"rear_wing_angle_deg": (15.5, _confidence(15.5))},
        pinned_to_observed_median=("rear_wing_angle_deg",),
    )
    out = render_full_setup_card(rec, car="bmw", most_recent_setup=_bmw_setup_blob())
    wing_line = next(line for line in out.splitlines() if "Rear Wing Angle" in line)
    assert "[OPT pin]" in wing_line
    assert "[OPT]" not in wing_line.replace("[OPT pin]", "")


def test_calculated_readouts_get_readout_tag() -> None:
    """Hard-coded readouts (RideHeight, *Defl, CornerWeight, LastHotPressure)
    must always carry the `[readout]` tag — never `[OPT]`, never `[past]`.
    """
    rec = _make_rec({"rear_wing_angle_deg": (15.5, _confidence(15.5))})
    out = render_full_setup_card(rec, car="bmw", most_recent_setup=_bmw_setup_blob())
    for leaf in ("Ride Height", "Heave Spring Defl", "Heave Slider Defl",
                 "Corner Weight", "Last Hot Pressure"):
        matching = [line for line in out.splitlines() if leaf in line]
        assert matching, f"expected at least one {leaf!r} line"
        for line in matching:
            assert "[readout]" in line, (
                f"{leaf!r} should be tagged [readout], got: {line!r}"
            )


def test_unbounded_carry_overs_get_past_tag() -> None:
    """Setup leaves that are NOT in the optimizer's recommendation AND not
    on the calculated-readouts list show the past value tagged `[past]`."""
    rec = _make_rec({"rear_wing_angle_deg": (15.5, _confidence(15.5))})
    out = render_full_setup_card(rec, car="bmw", most_recent_setup=_bmw_setup_blob())
    # ARB blade indices weren't in `rec.parameters` → carry over from past.
    arb_lines = [line for line in out.splitlines() if "Arb Blades" in line]
    assert arb_lines, "expected ARB lines in CHASSIS panel"
    for line in arb_lines:
        assert "[past]" in line


def test_render_skips_when_no_past_setup_available() -> None:
    rec = _make_rec({"rear_wing_angle_deg": (15.5, _confidence(15.5))})
    out = render_full_setup_card(rec, car="bmw", most_recent_setup=None)
    assert "skipped" in out
    assert "no past setup" in out


def test_render_skips_when_setup_blob_unparseable() -> None:
    rec = _make_rec({"rear_wing_angle_deg": (15.5, _confidence(15.5))})
    out = render_full_setup_card(rec, car="bmw", most_recent_setup="{not json")
    assert "skipped" in out
    assert "unparseable" in out


def test_render_accepts_json_string_setup() -> None:
    """`most_recent_setup` may arrive as a JSON string (the catalog stores
    it that way). The renderer must parse it transparently."""
    rec = _make_rec({"rear_wing_angle_deg": (15.5, _confidence(15.5))})
    out = render_full_setup_card(
        rec, car="bmw", most_recent_setup=json.dumps(_bmw_setup_blob()),
    )
    assert "TIRES & AERO" in out
    assert "Rear Wing Angle" in out


def test_optimizer_never_displays_a_non_user_settable_parameter() -> None:
    """Even if a hypothetical caller stuffs a `user_settable=False` parameter
    into `rec.parameters` (e.g. `static_ride_height_front_mm`), the renderer
    must not surface it as `[OPT]`. The hard guard lives in
    `_ontology_path_index`."""
    rec = _make_rec({
        "rear_wing_angle_deg": (15.5, _confidence(15.5)),
        # Leak a calculated readout into the optimizer's parameters dict.
        "static_ride_height_front_mm": (44.4, _confidence(44.4)),
    })
    out = render_full_setup_card(rec, car="bmw", most_recent_setup=_bmw_setup_blob())
    rh_lines = [
        line for line in out.splitlines()
        if "Ride Height" in line
    ]
    assert rh_lines
    for line in rh_lines:
        assert "[OPT]" not in line, (
            "non-user-settable params must never tag as [OPT]; "
            f"got: {line!r}"
        )
        assert "[readout]" in line


@pytest.mark.parametrize("car", ["acura", "bmw", "cadillac", "ferrari", "porsche"])
def test_render_works_for_every_canonical_car(car: str) -> None:
    """The renderer should not crash on any of the 5 GTP car keys; the
    panels and tagging logic are car-agnostic."""
    rec = SetupRecommendation(
        car=car, track="sebring", env=_env(),
        parameters={"rear_wing_angle_deg": (15.5, _confidence(15.5))},
        score_breakdown={},
        untrained_parameters=(),
        aero_correction_available=True,
    )
    out = render_full_setup_card(rec, car=car, most_recent_setup=_bmw_setup_blob())
    assert f"FULL SETUP CARD — {car} @ sebring" in out
