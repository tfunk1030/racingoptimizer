"""Tests for wind directional decomposition (S4.6)."""
from __future__ import annotations

import math

import pytest

from racingoptimizer.physics import aero_wind_modifier, decompose_wind


class TestDecomposeWind:
    def test_zero_wind_yields_zero_components(self) -> None:
        head, cross = decompose_wind(0.0, 0.0, 0.0)
        assert head == 0.0
        assert cross == 0.0

    def test_zero_wind_arbitrary_heading(self) -> None:
        head, cross = decompose_wind(0.0, 137.5, 42.0)
        assert head == 0.0
        assert cross == 0.0

    def test_pure_headwind_when_wind_opposes_car(self) -> None:
        # Car heads north (0°); wind blows from south → wind_dir = 180°.
        # That places wind directly into the car's nose → +headwind, 0 cross.
        head, cross = decompose_wind(10.0, 180.0, 0.0)
        assert head == pytest.approx(10.0)
        assert cross == pytest.approx(0.0, abs=1e-9)

    def test_pure_headwind_arbitrary_heading(self) -> None:
        # Car heading 90° (east); wind from west = wind_dir 270°. Headwind.
        head, cross = decompose_wind(7.5, 270.0, 90.0)
        assert head == pytest.approx(7.5)
        assert cross == pytest.approx(0.0, abs=1e-9)

    def test_tailwind_when_wind_matches_car_heading(self) -> None:
        # Car heads north; wind from north → blowing same direction as nose
        # points = pushing car from behind = tailwind = -headwind component.
        head, cross = decompose_wind(8.0, 0.0, 0.0)
        assert head == pytest.approx(-8.0)
        assert cross == pytest.approx(0.0, abs=1e-9)

    def test_crosswind_from_right_is_positive(self) -> None:
        # Car heads north (0°); wind from east (wind_dir = 90°) → wind comes
        # from the car's right side. Per docstring, that's positive crosswind.
        head, cross = decompose_wind(5.0, 90.0, 0.0)
        assert head == pytest.approx(0.0, abs=1e-9)
        assert cross == pytest.approx(5.0)

    def test_crosswind_from_left_is_negative(self) -> None:
        # Car heads north; wind from west (wind_dir = 270°) = car's left side.
        # rel_deg = 270° → sin(270°) = -1 → negative crosswind.
        head, cross = decompose_wind(5.0, 270.0, 0.0)
        assert head == pytest.approx(0.0, abs=1e-9)
        assert cross == pytest.approx(-5.0)

    def test_diagonal_wind_components_sum_in_quadrature(self) -> None:
        # 45° relative wind, magnitude 10 → both components ~7.07.
        head, cross = decompose_wind(10.0, 135.0, 0.0)
        # rel_deg = 135°: head = -10·cos(135°) = +7.07; cross = 10·sin(135°) = +7.07.
        assert head == pytest.approx(10.0 * math.sqrt(2) / 2)
        assert cross == pytest.approx(10.0 * math.sqrt(2) / 2)
        # Magnitude is preserved.
        assert math.hypot(head, cross) == pytest.approx(10.0)

    def test_heading_wraparound(self) -> None:
        # car_heading 350°, wind_dir 170° → rel_deg = (170 - 350) % 360 = 180°.
        # Pure headwind.
        head, cross = decompose_wind(6.0, 170.0, 350.0)
        assert head == pytest.approx(6.0)
        assert cross == pytest.approx(0.0, abs=1e-9)


class TestAeroWindModifier:
    def test_zero_wind_neutral_modifier(self) -> None:
        df_scale, balance_shift = aero_wind_modifier(0.0, 0.0)
        assert df_scale == pytest.approx(1.0)
        assert balance_shift == pytest.approx(0.0)

    def test_headwind_doubles_air_ratio_quadruples_downforce(self) -> None:
        # baseline 60 m/s, headwind 60 m/s → V_air = 120, ratio = 2 → df = 4.
        df_scale, balance_shift = aero_wind_modifier(60.0, 0.0, baseline_speed_ms=60.0)
        assert df_scale == pytest.approx(4.0)
        assert balance_shift == pytest.approx(0.0)

    def test_headwind_doubles_v_air_ratio_quadruples_downforce(self) -> None:
        # Spec test: doubling V_air ratio quadruples downforce.
        # baseline 50, headwind 50 → ratio 2.0 → scale 4.0.
        df_scale, _ = aero_wind_modifier(50.0, 0.0, baseline_speed_ms=50.0)
        assert df_scale == pytest.approx(4.0)

    def test_tailwind_reduces_downforce(self) -> None:
        # 30 m/s tailwind on 60 m/s baseline → V_air = 30 → ratio 0.5 → df 0.25.
        df_scale, _ = aero_wind_modifier(-30.0, 0.0, baseline_speed_ms=60.0)
        assert df_scale == pytest.approx(0.25)

    def test_extreme_tailwind_is_clamped(self) -> None:
        # Tailwind exceeding baseline would give negative V_air; clamp at 0.25.
        df_scale, _ = aero_wind_modifier(-200.0, 0.0, baseline_speed_ms=60.0)
        assert df_scale == pytest.approx(0.25)

    def test_crosswind_shifts_balance(self) -> None:
        # 1 m/s crosswind on 60 m/s baseline → ~5/60 ≈ 0.083 % shift.
        _, balance_shift = aero_wind_modifier(0.0, 12.0, baseline_speed_ms=60.0)
        assert balance_shift == pytest.approx(1.0)

    def test_negative_crosswind_shifts_balance_opposite(self) -> None:
        _, balance_shift = aero_wind_modifier(0.0, -12.0, baseline_speed_ms=60.0)
        assert balance_shift == pytest.approx(-1.0)

    def test_zero_baseline_is_safe(self) -> None:
        # Pathological input: avoid div-by-zero and return neutral modifiers.
        df_scale, balance_shift = aero_wind_modifier(10.0, 5.0, baseline_speed_ms=0.0)
        assert df_scale == pytest.approx(1.0)
        assert balance_shift == pytest.approx(0.0)
