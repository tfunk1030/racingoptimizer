"""`_post_clamp` rounds `is_discrete=True` parameters to nearest integer.

ARB blade indices, damper clicks, and other integer-valued garage controls
must reach the user as round numbers. The DE optimizer in
`physics/recommend.py` searches the bounded interval continuously, so
without this rounding step the briefing emits values like
"anti_roll_bar_front: 3.700" that the iRacing garage UI cannot accept.

This module pins the `is_discrete` round-to-int contract end-to-end
without requiring a real PhysicsModel fit (no LFS dependency).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from racingoptimizer.cli.recommend import _post_clamp
from racingoptimizer.confidence import Confidence
from racingoptimizer.constraints import load_constraints
from racingoptimizer.context import EnvironmentFrame


def _conf(value: float) -> Confidence:
    return Confidence(value=value, lo=value, hi=value, n_samples=10, regime="confident")


def _env() -> EnvironmentFrame:
    return EnvironmentFrame(
        air_temp_c=25.0, air_density=1.2, air_pressure_mbar=1013.0,
        relative_humidity=0.5, wind_vel_ms=0.0, wind_dir_deg=0.0,
        fog_level=0.0, track_temp_c=30.0, track_wetness=0.0,
        weather_declared_wet=False, precip_type=0, skies=0,
    )


@dataclass
class _StubRecommendation:
    car: str
    track: str
    env: EnvironmentFrame
    parameters: dict[str, tuple[float, Confidence]]
    score_breakdown: dict
    untrained_parameters: tuple[str, ...]
    aero_correction_available: bool
    pinned_to_observed_median: tuple[str, ...] = ()


def _stub_model(car: str = "bmw") -> SimpleNamespace:
    return SimpleNamespace(car=car)


def _make_rec(parameters: dict[str, tuple[float, Confidence]], car: str = "bmw"):
    return _StubRecommendation(
        car=car, track="sebring", env=_env(),
        parameters=parameters,
        score_breakdown={},
        untrained_parameters=(),
        aero_correction_available=True,
    )


def test_arb_value_rounds_to_nearest_integer() -> None:
    rec = _make_rec({"anti_roll_bar_front": (3.7, _conf(3.7))})
    out, clamp_warnings, _ = _post_clamp(rec, _stub_model(), load_constraints())
    value = out.parameters["anti_roll_bar_front"][0]
    assert value == 4.0, f"expected round-to-int 4, got {value!r}"
    assert "anti_roll_bar_front" in clamp_warnings
    assert "rounded" in clamp_warnings["anti_roll_bar_front"]


def test_arb_value_below_half_rounds_down() -> None:
    rec = _make_rec({"anti_roll_bar_rear": (3.4, _conf(3.4))})
    out, _, _ = _post_clamp(rec, _stub_model(), load_constraints())
    assert out.parameters["anti_roll_bar_rear"][0] == 3.0


def test_already_integer_arb_passes_through_unchanged() -> None:
    rec = _make_rec({"anti_roll_bar_front": (3.0, _conf(3.0))})
    out, clamp_warnings, _ = _post_clamp(rec, _stub_model(), load_constraints())
    assert out.parameters["anti_roll_bar_front"][0] == 3.0
    # No rounding warning when the value was already integer.
    assert "rounded" not in clamp_warnings.get("anti_roll_bar_front", "")


def test_continuous_parameter_keeps_fractional_value() -> None:
    """Springs / wing / pressures keep their floating-point precision —
    `is_discrete=False` so they are NOT rounded to integers."""
    rec = _make_rec({"rear_wing_angle_deg": (15.5, _conf(15.5))})
    out, _, _ = _post_clamp(rec, _stub_model(), load_constraints())
    assert out.parameters["rear_wing_angle_deg"][0] == 15.5


def test_arb_rounds_then_re_clamps_inside_bounds() -> None:
    """A pre-clamp value of 0.4 would round to 0 — outside the [1, 5]
    range. The post-round re-clamp must pull it back inside."""
    rec = _make_rec({"anti_roll_bar_front": (0.4, _conf(0.4))})
    out, _, _ = _post_clamp(rec, _stub_model(), load_constraints())
    value = out.parameters["anti_roll_bar_front"][0]
    assert 1.0 <= value <= 5.0


def test_arb_rounds_at_upper_edge() -> None:
    rec = _make_rec({"anti_roll_bar_front": (4.7, _conf(4.7))})
    out, _, _ = _post_clamp(rec, _stub_model(), load_constraints())
    assert out.parameters["anti_roll_bar_front"][0] == 5.0


def test_brake_bias_is_continuous_not_rounded() -> None:
    """Brake bias is `is_discrete=False` (continuous %); 47.3 % stays 47.3."""
    rec = _make_rec({"brake_bias_pct": (47.3, _conf(47.3))})
    out, _, _ = _post_clamp(rec, _stub_model(), load_constraints())
    assert out.parameters["brake_bias_pct"][0] == 47.3


def test_diff_preload_is_continuous_not_rounded() -> None:
    rec = _make_rec({"diff_preload_nm": (75.5, _conf(75.5))})
    out, _, _ = _post_clamp(rec, _stub_model(), load_constraints())
    assert out.parameters["diff_preload_nm"][0] == 75.5
