"""Sub-utilization + aggregator unit tests (spec §6)."""
from __future__ import annotations

from racingoptimizer.confidence import Confidence
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner import CornerPhaseKey, Phase
from racingoptimizer.physics.baselines import CarBaselines
from racingoptimizer.physics.model import CornerPhaseStateWithConfidence
from racingoptimizer.physics.score import (
    aero_eff,
    aggregate_utilization,
    balance,
    grip,
    platform,
    stability,
    traction,
)

# Synthetic baselines pin the score-normalisation scales the unit-test
# arithmetic relies on. Real per-car baselines come from the training
# corpus via `derive_baselines`; these are the equivalents of the old
# hardcoded literals in score.py so the assertions stay readable.
_BASELINES = CarBaselines(
    car="bmw",
    max_lateral_g=1.5,
    understeer_scale_rad=0.1,
    yaw_rate_scale_rad_s=2.0,
    wheelspin_scale_ms=5.0,
    ride_height_variance_scale_mm=5.0,
    shock_defl_scale_mm=25.0,
    aero_grip_baseline_g=1.5,
)


def _conf(value: float, *, regime: str = "confident", n: int = 100) -> Confidence:
    half = 0.05
    return Confidence(
        value=value, lo=value - half, hi=value + half,
        n_samples=n, regime=regime,  # type: ignore[arg-type]
    )


def _env() -> EnvironmentFrame:
    return EnvironmentFrame(
        air_density=1.225, track_temp_c=24.0, wind_vel_ms=2.0,
        wind_dir_deg=180.0, track_wetness=0.0,
    )


def _state(
    channels: dict[str, float], *, regime: str = "confident",
) -> CornerPhaseStateWithConfidence:
    states = {ch: _conf(v, regime=regime) for ch, v in channels.items()}
    return CornerPhaseStateWithConfidence(
        corner_phase_key=CornerPhaseKey(
            session_id="s", lap_index=0, corner_id=0, phase=Phase.MID_CORNER,
        ),
        states=states,
        untrained_channels=(),
    )


def test_grip_no_aero_uses_baseline() -> None:
    # max_g baseline = 1.5; lat_g = 0.75 -> util = 0.5.
    state = _state({"accel_lat_g_max": 0.75})
    util, conf = grip(state, _env(), aero=None, baselines=_BASELINES)
    assert util == 0.5
    assert isinstance(conf, Confidence)


def test_grip_clips_at_one() -> None:
    state = _state({"accel_lat_g_max": 99.0})
    util, _ = grip(state, _env(), aero=None, baselines=_BASELINES)
    assert util == 1.0


def test_grip_missing_channel_returns_neutral_sparse() -> None:
    state = _state({})
    util, conf = grip(state, _env(), aero=None, baselines=_BASELINES)
    assert util == 0.5
    assert conf.regime == "sparse"


def test_balance_neutral_understeer_full_score() -> None:
    state = _state({"understeer_angle_mean_rad": 0.0})
    util, _ = balance(state, _env(), aero=None, baselines=_BASELINES)
    assert util == 1.0


def test_balance_strong_understeer_zero_score() -> None:
    state = _state({"understeer_angle_mean_rad": 0.5})
    util, _ = balance(state, _env(), aero=None, baselines=_BASELINES)
    assert util == 0.0


def test_balance_half_scale() -> None:
    state = _state({"understeer_angle_mean_rad": 0.05})
    util, _ = balance(state, _env(), aero=None, baselines=_BASELINES)
    assert util == 0.5


def test_stability_yaw_rate_path() -> None:
    state = _state({"yaw_rate_max_rad_s": 1.0})  # half of 2.0 -> util 0.5
    util, _ = stability(state, _env(), aero=None, baselines=_BASELINES)
    assert util == 0.5


def test_stability_yaw_at_limit_zero_score() -> None:
    state = _state({"yaw_rate_max_rad_s": 5.0})
    util, _ = stability(state, _env(), aero=None, baselines=_BASELINES)
    assert util == 0.0


def test_stability_fallback_uses_lat_g_spread() -> None:
    state = _state({
        "accel_lat_g_max": 1.0,
        "accel_lat_g_mean": 0.7,
    })
    util, _ = stability(state, _env(), aero=None, baselines=_BASELINES)
    # 1 - clip(|1.0 - 0.7|, 0, 1) = 0.7
    assert abs(util - 0.7) < 1e-9


def test_traction_zero_diff_full_score() -> None:
    state = _state({"wheel_speed_max_diff_ms": 0.0})
    util, _ = traction(state, _env(), aero=None, baselines=_BASELINES)
    assert util == 1.0


def test_traction_high_diff_zero_score() -> None:
    state = _state({"wheel_speed_max_diff_ms": 10.0})
    util, _ = traction(state, _env(), aero=None, baselines=_BASELINES)
    assert util == 0.0


def test_traction_missing_channel_returns_neutral_sparse() -> None:
    state = _state({})
    util, conf = traction(state, _env(), aero=None, baselines=_BASELINES)
    assert util == 0.5
    assert conf.regime == "sparse"


def test_aero_eff_no_aero_returns_neutral_sparse() -> None:
    state = _state({})
    util, conf = aero_eff(state, _env(), aero=None, baselines=_BASELINES)
    assert util == 0.5
    assert conf.regime == "sparse"


def test_platform_uniform_ride_heights_full_score() -> None:
    state = _state({
        "lf_ride_height_mean_mm": 30.0,
        "rf_ride_height_mean_mm": 30.0,
        "lr_ride_height_mean_mm": 30.0,
        "rr_ride_height_mean_mm": 30.0,
    })
    util, _ = platform(state, _env(), aero=None, baselines=_BASELINES)
    assert util == 1.0


def test_platform_high_variance_low_score() -> None:
    # variance = mean((v - mean)^2). With 20/30/40/50 -> mean=35, var=125.
    state = _state({
        "lf_ride_height_mean_mm": 20.0,
        "rf_ride_height_mean_mm": 30.0,
        "lr_ride_height_mean_mm": 40.0,
        "rr_ride_height_mean_mm": 50.0,
    })
    util, _ = platform(state, _env(), aero=None, baselines=_BASELINES)
    # variance/5.0 clipped to 1.0 -> util = 0.0.
    assert util == 0.0


def test_platform_no_ride_heights_neutral_sparse() -> None:
    state = _state({})
    util, conf = platform(state, _env(), aero=None, baselines=_BASELINES)
    assert util == 0.5
    assert conf.regime == "sparse"


def test_aggregate_utilization_phase_weighted() -> None:
    state = _state({
        "accel_lat_g_max": 0.75,             # grip = 0.5
        "understeer_angle_mean_rad": 0.0,    # balance = 1.0
        "yaw_rate_max_rad_s": 1.0,           # stability = 0.5
        "wheel_speed_max_diff_ms": 0.0,      # traction = 1.0
        "lf_ride_height_mean_mm": 30.0,      # platform = 1.0
        "rf_ride_height_mean_mm": 30.0,
        "lr_ride_height_mean_mm": 30.0,
        "rr_ride_height_mean_mm": 30.0,
    })
    util, conf = aggregate_utilization(
        state, Phase.MID_CORNER, _env(), aero=None, baselines=_BASELINES,
    )
    # MID_CORNER weights: grip 0.4, balance 0.35, stability 0.10, traction 0.05,
    # platform 0.10. aero_eff weight is 0.0; aero_eff would be 0.5 sparse but
    # weight zero so excluded.
    expected = 0.4 * 0.5 + 0.35 * 1.0 + 0.10 * 0.5 + 0.05 * 1.0 + 0.10 * 1.0
    assert abs(util - expected) < 1e-9
    assert isinstance(conf, Confidence)


def test_aggregate_utilization_marks_sparse_when_significant_sub_sparse() -> None:
    # MID_CORNER weights grip at 0.4 (>0.1). Make grip sparse via missing channel.
    state = _state({
        "understeer_angle_mean_rad": 0.0,
    })
    _, conf = aggregate_utilization(
        state, Phase.MID_CORNER, _env(), aero=None, baselines=_BASELINES,
    )
    assert conf.regime == "sparse"


def test_score_setup_no_lap_time_reference() -> None:
    """Spec §6 hard rule: lap time NEVER appears in score_setup or recommend."""
    from pathlib import Path
    physics_dir = (
        Path(__file__).resolve().parents[2]
        / "src" / "racingoptimizer" / "physics"
    )
    for path in (physics_dir / "score.py", physics_dir / "recommend.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert "lap_time" not in text, f"lap_time reference found in {path.name}"
        assert "laptime" not in text, f"laptime reference found in {path.name}"
