"""Integration tests: hybrid score wired into score_breakdown / DE path."""
from __future__ import annotations

import pytest

from racingoptimizer.confidence import Confidence
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner import CornerPhaseKey, Phase
from racingoptimizer.physics.axle_grip import AxleGripCeiling
from racingoptimizer.physics.model import (
    CornerPhaseStateWithConfidence,
    PhysicsModel,
)
from racingoptimizer.physics.score import _corner_phase_objective_value


def _sparse_conf(value: float) -> Confidence:
    return Confidence(value=value, lo=value, hi=value, n_samples=0, regime="sparse")


def _minimal_model(*, ceilings: bool = True) -> PhysicsModel:
    ceilings_map = None
    if ceilings:
        ceilings_map = {
            "front": AxleGripCeiling(
                car="bmw", axle="front", mu_peak=2.5,
                n_samples=10, n_above_ceiling=1, percentile_used=95.0,
            ),
            "rear": AxleGripCeiling(
                car="bmw", axle="rear", mu_peak=2.5,
                n_samples=10, n_above_ceiling=1, percentile_used=95.0,
            ),
        }
    return PhysicsModel(
        car="bmw",
        session_ids=("s1",),
        aero_correction_available=True,
        axle_grip_ceilings=ceilings_map,
        feature_schema_version=4,
    )


def _state(lat_g: float = 2.0) -> CornerPhaseStateWithConfidence:
    cpkey = CornerPhaseKey(
        session_id="test", lap_index=1, corner_id=1, phase=Phase.MID_CORNER,
    )
    return CornerPhaseStateWithConfidence(
        corner_phase_key=cpkey,
        states={
            "accel_lat_g_max": _sparse_conf(lat_g),
            "lf_ride_height_mean_mm": _sparse_conf(30.0),
            "lr_ride_height_mean_mm": _sparse_conf(35.0),
        },
        untrained_channels=(),
    )


def test_hybrid_differs_from_surrogate_for_mid_corner() -> None:
    """Mid_corner hybrid blends physics; surrogate-only path differs."""
    model = _minimal_model()
    env = EnvironmentFrame(
        air_temp_c=25.0,
        air_density=1.2,
        air_pressure_mbar=1013.0,
        relative_humidity=0.5,
        wind_vel_ms=0.0,
        wind_dir_deg=0.0,
        fog_level=0.0,
        track_temp_c=30.0,
        track_wetness=0.0,
        weather_declared_wet=False,
        precip_type=0,
        skies=1,
    )
    setup = {"rear_wing_angle_deg": 15.0}
    weights = {1: 1.0}
    from racingoptimizer.physics.baselines import CarBaselines

    baselines = CarBaselines(
        car="bmw",
        max_lateral_g=2.0,
        understeer_scale_rad=0.15,
        yaw_rate_scale_rad_s=2.5,
        wheelspin_scale_ms=5.0,
        ride_height_variance_scale_mm=5.0,
        shock_defl_scale_mm=20.0,
        aero_grip_baseline_g=1.5,
    )
    state = _state(lat_g=2.2)
    archetype = {"corner_apex_speed_ms": 55.0}

    hybrid_val = _corner_phase_objective_value(
        model, setup, env, None, state, 1, "mid_corner",
        weights, baselines, hybrid=True, archetype=archetype,
    )
    surrogate_val = _corner_phase_objective_value(
        model, setup, env, None, state, 1, "mid_corner",
        weights, baselines, hybrid=False, archetype=archetype,
    )
    assert hybrid_val is not None
    assert surrogate_val is not None
    assert hybrid_val != pytest.approx(surrogate_val)


def test_braking_phase_is_surrogate_heavier_than_mid_corner_physics() -> None:
    """Braking gets w=0.1 physics vs mid_corner w=0.4 -- closer to surrogate."""
    model = _minimal_model()
    env = EnvironmentFrame(
        air_temp_c=25.0,
        air_density=1.2,
        air_pressure_mbar=1013.0,
        relative_humidity=0.5,
        wind_vel_ms=0.0,
        wind_dir_deg=0.0,
        fog_level=0.0,
        track_temp_c=30.0,
        track_wetness=0.0,
        weather_declared_wet=False,
        precip_type=0,
        skies=1,
    )
    setup = {"rear_wing_angle_deg": 15.0}
    weights = {1: 1.0}
    from racingoptimizer.physics.baselines import CarBaselines

    baselines = CarBaselines(
        car="bmw",
        max_lateral_g=2.0,
        understeer_scale_rad=0.15,
        yaw_rate_scale_rad_s=2.5,
        wheelspin_scale_ms=5.0,
        ride_height_variance_scale_mm=5.0,
        shock_defl_scale_mm=20.0,
        aero_grip_baseline_g=1.5,
    )
    state = _state(lat_g=2.0)
    archetype = {"corner_apex_speed_ms": 50.0}

    mid_hybrid = _corner_phase_objective_value(
        model, setup, env, None, state, 1, "mid_corner",
        weights, baselines, hybrid=True, archetype=archetype,
    )
    brake_hybrid = _corner_phase_objective_value(
        model, setup, env, None, state, 1, "braking",
        weights, baselines, hybrid=True, archetype=archetype,
    )
    mid_sur = _corner_phase_objective_value(
        model, setup, env, None, state, 1, "mid_corner",
        weights, baselines, hybrid=False, archetype=archetype,
    )
    assert mid_hybrid is not None and brake_hybrid is not None and mid_sur is not None
    mid_blend_gap = abs(mid_hybrid - mid_sur)
    brake_blend_gap = abs(brake_hybrid - mid_sur)
    assert mid_blend_gap > brake_blend_gap


def test_no_ceilings_falls_back_to_surrogate_only() -> None:
    model = _minimal_model(ceilings=False)
    env = EnvironmentFrame(
        air_temp_c=25.0,
        air_density=1.2,
        air_pressure_mbar=1013.0,
        relative_humidity=0.5,
        wind_vel_ms=0.0,
        wind_dir_deg=0.0,
        fog_level=0.0,
        track_temp_c=30.0,
        track_wetness=0.0,
        weather_declared_wet=False,
        precip_type=0,
        skies=1,
    )
    setup = {"rear_wing_angle_deg": 15.0}
    weights = {1: 1.0}
    from racingoptimizer.physics.baselines import CarBaselines

    baselines = CarBaselines(
        car="bmw",
        max_lateral_g=2.0,
        understeer_scale_rad=0.15,
        yaw_rate_scale_rad_s=2.5,
        wheelspin_scale_ms=5.0,
        ride_height_variance_scale_mm=5.0,
        shock_defl_scale_mm=20.0,
        aero_grip_baseline_g=1.5,
    )
    state = _state()
    hybrid_val = _corner_phase_objective_value(
        model, setup, env, None, state, 1, "mid_corner",
        weights, baselines, hybrid=True,
    )
    surrogate_val = _corner_phase_objective_value(
        model, setup, env, None, state, 1, "mid_corner",
        weights, baselines, hybrid=False,
    )
    assert hybrid_val == pytest.approx(surrogate_val)
