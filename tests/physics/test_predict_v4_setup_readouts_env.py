"""Regression: _predict_v4 must pass env into predict_setup_readouts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner.phase import CornerPhaseKey, Phase
from racingoptimizer.physics.model import PhysicsModel


def test_predict_v4_passes_env_to_predict_setup_readouts() -> None:
    """W6 regression -- single-arg call raised TypeError and zeroed the gate."""
    env = EnvironmentFrame(
        air_temp_c=20.0,
        air_density=1.225,
        air_pressure_mbar=1013.0,
        relative_humidity=0.5,
        wind_vel_ms=0.0,
        wind_dir_deg=0.0,
        fog_level=0.0,
        track_temp_c=30.0,
        track_wetness=0.0,
        weather_declared_wet=False,
        precip_type=-1,
        skies=-1,
    )
    model = PhysicsModel(
        car="bmw",
        session_ids=(),
        fitters={},
        baseline_setup={"heave_spring_rate_n_per_mm": 50.0},
        feature_schema_version=8,
    )
    key = CornerPhaseKey(
        session_id="s1",
        lap_index=0,
        corner_id=1,
        phase=Phase.MID_CORNER,
    )
    arch = {"corner_apex_speed_ms": 30.0, "corner_peak_lat_g": 1.5}

    with patch.object(
        PhysicsModel,
        "predict_setup_readouts",
        return_value={"setup_static_lf_ride_height_mm": 30.0},
    ) as mock_readouts:
        out = model._predict_v4(
            {"heave_spring_rate_n_per_mm": 50.0},
            env,
            key,
            arch,
            track="spa_2024_up",
        )
        mock_readouts.assert_called_once()
        assert mock_readouts.call_args[0][0] == {"heave_spring_rate_n_per_mm": 50.0}
        assert mock_readouts.call_args[0][1] is env
    assert out.states == {}
