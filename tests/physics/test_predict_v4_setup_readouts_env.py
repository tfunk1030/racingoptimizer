"""_predict_v4 <-> predict_setup_readouts interaction.

The aero-map fit features (W6) are the only reason _predict_v4 calls
predict_setup_readouts: it needs static-RH readouts to approximate the
map query at predict time. Those features are DISABLED by default since
2026-06-10 (AERO_MAP_FIT_FEATURES_ENABLED=False) after a held-out
ablation -- so the default path must NOT call predict_setup_readouts.

When the feature is re-enabled, the original W6 regression still bites:
predict_setup_readouts was once called single-arg (no env), which raised
TypeError and silently zeroed the gate. The enabled-path test below keeps
that guard with teeth.
"""

from __future__ import annotations

from unittest.mock import patch

import racingoptimizer.physics.aero_fit_features as aff
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner.phase import CornerPhaseKey, Phase
from racingoptimizer.physics.model import PhysicsModel


def _env() -> EnvironmentFrame:
    return EnvironmentFrame(
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


def _model() -> PhysicsModel:
    return PhysicsModel(
        car="bmw",
        session_ids=(),
        fitters={},
        baseline_setup={"heave_spring_rate_n_per_mm": 50.0},
        feature_schema_version=9,
    )


def _key() -> CornerPhaseKey:
    return CornerPhaseKey(
        session_id="s1", lap_index=0, corner_id=1, phase=Phase.MID_CORNER,
    )


def test_predict_v4_skips_setup_readouts_when_aero_disabled() -> None:
    """Default path: aero features off -> no predict_setup_readouts call."""
    assert aff.AERO_MAP_FIT_FEATURES_ENABLED is False
    model = _model()
    with patch.object(
        PhysicsModel, "predict_setup_readouts",
        return_value={"setup_static_lf_ride_height_mm": 30.0},
    ) as mock_readouts:
        out = model._predict_v4(
            {"heave_spring_rate_n_per_mm": 50.0}, _env(), _key(),
            {"corner_apex_speed_ms": 30.0, "corner_peak_lat_g": 1.5},
            track="spa_2024_up",
        )
        mock_readouts.assert_not_called()
    assert out.states == {}


def test_predict_v4_passes_env_to_predict_setup_readouts(monkeypatch) -> None:
    """Enabled path (W6 regression): single-arg call raised TypeError and
    zeroed the gate -- env must be threaded through as the 2nd positional."""
    monkeypatch.setattr(aff, "AERO_MAP_FIT_FEATURES_ENABLED", True)
    env = _env()
    model = _model()
    with patch.object(
        PhysicsModel, "predict_setup_readouts",
        return_value={"setup_static_lf_ride_height_mm": 30.0},
    ) as mock_readouts:
        out = model._predict_v4(
            {"heave_spring_rate_n_per_mm": 50.0}, env, _key(),
            {"corner_apex_speed_ms": 30.0, "corner_peak_lat_g": 1.5},
            track="spa_2024_up",
        )
        mock_readouts.assert_called_once()
        assert mock_readouts.call_args[0][0] == {"heave_spring_rate_n_per_mm": 50.0}
        assert mock_readouts.call_args[0][1] is env
    assert out.states == {}
