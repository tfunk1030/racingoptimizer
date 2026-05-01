"""`TARGET_OUTPUT_CHANNELS` covers VISION §3's named-physics-state surface.

VISION §3 explicitly lists shock velocities, ride heights, pitch angles,
aero balance, and understeer as the relationships the empirical model
must learn from setup variation. The fitter trains one regressor per
(corner_id, phase, channel) for every entry in `TARGET_OUTPUT_CHANNELS`,
so anything missing from this tuple is invisible to the optimizer's
score-and-search loop.

This test is the structural pin against silently dropping a channel.
"""
from __future__ import annotations

from racingoptimizer.physics.fitter import TARGET_OUTPUT_CHANNELS


def test_ride_heights_per_corner_are_targets() -> None:
    """Aero scoring (`physics/score._aero_ld_for_state`) reads
    `lf_ride_height_mean_mm` etc. — must be predictable from setup."""
    for ch in (
        "lf_ride_height_mean_mm",
        "rf_ride_height_mean_mm",
        "lr_ride_height_mean_mm",
        "rr_ride_height_mean_mm",
    ):
        assert ch in TARGET_OUTPUT_CHANNELS


def test_shock_deflections_per_corner_are_targets() -> None:
    for ch in (
        "lf_shock_defl_p99_mm",
        "rf_shock_defl_p99_mm",
        "lr_shock_defl_p99_mm",
        "rr_shock_defl_p99_mm",
    ):
        assert ch in TARGET_OUTPUT_CHANNELS


def test_damper_velocities_are_targets() -> None:
    """VISION §3: 'When you change the front heave spring from 30 to 50
    N/mm, what ACTUALLY happens to ... shock velocities ...'

    Prior to the second-pass audit's gap-B remediation, damper velocity
    columns were aggregated by `corner_phase_states` but never trained.
    """
    assert "damper_velocity_p99_mms" in TARGET_OUTPUT_CHANNELS
    assert "damper_velocity_mean_mms" in TARGET_OUTPUT_CHANNELS


def test_damper_forces_are_targets() -> None:
    """VISION §2: 'damper velocities vs forces'. The force columns
    (digressive-curve estimate from velocity, see
    `physics/damper_force.estimate_damper_force_n`) are part of the
    same physics surface and must be predictable from setup."""
    assert "damper_force_p99_n" in TARGET_OUTPUT_CHANNELS
    assert "damper_force_mean_n" in TARGET_OUTPUT_CHANNELS


def test_handling_state_channels_are_targets() -> None:
    for ch in (
        "accel_lat_g_max", "accel_lat_g_mean",
        "brake_max", "brake_mean",
        "throttle_max", "throttle_mean",
        "steering_max_rad",
        "understeer_angle_mean_rad",
    ):
        assert ch in TARGET_OUTPUT_CHANNELS


def test_targets_are_unique() -> None:
    """No duplicate channels (would double-train one regressor)."""
    assert len(TARGET_OUTPUT_CHANNELS) == len(set(TARGET_OUTPUT_CHANNELS))


def test_target_channels_are_aggregator_columns() -> None:
    """Every target channel must be produced by `corner_phase_states`,
    otherwise the fitter's `if output_channel not in sub.columns: continue`
    silently drops the entry and the regressor is never trained.
    """
    import inspect

    from racingoptimizer.corner import states as states_mod
    src = inspect.getsource(states_mod)
    missing = [
        ch for ch in TARGET_OUTPUT_CHANNELS
        if f'"{ch}"' not in src
    ]
    assert not missing, (
        f"TARGET_OUTPUT_CHANNELS references columns not in the corner "
        f"aggregator: {missing}. Either remove them or add them to "
        f"`corner_phase_states`."
    )
