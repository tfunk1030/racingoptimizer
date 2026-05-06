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
    """Every target channel must be produced by either `corner_phase_states`,
    `_attach_setup_readouts`, OR `_attach_dynamic_at_speed`, otherwise the
    fitter's ``if output_channel not in sub.columns: continue`` silently
    drops the entry and the regressor is never trained.

    Per-row sources:
    * Per-(corner, phase) aggregated channels — `corner_phase_states`.
    * `setup_static_*` / `setup_aero_*` (TRACK-INVARIANT, from iRacing's
      garage calculator) — `_attach_setup_readouts`.
    * `dynamic_*_rh_at_speed_mm` (TELEMETRY-DERIVED, real ride heights at
      high-speed straight-line samples) — `_attach_dynamic_at_speed` +
      `physics.dynamic_at_speed.DYNAMIC_AT_SPEED_CHANNELS`.
    """
    import inspect

    from racingoptimizer.corner import states as states_mod
    from racingoptimizer.physics import dynamic_at_speed as dyn_mod
    from racingoptimizer.physics import fitter as fitter_mod

    states_src = inspect.getsource(states_mod)
    fitter_src = inspect.getsource(fitter_mod)
    dyn_src = inspect.getsource(dyn_mod)
    missing = [
        ch for ch in TARGET_OUTPUT_CHANNELS
        if (f'"{ch}"' not in states_src
            and f'"{ch}"' not in fitter_src
            and f'"{ch}"' not in dyn_src)
    ]
    assert not missing, (
        f"TARGET_OUTPUT_CHANNELS references columns not in the corner "
        f"aggregator, setup-readout attacher, or dynamic-at-speed "
        f"attacher: {missing}."
    )


def test_dynamic_at_speed_channels_are_targets() -> None:
    """VISION §3 ('fit from what the car actually does'): real
    telemetry-derived at-speed ride heights complement iRacing's
    setup-only calculator estimates. score.platform prefers the dynamic
    channels for the bottoming penalty when they're predicted."""
    for ch in (
        "dynamic_lf_rh_at_speed_mm",
        "dynamic_rf_rh_at_speed_mm",
        "dynamic_lr_rh_at_speed_mm",
        "dynamic_rr_rh_at_speed_mm",
        "dynamic_front_rh_at_speed_mm",
        "dynamic_rear_rh_at_speed_mm",
    ):
        assert ch in TARGET_OUTPUT_CHANNELS


def test_setup_readout_channels_are_targets() -> None:
    """VISION §5 'chase the chain' — static ride heights are
    deterministic functions of garage parameters (no track component).
    Adding them as fittable targets gives the model a clean
    setup→equilibrium signal that doesn't get confounded by
    corner-archetype variance pooled across tracks.

    `TiresAero.AeroCalculator.*` fields are deliberately excluded — that
    panel is a user-input scratchpad, not a setup readout."""
    for ch in (
        "setup_static_lf_ride_height_mm",
        "setup_static_rf_ride_height_mm",
        "setup_static_lr_ride_height_mm",
        "setup_static_rr_ride_height_mm",
    ):
        assert ch in TARGET_OUTPUT_CHANNELS


def test_aero_calculator_fields_are_NOT_targets() -> None:
    """Hard pin — `TiresAero.AeroCalculator.*` fields are USER-INPUT
    scratchpad values, not setup-derived readouts. Including them as
    fittable targets would mean the model trains on whatever ride-height
    pairs the driver happened to type in, which doesn't carry any signal
    about the actual setup."""
    for ch in (
        "setup_aero_front_rh_at_speed_mm",
        "setup_aero_rear_rh_at_speed_mm",
        "setup_aero_downforce_balance_pct",
        "setup_aero_ld_ratio",
    ):
        assert ch not in TARGET_OUTPUT_CHANNELS, (
            f"{ch} is a user-input scratchpad value, not a setup readout. "
            f"Don't include it as a fittable target."
        )
