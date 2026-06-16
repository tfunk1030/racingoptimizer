"""Gate channel classification (2026-06-16 damper reclassification).

`damper_force_p99_n` was moved from gated to informational: it is a
deterministic transform of damper velocity (a structural fit ceiling),
and measured fit skill puts the whole damper family well outside the
setup-driven channels. These tests pin that decision so a future edit
to the threshold table doesn't silently re-gate it.
"""
from __future__ import annotations

import importlib

gate = importlib.import_module("scripts.holdout_accuracy_gate")


def test_damper_force_p99_not_gated() -> None:
    assert "damper_force_p99_n" not in gate._PER_CHANNEL_THRESHOLDS


def test_whole_damper_family_is_informational() -> None:
    for ch in gate._INFORMATIONAL_DAMPER_CHANNELS:
        assert ch not in gate._PER_CHANNEL_THRESHOLDS, f"{ch} should not be gated"


def test_setup_driven_channels_still_gated() -> None:
    # The genuinely setup-predictable channels must remain gated.
    for ch in (
        "accel_lat_g_max",
        "understeer_angle_mean_rad",
        "lf_ride_height_mean_mm",
        "rr_ride_height_mean_mm",
    ):
        assert ch in gate._PER_CHANNEL_THRESHOLDS


def test_per_channel_pass_skips_damper() -> None:
    """A damper row far over the old budget must NOT fail the gate now."""
    rows = [
        {"channel": "damper_force_p99_n", "mean_abs": 400.0,
         "normed_residual": 0.9, "actual_std": 500.0},
    ]
    ok, failed = gate._per_channel_pass(rows)
    assert ok is True
    assert failed == []
