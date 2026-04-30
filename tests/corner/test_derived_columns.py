"""S2.1 — synthetic verification of the 9 spec §6 derived corner-phase columns.

Each test constructs a fully-controlled `(corner_id, phase)`-labelled input
frame and asserts that `_aggregate` materialises the new column with the
arithmetic spelled out in `docs/superpowers/specs/2026-04-28-corner-phase-design.md`.

Why we hit `_aggregate` directly instead of going through `corner_phase_states`:
the public entry point loads from a parquet via slice A. Constructing a
synthetic parquet just to inject 5 hand-rolled samples would dwarf the
actual assertion. `_aggregate` is the function-of-record for the column
formulas, so testing it directly is the cleanest verification of the spec.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from racingoptimizer.corner.states import _aggregate

SID = "0123456789abcdef"
LAP = 7

# Common required cols every test frame needs (segment_lap was already run
# in concept; we materialise corner_id + phase directly).
_REQUIRED_BASE_COLS = (
    "t_s",
    "AccelLat",
    "Brake",
    "Throttle",
    "SteeringWheelAngle",
    "corner_id",
    "phase",
)


def _base_frame(n: int = 8) -> dict[str, np.ndarray]:
    """All-zeros base frame keyed to one corner / one phase, 60 Hz."""
    dt = 1.0 / 60.0
    return {
        "t_s": np.arange(n, dtype=np.float64) * dt,
        "AccelLat": np.full(n, 5.0, dtype=np.float64),  # avoid zero-norm cases
        "Brake": np.zeros(n, dtype=np.float64),
        "Throttle": np.zeros(n, dtype=np.float64),
        "SteeringWheelAngle": np.zeros(n, dtype=np.float64),
        "corner_id": np.zeros(n, dtype=np.int32),
        "phase": np.full(n, "mid_corner", dtype=object),
    }


def _aggregate_one(extra: dict[str, np.ndarray]) -> pl.DataFrame:
    cols = _base_frame(len(next(iter(extra.values()))))
    cols.update(extra)
    df = pl.DataFrame(cols)
    out = _aggregate(df, session_id=SID, lap_index=LAP)
    assert out.height == 1, "expected exactly one (corner, phase) row"
    return out


# ---------------------------------------------------------------------------
# 1. load_transfer_asymmetry_mean
#    formula: ((LFshockDefl + RRshockDefl) - (RFshockDefl + LRshockDefl)) * 1000
#    sign convention: positive = right-front + left-rear loaded
# ---------------------------------------------------------------------------


def test_load_transfer_asymmetry_left_diagonal_loaded() -> None:
    """LF + RR loaded (positive sign per spec §6)."""
    n = 6
    out = _aggregate_one(
        {
            "LFshockDefl": np.full(n, 0.030, dtype=np.float64),  # 30 mm
            "RRshockDefl": np.full(n, 0.020, dtype=np.float64),  # 20 mm
            "RFshockDefl": np.full(n, 0.010, dtype=np.float64),  # 10 mm
            "LRshockDefl": np.full(n, 0.005, dtype=np.float64),  # 5  mm
        }
    )
    # (30 + 20) - (10 + 5) = 35 mm
    assert out["load_transfer_asymmetry_mean"][0] == pytest.approx(35.0, abs=1e-3)


def test_load_transfer_asymmetry_zero_when_diagonals_balanced() -> None:
    n = 6
    out = _aggregate_one(
        {
            "LFshockDefl": np.full(n, 0.020, dtype=np.float64),
            "RRshockDefl": np.full(n, 0.020, dtype=np.float64),
            "RFshockDefl": np.full(n, 0.020, dtype=np.float64),
            "LRshockDefl": np.full(n, 0.020, dtype=np.float64),
        }
    )
    assert out["load_transfer_asymmetry_mean"][0] == pytest.approx(0.0, abs=1e-3)


def test_load_transfer_asymmetry_omitted_without_full_quad() -> None:
    """Acura case: drop RFshockDefl, the asym column should be omitted."""
    n = 6
    cols = _base_frame(n)
    cols["LFshockDefl"] = np.full(n, 0.030, dtype=np.float64)
    cols["LRshockDefl"] = np.full(n, 0.020, dtype=np.float64)
    cols["RRshockDefl"] = np.full(n, 0.010, dtype=np.float64)
    # RFshockDefl missing
    out = _aggregate(pl.DataFrame(cols), session_id=SID, lap_index=LAP)
    assert "load_transfer_asymmetry_mean" not in out.columns
    assert "damper_velocity_p99_mms" not in out.columns


# ---------------------------------------------------------------------------
# 2. traction_util_mean
#    formula: (max(*Speed) - min(*Speed)) / max(Speed, 1e-6), clipped [0, 1]
# ---------------------------------------------------------------------------


def test_traction_util_zero_when_wheels_match_speed() -> None:
    n = 6
    out = _aggregate_one(
        {
            "Speed": np.full(n, 60.0, dtype=np.float64),
            "LFspeed": np.full(n, 60.0, dtype=np.float64),
            "RFspeed": np.full(n, 60.0, dtype=np.float64),
            "LRspeed": np.full(n, 60.0, dtype=np.float64),
            "RRspeed": np.full(n, 60.0, dtype=np.float64),
        }
    )
    assert out["traction_util_mean"][0] == pytest.approx(0.0, abs=1e-6)


def test_traction_util_known_fraction() -> None:
    n = 6
    out = _aggregate_one(
        {
            "Speed": np.full(n, 100.0, dtype=np.float64),
            "LFspeed": np.full(n, 110.0, dtype=np.float64),  # max
            "RFspeed": np.full(n, 100.0, dtype=np.float64),
            "LRspeed": np.full(n, 100.0, dtype=np.float64),
            "RRspeed": np.full(n,  95.0, dtype=np.float64),  # min
        }
    )
    # (110 - 95) / 100 = 0.15
    assert out["traction_util_mean"][0] == pytest.approx(0.15, abs=1e-6)


def test_traction_util_clipped_at_one() -> None:
    n = 6
    out = _aggregate_one(
        {
            "Speed": np.full(n, 10.0, dtype=np.float64),
            "LFspeed": np.full(n, 50.0, dtype=np.float64),
            "RFspeed": np.full(n, 10.0, dtype=np.float64),
            "LRspeed": np.full(n, 10.0, dtype=np.float64),
            "RRspeed": np.full(n,  0.0, dtype=np.float64),
        }
    )
    # Raw ratio 5.0; clip pins to 1.0.
    assert out["traction_util_mean"][0] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3-5. aero_platform_front_rh_mean_mm, _rear_rh_mean_mm, _pitch_mean_mm
#      formulas:
#        front_rh_mm = (LFrideHeight + RFrideHeight) / 2 * 1000
#        rear_rh_mm  = (LRrideHeight + RRrideHeight) / 2 * 1000
#        pitch_mm    = rear_rh_mm - front_rh_mm
# ---------------------------------------------------------------------------


def test_aero_platform_front_rear_pitch_arithmetic() -> None:
    n = 6
    out = _aggregate_one(
        {
            "LFrideHeight": np.full(n, 0.020, dtype=np.float64),  # 20 mm
            "RFrideHeight": np.full(n, 0.024, dtype=np.float64),  # 24 mm
            "LRrideHeight": np.full(n, 0.040, dtype=np.float64),  # 40 mm
            "RRrideHeight": np.full(n, 0.050, dtype=np.float64),  # 50 mm
        }
    )
    # front = (20 + 24) / 2 = 22 mm; rear = (40 + 50) / 2 = 45 mm; pitch = 23.
    assert out["aero_platform_front_rh_mean_mm"][0] == pytest.approx(22.0, abs=1e-3)
    assert out["aero_platform_rear_rh_mean_mm"][0] == pytest.approx(45.0, abs=1e-3)
    assert out["aero_platform_pitch_mean_mm"][0] == pytest.approx(23.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 6. roll_angle_mean_rad — signed mean of Roll
# ---------------------------------------------------------------------------


def test_roll_angle_mean_signed() -> None:
    roll = np.array([-0.10, -0.05, 0.0, 0.05, 0.10, 0.15], dtype=np.float64)
    out = _aggregate_one({"Roll": roll})
    assert out["roll_angle_mean_rad"][0] == pytest.approx(roll.mean(), abs=1e-6)
    # roll_max_rad is unsigned (existing column); confirm both coexist.
    assert out["roll_max_rad"][0] == pytest.approx(0.15, abs=1e-6)


def test_roll_angle_mean_negative_when_signal_negative() -> None:
    n = 4
    roll = np.full(n, -0.20, dtype=np.float64)
    out = _aggregate_one({"Roll": roll})
    assert out["roll_angle_mean_rad"][0] == pytest.approx(-0.20, abs=1e-6)


# ---------------------------------------------------------------------------
# 7-8. damper_velocity_p99_mms / damper_velocity_mean_mms
#      Per-corner derivative of *shockDefl in m/s, * 1000 -> mm/s, abs.
#      p99 = max p99 across the four corners.
#      mean = mean of per-corner means.
# ---------------------------------------------------------------------------


def test_damper_velocities_constant_signal_is_zero() -> None:
    """If shock deflections are flat, velocity is identically zero."""
    n = 8
    out = _aggregate_one(
        {
            "LFshockDefl": np.full(n, 0.020, dtype=np.float64),
            "RFshockDefl": np.full(n, 0.022, dtype=np.float64),
            "LRshockDefl": np.full(n, 0.024, dtype=np.float64),
            "RRshockDefl": np.full(n, 0.026, dtype=np.float64),
        }
    )
    assert out["damper_velocity_p99_mms"][0] == pytest.approx(0.0, abs=1e-3)
    assert out["damper_velocity_mean_mms"][0] == pytest.approx(0.0, abs=1e-3)


def test_damper_velocities_known_ramp() -> None:
    """Ramp at 60 Hz: each corner moves N mm/sample => N * 60 mm/s velocity."""
    n = 10
    # The leading sample's diff is null and the aggregator ignores it, so
    # every per-corner column has n-1 valid samples all equal to its rate.
    lf = np.cumsum(np.full(n, 0.001))               # 1 mm per sample -> 60 mm/s
    rf = np.cumsum(np.full(n, 0.002))               # 2 mm per sample -> 120 mm/s
    lr = np.cumsum(np.full(n, 0.0005))              # 0.5 mm per sample -> 30 mm/s
    rr = np.full(n, 0.020, dtype=np.float64)        # flat -> 0 mm/s
    out = _aggregate_one(
        {
            "LFshockDefl": lf,
            "RFshockDefl": rf,
            "LRshockDefl": lr,
            "RRshockDefl": rr,
        }
    )
    # p99 = max p99 across (60, 120, 30, 0) = 120 mm/s.
    assert out["damper_velocity_p99_mms"][0] == pytest.approx(120.0, abs=1e-2)
    # mean of per-corner means = (60 + 120 + 30 + 0) / 4 = 52.5 mm/s.
    assert out["damper_velocity_mean_mms"][0] == pytest.approx(52.5, abs=1e-2)


def test_damper_velocity_oscillating_signal_p99_matches_amplitude() -> None:
    """A pure sine in shock deflection should show p99 ~ peak |dx/dt|."""
    n = 600  # 10 s of 60 Hz data
    t = np.arange(n) / 60.0
    amp_m = 0.005  # 5 mm
    freq_hz = 2.0
    omega = 2.0 * math.pi * freq_hz
    # x(t) = amp * sin(omega t); dx/dt peak = amp * omega (m/s).
    peak_mms = amp_m * omega * 1000.0
    signal = amp_m * np.sin(omega * t)
    flat = np.full(n, 0.020, dtype=np.float64)
    out = _aggregate_one(
        {
            "t_s": t,  # override t_s so dt is consistent
            "AccelLat": np.full(n, 5.0, dtype=np.float64),
            "Brake": np.zeros(n, dtype=np.float64),
            "Throttle": np.zeros(n, dtype=np.float64),
            "SteeringWheelAngle": np.zeros(n, dtype=np.float64),
            "corner_id": np.zeros(n, dtype=np.int32),
            "phase": np.full(n, "mid_corner", dtype=object),
            "LFshockDefl": signal,
            "RFshockDefl": flat,
            "LRshockDefl": flat,
            "RRshockDefl": flat,
        }
    )
    # p99 picks out the LF channel's near-peak velocity.
    assert out["damper_velocity_p99_mms"][0] == pytest.approx(peak_mms, rel=0.05)


# ---------------------------------------------------------------------------
# 9. data_quality_clean_frac — renamed from data_quality_pct, now [0, 1].
# ---------------------------------------------------------------------------


def test_data_quality_clean_frac_all_clean() -> None:
    n = 6
    out = _aggregate_one({"data_quality_mask": np.ones(n, dtype=bool)})
    assert "data_quality_pct" not in out.columns
    assert out["data_quality_clean_frac"][0] == pytest.approx(1.0, abs=1e-6)


def test_data_quality_clean_frac_half_clean() -> None:
    mask = np.array([True, True, True, False, False, False], dtype=bool)
    out = _aggregate_one({"data_quality_mask": mask})
    assert out["data_quality_clean_frac"][0] == pytest.approx(0.5, abs=1e-6)


def test_data_quality_clean_frac_all_dirty() -> None:
    n = 4
    out = _aggregate_one({"data_quality_mask": np.zeros(n, dtype=bool)})
    assert out["data_quality_clean_frac"][0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Cross-row sanity: aero_platform_pitch == rear - front for every group row.
# ---------------------------------------------------------------------------


def test_aero_pitch_consistent_across_two_phases() -> None:
    """Two corner-phase groups in one frame; pitch[i] == rear[i] - front[i]."""
    n = 12
    cols = _base_frame(n)
    # Half mid_corner, half exit, both in corner 0.
    cols["phase"] = np.array(
        ["mid_corner"] * (n // 2) + ["exit"] * (n // 2), dtype=object
    )
    # Different ride heights per half so the two rows differ.
    lf = np.concatenate([np.full(n // 2, 0.020), np.full(n // 2, 0.030)])
    rf = np.concatenate([np.full(n // 2, 0.022), np.full(n // 2, 0.032)])
    lr = np.concatenate([np.full(n // 2, 0.040), np.full(n // 2, 0.050)])
    rr = np.concatenate([np.full(n // 2, 0.044), np.full(n // 2, 0.054)])
    cols["LFrideHeight"] = lf
    cols["RFrideHeight"] = rf
    cols["LRrideHeight"] = lr
    cols["RRrideHeight"] = rr
    out = _aggregate(pl.DataFrame(cols), session_id=SID, lap_index=LAP)
    assert out.height == 2
    diff = (
        out["aero_platform_pitch_mean_mm"]
        - (out["aero_platform_rear_rh_mean_mm"] - out["aero_platform_front_rh_mean_mm"])
    ).abs()
    assert diff.max() < 1e-3
