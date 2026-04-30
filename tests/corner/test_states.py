"""Integration tests for `corner_phase_states` against a real BMW Sebring lap."""
from __future__ import annotations

from pathlib import Path

import pytest

from racingoptimizer.corner import corner_phase_states
from racingoptimizer.ingest.api import laps, learn


def _best_valid_lap(sid: str, corpus_root: Path) -> int:
    rows = laps(session_id=sid, valid_only=True, corpus_root=corpus_root)
    if rows.height == 0:
        pytest.skip("fixture has no valid laps")
    return int(rows["lap_index"][0])


def test_lap_index_minus_one_raises_value_error() -> None:
    with pytest.raises(ValueError, match="pre-grid sentinel"):
        corner_phase_states("00000000deadbeef", -1)


def test_corner_phase_states_against_real_bmw_lap(
    small_ibt: Path, tmp_corpus: Path
) -> None:
    sids = learn(small_ibt, corpus_root=tmp_corpus)
    sid = sids[0]
    lap_idx = _best_valid_lap(sid, tmp_corpus)

    out = corner_phase_states(sid, lap_idx, corpus_root=tmp_corpus)

    assert out.height > 0, "expected at least one (corner, phase) row"

    # At least one mid_corner row.
    phases_seen = set(out["phase"].to_list())
    assert "mid_corner" in phases_seen

    # Every row has samples.
    assert (out["n_samples"] > 0).all()

    # Corners contiguous starting at 0.
    distinct_corners = sorted(set(out["corner_id"].to_list()))
    assert distinct_corners == list(range(len(distinct_corners)))
    assert distinct_corners[0] == 0

    # Air density sanity envelope.
    if "air_density_mean" in out.columns:
        ad = out["air_density_mean"]
        assert ad.min() > 1.0
        assert ad.max() < 1.4

    if "track_temp_c_mean" in out.columns:
        tt = out["track_temp_c_mean"]
        assert tt.min() > 0.0
        assert tt.max() < 60.0

    # Circular wind dir mean must land in [0, 360).
    if "wind_dir_deg_mean" in out.columns:
        wd = out["wind_dir_deg_mean"]
        assert wd.min() >= 0.0
        assert wd.max() < 360.0

    # Spec §6 derived columns added in S2.1. BMW Sebring has the full
    # shock/ride-height/wheel-speed quads, so all gates are open.
    derived_required = {
        "load_transfer_asymmetry_mean",
        "traction_util_mean",
        "aero_platform_front_rh_mean_mm",
        "aero_platform_rear_rh_mean_mm",
        "aero_platform_pitch_mean_mm",
        "roll_angle_mean_rad",
        "damper_velocity_p99_mms",
        "damper_velocity_mean_mms",
        "data_quality_clean_frac",
    }
    missing = derived_required - set(out.columns)
    assert not missing, f"missing spec §6 derived columns: {sorted(missing)}"

    # data_quality_clean_frac is a fraction in [0, 1] — never a 0..100 pct.
    assert "data_quality_pct" not in out.columns, (
        "data_quality_pct was renamed to data_quality_clean_frac in S2.1"
    )
    dq = out["data_quality_clean_frac"]
    assert dq.min() >= 0.0
    assert dq.max() <= 1.0

    # Aero platform: ride heights are positive (mm), pitch = rear - front.
    front_rh = out["aero_platform_front_rh_mean_mm"]
    rear_rh = out["aero_platform_rear_rh_mean_mm"]
    pitch = out["aero_platform_pitch_mean_mm"]
    # Sanity envelope: GTP ride heights live in single/low-double digits mm.
    assert front_rh.min() > -50.0 and front_rh.max() < 200.0
    assert rear_rh.min() > -50.0 and rear_rh.max() < 200.0
    # pitch[i] should equal (rear_rh[i] - front_rh[i]) within float32 noise.
    diff = (pitch - (rear_rh - front_rh)).abs()
    assert diff.max() < 1e-3, f"pitch vs rear-front mismatch: {diff.max()}"

    # traction_util is a unitless fraction in [0, 1].
    tu = out["traction_util_mean"]
    assert tu.min() >= 0.0
    assert tu.max() <= 1.0

    # Damper velocities are non-negative mm/s and bounded by a sane envelope
    # (real GTP cars rarely sustain >2000 mm/s p99 inside a single phase).
    dvp = out["damper_velocity_p99_mms"]
    dvm = out["damper_velocity_mean_mms"]
    assert dvp.min() >= 0.0
    assert dvp.max() < 5000.0
    assert dvm.min() >= 0.0
    assert dvm.max() < 5000.0
    # mean must be <= p99 by construction.
    assert (dvm <= dvp + 1e-3).all()
    # Empirical understeer signal (S2.10) — the textbook `Speed^2` denominator
    # is gone. Per-phase mean of `SteeringWheelAngle - k_bmw * AccelLat` is
    # bounded by |max steering wheel angle| + k * |max AccelLat| (roughly
    # 2 rad + 0.06 * 25 m/s² ≈ 3.5 rad upper envelope).
    us = out["understeer_angle_mean_rad"]
    assert us.is_finite().all(), "understeer must be finite under the new formula"
    assert us.abs().max() < 3.0, (
        f"understeer signal exploded: max abs = {us.abs().max()} rad"
    )

    # Sort order: rows ordered by (corner_id, phase_order).
    phase_order = {
        "braking": 0,
        "trail_brake": 1,
        "mid_corner": 2,
        "exit": 3,
        "straight": 4,
    }
    keys = list(zip(out["corner_id"].to_list(), out["phase"].to_list(), strict=True))
    rank = [(c, phase_order[p]) for c, p in keys]
    assert rank == sorted(rank)


def test_corner_phase_states_is_deterministic(
    small_ibt: Path, tmp_corpus: Path
) -> None:
    sids = learn(small_ibt, corpus_root=tmp_corpus)
    sid = sids[0]
    lap_idx = _best_valid_lap(sid, tmp_corpus)

    a = corner_phase_states(sid, lap_idx, corpus_root=tmp_corpus)
    b = corner_phase_states(sid, lap_idx, corpus_root=tmp_corpus)
    assert a.equals(b)


def test_corner_phase_states_excludes_corner_id_minus_one(
    small_ibt: Path, tmp_corpus: Path
) -> None:
    sids = learn(small_ibt, corpus_root=tmp_corpus)
    sid = sids[0]
    lap_idx = _best_valid_lap(sid, tmp_corpus)
    out = corner_phase_states(sid, lap_idx, corpus_root=tmp_corpus)
    assert (out["corner_id"] >= 0).all()
