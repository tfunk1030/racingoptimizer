"""Per-car smoke: corner_phase_states runs end-to-end on each canonical car."""
from __future__ import annotations

from pathlib import Path

import pytest

from racingoptimizer.corner import corner_phase_states
from racingoptimizer.ingest.api import laps, learn

REPO_ROOT = Path(__file__).resolve().parents[2]
IBT_DIR = REPO_ROOT / "ibtfiles"

# Filename prefixes per the iRacing CarPath convention. `normalize_car_key`
# (slice A) maps these to the canonical car keys.
CAR_PREFIXES: dict[str, tuple[str, ...]] = {
    "bmw": ("bmwlmdh",),
    "acura": ("acuraarx06gtp",),
    "cadillac": ("cadillacvseriesrgtp",),
    "ferrari": ("ferrari499p",),
    "porsche": ("porsche963gtp",),
}


def _first_fixture_for(car_key: str) -> Path | None:
    if not IBT_DIR.is_dir():
        return None
    for prefix in CAR_PREFIXES[car_key]:
        for path in sorted(IBT_DIR.glob(f"{prefix}*.ibt")):
            return path
    return None


@pytest.mark.parametrize("car_key", sorted(CAR_PREFIXES))
def test_corner_phase_states_runs_for_each_canonical_car(
    car_key: str, tmp_corpus: Path
) -> None:
    fixture = _first_fixture_for(car_key)
    if fixture is None:
        pytest.skip(f"no {car_key} fixture present in ibtfiles/")

    sids = learn(fixture, corpus_root=tmp_corpus)
    assert sids
    sid = sids[0]

    valid = laps(session_id=sid, valid_only=True, corpus_root=tmp_corpus)
    if valid.height == 0:
        pytest.skip(f"{fixture.name} has no valid laps")

    out = corner_phase_states(sid, int(valid["lap_index"][0]), corpus_root=tmp_corpus)
    assert out.height > 0, f"no corners detected for {car_key} fixture {fixture.name}"
    assert (out["corner_id"] >= 0).all()

    # The S2.1 rename is universal: data_quality_pct must NOT appear, and
    # data_quality_clean_frac (0..1 fraction) MUST appear for every car.
    assert "data_quality_pct" not in out.columns, (
        f"{car_key}: data_quality_pct renamed to data_quality_clean_frac in S2.1"
    )
    assert "data_quality_clean_frac" in out.columns
    dq = out["data_quality_clean_frac"]
    assert dq.min() >= 0.0
    assert dq.max() <= 1.0

    # Channel-conditional spec §6 columns. Acura ARX-06 telemetry drops the
    # *shockDefl set entirely (per CLAUDE.md "Acura is a known divergence"),
    # so the shock-gated columns are absent for Acura but present for every
    # other car. Drive presence off the actual schema.
    has_shock_cols = "lf_shock_defl_p99_mm" in out.columns
    has_rh_cols = "lf_ride_height_mean_mm" in out.columns
    has_wheel_speeds = car_key != "acura"  # all 4 non-Acura cars expose them

    if has_shock_cols:
        assert "load_transfer_asymmetry_mean" in out.columns
        assert "damper_velocity_p99_mms" in out.columns
        assert "damper_velocity_mean_mms" in out.columns
        assert (out["damper_velocity_p99_mms"] >= 0.0).all()
        assert (out["damper_velocity_mean_mms"] >= 0.0).all()
        # mean <= p99 by construction.
        diffs = out["damper_velocity_p99_mms"] - out["damper_velocity_mean_mms"]
        assert (diffs >= -1e-3).all()
    else:
        # Acura — gated columns are correctly omitted.
        assert "load_transfer_asymmetry_mean" not in out.columns
        assert "damper_velocity_p99_mms" not in out.columns
        assert "damper_velocity_mean_mms" not in out.columns

    if has_rh_cols:
        assert "aero_platform_front_rh_mean_mm" in out.columns
        assert "aero_platform_rear_rh_mean_mm" in out.columns
        assert "aero_platform_pitch_mean_mm" in out.columns

    if has_wheel_speeds:
        assert "traction_util_mean" in out.columns
        tu = out["traction_util_mean"]
        assert tu.min() >= 0.0
        assert tu.max() <= 1.0

    if "Roll" in out.columns or "roll_angle_mean_rad" in out.columns:
        # Roll channel is present on every fixture so the signed mean lands.
        assert "roll_angle_mean_rad" in out.columns
    # S2.10: empirical understeer signal must be present + finite + sane for
    # every car. The textbook `Speed^2` denominator is gone; the per-car
    # coefficient table guarantees a non-zero, distinguishable signal. The
    # envelope tracks max(SteeringWheelAngle) + k * max(AccelLat) — for GTP
    # cars that's ~2 rad + 0.07 * 25 m/s² ≈ 3.5 rad upper bound.
    assert "understeer_angle_mean_rad" in out.columns
    us = out["understeer_angle_mean_rad"]
    assert us.is_finite().all(), (
        f"{car_key}: understeer must be finite under the new formula"
    )
    assert us.abs().max() < 3.0, (
        f"{car_key}: understeer signal exploded: max abs = {us.abs().max()} rad"
    )
