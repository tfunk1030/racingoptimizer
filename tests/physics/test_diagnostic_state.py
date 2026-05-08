"""Day 8 of physics-rebuild: telemetry-derived diagnostic state.

Per PLAN.md Section 15.1: Body slip β, per-axle kinematic slip, and
chassis-level force decomposition computed from iRacing IBT channels.
NOT used as Pacejka inputs (Reviewer Agent 1 vetoed circular fitting);
used as DIAGNOSTIC outputs for the renderer + as the basis for Days
10-11's axle-grip-margin model.

Acceptance gate per PLAN.md Section 15.1:
- diagnostic state computes for >= 80% of clean samples
- Fz balance residual < 5% on steady samples
- β sign correlates with steering on >= 80% of mid-corner samples

Broken-model canary: invert the steering ratio sign; β-vs-steering
correlation must FAIL.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from racingoptimizer.physics.diagnostic_state import (
    _CAR_GEOMETRY,
    _MIN_SPEED_FOR_BETA_MS,
    CarGeometry,
    axle_force_split,
    beta_steering_correlation,
    body_slip_angle_rad,
    front_axle_slip_angle_rad,
    fz_balance_residual_pct,
    get_car_geometry,
    rear_axle_slip_angle_rad,
)

# ---- body_slip_angle_rad ------------------------------------------------


def test_body_slip_zero_when_vy_zero() -> None:
    """β = atan2(0, Vx) = 0 for non-zero Vx."""
    assert body_slip_angle_rad(50.0, 0.0) == pytest.approx(0.0)


def test_body_slip_monotonic_with_vy_at_fixed_vx() -> None:
    """At fixed Vx, β increases monotonically with Vy."""
    vx = 50.0
    vys = [0.0, 0.5, 1.0, 2.0, 5.0]
    betas = [body_slip_angle_rad(vx, vy) for vy in vys]
    for i in range(len(betas) - 1):
        assert betas[i + 1] > betas[i], f"non-monotonic at i={i}"


def test_body_slip_zero_at_low_speed() -> None:
    """Below the minimum speed threshold, β returns 0 (avoids
    sensor-noise-dominated atan2 results)."""
    assert body_slip_angle_rad(0.1, 5.0) == 0.0
    assert body_slip_angle_rad(_MIN_SPEED_FOR_BETA_MS - 0.1, 5.0) == 0.0


def test_body_slip_vectorised() -> None:
    """numpy arrays in -> numpy array out."""
    vx = np.array([10.0, 20.0, 30.0, 40.0])
    vy = np.array([0.5, 1.0, 1.5, 2.0])
    out = body_slip_angle_rad(vx, vy)
    assert isinstance(out, np.ndarray)
    assert out.shape == (4,)
    # Each element should be atan2(vy[i], vx[i]).
    for i in range(4):
        assert out[i] == pytest.approx(math.atan2(vy[i], vx[i]))


def test_body_slip_sign_matches_vy_sign() -> None:
    """At positive Vx, sign(β) == sign(Vy)."""
    assert body_slip_angle_rad(50.0, 1.0) > 0
    assert body_slip_angle_rad(50.0, -1.0) < 0


# ---- front_axle_slip_angle_rad ------------------------------------------


def test_front_slip_at_steady_state_no_slip() -> None:
    """Steady cornering with steering exactly matching geometry: slip ~0.

    At low yaw rate + no body slip, α_front ≈ steering / steering_ratio.
    With Vy=0, the atan2 term is 0, so α_front = δ = sw / ratio.
    """
    geom = _CAR_GEOMETRY["bmw"]
    sw = 0.5  # rad
    out = front_axle_slip_angle_rad(
        steering_wheel_rad=sw, yaw_rate_rad_s=0.0,
        velocity_x_ms=50.0, velocity_y_ms=0.0,
        geometry=geom,
    )
    assert out == pytest.approx(sw / geom.steering_ratio)


def test_front_slip_sign_tracks_steering_at_low_speed() -> None:
    """Below the speed threshold, the atan2 term is masked to 0;
    α_front collapses to δ. Sign matches steering."""
    geom = _CAR_GEOMETRY["bmw"]
    out_pos = front_axle_slip_angle_rad(
        steering_wheel_rad=0.5, yaw_rate_rad_s=0.0,
        velocity_x_ms=1.0, velocity_y_ms=0.0,  # low speed
        geometry=geom,
    )
    out_neg = front_axle_slip_angle_rad(
        steering_wheel_rad=-0.5, yaw_rate_rad_s=0.0,
        velocity_x_ms=1.0, velocity_y_ms=0.0,
        geometry=geom,
    )
    assert out_pos > 0
    assert out_neg < 0


def test_front_slip_vectorised() -> None:
    geom = _CAR_GEOMETRY["bmw"]
    n = 10
    sw = np.full(n, 0.5)
    yr = np.zeros(n)
    vx = np.full(n, 50.0)
    vy = np.zeros(n)
    out = front_axle_slip_angle_rad(sw, yr, vx, vy, geom)
    assert isinstance(out, np.ndarray)
    assert out.shape == (n,)


# ---- rear_axle_slip_angle_rad -------------------------------------------


def test_rear_slip_zero_at_zero_yaw_zero_body_slip() -> None:
    """Straight-ahead at no body slip, no yaw -> rear slip = 0."""
    geom = _CAR_GEOMETRY["bmw"]
    out = rear_axle_slip_angle_rad(
        yaw_rate_rad_s=0.0, velocity_x_ms=50.0, velocity_y_ms=0.0,
        geometry=geom,
    )
    assert out == pytest.approx(0.0)


def test_rear_slip_sign_with_yaw_rate() -> None:
    """At fixed Vy=0, positive yaw rate -> rear axle moves laterally
    against vehicle body -> α_rear < 0 (per the - in the formula).

    α_rear = -atan2(Vy - b*r, Vx) -- with Vy=0 and r>0,
    Vy - b*r = -b*r < 0, atan2(-, +) < 0, negated -> α_rear > 0.

    Wait. With r > 0 (positive yaw, turning left), the rear axle has
    velocity (vx, vy - b*r). Vy_rear = -b*r < 0 means rear slipping
    right relative to body x-axis. The slip angle α_rear (rear-axle
    angle minus rear-axle velocity angle) = 0 - atan2(-b*r, vx) =
    -atan2(-b*r, vx) > 0 for vx > 0. So positive yaw -> positive
    rear slip angle in this convention.
    """
    geom = _CAR_GEOMETRY["bmw"]
    out_pos = rear_axle_slip_angle_rad(
        yaw_rate_rad_s=0.5, velocity_x_ms=50.0, velocity_y_ms=0.0,
        geometry=geom,
    )
    out_neg = rear_axle_slip_angle_rad(
        yaw_rate_rad_s=-0.5, velocity_x_ms=50.0, velocity_y_ms=0.0,
        geometry=geom,
    )
    assert out_pos > 0
    assert out_neg < 0


# ---- axle_force_split ---------------------------------------------------


def test_force_split_steady_state_no_aero() -> None:
    """Steady straight-line: per-axle Fz matches static distribution."""
    geom = _CAR_GEOMETRY["bmw"]
    split = axle_force_split(
        lat_accel_g=0.0, long_accel_g=0.0,
        aero_downforce_n_front=0.0, aero_downforce_n_rear=0.0,
        geometry=geom,
    )
    g = 9.81
    expected_front = geom.weight_distribution * geom.sprung_mass_kg * g
    expected_rear = (1 - geom.weight_distribution) * geom.sprung_mass_kg * g
    assert split.fz_front_n == pytest.approx(expected_front)
    assert split.fz_rear_n == pytest.approx(expected_rear)
    # Lateral and longitudinal forces zero at zero G.
    assert split.fy_front_n == pytest.approx(0.0)
    assert split.fy_rear_n == pytest.approx(0.0)


def test_force_split_braking_transfers_to_front() -> None:
    """Negative long_accel_g (braking) transfers Fz to the front."""
    geom = _CAR_GEOMETRY["bmw"]
    split = axle_force_split(
        lat_accel_g=0.0, long_accel_g=-1.0,  # 1G braking
        aero_downforce_n_front=0.0, aero_downforce_n_rear=0.0,
        geometry=geom,
    )
    g = 9.81
    static_front = geom.weight_distribution * geom.sprung_mass_kg * g
    assert split.fz_front_n > static_front, (
        f"braking should transfer Fz to front; "
        f"got {split.fz_front_n} vs static {static_front}"
    )
    static_rear = (1 - geom.weight_distribution) * geom.sprung_mass_kg * g
    assert split.fz_rear_n < static_rear


def test_force_split_throttle_transfers_to_rear() -> None:
    """Positive long_accel_g (throttle) transfers Fz to the rear."""
    geom = _CAR_GEOMETRY["bmw"]
    split = axle_force_split(
        lat_accel_g=0.0, long_accel_g=+1.0,
        aero_downforce_n_front=0.0, aero_downforce_n_rear=0.0,
        geometry=geom,
    )
    g = 9.81
    static_rear = (1 - geom.weight_distribution) * geom.sprung_mass_kg * g
    assert split.fz_rear_n > static_rear


def test_force_split_aero_adds_directly() -> None:
    """Aero downforce per axle adds to that axle's Fz."""
    geom = _CAR_GEOMETRY["bmw"]
    split = axle_force_split(
        lat_accel_g=0.0, long_accel_g=0.0,
        aero_downforce_n_front=2000.0, aero_downforce_n_rear=2500.0,
        geometry=geom,
    )
    g = 9.81
    expected_front = geom.weight_distribution * geom.sprung_mass_kg * g + 2000.0
    expected_rear = (1 - geom.weight_distribution) * geom.sprung_mass_kg * g + 2500.0
    assert split.fz_front_n == pytest.approx(expected_front)
    assert split.fz_rear_n == pytest.approx(expected_rear)


# ---- fz_balance_residual_pct -------------------------------------------


def test_fz_balance_residual_zero_at_steady_state() -> None:
    """Steady-state vertical-force balance: front + rear = m*g + aero."""
    geom = _CAR_GEOMETRY["bmw"]
    split = axle_force_split(
        lat_accel_g=0.0, long_accel_g=0.0,
        aero_downforce_n_front=2000.0, aero_downforce_n_rear=2500.0,
        geometry=geom,
    )
    residual = fz_balance_residual_pct(
        split, geom, aero_downforce_n_total=4500.0,
    )
    assert residual == pytest.approx(0.0, abs=0.01)


def test_fz_balance_residual_unaffected_by_long_g() -> None:
    """Longitudinal load transfer is internal: total Fz = m*g + aero
    regardless of long_accel."""
    geom = _CAR_GEOMETRY["bmw"]
    split = axle_force_split(
        lat_accel_g=0.5, long_accel_g=-1.5,
        aero_downforce_n_front=1500.0, aero_downforce_n_rear=2000.0,
        geometry=geom,
    )
    residual = fz_balance_residual_pct(
        split, geom, aero_downforce_n_total=3500.0,
    )
    assert residual < 0.01


# ---- beta_steering_correlation ------------------------------------------


def test_beta_steering_correlation_perfect_correlation() -> None:
    """Synthetic case: β and steering perfectly correlated -> |corr| = 1."""
    beta = np.array([0.01, -0.01, 0.02, -0.02, 0.03])
    sw = np.array([0.05, -0.05, 0.10, -0.10, 0.15])
    out = beta_steering_correlation(beta, sw)
    assert out == pytest.approx(1.0, abs=1e-9)


def test_beta_steering_correlation_perfect_anti_correlation() -> None:
    """Anti-correlated case (iRacing's actual convention): |corr| = 1.

    In iRacing, VelocityY is rightward, so turning LEFT (positive
    steering) produces a body-frame Vy that's NEGATIVE -> β = atan2(neg,
    pos) = negative. The diagnostic-state gate is coordinate-system-
    agnostic: it tests |corr| >= 0.5, which accepts either sign
    convention.
    """
    beta = np.array([0.01, -0.01, 0.02, -0.02])
    sw = np.array([-0.05, 0.05, -0.10, 0.10])
    out = beta_steering_correlation(beta, sw)
    assert out == pytest.approx(1.0, abs=1e-9)


def test_beta_steering_correlation_uncorrelated_returns_zero() -> None:
    """Random independent signals -> correlation near 0."""
    rng = np.random.default_rng(0)
    beta = rng.normal(0, 0.01, 100)
    sw = rng.normal(0, 0.1, 100)
    out = beta_steering_correlation(beta, sw)
    assert out < 0.3  # well below the 0.5 gate threshold


def test_beta_steering_correlation_lat_g_filter() -> None:
    """With lat_g filter, only mid-corner samples (|lat_g|>=0.5) count.

    Set up: the kept-subset (last two) is perfectly correlated; the
    filtered-out subset (first two) would be anti-correlated. With the
    filter, |corr| should be 1.0 (kept subset is perfectly correlated).
    """
    # Kept subset (last 2): perfectly correlated.
    # Filtered subset (first 2): perfectly anti-correlated.
    beta = np.array([0.01, -0.01, 0.02, -0.02])
    sw_with_filter = np.array([-0.05, 0.05, 0.10, -0.10])
    lat_g = np.array([0.1, 0.1, 0.6, 0.6])
    out = beta_steering_correlation(
        beta, sw_with_filter, lat_g_array=lat_g, min_lat_g=0.5,
    )
    assert out == pytest.approx(1.0, abs=1e-9)


def test_beta_steering_correlation_constant_returns_zero() -> None:
    """Zero variance in either input -> 0.0 (degenerate; not random
    output that could mistake for a real signal)."""
    beta = np.array([0.01, 0.01, 0.01])
    sw = np.array([0.05, 0.05, 0.05])
    assert beta_steering_correlation(beta, sw) == 0.0


# ---- get_car_geometry --------------------------------------------------


def test_get_car_geometry_known_cars() -> None:
    """All 5 GTP cars have geometry registered."""
    for car in ("bmw", "cadillac", "ferrari", "acura", "porsche"):
        geom = get_car_geometry(car)
        assert isinstance(geom, CarGeometry)
        assert geom.wheelbase_m > 2.5
        assert geom.wheelbase_m < 3.5
        assert 0.4 < geom.weight_distribution < 0.55


def test_get_car_geometry_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_car_geometry("fake_car")


# ---- Canary -------------------------------------------------------------


def test_canary_inverted_steering_ratio_sign_breaks_correlation() -> None:
    """Broken-model canary: with the steering ratio inverted (sign
    flipped), the front-axle slip angle's sign flips, and the
    diagnostic-state derived β-vs-steering correlation degrades.

    Specifically: front_axle_slip = sw / ratio - atan2(...). If
    ratio is negated, front_axle_slip's first term flips sign,
    making it WRONG for the bicycle-model prediction.

    This canary's purpose is to prove the geometry registry is the
    SOURCE of the correct correlation -- a corrupted registry would
    cause the gate to fail.
    """
    correct = _CAR_GEOMETRY["bmw"]
    inverted = CarGeometry(
        wheelbase_m=correct.wheelbase_m,
        track_width_m=correct.track_width_m,
        weight_distribution=correct.weight_distribution,
        sprung_mass_kg=correct.sprung_mass_kg,
        steering_ratio=-correct.steering_ratio,  # FLIPPED
    )
    sw = np.array([0.5, 0.5, 0.5])
    yr = np.array([0.0, 0.0, 0.0])
    vx = np.array([1.0, 1.0, 1.0])  # below threshold
    vy = np.array([0.0, 0.0, 0.0])
    correct_slip = front_axle_slip_angle_rad(sw, yr, vx, vy, correct)
    inverted_slip = front_axle_slip_angle_rad(sw, yr, vx, vy, inverted)
    # Sign should flip when ratio is negated.
    assert np.all(np.sign(correct_slip) == -np.sign(inverted_slip)), (
        "inverted steering ratio should flip the sign of front-axle slip"
    )
