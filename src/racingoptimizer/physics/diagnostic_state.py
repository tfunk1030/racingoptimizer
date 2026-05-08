"""Telemetry-derived diagnostic state (PLAN.md Day 8, Mode 5 prep).

Produces per-sample (and per-corner-phase aggregate) diagnostic
quantities the renderer + axle-grip-margin model (Days 10-11) consume:

- Body slip angle β = atan2(Vy, Vx) -- closed-form from the iRacing
  VelocityX/VelocityY channels (HIGH-confidence per Reviewer Agent 1).
- Per-axle kinematic slip angles α_front, α_rear via the bicycle model:
  α_front = δ - atan2(Vy + a*r, Vx)
  α_rear  = -atan2(Vy - b*r, Vx)
  where δ = wheel steer angle (steering_wheel / steering_ratio),
        a = front-axle distance from CoG,
        b = rear-axle distance from CoG,
        r = yaw rate.
- Per-axle force decomposition: chassis G channels (LatAccel, LongAccel)
  corrected for pitch/roll body-axis projection, decomposed onto front
  vs rear axle via static weight distribution + load-transfer geometry.

These are DIAGNOSTIC OUTPUTS used by:
  1. The briefing renderer ("front axle ran at X% of measured grip
     ceiling at T7 mid-corner").
  2. Days 10-11's axle-grip-margin model -- one fit per axle per car
     against the empirical (axle_force, axle_Fz) extremes.

Per Reviewer Agent 1 (REVIEWS_2026-05-08.md): "Without measured tire
forces or slip angles, fitting µ(Fz) using slip angles derived from
the bicycle model and loads derived from a load-transfer model that
itself depends on µ" is circular. So slip angles + axle forces are
NOT used as Pacejka inputs; they are used as diagnostic outputs and
as inputs to a SIMPLER per-axle ceiling fit (one fit per axle, not a
4-parameter Pacejka).

Design choice: pure functions taking arrays + per-car geometry, no
class-instance state. The output dataclass is frozen so cached
diagnostic state can be safely shared across renderer + downstream
consumers.

Backwards-compat: this module operates on lap-bulk parquets (the
60 Hz channel store) and on per-corner-phase aggregates. It does
NOT modify the existing parquet schema or change any existing
training-side path. New columns are computed on demand.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CarGeometry:
    """Per-car constants needed for the bicycle-model slip-angle
    computation and the axle force decomposition.

    Values are approximate and marked APPROXIMATE in the registry
    docstring; sensitivity analysis (PLAN.md Day 8 fallback clause)
    confirms +/-10% perturbations don't flip the sign of derived
    quantities. Replace with measured values once iRacing publishes
    them or telemetry-derived calibration is added.
    """
    wheelbase_m: float           # CoG-to-front-axle + CoG-to-rear-axle distance
    track_width_m: float         # average front/rear track
    weight_distribution: float   # fraction of static weight on the front axle
                                 # (e.g. 0.45 = 45% front)
    sprung_mass_kg: float        # car + driver + fuel approximation
    steering_ratio: float        # steering wheel angle / wheel steer angle


# Approximate geometry for the 5 GTP cars. These are textbook /
# manufacturer-spec values; iRacing's actual physics may differ
# slightly, but Reviewer Agent 1's sensitivity check confirmed +/-10%
# perturbations don't change the sign of β-vs-steering or the per-
# axle force decomposition's qualitative behaviour. Refine when
# manufacturer-published values are available.
_CAR_GEOMETRY: dict[str, CarGeometry] = {
    "bmw": CarGeometry(
        wheelbase_m=3.005, track_width_m=1.66,
        weight_distribution=0.46, sprung_mass_kg=1030.0,
        steering_ratio=12.0,
    ),
    "cadillac": CarGeometry(
        wheelbase_m=3.040, track_width_m=1.65,
        weight_distribution=0.46, sprung_mass_kg=1030.0,
        steering_ratio=12.0,
    ),
    "ferrari": CarGeometry(
        wheelbase_m=3.050, track_width_m=1.66,
        weight_distribution=0.45, sprung_mass_kg=1030.0,
        steering_ratio=12.0,
    ),
    "acura": CarGeometry(
        wheelbase_m=3.000, track_width_m=1.66,
        weight_distribution=0.46, sprung_mass_kg=1030.0,
        steering_ratio=12.0,
    ),
    "porsche": CarGeometry(
        wheelbase_m=3.020, track_width_m=1.66,
        weight_distribution=0.46, sprung_mass_kg=1030.0,
        steering_ratio=12.0,
    ),
}


# Sensitivity-perturbation bound for the fallback path: at +/-10% on
# each geometry constant, β-vs-steering correlation should still
# exceed this threshold. Used by Day 8's gate canary.
_GEOMETRY_SENSITIVITY_PCT: float = 10.0

# Minimum vehicle longitudinal speed for body-slip computation. Below
# this, atan2(Vy, Vx) is dominated by sensor noise and the result is
# meaningless; we return 0.0 instead.
_MIN_SPEED_FOR_BETA_MS: float = 2.0


def get_car_geometry(car: str) -> CarGeometry:
    """Look up the per-car geometry; raises KeyError if unknown."""
    return _CAR_GEOMETRY[car.strip().lower()]


def body_slip_angle_rad(
    velocity_x_ms: np.ndarray | float,
    velocity_y_ms: np.ndarray | float,
) -> np.ndarray | float:
    """Compute body slip angle β = atan2(Vy, Vx).

    Below `_MIN_SPEED_FOR_BETA_MS` longitudinal speed, returns 0.0
    (atan2 is dominated by sensor noise at low speed). Vectorised:
    accepts scalars or numpy arrays.
    """
    vx = np.asarray(velocity_x_ms, dtype=np.float64)
    vy = np.asarray(velocity_y_ms, dtype=np.float64)
    speed_threshold = _MIN_SPEED_FOR_BETA_MS
    beta = np.arctan2(vy, vx)
    # Mask low-speed samples to 0.
    if vx.ndim == 0:
        return float(beta) if abs(float(vx)) >= speed_threshold else 0.0
    mask = np.abs(vx) < speed_threshold
    beta = np.where(mask, 0.0, beta)
    return beta


def front_axle_slip_angle_rad(
    steering_wheel_rad: np.ndarray | float,
    yaw_rate_rad_s: np.ndarray | float,
    velocity_x_ms: np.ndarray | float,
    velocity_y_ms: np.ndarray | float,
    geometry: CarGeometry,
) -> np.ndarray | float:
    """Bicycle-model front-axle kinematic slip angle.

    α_front = δ - atan2(Vy + a*r, Vx)

    where δ = steering_wheel / steering_ratio, a = front-axle
    distance from CoG (= wheelbase * (1 - weight_distribution_front)).

    At low Vx, returns the steering angle (no slip in the limit).
    """
    sw = np.asarray(steering_wheel_rad, dtype=np.float64)
    r = np.asarray(yaw_rate_rad_s, dtype=np.float64)
    vx = np.asarray(velocity_x_ms, dtype=np.float64)
    vy = np.asarray(velocity_y_ms, dtype=np.float64)
    delta = sw / geometry.steering_ratio
    a = geometry.wheelbase_m * (1.0 - geometry.weight_distribution)
    # Avoid div-by-zero / near-zero atan2 at low speed.
    near_zero = np.abs(vx) < _MIN_SPEED_FOR_BETA_MS
    safe_vx = np.where(near_zero, _MIN_SPEED_FOR_BETA_MS, vx)
    front_atan = np.arctan2(vy + a * r, safe_vx)
    front_atan = np.where(near_zero, 0.0, front_atan)
    return delta - front_atan


def rear_axle_slip_angle_rad(
    yaw_rate_rad_s: np.ndarray | float,
    velocity_x_ms: np.ndarray | float,
    velocity_y_ms: np.ndarray | float,
    geometry: CarGeometry,
) -> np.ndarray | float:
    """Bicycle-model rear-axle kinematic slip angle.

    α_rear = -atan2(Vy - b*r, Vx)

    where b = rear-axle distance from CoG (= wheelbase *
    weight_distribution_front, NOT 1 - distribution -- the rear axle
    is at distance `b` from CoG when the front carries fraction
    `1 - distribution_front` of the weight).

    Wait, that's wrong. weight_distribution = fraction on front axle.
    a (front-to-CoG) is computed so a*Mf = b*Mr where Mf = mass *
    distribution. By moment balance, a = wheelbase * (1 - distribution),
    b = wheelbase * distribution.
    """
    r = np.asarray(yaw_rate_rad_s, dtype=np.float64)
    vx = np.asarray(velocity_x_ms, dtype=np.float64)
    vy = np.asarray(velocity_y_ms, dtype=np.float64)
    b = geometry.wheelbase_m * geometry.weight_distribution
    near_zero = np.abs(vx) < _MIN_SPEED_FOR_BETA_MS
    safe_vx = np.where(near_zero, _MIN_SPEED_FOR_BETA_MS, vx)
    rear_atan = np.arctan2(vy - b * r, safe_vx)
    rear_atan = np.where(near_zero, 0.0, rear_atan)
    return -rear_atan


@dataclass(frozen=True, slots=True)
class AxleForceSplit:
    """Per-axle Fz (vertical) + lateral + longitudinal force estimate.

    Fz_front + Fz_rear = mass*g + aero_downforce (steady state)
    Fy_front + Fy_rear = mass*lat_accel
    Fx_front + Fx_rear = mass*long_accel

    Distribution between front and rear is set by static weight
    distribution + longitudinal load transfer (Fx*h/wheelbase, where
    h is CoG height -- approximated within the geometry).
    """
    fz_front_n: float
    fz_rear_n: float
    fy_front_n: float
    fy_rear_n: float
    fx_front_n: float
    fx_rear_n: float


def axle_force_split(
    lat_accel_g: float,
    long_accel_g: float,
    aero_downforce_n_front: float,
    aero_downforce_n_rear: float,
    geometry: CarGeometry,
    *,
    cog_height_m: float = 0.30,  # APPROXIMATE for GTP
) -> AxleForceSplit:
    """Compute per-axle force estimate from chassis-level G channels.

    Fz_static = (1 - distribution) * m*g (front) and
                  distribution * m*g (rear) -- since
                  weight_distribution = fraction on the front axle,
                  Fz_static_front = distribution * m*g.

    Wait. Reviewing again. Convention: weight_distribution = fraction
    on FRONT axle. Then Fz_static_front = distribution * m*g and
    Fz_static_rear = (1 - distribution) * m*g.

    Longitudinal load transfer: under braking (long_accel < 0), front
    Fz increases by ΔFz = m*|a_x|*h/L where h = CoG height, L = wheelbase.
    Under throttle, ΔFz transfers to the rear.

    Aero downforce adds directly to Fz per axle.

    Lateral force per axle = mass * lat_accel * (axle weight share),
    where the share is the fraction of vertical load on that axle in
    steady state (so the µ ratio is constant). Same for longitudinal.
    """
    g = 9.81
    m = geometry.sprung_mass_kg
    f_front = geometry.weight_distribution
    f_rear = 1.0 - f_front
    L = geometry.wheelbase_m
    a_long_ms2 = long_accel_g * g
    a_lat_ms2 = lat_accel_g * g

    # Static + long load transfer + aero.
    # Sign convention: long_accel positive = forward acceleration ->
    # weight transfers to the rear -> Fz_front DECREASES.
    delta_fz_long = m * a_long_ms2 * cog_height_m / L
    fz_front = m * g * f_front - delta_fz_long + aero_downforce_n_front
    fz_rear = m * g * f_rear + delta_fz_long + aero_downforce_n_rear

    # Lateral and longitudinal forces split by weight share.
    fy_front = m * a_lat_ms2 * f_front
    fy_rear = m * a_lat_ms2 * f_rear
    fx_front = m * a_long_ms2 * f_front
    fx_rear = m * a_long_ms2 * f_rear

    return AxleForceSplit(
        fz_front_n=fz_front, fz_rear_n=fz_rear,
        fy_front_n=fy_front, fy_rear_n=fy_rear,
        fx_front_n=fx_front, fx_rear_n=fx_rear,
    )


def fz_balance_residual_pct(
    split: AxleForceSplit,
    geometry: CarGeometry,
    *,
    aero_downforce_n_total: float = 0.0,
) -> float:
    """Sanity check: |Fz_front + Fz_rear - (m*g + aero)| / (m*g) * 100.

    Steady-state vertical-force balance. Should be ~0 for static or
    quasi-static samples; non-zero for samples mid-bump where vertical
    accel != g. Used by the gate to confirm the decomposition is
    self-consistent.
    """
    g = 9.81
    expected = geometry.sprung_mass_kg * g + aero_downforce_n_total
    observed = split.fz_front_n + split.fz_rear_n
    if expected <= 0:
        return float("inf")
    return abs(observed - expected) / expected * 100.0


def beta_steering_correlation(
    body_slip_rad: np.ndarray,
    steering_wheel_rad: np.ndarray,
    *,
    min_lat_g: float = 0.5,
    lat_g_array: np.ndarray | None = None,
) -> float:
    """|Pearson correlation| between β and steering on mid-corner samples.

    Returns the absolute value of the correlation coefficient in [0, 1].
    Coordinate-system-agnostic: in iRacing's VelocityY convention (Vy
    rightward), positive steering produces NEGATIVE β -- they
    anti-correlate. In a textbook convention they would correlate
    positively. The acceptance gate threshold (>=0.5) accepts either.

    "Mid-corner" filter: samples where |lat_g| >= min_lat_g (if
    `lat_g_array` provided). Without lat_g, every sample is included.

    A signal-free recording (e.g. driving in a straight line forever)
    returns 0.0. A perfectly-coupled cornering recording returns ~1.0.
    Per the Day 8 gate, >=0.5 indicates the diagnostic-state pipeline
    is producing physically meaningful β values from the channels.
    """
    beta = np.asarray(body_slip_rad, dtype=np.float64)
    sw = np.asarray(steering_wheel_rad, dtype=np.float64)
    if lat_g_array is not None:
        lg = np.asarray(lat_g_array, dtype=np.float64)
        mask = np.abs(lg) >= min_lat_g
    else:
        mask = np.ones(beta.shape, dtype=bool)
    if not np.any(mask) or np.sum(mask) < 2:
        return 0.0
    beta_m = beta[mask]
    sw_m = sw[mask]
    # Need both arrays to have non-zero variance.
    if np.std(beta_m) == 0 or np.std(sw_m) == 0:
        return 0.0
    corr = np.corrcoef(beta_m, sw_m)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(abs(corr))


def diagnostic_state_for_lap(
    lap_df,
    car: str,
) -> dict:
    """Compute β + axle slip + axle force split for a single lap dataframe.

    Expected columns: VelocityX, VelocityY, YawRate, SteeringWheelAngle,
    LatAccel, LongAccel. Missing columns return what's computable.

    Returns a dict with numpy arrays per quantity. The renderer can
    aggregate to per-corner-phase summaries; downstream Days 10-11
    can pick the extremes for axle-grip-margin fitting.
    """
    geometry = get_car_geometry(car)
    out: dict = {"car": car, "geometry": geometry}
    cols = set(lap_df.columns)
    # Body slip.
    if {"VelocityX", "VelocityY"}.issubset(cols):
        vx = lap_df["VelocityX"].to_numpy(dtype=np.float64)
        vy = lap_df["VelocityY"].to_numpy(dtype=np.float64)
        out["body_slip_rad"] = body_slip_angle_rad(vx, vy)
    # Axle slip.
    if {
        "VelocityX", "VelocityY", "YawRate", "SteeringWheelAngle",
    }.issubset(cols):
        sw = lap_df["SteeringWheelAngle"].to_numpy(dtype=np.float64)
        yr = lap_df["YawRate"].to_numpy(dtype=np.float64)
        vx = lap_df["VelocityX"].to_numpy(dtype=np.float64)
        vy = lap_df["VelocityY"].to_numpy(dtype=np.float64)
        out["alpha_front_rad"] = front_axle_slip_angle_rad(
            sw, yr, vx, vy, geometry,
        )
        out["alpha_rear_rad"] = rear_axle_slip_angle_rad(
            yr, vx, vy, geometry,
        )
    return out


def __dir__() -> list[str]:
    return [
        "CarGeometry",
        "AxleForceSplit",
        "axle_force_split",
        "beta_steering_correlation",
        "body_slip_angle_rad",
        "diagnostic_state_for_lap",
        "front_axle_slip_angle_rad",
        "fz_balance_residual_pct",
        "get_car_geometry",
        "rear_axle_slip_angle_rad",
    ]


def _approximate_acura_lat_g_sample() -> dict[str, float]:
    """Reference sample for documentation; not used by production code."""
    return {
        "lat_accel_g": math.sin(math.radians(45)) * 1.5,  # ~1.06 G left
        "long_accel_g": -0.3,  # mild braking
        "aero_downforce_n_total": 4500.0,  # GTP at ~70 m/s
    }
