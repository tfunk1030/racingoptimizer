import numpy as np
import polars as pl

from racingoptimizer.corner import Phase, assign_phases
from racingoptimizer.corner.config import G_MS2

SAMPLE_HZ = 60


def _trace():
    n_brake = 30
    n_trail = 20
    n_mid = 30
    n_exit = 25
    n_straight = 15
    n = n_brake + n_trail + n_mid + n_exit + n_straight

    accel_lat_g = np.zeros(n)
    brake = np.zeros(n)
    throttle = np.zeros(n)
    steering = np.zeros(n)

    # BRAKING: hard brake, no steering yet, low lat_g (below 0.3g trail predicate).
    s = 0
    e = s + n_brake
    accel_lat_g[s:e] = np.linspace(0.0, 0.2, n_brake)
    brake[s:e] = 0.7
    throttle[s:e] = 0.0
    steering[s:e] = 0.0

    # TRAIL_BRAKE: brake released partially, steering loaded, high lat_g.
    s = e
    e = s + n_trail
    accel_lat_g[s:e] = 0.8
    brake[s:e] = 0.3
    throttle[s:e] = 0.0
    steering[s:e] = 0.5

    # MID_CORNER: off the brake, low throttle, lat_g peaks then decreases.
    s = e
    e = s + n_mid
    accel_lat_g[s:e] = np.concatenate(
        [np.full(10, 0.85), np.linspace(0.85, 0.7, n_mid - 10)]
    )
    brake[s:e] = 0.0
    throttle[s:e] = 0.1
    steering[s:e] = 0.5

    # EXIT: opening throttle, lat_g monotone decreasing, steering unwinding.
    s = e
    e = s + n_exit
    accel_lat_g[s:e] = np.linspace(0.7, 0.2, n_exit)
    brake[s:e] = 0.0
    throttle[s:e] = np.linspace(0.3, 0.7, n_exit)
    steering[s:e] = np.linspace(0.5, 0.1, n_exit)

    # STRAIGHT: pinned throttle, no steering, lat_g near zero.
    s = e
    e = s + n_straight
    accel_lat_g[s:e] = 0.0
    brake[s:e] = 0.0
    throttle[s:e] = 0.8
    steering[s:e] = 0.0

    df = pl.DataFrame(
        {
            "t_s": pl.Series("t_s", np.arange(n) / SAMPLE_HZ, dtype=pl.Float64),
            "AccelLat": pl.Series("AccelLat", accel_lat_g * G_MS2, dtype=pl.Float32),
            "Brake": pl.Series("Brake", brake, dtype=pl.Float32),
            "Throttle": pl.Series("Throttle", throttle, dtype=pl.Float32),
            "SteeringWheelAngle": pl.Series("SteeringWheelAngle", steering, dtype=pl.Float32),
        }
    )
    starts = {
        "brake": 0,
        "trail": n_brake,
        "mid": n_brake + n_trail,
        "exit": n_brake + n_trail + n_mid,
        "straight": n_brake + n_trail + n_mid + n_exit,
    }
    return df, starts


def test_full_phase_walk_in_single_corner():
    df, s = _trace()
    corner_ids = np.zeros(df.height, dtype=np.int32)
    phases = assign_phases(df, corner_ids)

    # Mid-BRAKING sample: well before any trail-brake predicate fires.
    assert phases[5] == Phase.BRAKING.value
    assert phases[20] == Phase.BRAKING.value
    # Mid-TRAIL_BRAKE sample.
    assert phases[s["trail"] + 5] == Phase.TRAIL_BRAKE.value
    # MID_CORNER: skip the first ~3 samples (brake_off_hold ≈ 50ms).
    assert phases[s["mid"] + 10] == Phase.MID_CORNER.value
    # EXIT: well past the throttle-up + lat_g-decrease trigger.
    assert phases[s["exit"] + 15] == Phase.EXIT.value
    # STRAIGHT segment: Throttle=0.8, Steering=0 so EXIT->STRAIGHT triggers.
    assert phases[s["straight"] + 5] == Phase.STRAIGHT.value


def test_forward_only_no_revert_on_lat_g_dip():
    df, s = _trace()
    # Inject a dip in lat_g partway through EXIT that, if the walker were
    # bidirectional, would look like a return to MID_CORNER.
    accel_lat = df["AccelLat"].to_numpy().copy()
    dip_start = s["exit"] + 10
    dip_end = s["exit"] + 18
    accel_lat[dip_start:dip_end] = 0.85 * G_MS2
    df = df.with_columns(pl.Series("AccelLat", accel_lat, dtype=pl.Float32))

    corner_ids = np.zeros(df.height, dtype=np.int32)
    phases = assign_phases(df, corner_ids)
    # Past the dip, the walker must remain at EXIT or STRAIGHT — never MID.
    for i in range(dip_start, df.height):
        assert phases[i] in (Phase.EXIT.value, Phase.STRAIGHT.value)


def test_corner_id_minus_one_is_straight():
    df, _ = _trace()
    corner_ids = np.full(df.height, -1, dtype=np.int32)
    phases = assign_phases(df, corner_ids)
    assert (phases == Phase.STRAIGHT.value).all()
