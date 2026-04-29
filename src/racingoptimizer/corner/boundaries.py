"""Phase-boundary state machine.

Per detected corner, walk the BRAKING -> TRAIL_BRAKE -> MID_CORNER -> EXIT
state machine forward only, then EXIT -> STRAIGHT once the car is unloaded.
Outside-corner samples (corner_id == -1) are STRAIGHT.

Simplification vs spec §5: every corner starts in BRAKING. Phases the
predicates never satisfy (e.g. a flat-throttle corner with no real braking)
collapse to zero samples and are omitted by the downstream aggregator.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from racingoptimizer.corner.config import (
    DEFAULT_THRESHOLDS,
    G_MS2,
    PhaseThresholds,
    ms_to_samples,
)
from racingoptimizer.corner.phase import Phase


def assign_phases(
    lap_df: pl.DataFrame,
    corner_ids: np.ndarray,
    *,
    thresholds: PhaseThresholds = DEFAULT_THRESHOLDS,
) -> np.ndarray:
    n = lap_df.height
    phases = np.full(n, Phase.STRAIGHT.value, dtype="<U12")
    if n == 0:
        return phases

    brake = lap_df["Brake"].to_numpy().astype(np.float64)
    throttle = lap_df["Throttle"].to_numpy().astype(np.float64)
    steering = lap_df["SteeringWheelAngle"].to_numpy().astype(np.float64)
    lat_g = np.abs(lap_df["AccelLat"].to_numpy().astype(np.float64)) / G_MS2

    hz = thresholds.sample_rate_hz
    brake_off_hold = ms_to_samples(thresholds.brake_off_hold_ms, hz, minimum=1)
    decreasing_window = ms_to_samples(thresholds.accel_lat_decreasing_window_ms, hz, minimum=1)
    trail_brake_lat_g = 0.6 * thresholds.lat_g_entry

    unique_corners = np.unique(corner_ids[corner_ids >= 0])
    for cid in unique_corners:
        idxs = np.flatnonzero(corner_ids == int(cid))
        if idxs.size == 0:
            continue
        state = Phase.BRAKING
        brake_off_run = 0
        for j, i in enumerate(idxs):
            if state is Phase.BRAKING:
                if (
                    brake[i] > 0.0
                    and abs(steering[i]) > thresholds.steering_active_rad
                    and lat_g[i] > trail_brake_lat_g
                ):
                    state = Phase.TRAIL_BRAKE
            elif state is Phase.TRAIL_BRAKE:
                if brake[i] < thresholds.brake_off_threshold:
                    brake_off_run += 1
                    if brake_off_run >= brake_off_hold:
                        state = Phase.MID_CORNER
                        brake_off_run = 0
                else:
                    brake_off_run = 0
            elif state is Phase.MID_CORNER:
                if throttle[i] > thresholds.throttle_active_threshold and j >= decreasing_window:
                    past = idxs[j - decreasing_window]
                    if lat_g[i] < lat_g[past]:
                        state = Phase.EXIT
            elif state is Phase.EXIT:
                if (
                    abs(steering[i]) < thresholds.steering_active_rad
                    and throttle[i] > thresholds.throttle_straight_threshold
                ):
                    state = Phase.STRAIGHT
            phases[i] = state.value
    return phases
