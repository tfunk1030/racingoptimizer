"""Per-session telemetry-derived ride heights at speed.

iRacing's setup blob exposes ``AeroCalculator.FrontRhAtSpeed`` and
``RearRhAtSpeed`` — but those are STATIC estimates iRacing computes from
the garage parameters using a simplified aero model. They don't include:

* real damper compression dynamics (tunable but not in the calculator)
* track-specific straight-line speed (Spa hits 304 km/h; Laguna ~273 km/h)
* curb / surface effects on the actual high-speed pose
* driver throttle modulation (lift vs flat-out)

The car's REAL at-speed ride height is in the 60Hz telemetry — when the
car is at high speed running straight, the four-corner ``*rideHeight``
channels are the genuinely-loaded aero pose.

This module computes per-session median ride heights over the subset of
samples where the car is at high speed AND going relatively straight AND
on full throttle (so brake / corner-entry dives don't pollute the
median). Each session contributes one value per corner; the values get
broadcast across every (corner, phase) row in the joint training frame
the same way the setup readouts do. The fitter then learns the
``setup → real_at_speed_rh`` mapping from observation, not from
iRacing's calculator.

VISION §3 ("Don't hardcode spring rate formulas — fit them from what the
car actually does") — this module realises that for the at-speed pose
specifically, by going to telemetry rather than the simulator's static
estimator.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from racingoptimizer.ingest import api as ingest_api
from racingoptimizer.ingest.paths import resolve_corpus_root

# Channels emitted as session-invariant columns. Same naming convention as
# the setup readouts (`setup_aero_*_rh_at_speed_mm`) but the values are
# pulled from real telemetry rather than the iRacing calculator. Order
# is locked because the fitter records `feature_names` against it.
DYNAMIC_AT_SPEED_CHANNELS: tuple[str, ...] = (
    "dynamic_lf_rh_at_speed_mm",
    "dynamic_rf_rh_at_speed_mm",
    "dynamic_lr_rh_at_speed_mm",
    "dynamic_rr_rh_at_speed_mm",
    "dynamic_front_rh_at_speed_mm",
    "dynamic_rear_rh_at_speed_mm",
)


# Filter thresholds for "high-speed straight-line" samples. Tuned against
# the GTP corpus across all 5 cars: top 20 % of session speed gets the
# car onto the long straights at Spa / Sebring / Daytona; |LatAccel| <
# 0.3 g excludes anything still committed to a corner; throttle > 0.7
# excludes braking and lift-and-coast sections so the predicted "at
# speed" pose reflects full aero load.
_HIGH_SPEED_PERCENTILE: float = 0.80
_MAX_LAT_G: float = 0.3
_GRAVITY_M_S2: float = 9.80665
_MIN_THROTTLE: float = 0.7

# Channel names in the per-lap parquet (matching `ingest.api.lap_data`
# raw column naming — the same ones `corner_phase_states` reads).
_RH_CHANNELS: tuple[tuple[str, str], ...] = (
    ("LFrideHeight", "dynamic_lf_rh_at_speed_mm"),
    ("RFrideHeight", "dynamic_rf_rh_at_speed_mm"),
    ("LRrideHeight", "dynamic_lr_rh_at_speed_mm"),
    ("RRrideHeight", "dynamic_rr_rh_at_speed_mm"),
)


def compute_dynamic_at_speed_rh(
    session_id: str,
    *,
    corpus_root: Path | str | None = None,
) -> dict[str, float]:
    """Median ride heights at high-speed straight-line samples.

    Returns ``{channel: float_mm}`` for every channel where the session
    has enough qualifying samples (≥ 50 samples). Channels with too few
    qualifying samples are omitted from the dict — callers attach NaN
    columns and the fitter drops those rows. ``dynamic_front_rh_at_speed_mm``
    and ``dynamic_rear_rh_at_speed_mm`` are returned as the LF/RF and
    LR/RR averages when both per-corner channels are present.
    """
    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    try:
        valid_laps = ingest_api.laps(
            session_id=session_id, valid_only=True, corpus_root=root,
        )
    except Exception:
        return {}
    if valid_laps.height == 0:
        return {}

    sample_chunks: dict[str, list[np.ndarray]] = {
        c: [] for c, _alias in _RH_CHANNELS
    }
    speed_chunks: list[np.ndarray] = []
    lat_g_chunks: list[np.ndarray] = []
    throttle_chunks: list[np.ndarray] = []
    quality_chunks: list[np.ndarray] = []

    columns_to_pull = (
        ["Speed", "LatAccel", "Throttle", "data_quality_mask"]
        + [c for c, _alias in _RH_CHANNELS]
    )

    for lap_idx in valid_laps["lap_index"].to_list():
        try:
            df = ingest_api.lap_data(
                session_id, int(lap_idx),
                channels=columns_to_pull, corpus_root=root,
            )
        except (KeyError, ValueError, FileNotFoundError):
            continue
        if df.height == 0:
            continue
        # Speed is required to threshold; skip laps without it.
        if "Speed" not in df.columns:
            continue
        # The schema in the parquet may legitimately omit some columns
        # (per-car shock channel coverage varies) — only stash arrays
        # for columns that exist.
        speed_chunks.append(df["Speed"].cast(pl.Float64).to_numpy())
        lat_g_chunks.append(
            df["LatAccel"].cast(pl.Float64).to_numpy()
            if "LatAccel" in df.columns
            else np.zeros(df.height, dtype=np.float64)
        )
        throttle_chunks.append(
            df["Throttle"].cast(pl.Float64).to_numpy()
            if "Throttle" in df.columns
            else np.ones(df.height, dtype=np.float64)
        )
        quality_chunks.append(
            df["data_quality_mask"].cast(pl.Boolean).to_numpy()
            if "data_quality_mask" in df.columns
            else np.ones(df.height, dtype=bool)
        )
        for raw, _alias in _RH_CHANNELS:
            if raw in df.columns:
                sample_chunks[raw].append(
                    df[raw].cast(pl.Float64).to_numpy()
                )

    if not speed_chunks:
        return {}

    speed = np.concatenate(speed_chunks)
    lat_g_abs = np.abs(np.concatenate(lat_g_chunks)) / _GRAVITY_M_S2
    throttle = np.concatenate(throttle_chunks)
    quality = np.concatenate(quality_chunks)

    if speed.size == 0:
        return {}

    # Speed threshold = high percentile of session speed. Capped at >=
    # 30 m/s (108 km/h) so a slow / out-lap session doesn't claim its
    # cool-down speeds as "at speed".
    speed_floor = max(
        float(np.quantile(speed, _HIGH_SPEED_PERCENTILE)),
        30.0,
    )

    mask = (
        (speed >= speed_floor)
        & (lat_g_abs <= _MAX_LAT_G)
        & (throttle >= _MIN_THROTTLE)
        & quality
    )

    out: dict[str, float] = {}
    if mask.sum() < 50:
        return {}

    for raw, alias in _RH_CHANNELS:
        chunks = sample_chunks.get(raw, [])
        if not chunks:
            continue
        arr = np.concatenate(chunks)
        if arr.size != mask.size:
            # Ragged session (some laps had the channel, others didn't).
            continue
        masked = arr[mask]
        if masked.size < 50:
            continue
        # Raw iRacing channel is in METERS. Convert to mm at the
        # boundary so the alias matches reality (same fix the
        # `corner_phase_states` aggregator now applies).
        out[alias] = float(np.median(masked) * 1000.0)

    if "dynamic_lf_rh_at_speed_mm" in out and "dynamic_rf_rh_at_speed_mm" in out:
        out["dynamic_front_rh_at_speed_mm"] = (
            out["dynamic_lf_rh_at_speed_mm"]
            + out["dynamic_rf_rh_at_speed_mm"]
        ) / 2.0
    if "dynamic_lr_rh_at_speed_mm" in out and "dynamic_rr_rh_at_speed_mm" in out:
        out["dynamic_rear_rh_at_speed_mm"] = (
            out["dynamic_lr_rh_at_speed_mm"]
            + out["dynamic_rr_rh_at_speed_mm"]
        ) / 2.0

    return out


__all__ = [
    "DYNAMIC_AT_SPEED_CHANNELS",
    "compute_dynamic_at_speed_rh",
]
