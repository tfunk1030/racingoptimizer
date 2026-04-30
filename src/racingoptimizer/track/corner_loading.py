"""Per-corner loading classifier (S4.5, VISION §9).

VISION §9: "How each corner loads the car differently (which corners are
front-limited, rear-limited, traction-limited, aero-limited)."

For every corner detected on `track`, stack `corner_phase_states` across
the supplied sessions / laps, then dominantly classify the corner into one of:

- ``front_limited``     — high understeer angle + low rear-shock-defl spread
                          across observations: front grip is the bottleneck.
- ``rear_limited``      — high yaw-rate at exit + low front-shock-defl spread:
                          rear breaks loose first.
- ``traction_limited``  — high traction_util_mean during EXIT + wheelspin events:
                          power exceeds rear grip.
- ``aero_limited``      — high-speed corner (mean speed > 55 m/s ≈ 200 km/h)
                          AND lat-G utilization rising with speed.
- ``mixed``             — none of the above dominate.

The output is one row per corner_id with `(classification, confidence,
n_observations)` where `confidence ∈ [0, 1]` reports how strongly the chosen
heuristic fired (the dominant signal's normalized score). Cold-start
(< 1 observation per corner) corners are dropped from the output entirely.

Channel availability is graceful: cars whose IBT exports omit per-corner
shock-deflection channels (Acura ARX-06) skip the front/rear-limited heuristic
inputs and fall through to ``traction_limited`` / ``aero_limited`` / ``mixed``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from racingoptimizer.corner import corner_phase_states
from racingoptimizer.corner.phase import Phase
from racingoptimizer.ingest import api as ingest_api

_LOG = logging.getLogger(__name__)

# Heuristic thresholds (rad / mm / m·s⁻¹ / dimensionless). All tunable; the
# defaults below are anchored to the synthetic fixtures in
# tests/track/test_corner_loading.py and the spec narrative.
_UNDERSTEER_HIGH_RAD: float = 0.05
_YAW_RATE_HIGH_RAD_S: float = 0.35
_SHOCK_SPREAD_LOW_MM: float = 5.0
_TRACTION_HIGH: float = 0.05
_AERO_SPEED_HIGH_MS: float = 55.0  # ≈ 200 km/h
_AERO_LAT_G_SCALING: float = 0.6  # corr(speed, lat_g) ≥ this → aero-limited

_CLASSIFY_OUTPUT_SCHEMA: dict[str, type[pl.DataType]] = {
    "corner_id": pl.Int32,
    "classification": pl.Utf8,
    "confidence": pl.Float64,
    "n_observations": pl.Int64,
}


def classify_corner_loading(
    track: str,
    session_ids: list[str],
    *,
    corpus_root: Path | str | None = None,
) -> pl.DataFrame:
    """Classify each detected corner on ``track`` by what loads the car most.

    Loops the supplied sessions, pulls every valid lap's
    `corner_phase_states`, stacks across `corner_id`, then runs the
    per-corner heuristic. Returns a `pl.DataFrame` with schema
    ``(corner_id, classification, confidence, n_observations)``.

    Returns an empty (schema-only) frame when no observations were collected
    — e.g. zero matching sessions, every lap missing required channels, or
    the catalog has no laps yet. Sessions that fail to load (missing
    parquet, no laps, channel mismatch) are skipped with a logged warning;
    the function does not raise on per-session failure.
    """
    stacked = _stack_corner_phase_states(track, session_ids, corpus_root=corpus_root)
    if stacked.height == 0:
        return _empty_output()

    rows: list[dict[str, object]] = []
    for corner_id, group in stacked.group_by("corner_id", maintain_order=True):
        cid = int(corner_id[0]) if isinstance(corner_id, tuple) else int(corner_id)
        if cid == -1:
            continue
        classification, confidence = _classify_one_corner(group)
        rows.append(
            {
                "corner_id": cid,
                "classification": classification,
                "confidence": float(confidence),
                "n_observations": int(group.height),
            }
        )

    if not rows:
        return _empty_output()
    return (
        pl.DataFrame(rows, schema=_CLASSIFY_OUTPUT_SCHEMA)
        .sort("corner_id", maintain_order=True)
    )


# ---- internals -----------------------------------------------------------


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(schema=_CLASSIFY_OUTPUT_SCHEMA)


def _stack_corner_phase_states(
    track: str,
    session_ids: list[str],
    *,
    corpus_root: Path | str | None,
) -> pl.DataFrame:
    """Concatenate `corner_phase_states` for every (session, valid lap) pair.

    Sessions outside ``session_ids`` are ignored — the caller controls the
    scope. Per-session / per-lap failures are logged as warnings; the loop
    continues so one bad lap does not poison the classifier.
    """
    if not session_ids:
        return pl.DataFrame()

    requested = set(session_ids)
    sessions_df = ingest_api.sessions(
        track=track, valid_only=True, corpus_root=corpus_root
    )
    if sessions_df.height == 0:
        return pl.DataFrame()

    candidate_sids = [
        sid for sid in sessions_df["session_id"].to_list() if sid in requested
    ]
    if not candidate_sids:
        return pl.DataFrame()

    frames: list[pl.DataFrame] = []
    for sid in candidate_sids:
        try:
            laps_df = ingest_api.laps(
                session_id=sid, valid_only=True, corpus_root=corpus_root
            )
        except (KeyError, FileNotFoundError) as exc:
            _LOG.warning("corner_loading: skip session=%s (%s)", sid, exc)
            continue
        for lap_idx in laps_df["lap_index"].to_list():
            try:
                cps = corner_phase_states(sid, int(lap_idx), corpus_root=corpus_root)
            except (KeyError, ValueError, FileNotFoundError) as exc:
                _LOG.warning(
                    "corner_loading: skip session=%s lap=%s (%s)", sid, lap_idx, exc
                )
                continue
            if cps.height > 0:
                frames.append(cps)

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _classify_one_corner(group: pl.DataFrame) -> tuple[str, float]:
    """Heuristic classification of one corner from its stacked phase rows.

    Returns ``(classification, confidence)``. The confidence is the dominant
    signal's normalized score in ``[0, 1]``; ``mixed`` returns 0.0 by
    convention since no heuristic dominated.
    """
    front_score = _front_limited_score(group)
    rear_score = _rear_limited_score(group)
    traction_score = _traction_limited_score(group)
    aero_score = _aero_limited_score(group)

    scores: dict[str, float] = {
        "front_limited": front_score,
        "rear_limited": rear_score,
        "traction_limited": traction_score,
        "aero_limited": aero_score,
    }
    # Pick the strongest signal; ties broken by the dict order above.
    best_label = max(scores, key=lambda k: scores[k])
    best_score = scores[best_label]
    if best_score <= 0.0:
        return "mixed", 0.0
    return best_label, float(np.clip(best_score, 0.0, 1.0))


def _front_limited_score(group: pl.DataFrame) -> float:
    """High understeer + low rear-shock-defl spread across observations."""
    understeer = _abs_mean(group, "understeer_angle_mean_rad")
    if understeer is None:
        return 0.0
    rear_spread = _spread(group, ("lr_shock_defl_p99_mm", "rr_shock_defl_p99_mm"))

    if understeer < _UNDERSTEER_HIGH_RAD:
        return 0.0
    # Saturate at 4× the threshold; gives dynamic range without rewarding outliers.
    understeer_norm = float(min(understeer / (4.0 * _UNDERSTEER_HIGH_RAD), 1.0))
    if rear_spread is None:
        # No rear-shock channels → understeer evidence alone, but discount
        # because we can't confirm low rear-shock spread.
        return 0.5 * understeer_norm
    spread_term = float(max(0.0, 1.0 - rear_spread / (2.0 * _SHOCK_SPREAD_LOW_MM)))
    return understeer_norm * spread_term


def _rear_limited_score(group: pl.DataFrame) -> float:
    """High yaw rate at EXIT + low front-shock-defl spread."""
    exit_rows = group.filter(pl.col("phase") == Phase.EXIT.value)
    if exit_rows.height == 0:
        return 0.0
    yaw_max = _abs_mean(exit_rows, "yaw_rate_max_rad_s")
    if yaw_max is None or yaw_max < _YAW_RATE_HIGH_RAD_S:
        return 0.0
    front_spread = _spread(group, ("lf_shock_defl_p99_mm", "rf_shock_defl_p99_mm"))

    yaw_norm = float(min(yaw_max / (2.0 * _YAW_RATE_HIGH_RAD_S), 1.0))
    if front_spread is None:
        return 0.5 * yaw_norm
    spread_term = float(max(0.0, 1.0 - front_spread / (2.0 * _SHOCK_SPREAD_LOW_MM)))
    return yaw_norm * spread_term


def _traction_limited_score(group: pl.DataFrame) -> float:
    """High traction_util during EXIT phase (proxies wheelspin events)."""
    exit_rows = group.filter(pl.col("phase") == Phase.EXIT.value)
    if exit_rows.height == 0 or "traction_util_mean" not in exit_rows.columns:
        return 0.0
    traction = _abs_mean(exit_rows, "traction_util_mean")
    if traction is None or traction < _TRACTION_HIGH:
        return 0.0
    # Saturate at 4× the threshold so a 0.20 traction_util reads as 1.0.
    return float(min(traction / (4.0 * _TRACTION_HIGH), 1.0))


def _aero_limited_score(group: pl.DataFrame) -> float:
    """High-speed corner whose lat-G rises with speed (downforce dominant)."""
    speeds = _values(group, "speed_mean_ms")
    lats = _values(group, "accel_lat_g_max")
    if speeds is None or lats is None or speeds.size < 2:
        return 0.0
    mean_speed = float(np.mean(speeds))
    if mean_speed < _AERO_SPEED_HIGH_MS:
        return 0.0

    speed_excess = min((mean_speed - _AERO_SPEED_HIGH_MS) / _AERO_SPEED_HIGH_MS, 1.0)

    # Without variance in both axes, corrcoef is undefined; fall back to a
    # speed-only score so a flat-speed sweep through a fast bend still scores
    # >0 instead of silently classifying as mixed.
    if float(np.std(speeds)) < 1e-6 or float(np.std(lats)) < 1e-6:
        return float(0.5 * speed_excess)

    corr = float(np.corrcoef(speeds, lats)[0, 1])
    if not np.isfinite(corr) or corr < _AERO_LAT_G_SCALING:
        return 0.0
    corr_term = float(np.clip(0.5 + 0.5 * (corr - _AERO_LAT_G_SCALING), 0.0, 1.0))
    return corr_term * (0.5 + 0.5 * speed_excess)


def _values(group: pl.DataFrame, col: str) -> np.ndarray | None:
    if col not in group.columns:
        return None
    arr = group[col].drop_nulls().to_numpy().astype(np.float64)
    return arr if arr.size > 0 else None


def _abs_mean(group: pl.DataFrame, col: str) -> float | None:
    arr = _values(group, col)
    if arr is None:
        return None
    return float(np.mean(np.abs(arr)))


def _spread(group: pl.DataFrame, cols: tuple[str, ...]) -> float | None:
    """Std-dev across the supplied per-corner shock-defl p99 columns.

    Returns None if any of the columns is missing — the caller treats this
    as "no rear/front shock evidence available" and discounts the score.
    """
    arrays: list[np.ndarray] = []
    for c in cols:
        arr = _values(group, c)
        if arr is None:
            return None
        arrays.append(arr)
    stacked = np.concatenate(arrays)
    if stacked.size < 2:
        return 0.0
    return float(np.std(stacked))


__all__ = ["classify_corner_loading"]
