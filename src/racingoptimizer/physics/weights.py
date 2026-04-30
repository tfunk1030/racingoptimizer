"""Per-corner time-sensitivity weights (spec §6, derivation only).

`weight_corners(track, model)` partitions held-out laps into high/low
utilization at each corner, computes the lap-time delta within env-similarity
buckets, and normalises across all corners. Lap time appears here as the
*source* of weights — never inside the optimisation objective. The grep test
in tests/physics/test_weight_corners.py asserts score.py / recommend.py
contain no lap_time reference.

When the available corpus is too thin (< 3 laps), the function returns
uniform weights so the recommender can still produce a setup.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from racingoptimizer.corner import corner_phase_states
from racingoptimizer.ingest import api as ingest_api
from racingoptimizer.physics.model import PhysicsModel


def weight_corners(
    track: str,
    model: PhysicsModel,
    *,
    env_buckets: int = 4,
    corpus_root: Path | str | None = None,
) -> dict[int, float]:
    """Per-corner weights for `track`, normalised to sum to 1.

    Lap time used here as the source of per-corner sensitivity (spec §6) —
    never inside the optimisation objective.
    """
    sessions_df = ingest_api.sessions(
        car=model.car, track=track, valid_only=True, corpus_root=corpus_root,
    )
    if sessions_df.height == 0:
        return _uniform(model)

    laps_df = ingest_api.laps(
        car=model.car, track=track, valid_only=True, corpus_root=corpus_root,
    )
    if laps_df.height < 3:
        return _uniform(model)

    rows: list[dict[str, float]] = []
    for sid, lap_idx, lap_time in zip(
        laps_df["session_id"].to_list(),
        laps_df["lap_index"].to_list(),
        laps_df["lap_time_s"].to_list(),
        strict=False,
    ):
        if lap_time is None:
            continue
        try:
            cps = corner_phase_states(sid, int(lap_idx), corpus_root=corpus_root)
        except (KeyError, ValueError, FileNotFoundError):
            continue
        if cps.height == 0 or "accel_lat_g_max" not in cps.columns:
            continue
        # air_density_mean is optional in the corner-phase frame; default to
        # 0 so all laps end up in one env bucket when telemetry lacks it.
        has_density = "air_density_mean" in cps.columns
        agg_exprs = [pl.col("accel_lat_g_max").max().alias("util_proxy")]
        if has_density:
            agg_exprs.append(
                pl.col("air_density_mean").mean().alias("air_density")
            )
        per_corner = (
            cps.group_by("corner_id")
            .agg(agg_exprs)
            .filter(pl.col("corner_id") != -1)
        )
        densities = (
            per_corner["air_density"].to_list()
            if has_density else [0.0] * per_corner.height
        )
        for corner_id, util, density in zip(
            per_corner["corner_id"].to_list(),
            per_corner["util_proxy"].to_list(),
            densities,
            strict=True,
        ):
            if util is None:
                continue
            rows.append({
                "corner_id": int(corner_id),
                "lap_time": float(lap_time),
                "util": float(util),
                "air_density": float(density) if density is not None else 0.0,
            })

    if not rows:
        return _uniform(model)

    df = pl.DataFrame(rows)
    corners = sorted({int(c) for c in df["corner_id"].to_list()})
    if not corners:
        return _uniform(model)

    # Negative sensitivities (high-util laps slower than low-util — likely
    # data noise) are clipped to zero so they don't subtract weight.
    sensitivity: dict[int, float] = {}
    for corner in corners:
        sub = df.filter(pl.col("corner_id") == corner)
        delta = (
            _bucketed_sensitivity(sub, env_buckets=env_buckets)
            if sub.height >= 2 else 0.0
        )
        sensitivity[corner] = max(delta, 0.0)

    total = sum(sensitivity.values())
    if total <= 0:
        return {c: 1.0 / len(corners) for c in corners}
    return {c: float(v / total) for c, v in sensitivity.items()}


def _bucketed_sensitivity(sub: pl.DataFrame, *, env_buckets: int) -> float:
    """Per-corner mean of (low_util_lap_time - high_util_lap_time) across env buckets."""
    densities = np.asarray(sub["air_density"].to_list(), dtype=np.float64)
    bucket_count = min(env_buckets, max(len(densities), 1))
    if bucket_count <= 1 or float(densities.std()) == 0.0:
        return _split_sensitivity(sub)

    edges = np.quantile(densities, np.linspace(0.0, 1.0, bucket_count + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return _split_sensitivity(sub)

    bucket_idx = np.clip(
        np.searchsorted(edges, densities, side="right") - 1, 0, len(edges) - 2,
    )
    deltas: list[float] = []
    for b in range(len(edges) - 1):
        mask = bucket_idx == b
        if mask.sum() < 2:
            continue
        bucket = sub.filter(pl.Series("__mask", mask.tolist()))
        deltas.append(_split_sensitivity(bucket))
    if not deltas:
        return _split_sensitivity(sub)
    return float(np.mean(deltas))


def _split_sensitivity(sub: pl.DataFrame) -> float:
    if sub.height < 2:
        return 0.0
    median = sub["util"].median()
    high = sub.filter(pl.col("util") >= median)
    low = sub.filter(pl.col("util") < median)
    if high.height == 0 or low.height == 0:
        return 0.0
    return float(low["lap_time"].mean() - high["lap_time"].mean())


def _uniform(model: PhysicsModel) -> dict[int, float]:
    """Uniform per-corner weights when the corpus is too thin to derive any.

    Stage-3 keys are 3-tuples (corner_id, phase, channel); legacy v1/v2
    keys were 4-tuples (param, corner_id, phase, channel). Handle both.
    """
    corners: set[int] = set()
    for key in model.fitters:
        if len(key) == 3:
            corners.add(int(key[0]))
        elif len(key) == 4:
            corners.add(int(key[1]))
    if not corners:
        return {}
    sorted_corners = sorted(corners)
    return {c: 1.0 / len(sorted_corners) for c in sorted_corners}


__all__ = ["weight_corners"]
