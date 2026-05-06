"""Per-corner schedule extractor for the per-car (v4) physics model.

A ``CornerSchedule`` is the bridge between a track-agnostic per-car
``PhysicsModel`` and a target track's geometry. It enumerates every
``(corner_id, phase)`` the recommender should score on the target track and
attaches an archetype-feature dict per corner — apex speed, peak lat-G,
corner duration, max/min speed — pulled from observed lap data on the
target track.

The schedule is used both:

* during ``recommend()``: each candidate setup is scored at every
  ``(corner_id, phase)`` in the schedule by passing the corner's
  archetype features into ``PhysicsModel._predict_v4``.
* during the per-corner weighting (``physics.weights.weight_corners``):
  the per-corner time-sensitivity weight is computed from the schedule's
  archetype values.

Cold-start handling: a target track with as few as one valid lap on any
car still produces a usable schedule — the archetype values aggregate
median across whatever (session, lap) rows exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from racingoptimizer.corner import corner_phase_states
from racingoptimizer.ingest import api as ingest_api
from racingoptimizer.ingest.paths import resolve_corpus_root


# Keys carried in the archetype dict — same order/names as
# `racingoptimizer.physics.fitter.CORNER_ARCHETYPE_COLUMNS` so the
# `_assemble_feature_row_v4` lookup matches the trained feature_names.
ARCHETYPE_KEYS: tuple[str, ...] = (
    "corner_apex_speed_ms",
    "corner_peak_lat_g",
    "corner_max_speed_ms",
    "corner_min_speed_ms",
    "corner_duration_s",
)


@dataclass(frozen=True, slots=True)
class CornerScheduleEntry:
    corner_id: int
    phase: str
    archetype: dict[str, float]


def build_corner_schedule(
    session_ids: Iterable[str],
    *,
    corpus_root: Path | str | None = None,
) -> list[CornerScheduleEntry]:
    """Return the per-(corner_id, phase) schedule for a target track.

    Walks every valid lap of every session in ``session_ids``, runs
    ``corner_phase_states`` per lap, then groups by ``(corner_id, phase)``
    and computes per-group archetype values via the same window aggregations
    the per-car fitter uses at training time.

    The returned schedule is sorted by ``(corner_id, phase)`` for stable
    iteration. Empty list if no laps survive the lap-loader (e.g., target
    track has only out-laps with no detected corners).
    """
    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    sids = sorted(session_ids)
    frames: list[pl.DataFrame] = []
    for sid in sids:
        try:
            laps_df = ingest_api.laps(
                session_id=sid, valid_only=True, corpus_root=root,
            )
        except Exception:
            continue
        if laps_df.height == 0:
            continue
        for lap_idx in laps_df["lap_index"].to_list():
            try:
                cps = corner_phase_states(
                    sid, int(lap_idx), corpus_root=root,
                )
            except (KeyError, ValueError, FileNotFoundError):
                continue
            if cps.height == 0:
                continue
            frames.append(cps)

    if not frames:
        return []

    pooled = pl.concat(frames, how="diagonal_relaxed")
    if "corner_id" not in pooled.columns:
        return []
    pooled = pooled.filter(pl.col("corner_id") >= 0)
    if pooled.height == 0:
        return []

    # Per-corner archetype aggregations. Use median across (session, lap)
    # to be robust to one-off slow laps / driver inputs. The per-corner
    # archetype is the same for every phase of the corner — it describes
    # the corner, not the phase.
    duration_per_phase = pl.col("t_end_s") - pl.col("t_start_s")
    per_corner = (
        pooled.group_by("corner_id")
        .agg(
            [
                pl.col("speed_min_ms").min().alias("corner_apex_speed_ms"),
                pl.col("accel_lat_g_max").max().alias("corner_peak_lat_g"),
                pl.col("speed_max_ms").max().alias("corner_max_speed_ms"),
                pl.col("speed_min_ms").min().alias("corner_min_speed_ms"),
                # Sum durations across phases AND laps, then divide by
                # number of distinct (session, lap) pairs to get the
                # per-lap corner duration.
                duration_per_phase.sum().alias("_total_duration_s"),
                pl.struct(["session_id", "lap_index"]).n_unique().alias("_n_laps"),
            ]
        )
        .with_columns(
            (
                pl.col("_total_duration_s") / pl.max_horizontal("_n_laps", pl.lit(1))
            ).alias("corner_duration_s")
        )
    )

    # Distinct phases observed for each corner.
    phases_per_corner = (
        pooled.select(["corner_id", "phase"])
        .unique()
        .sort(["corner_id", "phase"])
    )

    archetype_by_corner: dict[int, dict[str, float]] = {}
    for row in per_corner.iter_rows(named=True):
        cid = int(row["corner_id"])
        archetype_by_corner[cid] = {
            "corner_apex_speed_ms": float(row["corner_apex_speed_ms"] or 0.0),
            "corner_peak_lat_g": float(row["corner_peak_lat_g"] or 0.0),
            "corner_max_speed_ms": float(row["corner_max_speed_ms"] or 0.0),
            "corner_min_speed_ms": float(row["corner_min_speed_ms"] or 0.0),
            "corner_duration_s": float(row["corner_duration_s"] or 0.0),
        }

    schedule: list[CornerScheduleEntry] = []
    for row in phases_per_corner.iter_rows(named=True):
        cid = int(row["corner_id"])
        phase_str = str(row["phase"])
        archetype = archetype_by_corner.get(cid)
        if archetype is None:
            continue
        schedule.append(
            CornerScheduleEntry(
                corner_id=cid,
                phase=phase_str,
                archetype=dict(archetype),
            )
        )
    return schedule


__all__ = [
    "ARCHETYPE_KEYS",
    "CornerScheduleEntry",
    "build_corner_schedule",
]
