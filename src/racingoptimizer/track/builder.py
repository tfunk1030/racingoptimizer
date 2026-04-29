"""Track-model builder + cache (slice D-1, unit U6).

Aggregates per-session per-bin shock-velocity p99 and lateral-G p95/median
across `track_pos_m` bins, then collapses across sessions into a track-wide
summary. Persistent on disk so downstream slices read parquet, not raw
60 Hz data. Curb / bump detection (the actual likelihoods) is U7's job —
this unit ships placeholder zeros and the bin pipeline.

Determinism contract: same `(track, sorted(session_ids))` → byte-identical
persisted parquet. Sort orders are pinned explicitly.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from racingoptimizer.ingest import api as ingest_api
from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.paths import catalog_path, resolve_corpus_root
from racingoptimizer.track.bins import DEFAULT_BIN_SIZE_M, bin_index, track_pos_m_from_pct
from racingoptimizer.track.paths import (
    cache_path,
    latest_pointer_path,
    sessions_hash,
    summary_path,
)

_COLD_START_THRESHOLD = 3
_GRAVITY_M_S2 = 9.80665

_PER_SESSION_SCHEMA: dict[str, type[pl.DataType]] = {
    "session_id": pl.Utf8,
    "track_pos_m": pl.Float64,
    "n_samples": pl.Int64,
    "shock_v_p99_mm_s": pl.Float64,
    "lateral_g_p95": pl.Float64,
    "lateral_g_median": pl.Float64,
}

_BUMP_SCHEMA: dict[str, type[pl.DataType]] = {
    "track_pos_m": pl.Float64,
    "shock_v_p99_mm_s": pl.Float64,
    "n_samples": pl.Int64,
    "n_sessions": pl.Int64,
    "curb_likelihood": pl.Float64,
    "bump_likelihood": pl.Float64,
}

_GRIP_SCHEMA: dict[str, type[pl.DataType]] = {
    "track_pos_m": pl.Float64,
    "lateral_g_p95": pl.Float64,
    "lateral_g_median": pl.Float64,
    "n_samples": pl.Int64,
    "n_sessions": pl.Int64,
}

# Summary parquet merges bump + grip on track_pos_m. Consumers project columns.
_SUMMARY_SCHEMA: dict[str, type[pl.DataType]] = {
    "track_pos_m": pl.Float64,
    "shock_v_p99_mm_s": pl.Float64,
    "lateral_g_p95": pl.Float64,
    "lateral_g_median": pl.Float64,
    "n_samples": pl.Int64,
    "n_sessions": pl.Int64,
    "curb_likelihood": pl.Float64,
    "bump_likelihood": pl.Float64,
}


@dataclass(frozen=True)
class TrackModel:
    track: str
    regime: Literal["cold_start", "compounding"]
    session_ids: tuple[str, ...]
    bin_size_m: float
    bump_map: pl.DataFrame
    grip_map: pl.DataFrame
    cache_path: Path
    summary_path: Path

    def curb_mask(self, lap_df: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError("curb_mask deferred to U7 (Wave 2)")

    def off_track_mask(self, lap_df: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError("off_track_mask deferred to U7 (Wave 2)")


def build_track_model(
    track: str,
    session_ids: list[str],
    *,
    bin_size_m: float = DEFAULT_BIN_SIZE_M,
    corpus_root: Path | str | None = None,
) -> TrackModel:
    sorted_ids = sorted(session_ids)
    digest = sessions_hash(sorted_ids)

    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    cache = cache_path(root, track, digest)
    summary = summary_path(root, track, digest)

    regime: Literal["cold_start", "compounding"] = (
        "compounding" if len(sorted_ids) >= _COLD_START_THRESHOLD else "cold_start"
    )

    if cache.exists() and summary.exists():
        summary_df = pl.read_parquet(summary)
        bump_map, grip_map = _project_maps(summary_df)
        _write_pointer(root, track, digest, regime, len(sorted_ids))
        return TrackModel(
            track=track,
            regime=regime,
            session_ids=tuple(sorted_ids),
            bin_size_m=bin_size_m,
            bump_map=bump_map,
            grip_map=grip_map,
            cache_path=cache,
            summary_path=summary,
        )

    if regime == "cold_start":
        per_session = pl.DataFrame(schema=_PER_SESSION_SCHEMA)
        summary_df = pl.DataFrame(schema=_SUMMARY_SCHEMA)
    else:
        per_session = _aggregate_per_session(
            sorted_ids, bin_size_m=bin_size_m, corpus_root=root
        )
        summary_df = _collapse_across_sessions(per_session)

    per_session.write_parquet(cache, compression="zstd")
    summary_df.write_parquet(summary, compression="zstd")
    _write_pointer(root, track, digest, regime, len(sorted_ids))

    bump_map, grip_map = _project_maps(summary_df)
    return TrackModel(
        track=track,
        regime=regime,
        session_ids=tuple(sorted_ids),
        bin_size_m=bin_size_m,
        bump_map=bump_map,
        grip_map=grip_map,
        cache_path=cache,
        summary_path=summary,
    )


# ---- internals ----

def _project_maps(summary_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    bump_map = summary_df.select(list(_BUMP_SCHEMA.keys()))
    grip_map = summary_df.select(list(_GRIP_SCHEMA.keys()))
    return bump_map, grip_map


def _write_pointer(
    corpus_root: Path | None,
    track: str,
    digest: str,
    regime: str,
    n_sessions: int,
) -> None:
    pointer = latest_pointer_path(corpus_root, track)
    payload = {"sessions_hash": digest, "regime": regime, "n_sessions": n_sessions}
    pointer.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _aggregate_per_session(
    session_ids: list[str],
    *,
    bin_size_m: float,
    corpus_root: Path,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    with cat.open_catalog(catalog_path(corpus_root)) as conn:
        for sid in session_ids:
            sess = cat.get_session(conn, sid)
            if sess is None or sess.parquet_path is None:
                continue
            lap_length_m = _lap_length_for_session(conn, sid, corpus_root=corpus_root)
            if lap_length_m is None or lap_length_m <= 0.0:
                continue
            session_frame = _aggregate_one_session(
                sid, bin_size_m=bin_size_m, lap_length_m=lap_length_m, corpus_root=corpus_root
            )
            if session_frame.height > 0:
                frames.append(session_frame)
    if not frames:
        return pl.DataFrame(schema=_PER_SESSION_SCHEMA)
    out = pl.concat(frames, how="vertical_relaxed")
    return out.sort(["session_id", "track_pos_m"], maintain_order=True)


def _aggregate_one_session(
    session_id: str,
    *,
    bin_size_m: float,
    lap_length_m: float,
    corpus_root: Path,
) -> pl.DataFrame:
    lap_rows = ingest_api.laps(
        session_id=session_id, valid_only=True, corpus_root=corpus_root
    )
    if lap_rows.height == 0:
        return pl.DataFrame(schema=_PER_SESSION_SCHEMA)

    needed = [
        "lap_dist_pct",
        "LFshockVel",
        "RFshockVel",
        "LRshockVel",
        "RRshockVel",
        "AccelLat",
    ]
    sample_frames: list[pl.DataFrame] = []
    for lap_idx in lap_rows["lap_index"].to_list():
        try:
            df = ingest_api.lap_data(
                session_id, int(lap_idx), channels=needed, corpus_root=corpus_root
            )
        except Exception:
            continue
        if df.height == 0:
            continue
        track_pos = track_pos_m_from_pct(df["lap_dist_pct"].to_numpy(), lap_length_m)
        idx = bin_index(track_pos, bin_size_m=bin_size_m)
        shock = np.maximum.reduce(
            [
                np.abs(df["LFshockVel"].to_numpy()),
                np.abs(df["RFshockVel"].to_numpy()),
                np.abs(df["LRshockVel"].to_numpy()),
                np.abs(df["RRshockVel"].to_numpy()),
            ]
        )
        lat_g = np.abs(df["AccelLat"].to_numpy()) / _GRAVITY_M_S2
        sample_frames.append(
            pl.DataFrame(
                {
                    "bin": idx.astype(np.int64),
                    "shock": shock.astype(np.float64),
                    "lat_g": lat_g.astype(np.float64),
                }
            )
        )

    if not sample_frames:
        return pl.DataFrame(schema=_PER_SESSION_SCHEMA)

    samples = pl.concat(sample_frames, how="vertical_relaxed").filter(pl.col("bin") >= 0)
    if samples.height == 0:
        return pl.DataFrame(schema=_PER_SESSION_SCHEMA)

    out = (
        samples.group_by("bin")
        .agg(
            pl.len().cast(pl.Int64).alias("n_samples"),
            pl.col("shock").quantile(0.99).alias("shock_v_p99_mm_s"),
            pl.col("lat_g").quantile(0.95).alias("lateral_g_p95"),
            pl.col("lat_g").median().alias("lateral_g_median"),
        )
        .with_columns(
            pl.lit(session_id).alias("session_id"),
            (pl.col("bin").cast(pl.Float64) * bin_size_m).alias("track_pos_m"),
        )
        .select(list(_PER_SESSION_SCHEMA.keys()))
    )
    return out.sort("track_pos_m", maintain_order=True)


def _collapse_across_sessions(per_session: pl.DataFrame) -> pl.DataFrame:
    if per_session.height == 0:
        return pl.DataFrame(schema=_SUMMARY_SCHEMA)
    grouped = (
        per_session.group_by("track_pos_m")
        .agg(
            pl.col("shock_v_p99_mm_s").quantile(0.99).alias("shock_v_p99_mm_s"),
            pl.col("lateral_g_p95").median().alias("lateral_g_p95"),
            pl.col("lateral_g_median").median().alias("lateral_g_median"),
            pl.col("n_samples").sum().alias("n_samples"),
            pl.col("session_id").n_unique().cast(pl.Int64).alias("n_sessions"),
        )
        .with_columns(
            pl.lit(0.0, dtype=pl.Float64).alias("curb_likelihood"),
            pl.lit(0.0, dtype=pl.Float64).alias("bump_likelihood"),
        )
        .select(list(_SUMMARY_SCHEMA.keys()))
    )
    return grouped.sort("track_pos_m", maintain_order=True)


_TRACK_LENGTH_PATTERN = re.compile(r"([\d.]+)\s*(km|m)\b", re.IGNORECASE)


def _lap_length_for_session(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    corpus_root: Path,
) -> float | None:
    sess = cat.get_session(conn, session_id)
    if sess is None or sess.setup is None:
        return _lap_length_from_speed_fallback(session_id, corpus_root=corpus_root)
    try:
        setup = json.loads(sess.setup)
    except json.JSONDecodeError:
        setup = {}
    weekend = setup.get("WeekendInfo") if isinstance(setup, dict) else None
    raw = weekend.get("TrackLength") if isinstance(weekend, dict) else None
    parsed = _parse_track_length(raw)
    if parsed is not None and parsed > 0.0:
        return parsed
    # Fallback: integrate Speed × dt over the longest valid lap. Used when the
    # IBT YAML header does not expose TrackLength (or it parses to zero).
    return _lap_length_from_speed_fallback(session_id, corpus_root=corpus_root)


def _parse_track_length(raw: object) -> float | None:
    if not isinstance(raw, str):
        return None
    m = _TRACK_LENGTH_PATTERN.search(raw)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    return value * 1000.0 if unit == "km" else value


def _lap_length_from_speed_fallback(
    session_id: str, *, corpus_root: Path
) -> float | None:
    lap_rows = ingest_api.laps(
        session_id=session_id, valid_only=True, corpus_root=corpus_root
    )
    if lap_rows.height == 0:
        return None
    durations = (
        lap_rows["end_sample"].to_numpy() - lap_rows["start_sample"].to_numpy()
    )
    longest = int(lap_rows["lap_index"].to_numpy()[int(np.argmax(durations))])
    try:
        df = ingest_api.lap_data(
            session_id, longest, channels=["Speed"], corpus_root=corpus_root
        )
    except Exception:
        return None
    if df.height == 0:
        return None
    return float(np.sum(df["Speed"].to_numpy()) / 60.0)
