"""Track-model builder + cache (slice D-1 / D-2, units U6 + U7).

Aggregates per-session per-bin shock-velocity p99 and lateral-G p95/median
across `track_pos_m` bins, then collapses across sessions into a track-wide
summary. Persistent on disk so downstream slices read parquet, not raw
60 Hz data. Curb / bump / off-track classification (U7) plugs into the
compounding regime via `racingoptimizer.track.masks` aggregators; cold-start
keeps the U6 placeholder zeros.

Determinism contract: same `(track, sorted(session_ids))` → byte-identical
persisted parquet. Sort orders are pinned explicitly. The build-time
`lap_length_m` is stored as a constant column on the summary parquet so
`TrackModel.lap_length_m` and `curb_mask` / `off_track_mask` can recover it
without re-reading the IBT YAML.
"""
from __future__ import annotations

import json
import logging
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
from racingoptimizer.track.masks import (
    _max_abs_shock_vel,
    aggregate_curb_likelihood,
    aggregate_grip_map,
    compute_curb_mask,
    compute_off_track_mask,
    shock_vel_channels,
)
from racingoptimizer.track.paths import (
    cache_path,
    latest_pointer_path,
    sessions_hash,
    summary_path,
)

_LOG = logging.getLogger(__name__)

_COLD_START_THRESHOLD = 3
_GRAVITY_M_S2 = 9.80665

_PER_SESSION_SCHEMA: dict[str, type[pl.DataType]] = {
    "session_id": pl.Utf8,
    "track_pos_m": pl.Float64,
    "n_samples": pl.Int64,
    "shock_v_p99_mm_s": pl.Float64,
    "lateral_g_p95": pl.Float64,
    "lateral_g_median": pl.Float64,
    "speed_min_ms": pl.Float64,
    "speed_median_ms": pl.Float64,
    "speed_max_ms": pl.Float64,
}

_BUMP_SCHEMA: dict[str, type[pl.DataType]] = {
    "bin_index": pl.Int32,
    "track_pos_m": pl.Float64,
    "shock_v_p99_mm_s": pl.Float64,
    "n_samples": pl.Int64,
    "n_sessions": pl.Int64,
    "curb_likelihood": pl.Float64,
    "bump_likelihood": pl.Float64,
}

_GRIP_SCHEMA: dict[str, type[pl.DataType]] = {
    "bin_index": pl.Int32,
    "track_pos_m": pl.Float64,
    "lateral_g_p95": pl.Float64,
    "lateral_g_median": pl.Float64,
    "n_samples": pl.Int64,
    "n_sessions": pl.Int64,
}

# Speed envelope (VISION §9: "Typical speed envelope at every point — min/median/max
# from all laps"). Speed is m/s as iRacing emits it; downstream consumers convert
# to km/h or mph at render time.
_SPEED_ENVELOPE_SCHEMA: dict[str, type[pl.DataType]] = {
    "bin_index": pl.Int32,
    "track_pos_m": pl.Float64,
    "speed_min_ms": pl.Float64,
    "speed_median_ms": pl.Float64,
    "speed_max_ms": pl.Float64,
    "n_samples": pl.Int64,
    "n_sessions": pl.Int64,
}

# Summary parquet merges bump + grip + speed envelope on bin_index. lap_length_m
# repeats per row (constant per build) so consumers reconstruct track_pos_m without
# sidecars.
_SUMMARY_SCHEMA: dict[str, type[pl.DataType]] = {
    "bin_index": pl.Int32,
    "track_pos_m": pl.Float64,
    "shock_v_p99_mm_s": pl.Float64,
    "lateral_g_p95": pl.Float64,
    "lateral_g_median": pl.Float64,
    "speed_min_ms": pl.Float64,
    "speed_median_ms": pl.Float64,
    "speed_max_ms": pl.Float64,
    "n_samples": pl.Int64,
    "n_sessions": pl.Int64,
    "curb_likelihood": pl.Float64,
    "bump_likelihood": pl.Float64,
    "lap_length_m": pl.Float64,
}


@dataclass(frozen=True)
class TrackModel:
    track: str
    regime: Literal["cold_start", "compounding"]
    session_ids: tuple[str, ...]
    bin_size_m: float
    bump_map: pl.DataFrame
    grip_map: pl.DataFrame
    speed_envelope: pl.DataFrame
    cache_path: Path
    summary_path: Path
    corpus_root: Path | None = None
    # Per-IBT recording rate detected at ingest (S1.4). Defaults to 60 Hz
    # for backward-compat with TrackModels built before this field existed
    # (the catalog stores `sample_rate_hz` as nullable so old rows return
    # None; we coalesce to the iRacing default at lookup time).
    sample_rate_hz: float = 60.0
    # Canonical car key (e.g. "acura", "bmw"). Drives per-car threshold lookups
    # in `compute_curb_mask` (Acura uses a lower curb-agreement fraction —
    # see `racingoptimizer.track.masks._PER_CAR_CURB_AGREEMENT_FRACTION`).
    # `None` keeps the legacy four-corner defaults.
    car: str | None = None

    @property
    def geometry(self) -> pl.DataFrame:
        """Per-bin elevation gradient + camber ratio proxies (S4.4, VISION §9).

        Lazily computed on first access — walks every session's clean
        laps, fits the corpus-wide ``LongAccel ~ Speed`` and
        ``|LatAccel|/g ~ |SteeringWheelAngle|`` baselines, and returns a
        per-bin median frame. Cached on the instance for subsequent
        accesses. Cold-start returns an empty frame.

        Schema: ``(bin_index, track_pos_m, elevation_gradient_proxy,
        camber_ratio_proxy, n_samples, n_sessions)``.
        """
        from racingoptimizer.track.geometry import (
            compute_track_geometry,
            empty_geometry_frame,
        )

        cached = self.__dict__.get("_geometry_cache")
        if cached is not None:
            return cached
        if self.regime == "cold_start":
            value = empty_geometry_frame()
        else:
            value = compute_track_geometry(
                self.track,
                list(self.session_ids),
                bin_size_m=self.bin_size_m,
                corpus_root=self.corpus_root,
            )
        # Frozen dataclass blocks setattr; reach into __dict__ directly.
        object.__setattr__(self, "_geometry_cache", value)
        return value

    @property
    def lap_length_m(self) -> float | None:
        if not self.summary_path.exists():
            return None
        df = pl.read_parquet(self.summary_path, columns=["lap_length_m"])
        if df.height == 0:
            return None
        value = df["lap_length_m"][0]
        return float(value) if value is not None else None

    def curb_mask(self, lap_df: pl.DataFrame) -> np.ndarray:
        if self.regime == "cold_start":
            return np.zeros(lap_df.height, dtype=bool)
        lap_length = self._resolve_lap_length(lap_df)
        return compute_curb_mask(
            lap_df,
            self.bump_map,
            lap_length_m=lap_length,
            bin_size_m=self.bin_size_m,
            car=self.car,
        )

    def off_track_mask(self, lap_df: pl.DataFrame) -> np.ndarray:
        if self.regime == "cold_start":
            return np.zeros(lap_df.height, dtype=bool)
        lap_length = self._resolve_lap_length(lap_df)
        return compute_off_track_mask(
            lap_df,
            self.grip_map,
            lap_length_m=lap_length,
            bin_size_m=self.bin_size_m,
            sample_rate_hz=int(round(self.sample_rate_hz)),
        )

    def _resolve_lap_length(self, lap_df: pl.DataFrame) -> float:
        stored = self.lap_length_m
        if stored is not None and stored > 0.0:
            return stored
        if "Speed" in lap_df.columns and lap_df.height > 0:
            # VISION §1: per-IBT sample rate, never the hardcoded 60 Hz default.
            return float(np.sum(lap_df["Speed"].to_numpy()) / self.sample_rate_hz)
        return 1.0

    # ---- S4.1: predict expected & flag anomalies (VISION §9) ----

    def expected(self, track_pos_m: float, channel: str):
        """Return `(mean, p99)` of `channel` across sessions for the bin at `track_pos_m`.

        Channels: ``shock_v_p99_mm_s``, ``lateral_g_p95``, ``lateral_g_median``.
        Returns ``None`` on cold-start (< 3 sessions for the bin) — the model
        does not yet know what "expected" looks like there.

        See `racingoptimizer.track.predict` for the data source (per-session
        cache parquet) and threshold rationale.
        """
        from racingoptimizer.track.predict import expected_from_cache

        if self.regime == "cold_start" or not self.cache_path.exists():
            return None
        cache_df = pl.read_parquet(self.cache_path)
        return expected_from_cache(
            cache_df,
            track_pos_m=track_pos_m,
            channel=channel,
            bin_size_m=self.bin_size_m,
        )

    @property
    def corner_landmarks(self) -> pl.DataFrame:
        """Cross-lap braking / apex / exit positions per corner (S4.2, VISION §9).

        Lazily delegates to `racingoptimizer.track.corners.compute_corner_landmarks`
        for the model's `(track, session_ids)`. Returns an empty frame on
        cold-start or when no session yields a usable lap. Computed on each
        access — no caching at the model layer because the underlying source
        parquets are already on disk and the per-lap detector is fast.
        """
        from racingoptimizer.track.corners import (
            _empty_landmarks_frame,
            compute_corner_landmarks,
        )

        if self.regime == "cold_start":
            return _empty_landmarks_frame()
        # cache_path = <corpus_root>/track_models/<track>.<hash>.parquet, so
        # the corpus_root is the file's grandparent.
        corpus_root = self.cache_path.parent.parent
        return compute_corner_landmarks(
            self.track, list(self.session_ids), corpus_root=corpus_root
        )

    def flag_anomalies(
        self, lap_df: pl.DataFrame, *, car: str | None = None, z_threshold: float = 3.0
    ) -> pl.DataFrame:
        """Flag samples in `lap_df` whose channels deviate far from this model's expectation.

        Returns a polars DataFrame with columns
        ``(sample_idx, channel, observed, expected, z_score, label)``.
        Empty frame on cold-start, on missing source channels, or when no
        sample crosses ``|z_score| > z_threshold``. See
        `racingoptimizer.track.anomaly` for the heuristic label rules.
        """
        from racingoptimizer.track.anomaly import _empty_anomaly_frame, flag_anomalies_from_cache

        if self.regime == "cold_start" or not self.cache_path.exists():
            return _empty_anomaly_frame()
        cache_df = pl.read_parquet(self.cache_path)
        lap_length = self._resolve_lap_length(lap_df)
        return flag_anomalies_from_cache(
            lap_df,
            cache_df,
            lap_length_m=lap_length,
            bin_size_m=self.bin_size_m,
            car=car,
            z_threshold=z_threshold,
        )

    # ---- S4.5: per-corner loading classification (VISION §9) ----

    @property
    def corner_loading(self) -> pl.DataFrame:
        """Per-corner classification (front/rear/traction/aero/mixed).

        Lazy: walks every session in this model and stacks each session's
        valid laps' `corner_phase_states`, then runs the heuristic
        classifier in `racingoptimizer.track.corner_loading`. Returns an
        empty (schema-only) frame when no observations are available
        (cold-start regime, or no laps survived the channel-presence
        filters).

        See :func:`classify_corner_loading` for the full heuristic.
        """
        from racingoptimizer.track.corner_loading import classify_corner_loading

        return classify_corner_loading(self.track, list(self.session_ids))


def build_track_model(
    track: str,
    session_ids: list[str],
    *,
    bin_size_m: float = DEFAULT_BIN_SIZE_M,
    corpus_root: Path | str | None = None,
    car: str | None = None,
) -> TrackModel:
    sorted_ids = sorted(session_ids)
    digest = sessions_hash(sorted_ids)

    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    cache = cache_path(root, track, digest)
    summary = summary_path(root, track, digest)

    regime: Literal["cold_start", "compounding"] = (
        "compounding" if len(sorted_ids) >= _COLD_START_THRESHOLD else "cold_start"
    )

    sample_rate_hz = _resolve_session_sample_rate(sorted_ids, root)
    # When the caller doesn't pin a car, infer one from the catalog so per-car
    # threshold lookups (curb-agreement, shock channels) still apply. Falls
    # back to the dominant car across the requested session ids.
    resolved_car = car if car else _resolve_session_car(sorted_ids, root)

    if cache.exists() and summary.exists():
        summary_df = pl.read_parquet(summary)
        bump_map, grip_map, speed_envelope = _project_maps(summary_df)
        _write_pointer(root, track, digest, regime, len(sorted_ids))
        return TrackModel(
            track=track,
            regime=regime,
            session_ids=tuple(sorted_ids),
            bin_size_m=bin_size_m,
            bump_map=bump_map,
            grip_map=grip_map,
            speed_envelope=speed_envelope,
            cache_path=cache,
            summary_path=summary,
            corpus_root=root,
            sample_rate_hz=sample_rate_hz,
            car=resolved_car,
        )

    if regime == "cold_start":
        per_session = pl.DataFrame(schema=_PER_SESSION_SCHEMA)
        summary_df = pl.DataFrame(schema=_SUMMARY_SCHEMA)
    else:
        per_session, lap_length_m = _aggregate_per_session(
            sorted_ids, bin_size_m=bin_size_m, corpus_root=root, track=track
        )
        summary_df = _collapse_across_sessions(
            per_session,
            bin_size_m=bin_size_m,
            lap_length_m=lap_length_m,
            car=resolved_car,
        )

    per_session.write_parquet(cache, compression="zstd")
    summary_df.write_parquet(summary, compression="zstd")
    _write_pointer(root, track, digest, regime, len(sorted_ids))

    bump_map, grip_map, speed_envelope = _project_maps(summary_df)
    return TrackModel(
        track=track,
        regime=regime,
        session_ids=tuple(sorted_ids),
        bin_size_m=bin_size_m,
        bump_map=bump_map,
        grip_map=grip_map,
        speed_envelope=speed_envelope,
        cache_path=cache,
        summary_path=summary,
        corpus_root=root,
        sample_rate_hz=sample_rate_hz,
        car=resolved_car,
    )


def _resolve_session_car(session_ids: list[str], corpus_root: Path) -> str | None:
    """Look up the dominant canonical car key across the requested sessions.

    Returns ``None`` when no rows exist or the catalog is empty — callers
    treat ``None`` as "use the four-corner default thresholds". When sessions
    span multiple cars (which is unusual but legal — one track may host
    different cars on different days), the most-frequent car wins.
    """
    if not session_ids:
        return None
    try:
        df = ingest_api.sessions(corpus_root=corpus_root)
    except Exception:
        return None
    if df.height == 0 or "car" not in df.columns:
        return None
    matching = df.filter(pl.col("session_id").is_in(session_ids))
    cars = [c for c in matching["car"].to_list() if c]
    if not cars:
        return None
    counts: dict[str, int] = {}
    for c in cars:
        counts[c] = counts.get(c, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _resolve_session_sample_rate(session_ids: list[str], corpus_root: Path) -> float:
    """Look up the per-IBT sample rate for the dominant session.

    Reads ``sessions(...).sample_rate_hz`` for the requested ids and returns
    the median (most non-trivial corpora are recorded at a single rate).
    Falls back to 60.0 Hz when the catalog is empty or every row is null —
    matching iRacing's default IBT recording rate.
    """
    if not session_ids:
        return 60.0
    try:
        df = ingest_api.sessions(corpus_root=corpus_root)
    except Exception:
        return 60.0
    if df.height == 0 or "sample_rate_hz" not in df.columns:
        return 60.0
    matching = df.filter(pl.col("session_id").is_in(session_ids))
    rates = [r for r in matching["sample_rate_hz"].to_list() if r is not None and r > 0]
    if not rates:
        return 60.0
    return float(sorted(rates)[len(rates) // 2])


# ---- internals ----

def _project_maps(
    summary_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return (
        summary_df.select(list(_BUMP_SCHEMA.keys())),
        summary_df.select(list(_GRIP_SCHEMA.keys())),
        summary_df.select(list(_SPEED_ENVELOPE_SCHEMA.keys())),
    )


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
    track: str,
) -> tuple[pl.DataFrame, float]:
    frames: list[pl.DataFrame] = []
    chosen_lap_length: float | None = None
    with cat.open_catalog(catalog_path(corpus_root)) as conn:
        for sid in session_ids:
            sess = cat.get_session(conn, sid)
            if sess is None or sess.parquet_path is None:
                continue
            lap_length_m = _lap_length_for_session(conn, sid, corpus_root=corpus_root)
            if lap_length_m is None or lap_length_m <= 0.0:
                continue
            if chosen_lap_length is None:
                chosen_lap_length = float(lap_length_m)
            session_frame = _aggregate_one_session(
                sid,
                bin_size_m=bin_size_m,
                lap_length_m=lap_length_m,
                corpus_root=corpus_root,
                car=sess.car,
                track=track,
            )
            if session_frame.height > 0:
                frames.append(session_frame)
    if not frames:
        return pl.DataFrame(schema=_PER_SESSION_SCHEMA), float(chosen_lap_length or 0.0)
    out = pl.concat(frames, how="vertical_relaxed")
    return (
        out.sort(["session_id", "track_pos_m"], maintain_order=True),
        float(chosen_lap_length or 0.0),
    )


def _aggregate_one_session(
    session_id: str,
    *,
    bin_size_m: float,
    lap_length_m: float,
    corpus_root: Path,
    car: str | None,
    track: str,
) -> pl.DataFrame:
    lap_rows = ingest_api.laps(
        session_id=session_id, valid_only=True, corpus_root=corpus_root
    )
    if lap_rows.height == 0:
        return pl.DataFrame(schema=_PER_SESSION_SCHEMA)

    shock_channels = shock_vel_channels(car)
    needed = ["lap_dist_pct", *shock_channels, "LatAccel", "Speed"]
    sample_frames: list[pl.DataFrame] = []
    for lap_idx in lap_rows["lap_index"].to_list():
        try:
            df = ingest_api.lap_data(
                session_id, int(lap_idx), channels=needed, corpus_root=corpus_root
            )
        except pl.exceptions.ColumnNotFoundError as exc:
            # Spec §9: skip curb/bump for the session and log; do not silently
            # drop the lap. Log once per session — the mismatch is structural.
            _LOG.warning(
                "track=%s session=%s: skipped curb/bump (channel absent: %s)",
                track,
                session_id,
                exc,
            )
            return pl.DataFrame(schema=_PER_SESSION_SCHEMA)
        if df.height == 0:
            continue
        track_pos = track_pos_m_from_pct(df["lap_dist_pct"].to_numpy(), lap_length_m)
        idx = bin_index(track_pos, bin_size_m=bin_size_m)
        shock = _max_abs_shock_vel(df, shock_channels)
        lat_g = np.abs(df["LatAccel"].to_numpy()) / _GRAVITY_M_S2
        speed = df["Speed"].to_numpy().astype(np.float64)
        sample_frames.append(
            pl.DataFrame(
                {
                    "bin": idx.astype(np.int64),
                    "shock": shock.astype(np.float64),
                    "lat_g": lat_g.astype(np.float64),
                    "speed": speed,
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
            pl.col("shock").quantile(0.99, "linear").alias("shock_v_p99_mm_s"),
            pl.col("lat_g").quantile(0.95, "linear").alias("lateral_g_p95"),
            pl.col("lat_g").median().alias("lateral_g_median"),
            pl.col("speed").min().alias("speed_min_ms"),
            pl.col("speed").median().alias("speed_median_ms"),
            pl.col("speed").max().alias("speed_max_ms"),
        )
        .with_columns(
            pl.lit(session_id).alias("session_id"),
            (pl.col("bin").cast(pl.Float64) * bin_size_m).alias("track_pos_m"),
        )
        .select(list(_PER_SESSION_SCHEMA.keys()))
    )
    return out.sort("track_pos_m", maintain_order=True)


def _collapse_across_sessions(
    per_session: pl.DataFrame,
    *,
    bin_size_m: float,
    lap_length_m: float,
    car: str | None = None,
) -> pl.DataFrame:
    if per_session.height == 0:
        return pl.DataFrame(schema=_SUMMARY_SCHEMA)

    indexed = per_session.with_columns(
        (pl.col("track_pos_m") / bin_size_m).round().cast(pl.Int32).alias("bin_index")
    )

    per_session_p99 = [
        sub.select(["bin_index", "shock_v_p99_mm_s"])
        for _, sub in indexed.group_by("session_id", maintain_order=True)
    ]
    per_session_grip = [
        sub.select(["bin_index", "lateral_g_p95", "lateral_g_median", "n_samples"])
        .with_columns(
            pl.col("n_samples").cast(pl.UInt32),
            pl.lit(1, dtype=pl.Int64).alias("n_sessions"),
        )
        for _, sub in indexed.group_by("session_id", maintain_order=True)
    ]

    bump = aggregate_curb_likelihood(per_session_p99, car=car)
    grip = aggregate_grip_map(per_session_grip)
    speed_env = _aggregate_speed_envelope(indexed)

    n_samples_per_bin = (
        indexed.group_by("bin_index")
        .agg(pl.col("n_samples").sum().cast(pl.Int64).alias("n_samples_total"))
    )

    grip_no_n = grip.drop("n_sessions")
    merged = (
        bump.join(grip_no_n, on="bin_index", how="left")
        .join(speed_env, on="bin_index", how="left")
        .join(n_samples_per_bin, on="bin_index", how="left")
        .with_columns(
            (pl.col("bin_index").cast(pl.Float64) * bin_size_m).alias("track_pos_m"),
            pl.lit(lap_length_m, dtype=pl.Float64).alias("lap_length_m"),
            pl.col("n_samples_total").fill_null(0).cast(pl.Int64).alias("n_samples"),
            pl.col("lateral_g_p95").fill_null(0.0),
            pl.col("lateral_g_median").fill_null(0.0),
            pl.col("speed_min_ms").fill_null(0.0),
            pl.col("speed_median_ms").fill_null(0.0),
            pl.col("speed_max_ms").fill_null(0.0),
        )
        .select(list(_SUMMARY_SCHEMA.keys()))
        .sort("bin_index", maintain_order=True)
    )
    return merged


def _aggregate_speed_envelope(indexed: pl.DataFrame) -> pl.DataFrame:
    """Cross-session speed envelope per bin.

    `min` collapses to the smallest seen across all sessions; `max` to the
    largest. `median_ms` is the median of the per-session medians — using the
    median (not the mean) so a single anomalous session does not dominate, the
    same convention as `aggregate_grip_map` (spec §4.3).
    """
    return (
        indexed.group_by("bin_index")
        .agg(
            pl.col("speed_min_ms").min().alias("speed_min_ms"),
            pl.col("speed_median_ms").median().alias("speed_median_ms"),
            pl.col("speed_max_ms").max().alias("speed_max_ms"),
        )
    )


_TRACK_LENGTH_PATTERN = re.compile(r"([\d.]+)\s*(km|m)\b", re.IGNORECASE)


def _lap_length_for_session(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    corpus_root: Path,
) -> float | None:
    sess = cat.get_session(conn, session_id)
    # Per-IBT recording rate (S1.4) — None when the catalog row predates the
    # column or the session failed to ingest a header rate. Coalesce to the
    # iRacing default so the speed-integration fallback stays usable.
    rate_hz = float(sess.sample_rate_hz) if (
        sess is not None and sess.sample_rate_hz is not None and sess.sample_rate_hz > 0
    ) else 60.0
    if sess is None or sess.setup is None:
        return _lap_length_from_speed_fallback(
            session_id, corpus_root=corpus_root, sample_rate_hz=rate_hz
        )
    try:
        setup = json.loads(sess.setup)
    except json.JSONDecodeError:
        setup = {}
    weekend = setup.get("WeekendInfo") if isinstance(setup, dict) else None
    raw = weekend.get("TrackLength") if isinstance(weekend, dict) else None
    parsed = _parse_track_length(raw)
    if parsed is not None and parsed > 0.0:
        return parsed
    return _lap_length_from_speed_fallback(
        session_id, corpus_root=corpus_root, sample_rate_hz=rate_hz
    )


def _parse_track_length(raw: object) -> float | None:
    if not isinstance(raw, str):
        return None
    m = _TRACK_LENGTH_PATTERN.search(raw)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    return value * 1000.0 if unit == "km" else value


_RACING_LAP_MIN_MEAN_SPEED_M_S = 30.0


def _lap_length_from_speed_fallback(
    session_id: str, *, corpus_root: Path, sample_rate_hz: float = 60.0
) -> float | None:
    """Estimate lap length by integrating Speed × dt over a clean racing lap.

    Earlier versions picked the wallclock-longest lap by `argmax(end_sample -
    start_sample)`, which on Porsche/Algarve corpora selected a 350 s pit-out
    lap whose mean Speed was ~1 m/s and integrated to ~412 m instead of the
    real ~4600 m. The wrong `lap_length_m` then poisoned every downstream
    `track_pos_m_from_pct` call.

    Fix: filter out non-racing laps by minimum mean Speed (GTP cars average
    well above 30 m/s on a real lap), then pick the candidate with the
    highest mean Speed — most likely a clean timed lap. Returns None when
    no lap clears the threshold so the caller can skip the session.

    `sample_rate_hz` defaults to 60.0 (iRacing's standard IBT rate) but the
    caller MUST pass the per-IBT detected rate when known — high-frequency
    recordings (e.g. 360 Hz) would otherwise overestimate lap length 6×.
    """
    rate = float(sample_rate_hz) if sample_rate_hz and sample_rate_hz > 0 else 60.0
    lap_rows = ingest_api.laps(
        session_id=session_id, valid_only=True, corpus_root=corpus_root
    )
    if lap_rows.height == 0:
        return None

    best_mean_speed = -1.0
    best_lap_length: float | None = None
    for lap_idx in lap_rows["lap_index"].to_list():
        try:
            df = ingest_api.lap_data(
                session_id, int(lap_idx), channels=["Speed"], corpus_root=corpus_root
            )
        except Exception:
            continue
        if df.height == 0:
            continue
        speed = df["Speed"].to_numpy()
        mean_speed = float(np.mean(speed))
        if mean_speed < _RACING_LAP_MIN_MEAN_SPEED_M_S:
            continue
        if mean_speed > best_mean_speed:
            best_mean_speed = mean_speed
            best_lap_length = float(np.sum(speed) / rate)

    return best_lap_length


