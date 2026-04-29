"""Atomic in-place parquet rewrite for `data_quality_mask` (slice D-3, unit U8).

Updates a session's parquet so dirty samples (curb strikes, off-track excursions)
flip `data_quality_mask` to False; the prior column snapshots into
`data_quality_mask_v0` for one-cycle rollback. Cold-start sessions are no-ops on
the mask but still establish the `_v0` baseline so the on-disk schema is uniform.

Atomicity: write to `<parquet>.tmp.<pid>`, fsync the file, then `os.replace`.
On any failure (including KeyboardInterrupt / SystemExit) the temp file is
removed before re-raising; the original parquet is left untouched.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.paths import catalog_path, resolve_corpus_root
from racingoptimizer.track.builder import TrackModel, build_track_model

_MASK_COL = "data_quality_mask"
_MASK_V0_COL = "data_quality_mask_v0"


@dataclass(frozen=True)
class ApplyMaskResult:
    session_id: str
    track: str
    parquet_path: Path
    n_samples_total: int
    n_samples_clean_before: int
    n_samples_clean_after: int
    regime: Literal["cold_start", "compounding"]
    noop: bool


def apply_quality_mask(
    session_id: str,
    *,
    track_model: TrackModel | None = None,
    corpus_root: Path | str | None = None,
) -> ApplyMaskResult:
    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    with cat.open_catalog(catalog_path(root)) as conn:
        sess = cat.get_session(conn, session_id)
        if sess is None:
            raise KeyError(f"unknown session_id: {session_id}")
        if sess.parquet_path is None:
            raise FileNotFoundError(f"session {session_id} has no parquet_path in catalog")
        pq_path = root / sess.parquet_path
        if not pq_path.exists():
            raise FileNotFoundError(f"parquet missing for session {session_id}: {pq_path}")

        if track_model is None:
            other = cat.query_sessions(conn, car=sess.car, track=sess.track, valid_only=True)
            sids = sorted({s.session_id for s in other} | {session_id})
            track_model = build_track_model(sess.track, sids, corpus_root=root)

    df = pl.read_parquet(pq_path)
    n_total = df.height

    if _MASK_COL not in df.columns:
        df = df.with_columns(pl.lit(True, dtype=pl.Boolean).alias(_MASK_COL))

    prior_mask = df[_MASK_COL].to_numpy().astype(bool)
    n_clean_before = int(prior_mask.sum())

    if _MASK_V0_COL in df.columns:
        df = df.drop(_MASK_V0_COL)

    if track_model.regime == "cold_start":
        new_mask = prior_mask.copy()
        noop = True
    else:
        new_mask = _compute_clean_mask(df, track_model)
        noop = False

    n_clean_after = int(new_mask.sum())

    rebuilt = df.with_columns(
        pl.Series(_MASK_COL, new_mask, dtype=pl.Boolean),
        pl.Series(_MASK_V0_COL, prior_mask, dtype=pl.Boolean),
    )
    rebuilt = _reorder_columns(rebuilt, df.columns)

    _atomic_write(rebuilt, pq_path)

    return ApplyMaskResult(
        session_id=session_id,
        track=sess.track,
        parquet_path=pq_path,
        n_samples_total=n_total,
        n_samples_clean_before=n_clean_before,
        n_samples_clean_after=n_clean_after,
        regime=track_model.regime,
        noop=noop,
    )


def _compute_clean_mask(df: pl.DataFrame, track_model: TrackModel) -> np.ndarray:
    n = df.height
    if "lap_index" not in df.columns:
        return np.ones(n, dtype=bool)

    dirty = np.zeros(n, dtype=bool)
    lap_index = df["lap_index"].to_numpy()
    for lap in np.unique(lap_index):
        if int(lap) < 0:
            continue
        positions = np.flatnonzero(lap_index == lap)
        start, stop = int(positions[0]), int(positions[-1]) + 1
        lap_df = df.slice(start, stop - start)
        dirty[start:stop] = track_model.curb_mask(lap_df) | track_model.off_track_mask(lap_df)
    return ~dirty


def _reorder_columns(rebuilt: pl.DataFrame, original_columns: list[str]) -> pl.DataFrame:
    ordered: list[str] = []
    for col in original_columns:
        if col in rebuilt.columns:
            ordered.append(col)
    for col in rebuilt.columns:
        if col not in ordered:
            ordered.append(col)
    return rebuilt.select(ordered)


def _atomic_write(df: pl.DataFrame, target: Path) -> None:
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    try:
        df.write_parquet(tmp, compression="zstd")
        fd = os.open(tmp, os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, target)
    except BaseException:
        # Catches KeyboardInterrupt / SystemExit too: tmp must never linger.
        _remove_quietly(tmp)
        raise


def _remove_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
