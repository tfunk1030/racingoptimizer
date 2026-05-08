"""Public API for the ingest module."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.detect import (
    UnknownCarError,
    detect_car,
    detect_car_from_filename,
    detect_track_from_filename,
    slugify_track,
)
from racingoptimizer.ingest.parser import parse_ibt
from racingoptimizer.ingest.paths import (
    catalog_path,
    parquet_path,
    resolve_corpus_root,
)
from racingoptimizer.ingest.writer import _now_iso, session_id_from_bytes, write_session


def learn(
    path: Path | str,
    corpus_root: Path | str | None = None,
    *,
    reparse: bool = False,
    apply_quality_masks: bool = True,
) -> list[str]:
    """Ingest a .ibt file or every .ibt under a directory.

    Returns the list of session_ids for every file processed (existing or new),
    regardless of status. Caller can join against `sessions(...)` to inspect
    per-session outcomes.

    ``reparse=True`` forces re-processing of sessions whose catalog row
    is already ``status=ok``. Use this after a parser change to refresh
    stale fields (e.g. ``recorded_at`` after the filename-derivation
    fix); normal incremental ingest skips already-ok sessions.

    ``apply_quality_masks=True`` (default) applies the slice-D track-quality
    mask to every newly-ingested session as a final step. Without this the
    parquet's `data_quality_mask` column stays at the all-True default
    written by the parser, and the curb / off-track filtering the track
    model produces never reaches the fitter. The mask uses every session
    on the same (car, track) for cross-session curb agreement, so the
    deferred pass runs once per (car, track) combination touched. Pass
    `False` to skip (e.g. if the track model is unavailable or you're
    re-running ingest in a tight loop).
    """
    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    db = catalog_path(root)

    targets = list(_iter_ibt_paths(Path(path)))
    out: list[str] = []
    written_this_run: list[str] = []
    with cat.open_catalog(db) as conn:
        for ibt in targets:
            sid, was_written = _process_one(conn, root, ibt, reparse=reparse)
            out.append(sid)
            if was_written:
                written_this_run.append(sid)
    if apply_quality_masks and written_this_run:
        _apply_masks_for_session_ids(written_this_run, root=root)
    return out


def _apply_masks_for_session_ids(
    session_ids: list[str], *, root: Path,
) -> None:
    """Apply the slice-D quality mask to each session_id, batched per
    (car, track) so the TrackModel only builds once per group."""
    # Lazy import: track is a heavy module that pulls in numpy/polars
    # for every callsite, and `learn` is called by tests that don't need
    # the track machinery (e.g. ingest unit tests).
    from racingoptimizer.track.builder import build_track_model
    from racingoptimizer.track.rewrite import apply_quality_mask

    with cat.open_catalog(catalog_path(root)) as conn:
        rows = [cat.get_session(conn, sid) for sid in session_ids]
    by_track: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        if row is None or row.status != "ok":
            continue
        by_track.setdefault((row.car, row.track), []).append(row.session_id)
    if not by_track:
        return
    for (car, track), batch_ids in by_track.items():
        with cat.open_catalog(catalog_path(root)) as conn:
            all_for_track = cat.query_sessions(
                conn, car=car, track=track, valid_only=True,
            )
        all_sids = sorted({s.session_id for s in all_for_track})
        if not all_sids:
            continue
        tm = build_track_model(track, all_sids, corpus_root=root, car=car)
        for sid in batch_ids:
            try:
                apply_quality_mask(sid, track_model=tm, corpus_root=root)
            except Exception as exc:  # noqa: BLE001 -- best-effort: never block ingest
                # The mask pass is informational; ingest already wrote
                # the parquet successfully. Surface failures via
                # warnings so the user can see them, but do NOT abort
                # the whole `learn` run for one bad session.
                import warnings
                warnings.warn(
                    f"apply_quality_mask({sid!r}) failed: "
                    f"{type(exc).__name__}: {exc}; ingest continues",
                    stacklevel=2,
                )


def sessions(
    car: str | None = None,
    track: str | None = None,
    valid_only: bool = True,
    corpus_root: Path | str | None = None,
    *,
    include_held_out: bool = False,
) -> pl.DataFrame:
    """Return one row per session, ordered by recorded_at.

    `include_held_out` defaults to False so production paths never see
    gate-only IBTs marked via `set_held_out_sessions` (physics-rebuild
    PLAN.md Section 7). Gate validation scripts opt in explicitly.
    """
    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    with cat.open_catalog(catalog_path(root)) as conn:
        rows = cat.query_sessions(
            conn, car=car, track=track, valid_only=valid_only,
            include_held_out=include_held_out,
        )
    return pl.DataFrame(
        {
            "session_id": [r.session_id for r in rows],
            "car": [r.car for r in rows],
            "track": [r.track for r in rows],
            "recorded_at": [r.recorded_at for r in rows],
            "ingested_at": [r.ingested_at for r in rows],
            "duration_s": [r.duration_s for r in rows],
            "lap_count": [r.lap_count for r in rows],
            "weather_summary": [r.weather_summary for r in rows],
            "setup": [r.setup for r in rows],
            "status": [r.status for r in rows],
            "error": [r.error for r in rows],
            "parquet_path": [r.parquet_path for r in rows],
            "dropped_channels": [r.dropped_channels for r in rows],
            "sample_rate_hz": [r.sample_rate_hz for r in rows],
            "held_out": [r.held_out for r in rows],
        }
    )


def laps(
    session_id: str | None = None,
    car: str | None = None,
    track: str | None = None,
    valid_only: bool = True,
    corpus_root: Path | str | None = None,
) -> pl.DataFrame:
    """Return one row per lap matching the filters."""
    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    with cat.open_catalog(catalog_path(root)) as conn:
        sid_list: list[str]
        if session_id is not None:
            sid_list = [session_id]
        else:
            sid_list = [
                s.session_id
                for s in cat.query_sessions(conn, car=car, track=track, valid_only=valid_only)
            ]

        all_rows: list[cat.LapRow] = []
        for sid in sid_list:
            rows = cat.get_laps(conn, sid)
            if valid_only:
                rows = [r for r in rows if r.valid == 1]
            all_rows.extend(rows)

    return pl.DataFrame(
        {
            "session_id": [r.session_id for r in all_rows],
            "lap_index": [r.lap_index for r in all_rows],
            "lap_time_s": [r.lap_time_s for r in all_rows],
            "start_sample": [r.start_sample for r in all_rows],
            "end_sample": [r.end_sample for r in all_rows],
            "valid": [r.valid for r in all_rows],
            "best": [r.best for r in all_rows],
        }
    )


def lap_data(
    session_id: str,
    lap_index: int,
    channels: list[str] | None = None,
    corpus_root: Path | str | None = None,
) -> pl.DataFrame:
    """Read one lap's bulk 60 Hz channels from the session's parquet."""
    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    with cat.open_catalog(catalog_path(root)) as conn:
        sess = cat.get_session(conn, session_id)
        if sess is None:
            raise KeyError(f"unknown session_id: {session_id}")
        rows = [r for r in cat.get_laps(conn, session_id) if r.lap_index == lap_index]
        if not rows:
            raise KeyError(f"no lap_index={lap_index} in session {session_id}")
        lap = rows[0]
        pq = parquet_path(root, car=sess.car, track=sess.track, session_id=session_id)

    lf = pl.scan_parquet(pq)
    if channels is not None:
        lf = lf.select(channels)
    df = lf.slice(lap.start_sample, lap.end_sample - lap.start_sample).collect()
    return df


# ---- internal helpers ----

def _iter_ibt_paths(p: Path) -> Iterable[Path]:
    if p.is_file() and p.suffix.lower() == ".ibt":
        yield p
    elif p.is_dir():
        yield from sorted(p.rglob("*.ibt"))


def _record_failure(
    conn: sqlite3.Connection, *, sid: str, source_path: str, exc: BaseException
) -> None:
    """Stamp a `status='failed'` row when no salvage is possible."""
    cat.upsert_session(
        conn,
        cat.SessionRow(
            session_id=sid,
            car="unknown",
            track="unknown",
            recorded_at=None,
            duration_s=None,
            lap_count=None,
            weather_summary=None,
            setup=None,
            source_path=source_path,
            ingested_at=_now_iso(),
            parquet_path=None,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            dropped_channels=None,
            sample_rate_hz=None,
        ),
    )


def _process_one(
    conn: sqlite3.Connection,
    root: Path,
    ibt_path: Path,
    *,
    reparse: bool = False,
) -> tuple[str, bool]:
    """Ingest one IBT, recording an outcome row no matter what.

    Returns ``(session_id, was_written)``. ``was_written`` is ``True``
    when the catalog row / parquet were (re-)written this call --
    either because the session was new, or it had `status != "ok"`,
    or `reparse=True` forced re-processing. ``False`` means the call
    short-circuited on a pre-existing `status="ok"` row. Used by
    `learn` to skip the slice-D quality-mask pass on duplicates so
    re-ingest stays idempotent.

    Status semantics (VISION §1 "use everything, lose nothing"):

    * ``"ok"`` — YAML, channels, lap segmentation, and car/track detection all
      succeeded. Parquet written, catalog row complete.
    * ``"partial"`` — channels parsed and were persisted, but at least one of
      the following is true: lap segmentation produced no spans (the corpus
      keeps the bulk channels but with ``lap_count=0`` and no `laps` rows);
      car detection failed (``car="unknown"``); or track detection failed
      (``track="unknown"``). The parquet exists and is queryable.
    * ``"failed"`` — nothing salvageable. Either the file could not be read
      from disk (OSError), the IBT/YAML failed to parse so we have no
      channels at all, or the parquet writer itself errored mid-stream.
      No parquet is left behind on a failed read or parse; one may be left
      behind on a writer mid-stream failure (next re-ingest will overwrite).

    Re-ingesting a previously ``"partial"`` or ``"failed"`` session retries it;
    only ``"ok"`` short-circuits.
    """
    # Stage 0: read bytes off disk. A filesystem error here means we have
    # literally nothing to work with — not even a session_id (which we hash
    # from the bytes), so we synthesize one from the path and bail.
    try:
        raw = ibt_path.read_bytes()
    except OSError as exc:
        sid = session_id_from_bytes(str(ibt_path).encode("utf-8"))
        _record_failure(conn, sid=sid, source_path=str(ibt_path), exc=exc)
        return sid, True

    sid = session_id_from_bytes(raw)
    existing = cat.get_session(conn, sid)
    if existing is not None and existing.status == "ok" and not reparse:
        return sid, False

    # Stage 1: parse YAML header + channels. If this raises we have no
    # channels to keep — record `failed` and return.
    try:
        parse = parse_ibt(ibt_path)
    except Exception as exc:  # noqa: BLE001 — every parse failure must register
        _record_failure(conn, sid=sid, source_path=str(ibt_path), exc=exc)
        return sid, True

    # Stage 2: detect car + track. Either failure is salvageable (we still
    # have channels), so accept "unknown" and downgrade status to "partial".
    car = "unknown"
    try:
        car = detect_car(
            yaml_car=parse.yaml_car,
            filename_car=detect_car_from_filename(ibt_path.name),
        )
    except UnknownCarError:
        car = "unknown"

    # slugify_track("") returns "" — collapse to "unknown" so neither field
    # sneaks an empty string into the catalog.
    raw_track = parse.yaml_track or detect_track_from_filename(ibt_path.name) or ""
    track = slugify_track(raw_track) or "unknown"

    # Decide on status. lap_spans empty → no per-lap rows but channels exist.
    # car/track unknown → channels exist but indexing is degraded. Either is
    # "partial". All three healthy → "ok".
    status = "ok"
    error: str | None = None
    if not parse.lap_spans:
        status = "partial"
        error = "no laps detected during segmentation"
    if car == "unknown" or track == "unknown":
        status = "partial"
        # Compose error reason; preserve any prior reason.
        unknown_bits = []
        if car == "unknown":
            unknown_bits.append("car")
        if track == "unknown":
            unknown_bits.append("track")
        unknown_msg = "unknown " + " and ".join(unknown_bits) + " detection failed"
        error = unknown_msg if error is None else f"{error}; {unknown_msg}"

    # Stage 3: write parquet + catalog row. A writer failure here is the
    # least-recoverable case: we may have left a half-written parquet file
    # on disk, but the next re-ingest will overwrite it. Mark `failed` so a
    # re-run picks it up (an `ok` short-circuits otherwise).
    try:
        write_session(
            conn=conn,
            corpus_root=root,
            session_id=sid,
            source_path=str(ibt_path),
            parse=parse,
            car=car,
            track=track,
            status=status,
            error=error,
        )
    except Exception as exc:  # noqa: BLE001 — every writer failure must register
        _record_failure(conn, sid=sid, source_path=str(ibt_path), exc=exc)
    return sid, True
