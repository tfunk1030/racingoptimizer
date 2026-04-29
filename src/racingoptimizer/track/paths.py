"""Filesystem layout for persisted track models.

Layout (under the corpus root resolved by `racingoptimizer.ingest.paths`):

    <corpus_root>/
        track_models/
            <track>.<sessions_hash>.parquet         # per-session per-bin
            <track>.<sessions_hash>.summary.parquet # cross-session collapse
            <track>.latest.json                     # pointer to freshest hash
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from racingoptimizer.ingest.paths import resolve_corpus_root


def track_models_root(corpus_root: Path | None = None) -> Path:
    root = resolve_corpus_root(corpus_root) / "track_models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_path(corpus_root: Path | None, track: str, sessions_hash: str) -> Path:
    return track_models_root(corpus_root) / f"{track}.{sessions_hash}.parquet"


def summary_path(corpus_root: Path | None, track: str, sessions_hash: str) -> Path:
    return track_models_root(corpus_root) / f"{track}.{sessions_hash}.summary.parquet"


def latest_pointer_path(corpus_root: Path | None, track: str) -> Path:
    return track_models_root(corpus_root) / f"{track}.latest.json"


def sessions_hash(session_ids: list[str]) -> str:
    return hashlib.sha256(",".join(sorted(session_ids)).encode("utf-8")).hexdigest()[:16]
