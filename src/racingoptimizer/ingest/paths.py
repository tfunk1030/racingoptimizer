"""Filesystem layout for the parsed corpus.

Layout:
    <corpus_root>/
        catalog.sqlite
        sessions/<car>/<track>/<session_id>.parquet
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "RACINGOPTIMIZER_CORPUS"


def default_corpus_root() -> Path:
    """Repo-relative default for the corpus."""
    # paths.py lives at .../src/racingoptimizer/ingest/paths.py
    # repo root is four parents up.
    return Path(__file__).resolve().parents[3] / "corpus"


def resolve_corpus_root(explicit: Path | None) -> Path:
    """Pick the corpus root: explicit arg > env var > default."""
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env)
    return default_corpus_root()


def catalog_path(corpus_root: Path) -> Path:
    return corpus_root / "catalog.sqlite"


def parquet_path(corpus_root: Path, *, car: str, track: str, session_id: str) -> Path:
    return corpus_root / "sessions" / car / track / f"{session_id}.parquet"
