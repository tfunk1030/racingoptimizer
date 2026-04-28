from __future__ import annotations

from pathlib import Path

import pytest

from racingoptimizer.ingest.paths import (
    catalog_path,
    default_corpus_root,
    parquet_path,
    resolve_corpus_root,
)


def test_resolve_corpus_root_uses_explicit_arg(tmp_path: Path) -> None:
    assert resolve_corpus_root(tmp_path) == tmp_path


def test_resolve_corpus_root_uses_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RACINGOPTIMIZER_CORPUS", str(tmp_path))
    assert resolve_corpus_root(None) == tmp_path


def test_resolve_corpus_root_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RACINGOPTIMIZER_CORPUS", raising=False)
    assert resolve_corpus_root(None) == default_corpus_root()


def test_catalog_path_is_inside_root(tmp_path: Path) -> None:
    p = catalog_path(tmp_path)
    assert p == tmp_path / "catalog.sqlite"


def test_parquet_path_is_per_car_per_track(tmp_path: Path) -> None:
    p = parquet_path(tmp_path, car="porsche", track="laguna_seca", session_id="abcdef0123456789")
    assert p == tmp_path / "sessions" / "porsche" / "laguna_seca" / "abcdef0123456789.parquet"
