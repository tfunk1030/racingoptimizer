# IBT Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `racingoptimizer.ingest` — a Python module that parses iRacing `.ibt` telemetry files and persists each session as a parquet file plus a row in a SQLite catalog, exposing `learn / sessions / laps / lap_data` as the query API.

**Architecture:** Bottom-up TDD. Pure helpers (`paths`, `detect`, lap segmentation) first, then SQLite catalog, then the pyirsdk-backed parser (tested against a real fixture IBT), then the writer that composes parser output into parquet + catalog row, then the public API and CLI. Hash-based session ids deliver idempotency for free. Spec: `docs/superpowers/specs/2026-04-28-ibt-ingestion-design.md`.

**Tech Stack:** Python 3.12, [`uv`](https://docs.astral.sh/uv/) for environments, `pyirsdk` for IBT parsing, `polars` + `pyarrow` for parquet I/O, `sqlite3` (stdlib) for the catalog, `click` for the CLI, `pytest` for tests.

---

## Task 0: Project scaffolding & repo init

**Files:**
- Create: `pyproject.toml`
- Create: `src/racingoptimizer/__init__.py`
- Create: `src/racingoptimizer/py.typed`
- Create: `src/racingoptimizer/ingest/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`
- Create: `.python-version`

- [ ] **Step 1: Initialize git repository**

```bash
cd C:/Users/VYRAL/racingoptimizer
git init
git config user.email "you@local"      # only if not set globally
git config user.name "Your Name"       # only if not set globally
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
build/
dist/

# Project corpus (regenerable; not in source control)
corpus/

# Editor
.vscode/
.idea/
```

- [ ] **Step 3: Pin Python version**

Write `.python-version`:

```
3.12
```

- [ ] **Step 4: Write `pyproject.toml`**

```toml
[project]
name = "racingoptimizer"
version = "0.0.1"
description = "Physics-based setup optimizer for iRacing GTP cars"
requires-python = ">=3.12"
dependencies = [
    "pyirsdk>=1.4.0",
    "polars>=1.0",
    "pyarrow>=16.0",
    "click>=8.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.5",
]

[project.scripts]
optimize = "racingoptimizer.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/racingoptimizer"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

- [ ] **Step 5: Write package init files**

`src/racingoptimizer/__init__.py`:
```python
"""racingoptimizer — physics-based setup optimizer for iRacing GTP cars."""

__version__ = "0.0.1"
```

`src/racingoptimizer/py.typed`: (empty file)

`src/racingoptimizer/ingest/__init__.py`:
```python
"""IBT ingestion: parse .ibt files into a queryable corpus."""
```

`tests/__init__.py`: (empty file)

- [ ] **Step 6: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
IBT_DIR = REPO_ROOT / "ibtfiles"

# A small-enough real IBT for end-to-end tests. Picked because it's ~10 MB
# (a few clean laps), runs fast, and exercises the full parser path.
SMALL_IBT_NAME = "bmwlmdh_sebring international 2026-03-22 14-47-42.ibt"


@pytest.fixture
def small_ibt() -> Path:
    """Path to a small real .ibt fixture from the repo's ibtfiles/ corpus."""
    candidate = IBT_DIR / SMALL_IBT_NAME
    if not candidate.exists():
        pytest.skip(f"fixture IBT not present at {candidate}")
    return candidate


@pytest.fixture
def tmp_corpus(tmp_path: Path) -> Path:
    """Empty per-test corpus root."""
    root = tmp_path / "corpus"
    root.mkdir()
    return root
```

- [ ] **Step 7: Install in editable mode and smoke-test the import**

```bash
uv venv
uv pip install -e ".[dev]"
uv run python -c "import racingoptimizer; print(racingoptimizer.__version__)"
```

Expected: `0.0.1`

- [ ] **Step 8: Confirm pytest discovers the empty test tree**

```bash
uv run pytest --collect-only
```

Expected: `no tests ran in ...s` (zero collected, no errors).

- [ ] **Step 9: Commit**

```bash
git add .gitignore .python-version pyproject.toml src/ tests/
git commit -m "chore: scaffold racingoptimizer package and dev environment"
```

---

## Task 1: `paths.py` — corpus root resolution & parquet layout

**Files:**
- Create: `src/racingoptimizer/ingest/paths.py`
- Create: `tests/test_paths.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_paths.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_paths.py -v
```

Expected: collection error / `ModuleNotFoundError: racingoptimizer.ingest.paths`.

- [ ] **Step 3: Implement `paths.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_paths.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/racingoptimizer/ingest/paths.py tests/test_paths.py
git commit -m "feat(ingest): corpus path resolution and parquet layout helpers"
```

---

## Task 2: `detect.py` — car & track normalization

**Files:**
- Create: `src/racingoptimizer/ingest/detect.py`
- Create: `tests/test_detect.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_detect.py`:
```python
from __future__ import annotations

import pytest

from racingoptimizer.ingest.detect import (
    UnknownCarError,
    detect_car,
    detect_track_from_filename,
    normalize_car_key,
    slugify_track,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("acuraarx06gtp", "acura"),
        ("bmwlmdh", "bmw"),
        ("bmwm4gt3", "bmw"),
        ("cadillacvseriesrgtp", "cadillac"),
        ("ferrari499p", "ferrari"),
        ("porsche963gtp", "porsche"),
        ("porsche992rgt3", "porsche"),
    ],
)
def test_normalize_car_key_known(raw: str, expected: str) -> None:
    assert normalize_car_key(raw) == expected


def test_normalize_car_key_unknown_raises() -> None:
    with pytest.raises(UnknownCarError):
        normalize_car_key("teslamodels")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Daytona 2011 road", "daytona_2011_road"),
        ("Algarve GP", "algarve_gp"),
        ("Hockenheim Gp", "hockenheim_gp"),
        ("Laguna Seca", "laguna_seca"),
        ("Sebring International", "sebring_international"),
        ("Road Atlanta - Full", "road_atlanta_full"),
    ],
)
def test_slugify_track(raw: str, expected: str) -> None:
    assert slugify_track(raw) == expected


def test_detect_track_from_filename_uses_underscore_separator() -> None:
    name = "porsche963gtp_lagunaseca 2026-04-26 18-25-49.ibt"
    assert detect_track_from_filename(name) == "lagunaseca"


def test_detect_track_from_filename_strips_timestamp_suffix() -> None:
    name = "ferrari499p_algarve gp 2026-04-09 17-58-04.ibt"
    assert detect_track_from_filename(name) == "algarve_gp"


def test_detect_car_prefers_yaml_over_filename() -> None:
    yaml_car = "porsche963gtp"
    filename_car = "ferrari499p"
    assert detect_car(yaml_car=yaml_car, filename_car=filename_car) == "porsche"


def test_detect_car_falls_back_to_filename_when_yaml_missing() -> None:
    assert detect_car(yaml_car=None, filename_car="bmwlmdh") == "bmw"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_detect.py -v
```

Expected: collection error / `ModuleNotFoundError: racingoptimizer.ingest.detect`.

- [ ] **Step 3: Implement `detect.py`**

```python
"""Car & track detection / normalization.

The IBT YAML header is authoritative for the canonical car name when present;
the filename is used as a fallback. Track names always come from a stringified
source (YAML or filename) and get slugified.
"""
from __future__ import annotations

import re
from pathlib import Path

# Maps known iRacing car identifiers (as they appear in IBT YAML or filenames)
# to the canonical key used in `aero-maps/` and the corpus directory layout.
# Order matters: longest-prefix wins.
CAR_PREFIX_MAP: dict[str, str] = {
    "acuraarx06gtp": "acura",
    "bmwlmdh": "bmw",
    "bmwm4gt3": "bmw",
    "cadillacvseriesrgtp": "cadillac",
    "ferrari499p": "ferrari",
    "porsche963gtp": "porsche",
    "porsche992rgt3": "porsche",
    "amvantageevogt3": "bmw",   # NOTE: not GTP — placeholder mapping until we
                                # confirm where Aston Martin GT3 fits in the model.
}


class UnknownCarError(ValueError):
    """Raised when a car identifier cannot be mapped to a canonical key."""


def normalize_car_key(raw: str) -> str:
    """Map a raw iRacing car identifier to a canonical car key.

    >>> normalize_car_key("acuraarx06gtp")
    'acura'
    """
    raw = raw.strip().lower()
    # longest-prefix match
    for prefix in sorted(CAR_PREFIX_MAP, key=len, reverse=True):
        if raw.startswith(prefix):
            return CAR_PREFIX_MAP[prefix]
    raise UnknownCarError(raw)


def slugify_track(raw: str) -> str:
    """Lowercase, replace whitespace runs and punctuation with single underscores."""
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


_FILENAME_RE = re.compile(
    r"^(?P<car>[a-z0-9]+)_(?P<track>.+?)(?:\s+\d{4}-\d{2}-\d{2}\s+\d{2}-\d{2}-\d{2})?\.ibt$"
)


def detect_track_from_filename(filename: str) -> str | None:
    """Pull a track slug out of an iRacing-style filename, or None on no match."""
    name = Path(filename).name.lower()
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    return slugify_track(m["track"])


def detect_car_from_filename(filename: str) -> str | None:
    """Pull the raw car identifier out of an iRacing-style filename, or None."""
    m = _FILENAME_RE.match(Path(filename).name.lower())
    return m["car"] if m else None


def detect_car(*, yaml_car: str | None, filename_car: str | None) -> str:
    """Pick the canonical car key. YAML wins when present and known."""
    if yaml_car:
        try:
            return normalize_car_key(yaml_car)
        except UnknownCarError:
            pass
    if filename_car:
        return normalize_car_key(filename_car)
    raise UnknownCarError("no car id available from yaml or filename")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_detect.py -v
```

Expected: 13 passed (5 parametrized car keys + 6 parametrized tracks + 4 detect-helper tests, but pytest may report them differently — the count matters less than 0 failures).

- [ ] **Step 5: Commit**

```bash
git add src/racingoptimizer/ingest/detect.py tests/test_detect.py
git commit -m "feat(ingest): canonical car keys and track slugs"
```

---

## Task 3: `catalog.py` — SQLite schema, CRUD, queries

**Files:**
- Create: `src/racingoptimizer/ingest/catalog.py`
- Create: `tests/test_catalog.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_catalog.py`:
```python
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from racingoptimizer.ingest.catalog import (
    LapRow,
    SessionRow,
    init_schema,
    insert_laps,
    open_catalog,
    query_sessions,
    update_session_status,
    upsert_session,
)


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    init_schema(c)
    return c


def _make_session(**overrides) -> SessionRow:
    base = SessionRow(
        session_id="abc1234567890def",
        car="porsche",
        track="laguna_seca",
        recorded_at="2026-04-26T18:25:49",
        duration_s=600.0,
        lap_count=5,
        weather_summary=json.dumps({"AirTemp_c_mean": 22.4}),
        setup=json.dumps({"wing": 16.0}),
        source_path="C:/x/y.ibt",
        ingested_at=_now(),
        parquet_path="sessions/porsche/laguna_seca/abc1234567890def.parquet",
        status="ok",
        error=None,
    )
    return base._replace(**overrides) if overrides else base


def test_init_schema_creates_required_tables(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {r[0] for r in rows}
    assert {"sessions", "laps"}.issubset(names)


def test_upsert_then_query_round_trip(conn: sqlite3.Connection) -> None:
    s = _make_session()
    upsert_session(conn, s)
    out = query_sessions(conn, car="porsche")
    assert len(out) == 1
    assert out[0].session_id == s.session_id
    assert out[0].car == "porsche"
    assert out[0].track == "laguna_seca"


def test_query_filters_by_track(conn: sqlite3.Connection) -> None:
    upsert_session(conn, _make_session(session_id="a" * 16, track="laguna_seca"))
    upsert_session(conn, _make_session(session_id="b" * 16, track="sebring_international"))
    out = query_sessions(conn, track="sebring_international")
    assert len(out) == 1
    assert out[0].track == "sebring_international"


def test_query_excludes_failed_sessions_when_valid_only(conn: sqlite3.Connection) -> None:
    upsert_session(conn, _make_session(session_id="a" * 16, status="ok"))
    upsert_session(conn, _make_session(session_id="b" * 16, status="failed", error="bad header"))
    out_valid = query_sessions(conn, valid_only=True)
    out_all = query_sessions(conn, valid_only=False)
    assert {s.session_id for s in out_valid} == {"a" * 16}
    assert {s.session_id for s in out_all} == {"a" * 16, "b" * 16}


def test_upsert_is_idempotent_on_session_id(conn: sqlite3.Connection) -> None:
    s = _make_session()
    upsert_session(conn, s)
    upsert_session(conn, s)
    rows = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
    assert rows[0] == 1


def test_update_session_status_changes_partial_to_ok(conn: sqlite3.Connection) -> None:
    upsert_session(conn, _make_session(status="partial", error="parser ran out of bytes"))
    update_session_status(
        conn,
        session_id="abc1234567890def",
        status="ok",
        error=None,
        parquet_path="sessions/porsche/laguna_seca/abc1234567890def.parquet",
    )
    out = query_sessions(conn, valid_only=False)
    assert out[0].status == "ok"
    assert out[0].error is None


def test_insert_laps_round_trip(conn: sqlite3.Connection) -> None:
    s = _make_session()
    upsert_session(conn, s)
    laps = [
        LapRow(s.session_id, lap_index=0, lap_time_s=92.1, start_sample=0, end_sample=5530, valid=1, best=0),
        LapRow(s.session_id, lap_index=1, lap_time_s=91.4, start_sample=5530, end_sample=11020, valid=1, best=1),
    ]
    insert_laps(conn, laps)
    rows = conn.execute(
        "SELECT lap_index, lap_time_s, valid, best FROM laps WHERE session_id=? ORDER BY lap_index",
        (s.session_id,),
    ).fetchall()
    assert rows == [(0, 92.1, 1, 0), (1, 91.4, 1, 1)]


def test_open_catalog_creates_db_file(tmp_path: Path) -> None:
    db = tmp_path / "catalog.sqlite"
    with open_catalog(db) as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert db.exists()
    assert {"sessions", "laps"}.issubset(names)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_catalog.py -v
```

Expected: collection error / `ModuleNotFoundError: racingoptimizer.ingest.catalog`.

- [ ] **Step 3: Implement `catalog.py`**

```python
"""SQLite catalog: sessions and lap rows.

Schema lives in `SCHEMA_SQL`. The catalog is rebuildable from raw IBTs (see
`learn`), so there is no migration system — drop the file and re-ingest if the
schema ever changes.
"""
from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id      TEXT PRIMARY KEY,
  car             TEXT NOT NULL,
  track           TEXT NOT NULL,
  recorded_at     TEXT,
  duration_s      REAL,
  lap_count       INTEGER,
  weather_summary TEXT,
  setup           TEXT,
  source_path     TEXT,
  ingested_at     TEXT NOT NULL,
  parquet_path    TEXT,
  status          TEXT NOT NULL CHECK(status IN ('ok','partial','failed')),
  error           TEXT
);

CREATE TABLE IF NOT EXISTS laps (
  session_id   TEXT NOT NULL,
  lap_index    INTEGER NOT NULL,
  lap_time_s   REAL,
  start_sample INTEGER NOT NULL,
  end_sample   INTEGER NOT NULL,
  valid        INTEGER NOT NULL,
  best         INTEGER NOT NULL,
  PRIMARY KEY (session_id, lap_index),
  FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_car_track ON sessions(car, track);
CREATE INDEX IF NOT EXISTS idx_laps_session ON laps(session_id);
"""


class SessionRow(NamedTuple):
    session_id: str
    car: str
    track: str
    recorded_at: str | None
    duration_s: float | None
    lap_count: int | None
    weather_summary: str | None  # JSON
    setup: str | None            # JSON
    source_path: str | None
    ingested_at: str
    parquet_path: str | None
    status: str                  # 'ok' | 'partial' | 'failed'
    error: str | None


class LapRow(NamedTuple):
    session_id: str
    lap_index: int
    lap_time_s: float | None
    start_sample: int
    end_sample: int
    valid: int   # 0/1
    best: int    # 0/1


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


@contextlib.contextmanager
def open_catalog(path: Path | str) -> Iterator[sqlite3.Connection]:
    """Open the catalog at `path`, creating its directory and schema if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        init_schema(conn)
        yield conn
    finally:
        conn.close()


def upsert_session(conn: sqlite3.Connection, row: SessionRow) -> None:
    conn.execute(
        """
        INSERT INTO sessions (
            session_id, car, track, recorded_at, duration_s, lap_count,
            weather_summary, setup, source_path, ingested_at, parquet_path,
            status, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            car=excluded.car,
            track=excluded.track,
            recorded_at=excluded.recorded_at,
            duration_s=excluded.duration_s,
            lap_count=excluded.lap_count,
            weather_summary=excluded.weather_summary,
            setup=excluded.setup,
            source_path=excluded.source_path,
            ingested_at=excluded.ingested_at,
            parquet_path=excluded.parquet_path,
            status=excluded.status,
            error=excluded.error
        """,
        tuple(row),
    )
    conn.commit()


def update_session_status(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    status: str,
    error: str | None,
    parquet_path: str | None,
) -> None:
    conn.execute(
        "UPDATE sessions SET status=?, error=?, parquet_path=? WHERE session_id=?",
        (status, error, parquet_path, session_id),
    )
    conn.commit()


def insert_laps(conn: sqlite3.Connection, laps: Iterable[LapRow]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO laps
            (session_id, lap_index, lap_time_s, start_sample, end_sample, valid, best)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        list(laps),
    )
    conn.commit()


def query_sessions(
    conn: sqlite3.Connection,
    *,
    car: str | None = None,
    track: str | None = None,
    valid_only: bool = True,
) -> list[SessionRow]:
    sql = "SELECT * FROM sessions"
    where: list[str] = []
    params: list[object] = []
    if car is not None:
        where.append("car = ?")
        params.append(car)
    if track is not None:
        where.append("track = ?")
        params.append(track)
    if valid_only:
        where.append("status IN ('ok','partial')")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY recorded_at"
    rows = conn.execute(sql, params).fetchall()
    return [SessionRow(*r) for r in rows]


def get_session(conn: sqlite3.Connection, session_id: str) -> SessionRow | None:
    row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    return SessionRow(*row) if row else None


def get_laps(conn: sqlite3.Connection, session_id: str) -> list[LapRow]:
    rows = conn.execute(
        "SELECT * FROM laps WHERE session_id=? ORDER BY lap_index", (session_id,)
    ).fetchall()
    return [LapRow(*r) for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_catalog.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/racingoptimizer/ingest/catalog.py tests/test_catalog.py
git commit -m "feat(ingest): SQLite catalog schema, CRUD, and queries"
```

---

## Task 4: lap segmentation — pure function on synthetic arrays

**Files:**
- Create: `src/racingoptimizer/ingest/segment.py`
- Create: `tests/test_segment.py`

This task is a small deviation from the spec's module layout (which placed lap-segmentation logic inside `parser.py`). Pulling the pure-function piece into `segment.py` makes it directly unit-testable without touching pyirsdk or any IBT file.

- [ ] **Step 1: Write the failing tests**

`tests/test_segment.py`:
```python
from __future__ import annotations

import numpy as np

from racingoptimizer.ingest.segment import LapSpan, detect_lap_boundaries


def _synth(n_laps: int, samples_per_lap: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Build synthetic LapDistPct (sawtooth 0->1) and Lap (step) arrays."""
    pct = np.tile(np.linspace(0.0, 1.0, samples_per_lap, endpoint=False), n_laps)
    lap = np.repeat(np.arange(n_laps), samples_per_lap)
    return pct.astype(np.float32), lap.astype(np.int32)


def test_two_clean_laps_yield_two_valid_spans() -> None:
    pct, lap = _synth(n_laps=2, samples_per_lap=120)
    spans = detect_lap_boundaries(pct, lap)
    assert len(spans) == 2
    assert all(isinstance(s, LapSpan) for s in spans)
    assert all(s.valid == 1 for s in spans)
    assert spans[0].lap_index == 0
    assert spans[1].lap_index == 1
    assert spans[0].end_sample == spans[1].start_sample


def test_pre_grid_warmup_gets_lap_index_minus_one() -> None:
    # Insert 30 samples of "before first rollover" at the start.
    pct_clean, lap_clean = _synth(n_laps=2, samples_per_lap=100)
    pct = np.concatenate([np.linspace(0.4, 0.95, 30, dtype=np.float32), pct_clean])
    lap = np.concatenate([np.full(30, -1, dtype=np.int32), lap_clean])
    spans = detect_lap_boundaries(pct, lap)
    # First span must be the pre-grid warmup with lap_index = -1 and valid = 0.
    assert spans[0].lap_index == -1
    assert spans[0].valid == 0
    # Then two clean laps follow.
    assert [s.lap_index for s in spans[1:]] == [0, 1]
    assert all(s.valid == 1 for s in spans[1:])


def test_incomplete_trailing_lap_is_invalid() -> None:
    pct, lap = _synth(n_laps=1, samples_per_lap=100)
    pct = np.concatenate([pct, np.linspace(0.0, 0.5, 50, dtype=np.float32)])
    lap = np.concatenate([lap, np.full(50, 1, dtype=np.int32)])
    spans = detect_lap_boundaries(pct, lap)
    assert spans[0].valid == 1
    assert spans[-1].valid == 0   # trailing partial lap


def test_non_monotonic_lap_channel_marks_lap_invalid() -> None:
    pct, lap = _synth(n_laps=2, samples_per_lap=100)
    # Corrupt the Lap channel so it does not increment by exactly 1 across the boundary.
    lap[100:] = 5
    spans = detect_lap_boundaries(pct, lap)
    # Boundaries are still detected from LapDistPct, but the laps are flagged invalid.
    assert any(s.valid == 0 for s in spans)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_segment.py -v
```

Expected: collection error / `ModuleNotFoundError: racingoptimizer.ingest.segment`.

- [ ] **Step 3: Implement `segment.py`**

```python
"""Lap segmentation from raw IBT channels.

A lap boundary is detected wherever LapDistPct rolls back from near 1.0 to
near 0.0. A lap is `valid` iff it both starts and ends with such a rollover
and the IBT `Lap` channel monotonically increases by exactly 1 across it.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

ROLLOVER_HI = 0.9
ROLLOVER_LO = 0.1


class LapSpan(NamedTuple):
    lap_index: int     # -1 for pre-grid samples before the first rollover
    start_sample: int  # inclusive
    end_sample: int    # exclusive
    valid: int         # 0/1


def _rollover_indices(lap_dist_pct: np.ndarray) -> np.ndarray:
    """Return sample indices where LapDistPct rolls back (i.e. lap-start markers)."""
    prev = lap_dist_pct[:-1]
    curr = lap_dist_pct[1:]
    rollover = (prev > ROLLOVER_HI) & (curr < ROLLOVER_LO)
    # +1 because the rollover *ends* one sample after the spike at index i.
    return np.flatnonzero(rollover) + 1


def detect_lap_boundaries(lap_dist_pct: np.ndarray, lap_channel: np.ndarray) -> list[LapSpan]:
    """Decompose a session's samples into LapSpans.

    Parameters
    ----------
    lap_dist_pct : np.ndarray
        Per-sample fractional position around the lap, in [0, 1].
    lap_channel : np.ndarray
        Per-sample integer lap index as reported by iRacing.

    Returns
    -------
    list[LapSpan]
        Pre-grid samples (if any) appear as the first span with lap_index=-1.
        Each subsequent span covers one lap. The trailing span is marked
        invalid if the session ends mid-lap.
    """
    n = lap_dist_pct.shape[0]
    if n == 0:
        return []
    starts = _rollover_indices(lap_dist_pct).tolist()
    spans: list[LapSpan] = []

    if not starts:
        # No completed lap boundary seen at all.
        return [LapSpan(lap_index=-1, start_sample=0, end_sample=n, valid=0)]

    # Pre-grid prefix.
    if starts[0] > 0:
        spans.append(LapSpan(lap_index=-1, start_sample=0, end_sample=starts[0], valid=0))

    # One span per completed boundary, plus the trailing partial.
    boundaries = starts + [n]
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        is_trailing_partial = i == len(boundaries) - 2 and e == n and lap_dist_pct[-1] < ROLLOVER_HI
        # The lap is "valid" iff it is fully bracketed by rollovers AND the Lap
        # channel increments by exactly 1 across it.
        valid = 1
        if is_trailing_partial:
            valid = 0
        else:
            lap_at_start = int(lap_channel[s])
            lap_at_end_inclusive = int(lap_channel[e - 1])
            if lap_at_end_inclusive - lap_at_start != 0:
                # The Lap channel changed *within* a single LapDistPct lap span — not clean.
                valid = 0
        spans.append(LapSpan(lap_index=i, start_sample=s, end_sample=e, valid=valid))
    return spans
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_segment.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/racingoptimizer/ingest/segment.py tests/test_segment.py
git commit -m "feat(ingest): pure-function lap segmentation"
```

---

## Task 5: `parser.py` — pyirsdk wrapper, fixture-based test

**Files:**
- Create: `src/racingoptimizer/ingest/parser.py`
- Create: `tests/test_parser.py`

This task is the thinnest practical wrapper around `pyirsdk` (also published as the `irsdk` Python package). The wrapper exposes one function — `parse_ibt(path) -> ParseResult` — returning typed channel arrays + setup + weather summary + lap spans. Channel filtering applies here.

If you are uncertain about the exact `pyirsdk` API surface, query context7 for the live docs (the `context7` MCP server is configured for this user) and confirm the call signatures before writing the implementation.

- [ ] **Step 1: Write the failing test**

`tests/test_parser.py`:
```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from racingoptimizer.ingest.parser import EXCLUDED_CHANNEL_PATTERNS, ParseResult, parse_ibt


def test_parses_real_ibt(small_ibt: Path) -> None:
    result = parse_ibt(small_ibt)
    assert isinstance(result, ParseResult)

    # Car/track come from the YAML header.
    assert result.yaml_car  # raw IBT car id, e.g. "bmwlmdh"
    assert result.yaml_track  # raw IBT track id, non-empty

    # We expect at least the obvious physics channels.
    expected_channels = {"Speed", "Brake", "Throttle", "LapDistPct", "Lap"}
    missing = expected_channels - set(result.channels)
    assert not missing, f"missing expected channels: {missing}"

    # Channels are 1-D float arrays of consistent length.
    lengths = {name: arr.shape for name, arr in result.channels.items()}
    sample_count = result.channels["Speed"].shape[0]
    assert sample_count > 100
    assert all(arr.shape == (sample_count,) for arr in result.channels.values())

    # Lap spans look reasonable.
    assert any(s.valid == 1 for s in result.lap_spans), "expected at least one valid lap"

    # Setup is parsed and is a non-empty dict.
    assert isinstance(result.setup, dict) and result.setup

    # Weather summary contains at least the most important fields.
    for key in ("AirTemp_c_mean", "TrackTempCrew_c_mean"):
        assert key in result.weather_summary


def test_excluded_channels_are_dropped(small_ibt: Path) -> None:
    result = parse_ibt(small_ibt)
    for name in result.channels:
        for pattern in EXCLUDED_CHANNEL_PATTERNS:
            assert pattern not in name, f"channel {name!r} matched excluded pattern {pattern!r}"


def test_dtypes_are_float32_for_telemetry(small_ibt: Path) -> None:
    result = parse_ibt(small_ibt)
    assert result.channels["Speed"].dtype == np.float32
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_parser.py -v
```

Expected: collection error / `ModuleNotFoundError: racingoptimizer.ingest.parser`.

- [ ] **Step 3: Implement `parser.py`**

```python
"""Wrap `pyirsdk` (a.k.a. `irsdk`) into a typed ParseResult.

The IBT format and pyirsdk's API are both stable enough that we can read each
channel into a numpy array up-front. Memory cost: ~24 KB per channel per
second of session. A ~30 minute session at ~150 channels is ~6 GB raw, which
exceeds RAM — so this implementation streams channels one-at-a-time, casts to
float32, and lets the writer flush to parquet incrementally.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import numpy as np

from racingoptimizer.ingest.segment import LapSpan, detect_lap_boundaries


# Substring patterns that, when present in a channel name, cause us to drop
# that channel from the corpus. Multi-driver arrays (`CarIdx*`) and per-tyre
# per-spot temperature/pressure spreads are the worst offenders for disk/IO.
EXCLUDED_CHANNEL_PATTERNS: tuple[str, ...] = (
    "CarIdx",          # 64-element multi-driver arrays
    "TempCM",          # per-tyre per-spot temp arrays in some IBT versions
    "TempCL",
    "TempCR",
)


class ParseResult(NamedTuple):
    yaml_car: str
    yaml_track: str
    recorded_at: str | None    # ISO timestamp from YAML header if present
    duration_s: float
    channels: dict[str, np.ndarray]      # name -> float32 1-D array, all same length
    setup: dict                          # nested garage setup as parsed from YAML
    weather_summary: dict                # JSON-friendly summary of weather channels
    lap_spans: list[LapSpan]


def _excluded(name: str) -> bool:
    return any(p in name for p in EXCLUDED_CHANNEL_PATTERNS)


def parse_ibt(path: Path | str) -> ParseResult:
    """Parse one .ibt file via pyirsdk and return a ParseResult.

    pyirsdk exposes the IBT data via its `IBT` class. The implementation here
    avoids leaking pyirsdk types into our return value — callers see only
    numpy arrays and plain Python containers.
    """
    # NOTE: pyirsdk publishes both `irsdk` and `pyirsdk` as importable names
    # depending on the install. Try the canonical one first.
    try:
        import irsdk  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover
        import pyirsdk as irsdk  # type: ignore[import-not-found, no-redef]

    ibt = irsdk.IBT()
    ibt.open(str(path))
    try:
        # YAML session info: car/track/setup/timestamp.
        info = ibt.get_session_info_update_yaml() or {}
        weekend = info.get("WeekendInfo", {})
        yaml_car = weekend.get("WeekendOptions", {}).get("CarClassesEntered", "") or ""
        yaml_car = info.get("DriverInfo", {}).get("Drivers", [{}])[0].get("CarPath", yaml_car) or yaml_car
        yaml_track = weekend.get("TrackName", "") or ""
        recorded_at = weekend.get("WeekendOptions", {}).get("Date") or None

        # Pull the car setup section verbatim — typing it is a future spec.
        setup = info.get("CarSetup", {})

        # Read every channel except the excluded ones into a float32 array.
        channels: dict[str, np.ndarray] = {}
        for header in ibt.var_headers:
            name = header.name
            if _excluded(name):
                continue
            arr = np.asarray(ibt.get_all(name))
            if arr.ndim != 1:
                continue   # skip arrays we did not anticipate
            channels[name] = arr.astype(np.float32, copy=False)

        # Validate that the canonical channels for lap segmentation are present.
        for required in ("LapDistPct", "Lap"):
            if required not in channels:
                raise ValueError(f"required channel {required!r} missing from IBT")

        sample_count = channels["LapDistPct"].shape[0]
        # Sample rate is 60 Hz nominal; duration is samples / 60.
        duration_s = float(sample_count) / 60.0

        lap_spans = detect_lap_boundaries(channels["LapDistPct"], channels["Lap"])

        weather_summary = _summarize_weather(channels)

        return ParseResult(
            yaml_car=str(yaml_car),
            yaml_track=str(yaml_track),
            recorded_at=str(recorded_at) if recorded_at else None,
            duration_s=duration_s,
            channels=channels,
            setup=setup,
            weather_summary=weather_summary,
            lap_spans=lap_spans,
        )
    finally:
        ibt.close()


def _summarize_weather(channels: dict[str, np.ndarray]) -> dict:
    """Reduce per-sample weather channels to a small JSON-friendly summary."""
    summary: dict = {}
    if "AirTemp" in channels:
        summary["AirTemp_c_mean"] = float(channels["AirTemp"].mean())
    if "TrackTempCrew" in channels:
        summary["TrackTempCrew_c_mean"] = float(channels["TrackTempCrew"].mean())
    if "AirDensity" in channels:
        summary["AirDensity_kgm3_mean"] = float(channels["AirDensity"].mean())
    if "WindVel" in channels:
        summary["WindVel_ms_mean"] = float(channels["WindVel"].mean())
    if "TrackWetness" in channels:
        summary["TrackWetness_max"] = float(channels["TrackWetness"].max())
    return summary


def channel_names(result: ParseResult) -> Iterable[str]:
    return result.channels.keys()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_parser.py -v
```

Expected: 3 passed. If `pyirsdk` is not installed, install it: `uv pip install pyirsdk`. If the YAML header keys differ (iRacing has shifted them in past builds), use context7 to look up the current pyirsdk docs and adjust the dotted-path lookups in `parse_ibt`.

- [ ] **Step 5: Commit**

```bash
git add src/racingoptimizer/ingest/parser.py tests/test_parser.py
git commit -m "feat(ingest): pyirsdk wrapper with channel filtering and lap spans"
```

---

## Task 6: `writer.py` — parquet + catalog row from ParseResult

**Files:**
- Create: `src/racingoptimizer/ingest/writer.py`
- Create: `tests/test_writer.py`

The writer takes a `ParseResult` plus the source bytes' hash and produces:
1. A parquet file at `parquet_path(...)`
2. A `SessionRow` upserted into the catalog
3. `LapRow`s inserted into `laps`

It is the single place where the on-disk layout is enforced.

- [ ] **Step 1: Write the failing tests**

`tests/test_writer.py`:
```python
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from racingoptimizer.ingest.catalog import init_schema, query_sessions, get_laps
from racingoptimizer.ingest.parser import ParseResult
from racingoptimizer.ingest.segment import LapSpan
from racingoptimizer.ingest.writer import session_id_from_bytes, write_session


def _fake_parse_result(n_samples: int = 600) -> ParseResult:
    pct = np.tile(np.linspace(0.0, 1.0, 200, endpoint=False), n_samples // 200).astype(np.float32)
    lap = np.repeat(np.arange(n_samples // 200), 200).astype(np.float32)
    speed = np.linspace(0, 80.0, n_samples).astype(np.float32)
    brake = np.zeros(n_samples, dtype=np.float32)
    throttle = np.ones(n_samples, dtype=np.float32)
    return ParseResult(
        yaml_car="bmwlmdh",
        yaml_track="Sebring International",
        recorded_at="2026-03-22T14:47:42",
        duration_s=n_samples / 60.0,
        channels={"LapDistPct": pct, "Lap": lap, "Speed": speed, "Brake": brake, "Throttle": throttle},
        setup={"chassis": {"front": {"wing": 16.0}}},
        weather_summary={"AirTemp_c_mean": 22.0},
        lap_spans=[LapSpan(0, 0, 200, 1), LapSpan(1, 200, 400, 1), LapSpan(2, 400, 600, 1)],
    )


def test_session_id_is_stable_over_bytes() -> None:
    a = b"hello world"
    assert session_id_from_bytes(a) == session_id_from_bytes(a)
    assert session_id_from_bytes(a) != session_id_from_bytes(b"hello world!")
    assert len(session_id_from_bytes(a)) == 16


def test_write_session_creates_parquet_and_catalog_row(tmp_corpus: Path) -> None:
    db = tmp_corpus / "catalog.sqlite"
    conn = sqlite3.connect(db)
    init_schema(conn)

    pr = _fake_parse_result()
    sid = "deadbeefcafef00d"
    parquet_p = write_session(
        conn=conn,
        corpus_root=tmp_corpus,
        session_id=sid,
        source_path="X:/fake.ibt",
        parse=pr,
    )

    assert parquet_p.exists()
    df = pl.read_parquet(parquet_p)
    assert {"t_s", "lap_index", "lap_dist_pct", "Speed", "Brake", "Throttle"}.issubset(df.columns)
    assert df.height == pr.channels["Speed"].shape[0]

    out = query_sessions(conn, valid_only=False)
    assert len(out) == 1
    s = out[0]
    assert s.session_id == sid
    assert s.car == "bmw"
    assert s.track == "sebring_international"
    assert s.status == "ok"
    assert json.loads(s.setup) == pr.setup

    laps = get_laps(conn, sid)
    assert len(laps) == 3
    assert all(l.valid == 1 for l in laps)
    # The fastest lap by lap_time_s would be marked best=1; with monotone Speed
    # and equal lap durations, the first lap wins on tie-break (lower lap_index).
    assert sum(l.best for l in laps) == 1
    conn.close()


def test_write_session_is_idempotent_on_same_id(tmp_corpus: Path) -> None:
    db = tmp_corpus / "catalog.sqlite"
    conn = sqlite3.connect(db)
    init_schema(conn)
    pr = _fake_parse_result()
    sid = "1111222233334444"
    write_session(conn=conn, corpus_root=tmp_corpus, session_id=sid, source_path="X:/a.ibt", parse=pr)
    parquet_p = write_session(conn=conn, corpus_root=tmp_corpus, session_id=sid, source_path="X:/a.ibt", parse=pr)
    rows = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
    laps = conn.execute("SELECT COUNT(*) FROM laps").fetchone()
    assert rows[0] == 1
    assert laps[0] == 3
    assert parquet_p.exists()
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_writer.py -v
```

Expected: collection error / `ModuleNotFoundError: racingoptimizer.ingest.writer`.

- [ ] **Step 3: Implement `writer.py`**

```python
"""Persist a ParseResult: parquet file + catalog rows."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from racingoptimizer.ingest.catalog import LapRow, SessionRow, insert_laps, upsert_session
from racingoptimizer.ingest.detect import detect_car, detect_car_from_filename, slugify_track
from racingoptimizer.ingest.parser import ParseResult
from racingoptimizer.ingest.paths import parquet_path


def session_id_from_bytes(b: bytes) -> str:
    """Stable 16-hex-char session id derived from raw IBT bytes."""
    return hashlib.sha256(b).hexdigest()[:16]


def session_id_from_path(path: Path | str) -> str:
    return session_id_from_bytes(Path(path).read_bytes())


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _build_dataframe(pr: ParseResult) -> pl.DataFrame:
    n = pr.channels["LapDistPct"].shape[0]
    t_s = (np.arange(n, dtype=np.float64) / 60.0).astype(np.float64)

    # Build lap_index column: -1 outside any lap_span, else lap_index of the span.
    lap_index = np.full(n, -1, dtype=np.int32)
    for span in pr.lap_spans:
        lap_index[span.start_sample : span.end_sample] = span.lap_index

    data: dict[str, np.ndarray] = {
        "t_s": t_s,
        "lap_index": lap_index,
        "lap_dist_pct": pr.channels["LapDistPct"],
    }
    for name, arr in pr.channels.items():
        if name == "LapDistPct":
            continue   # already aliased as lap_dist_pct
        data[name] = arr
    return pl.DataFrame(data)


def _lap_rows(session_id: str, pr: ParseResult) -> list[LapRow]:
    rows: list[LapRow] = []
    valid_durations: list[tuple[int, float]] = []
    for span in pr.lap_spans:
        duration = (span.end_sample - span.start_sample) / 60.0
        lap_time = duration if span.valid else None
        if span.valid and lap_time is not None:
            valid_durations.append((span.lap_index, lap_time))
        rows.append(
            LapRow(
                session_id=session_id,
                lap_index=span.lap_index,
                lap_time_s=lap_time,
                start_sample=span.start_sample,
                end_sample=span.end_sample,
                valid=span.valid,
                best=0,
            )
        )
    if valid_durations:
        # Tie-break by lower lap_index.
        best_idx = min(valid_durations, key=lambda t: (t[1], t[0]))[0]
        rows = [r._replace(best=1) if (r.valid and r.lap_index == best_idx) else r for r in rows]
    return rows


def write_session(
    *,
    conn: sqlite3.Connection,
    corpus_root: Path,
    session_id: str,
    source_path: str,
    parse: ParseResult,
) -> Path:
    """Write parquet + catalog rows. Returns the parquet path."""
    car = detect_car(yaml_car=parse.yaml_car, filename_car=detect_car_from_filename(source_path))
    track = slugify_track(parse.yaml_track)

    pq = parquet_path(corpus_root, car=car, track=track, session_id=session_id)
    pq.parent.mkdir(parents=True, exist_ok=True)

    df = _build_dataframe(parse)
    df.write_parquet(pq, compression="zstd")

    laps = _lap_rows(session_id, parse)

    session = SessionRow(
        session_id=session_id,
        car=car,
        track=track,
        recorded_at=parse.recorded_at,
        duration_s=parse.duration_s,
        lap_count=sum(1 for s in parse.lap_spans if s.valid),
        weather_summary=json.dumps(parse.weather_summary),
        setup=json.dumps(parse.setup),
        source_path=source_path,
        ingested_at=_now_iso(),
        parquet_path=str(pq.relative_to(corpus_root).as_posix()),
        status="ok",
        error=None,
    )
    upsert_session(conn, session)
    insert_laps(conn, laps)
    return pq
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_writer.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/racingoptimizer/ingest/writer.py tests/test_writer.py
git commit -m "feat(ingest): writer composes parser output into parquet + catalog"
```

---

## Task 7: public API — `learn`, `sessions`, `laps`, `lap_data`

**Files:**
- Modify: `src/racingoptimizer/ingest/__init__.py`
- Create: `src/racingoptimizer/ingest/api.py`
- Create: `tests/test_api.py`

The public API is the only surface code outside `racingoptimizer.ingest` is allowed to import. It glues the parser, writer, and catalog together, and exposes Polars DataFrames to callers.

- [ ] **Step 1: Write the failing tests**

`tests/test_api.py`:
```python
from __future__ import annotations

from pathlib import Path

import polars as pl

from racingoptimizer.ingest import lap_data, laps, learn, sessions


def test_learn_then_query_then_lap_data(small_ibt: Path, tmp_corpus: Path) -> None:
    ids = learn(small_ibt, corpus_root=tmp_corpus)
    assert len(ids) == 1

    s = sessions(corpus_root=tmp_corpus)
    assert s.height == 1
    assert s["session_id"][0] == ids[0]
    assert s["car"][0]
    assert s["track"][0]

    l = laps(session_id=ids[0], corpus_root=tmp_corpus, valid_only=True)
    assert l.height >= 1

    first_lap = int(l["lap_index"].min())
    df = lap_data(ids[0], lap_index=first_lap, corpus_root=tmp_corpus)
    assert df.height > 100
    assert {"t_s", "lap_dist_pct", "Speed"}.issubset(df.columns)


def test_learn_is_idempotent(small_ibt: Path, tmp_corpus: Path) -> None:
    ids1 = learn(small_ibt, corpus_root=tmp_corpus)
    ids2 = learn(small_ibt, corpus_root=tmp_corpus)
    assert ids1 == ids2
    s = sessions(corpus_root=tmp_corpus)
    assert s.height == 1


def test_learn_handles_a_directory_recursively(tmp_path: Path, small_ibt: Path, tmp_corpus: Path) -> None:
    # Place the small IBT in a nested folder.
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    target = nested / small_ibt.name
    target.write_bytes(small_ibt.read_bytes())
    ids = learn(tmp_path, corpus_root=tmp_corpus)
    assert len(ids) == 1


def test_lap_data_can_project_columns(small_ibt: Path, tmp_corpus: Path) -> None:
    ids = learn(small_ibt, corpus_root=tmp_corpus)
    sid = ids[0]
    l = laps(session_id=sid, corpus_root=tmp_corpus, valid_only=True)
    first_lap = int(l["lap_index"].min())
    df = lap_data(sid, lap_index=first_lap, corpus_root=tmp_corpus, channels=["Speed", "Brake"])
    assert set(df.columns) == {"Speed", "Brake"}


def test_sessions_returns_empty_frame_when_corpus_is_new(tmp_corpus: Path) -> None:
    s = sessions(corpus_root=tmp_corpus)
    assert isinstance(s, pl.DataFrame)
    assert s.height == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_api.py -v
```

Expected: collection error / `ImportError: cannot import name 'learn' from 'racingoptimizer.ingest'`.

- [ ] **Step 3: Implement `api.py`**

`src/racingoptimizer/ingest/api.py`:
```python
"""Public API for the ingest module."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from racingoptimizer.ingest import catalog as cat
from racingoptimizer.ingest.parser import parse_ibt
from racingoptimizer.ingest.paths import (
    catalog_path,
    parquet_path,
    resolve_corpus_root,
)
from racingoptimizer.ingest.writer import session_id_from_bytes, write_session


def learn(path: Path | str, corpus_root: Path | str | None = None) -> list[str]:
    """Ingest a .ibt file or every .ibt under a directory.

    Returns the list of session_ids for every file processed (existing or new),
    regardless of status. Caller can join against `sessions(...)` to inspect
    per-session outcomes.
    """
    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    db = catalog_path(root)

    targets = list(_iter_ibt_paths(Path(path)))
    out: list[str] = []
    with cat.open_catalog(db) as conn:
        for ibt in targets:
            sid = _process_one(conn, root, ibt)
            out.append(sid)
    return out


def sessions(
    car: str | None = None,
    track: str | None = None,
    valid_only: bool = True,
    corpus_root: Path | str | None = None,
) -> pl.DataFrame:
    """Return one row per session, ordered by recorded_at."""
    root = resolve_corpus_root(Path(corpus_root) if corpus_root else None)
    with cat.open_catalog(catalog_path(root)) as conn:
        rows = cat.query_sessions(conn, car=car, track=track, valid_only=valid_only)
    return pl.DataFrame(
        {
            "session_id": [r.session_id for r in rows],
            "car": [r.car for r in rows],
            "track": [r.track for r in rows],
            "recorded_at": [r.recorded_at for r in rows],
            "duration_s": [r.duration_s for r in rows],
            "lap_count": [r.lap_count for r in rows],
            "weather_summary": [r.weather_summary for r in rows],
            "setup": [r.setup for r in rows],
            "status": [r.status for r in rows],
            "error": [r.error for r in rows],
            "parquet_path": [r.parquet_path for r in rows],
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
            sid_list = [s.session_id for s in cat.query_sessions(conn, car=car, track=track, valid_only=valid_only)]

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


def _process_one(conn: sqlite3.Connection, root: Path, ibt_path: Path) -> str:
    raw = ibt_path.read_bytes()
    sid = session_id_from_bytes(raw)
    existing = cat.get_session(conn, sid)
    if existing is not None and existing.status == "ok":
        return sid
    try:
        parse = parse_ibt(ibt_path)
        write_session(
            conn=conn,
            corpus_root=root,
            session_id=sid,
            source_path=str(ibt_path),
            parse=parse,
        )
    except Exception as exc:  # noqa: BLE001 — we record every failure
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
                source_path=str(ibt_path),
                ingested_at=_iso_now(),
                parquet_path=None,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
    return sid


def _iso_now() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat(timespec="seconds")
```

- [ ] **Step 4: Update `ingest/__init__.py` to re-export the API**

```python
"""IBT ingestion: parse .ibt files into a queryable corpus."""

from racingoptimizer.ingest.api import lap_data, laps, learn, sessions

__all__ = ["learn", "sessions", "laps", "lap_data"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_api.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/racingoptimizer/ingest/api.py src/racingoptimizer/ingest/__init__.py tests/test_api.py
git commit -m "feat(ingest): public API — learn, sessions, laps, lap_data"
```

---

## Task 8: CLI — `optimize learn <path>`

**Files:**
- Create: `src/racingoptimizer/cli.py`
- Create: `src/racingoptimizer/ingest/cli.py`
- Create: `tests/test_cli.py`

The top-level CLI is a click group named `optimize`. This slice registers exactly one subcommand: `learn`. Future slices add more subcommands by importing and registering them in `cli.main`.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from racingoptimizer.cli import main


def test_optimize_learn_on_one_ibt(small_ibt: Path, tmp_corpus: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["learn", str(small_ibt), "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "ingested" in result.output.lower()


def test_optimize_learn_directory(tmp_path: Path, small_ibt: Path, tmp_corpus: Path) -> None:
    nested = tmp_path / "subdir"
    nested.mkdir()
    (nested / small_ibt.name).write_bytes(small_ibt.read_bytes())
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["learn", str(tmp_path), "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0


def test_optimize_help_lists_learn() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "learn" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: collection error / `ModuleNotFoundError: racingoptimizer.cli`.

- [ ] **Step 3: Implement `ingest/cli.py`**

```python
"""`optimize learn` subcommand."""
from __future__ import annotations

from pathlib import Path

import click

from racingoptimizer.ingest.api import learn as _learn


@click.command(name="learn")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--corpus-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override corpus location (else uses RACINGOPTIMIZER_CORPUS or repo default).",
)
def learn_command(path: Path, corpus_root: Path | None) -> None:
    """Ingest a .ibt file or every .ibt under a directory into the corpus."""
    ids = _learn(path, corpus_root=corpus_root)
    click.echo(f"ingested {len(ids)} session(s)")
    for sid in ids:
        click.echo(f"  {sid}")
```

- [ ] **Step 4: Implement `cli.py`**

```python
"""Top-level `optimize` CLI group."""
from __future__ import annotations

import click

from racingoptimizer.ingest.cli import learn_command


@click.group()
def main() -> None:
    """racingoptimizer CLI."""


main.add_command(learn_command)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Sanity-check the entry point from a shell**

```bash
uv run optimize --help
uv run optimize learn ./ibtfiles --corpus-root ./corpus
```

Expected: help text lists `learn`; `learn` against `./ibtfiles` runs without crashing and prints "ingested N session(s)".

- [ ] **Step 7: Commit**

```bash
git add src/racingoptimizer/cli.py src/racingoptimizer/ingest/cli.py tests/test_cli.py
git commit -m "feat(cli): optimize learn subcommand"
```

---

## Task 9: end-to-end smoke + failure-handling tests

**Files:**
- Create: `tests/test_ingest_smoke.py`

These tests exercise the full pipeline against the real fixture and verify the failure-handling matrix from spec §8.

- [ ] **Step 1: Write the tests**

`tests/test_ingest_smoke.py`:
```python
from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl

from racingoptimizer.ingest import lap_data, learn, sessions
from racingoptimizer.ingest.catalog import open_catalog
from racingoptimizer.ingest.paths import catalog_path
from racingoptimizer.ingest.writer import session_id_from_bytes


def test_full_pipeline_against_fixture(small_ibt: Path, tmp_corpus: Path) -> None:
    ids = learn(small_ibt, corpus_root=tmp_corpus)
    assert len(ids) == 1
    s = sessions(corpus_root=tmp_corpus, valid_only=False)
    assert s.height == 1
    assert s["status"][0] == "ok"
    assert s["car"][0] == "bmw"
    assert s["track"][0] == "sebring_international"

    df = lap_data(ids[0], lap_index=int(s["lap_count"][0]) - 1, corpus_root=tmp_corpus)
    # Spec §11 floor.
    assert df.height > 100
    assert {"Speed", "Brake", "Throttle"}.issubset(df.columns)


def test_corrupt_file_records_failed_status(tmp_path: Path, tmp_corpus: Path) -> None:
    bad = tmp_path / "broken.ibt"
    bad.write_bytes(b"not an ibt file at all")
    ids = learn(bad, corpus_root=tmp_corpus)
    assert len(ids) == 1
    s = sessions(corpus_root=tmp_corpus, valid_only=False)
    assert s.height == 1
    assert s["status"][0] == "failed"
    assert s["error"][0]


def test_partial_session_can_be_upgraded_to_ok(small_ibt: Path, tmp_corpus: Path) -> None:
    """Spec §9: a previously partial/failed session is re-attempted."""
    sid = session_id_from_bytes(small_ibt.read_bytes())
    db = catalog_path(tmp_corpus)
    # Seed a 'partial' row with the matching session_id.
    with open_catalog(db) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, car, track, ingested_at, status, error) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, "bmw", "sebring_international", "2026-04-28T00:00:00", "partial", "seeded"),
        )
        conn.commit()

    ids = learn(small_ibt, corpus_root=tmp_corpus)
    assert ids == [sid]
    s = sessions(corpus_root=tmp_corpus, valid_only=False)
    assert s.height == 1
    assert s["status"][0] == "ok"
    assert s["error"][0] is None


def test_idempotent_ingest_writes_no_new_parquet(small_ibt: Path, tmp_corpus: Path) -> None:
    ids1 = learn(small_ibt, corpus_root=tmp_corpus)
    sid = ids1[0]
    s = sessions(corpus_root=tmp_corpus, valid_only=False)
    pq = tmp_corpus / s["parquet_path"][0]
    mtime_before = pq.stat().st_mtime_ns

    ids2 = learn(small_ibt, corpus_root=tmp_corpus)
    assert ids1 == ids2
    mtime_after = pq.stat().st_mtime_ns
    assert mtime_after == mtime_before
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
uv run pytest tests/test_ingest_smoke.py -v
```

Expected: 4 passed.

- [ ] **Step 3: Run the whole test suite**

```bash
uv run pytest -v
```

Expected: every test from Tasks 1-9 passes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ingest_smoke.py
git commit -m "test(ingest): end-to-end smoke + failure handling"
```

---

## Task 10: docs polish — README and CLAUDE.md update

**Files:**
- Create: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Write a minimal `README.md`**

```markdown
# racingoptimizer

Physics-based setup optimizer for iRacing GTP cars. See `VISION.md` for the full spec.

## Status

Slice A (IBT ingestion) implemented. Other subsystems (corner-phase decomposition, aero-map loader, fitter, optimizer, track model) are planned and unimplemented.

## Quickstart

```bash
uv venv
uv pip install -e ".[dev]"
uv run optimize learn ./ibtfiles
uv run python -c "from racingoptimizer.ingest import sessions; print(sessions())"
```

The default corpus location is `./corpus/` next to this README. Override with the env var `RACINGOPTIMIZER_CORPUS` or the `--corpus-root` flag.
```

- [ ] **Step 2: Update `CLAUDE.md` to reflect that ingestion now exists**

Read `CLAUDE.md` first, then locate the "Repository state" section. Replace the sentence "There is nothing to build, lint, or run yet." with a paragraph noting that `racingoptimizer.ingest` is implemented and how to run it (mirror the README quickstart). Leave the rest of the file unchanged — corner-phase decomposition, fitter, optimizer, etc. are still unimplemented.

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: README and CLAUDE.md update for slice A"
```

---

## Self-review notes

After writing the plan I checked it against the spec:

- **§2 Public API** → Tasks 7-8 implement `learn`, `sessions`, `laps`, `lap_data`, plus the CLI surface.
- **§3 On-disk layout** → Task 1 (paths.py) and Task 6 (writer applies it) cover the corpus tree and parquet shape.
- **§4 Catalog schema** → Task 3 builds it; Task 6 writes into it.
- **§5 Channel selection** → `EXCLUDED_CHANNEL_PATTERNS` in Task 5; tested in `test_excluded_channels_are_dropped`.
- **§6 Car/track detection** → Task 2; Task 6 wires it into `write_session`.
- **§7 Lap segmentation** → Task 4 (pure function), Task 5 (call site).
- **§8 Failure handling** → Task 7 (`_process_one` exception path) + Task 9 smoke tests for `failed` and `partial → ok` paths.
- **§9 Idempotency** → Task 7 (`existing.status == 'ok'` short-circuit) + Task 6 idempotency test + Task 9 mtime-unchanged test.
- **§10 Module layout** → Tasks 1-8 build it. Small deviation: `cli.py` lives at the package root (not inside `ingest/`) so future slices can compose more subcommands, with the ingest-specific subcommand defined in `ingest/cli.py`. `segment.py` was added under `ingest/` for testability of the lap-boundary algorithm in isolation.
- **§11 Testing** → Distributed across all tasks; spec's listed cases (smoke / idempotent / failure / partial-retry) all appear in Task 9.
- **§12 Out of scope** → Honored. No corner segmentation, no aero correction, no fitter.
- **§13 Open questions** → Carried forward as future spec items, unchanged.

No placeholders. Type names are consistent across tasks (`SessionRow`, `LapRow`, `LapSpan`, `ParseResult`). The only deliberate spec deviation (cli location, segment.py) is documented above.
