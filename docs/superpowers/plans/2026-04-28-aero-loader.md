# Aero-Map Loader & Interpolator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `racingoptimizer.aero` — a Python module that loads the 33 aero-map JSONs in `aero-maps/`, exposes a per-car `AeroSurface` cache, and answers `(balance_pct, ld_ratio)` queries at any `(front_rh_mm, rear_rh_mm, wing_deg, air_density)` inside (or clamped to) the documented envelope.

**Architecture:** Bottom-up TDD. Pure-data parsing helpers (`loader.py`) first — no scipy required, fully testable against in-temp-dir synthetic JSONs and the real corpus. Then the interpolator (`interpolator.py`) — wraps a parsed `AeroMapData` in an `AeroSurface` with cached per-wing `RegularGridInterpolator`s, applies the air-density correction, and clamps out-of-envelope queries with a logged warning. Finally a smoke test against the real porsche maps that nails the full path. Spec: `docs/superpowers/specs/2026-04-28-aero-loader-design.md`.

**Tech Stack:** Python 3.12+, [`uv`](https://docs.astral.sh/uv/) for environments, `numpy>=1.26` and `scipy>=1.13` for the interpolator, `pytest>=8.0` for tests. The worktree forks from master HEAD; no other package manifest exists. We create a minimal `pyproject.toml` for this slice only (see §13 of the spec for the rationale on parallel-slice manifest reconciliation).

---

## Task 0: Project scaffolding subset

**Files:**
- Create: `pyproject.toml`
- Create: `src/racingoptimizer/__init__.py`
- Create: `src/racingoptimizer/py.typed`
- Create: `src/racingoptimizer/aero/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/aero/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`

- [ ] **Step 1: Write `.gitignore`**

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

# Editor
.vscode/
.idea/
```

- [ ] **Step 2: Write a minimal `pyproject.toml` for this slice**

```toml
[project]
name = "racingoptimizer"
version = "0.0.1"
description = "Physics-based setup optimizer for iRacing GTP cars"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.13",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/racingoptimizer"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

> **Note on parallel slices:** Slice A's worker will write its own `pyproject.toml` adding `pyirsdk, polars, pyarrow, click, pytest-cov, ruff`. The user reconciles at merge time. Do not coordinate with slice A's worker.

- [ ] **Step 3: Write package init files**

`src/racingoptimizer/__init__.py`:
```python
"""racingoptimizer — physics-based setup optimizer for iRacing GTP cars."""
```

> Exactly one line, matching slice A's worker.

`src/racingoptimizer/py.typed`: (empty file)

`src/racingoptimizer/aero/__init__.py`:
```python
"""Aero-map loader and interpolator: ride-height + wing -> (balance, l/d)."""
```

`tests/__init__.py`: (empty file)

`tests/aero/__init__.py`: (empty file)

- [ ] **Step 4: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures for the racingoptimizer test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
AERO_DIR = REPO_ROOT / "aero-maps"


@pytest.fixture
def aero_dir() -> Path:
    """Path to the real aero-maps/ directory in the repo."""
    if not AERO_DIR.is_dir():
        pytest.skip(f"aero-maps/ not present at {AERO_DIR}")
    return AERO_DIR
```

- [ ] **Step 5: Install in editable mode and smoke-test the import**

```bash
uv venv
uv pip install -e ".[dev]"
uv run python -c "import racingoptimizer; import racingoptimizer.aero; print('ok')"
```

Expected: `ok`.

- [ ] **Step 6: Confirm pytest discovers the empty test tree**

```bash
uv run pytest --collect-only
```

Expected: `no tests ran` (zero collected, no errors).

- [ ] **Step 7: Commit**

```bash
git add .gitignore pyproject.toml src/ tests/
git commit -m "chore: scaffold racingoptimizer package and aero/ module"
```

---

## Task 1: `loader.py` — JSON parsing & schema validation

**Files:**
- Create: `src/racingoptimizer/aero/loader.py`
- Create: `tests/aero/test_loader.py`

- [ ] **Step 1: Write the failing tests**

`tests/aero/test_loader.py`:
```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from racingoptimizer.aero.loader import (
    AeroLoadError,
    AeroMapData,
    load_aero_map_data,
    parse_aero_filename,
)


# --- filename parser ---------------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        ("acura_wing_6.0.json", ("acura", 6.0)),
        ("acura_wing_6.5.json", ("acura", 6.5)),
        ("acura_wing_10.0.json", ("acura", 10.0)),
        ("bmw_wing_15.0.json", ("bmw", 15.0)),
        ("porsche_wing_17.0.json", ("porsche", 17.0)),
    ],
)
def test_parse_aero_filename_known(name: str, expected: tuple[str, float]) -> None:
    assert parse_aero_filename(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "wing_6.0.json",                    # missing car
        "acura_wing.json",                  # missing wing angle
        "acura_wing_6.json",                # wing not float-formatted
        "acura_wing_6.0.txt",               # wrong extension
        "acura_6.0.json",                   # missing 'wing_'
    ],
)
def test_parse_aero_filename_invalid(name: str) -> None:
    with pytest.raises(ValueError):
        parse_aero_filename(name)


# --- real-corpus loader ------------------------------------------------------

@pytest.mark.parametrize(
    "car, expected_n_wing, expected_first_wing, expected_last_wing",
    [
        ("acura", 9, 6.0, 10.0),
        ("bmw", 6, 12.0, 17.0),
        ("cadillac", 6, 12.0, 17.0),
        ("ferrari", 6, 12.0, 17.0),
        ("porsche", 6, 12.0, 17.0),
    ],
)
def test_load_real_corpus_per_car(
    aero_dir: Path,
    car: str,
    expected_n_wing: int,
    expected_first_wing: float,
    expected_last_wing: float,
) -> None:
    data = load_aero_map_data(car, aero_dir=aero_dir)
    assert isinstance(data, AeroMapData)
    assert data.car == car
    assert len(data.wing_angles) == expected_n_wing
    assert data.wing_angles[0] == expected_first_wing
    assert data.wing_angles[-1] == expected_last_wing
    # Verified shapes from the spec §4
    assert data.front_rh_mm.shape == (51,)
    assert data.rear_rh_mm.shape == (46,)
    assert data.balance_pct.shape == (expected_n_wing, 51, 46)
    assert data.ld_ratio.shape == (expected_n_wing, 51, 46)
    # Strictly-increasing axes
    assert np.all(np.diff(data.front_rh_mm) > 0)
    assert np.all(np.diff(data.rear_rh_mm) > 0)
    assert np.all(np.diff(np.array(data.wing_angles)) > 0)
    # No NaNs
    assert not np.isnan(data.balance_pct).any()
    assert not np.isnan(data.ld_ratio).any()


def test_load_unknown_car_raises(aero_dir: Path) -> None:
    with pytest.raises(AeroLoadError, match="no aero maps"):
        load_aero_map_data("teslamodels", aero_dir=aero_dir)


def test_load_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_aero_map_data("acura", aero_dir=tmp_path / "does-not-exist")


# --- synthetic schema-violation tests ----------------------------------------

def _write_minimal_aero_json(path: Path, car: str, wing: float, **overrides) -> None:
    payload = {
        "car": car,
        "wing": wing,
        "front_rh_mm": list(range(25, 76)),    # 51 entries, [25, 75]
        "rear_rh_mm": list(range(5, 51)),      # 46 entries, [5, 50]
        "balance_pct": [[50.0] * 46 for _ in range(51)],
        "ld_ratio": [[3.5] * 46 for _ in range(51)],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload))


def test_missing_required_key_raises(tmp_path: Path) -> None:
    bad = tmp_path / "synth_wing_6.0.json"
    payload = {
        "car": "synth",
        "wing": 6.0,
        "front_rh_mm": list(range(25, 76)),
        # missing "rear_rh_mm"
        "balance_pct": [[50.0] * 46 for _ in range(51)],
        "ld_ratio": [[3.5] * 46 for _ in range(51)],
    }
    bad.write_text(json.dumps(payload))
    with pytest.raises(AeroLoadError, match="rear_rh_mm"):
        load_aero_map_data("synth", aero_dir=tmp_path)


def test_balance_shape_mismatch_raises(tmp_path: Path) -> None:
    _write_minimal_aero_json(
        tmp_path / "synth_wing_6.0.json",
        "synth",
        6.0,
        balance_pct=[[50.0] * 46 for _ in range(50)],   # 50 rows, not 51
    )
    with pytest.raises(AeroLoadError, match="balance_pct"):
        load_aero_map_data("synth", aero_dir=tmp_path)


def test_axis_disagreement_across_wings_raises(tmp_path: Path) -> None:
    _write_minimal_aero_json(tmp_path / "synth_wing_6.0.json", "synth", 6.0)
    _write_minimal_aero_json(
        tmp_path / "synth_wing_7.0.json",
        "synth",
        7.0,
        front_rh_mm=list(range(20, 71)),    # 51 entries but different range
    )
    with pytest.raises(AeroLoadError, match="axis"):
        load_aero_map_data("synth", aero_dir=tmp_path)


def test_nan_cell_raises(tmp_path: Path) -> None:
    bad_balance = [[50.0] * 46 for _ in range(51)]
    bad_balance[10][10] = float("nan")
    _write_minimal_aero_json(
        tmp_path / "synth_wing_6.0.json", "synth", 6.0, balance_pct=bad_balance
    )
    with pytest.raises(AeroLoadError, match="NaN"):
        load_aero_map_data("synth", aero_dir=tmp_path)


def test_wing_angles_returned_sorted(tmp_path: Path) -> None:
    # Write in deliberately scrambled order
    for w in [9.0, 6.0, 7.5, 8.0]:
        _write_minimal_aero_json(tmp_path / f"synth_wing_{w}.json", "synth", w)
    data = load_aero_map_data("synth", aero_dir=tmp_path)
    assert list(data.wing_angles) == [6.0, 7.5, 8.0, 9.0]


def test_filename_wing_must_match_payload_wing(tmp_path: Path) -> None:
    _write_minimal_aero_json(tmp_path / "synth_wing_6.0.json", "synth", 7.0)
    with pytest.raises(AeroLoadError, match="wing"):
        load_aero_map_data("synth", aero_dir=tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/aero/test_loader.py -v
```

Expected: collection error / `ModuleNotFoundError: racingoptimizer.aero.loader`.

- [ ] **Step 3: Implement `loader.py`**

```python
"""Disk -> AeroMapData. Pure data parsing; no interpolation, no scipy.

Reads `aero-maps/<car>_wing_<X.X>.json` files and returns an immutable
container with axes and 3D matrices stacked along the wing dimension.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REQUIRED_KEYS = ("car", "wing", "front_rh_mm", "rear_rh_mm", "balance_pct", "ld_ratio")
EXPECTED_FRONT_RH_LEN = 51
EXPECTED_REAR_RH_LEN = 46

_FILENAME_RE = re.compile(r"^(?P<car>[a-z0-9]+)_wing_(?P<wing>\d+\.\d+)\.json$")


class AeroLoadError(ValueError):
    """Raised when an aero JSON fails schema validation or a car has no maps."""


@dataclass(frozen=True)
class AeroMapData:
    """Stacked aero data for one car across all loaded wing angles.

    Wing axis is leading so each per-wing 2D slice is a contiguous (51, 46)
    array — cheap to hand to a per-wing RegularGridInterpolator.
    """

    car: str
    wing_angles: tuple[float, ...]   # sorted ascending
    front_rh_mm: np.ndarray          # shape (51,)
    rear_rh_mm: np.ndarray           # shape (46,)
    balance_pct: np.ndarray          # shape (n_wing, 51, 46)
    ld_ratio: np.ndarray             # shape (n_wing, 51, 46)


def parse_aero_filename(name: str) -> tuple[str, float]:
    """Pull (car, wing_deg) from `<car>_wing_<X.X>.json`.

    >>> parse_aero_filename("acura_wing_6.5.json")
    ('acura', 6.5)
    """
    m = _FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"not an aero-map filename: {name!r}")
    return m["car"], float(m["wing"])


def load_aero_map_data(car: str, *, aero_dir: Path) -> AeroMapData:
    """Load every <car>_wing_*.json under aero_dir and stack into AeroMapData."""
    aero_dir = Path(aero_dir)
    if not aero_dir.is_dir():
        raise FileNotFoundError(f"aero directory not found: {aero_dir}")

    matches: list[tuple[float, Path]] = []
    for entry in sorted(aero_dir.iterdir()):
        if not entry.is_file() or not entry.name.endswith(".json"):
            continue
        try:
            file_car, file_wing = parse_aero_filename(entry.name)
        except ValueError:
            continue
        if file_car == car:
            matches.append((file_wing, entry))

    if not matches:
        raise AeroLoadError(f"no aero maps for car {car!r} under {aero_dir}")

    matches.sort(key=lambda x: x[0])

    front_rh: np.ndarray | None = None
    rear_rh: np.ndarray | None = None
    balance_stack: list[np.ndarray] = []
    ld_stack: list[np.ndarray] = []
    wings: list[float] = []

    for filename_wing, path in matches:
        payload = json.loads(path.read_text())
        for key in REQUIRED_KEYS:
            if key not in payload:
                raise AeroLoadError(f"{path.name}: missing required key {key!r}")

        if payload["car"] != car:
            raise AeroLoadError(
                f"{path.name}: payload car {payload['car']!r} != filename car {car!r}"
            )
        if float(payload["wing"]) != filename_wing:
            raise AeroLoadError(
                f"{path.name}: payload wing {payload['wing']} != filename wing {filename_wing}"
            )

        f_axis = np.asarray(payload["front_rh_mm"], dtype=float)
        r_axis = np.asarray(payload["rear_rh_mm"], dtype=float)
        if f_axis.shape != (EXPECTED_FRONT_RH_LEN,):
            raise AeroLoadError(
                f"{path.name}: front_rh_mm length {f_axis.shape[0]} != {EXPECTED_FRONT_RH_LEN}"
            )
        if r_axis.shape != (EXPECTED_REAR_RH_LEN,):
            raise AeroLoadError(
                f"{path.name}: rear_rh_mm length {r_axis.shape[0]} != {EXPECTED_REAR_RH_LEN}"
            )

        if front_rh is None:
            front_rh, rear_rh = f_axis, r_axis
        elif not (np.array_equal(front_rh, f_axis) and np.array_equal(rear_rh, r_axis)):
            raise AeroLoadError(
                f"{path.name}: rh axis disagrees with sibling wing files for car {car!r}"
            )

        balance = np.asarray(payload["balance_pct"], dtype=float)
        ld = np.asarray(payload["ld_ratio"], dtype=float)
        expected_shape = (EXPECTED_FRONT_RH_LEN, EXPECTED_REAR_RH_LEN)
        if balance.shape != expected_shape:
            raise AeroLoadError(
                f"{path.name}: balance_pct shape {balance.shape} != {expected_shape}"
            )
        if ld.shape != expected_shape:
            raise AeroLoadError(
                f"{path.name}: ld_ratio shape {ld.shape} != {expected_shape}"
            )
        if np.isnan(balance).any() or np.isnan(ld).any():
            raise AeroLoadError(f"{path.name}: NaN cell in balance_pct or ld_ratio")

        wings.append(filename_wing)
        balance_stack.append(balance)
        ld_stack.append(ld)

    assert front_rh is not None and rear_rh is not None  # for type-checker

    return AeroMapData(
        car=car,
        wing_angles=tuple(wings),
        front_rh_mm=front_rh,
        rear_rh_mm=rear_rh,
        balance_pct=np.stack(balance_stack, axis=0),
        ld_ratio=np.stack(ld_stack, axis=0),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/aero/test_loader.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/racingoptimizer/aero/loader.py tests/aero/test_loader.py
git commit -m "feat(aero): JSON loader with schema validation"
```

---

## Task 2: `interpolator.py` — `AeroSurface` + clamp + log + air-density correction

**Files:**
- Create: `src/racingoptimizer/aero/interpolator.py`
- Create: `tests/aero/test_interpolator.py`

- [ ] **Step 1: Write the failing tests**

`tests/aero/test_interpolator.py`:
```python
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from racingoptimizer.aero.interpolator import (
    BASELINE_AIR_DENSITY,
    AeroBounds,
    AeroSurface,
)
from racingoptimizer.aero.loader import load_aero_map_data


@pytest.fixture
def porsche_surface(aero_dir: Path) -> AeroSurface:
    return AeroSurface(load_aero_map_data("porsche", aero_dir=aero_dir))


@pytest.fixture
def acura_surface(aero_dir: Path) -> AeroSurface:
    return AeroSurface(load_aero_map_data("acura", aero_dir=aero_dir))


# --- bounds ------------------------------------------------------------------

def test_bounds_envelope(porsche_surface: AeroSurface) -> None:
    b: AeroBounds = porsche_surface.bounds
    assert b.front_rh_mm == (25.0, 75.0)
    assert b.rear_rh_mm == (5.0, 50.0)
    assert b.wing_deg == (12.0, 17.0)
    assert b.wing_angles == (12.0, 13.0, 14.0, 15.0, 16.0, 17.0)


def test_bounds_acura_half_degree_steps(acura_surface: AeroSurface) -> None:
    assert acura_surface.bounds.wing_deg == (6.0, 10.0)
    assert acura_surface.bounds.wing_angles == (6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0)


# --- exact node lookup -------------------------------------------------------

def test_interpolate_at_grid_node_returns_stored_value(
    porsche_surface: AeroSurface, aero_dir: Path
) -> None:
    raw = load_aero_map_data("porsche", aero_dir=aero_dir)
    # Pick any on-grid (front_rh, rear_rh, wing): 5th front, 10th rear, second wing
    fi, ri, wi = 5, 10, 1
    front = float(raw.front_rh_mm[fi])
    rear = float(raw.rear_rh_mm[ri])
    wing = float(raw.wing_angles[wi])
    bal, ld = porsche_surface.interpolate(front, rear, wing, BASELINE_AIR_DENSITY)
    assert bal == pytest.approx(raw.balance_pct[wi, fi, ri], abs=1e-12)
    assert ld == pytest.approx(raw.ld_ratio[wi, fi, ri], abs=1e-12)


# --- midway-on-wing axis -----------------------------------------------------

def test_interpolate_midway_between_wings_at_grid_node(
    porsche_surface: AeroSurface, aero_dir: Path
) -> None:
    raw = load_aero_map_data("porsche", aero_dir=aero_dir)
    fi, ri = 12, 8
    w_lo = float(raw.wing_angles[2])    # 14.0
    w_hi = float(raw.wing_angles[3])    # 15.0
    front = float(raw.front_rh_mm[fi])
    rear = float(raw.rear_rh_mm[ri])
    bal, ld = porsche_surface.interpolate(front, rear, 0.5 * (w_lo + w_hi), BASELINE_AIR_DENSITY)
    bal_ref = 0.5 * (raw.balance_pct[2, fi, ri] + raw.balance_pct[3, fi, ri])
    ld_ref = 0.5 * (raw.ld_ratio[2, fi, ri] + raw.ld_ratio[3, fi, ri])
    assert bal == pytest.approx(bal_ref, abs=1e-12)
    assert ld == pytest.approx(ld_ref, abs=1e-12)


# --- bilinear at fixed wing --------------------------------------------------

def test_interpolate_bilinear_in_rh_at_fixed_wing(
    porsche_surface: AeroSurface, aero_dir: Path
) -> None:
    raw = load_aero_map_data("porsche", aero_dir=aero_dir)
    wi = 2  # 14.0°
    fi = 5
    ri = 10
    f0, f1 = float(raw.front_rh_mm[fi]), float(raw.front_rh_mm[fi + 1])
    r0, r1 = float(raw.rear_rh_mm[ri]), float(raw.rear_rh_mm[ri + 1])
    f_query = 0.5 * (f0 + f1)
    r_query = 0.5 * (r0 + r1)
    expected_bal = 0.25 * (
        raw.balance_pct[wi, fi, ri]
        + raw.balance_pct[wi, fi + 1, ri]
        + raw.balance_pct[wi, fi, ri + 1]
        + raw.balance_pct[wi, fi + 1, ri + 1]
    )
    expected_ld = 0.25 * (
        raw.ld_ratio[wi, fi, ri]
        + raw.ld_ratio[wi, fi + 1, ri]
        + raw.ld_ratio[wi, fi, ri + 1]
        + raw.ld_ratio[wi, fi + 1, ri + 1]
    )
    bal, ld = porsche_surface.interpolate(
        f_query, r_query, float(raw.wing_angles[wi]), BASELINE_AIR_DENSITY
    )
    assert bal == pytest.approx(expected_bal, abs=1e-12)
    assert ld == pytest.approx(expected_ld, abs=1e-12)


# --- air-density correction --------------------------------------------------

def test_air_density_baseline_returns_raw_ld(
    porsche_surface: AeroSurface, aero_dir: Path
) -> None:
    raw = load_aero_map_data("porsche", aero_dir=aero_dir)
    fi, ri, wi = 5, 10, 1
    bal, ld = porsche_surface.interpolate(
        float(raw.front_rh_mm[fi]),
        float(raw.rear_rh_mm[ri]),
        float(raw.wing_angles[wi]),
        BASELINE_AIR_DENSITY,
    )
    assert ld == pytest.approx(raw.ld_ratio[wi, fi, ri], abs=1e-12)


def test_air_density_doubles_ld_when_density_doubles(
    porsche_surface: AeroSurface, aero_dir: Path
) -> None:
    raw = load_aero_map_data("porsche", aero_dir=aero_dir)
    fi, ri, wi = 5, 10, 1
    front = float(raw.front_rh_mm[fi])
    rear = float(raw.rear_rh_mm[ri])
    wing = float(raw.wing_angles[wi])
    bal_base, ld_base = porsche_surface.interpolate(front, rear, wing, BASELINE_AIR_DENSITY)
    bal_2x, ld_2x = porsche_surface.interpolate(front, rear, wing, 2 * BASELINE_AIR_DENSITY)
    assert ld_2x == pytest.approx(2 * ld_base, abs=1e-12)
    # balance is dimensionless and unaffected by density
    assert bal_2x == pytest.approx(bal_base, abs=1e-12)


def test_zero_or_negative_air_density_raises(porsche_surface: AeroSurface) -> None:
    with pytest.raises(ValueError):
        porsche_surface.interpolate(40.0, 20.0, 14.0, 0.0)
    with pytest.raises(ValueError):
        porsche_surface.interpolate(40.0, 20.0, 14.0, -1.0)


# --- clamp behaviour ---------------------------------------------------------

def test_out_of_envelope_front_rh_clamps_to_edge(
    porsche_surface: AeroSurface, aero_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw = load_aero_map_data("porsche", aero_dir=aero_dir)
    rear = float(raw.rear_rh_mm[10])
    wing = float(raw.wing_angles[2])
    # Reference: at the upper edge (front_rh = 75)
    expected_bal, expected_ld = porsche_surface.interpolate(75.0, rear, wing, BASELINE_AIR_DENSITY)
    with caplog.at_level(logging.WARNING, logger="racingoptimizer.aero"):
        bal, ld = porsche_surface.interpolate(200.0, rear, wing, BASELINE_AIR_DENSITY)
    assert bal == pytest.approx(expected_bal, abs=1e-12)
    assert ld == pytest.approx(expected_ld, abs=1e-12)
    assert any("front_rh_mm" in r.message for r in caplog.records)


def test_out_of_envelope_wing_clamps_to_edge(
    porsche_surface: AeroSurface, caplog: pytest.LogCaptureFixture
) -> None:
    front, rear = 42.5, 22.5
    expected_bal, expected_ld = porsche_surface.interpolate(front, rear, 17.0, BASELINE_AIR_DENSITY)
    with caplog.at_level(logging.WARNING, logger="racingoptimizer.aero"):
        bal, ld = porsche_surface.interpolate(front, rear, 25.0, BASELINE_AIR_DENSITY)
    assert bal == pytest.approx(expected_bal, abs=1e-12)
    assert ld == pytest.approx(expected_ld, abs=1e-12)
    assert any("wing_deg" in r.message for r in caplog.records)


def test_out_of_envelope_does_not_raise(porsche_surface: AeroSurface) -> None:
    # All three axes out of bounds + extreme density. No exception.
    bal, ld = porsche_surface.interpolate(1000.0, -50.0, 0.0, BASELINE_AIR_DENSITY)
    assert np.isfinite(bal) and np.isfinite(ld)


# --- car attribute -----------------------------------------------------------

def test_aero_surface_exposes_car(porsche_surface: AeroSurface) -> None:
    assert porsche_surface.car == "porsche"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/aero/test_interpolator.py -v
```

Expected: collection error / `ModuleNotFoundError: racingoptimizer.aero.interpolator`.

- [ ] **Step 3: Implement `interpolator.py`**

```python
"""AeroMapData -> AeroSurface. Per-wing 2D RegularGridInterpolator + linear
blend on the wing axis + per-call air-density correction.

Out-of-envelope inputs clamp to the nearest grid edge; one warning per axis
that clamps. Calls never raise on geometry — only on physically-invalid
air density.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from racingoptimizer.aero.loader import AeroMapData

logger = logging.getLogger("racingoptimizer.aero")

# ISA sea-level standard atmosphere. Spec §6: per-car overrides land when a
# corpus-mean reference is available from slice A.
BASELINE_AIR_DENSITY: float = 1.225


@dataclass(frozen=True)
class AeroBounds:
    front_rh_mm: tuple[float, float]
    rear_rh_mm: tuple[float, float]
    wing_deg: tuple[float, float]
    wing_angles: tuple[float, ...]


def _clamp(value: float, lo: float, hi: float) -> tuple[float, bool]:
    """Clamp value to [lo, hi]; second return is True if clamping happened."""
    if value < lo:
        return lo, True
    if value > hi:
        return hi, True
    return value, False


class AeroSurface:
    """Queryable aero surface for one car across all loaded wing angles.

    Caches one (balance, ld_ratio) RegularGridInterpolator pair per wing slice;
    a query bracket-searches the wing axis, evaluates the two bracketing 2D
    interpolators, and linearly blends.
    """

    def __init__(
        self,
        data: AeroMapData,
        *,
        baseline_air_density: float = BASELINE_AIR_DENSITY,
    ) -> None:
        self._data = data
        self._baseline_air_density = baseline_air_density
        self._wing_axis = np.asarray(data.wing_angles, dtype=float)

        # One per-wing pair of 2D interpolators. linear method = bilinear.
        self._balance_interps: list[RegularGridInterpolator] = []
        self._ld_interps: list[RegularGridInterpolator] = []
        for wi in range(len(data.wing_angles)):
            self._balance_interps.append(
                RegularGridInterpolator(
                    (data.front_rh_mm, data.rear_rh_mm),
                    data.balance_pct[wi],
                    method="linear",
                    bounds_error=False,
                    fill_value=None,   # extrapolate -> we'll clamp inputs first anyway
                )
            )
            self._ld_interps.append(
                RegularGridInterpolator(
                    (data.front_rh_mm, data.rear_rh_mm),
                    data.ld_ratio[wi],
                    method="linear",
                    bounds_error=False,
                    fill_value=None,
                )
            )

    # --- public surface ------------------------------------------------------

    @property
    def car(self) -> str:
        return self._data.car

    @property
    def baseline_air_density(self) -> float:
        return self._baseline_air_density

    @property
    def bounds(self) -> AeroBounds:
        d = self._data
        return AeroBounds(
            front_rh_mm=(float(d.front_rh_mm[0]), float(d.front_rh_mm[-1])),
            rear_rh_mm=(float(d.rear_rh_mm[0]), float(d.rear_rh_mm[-1])),
            wing_deg=(float(d.wing_angles[0]), float(d.wing_angles[-1])),
            wing_angles=tuple(float(w) for w in d.wing_angles),
        )

    def interpolate(
        self,
        front_rh_mm: float,
        rear_rh_mm: float,
        wing_deg: float,
        air_density: float,
    ) -> tuple[float, float]:
        """Return (balance_pct, ld_ratio_corrected) at the queried point."""
        if air_density <= 0:
            raise ValueError(f"air_density must be > 0, got {air_density!r}")

        d = self._data

        front_clamped, front_was_clamped = _clamp(
            float(front_rh_mm), float(d.front_rh_mm[0]), float(d.front_rh_mm[-1])
        )
        rear_clamped, rear_was_clamped = _clamp(
            float(rear_rh_mm), float(d.rear_rh_mm[0]), float(d.rear_rh_mm[-1])
        )
        wing_clamped, wing_was_clamped = _clamp(
            float(wing_deg), float(self._wing_axis[0]), float(self._wing_axis[-1])
        )

        if front_was_clamped:
            logger.warning(
                "front_rh_mm=%s out of envelope %s for car %s; clamped to %s",
                front_rh_mm, (float(d.front_rh_mm[0]), float(d.front_rh_mm[-1])),
                d.car, front_clamped,
            )
        if rear_was_clamped:
            logger.warning(
                "rear_rh_mm=%s out of envelope %s for car %s; clamped to %s",
                rear_rh_mm, (float(d.rear_rh_mm[0]), float(d.rear_rh_mm[-1])),
                d.car, rear_clamped,
            )
        if wing_was_clamped:
            logger.warning(
                "wing_deg=%s out of envelope %s for car %s; clamped to %s",
                wing_deg, (float(self._wing_axis[0]), float(self._wing_axis[-1])),
                d.car, wing_clamped,
            )

        # Bracket the wing axis. searchsorted with 'right' gives the first index
        # strictly greater than wing_clamped; subtract 1 to get the lower bracket.
        idx = int(np.searchsorted(self._wing_axis, wing_clamped, side="right")) - 1
        idx = max(0, min(idx, len(self._wing_axis) - 2))
        w_lo = float(self._wing_axis[idx])
        w_hi = float(self._wing_axis[idx + 1])
        if w_hi == w_lo:
            t = 0.0
        else:
            t = (wing_clamped - w_lo) / (w_hi - w_lo)

        rh_query = np.array([[front_clamped, rear_clamped]])
        bal_lo = float(self._balance_interps[idx](rh_query)[0])
        bal_hi = float(self._balance_interps[idx + 1](rh_query)[0])
        ld_lo = float(self._ld_interps[idx](rh_query)[0])
        ld_hi = float(self._ld_interps[idx + 1](rh_query)[0])

        balance = (1.0 - t) * bal_lo + t * bal_hi
        ld_raw = (1.0 - t) * ld_lo + t * ld_hi

        ld_corrected = ld_raw * (air_density / self._baseline_air_density)
        return balance, ld_corrected
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/aero/test_interpolator.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/racingoptimizer/aero/interpolator.py tests/aero/test_interpolator.py
git commit -m "feat(aero): AeroSurface interpolator with clamp + air-density correction"
```

---

## Task 3: `aero/__init__.py` re-exports + smoke test

**Files:**
- Edit: `src/racingoptimizer/aero/__init__.py`
- Create: `tests/aero/test_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

`tests/aero/test_smoke.py`:
```python
"""End-to-end smoke tests against the real aero-maps/ corpus."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from racingoptimizer.aero import AeroBounds, AeroSurface, load_aero_maps
from racingoptimizer.aero.interpolator import BASELINE_AIR_DENSITY
from racingoptimizer.aero.loader import load_aero_map_data


CARS_AND_BOUNDS = [
    ("acura",    (6.0, 10.0), 9),
    ("bmw",      (12.0, 17.0), 6),
    ("cadillac", (12.0, 17.0), 6),
    ("ferrari",  (12.0, 17.0), 6),
    ("porsche",  (12.0, 17.0), 6),
]


@pytest.mark.parametrize("car, wing_range, n_wings", CARS_AND_BOUNDS)
def test_load_aero_maps_per_car_smoke(
    aero_dir: Path, car: str, wing_range: tuple[float, float], n_wings: int
) -> None:
    surf = load_aero_maps(car, aero_dir=aero_dir)
    assert isinstance(surf, AeroSurface)
    assert isinstance(surf.bounds, AeroBounds)
    assert surf.car == car
    assert surf.bounds.front_rh_mm == (25.0, 75.0)
    assert surf.bounds.rear_rh_mm == (5.0, 50.0)
    assert surf.bounds.wing_deg == wing_range
    assert len(surf.bounds.wing_angles) == n_wings


def test_porsche_full_envelope_interpolates(aero_dir: Path) -> None:
    """Spec §9 / master-plan e2e: porsche at (42.5, 22.5, 14.5°, 1.225) is
    finite, balance in [0, 100], ld > 0, and matches the hand-precomputed
    reference from the four bracketing JSON corners within 1e-9."""
    surf = load_aero_maps("porsche", aero_dir=aero_dir)
    raw = load_aero_map_data("porsche", aero_dir=aero_dir)

    front, rear, wing = 42.5, 22.5, 14.5
    bal, ld = surf.interpolate(front, rear, wing, BASELINE_AIR_DENSITY)

    assert np.isfinite(bal) and np.isfinite(ld)
    assert 0.0 <= bal <= 100.0
    assert ld > 0.0

    # Hand-precompute reference. front_rh_mm = [25, 26, ..., 75] step 1, so 42.5
    # bisects index 17 (=42) and 18 (=43). Same logic for rear: 22.5 bisects
    # index 17 (=22) and 18 (=23). Wing 14.5 bisects index 2 (=14.0) and 3
    # (=15.0). Trilinear average of the eight corners.
    fi = int(np.searchsorted(raw.front_rh_mm, front, side="right")) - 1
    ri = int(np.searchsorted(raw.rear_rh_mm, rear, side="right")) - 1
    wi = int(np.searchsorted(np.asarray(raw.wing_angles), wing, side="right")) - 1

    f0, f1 = raw.front_rh_mm[fi], raw.front_rh_mm[fi + 1]
    r0, r1 = raw.rear_rh_mm[ri], raw.rear_rh_mm[ri + 1]
    w0, w1 = raw.wing_angles[wi], raw.wing_angles[wi + 1]

    tf = (front - f0) / (f1 - f0)
    tr = (rear - r0) / (r1 - r0)
    tw = (wing - w0) / (w1 - w0)

    def trilinear(arr: np.ndarray) -> float:
        c000 = arr[wi,     fi,     ri]
        c100 = arr[wi + 1, fi,     ri]
        c010 = arr[wi,     fi + 1, ri]
        c110 = arr[wi + 1, fi + 1, ri]
        c001 = arr[wi,     fi,     ri + 1]
        c101 = arr[wi + 1, fi,     ri + 1]
        c011 = arr[wi,     fi + 1, ri + 1]
        c111 = arr[wi + 1, fi + 1, ri + 1]
        c00 = c000 * (1 - tw) + c100 * tw
        c10 = c010 * (1 - tw) + c110 * tw
        c01 = c001 * (1 - tw) + c101 * tw
        c11 = c011 * (1 - tw) + c111 * tw
        c0 = c00 * (1 - tf) + c10 * tf
        c1 = c01 * (1 - tf) + c11 * tf
        return c0 * (1 - tr) + c1 * tr

    expected_bal = trilinear(raw.balance_pct)
    expected_ld = trilinear(raw.ld_ratio)   # baseline density => raw

    assert bal == pytest.approx(expected_bal, abs=1e-9)
    assert ld == pytest.approx(expected_ld, abs=1e-9)


def test_negative_query_does_not_raise(aero_dir: Path) -> None:
    """Out-of-envelope on every axis still returns finite numbers."""
    surf = load_aero_maps("porsche", aero_dir=aero_dir)
    bal, ld = surf.interpolate(200.0, 200.0, 0.0, BASELINE_AIR_DENSITY)
    assert np.isfinite(bal) and np.isfinite(ld)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/aero/test_smoke.py -v
```

Expected: `ImportError: cannot import name 'load_aero_maps' from 'racingoptimizer.aero'`.

- [ ] **Step 3: Wire the public API**

`src/racingoptimizer/aero/__init__.py`:
```python
"""Aero-map loader and interpolator: ride-height + wing -> (balance, l/d).

Public API:
    load_aero_maps(car, *, aero_dir=None) -> AeroSurface
    AeroSurface.interpolate(front_rh_mm, rear_rh_mm, wing_deg, air_density)
    AeroSurface.bounds -> AeroBounds
"""
from __future__ import annotations

from pathlib import Path

from racingoptimizer.aero.interpolator import (
    BASELINE_AIR_DENSITY,
    AeroBounds,
    AeroSurface,
)
from racingoptimizer.aero.loader import (
    AeroLoadError,
    AeroMapData,
    load_aero_map_data,
)

__all__ = [
    "AeroBounds",
    "AeroLoadError",
    "AeroSurface",
    "BASELINE_AIR_DENSITY",
    "load_aero_maps",
]


def _default_aero_dir() -> Path:
    """Repo-relative aero-maps/. __init__.py lives at
    .../src/racingoptimizer/aero/__init__.py — repo root is four parents up."""
    return Path(__file__).resolve().parents[3] / "aero-maps"


def load_aero_maps(car: str, *, aero_dir: Path | None = None) -> AeroSurface:
    """Load every aero-maps/<car>_wing_*.json and wrap in an AeroSurface."""
    root = Path(aero_dir) if aero_dir is not None else _default_aero_dir()
    return AeroSurface(load_aero_map_data(car, aero_dir=root))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/aero/ -v
```

Expected: all green across `test_loader.py`, `test_interpolator.py`, `test_smoke.py`.

- [ ] **Step 5: Commit**

```bash
git add src/racingoptimizer/aero/__init__.py tests/aero/test_smoke.py
git commit -m "feat(aero): public load_aero_maps API and smoke tests"
```

---

## Task 4: Documentation handoff

**Files:**
- (no new files)

This slice does not need its own README. The spec at `docs/superpowers/specs/2026-04-28-aero-loader-design.md` is the authoritative documentation for module behaviour and the open questions list. Downstream slices (D, E, F) consume the public API documented in `src/racingoptimizer/aero/__init__.py`'s docstring + the spec.

- [ ] **Step 1: Confirm the public docstring lists `load_aero_maps`, `AeroSurface.interpolate`, `AeroSurface.bounds`**

```bash
uv run python -c "from racingoptimizer.aero import load_aero_maps; help(load_aero_maps)"
```

Expected: shows the docstring with `(car, *, aero_dir=None)` signature.

- [ ] **Step 2: Run the full aero suite one more time**

```bash
uv run pytest tests/aero/ -v
```

Expected: all green.

- [ ] **Step 3: No commit needed** (documentation is in the spec).

---

## Verification gate

Before declaring the slice done:

- [ ] `uv run pytest tests/aero/ -v` — all green.
- [ ] Spec at `docs/superpowers/specs/2026-04-28-aero-loader-design.md` exists and describes the implemented API.
- [ ] No file under `aero-maps/` has been modified, moved, or deleted.
- [ ] No file under `src/racingoptimizer/ingest/` has been touched.
- [ ] `pyproject.toml` declares only this slice's deps (`numpy`, `scipy`, dev `pytest`); the user reconciles with slice A's PR at merge time.
