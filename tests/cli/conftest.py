"""Per-car fixture catalog for the CLI smoke matrix."""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
IBT_DIR = REPO_ROOT / "ibtfiles"


# Canonical fixture per car for the per-car smoke matrix. Each fixture must
# contain ≥3 valid laps so slice E's `fit` produces ≥1 trained quadruple
# (1-lap fixtures collapse to 1 row per (corner, phase) and yield no fitters).
PER_CAR_FIXTURES: dict[str, tuple[str, str]] = {
    # car -> (filename, expected track substring)
    "bmw": (
        "bmwlmdh_sebring international 2026-03-22 14-52-24.ibt",
        "sebring",
    ),
    "acura": (
        "acuraarx06gtp_daytona 2011 road 2026-04-03 20-31-55.ibt",
        "daytona",
    ),
    "cadillac": (
        "cadillacvseriesrgtp_lagunaseca 2026-04-27 19-50-46.ibt",
        "lagunaseca",
    ),
    "ferrari": (
        "ferrari499p_hockenheim gp 2026-03-31 15-49-42.ibt",
        "hockenheim",
    ),
    "porsche": (
        "porsche963gtp_algarve gp 2026-04-07 15-49-17.ibt",
        "algarve",
    ),
}


def fixture_path(car: str) -> Path | None:
    """Return the canonical fixture path for `car` if it exists, else None."""
    name, _track = PER_CAR_FIXTURES[car]
    p = IBT_DIR / name
    return p if p.exists() else None


@pytest.fixture
def per_car_fixture(request) -> tuple[str, Path, str]:
    """Parametrised (car, ibt_path, track_substring) tuple per canonical car."""
    car = request.param
    name, track_sub = PER_CAR_FIXTURES[car]
    p = IBT_DIR / name
    if not p.exists():
        pytest.skip(f"fixture not present for {car}: {p}")
    return car, p, track_sub
