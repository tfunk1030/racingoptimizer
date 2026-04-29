"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
IBT_DIR = REPO_ROOT / "ibtfiles"
AERO_DIR = REPO_ROOT / "aero-maps"

# A small-enough real IBT for end-to-end tests. Picked because it's ~10 MB
# (a few clean laps), runs fast, and exercises the full parser path.
SMALL_IBT_NAME = "bmwlmdh_sebring international 2026-03-22 14-47-42.ibt"

# A multi-lap BMW Sebring fixture. Slice E's `fit` needs ≥3 rows per
# (parameter, corner, phase, channel) quadruple — 1-lap fixtures collapse
# to 1 row per quadruple and produce zero fitters. ~56 MB / 7 valid laps.
MULTI_LAP_IBT_NAME = "bmwlmdh_sebring international 2026-03-22 14-52-24.ibt"


@pytest.fixture
def small_ibt() -> Path:
    """Path to a small real .ibt fixture from the repo's ibtfiles/ corpus."""
    candidate = IBT_DIR / SMALL_IBT_NAME
    if not candidate.exists():
        pytest.skip(f"fixture IBT not present at {candidate}")
    return candidate


@pytest.fixture
def multi_lap_ibt() -> Path:
    """Path to a multi-lap real .ibt fixture (BMW Sebring, ≥3 valid laps)."""
    candidate = IBT_DIR / MULTI_LAP_IBT_NAME
    if not candidate.exists():
        pytest.skip(f"multi-lap fixture not present at {candidate}")
    return candidate


@pytest.fixture
def tmp_corpus(tmp_path: Path) -> Path:
    """Empty per-test corpus root."""
    root = tmp_path / "corpus"
    root.mkdir()
    return root


@pytest.fixture
def aero_dir() -> Path:
    """Path to the real aero-maps/ directory in the repo."""
    if not AERO_DIR.is_dir():
        pytest.skip(f"aero-maps/ not present at {AERO_DIR}")
    return AERO_DIR
