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
