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


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--regenerate-golden",
        action="store_true",
        default=False,
        help="Regenerate golden snapshot files instead of comparing to them.",
    )


_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _is_unmaterialised_lfs_pointer(path: Path) -> bool:
    """The IBT corpus is tracked in git-lfs.

    On checkouts where `git lfs pull` hasn't been run, the file at the
    fixture path is a ~130-byte pointer text instead of the actual binary.
    Feeding that pointer through the IRSDK parser causes runaway memory
    allocation (the pointer text gets misinterpreted as a multi-GB
    `session_info_len`). Detect the pointer header and skip cleanly.
    """
    try:
        if path.stat().st_size > 4096:
            return False
        with path.open("rb") as fh:
            return fh.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def _resolve_real_ibt_or_skip(candidate: Path, label: str) -> Path:
    if not candidate.exists():
        pytest.skip(f"{label} not present at {candidate}")
    if _is_unmaterialised_lfs_pointer(candidate):
        pytest.skip(
            f"{label} at {candidate.name} is an unmaterialised git-lfs pointer; "
            "run `git lfs pull` before invoking IBT-loading tests"
        )
    return candidate


@pytest.fixture
def small_ibt() -> Path:
    """Path to a small real .ibt fixture from the repo's ibtfiles/ corpus."""
    return _resolve_real_ibt_or_skip(IBT_DIR / SMALL_IBT_NAME, "fixture IBT")


@pytest.fixture
def multi_lap_ibt() -> Path:
    """Path to a multi-lap real .ibt fixture (BMW Sebring, ≥3 valid laps)."""
    return _resolve_real_ibt_or_skip(
        IBT_DIR / MULTI_LAP_IBT_NAME, "multi-lap fixture",
    )


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
