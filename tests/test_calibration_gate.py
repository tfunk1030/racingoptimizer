"""Weekly calibration gate (Phase 2 corpus strategy).

Marked ``calibration`` -- run on schedule via CI or locally:
    uv run pytest -q -m calibration
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_CATALOG = REPO_ROOT / "corpus" / "catalog.sqlite"


@pytest.mark.calibration
def test_day_12b_evaluator_calibration_script_runs() -> None:
    """Day 12b gate must pass when a local corpus is present."""
    script = REPO_ROOT / "scripts" / "day_12b_calibrate_evaluator.py"
    if not script.is_file():
        pytest.skip("day_12b_calibrate_evaluator.py missing")
    if not _CATALOG.is_file():
        pytest.skip("no local corpus/catalog.sqlite for calibration gate")
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


@pytest.mark.calibration
def test_holdout_accuracy_gate_script_runs() -> None:
    """Held-out H1-H5 accuracy gate should execute cleanly when corpus exists."""
    script = REPO_ROOT / "scripts" / "holdout_accuracy_gate.py"
    if not script.is_file():
        pytest.skip("holdout_accuracy_gate.py missing")
    if not _CATALOG.is_file():
        pytest.skip("no local corpus/catalog.sqlite for calibration gate")
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
