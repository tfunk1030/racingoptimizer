from __future__ import annotations

import importlib.util
from pathlib import Path

from racingoptimizer.corner.phase import Phase

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "holdout_accuracy_gate.py"

_SPEC = importlib.util.spec_from_file_location("holdout_accuracy_gate", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_corner_phase_key_parser_accepts_current_row_shape() -> None:
    row = {
        "session_id": "held-sid",
        "lap_index": 4,
        "corner_id": 7,
        "phase": "mid_corner",
    }
    key = _MODULE._corner_phase_key_from_row(row, fallback_session_id="fallback")
    assert key is not None
    assert key.session_id == "held-sid"
    assert key.lap_index == 4
    assert key.corner_id == 7
    assert key.phase is Phase.MID_CORNER


def test_corner_phase_key_parser_accepts_legacy_row_shape() -> None:
    row = {
        "source_session_id": "legacy-sid",
        "lap": 2,
        "corner_id": 3,
        "phase": "exit",
    }
    key = _MODULE._corner_phase_key_from_row(row, fallback_session_id="fallback")
    assert key is not None
    assert key.session_id == "legacy-sid"
    assert key.lap_index == 2
    assert key.corner_id == 3
    assert key.phase is Phase.EXIT


def test_corner_phase_key_parser_uses_fallback_session_id() -> None:
    row = {"lap_index": 1, "corner_id": 9, "phase": "braking"}
    key = _MODULE._corner_phase_key_from_row(row, fallback_session_id="heldout")
    assert key is not None
    assert key.session_id == "heldout"
    assert key.lap_index == 1
    assert key.corner_id == 9
    assert key.phase is Phase.BRAKING


def test_corner_phase_key_parser_rejects_invalid_phase() -> None:
    row = {"session_id": "sid", "lap_index": 0, "corner_id": 1, "phase": "foo"}
    key = _MODULE._corner_phase_key_from_row(row, fallback_session_id="heldout")
    assert key is None


def test_gate_pass_checks_thresholds() -> None:
    passing = {
        "car": "bmw",
        "channels": [
            {"coverage": 0.90, "normed_residual": 0.8, "regime": "dense"},
            {"coverage": 0.88, "normed_residual": 1.2, "regime": "dense"},
            {"coverage": 0.55, "normed_residual": 1.5, "regime": "sparse"},
        ],
    }
    ok, why = _MODULE._gate_pass(passing)
    assert ok
    assert why == "ok"

    failing = {
        "car": "bmw",
        "channels": [
            {"coverage": 0.45, "normed_residual": 2.5, "regime": "dense"},
            {"coverage": 0.50, "normed_residual": 2.4, "regime": "dense"},
            {"coverage": 0.48, "normed_residual": 2.3, "regime": "sparse"},
        ],
    }
    ok, why = _MODULE._gate_pass(failing)
    assert not ok
    assert "median_cov" in why
    assert "median_normed" in why
    assert "dense_mean_cov" in why
