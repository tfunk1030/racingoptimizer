"""P1.2 -- per-(car, track) lap-time Spearman gate helper coverage.

The orchestration layer in ``scripts/lap_time_correlation_gate.py``
walks the catalog and runs LOSO refits per session, which is too
heavy to exercise in CI. These tests cover the standalone helpers:
the rank-correlation math, the qualifying-pair filter, and the
per-pair pass/fail logic. Run synthetic ranking data through them
to verify the gate's pass/fail semantics.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_gate_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "lap_time_correlation_gate.py"
    spec = importlib.util.spec_from_file_location(
        "_lap_time_gate_under_test", script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate_mod():
    return _load_gate_module()


def test_rankdata_matches_average_ranking(gate_mod) -> None:
    ranks = gate_mod._rankdata([10.0, 20.0, 30.0])
    assert ranks == [1.0, 2.0, 3.0]


def test_rankdata_handles_ties(gate_mod) -> None:
    ranks = gate_mod._rankdata([10.0, 10.0, 30.0])
    # Two tied at the bottom: average rank = (1+2)/2 = 1.5
    assert ranks == [1.5, 1.5, 3.0]


def test_spearman_perfect_positive(gate_mod) -> None:
    pairs = [(i, 2.0 * i) for i in range(10)]
    rho = gate_mod._spearman_correlation(pairs)
    assert rho is not None
    assert rho == pytest.approx(1.0, abs=1e-9)


def test_spearman_perfect_negative(gate_mod) -> None:
    pairs = [(i, -2.0 * i) for i in range(10)]
    rho = gate_mod._spearman_correlation(pairs)
    assert rho is not None
    assert rho == pytest.approx(-1.0, abs=1e-9)


def test_spearman_returns_none_under_3_pairs(gate_mod) -> None:
    assert gate_mod._spearman_correlation([(1.0, 2.0), (3.0, 4.0)]) is None


def test_spearman_returns_none_when_either_constant(gate_mod) -> None:
    constant_x = [(5.0, 1.0), (5.0, 2.0), (5.0, 3.0)]
    assert gate_mod._spearman_correlation(constant_x) is None
    constant_y = [(1.0, 5.0), (2.0, 5.0), (3.0, 5.0)]
    assert gate_mod._spearman_correlation(constant_y) is None


def test_spearman_drops_non_finite_pairs(gate_mod) -> None:
    pairs = [
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 3.0),
        (float("nan"), 99.0),
        (float("inf"), 50.0),
    ]
    rho = gate_mod._spearman_correlation(pairs)
    assert rho == pytest.approx(1.0, abs=1e-9)


def test_qualifying_pairs_filters_below_threshold(gate_mod) -> None:
    sessions = {
        ("bmw", "spa"): [f"s{i}" for i in range(15)],   # qualifies
        ("bmw", "monza"): [f"s{i}" for i in range(8)],   # does not
        ("acura", "daytona"): [f"s{i}" for i in range(10)],  # boundary
    }
    qualifying = gate_mod._qualifying_pairs(sessions)
    assert ("bmw", "spa") in qualifying
    assert ("acura", "daytona") in qualifying
    assert ("bmw", "monza") not in qualifying


def test_evaluate_pair_score_passes_above_target(gate_mod) -> None:
    ok, why = gate_mod._evaluate_pair_score(0.42)
    assert ok is True
    assert "rho=0.420" in why


def test_evaluate_pair_score_fails_below_target(gate_mod) -> None:
    ok, why = gate_mod._evaluate_pair_score(0.10)
    assert ok is False
    assert "rho=0.100" in why
    assert "<" in why


def test_evaluate_pair_score_fails_when_correlation_is_none(gate_mod) -> None:
    ok, why = gate_mod._evaluate_pair_score(None)
    assert ok is False
    assert why == "insufficient_data"


def test_main_returns_zero_when_catalog_empty(monkeypatch, gate_mod) -> None:
    """When the catalog can't be reached, the gate is non-blocking."""
    monkeypatch.setattr(
        gate_mod,
        "_build_pair_sessions_from_catalog",
        lambda: {},
    )
    assert gate_mod.main() == 0
