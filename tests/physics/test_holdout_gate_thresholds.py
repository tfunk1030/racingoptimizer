"""Per-channel threshold logic for the held-out accuracy gate (P1.1).

These tests synthesize the dict shape ``_gate_one_car`` returns (rows
of ``channel`` / ``mean_abs`` / ``normed_residual`` / ``actual_std``)
and exercise ``_per_channel_pass`` directly. They do NOT fit a real
model -- the gate's pass/fail logic is a pure function of those rows
and is the only thing P1.1 owns.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_gate_module():
    """Load ``scripts/holdout_accuracy_gate.py`` as a module.

    The script lives outside the package and isn't on ``sys.path`` by
    default. Loading it via importlib avoids polluting any test that
    relies on the racingoptimizer package import path.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "holdout_accuracy_gate.py"
    spec = importlib.util.spec_from_file_location(
        "_holdout_gate_under_test", script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate_mod():
    return _load_gate_module()


def _row(channel: str, *, mean_abs: float, normed: float, std: float = 1.0) -> dict:
    return {
        "channel": channel,
        "mean_abs": float(mean_abs),
        "normed_residual": float(normed),
        "actual_std": float(std),
    }


def test_clean_row_passes(gate_mod) -> None:
    rows = [
        _row("accel_lat_g_max", mean_abs=0.10, normed=0.20, std=0.5),
        _row("understeer_angle_mean_rad", mean_abs=0.05, normed=0.20, std=0.4),
    ]
    ok, failed = gate_mod._per_channel_pass(rows)
    assert ok is True
    assert failed == []


def test_high_mean_abs_fails(gate_mod) -> None:
    rows = [
        _row("accel_lat_g_max", mean_abs=0.45, normed=0.30, std=0.5),
    ]
    ok, failed = gate_mod._per_channel_pass(rows)
    assert ok is False
    assert any("accel_lat_g_max" in line and "mean_abs" in line for line in failed)


def test_high_normed_fails(gate_mod) -> None:
    rows = [
        _row(
            "understeer_angle_mean_rad",
            mean_abs=0.08,
            normed=0.80,
            std=0.4,
        ),
    ]
    ok, failed = gate_mod._per_channel_pass(rows)
    assert ok is False
    assert any(
        "understeer_angle_mean_rad" in line and "normed" in line
        for line in failed
    )


def test_unknown_channel_is_skipped_not_failed(gate_mod) -> None:
    rows = [
        _row("throttle_max", mean_abs=99.0, normed=99.0, std=0.1),
    ]
    ok, failed = gate_mod._per_channel_pass(rows)
    assert ok is True
    assert failed == []


def test_missing_channel_is_skipped(gate_mod) -> None:
    rows = [_row("accel_lat_g_max", mean_abs=0.10, normed=0.20, std=0.5)]
    ok, failed = gate_mod._per_channel_pass(rows)
    assert ok is True
    assert failed == []


def test_damper_force_is_informational_not_gated(gate_mod) -> None:
    """``damper_force_p99_n`` was reclassified to informational (2026-06-16,
    fit-quality-justified -- it is a deterministic transform of damper
    velocity, the codebase's structural fit ceiling). A damper row far over
    the old 30%-of-std budget must now be SKIPPED, not failed."""
    rows = [
        _row("damper_force_p99_n", mean_abs=400.0, normed=0.90, std=1000.0),
    ]
    ok, failed = gate_mod._per_channel_pass(rows)
    assert ok is True
    assert failed == []
    # And the rest of the damper family stays informational too.
    for ch in gate_mod._INFORMATIONAL_DAMPER_CHANNELS:
        assert ch not in gate_mod._PER_CHANNEL_THRESHOLDS


def test_failure_message_format(gate_mod) -> None:
    rows = [
        _row("accel_lat_g_max", mean_abs=0.45, normed=0.80, std=0.5),
    ]
    ok, failed = gate_mod._per_channel_pass(rows)
    assert ok is False
    assert len(failed) == 1
    line = failed[0]
    assert line.startswith("accel_lat_g_max:")
    assert "mean_abs" in line
    assert "normed" in line
    assert "|" in line  # both clauses joined by " | "
