"""P3.1 -- briefing header surfaces per-channel held-out error budget.

The legacy header line ``Confidence: <regime> (median n=N)`` was
internally inconsistent with the per-parameter regime breakdown (see
PLAN.md 2.2). P3.1 replaces it with per-channel ``mean_abs`` numbers
pulled from ``docs/physics-rebuild/holdout_accuracy_latest.json`` for
the matching ``(car, track)`` row, falling back to the legacy line
when no row matches.
"""
from __future__ import annotations

import json
from pathlib import Path

from racingoptimizer.explain.narrative import (
    _HEADER_ERROR_BUDGET_CHANNELS,
    _render_error_budget_block,
)


def _holdout_payload(car: str, track: str, channels: list[dict]) -> list[dict]:
    return [
        {
            "car": car,
            "session_id": "synthetic",
            "track": track,
            "n_prod_sessions": 30,
            "n_holdout_rows": 100,
            "channels": channels,
        }
    ]


def _fake_channels() -> list[dict]:
    return [
        {
            "channel": "accel_lat_g_max",
            "n": 100,
            "mean_abs": 0.42,
            "actual_std": 0.5,
            "normed_residual": 0.84,
            "coverage": 0.85,
            "regime": "noisy",
        },
        {
            "channel": "understeer_angle_mean_rad",
            "n": 100,
            "mean_abs": 0.18,
            "actual_std": 0.4,
            "normed_residual": 0.45,
            "coverage": 0.92,
            "regime": "noisy",
        },
        {
            "channel": "setup_static_lf_ride_height_mm",
            "n": 100,
            "mean_abs": 0.4,
            "actual_std": 5.0,
            "normed_residual": 0.08,
            "coverage": 0.99,
            "regime": "dense",
        },
        {
            "channel": "damper_force_p99_n",
            "n": 100,
            "mean_abs": 400.0,
            "actual_std": 1500.0,
            "normed_residual": 0.27,
            "coverage": 0.95,
            "regime": "noisy",
        },
    ]


def _write_holdout(tmp_path: Path, payload: list[dict]) -> Path:
    docs = tmp_path / "docs" / "physics-rebuild"
    docs.mkdir(parents=True, exist_ok=True)
    target = docs / "holdout_accuracy_latest.json"
    target.write_text(json.dumps(payload, indent=2))
    return target


def test_header_block_renders_when_row_matches(tmp_path, monkeypatch) -> None:
    payload = _holdout_payload("acura", "belleisle", _fake_channels())
    _write_holdout(tmp_path, payload)
    monkeypatch.chdir(tmp_path)
    block = _render_error_budget_block("acura", "belleisle")
    assert block, "expected a non-empty error budget block"
    assert block[0].startswith("Predicted error on this car/track")
    joined = "\n".join(block)
    assert "peak lateral G" in joined
    assert "+/- 0.42 g" in joined
    assert "understeer angle" in joined
    assert "+/- 0.18 rad" in joined
    assert "static front RH" in joined
    assert "+/- 0.40 mm" in joined
    assert "damper force p99" in joined
    assert "+/- 400 N" in joined
    assert "(noisy)" in joined
    assert "(dense)" in joined


def test_header_block_falls_back_when_file_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # no docs/ tree
    block = _render_error_budget_block("acura", "belleisle")
    assert block == []


def test_header_block_falls_back_when_no_matching_row(tmp_path, monkeypatch) -> None:
    payload = _holdout_payload("bmw", "spa_2024_up", _fake_channels())
    _write_holdout(tmp_path, payload)
    monkeypatch.chdir(tmp_path)
    block = _render_error_budget_block("acura", "belleisle")
    assert block == []


def test_header_block_omits_channels_missing_from_gate(tmp_path, monkeypatch) -> None:
    """When the held-out IBT didn't score one of the budget channels
    (e.g. damper force absent), the block omits that line rather than
    rendering ``+/- nan`` or duplicating a header. Only channels that
    exist in the gate JSON appear."""
    partial = [c for c in _fake_channels() if c["channel"] != "damper_force_p99_n"]
    payload = _holdout_payload("acura", "belleisle", partial)
    _write_holdout(tmp_path, payload)
    monkeypatch.chdir(tmp_path)
    block = _render_error_budget_block("acura", "belleisle")
    assert block, "expected a non-empty error budget block"
    joined = "\n".join(block)
    assert "damper force p99" not in joined
    assert "peak lateral G" in joined


def test_header_block_falls_back_when_all_channels_missing(
    tmp_path, monkeypatch,
) -> None:
    """When NONE of the budget channels are scored, the block returns
    empty so the renderer falls back to the legacy ``Confidence: ...``
    line rather than printing only the header."""
    payload = _holdout_payload(
        "acura",
        "belleisle",
        [
            {
                "channel": "throttle_max",
                "n": 100,
                "mean_abs": 0.05,
                "actual_std": 0.1,
                "normed_residual": 0.5,
                "coverage": 0.9,
                "regime": "noisy",
            }
        ],
    )
    _write_holdout(tmp_path, payload)
    monkeypatch.chdir(tmp_path)
    block = _render_error_budget_block("acura", "belleisle")
    assert block == []


def test_header_block_handles_malformed_json(tmp_path, monkeypatch) -> None:
    docs = tmp_path / "docs" / "physics-rebuild"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "holdout_accuracy_latest.json").write_text("not json {")
    monkeypatch.chdir(tmp_path)
    block = _render_error_budget_block("acura", "belleisle")
    assert block == []


def test_header_channels_match_per_channel_thresholds_subset() -> None:
    """The budget's channels must be a subset of the gating dict in
    ``scripts/holdout_accuracy_gate.py`` so the header doesn't reference
    a channel the gate doesn't track."""
    import importlib.util

    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_holdout_gate_for_test",
        repo_root / "scripts" / "holdout_accuracy_gate.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gate_channels = set(mod._PER_CHANNEL_THRESHOLDS)
    header_channels = {ch for ch, _label, _unit in _HEADER_ERROR_BUDGET_CHANNELS}
    assert header_channels.issubset(gate_channels), (
        f"header references channels not in gate: "
        f"{header_channels - gate_channels}"
    )
