"""Machine-readable JSON renderers (spec §4 layout).

Snake-case keys; `sort_keys=False`; floats unrounded. Mirrors the dataclass
fields one-to-one so downstream consumers can deserialise without a separate
schema.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from racingoptimizer.explain.comparison import SetupComparison
from racingoptimizer.explain.justification import (
    SetupJustification,
)
from racingoptimizer.explain.status import ModelStatus

if TYPE_CHECKING:
    from racingoptimizer.physics import PhysicsModel, SetupRecommendation


_REGIME_RANK: dict[str, int] = {"sparse": 0, "noisy": 1, "confident": 2, "dense": 3}


def render_recommendation_json(
    rec: SetupRecommendation,
    model: PhysicsModel,
    *,
    justifications: list[SetupJustification] | None = None,
    pinned: dict[str, float] | None = None,
    warnings: list[str] | None = None,
    track_display: str | None = None,
) -> dict[str, Any]:
    if justifications is None:
        from racingoptimizer.explain.justification import build_justifications
        justifications = build_justifications(rec, model, pinned=pinned)
    pinned = pinned or {}
    warnings = list(warnings or [])
    if rec.untrained_parameters:
        warnings.append(
            f"untrained_parameters: {', '.join(rec.untrained_parameters)}"
        )
    rolled = _roll_up_confidence(justifications)
    return {
        "car": rec.car,
        "track": rec.track,
        "track_display": track_display or rec.track,
        "environment": _env_to_json(rec.env),
        "confidence": rolled,
        "pinned": dict(pinned),
        "parameters": [_justification_to_json(j) for j in justifications],
        "untrained_parameters": list(rec.untrained_parameters),
        "aero_correction_available": bool(rec.aero_correction_available),
        "warnings": warnings,
    }


def render_comparison_json(cmp: SetupComparison) -> dict[str, Any]:
    return {
        "car": cmp.car,
        "track": cmp.track,
        "setup_a_id": cmp.setup_a_id,
        "setup_b_id": cmp.setup_b_id,
        "total_score_a": float(cmp.total_score_a),
        "total_score_b": float(cmp.total_score_b),
        "total_delta": float(cmp.total_score_b - cmp.total_score_a),
        "per_corner_phase": [
            {
                "corner_id": int(d.corner_id),
                "phase": d.phase.value,
                "score_a": float(d.score_a),
                "score_b": float(d.score_b),
                "delta": float(d.delta),
                "drivers": list(d.drivers),
            }
            for d in cmp.per_corner_phase
        ],
        "notes": list(cmp.notes),
    }


def render_status_json(status: ModelStatus) -> dict[str, Any]:
    return {
        "car": status.car,
        "overall_regime": status.overall_regime,
        "coverage": [
            {
                "track": cov.track,
                "n_sessions": int(cov.n_sessions),
                "n_valid_laps": int(cov.n_valid_laps),
                "n_clean_corner_phases": int(cov.n_clean_corner_phases),
                "fit_quality": (
                    float(cov.fit_quality) if cov.fit_quality is not None else None
                ),
                "regime": cov.regime,
            }
            for cov in status.coverage
        ],
        "notes": list(status.notes),
    }


# ---- internals -----------------------------------------------------------


def _env_to_json(env) -> dict[str, float]:
    return {
        "air_density_kg_m3": float(env.air_density),
        "track_temp_c": float(env.track_temp_c),
        "wind_vel_ms": float(env.wind_vel_ms),
        "wind_dir_deg": float(env.wind_dir_deg),
        "wetness": float(env.track_wetness),
    }


def _justification_to_json(j: SetupJustification) -> dict[str, Any]:
    return {
        "parameter": j.parameter,
        "value": float(j.value),
        "unit": j.unit,
        "pinned": bool(j.pinned),
        "confidence": {
            "regime": j.confidence.regime,
            "n_samples": int(j.confidence.n_samples),
            "value": float(j.confidence.value),
            "lo": float(j.confidence.lo),
            "hi": float(j.confidence.hi),
        },
        "sensitivity_minus_1_click": float(j.sensitivity_minus_1_click),
        "sensitivity_plus_1_click": float(j.sensitivity_plus_1_click),
        "corners_helped": [_impact_to_json(i) for i in j.corners_helped],
        "corners_hurt": [_impact_to_json(i) for i in j.corners_hurt],
        "telemetry_evidence": list(j.telemetry_evidence),
    }


def _impact_to_json(impact) -> dict[str, Any]:
    return {
        "corner_id": int(impact.corner_id),
        "phase": impact.phase.value,
        "score_delta": float(impact.score_delta),
        "note": impact.note,
    }


def _roll_up_confidence(justifications: list[SetupJustification]) -> dict[str, Any]:
    if not justifications:
        return {"regime": "sparse", "n_samples": 0, "n_parameters": 0}
    worst_rank = min(_REGIME_RANK[j.confidence.regime] for j in justifications)
    rank_to_regime = {v: k for k, v in _REGIME_RANK.items()}
    regime = rank_to_regime[worst_rank]
    samples_for_regime = [
        j.confidence.n_samples for j in justifications
        if j.confidence.regime == regime
    ]
    return {
        "regime": regime,
        "n_samples": int(max(samples_for_regime) if samples_for_regime else 0),
        "n_parameters": len(justifications),
    }


__all__ = [
    "render_comparison_json",
    "render_recommendation_json",
    "render_status_json",
]
