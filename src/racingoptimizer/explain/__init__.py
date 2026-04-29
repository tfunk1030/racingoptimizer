"""Setup justification dataclasses + briefing renderers (slice F).

Public surface:
    SetupJustification, CornerPhaseImpact, IncompleteJustificationError
    SetupComparison, CornerPhaseDelta
    ModelStatus, TrackCoverage
    build_justifications(rec, model) -> list[SetupJustification]
    render_recommendation_text(rec, model, **opts) -> str
    render_recommendation_json(rec, model, **opts) -> dict
    render_comparison_text(cmp) / render_comparison_json(cmp)
    render_status_text(status) / render_status_json(status)
"""
from __future__ import annotations

from racingoptimizer.explain.comparison import CornerPhaseDelta, SetupComparison
from racingoptimizer.explain.justification import (
    CornerPhaseImpact,
    IncompleteJustificationError,
    SetupJustification,
    build_justifications,
)
from racingoptimizer.explain.render_json import (
    render_comparison_json,
    render_recommendation_json,
    render_status_json,
)
from racingoptimizer.explain.render_text import (
    render_comparison_text,
    render_recommendation_text,
    render_status_text,
)
from racingoptimizer.explain.status import ModelStatus, TrackCoverage

__all__ = [
    "CornerPhaseDelta",
    "CornerPhaseImpact",
    "IncompleteJustificationError",
    "ModelStatus",
    "SetupComparison",
    "SetupJustification",
    "TrackCoverage",
    "build_justifications",
    "render_comparison_json",
    "render_comparison_text",
    "render_recommendation_json",
    "render_recommendation_text",
    "render_status_json",
    "render_status_text",
]
