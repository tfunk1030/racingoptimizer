"""Confidence type for physics-state predictions.

Public API:
    Confidence(value, lo, hi, n_samples, regime)
    Confidence.derive(value=..., n_samples=..., cv_residual_std=..., signal_std=...)
"""
from __future__ import annotations

from racingoptimizer.confidence.confidence import Confidence

__all__ = ["Confidence"]
