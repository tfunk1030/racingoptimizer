"""Fitter family implementations: GP for continuous low-dim, RF otherwise."""
from __future__ import annotations

from racingoptimizer.physics.fitters.base import FitterBase
from racingoptimizer.physics.fitters.forest import ForestFitter
from racingoptimizer.physics.fitters.gp import GPFitter

__all__ = ["FitterBase", "ForestFitter", "GPFitter"]
