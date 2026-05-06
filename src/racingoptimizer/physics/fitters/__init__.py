"""Fitter family implementations: GP / RF / Ridge.

* `GPFitter` — continuous low-dim (per-(car, track) v3 heave / wing /
  spring / pressure cohort).
* `ForestFitter` — high-dim mixed-scale (per-car v4 joint vector,
  damper-bound vector with discrete clicks).
* `RidgeFitter` — deterministic setup-readout channels (static RH +
  aero-calc readouts that iRacing emits as a function of garage
  parameters alone, with no track / archetype / env dependency).
"""
from __future__ import annotations

from racingoptimizer.physics.fitters.base import FitterBase
from racingoptimizer.physics.fitters.forest import ForestFitter
from racingoptimizer.physics.fitters.gp import GPFitter
from racingoptimizer.physics.fitters.ridge import RidgeFitter

# Bump when the fitters package layout changes in a way that breaks
# pickle revival (class added/removed, module path renamed, etc.). The
# per-car model cache key folds this so a layout change invalidates
# pre-existing pickles instead of leaving them with a valid digest that
# no longer loads (e.g. `ModuleNotFoundError:
# racingoptimizer.physics.fitters.ridge` after a rename).
FITTERS_LAYOUT_VERSION: int = 2

__all__ = [
    "FITTERS_LAYOUT_VERSION",
    "FitterBase",
    "ForestFitter",
    "GPFitter",
    "RidgeFitter",
]
