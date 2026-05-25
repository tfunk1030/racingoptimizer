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
# pickle revival (class added/removed, module path renamed, etc.) OR
# when PhysicsModel itself gains a non-trivial field that production
# code paths read (forces a refit so the field is populated rather than
# default-empty). The per-car model cache key folds this so a layout
# change invalidates pre-existing pickles instead of leaving them with
# a valid digest that no longer loads (e.g. `ModuleNotFoundError:
# racingoptimizer.physics.fitters.ridge` after a rename) OR with the
# new field empty (e.g. `bayes_posteriors` added by physics-rebuild
# Day 4 -- pre-Day-4 pickles default-revive with `bayes_posteriors={}`,
# which would defeat Mode 1 closure on cached models).
#
# Version history:
#   1 -- pre-Stage-3 layout
#   2 -- Stage-3 joint multi-input model
#   3 -- physics-rebuild Day 4: PhysicsModel.bayes_posteriors field
#   4 -- post-rebuild hybrid-DE wiring: PhysicsModel.axle_grip_ceilings
#        field, consumed by physics/recommend.py guardrail penalty in
#        the DE objective. Pickles from v3 will be revived with
#        ceilings=None (no-op) but the cache key change forces a
#        refit on next recommend so ceilings populate.
#   5 -- FitRecord.bootstrap_std for forest bootstrap-CI brackets;
#        corner_compression_demand_mms archetype (schema v5).
#   6 -- PhysicsModel.aero_residual_correction field, consumed by
#        physics/score.py evaluator path in DE and guardrail warnings.
#   7 -- PhysicsModel.__setstate__ slot-shift repair for pre-v7 pickles
#        (per_track_residuals / axle_grip_ceilings / aero correction
#        mis-alignment) + within-track lap-time anchor for N=2 params.
#   8 -- (interim layout, not durably distinguished from 7)
#   9 -- 2026-05-24 P0.1: per_track_residuals retired. The previous
#        layer added ``track_median - global_median`` to every
#        prediction at that track, double-counting track bias and
#        flattening setup gradient in DE. Pickles from <=8 will revive
#        with the (broken) residuals dict intact; bumping the cache key
#        forces a refit so new models ship with an empty
#        per_track_residuals slot and predict() no longer reads it.
#  10 -- 2026-05-24 P0.2: PhysicsModel.static_rh_kinematic field added.
#        Per-car deterministic linear fit replaces the Ridge/Forest
#        surrogate for the four ``setup_static_*_ride_height_mm``
#        channels when it ships (R^2 >= 0.98). Pickles from <=9 revive
#        with static_rh_kinematic=None (predict_setup_readouts falls
#        through to the surrogate); bumping the cache key forces a
#        refit so new models carry the kinematic fit.
FITTERS_LAYOUT_VERSION: int = 10

__all__ = [
    "FITTERS_LAYOUT_VERSION",
    "FitterBase",
    "ForestFitter",
    "GPFitter",
    "RidgeFitter",
]
