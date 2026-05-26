"""Track-independence guards for the v4 predict path.

Regression tests for ``docs/accuracy-rebuild-2026-05-24/PLAN.md`` P0.1:
``per_track_residuals`` was an additive ``track_median - global_median``
correction added to every prediction at that track. It double-counted
track bias (the surrogate is already trained on those rows) and
flattened the setup -> output gradient that DE needs. The proximate
cause of ``+1 click +/-0.000 score`` lines in recent briefings.

The slot is preserved on ``PhysicsModel`` for pickle compat; the
prediction path must no longer read it.
"""
from __future__ import annotations

import inspect

from racingoptimizer.physics import fitter as _fitter_module
from racingoptimizer.physics import model as _model_module


def test_predict_v4_does_not_consume_per_track_residuals() -> None:
    """``_predict_v4`` must not add ``per_track_residuals`` to its output.

    Source-inspection guard: the prior implementation read
    ``getattr(self, "per_track_residuals", {})`` inside the per-channel
    loop and added the entry to ``mean_value``. Reintroducing that
    addition re-introduces the double-counted track bias.

    P2.2 note: ``_predict_v4`` IS allowed to mutate ``mean_value``
    post-fitter via ``track_random_intercepts`` (closed-form Bayes
    random intercepts on residuals with empirical-Bayes shrinkage).
    That correction is mathematically distinct from the retired
    ``per_track_residuals`` (raw observation bias) and is the
    intended P2.2 replacement. This test only guards against
    re-introduction of the *banned* ``per_track_residuals`` pattern.
    """
    src = inspect.getsource(_model_module.PhysicsModel._predict_v4)
    assert "per_track_residuals" in src, (
        "expected the doc comment naming the retired field to stay"
    )
    # No active read of the retired field beyond the doc comment.
    bad_patterns = (
        'getattr(self, "per_track_residuals"',
        "self.per_track_residuals",
        "+= float(track_res",
        "mean_value += float(track_res",
    )
    for pat in bad_patterns:
        assert pat not in src, (
            f"_predict_v4 reintroduced banned pattern {pat!r} -- "
            "this re-enables the P0.1 bug (track-bias double-count)."
        )


def test_fit_per_car_does_not_compute_per_track_residuals() -> None:
    """``fit_per_car`` must ship an empty ``per_track_residuals`` slot.

    Companion guard to the predict-side test: even if a future change
    re-introduces consumption, this guards the producer side.
    """
    src = inspect.getsource(_fitter_module.fit_per_car)
    # The variable name itself remains (slot preservation), but the
    # computation block must not exist.
    banned = (
        "global_medians: dict[str, float] = {}",
        "track_res[ch] = track_med - global_medians[ch]",
        "per_track_residuals[track] = track_res",
    )
    for pat in banned:
        assert pat not in src, (
            f"fit_per_car reintroduced banned pattern {pat!r} -- "
            "see docs/accuracy-rebuild-2026-05-24/PLAN.md P0.1."
        )


def test_fitters_layout_version_bumped_past_residuals_era() -> None:
    """Cache key must invalidate every pre-2026-05-24 pickle.

    Pre-v9 pickles will revive with the (broken) residuals dict intact
    even though predict() no longer reads it -- by bumping the cache
    key we force a refit so new models ship without the data.
    """
    from racingoptimizer.physics.fitters import FITTERS_LAYOUT_VERSION
    assert FITTERS_LAYOUT_VERSION >= 9, (
        "FITTERS_LAYOUT_VERSION must be >=9 so pre-P0.1 pickles are "
        "invalidated and refit without per_track_residuals data."
    )
