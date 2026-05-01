"""`_pin_or_trust_bounds` — pin search to baseline when training holds a parameter near-constant.

VISION §3 forbids extrapolating beyond observed data. When every session in
the training corpus ran the same value for a parameter (e.g. tyre cold P at
152 kPa across 16 Cadillac sessions), the joint surrogate has no signal
about the response surface and the DE search drifts to whichever bound the
noise gradient points at — producing absurd constraint-edge recommendations.
The recommender now detects this via per-parameter observed std and pins
the search window around the observed median.
"""
from __future__ import annotations

from racingoptimizer.physics.recommend import (
    _NEAR_CONSTANT_FRACTION,
    _pin_or_trust_bounds,
    _trust_bounds,
)


def test_pin_when_observed_std_is_zero() -> None:
    """Single-session corpus → std=0 → pin to baseline."""
    bound = (138.0, 166.0)  # tyre cold P range
    baseline = 152.0
    sub_bounds, was_pinned = _pin_or_trust_bounds(
        bound=bound, baseline=baseline, regime="noisy", observed_std=0.0,
    )
    assert was_pinned is True
    lo, hi = sub_bounds
    assert lo <= baseline <= hi
    assert (hi - lo) < 1e-3, "pinned window must be effectively zero-width"


def test_pin_when_observed_std_below_threshold() -> None:
    """Spread of 0.5 kPa over a 28 kPa range = 1.8% — below the 2% floor."""
    bound = (138.0, 166.0)
    baseline = 152.0
    span = bound[1] - bound[0]
    near_constant_std = span * (_NEAR_CONSTANT_FRACTION * 0.9)
    _, was_pinned = _pin_or_trust_bounds(
        bound=bound, baseline=baseline, regime="noisy",
        observed_std=near_constant_std,
    )
    assert was_pinned is True


def test_no_pin_when_observed_std_above_threshold() -> None:
    """Real per-session variation → fall through to trust-radius logic."""
    bound = (138.0, 166.0)
    baseline = 152.0
    span = bound[1] - bound[0]
    real_variation_std = span * (_NEAR_CONSTANT_FRACTION * 5.0)
    sub_bounds, was_pinned = _pin_or_trust_bounds(
        bound=bound, baseline=baseline, regime="noisy",
        observed_std=real_variation_std,
    )
    assert was_pinned is False
    # Should match the existing trust-radius behaviour for "noisy" (50%).
    assert sub_bounds == _trust_bounds(bound, baseline, "noisy")


def test_pin_respects_constraint_bounds_at_edge() -> None:
    """A baseline at the constraint edge still produces a valid lo<hi window."""
    bound = (138.0, 166.0)
    baseline = 166.0  # right at the upper bound
    sub_bounds, was_pinned = _pin_or_trust_bounds(
        bound=bound, baseline=baseline, regime="sparse", observed_std=0.0,
    )
    assert was_pinned is True
    lo, hi = sub_bounds
    assert lo < hi, "scipy.optimize.differential_evolution requires lo < hi"
    assert lo >= bound[0]
    assert hi <= bound[1]


def test_pin_with_zero_span_bound_is_safe() -> None:
    """Degenerate constraint (lo == hi) should not crash."""
    bound = (10.0, 10.0)
    sub_bounds, was_pinned = _pin_or_trust_bounds(
        bound=bound, baseline=10.0, regime="confident", observed_std=0.0,
    )
    # Span is zero so the std-fraction check short-circuits to "no pin"
    # (we only pin when there's room to learn). Falls through to trust_bounds.
    assert was_pinned is False
    assert sub_bounds == bound


def test_full_recommend_pins_near_constant_param(bmw_model_session) -> None:
    """End-to-end: a model whose `parameter_observed_std` says everything is
    constant must produce a recommendation whose values stay at the
    legal-clamped baseline (no DE drift to a constraint extreme on noise).

    Note: when the observed median is OUTSIDE the legal constraint range
    (e.g. user ran tyre P at 152 kPa but the legal min is 165 kPa), the
    recommender first clamps the baseline into the legal range, then pins
    around that clamped value. That is the correct VISION behaviour —
    illegal setups get clipped to the closest legal value, not exposed.
    """
    from racingoptimizer.constraints import load_constraints
    from racingoptimizer.context import EnvironmentFrame
    from racingoptimizer.physics.recommend import recommend

    model, track, _ = bmw_model_session
    # Force every parameter into the pinned regime by overriding the field.
    object.__setattr__(model, "parameter_observed_std", {
        name: 0.0 for name in model.baseline_setup
    })

    constraints = load_constraints()
    env = EnvironmentFrame(
        air_density=1.18, track_temp_c=24.0, wind_vel_ms=2.5,
        wind_dir_deg=120.0, track_wetness=0.0,
    )
    rec = recommend(model, track, env, constraints)

    # Every fittable parameter that ended up in the recommendation must be
    # within the pinning tolerance of the LEGAL-CLAMPED baseline (i.e. the
    # observed median, then clipped to the constraint range).
    pin_tolerance_per_param: dict[str, float] = {}
    for name in rec.parameters:
        bound = constraints.bounds(model.car, name)
        if bound is None:
            continue
        # Match `_PIN_HALF_WIDTH_FRACTION` * span * 2 (full pin window) plus
        # a small slack for clamp rounding.
        pin_tolerance_per_param[name] = max(
            (bound[1] - bound[0]) * 1e-5, 1e-3,
        )

    for name in rec.pinned_to_observed_median:
        assert name in rec.parameters, (
            f"pinned-to-observed-median lists {name} but it's missing from parameters"
        )
        bound = constraints.bounds(model.car, name)
        observed_baseline = model.baseline_setup.get(name)
        if bound is None or observed_baseline is None:
            continue
        # Recommender clamps baseline into the legal range before pinning.
        legal_baseline = min(max(observed_baseline, bound[0]), bound[1])
        value = rec.parameters[name][0]
        assert abs(value - legal_baseline) < pin_tolerance_per_param[name], (
            f"{name} drifted from legal-clamped baseline {legal_baseline} "
            f"to {value} despite pinning (observed median: {observed_baseline}, "
            f"bounds: {bound})"
        )

    # And the recommendation must mention which parameters were pinned.
    assert len(rec.pinned_to_observed_median) > 0
