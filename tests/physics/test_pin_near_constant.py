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
    _apply_within_track_bounds,
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


def test_no_pin_when_empirical_range_dominates_wide_constraint() -> None:
    """Wide constraint envelope must not mask real corpus variation.

    Regression: BMW heave was pinned at 50 N/mm despite the corpus carrying
    values 30/40/50/60/80 across 9 Sebring sessions, because the constraint
    span had been widened to the full BMWBounds.md legal envelope (0..900
    N/mm) and `observed_std=11.48 / span=900 = 1.27%` fell below the 2%
    threshold. The denominator should be the empirical training range
    (max-min observed = 50 N/mm) so meaningful variation in a wide
    envelope is preserved: `11.48 / 50 = 23%` → not pinned.
    """
    bound = (0.0, 900.0)
    baseline = 50.0
    observed_std = 11.48  # BMW heave global stddev across all sessions
    empirical_range = 50.0  # 80 - 30 across BMW Sebring corpus

    _, was_pinned_with_range = _pin_or_trust_bounds(
        bound=bound, baseline=baseline, regime="confident",
        observed_std=observed_std,
        empirical_range=empirical_range,
    )
    assert was_pinned_with_range is False, (
        "empirical range 50 + std 11.48 = 23% spread; must not pin"
    )

    # Without empirical_range (legacy path / v3 model) the wide constraint
    # span swallows the same std and pins — documents the regression we fixed.
    _, was_pinned_legacy = _pin_or_trust_bounds(
        bound=bound, baseline=baseline, regime="confident",
        observed_std=observed_std,
    )
    assert was_pinned_legacy is True, (
        "without empirical_range the wide constraint span pins (legacy "
        "behaviour we've fixed for per-car v4 models)"
    )


def test_pin_when_empirical_range_is_zero_falls_back_to_span() -> None:
    """Truly constant params (single observed value) must still pin via span fallback."""
    bound = (0.0, 11.0)  # damper click range
    baseline = 11.0
    # Driver only ever ran one click value — std=0, range=0.
    _, was_pinned = _pin_or_trust_bounds(
        bound=bound, baseline=baseline, regime="confident",
        observed_std=0.0, empirical_range=0.0,
    )
    assert was_pinned is True


def test_user_pin_outside_target_observed_does_not_invert() -> None:
    """User constraint-pin must override an out-of-window empirical envelope.

    Regression: ``--fuel 8`` collapses the fuel constraint to ``(8, 8)``,
    but the BMW Spa training corpus only has 58 L observed. The old logic
    built ``empirical_lo = max(8, 58) = 58`` and ``empirical_hi =
    min(8, 58) = 8``, then expanded the degenerate single value by a
    click, producing ``(57, 8)`` — lo > hi — which DE's seed_population
    rejected with ``ValueError: high - low < 0`` and crashed the entire
    recommend run.

    The fix clips ``target_observed`` to the constraint envelope FIRST so
    out-of-window observations are dropped instead of inverting the bound.
    """
    bound = (8.0, 8.0)  # constraint-pinned to 8 L by --fuel 8
    target_observed = (58.0,)  # BMW Spa corpus only has 58 L
    sub_bounds, was_pinned = _pin_or_trust_bounds(
        bound=bound, baseline=8.0, regime="confident",
        observed_std=0.0, target_observed=target_observed,
        click_step=1.0, empirical_range=0.0,
    )
    lo, hi = sub_bounds
    assert lo <= hi, (
        f"DE requires lo<=hi but got ({lo}, {hi}); "
        "constraint pin + out-of-window empirical envelope inverted bound"
    )
    # Should land inside the constraint window (i.e. 8 L) regardless of
    # which branch resolved it — the user pinned it for a reason.
    assert 8.0 - 1e-6 <= lo <= hi <= 8.0 + 1e-6


def test_out_of_bound_baseline_does_not_invert() -> None:
    """Bayes anchor outside the constraint bound must not collapse to inverted DE bounds.

    Regression: ``_bayes_trust_anchor`` returns the unclamped empirical Bayes
    posterior mean, which can be far outside the constraint bound when
    constraints.md is wrong (or the driver's setup drifts outside our coded
    legal envelope). The PIN branch then built
    ``pinned_lo = max(lo, baseline - eps) = baseline - eps`` and
    ``pinned_hi = min(hi, baseline + eps) = hi``, producing
    e.g. baseline=10.0 against bound (1.0, 5.0) -> (9.999996, 5.0) — inverted.
    The fix clamps ``baseline`` to ``[lo, hi]`` at function entry so every
    downstream branch sees a valid baseline.
    """
    bound = (1.0, 5.0)  # Porsche anti_roll_bar_rear pre-override default
    out_of_bound_baseline = 10.0  # actual observed Porsche value
    sub_bounds, was_pinned = _pin_or_trust_bounds(
        bound=bound, baseline=out_of_bound_baseline, regime="sparse",
        observed_std=0.0, target_observed=(10.0,),
        click_step=0.04, empirical_range=0.0,
    )
    lo, hi = sub_bounds
    assert lo <= hi, (
        f"DE requires lo<=hi but got ({lo}, {hi}); "
        "out-of-bound baseline produced inverted window"
    )
    # Result must sit inside the legal bound.
    assert bound[0] <= lo <= hi <= bound[1]


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


# --------------------------------------------------------------------------
# `--reset` mode: 30% widening around corpus envelope, skip pin check
# --------------------------------------------------------------------------


def test_reset_widens_envelope_around_corpus() -> None:
    """Reset opens the bound to corpus-envelope +/- 30% of constraint span
    on each side, clipped to the legal bound."""
    bound = (0.0, 200.0)  # wide constraint
    span = bound[1] - bound[0]
    sub_bounds, was_pinned = _pin_or_trust_bounds(
        bound=bound,
        baseline=60.0,
        regime="dense",
        observed_std=5.0,
        target_observed=(50.0, 60.0, 70.0),
        empirical_range=20.0,
        reset_mode=True,
    )
    assert was_pinned is False
    lo, hi = sub_bounds
    # Expected: [50 - 0.3*200, 70 + 0.3*200] = [-10, 130], clipped to [0, 130].
    assert lo == 0.0
    assert hi == 70.0 + 0.3 * span


def test_reset_skips_pin_check_for_constant_observation() -> None:
    """Reset must allow a parameter the driver always ran at one value
    to move (skips the std-based pin check)."""
    bound = (0.0, 200.0)
    sub_bounds, was_pinned = _pin_or_trust_bounds(
        bound=bound,
        baseline=60.0,
        regime="confident",
        observed_std=0.0,  # would normally pin
        target_observed=(60.0,),
        empirical_range=0.0,
        reset_mode=True,
    )
    assert was_pinned is False
    lo, hi = sub_bounds
    # Search window is wider than the legacy pin would allow.
    assert (hi - lo) > 1.0


def test_reset_clipped_to_legal_bounds() -> None:
    """Widening past the legal constraint envelope clamps to the bound."""
    bound = (10.0, 50.0)
    sub_bounds, was_pinned = _pin_or_trust_bounds(
        bound=bound,
        baseline=30.0,
        regime="dense",
        observed_std=2.0,
        target_observed=(20.0, 30.0, 40.0),
        empirical_range=20.0,
        reset_mode=True,
    )
    assert was_pinned is False
    lo, hi = sub_bounds
    assert lo == 10.0  # 20 - 0.3*40 = 8 -> clipped to 10
    assert hi == 50.0  # 40 + 0.3*40 = 52 -> clipped to 50


# --------------------------------------------------------------------------
# `--explore N` widening
# --------------------------------------------------------------------------


def test_explore_widens_empirical_envelope_by_pct_each_side() -> None:
    """`--explore 10` widens the corpus envelope by 10% of constraint span
    on each side."""
    bound = (0.0, 200.0)
    sub_bounds, was_pinned = _pin_or_trust_bounds(
        bound=bound,
        baseline=60.0,
        regime="dense",
        observed_std=5.0,
        target_observed=(50.0, 60.0, 70.0),
        empirical_range=20.0,
        explore_pct=10.0,
    )
    assert was_pinned is False
    lo, hi = sub_bounds
    # Expected envelope: [50 - 0.1*200, 70 + 0.1*200] = [30, 90].
    # The trust radius for 'dense' is the full constraint, so the
    # intersection is the empirical-envelope clip itself.
    assert lo == 30.0
    assert hi == 90.0


def test_explore_clipped_to_constraint_bounds() -> None:
    """Aggressive `--explore N` clamps to the legal envelope."""
    bound = (10.0, 50.0)
    sub_bounds, was_pinned = _pin_or_trust_bounds(
        bound=bound,
        baseline=30.0,
        regime="dense",
        observed_std=2.0,
        target_observed=(20.0, 30.0, 40.0),
        empirical_range=20.0,
        explore_pct=80.0,  # 80% of 40 = 32 -> would push past both edges
    )
    assert was_pinned is False
    lo, hi = sub_bounds
    assert lo == bound[0]
    assert hi == bound[1]


def test_explore_zero_matches_strict_empirical() -> None:
    """`--explore 0` (default) is strict empirical -- no widening."""
    bound = (0.0, 200.0)
    sub_bounds_no, _ = _pin_or_trust_bounds(
        bound=bound,
        baseline=60.0,
        regime="dense",
        observed_std=5.0,
        target_observed=(50.0, 60.0, 70.0),
        empirical_range=20.0,
        explore_pct=0.0,
    )
    lo, hi = sub_bounds_no
    assert lo == 50.0
    assert hi == 70.0


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
    from racingoptimizer.physics.corner_schedule import build_corner_schedule
    from racingoptimizer.physics.recommend import recommend
    from racingoptimizer.ingest.api import sessions as ingest_sessions

    model, track, corpus_root = bmw_model_session
    # Force every parameter into the pinned regime by overriding the field.
    object.__setattr__(model, "parameter_observed_std", {
        name: 0.0 for name in model.baseline_setup
    })

    constraints = load_constraints()
    env = EnvironmentFrame(
        air_density=1.18, track_temp_c=24.0, wind_vel_ms=2.5,
        wind_dir_deg=120.0, track_wetness=0.0,
    )
    sess_df = ingest_sessions(corpus_root=corpus_root)
    session_ids = sess_df["session_id"].to_list()
    schedule = build_corner_schedule(session_ids, corpus_root=corpus_root)
    rec = recommend(model, track, env, constraints, schedule=schedule)

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


def test_apply_within_track_bounds_pins_single_value() -> None:
    bounds, pinned, thin = _apply_within_track_bounds(
        sub_bounds=(100.0, 200.0),
        was_pinned=False,
        track_observed=(10.0,),
        bound=(0.0, 300.0),
        reset_mode=False,
    )
    assert bounds == (10.0, 10.0)
    assert pinned is True
    assert thin is True


def test_apply_within_track_bounds_caps_to_two_local_values() -> None:
    bounds, pinned, thin = _apply_within_track_bounds(
        sub_bounds=(50.0, 250.0),
        was_pinned=False,
        track_observed=(160.0, 180.0),
        bound=(0.0, 900.0),
        reset_mode=False,
    )
    assert bounds == (160.0, 180.0)
    assert pinned is False
    assert thin is False


def test_apply_within_track_bounds_pins_faster_of_two_local_values() -> None:
    bounds, pinned, thin = _apply_within_track_bounds(
        sub_bounds=(6.0, 12.0),
        was_pinned=False,
        track_observed=(8.0, 10.0),
        bound=(6.0, 10.0),
        reset_mode=False,
        track_best_value=10.0,
    )
    assert bounds == (10.0, 10.0)
    assert pinned is True
    assert thin is True


def test_apply_within_track_bounds_skips_when_three_or_more() -> None:
    bounds, pinned, thin = _apply_within_track_bounds(
        sub_bounds=(50.0, 250.0),
        was_pinned=False,
        track_observed=(160.0, 180.0, 210.0),
        bound=(0.0, 900.0),
        reset_mode=False,
    )
    assert bounds == (50.0, 250.0)
    assert thin is False
