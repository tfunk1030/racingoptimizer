"""Days 3-5 of the physics-rebuild plan: hierarchical Bayesian retrofit (Mode 1).

Background (PLAN.md Section 14.3):
- Mode 1 evidence: per-car (v4) RandomForest fits pool every track
  and are dragged by sample weight regardless of corner-archetype
  features. CLAUDE.md lines 112-118 documents Hockenheim wing=17
  across 24 sessions vs Spa wing=14-15 across 6 sessions producing
  Spa recommendations that inherit Hockenheim philosophy.
- The retrofit fits a one-way random-intercept Gaussian model per
  parameter; method-of-moments hyperparameters; closed-form per-track
  posterior. Mathematically equivalent to the limit of infinite MCMC
  samples.

Acceptance gate (PLAN.md Section 14.3): for BMW with H1 (Spa) held
out, the Bayesian retrofit's posterior mean prediction on H1 must
beat the current v4 surrogate prediction by >= 5% in MAE on setup-
readout target columns. Plus 95% interval covers >= 80% of
held-out setup readouts. (Day 5 task; the full-corpus held-out gate
runs on Day 5 once the wire-in to fit_per_car lands on Day 4.)

Day 3 scope (this test file): standalone module correctness across
12+ test cases covering the math, the canonical Mode 1 case, and
the broken-model canary.
"""
from __future__ import annotations

import math

import pytest

from racingoptimizer.physics.bayes_retrofit import (
    BayesPosterior,
    fit_all_parameters,
    fit_per_parameter,
)

# ---- correctness on synthetic data ---------------------------------------


def test_single_track_returns_empirical_mean_no_shrinkage() -> None:
    """One track in the data: degraded path returns its own empirical
    mean+std, shrinkage = 0 (nothing to shrink toward)."""
    out = fit_per_parameter({"sebring": [10.0, 11.0, 12.0]})
    assert "sebring" in out
    p = out["sebring"]
    assert p.mean == pytest.approx(11.0, abs=1e-9)
    assert p.std > 0
    assert p.shrinkage == 0.0
    assert p.n_samples == 3


def test_two_tracks_with_identical_means_no_shrinkage_drift() -> None:
    """Two tracks with the same mean: posterior mean equals track mean
    (grand mean equals track means too, so no information lost). Some
    shrinkage may be applied because between-track variance is small."""
    data = {
        "sebring": [10.0, 10.0, 10.0],
        "spa_2024_up": [10.0, 10.0, 10.0],
    }
    out = fit_per_parameter(data)
    for _t, p in out.items():
        assert p.mean == pytest.approx(10.0, abs=1e-6)


def test_high_n_high_separation_keeps_track_specific_mean() -> None:
    """Two tracks with clearly different means and decent sample sizes
    each: the hierarchical fit should NOT pull them toward each other
    much (between-track variance is large)."""
    data = {
        "sebring": [17.0] * 20 + [17.5] * 4,  # n=24, mean ~17.08
        "spa_2024_up": [14.5] * 4 + [14.0, 15.0],  # n=6, mean ~14.5
    }
    out = fit_per_parameter(data)
    sebring = out["sebring"]
    spa = out["spa_2024_up"]
    # Posterior means should stay close to their empirical means.
    assert abs(sebring.mean - 17.08) < 0.2, (
        f"sebring posterior {sebring.mean:.3f} drifted too far from 17.08"
    )
    assert abs(spa.mean - 14.5) < 0.5, (
        f"spa posterior {spa.mean:.3f} drifted too far from 14.5; this "
        f"is the Mode 1 failure case"
    )
    # Spa has fewer samples so its posterior std should be > sebring's.
    assert spa.std > sebring.std


def test_canonical_mode_1_case_protects_minority_track() -> None:
    """The Mode 1 canonical scenario: 24-sample track at one value, 6-
    sample track at a clearly different value. The minority track's
    posterior mean must NOT be pulled to the majority -- this is the
    whole point of the hierarchical retrofit.

    Maps to: BMW Ferrari wing values per CLAUDE.md lines 112-118.
    """
    data = {
        "hockenheim_gp": [17.0] * 24,  # majority, n=24, perfectly constant
        "spa_2024_up": [14.0] * 3 + [15.0] * 3,  # n=6, mean=14.5
    }
    out = fit_per_parameter(data, parameter_name="rear_wing_angle_deg")
    spa = out["spa_2024_up"]

    # The OLD per-car surrogate would predict ~16.5 at Spa (sample-weighted
    # grand mean across both tracks, dragged toward Hockenheim 17). The
    # hierarchical retrofit must produce a posterior closer to 14.5.
    # Specifically: posterior_mean must be closer to Spa's 14.5 than to
    # the grand mean 16.5.
    grand_mean = (24 * 17.0 + 6 * 14.5) / 30
    assert abs(spa.mean - 14.5) < abs(spa.mean - grand_mean), (
        f"Mode 1 NOT closed: Spa posterior {spa.mean:.3f} closer to grand "
        f"mean {grand_mean:.3f} than to Spa's empirical 14.5"
    )

    # Stronger: Spa posterior should be within 1 unit of Spa's empirical
    # 14.5 (well inside the corpus envelope that user actually drove).
    assert abs(spa.mean - 14.5) <= 1.0, (
        f"Spa posterior {spa.mean:.3f} drifted >1.0 from empirical 14.5"
    )


def test_low_n_track_gets_wider_posterior_std() -> None:
    """Posterior std reflects sample-count uncertainty: a 2-sample
    track has wider std than a 100-sample track."""
    data = {
        "track_a": [10.0] * 100,
        "track_b": [11.0, 11.0],
    }
    out = fit_per_parameter(data)
    assert out["track_b"].std > out["track_a"].std


def test_shrinkage_in_unit_interval() -> None:
    """`shrinkage = 1 - lambda_t` is always in [0, 1]."""
    data = {
        "a": [10.0, 11.0, 12.0],
        "b": [20.0, 21.0],
        "c": [30.0],  # singleton
    }
    out = fit_per_parameter(data)
    for p in out.values():
        assert 0.0 <= p.shrinkage <= 1.0


def test_empty_input_returns_empty() -> None:
    out = fit_per_parameter({})
    assert out == {}


def test_filters_zero_observation_tracks() -> None:
    """A track with an empty value list is silently dropped."""
    data = {
        "ok": [10.0, 11.0],
        "empty": [],
    }
    out = fit_per_parameter(data)
    assert "ok" in out
    assert "empty" not in out


def test_min_samples_per_track_drops_singletons_when_set() -> None:
    """`min_samples_per_track=2` lets the caller exclude singleton
    tracks so they don't enter the hierarchical estimation. The
    full-corpus default is 1 (tolerant)."""
    data = {
        "ok": [10.0, 11.0, 12.0],
        "singleton": [99.0],
    }
    out = fit_per_parameter(data, min_samples_per_track=2)
    assert "ok" in out
    assert "singleton" not in out


def test_determinism_same_input_same_output() -> None:
    """Pure-Python statistics module: no RNG, no parallelism.
    Re-running on identical input yields bit-identical posteriors."""
    data = {
        "a": [10.0, 11.0, 12.0, 13.0],
        "b": [20.0, 21.0],
    }
    out1 = fit_per_parameter(data)
    out2 = fit_per_parameter(data)
    for t in out1:
        assert out1[t] == out2[t]


def test_negative_values_handled() -> None:
    """Signed parameters (perch offsets, toe) work."""
    data = {
        "a": [-50.0, -49.0, -51.0],
        "b": [-30.0, -31.0],
    }
    out = fit_per_parameter(data)
    assert out["a"].mean < 0
    assert out["b"].mean < 0


def test_posterior_dataclass_immutable() -> None:
    """Frozen dataclass rejects field reassignment."""
    from dataclasses import FrozenInstanceError

    p = BayesPosterior(
        parameter="x", track="t", mean=1.0, std=0.5,
        n_samples=10, shrinkage=0.3,
    )
    with pytest.raises(FrozenInstanceError):
        p.mean = 2.0  # type: ignore[misc]


# ---- fit_all_parameters wrapper -----------------------------------------


def test_fit_all_parameters_reshapes_correctly() -> None:
    """Wrapper takes track->param->values and returns (param, track) keys."""
    data = {
        "sebring": {
            "wing": [17.0, 17.0, 17.0],
            "heave": [60.0, 65.0, 70.0],
        },
        "spa_2024_up": {
            "wing": [14.0, 15.0],
            "heave": [80.0, 85.0],
        },
    }
    out = fit_all_parameters(data)
    assert ("wing", "sebring") in out
    assert ("wing", "spa_2024_up") in out
    assert ("heave", "sebring") in out
    assert ("heave", "spa_2024_up") in out


def test_fit_all_parameters_carries_parameter_name_in_posterior() -> None:
    data = {
        "sebring": {"wing": [17.0, 17.0]},
        "spa_2024_up": {"wing": [14.0, 14.0]},
    }
    out = fit_all_parameters(data)
    for (param, track), post in out.items():
        assert post.parameter == param
        assert post.track == track


# ---- canary -------------------------------------------------------------


def test_canary_pooled_regression_drifts_minority_to_grand_mean() -> None:
    """Broken-model canary (PLAN.md Section 14.3): replace the
    hierarchical model with simple pooling (just compute the grand
    mean, ignore track structure). Verify that produces a Mode 1
    failure -- minority track is dragged to grand mean.

    This is the inverse-direction proof: the canary path produces
    the bug we're closing, so the hierarchical fix must be doing
    something specific (and not just by accident).
    """
    data = {
        "hockenheim_gp": [17.0] * 24,
        "spa_2024_up": [14.0] * 3 + [15.0] * 3,
    }

    # Pooled (broken) regression: weighted grand mean for both tracks.
    total = 24 * 17.0 + 6 * 14.5
    n = 30
    pooled_mean = total / n  # ~16.5
    spa_pooled_error = abs(pooled_mean - 14.5)

    # Hierarchical fit:
    out = fit_per_parameter(data, parameter_name="wing")
    spa_hierarchical_error = abs(out["spa_2024_up"].mean - 14.5)

    # The hierarchical fit's Spa posterior must be MUCH closer to Spa's
    # empirical 14.5 than the pooled-regression baseline.
    assert spa_hierarchical_error < spa_pooled_error, (
        f"hierarchical not better than pooled: "
        f"hierarchical_err={spa_hierarchical_error:.3f}, "
        f"pooled_err={spa_pooled_error:.3f}"
    )
    # And the improvement must be substantial (>50% reduction in error).
    assert spa_hierarchical_error < spa_pooled_error * 0.5, (
        f"hierarchical only marginally better than pooled "
        f"({spa_hierarchical_error:.3f} vs {spa_pooled_error:.3f}); "
        f"expect >50% reduction"
    )


def test_posterior_std_smaller_than_sigma_eps_for_high_n_track() -> None:
    """Posterior std for a high-n, high-shrinkage track scales as
    sqrt(sigma_eps^2 / n) at most (and is multiplied by 1-lambda).
    Verify: a 100-sample track has post_std << sigma_eps (=0.5)."""
    data = {
        "a": [10.0 + 0.5 * (i % 3 - 1) for i in range(100)],  # mean=10
        "b": [11.0, 11.0, 11.0, 11.0],  # mean=11, n=4
    }
    out = fit_per_parameter(data)
    # within-track sigma_eps is roughly sqrt(pvariance(a values)) ~ 0.4
    sigma_eps = math.sqrt(0.5 ** 2 * 2 / 3)  # rough estimate for the 'a' track
    # The high-n track's posterior std should be much smaller.
    assert out["a"].std < sigma_eps, (
        f"high-n posterior std {out['a'].std:.4f} not much smaller than "
        f"sigma_eps {sigma_eps:.4f}"
    )
