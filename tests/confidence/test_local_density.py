"""Day 2 of the physics-rebuild plan: per-parameter local density (Mode 4).

Background (PLAN.md Section 14.2):
- `Confidence.derive` answers "how good is the FITTER?" via global
  noise-ratio + sample count.
- The recommender ALSO needs "how good is the fitter AT THIS RECOMMENDED
  VALUE?" Even a well-fitted model has ~zero training density at a
  parameter value 10 steps from any observed sample.
- A polluted corpus reports `dense` while the recommended value has zero
  physics anchor (Mode 4 evidence: `confidence/confidence.py:60-92`,
  `CLAUDE.md` lines 108-110).
- Cheapest fix: per-parameter local density check on
  `min(|recommended - obs| for obs in observed_values)`.

Acceptance gate (per PLAN.md Section 14.2): for 5 representative
recommendations, every parameter whose recommended value is more than 3
*step* units from its nearest observed value gets regime label one
worse than its global label, OR `noisy`/`sparse` already.

Broken-model canary: with the local-density threshold set to a huge
number (effectively disabled), regime labels stay confident even when
recommended values drift far from observed. The unit test
`test_canary_huge_threshold_disables_downgrade` exercises that path
directly.
"""
from __future__ import annotations

import pytest

from racingoptimizer.confidence import Confidence
from racingoptimizer.confidence.confidence import (
    _LOCAL_DENSITY_THRESHOLD_STEPS,
    _REGIME_ORDER,
)


def _conf(regime: str = "dense", value: float = 1.0, n: int = 50) -> Confidence:
    return Confidence(value=value, lo=value, hi=value, n_samples=n, regime=regime)


# ---- Confidence.downgrade -----------------------------------------------

def test_downgrade_one_level_walks_hierarchy() -> None:
    """`dense -> confident -> noisy -> sparse`, one tier per call."""
    c = _conf("dense")
    assert c.downgrade(levels=1).regime == "confident"
    assert c.downgrade(levels=1).downgrade(levels=1).regime == "noisy"
    assert (
        c.downgrade(levels=1).downgrade(levels=1).downgrade(levels=1).regime
        == "sparse"
    )


def test_downgrade_below_sparse_floors_at_sparse() -> None:
    c = _conf("sparse")
    assert c.downgrade(levels=1).regime == "sparse"
    assert c.downgrade(levels=5).regime == "sparse"


def test_downgrade_zero_levels_is_noop() -> None:
    c = _conf("confident")
    out = c.downgrade(levels=0)
    assert out is c  # identity


def test_downgrade_negative_levels_raises() -> None:
    with pytest.raises(ValueError, match="levels must be >= 0"):
        _conf("dense").downgrade(levels=-1)


def test_downgrade_preserves_other_fields() -> None:
    c = Confidence(value=42.0, lo=40.0, hi=44.0, n_samples=88, regime="confident")
    out = c.downgrade(levels=1)
    assert out.value == 42.0
    assert out.lo == 40.0
    assert out.hi == 44.0
    assert out.n_samples == 88
    assert out.regime == "noisy"


# ---- Confidence.with_local_density --------------------------------------

def test_in_cluster_keeps_global_regime() -> None:
    """A recommended value inside the observed cluster keeps the global label."""
    c = _conf("dense")
    out = c.with_local_density(
        recommended=10.5, observed_values=[10.0, 11.0, 12.0], step=1.0,
    )
    assert out.regime == "dense"


def test_three_steps_away_triggers_downgrade() -> None:
    """Threshold is exclusive: > 3 * step downgrades; <= 3 * step stays."""
    observed = [10.0]
    step = 1.0
    # Exactly at threshold: stays.
    c1 = _conf("dense").with_local_density(
        recommended=13.0, observed_values=observed, step=step,
    )
    assert c1.regime == "dense"
    # Just past threshold: downgrades.
    c2 = _conf("dense").with_local_density(
        recommended=13.5, observed_values=observed, step=step,
    )
    assert c2.regime == "confident"


def test_downgrade_chain_dense_to_confident_to_noisy() -> None:
    observed = [10.0]
    step = 1.0
    far = 100.0
    assert _conf("dense").with_local_density(
        recommended=far, observed_values=observed, step=step,
    ).regime == "confident"
    assert _conf("confident").with_local_density(
        recommended=far, observed_values=observed, step=step,
    ).regime == "noisy"


def test_already_noisy_downgrades_to_sparse() -> None:
    out = _conf("noisy").with_local_density(
        recommended=100.0, observed_values=[10.0], step=1.0,
    )
    assert out.regime == "sparse"


def test_already_sparse_stays_sparse() -> None:
    """The sparse floor: no further downgrade is possible."""
    out = _conf("sparse").with_local_density(
        recommended=100.0, observed_values=[10.0], step=1.0,
    )
    assert out.regime == "sparse"


def test_no_observed_values_returns_self_unchanged() -> None:
    """Caller decides what an empty observed set means -- the helper is a no-op.

    For Mode 5 (untrained car/track), the caller upstream already enforces
    sparse; this helper does not silently re-decide.
    """
    c = _conf("dense")
    out = c.with_local_density(
        recommended=100.0, observed_values=[], step=1.0,
    )
    assert out is c


def test_zero_step_returns_self_unchanged() -> None:
    """Defensive: bad input doesn't crash; we keep the global label."""
    c = _conf("dense")
    out = c.with_local_density(
        recommended=100.0, observed_values=[10.0], step=0.0,
    )
    assert out is c


def test_canary_huge_threshold_disables_downgrade() -> None:
    """Broken-model canary: a huge threshold makes the helper a no-op
    even when recommended drifts FAR from observed.

    PLAN.md Section 14.2's runtime canary is "set the local-density
    threshold to a huge number, gate must FAIL." This unit test
    exercises that path directly: with `threshold_steps=1e9`, no
    downgrade occurs even at distance 100 * step.
    """
    out = _conf("dense").with_local_density(
        recommended=100.0,
        observed_values=[10.0],
        step=1.0,
        threshold_steps=1e9,
    )
    assert out.regime == "dense"


def test_negative_observed_value_handled() -> None:
    """Defensive: signed parameters (e.g. perch offsets, toe) work."""
    # -97 to {-105, -95}: nearest is -95, distance 2.0 (within 3*step=3.0).
    out = _conf("dense").with_local_density(
        recommended=-97.0, observed_values=[-105.0, -95.0], step=1.0,
    )
    assert out.regime == "dense"
    # -200 to {-105, -95}: nearest is -95, distance 105 (>> 3*step).
    out = _conf("dense").with_local_density(
        recommended=-200.0, observed_values=[-105.0, -95.0], step=1.0,
    )
    assert out.regime == "confident"


def test_default_threshold_is_three_steps() -> None:
    """The exported _LOCAL_DENSITY_THRESHOLD_STEPS constant matches the
    plan's documented '> 3 step units'."""
    assert _LOCAL_DENSITY_THRESHOLD_STEPS == 3.0


def test_regime_order_matches_audit_constant() -> None:
    """`_REGIME_ORDER` is the canonical hierarchy used by both
    `Confidence.derive` (for setting) and `downgrade`/`with_local_density`
    (for tier walks)."""
    assert _REGIME_ORDER == ("dense", "confident", "noisy", "sparse")
