"""Integration tests for `_observed_values_for_param` (PLAN.md Day 2).

The helper is the bridge between `Confidence.with_local_density(...)` and
the recommend pipeline. These tests verify the right observed-value list
is returned for each model layout (v3 / v4 / cross-car-borrow / no-data).
"""
from __future__ import annotations

from types import SimpleNamespace

from racingoptimizer.confidence import Confidence
from racingoptimizer.physics.recommend import _observed_values_for_param


def _model(
    *,
    per_track: dict | None = None,
    baseline: dict | None = None,
    std: dict | None = None,
):
    """Build a synthetic model with just the attributes the helper reads."""
    return SimpleNamespace(
        per_track_parameter_observed=per_track,
        baseline_setup=baseline or {},
        parameter_observed_std=std or {},
    )


def test_v4_returns_per_track_observed() -> None:
    """For per-car (v4) models, prefer the per-track observed list."""
    m = _model(
        per_track={
            "spa_2024_up": {"rear_wing_angle_deg": (14.0, 14.0, 15.0)},
            "sebring": {"rear_wing_angle_deg": (17.0, 17.0, 17.5)},
        },
    )
    out = _observed_values_for_param(m, "spa_2024_up", "rear_wing_angle_deg")
    assert out == (14.0, 14.0, 15.0)
    out2 = _observed_values_for_param(m, "sebring", "rear_wing_angle_deg")
    assert out2 == (17.0, 17.0, 17.5)


def test_v4_unknown_track_falls_through() -> None:
    """A v4 model with no entry for the target track falls through to the
    v3 baseline+/-std synthesis."""
    m = _model(
        per_track={"sebring": {"rear_wing_angle_deg": (17.0,)}},
        baseline={"rear_wing_angle_deg": 14.0},
        std={"rear_wing_angle_deg": 1.0},
    )
    out = _observed_values_for_param(m, "spa_2024_up", "rear_wing_angle_deg")
    assert out == (13.0, 14.0, 15.0)


def test_v3_synthesises_cluster_from_baseline_and_std() -> None:
    """v3 models have no per-track raw values; we approximate with a
    3-point cluster (baseline ± std)."""
    m = _model(
        per_track=None,
        baseline={"rear_wing_angle_deg": 14.0},
        std={"rear_wing_angle_deg": 1.5},
    )
    out = _observed_values_for_param(m, "spa_2024_up", "rear_wing_angle_deg")
    assert out == (12.5, 14.0, 15.5)


def test_v3_zero_std_returns_singleton() -> None:
    """When std is 0 (parameter held constant in corpus), the cluster is
    the singleton baseline value."""
    m = _model(
        per_track=None,
        baseline={"tyre_cold_pressure_kpa": 152.0},
        std={"tyre_cold_pressure_kpa": 0.0},
    )
    out = _observed_values_for_param(m, "any", "tyre_cold_pressure_kpa")
    assert out == (152.0,)


def test_no_data_returns_empty() -> None:
    """No baseline + no per-track data = empty tuple. Caller leaves the
    global regime label alone."""
    m = _model(per_track=None, baseline={}, std={})
    out = _observed_values_for_param(m, "any", "missing_param")
    assert out == ()


def test_with_local_density_in_cluster_keeps_regime() -> None:
    """End-to-end: helper -> with_local_density returns global regime when
    recommended is inside the v4 per-track cluster."""
    m = _model(
        per_track={
            "spa_2024_up": {"rear_wing_angle_deg": (14.0, 14.0, 15.0)},
        },
    )
    obs = _observed_values_for_param(m, "spa_2024_up", "rear_wing_angle_deg")
    c = Confidence(value=14.0, lo=13.5, hi=14.5, n_samples=50, regime="dense")
    out = c.with_local_density(
        recommended=14.0, observed_values=obs, step=1.0,
    )
    assert out.regime == "dense"


def test_with_local_density_out_of_cluster_downgrades() -> None:
    """A recommended wing angle 5 steps off the observed cluster
    downgrades dense -> confident."""
    m = _model(
        per_track={
            "spa_2024_up": {"rear_wing_angle_deg": (14.0, 14.0, 15.0)},
        },
    )
    obs = _observed_values_for_param(m, "spa_2024_up", "rear_wing_angle_deg")
    c = Confidence(value=14.0, lo=13.5, hi=14.5, n_samples=50, regime="dense")
    out = c.with_local_density(
        recommended=20.0, observed_values=obs, step=1.0,
    )
    assert out.regime == "confident"


def test_canary_disabled_helper_keeps_dense_when_far() -> None:
    """Broken-model canary (PLAN.md Section 14.2): if the
    `_observed_values_for_param` helper or the wire-in is reverted, a
    far-from-cluster recommendation keeps the global regime label.

    This test exercises the bypass directly: with NO with_local_density
    call, the same far-recommended value retains its dense label.
    """
    c = Confidence(value=14.0, lo=13.5, hi=14.5, n_samples=50, regime="dense")
    # No call to with_local_density.
    assert c.regime == "dense"
    # Verify the helper would HAVE downgraded it (positive control):
    out = c.with_local_density(
        recommended=20.0, observed_values=(14.0, 15.0), step=1.0,
    )
    assert out.regime == "confident"
