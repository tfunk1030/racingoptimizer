"""P2.4 -- corner archetype carried into the v4 fit-time feature matrix.

Today the corner archetype is built per-corner at train time and per-
schedule at predict time. P2.4 promotes the per-PHASE duration into
the joint feature vector so the surrogate can learn within-corner
phase asymmetry (e.g. brake bias matters more on corners with extended
braking phases). Coverage:

* ``phase_duration_s`` appears in ``CORNER_ARCHETYPE_COLUMNS`` after
  the schema bump.
* ``_attach_corner_archetypes`` materialises a per-row
  ``phase_duration_s`` column (= ``t_end_s - t_start_s`` per row).
* A Forest fit on a synthetic two-radius corner pair where one
  parameter's effect varies with radius produces a non-zero feature
  importance for the radius-style archetype column.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from racingoptimizer.physics.fitter import (
    CORNER_ARCHETYPE_COLUMNS,
    ENV_FEATURE_SCHEMA_VERSION_PER_CAR,
    _attach_corner_archetypes,
)
from racingoptimizer.physics.fitters.forest import ForestFitter


def test_phase_duration_s_in_archetype_columns() -> None:
    assert "phase_duration_s" in CORNER_ARCHETYPE_COLUMNS


def test_schema_version_bumped_for_archetype_changes() -> None:
    # Was 5 pre-P2.4; bump captures the feature-vector layout change so
    # old pickles don't try to revive against a longer feature vector.
    assert ENV_FEATURE_SCHEMA_VERSION_PER_CAR >= 6


def test_attach_corner_archetypes_emits_phase_duration_s() -> None:
    """Every (corner, phase) row gets its phase duration as a feature.
    The corner-level keys (apex_speed, duration_s) are session-pooled
    and identical across phases of the same corner; phase_duration_s
    differentiates phases within a corner."""
    frame = pl.DataFrame(
        {
            "session_id": ["s1"] * 4,
            "corner_id": [1, 1, 1, 1],
            "phase": ["braking", "trail_brake", "mid_corner", "exit"],
            "speed_min_ms": [40.0, 35.0, 30.0, 35.0],
            "speed_max_ms": [80.0, 60.0, 50.0, 80.0],
            "accel_lat_g_max": [0.5, 1.0, 1.4, 0.9],
            "t_start_s": [0.0, 1.0, 2.0, 3.5],
            "t_end_s": [1.0, 2.0, 3.5, 4.5],
        }
    )
    out = _attach_corner_archetypes(frame)
    assert "phase_duration_s" in out.columns
    by_phase = dict(
        zip(out["phase"].to_list(), out["phase_duration_s"].to_list(), strict=True)
    )
    assert by_phase["braking"] == 1.0
    assert by_phase["trail_brake"] == 1.0
    assert by_phase["mid_corner"] == 1.5
    assert by_phase["exit"] == 1.0


def test_forest_picks_up_archetype_radius_interaction() -> None:
    """Synthetic two-radius case: y = wing * f(radius) + noise. Without
    the radius archetype feature, Forest can't separate the two
    radii's regimes and produces low feature importance for radius.
    With the archetype feature included, importance becomes non-zero
    AND prediction at the held-out radius regime is closer to truth.
    """
    rng = np.random.default_rng(2026)
    n = 200
    wing = rng.uniform(5.0, 15.0, size=n)
    radius = rng.choice([30.0, 200.0], size=n)  # tight vs sweeping
    # At tight radius: wing has STRONG effect on lateral G.
    # At sweeping radius: wing has near-zero effect.
    sensitivity = np.where(radius < 100.0, 0.10, 0.005)
    y = sensitivity * wing + rng.normal(0.0, 0.05, size=n)

    # Without the radius feature: Forest only sees wing.
    X_without = wing.reshape(-1, 1)
    f_wo = ForestFitter(random_state=42)
    f_wo.fit(X_without, y)

    # With the radius feature in the matrix.
    X_with = np.column_stack([wing, radius])
    f_with = ForestFitter(random_state=42)
    f_with.fit(X_with, y)

    # Probe at tight-radius wing=15 (should be ~1.5)
    probe_with = np.array([[15.0, 30.0]])
    probe_without = np.array([[15.0]])
    pred_with, _ = f_with.predict(probe_with)
    pred_wo, _ = f_wo.predict(probe_without)
    truth_at_tight = 0.10 * 15.0  # ~1.5

    err_with = abs(float(pred_with[0]) - truth_at_tight)
    err_wo = abs(float(pred_wo[0]) - truth_at_tight)
    assert err_with < err_wo, (
        f"with-archetype prediction err {err_with:.3f} should beat "
        f"without-archetype err {err_wo:.3f} for tight-radius wing=15"
    )

    # Forest feature importance: radius non-zero when included.
    assert f_with._rf is not None  # type: ignore[attr-defined]
    importances = f_with._rf.feature_importances_  # type: ignore[attr-defined]
    radius_importance = float(importances[1])
    assert radius_importance > 0.05, (
        f"radius feature importance {radius_importance:.3f} too low; "
        "expected >= 0.05 once the interaction is exposed to the fit"
    )
