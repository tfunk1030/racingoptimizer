"""P2.3 -- inverse-track-sample-count training weights.

When the per-car v4 surrogate pools every session across every track,
an over-represented track (e.g. BMW Sebring 37 sessions vs Spa 11)
otherwise dominates the Forest's split criteria so under-sampled
tracks inherit the over-represented track's setup philosophy. P2.3
attaches a ``_track_balance_weight`` column = ``1 / sqrt(n_track_rows)``
and pipes it through to the fitter as ``sample_weight``.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl

from racingoptimizer.physics.fitter import (
    _TRACK_BALANCE_WEIGHT_COLUMN,
    _attach_track_balance_weights,
)
from racingoptimizer.physics.fitters.forest import ForestFitter


def test_attach_track_balance_weights_inverse_sqrt() -> None:
    """A 10:1 row-count split produces ``1/sqrt(10) ~= 0.316`` vs
    ``1/sqrt(1) = 1.0``."""
    rows = []
    for i in range(10):
        rows.append({"session_id": f"sebring_{i}", "feature": float(i)})
    rows.append({"session_id": "spa_0", "feature": 99.0})
    frame = pl.DataFrame(rows)
    track_per_session = {f"sebring_{i}": "sebring" for i in range(10)}
    track_per_session["spa_0"] = "spa"

    out = _attach_track_balance_weights(frame, track_per_session=track_per_session)
    assert _TRACK_BALANCE_WEIGHT_COLUMN in out.columns

    sebring_weights = (
        out.filter(pl.col("session_id").str.starts_with("sebring_"))
        [_TRACK_BALANCE_WEIGHT_COLUMN]
        .to_list()
    )
    spa_weight = (
        out.filter(pl.col("session_id") == "spa_0")
        [_TRACK_BALANCE_WEIGHT_COLUMN]
        .to_list()[0]
    )
    assert all(math.isclose(w, 1.0 / math.sqrt(10), rel_tol=1e-6) for w in sebring_weights)
    assert math.isclose(spa_weight, 1.0, rel_tol=1e-6)


def test_attach_track_balance_weights_missing_track_defaults_to_one() -> None:
    frame = pl.DataFrame(
        {"session_id": ["a", "b", "c"], "feature": [1.0, 2.0, 3.0]}
    )
    out = _attach_track_balance_weights(
        frame, track_per_session={"a": "sebring"},
    )
    weights = dict(
        zip(
            out["session_id"].to_list(),
            out[_TRACK_BALANCE_WEIGHT_COLUMN].to_list(),
            strict=True,
        )
    )
    # session 'a' has 1 row at sebring → 1/sqrt(1)=1.0
    assert math.isclose(weights["a"], 1.0, rel_tol=1e-6)
    # b, c have no track mapping → fallback weight 1.0
    assert math.isclose(weights["b"], 1.0, rel_tol=1e-6)
    assert math.isclose(weights["c"], 1.0, rel_tol=1e-6)


def test_attach_track_balance_weights_empty_frame_is_noop() -> None:
    frame = pl.DataFrame(
        {"session_id": [], "feature": []},
        schema={"session_id": pl.Utf8, "feature": pl.Float64},
    )
    out = _attach_track_balance_weights(
        frame, track_per_session={"a": "sebring"},
    )
    assert out.height == 0


def test_attach_track_balance_weights_no_track_dict_uses_unit_weight() -> None:
    frame = pl.DataFrame(
        {"session_id": ["a", "b"], "feature": [1.0, 2.0]}
    )
    out = _attach_track_balance_weights(frame, track_per_session={})
    weights = out[_TRACK_BALANCE_WEIGHT_COLUMN].to_list()
    assert weights == [1.0, 1.0]


def test_forest_fitter_honours_sample_weight_pulling_undersampled_track() -> None:
    """Synthetic two-track corpus where one track has 10x the rows.
    With the under-sampled track flagged via sample_weight, the fit's
    prediction at the under-sampled feature region must move closer to
    that track's actual y values than an unweighted fit does.
    """
    rng = np.random.default_rng(2026)
    # Track A (over-represented): y = +1 for x < 0.5
    # Track B (under-sampled): y = -1 for x < 0.5 (opposite signal at same x)
    n_a = 200
    n_b = 20
    x_a = rng.uniform(0.0, 0.5, size=(n_a, 1))
    y_a = np.full(n_a, 1.0) + rng.normal(0.0, 0.05, size=n_a)
    x_b = rng.uniform(0.0, 0.5, size=(n_b, 1))
    y_b = np.full(n_b, -1.0) + rng.normal(0.0, 0.05, size=n_b)

    X = np.concatenate([x_a, x_b], axis=0)
    y = np.concatenate([y_a, y_b], axis=0)
    weights = np.concatenate(
        [
            np.full(n_a, 1.0 / math.sqrt(n_a)),
            np.full(n_b, 1.0 / math.sqrt(n_b)),
        ],
    )

    probe_x = np.array([[0.25]])
    expected_at_b = -1.0

    f_un = ForestFitter(random_state=42)
    f_un.fit(X, y)
    pred_un, _ = f_un.predict(probe_x)

    f_w = ForestFitter(random_state=42)
    f_w.fit(X, y, sample_weight=weights)
    pred_w, _ = f_w.predict(probe_x)

    err_un = abs(float(pred_un[0]) - expected_at_b)
    err_w = abs(float(pred_w[0]) - expected_at_b)
    assert err_w < err_un, (
        f"weighted fit prediction ({float(pred_w[0]):.3f}) should be "
        f"closer to under-sampled track value ({expected_at_b}) than "
        f"unweighted ({float(pred_un[0]):.3f})"
    )


def test_ridge_and_gp_fitters_silently_accept_sample_weight_kwarg() -> None:
    """Ridge and GP do not honour sample_weight (track-invariant
    targets) but must accept the kwarg without crashing -- callers
    pass it uniformly across fitter families."""
    from racingoptimizer.physics.fitters.gp import GPFitter
    from racingoptimizer.physics.fitters.ridge import RidgeFitter

    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    y = X.sum(axis=1) + rng.normal(scale=0.05, size=30)
    weights = rng.uniform(0.1, 1.0, size=30)

    ridge = RidgeFitter(random_state=0)
    ridge.fit(X, y, sample_weight=weights)
    assert ridge.is_trained

    gp = GPFitter(random_state=0)
    gp.fit(X, y, sample_weight=weights)
    assert gp.is_trained
