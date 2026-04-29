from __future__ import annotations

import numpy as np
import pytest

from racingoptimizer.confidence import Confidence
from racingoptimizer.physics.exceptions import UntrainedError
from racingoptimizer.physics.fitters import ForestFitter


def _synthetic_damper_data(n: int = 200, noise: float = 0.05, seed: int = 7):
    rng = np.random.default_rng(seed)
    lsc = rng.uniform(0.0, 10.0, n)
    hsc = rng.uniform(0.0, 10.0, n)
    y = 0.3 * lsc + 0.7 * hsc + rng.normal(scale=noise, size=n)
    X = np.stack([lsc, hsc], axis=1)
    return X, y


def test_forest_recovers_mean_close_to_truth() -> None:
    X, y = _synthetic_damper_data()
    fitter = ForestFitter(random_state=7)
    fitter.fit(X, y)
    assert fitter.is_trained
    assert fitter.n_samples == X.shape[0]

    rng = np.random.default_rng(11)
    test_lsc = rng.uniform(2.0, 8.0, 30)
    test_hsc = rng.uniform(2.0, 8.0, 30)
    truth = 0.3 * test_lsc + 0.7 * test_hsc
    Xt = np.stack([test_lsc, test_hsc], axis=1)
    mean, std = fitter.predict(Xt)
    # Random forest on noisy synthetic stays within 1.0 of truth on the
    # interior of the training envelope.
    assert np.max(np.abs(mean - truth)) < 1.0
    assert std.shape == mean.shape
    # Per-tree std must be non-negative; never NaN.
    assert np.all(std >= 0.0)
    assert not np.isnan(std).any()


def test_forest_derived_confidence_regime() -> None:
    X, y = _synthetic_damper_data(n=200, noise=0.05)
    fitter = ForestFitter(random_state=7)
    fitter.fit(X, y)

    # Use the per-row predictions to build a typical CV residual estimate.
    mean, _ = fitter.predict(X)
    cv_residual_std = float(np.std(mean - y, ddof=0))
    signal_std = float(y.std(ddof=0))
    # On clean synthetic with > 100 samples and tiny residuals the regime
    # must land at one of the well-known tiers.
    conf = Confidence.derive(
        value=float(mean[0]),
        n_samples=fitter.n_samples,
        cv_residual_std=cv_residual_std,
        signal_std=signal_std,
    )
    assert conf.regime in {"dense", "confident"}


def test_forest_predict_untrained_raises() -> None:
    fitter = ForestFitter()
    with pytest.raises(UntrainedError):
        fitter.predict(np.array([[1.0, 2.0]]))


def test_forest_one_d_input_reshapes() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 60)
    y = 2.0 * x
    fitter = ForestFitter(random_state=0)
    fitter.fit(x, y)
    mean, _ = fitter.predict(np.array([0.4]))
    assert mean.shape == (1,)
