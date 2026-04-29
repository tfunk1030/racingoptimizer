from __future__ import annotations

import numpy as np
import pytest

from racingoptimizer.confidence import Confidence
from racingoptimizer.physics.exceptions import UntrainedError
from racingoptimizer.physics.fitters import GPFitter


def _synthetic_spring_data(n: int = 200, noise: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, n)
    y = 2.0 * x + rng.normal(scale=noise, size=n)
    return x.reshape(-1, 1), y


def test_gp_recovers_synthetic_slope() -> None:
    X, y = _synthetic_spring_data()
    fitter = GPFitter(random_state=42)
    fitter.fit(X, y)
    assert fitter.is_trained
    assert fitter.n_samples == X.shape[0]

    # Predict at the median of x — slope ≈ 2 ± 0.1 implies prediction near 10.
    median_x = np.array([[5.0]])
    mean, std = fitter.predict(median_x)
    assert mean.shape == (1,)
    assert std.shape == (1,)
    assert mean[0] == pytest.approx(10.0, abs=0.5)
    # Local slope check: predict at 4.5 and 5.5; (y2 - y1) / 1.0 ≈ 2.0.
    near = np.array([[4.5], [5.5]])
    pmean, _ = fitter.predict(near)
    slope = (pmean[1] - pmean[0]) / 1.0
    assert slope == pytest.approx(2.0, abs=0.1)


def test_gp_derived_confidence_dense_for_clean_synthetic() -> None:
    X, y = _synthetic_spring_data(n=200, noise=0.05)
    fitter = GPFitter(random_state=42)
    fitter.fit(X, y)
    mean, std = fitter.predict(np.array([[5.0]]))

    cv_residual_std = 0.05  # noise σ
    signal_std = float(y.std(ddof=0))
    conf = Confidence.derive(
        value=float(mean[0]),
        n_samples=fitter.n_samples,
        cv_residual_std=cv_residual_std,
        signal_std=signal_std,
    )
    assert conf.regime == "dense"


def test_gp_predict_untrained_raises() -> None:
    fitter = GPFitter()
    with pytest.raises(UntrainedError):
        fitter.predict(np.array([[1.0]]))


def test_gp_handles_one_d_input() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 50)
    y = 3.0 * x + rng.normal(scale=0.05, size=50)
    fitter = GPFitter(random_state=0)
    fitter.fit(x, y)  # 1-D input — must be auto-reshaped.
    assert fitter.is_trained
    mean, _ = fitter.predict(np.array([0.5]))
    assert mean[0] == pytest.approx(1.5, abs=0.2)
