"""Gaussian-process fitter for continuous low-dim parameters.

Spec §5: continuous (springs, sliders, ride heights, wings, tyre pressures)
gets a GP because the posterior variance becomes `Confidence.lo/hi` directly
and sparse data widens naturally rather than silently extrapolating.

Fallback chain: Matérn-2.5 + WhiteKernel → RBF + WhiteKernel → untrained.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

from racingoptimizer.physics.fitters.base import FitterBase


def _matern_kernel():
    return Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)


def _rbf_kernel():
    return RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)


class GPFitter(FitterBase):
    """Wraps `GaussianProcessRegressor`. `predict` returns `(mean, std)`."""

    def __init__(self, *, random_state: int = 0xC0FFEE) -> None:
        super().__init__()
        self._random_state = int(random_state)
        self._gp: GaussianProcessRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Try Matérn first; on convergence failure or singularity, retry with RBF.
        # ConvergenceWarning alone is non-fatal — the GP still has a usable fit
        # (just at a sub-optimal kernel hyperparameter); only LinAlg/Value
        # errors during `.fit()` push us off to the fallback kernel.
        for kernel_factory in (_matern_kernel, _rbf_kernel):
            gp = GaussianProcessRegressor(
                kernel=kernel_factory(),
                normalize_y=True,
                n_restarts_optimizer=0,
                random_state=self._random_state,
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    gp.fit(X, y)
                self._gp = gp
                self.is_trained = True
                self.n_samples = X.shape[0]
                return
            except (np.linalg.LinAlgError, ValueError):
                continue
        # Both kernels failed — leave `is_trained=False`.
        self._gp = None

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_trained()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        assert self._gp is not None
        mean, std = self._gp.predict(X, return_std=True)
        return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
