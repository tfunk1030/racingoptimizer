"""Random-forest fitter for high-dim / discrete parameters.

Spec §5: dampers (16-d clicks), corner weights, brake bias, diff get an
RF because they are discrete / coupled and tree ensembles handle interactions
without wild extrapolation. Bootstrap CI is approximated by the std across
the per-estimator predictions for each input row.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from racingoptimizer.physics.fitters.base import FitterBase


class ForestFitter(FitterBase):
    """Wraps `RandomForestRegressor`. `predict` returns `(mean, per-tree std)`."""

    def __init__(
        self,
        *,
        random_state: int = 0xC0FFEE,
        n_estimators: int = 100,
        max_depth: int | None = None,
    ) -> None:
        super().__init__()
        self._random_state = int(random_state)
        self._n_estimators = int(n_estimators)
        self._max_depth = max_depth
        self._rf: RandomForestRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        for n_est in (self._n_estimators, max(self._n_estimators // 2, 10)):
            try:
                rf = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=self._max_depth,
                    random_state=self._random_state,
                    n_jobs=1,
                )
                rf.fit(X, y)
                self._rf = rf
                self.is_trained = True
                self.n_samples = X.shape[0]
                return
            except (MemoryError, ValueError):
                continue
        self._rf = None

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_trained()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        assert self._rf is not None
        # Stack per-estimator predictions (n_estimators, n_rows) so std across
        # axis=0 gives the bootstrap-style per-row uncertainty.
        per_tree = np.stack(
            [tree.predict(X) for tree in self._rf.estimators_], axis=0
        )
        mean = per_tree.mean(axis=0)
        std = per_tree.std(axis=0, ddof=0)
        return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
