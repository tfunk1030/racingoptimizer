"""Common interface for the GP / RF fitter families.

Both subclasses expose `fit(X, y)` and `predict(X) -> (mean, std)` so the
orchestrator in `racingoptimizer.physics.fitter` can treat them uniformly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from racingoptimizer.physics.exceptions import UntrainedError


class FitterBase(ABC):
    is_trained: bool
    n_samples: int

    def __init__(self) -> None:
        self.is_trained = False
        self.n_samples = 0

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit the model.

        ``sample_weight`` (optional) lets callers up-weight or down-weight
        individual rows -- used by P2.3 (inverse-track-sample-count
        weights) to keep a Sebring-heavy corpus from drowning Spa rows.
        Implementations that cannot honour weights MUST silently ignore
        them so the orchestrator can pass them unconditionally.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def _ensure_trained(self) -> None:
        if not self.is_trained:
            raise UntrainedError(f"{type(self).__name__} has not been trained")
