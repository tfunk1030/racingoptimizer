"""Ridge-regression fitter for deterministic setup-readout channels.

VISION §3 / §5 ("chase the chain"): static ride heights, aero balance %,
L/D ratio are deterministic readouts iRacing's garage calculator emits as
a function of the bounded setup parameters alone — they do NOT depend on
track, corner archetype, environmental context, or driver execution. The
mapping is approximately linear in the local neighbourhood of any base
setup (spring equilibrium math), so a regularised linear model fits this
class of target with far higher fidelity in low-data regimes than an
ensemble of trees.

Forest with ~18 unique (setup → static_lr_rh) data points underfits to
near-constant predictions — the chain shows up as ±0.04 mm of response
across a 20 N/mm heave sweep. Ridge with the same data captures the
clean linear correlation that's actually present in the calculator
output, lifting the chain response to physically meaningful magnitudes
(driver-perceptible mm-per-click).

The fitter exposes the same `(mean, std)` predict signature as
`GPFitter` / `ForestFitter` so the orchestrator can route channels to it
without special-casing the predict path. ``std`` is approximated from
the training-set residual std — there is no per-row posterior because
ridge is a point estimator.
"""
from __future__ import annotations

import pickle

import numpy as np
from sklearn.linear_model import Ridge

from racingoptimizer.physics.fitters.base import FitterBase

# Ridge regularisation strength. Small values (favour fit) suit setup-
# readout targets because the underlying mapping is deterministic — we
# expect the corpus-observed (setup → readout) pairs to fall almost
# exactly on a hyperplane modulo iRacing's display-rounding (0.1 mm,
# 0.01%). Values much larger than 1.0 wash out the very signal we
# added Ridge to capture.
_DEFAULT_ALPHA: float = 0.1

# Numerical floor for per-feature std. Constant features (e.g. tyre
# pressure pinned at 152 kPa across every session) have std=0 and zero-
# divide. Treating them as 1.0 leaves their normalised value at 0 and
# Ridge correctly attributes zero coefficient.
_STD_FLOOR: float = 1e-9


class RidgeFitter(FitterBase):
    """Wraps `sklearn.linear_model.Ridge`. Returns ``(mean, std)`` where
    ``std`` is the constant training-set residual std (broadcast across
    every prediction row).

    Inputs are standardised per-feature so the single regularisation
    strength applies uniformly across heterogeneous-scale features
    (heave 30..50, wing 0..30, archetype 5..100, env 0..1000).
    """

    def __init__(
        self,
        *,
        random_state: int = 0xC0FFEE,
        alpha: float = _DEFAULT_ALPHA,
    ) -> None:
        super().__init__()
        self._random_state = int(random_state)
        self._alpha = float(alpha)
        self._ridge: Ridge | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._residual_std: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0, ddof=0)
        x_std = np.where(x_std < _STD_FLOOR, 1.0, x_std)
        X_norm = (X - x_mean) / x_std

        try:
            ridge = Ridge(
                alpha=self._alpha,
                random_state=self._random_state,
            )
            ridge.fit(X_norm, y)
            # Residual std for Confidence bracketing. Using the in-sample
            # residual is acceptable for setup-readout channels because
            # the underlying mapping is deterministic — out-of-sample
            # extrapolation noise is dominated by the ridge prior, not
            # by data noise.
            residuals = y - ridge.predict(X_norm)
            self._residual_std = float(np.std(residuals, ddof=0))
            # Pickle round-trip for byte-identical re-pickle (matches
            # GPFitter / ForestFitter convention; see GPFitter.fit for
            # the dtype-singleton rationale). sklearn's Ridge needs
            # TWO round-trips: the first pickle materialises lazy
            # internal state (`solver_`, etc.) and grows the pickle
            # by ~34 bytes; the second round-trip is stable. The
            # WRAPPER (this fitter's __dict__) also needs one full
            # round-trip after that because the multiple numpy
            # arrays (`coef_` inside Ridge, plus `_x_mean`, `_x_std`)
            # share dtype singletons via pickle memos at first
            # pickling — after unpickling the memos differ slightly
            # and the next pickle gains ~34 bytes. Without both
            # fixes `test_pickle_round_trip_per_car` fails on every
            # car that has any readout-channel fitter.
            self._ridge = pickle.loads(pickle.dumps(
                pickle.loads(
                    pickle.dumps(ridge, protocol=pickle.HIGHEST_PROTOCOL)
                ),
                protocol=pickle.HIGHEST_PROTOCOL,
            ))
            self._x_mean = np.asarray(x_mean, dtype=np.float64)
            self._x_std = np.asarray(x_std, dtype=np.float64)
            self.is_trained = True
            self.n_samples = X.shape[0]
            # Final wrapper-level round-trip to stabilise the full
            # `self.__dict__` pickle (see comment above).
            stable_state = pickle.loads(
                pickle.dumps(self.__dict__, protocol=pickle.HIGHEST_PROTOCOL)
            )
            self.__dict__.update(stable_state)
        except (np.linalg.LinAlgError, ValueError):
            self._ridge = None

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_trained()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        assert self._ridge is not None and self._x_mean is not None
        X_norm = (X - self._x_mean) / self._x_std
        mean = self._ridge.predict(X_norm)
        # Uniform residual-std bracket per prediction row.
        std = np.full(X_norm.shape[0], self._residual_std, dtype=np.float64)
        return np.asarray(mean, dtype=np.float64), std
