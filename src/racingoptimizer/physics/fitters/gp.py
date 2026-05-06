"""Gaussian-process fitter for continuous low-dim parameters.

Spec §5: continuous (springs, sliders, ride heights, wings, tyre pressures)
gets a GP because the posterior variance becomes `Confidence.lo/hi` directly
and sparse data widens naturally rather than silently extrapolating.

Fallback chain: Matérn-2.5 + WhiteKernel → RBF + WhiteKernel → untrained.

VISION §3 / §5 ("chase the chain"): the fitter MUST learn coupled
responses across heterogeneous features — setup parameters (heave 30..50,
wing 0..30), env channels (air_density ~1.14, wind_dir 0..360), and
corner archetype features (apex_speed 5..90, peak_lat_g 1..4). With a
SCALAR length scale the GP collapses onto whichever feature has the most
training-set variance and ignores the others; this is why the per-car v4
joint surrogate previously returned constant ride-height predictions
under setup perturbation. We standardise X (subtract per-feature mean,
divide by per-feature std) BEFORE fitting and at predict time so a single
shared length scale is meaningful for every feature axis.
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

from racingoptimizer.physics.fitters.base import FitterBase


def _matern_kernel(n_features: int = 1):
    """Isotropic Matérn (single shared length scale).

    Anisotropic ARD (per-feature length scales) was tried but the L-BFGS
    hyperparameter optimizer hangs on >20-dim length-scale vectors —
    impractical for the 100+ fitter per-car build. The X-standardization
    in `GPFitter.fit` puts every feature on O(1) scale before fitting so
    the shared length scale stays meaningful, but with high-dim mixed-scale
    inputs the per-car path forces ForestFitter (in `fitter.fit_per_car`)
    rather than relying on the GP to disentangle.
    """
    return (
        Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
    )


def _rbf_kernel(n_features: int = 1):
    return (
        RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
    )


# Numerical floor for per-feature std. Features that are constant across
# training (e.g. tyre pressure pinned at 152 kPa) get a 0.0 std which would
# zero-divide. Treating them as 1.0 leaves their normalised value at 0 and
# the GP correctly attributes zero importance.
_STD_FLOOR: float = 1e-9


class GPFitter(FitterBase):
    """Wraps `GaussianProcessRegressor`. `predict` returns `(mean, std)`.

    Inputs are standardised to zero mean / unit variance per feature so the
    Matérn kernel's single shared length scale has a uniform meaning across
    setup / env / archetype dimensions.
    """

    def __init__(self, *, random_state: int = 0xC0FFEE) -> None:
        super().__init__()
        self._random_state = int(random_state)
        self._gp: GaussianProcessRegressor | None = None
        # Per-feature mean/std captured at fit time so predict can apply the
        # same transform. Stored as numpy arrays — pickled along with the GP.
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Per-feature standardisation. Floor std so constant features
        # (tyre pressure pinned at 152, a single-session damper not in
        # constraints) don't zero-divide; their normalised column stays at 0.
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0, ddof=0)
        x_std = np.where(x_std < _STD_FLOOR, 1.0, x_std)
        X_norm = (X - x_mean) / x_std

        n_features = X_norm.shape[1]
        # Try Matérn first; on convergence failure or singularity, retry with RBF.
        # ConvergenceWarning alone is non-fatal — the GP still has a usable fit
        # (just at a sub-optimal kernel hyperparameter); only LinAlg/Value
        # errors during `.fit()` push us off to the fallback kernel.
        for kernel_factory in (_matern_kernel, _rbf_kernel):
            gp = GaussianProcessRegressor(
                kernel=kernel_factory(n_features),
                normalize_y=True,
                n_restarts_optimizer=0,
                random_state=self._random_state,
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    gp.fit(X_norm, y)
                # Pickle round-trip canonicalizes the fitted estimator so
                # subsequent `pickle.dumps` calls are byte-identical (spec
                # §12). A freshly-fit sklearn estimator shares numpy dtype
                # singletons across its fitted scalars/arrays via Python
                # identity; pickle's memo dedupes those once, but after a
                # load/dump cycle the array dtypes are fresh per-array
                # objects (no longer identity-shared with scalar dtypes),
                # so a re-pickle emits dtypes inline and grows by ~30 bytes
                # per estimator. One eager round-trip here puts the state
                # at the post-pickle fixed point.
                self._gp = pickle.loads(pickle.dumps(gp, protocol=pickle.HIGHEST_PROTOCOL))
                self._x_mean = np.asarray(x_mean, dtype=np.float64)
                self._x_std = np.asarray(x_std, dtype=np.float64)
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
        # Apply the same per-feature standardisation captured at fit time.
        if self._x_mean is not None and self._x_std is not None:
            X = (X - self._x_mean) / self._x_std
        mean, std = self._gp.predict(X, return_std=True)
        return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
