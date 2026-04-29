"""Physics fitter exception hierarchy."""
from __future__ import annotations


class UntrainedError(RuntimeError):
    """Raised when prediction is requested for a fitter that never trained."""


class InsufficientDataError(RuntimeError):
    """Raised when `fit()` produced zero working fitters for a car."""
