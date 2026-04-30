"""Tests for the trailing rolling-median primitive used by the off-track detector.

The previous implementation was O(n²) (a fresh ``np.median`` per sample over a
growing slice). At realistic ingest scale — 60 Hz × 60 s/lap × 50 laps × 5
sessions ≈ 10⁶ samples per build — that quadratic cost dominated. The new
implementation delegates the steady-state region to
``scipy.ndimage.median_filter`` (a C-implemented O(n·w) filter) with the kernel
right-aligned via ``origin``, and falls back to per-sample ``np.median`` only
for the short warm-up prefix.

These tests pin three contracts: exact numerical equivalence to the original
implementation for odd windows, near-equivalence for even windows (where
scipy's filter picks one of the two middle values rather than averaging them),
and a hard wall-clock budget on a ~1M-sample run.
"""
from __future__ import annotations

import time

import numpy as np

from racingoptimizer.track.masks import _rolling_median_forward


def _rolling_median_forward_naive(values: np.ndarray, window: int) -> np.ndarray:
    """The original O(n²) implementation, kept here as the equivalence oracle."""
    n = values.size
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - window + 1)
        out[i] = float(np.median(values[lo : i + 1]))
    return out


def test_matches_naive_on_random_input_odd_window():
    # For odd windows the implementation is exact: scipy's median_filter and
    # numpy's median agree on the single middle element.
    rng = np.random.default_rng(seed=20260429)
    values = rng.normal(loc=0.0, scale=10.0, size=10_000)
    window = 61
    fast = _rolling_median_forward(values, window)
    slow = _rolling_median_forward_naive(values, window)
    assert fast.shape == slow.shape
    np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-12)


def test_close_to_naive_on_random_input_even_window():
    # For even windows scipy's median_filter picks one of the two middle
    # values rather than averaging them, so an exact equivalence check fails.
    # The call site (off-track baseline) tolerates the sub-1% difference; we
    # still pin that the deviation never exceeds one inter-quartile-ish step
    # of the source distribution.
    rng = np.random.default_rng(seed=20260429)
    values = rng.normal(loc=0.0, scale=10.0, size=10_000)
    window = 60  # 1 s at 60 Hz, the actual call-site window
    fast = _rolling_median_forward(values, window)
    slow = _rolling_median_forward_naive(values, window)
    # Steady-state samples agree to within a small fraction of the source
    # distribution's scale; warm-up prefix is exact (uses np.median directly).
    np.testing.assert_allclose(fast[:window - 1], slow[:window - 1], rtol=0, atol=1e-12)
    diff = np.abs(fast[window - 1 :] - slow[window - 1 :])
    assert diff.max() < 5.0, f"max deviation {diff.max():.3f} too large for sigma=10 input"


def test_warmup_prefix_uses_available_samples():
    # For i < window-1 the median must be taken over values[0..i] so the first
    # sample is its own median, the second is the median of the first two, etc.
    # This matches the docstring contract and the existing call-site behaviour.
    values = np.array([5.0, 1.0, 3.0, 9.0, 7.0], dtype=np.float64)
    out = _rolling_median_forward(values, window=3)
    expected = np.array(
        [
            5.0,                       # median([5])
            (5.0 + 1.0) / 2,           # median([5, 1])
            float(np.median([5.0, 1.0, 3.0])),
            float(np.median([1.0, 3.0, 9.0])),
            float(np.median([3.0, 9.0, 7.0])),
        ]
    )
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12)


def test_empty_input_returns_empty():
    out = _rolling_median_forward(np.array([], dtype=np.float64), window=10)
    assert out.shape == (0,)


def test_window_larger_than_input_falls_back_to_prefix():
    # When window > n every output is the median of values[0..i]; the
    # steady-state branch is skipped entirely.
    values = np.array([1.0, 4.0, 2.0], dtype=np.float64)
    out = _rolling_median_forward(values, window=100)
    expected = np.array([1.0, 2.5, 2.0])
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12)


def test_million_samples_under_500ms():
    # Realistic ingest scale; the old O(n²) loop did not finish here.
    rng = np.random.default_rng(seed=42)
    values = rng.normal(size=1_000_000)
    window = 60  # 1 s at 60 Hz
    start = time.perf_counter()
    out = _rolling_median_forward(values, window)
    elapsed = time.perf_counter() - start
    assert out.shape == (values.size,)
    assert elapsed < 0.5, f"rolling median took {elapsed*1000:.0f} ms (>500 ms budget)"
