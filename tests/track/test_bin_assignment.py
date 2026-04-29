"""Pin the binning math + sessions_hash invariants (slice D-1, U6)."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from racingoptimizer.track import bin_index, sessions_hash, track_pos_m_from_pct


def test_track_pos_m_from_pct_round_lap():
    out = track_pos_m_from_pct(np.array([0.0, 0.5, 1.0]), 4574.0)
    assert np.array_equal(out, np.array([0.0, 2287.0, 4574.0]))
    assert out.dtype == np.float64


def test_bin_index_default_5m():
    out = bin_index(np.array([2287.0]), bin_size_m=5.0)
    assert np.array_equal(out, np.array([457], dtype=np.int32))
    assert out.dtype == np.int32


def test_bin_index_exact_boundary_floors():
    assert bin_index(np.array([5.0]), bin_size_m=5.0).tolist() == [1]


def test_bin_index_negative_passes_through():
    out = bin_index(np.array([-1.0, -5.0, -7.5]), bin_size_m=5.0)
    assert out.tolist() == [-1, -1, -2]


def test_bin_index_is_pure():
    arr = np.array([10.0, 20.0, 30.0])
    a = bin_index(arr, bin_size_m=5.0)
    b = bin_index(arr, bin_size_m=5.0)
    assert np.array_equal(a, b)


def test_track_pos_m_from_pct_is_pure():
    arr = np.array([0.1, 0.2, 0.3])
    a = track_pos_m_from_pct(arr, 5000.0)
    b = track_pos_m_from_pct(arr, 5000.0)
    assert np.array_equal(a, b)


def test_sessions_hash_is_sort_invariant():
    assert sessions_hash(["a", "b", "c"]) == sessions_hash(["c", "b", "a"])


def test_sessions_hash_empty_known_value():
    expected = hashlib.sha256(b"").hexdigest()[:16]
    assert sessions_hash([]) == expected


@pytest.mark.parametrize(
    "ids",
    [
        ["00000000deadbeef"],
        ["aaaa", "bbbb"],
        ["x", "y", "z"],
    ],
)
def test_sessions_hash_stable_across_repeats(ids: list[str]):
    assert sessions_hash(ids) == sessions_hash(list(ids))
