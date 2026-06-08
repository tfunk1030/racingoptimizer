"""Guards for the hardened pickle loader (`physics.io`).

The production model-cache load path (`cli.recommend`) routes through
`physics.io.load` so a stale/wrong-typed/corrupt cache is rejected (and the
caller refits) rather than silently revived. See AUDIT M3.
"""
from __future__ import annotations

import pickle

import pytest

from racingoptimizer.physics import io as physics_io


def test_loads_rejects_non_model() -> None:
    """A pickle that is not a PhysicsModel must raise, not be returned."""
    with pytest.raises(TypeError):
        physics_io.loads(pickle.dumps({"not": "a model"}))


def test_load_rejects_corrupt_bytes(tmp_path) -> None:
    """A corrupt cache file must raise (caller catches and refits)."""
    bad = tmp_path / "corrupt.pickle"
    bad.write_bytes(b"\x80\x04not-a-valid-pickle-stream")
    with pytest.raises((pickle.UnpicklingError, EOFError, ValueError)):
        physics_io.load(bad)
