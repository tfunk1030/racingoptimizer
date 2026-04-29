"""Pickle helpers for `PhysicsModel`.

The frozen dataclass + sklearn fitters + numpy arrays + ConstraintsTable
(also a frozen dataclass) are all picklable, so `pickle.dumps/loads` works
out of the box. These thin helpers keep the call site explicit and exist
so the determinism check in tests has a single import path.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from racingoptimizer.physics.model import PhysicsModel


def dumps(model: PhysicsModel) -> bytes:
    return pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)


def loads(data: bytes) -> PhysicsModel:
    obj = pickle.loads(data)  # noqa: S301 — trusted offline artefact
    if not isinstance(obj, PhysicsModel):
        raise TypeError(f"expected PhysicsModel, got {type(obj).__name__}")
    return obj


def save(model: PhysicsModel, path: Path | str) -> None:
    Path(path).write_bytes(dumps(model))


def load(path: Path | str) -> PhysicsModel:
    return loads(Path(path).read_bytes())


__all__ = ["dumps", "load", "loads", "save"]
