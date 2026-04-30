"""Unit tests for the per-car steering-geometry coefficient (S2.10).

VISION §3 forbids the textbook bicycle-model `Speed^2` denominator that the
old understeer formula used. S2.10 replaces it with a per-car linear
yaw-deficiency proxy and exposes the lookup helpers tested here:

- :data:`STEERING_GEOMETRY_COEFFICIENT` — per-car table.
- :data:`DEFAULT_STEERING_GEOMETRY_COEFFICIENT` — fallback.
- :func:`steering_geometry_for` — case-insensitive resolver.
"""
from __future__ import annotations

import pytest

from racingoptimizer.corner.states import (
    DEFAULT_STEERING_GEOMETRY_COEFFICIENT,
    STEERING_GEOMETRY_COEFFICIENT,
    steering_geometry_for,
)

CANONICAL_CARS = ("bmw", "acura", "cadillac", "ferrari", "porsche")


def test_default_is_floating_point_in_expected_band() -> None:
    """Anchor: the default sits in the same order-of-magnitude band as the
    per-car seeds (rad / m·s⁻²)."""
    assert 0.04 <= DEFAULT_STEERING_GEOMETRY_COEFFICIENT <= 0.10


@pytest.mark.parametrize("car", CANONICAL_CARS)
def test_seeded_table_covers_every_canonical_car(car: str) -> None:
    """All five GTP cars must have a seeded coefficient — no silent fallback."""
    assert car in STEERING_GEOMETRY_COEFFICIENT
    val = STEERING_GEOMETRY_COEFFICIENT[car]
    assert 0.04 <= val <= 0.10, f"{car}: {val} outside expected GTP band"


@pytest.mark.parametrize("car", CANONICAL_CARS)
def test_steering_geometry_for_returns_seeded_value(car: str) -> None:
    assert steering_geometry_for(car) == STEERING_GEOMETRY_COEFFICIENT[car]


@pytest.mark.parametrize("car", CANONICAL_CARS)
def test_steering_geometry_for_is_case_insensitive(car: str) -> None:
    assert steering_geometry_for(car.upper()) == STEERING_GEOMETRY_COEFFICIENT[car]
    assert steering_geometry_for(car.title()) == STEERING_GEOMETRY_COEFFICIENT[car]


def test_steering_geometry_for_strips_whitespace() -> None:
    assert steering_geometry_for(" bmw ") == STEERING_GEOMETRY_COEFFICIENT["bmw"]


@pytest.mark.parametrize(
    "raw,canonical",
    [
        ("bmwlmdh", "bmw"),
        ("acuraarx06gtp", "acura"),
        ("cadillacvseriesrgtp", "cadillac"),
        ("ferrari499p", "ferrari"),
        ("porsche963gtp", "porsche"),
    ],
)
def test_steering_geometry_for_normalises_raw_iracing_identifiers(
    raw: str, canonical: str
) -> None:
    """Should also work when handed the raw IBT/filename car identifier."""
    assert steering_geometry_for(raw) == STEERING_GEOMETRY_COEFFICIENT[canonical]


@pytest.mark.parametrize("bad", [None, "", "   "])
def test_steering_geometry_for_falls_back_on_empty_input(bad: str | None) -> None:
    assert steering_geometry_for(bad) == DEFAULT_STEERING_GEOMETRY_COEFFICIENT


@pytest.mark.parametrize("bad", ["unknown_car", "tesla_roadster", "f1_2026"])
def test_steering_geometry_for_falls_back_on_unknown_car(bad: str) -> None:
    assert steering_geometry_for(bad) == DEFAULT_STEERING_GEOMETRY_COEFFICIENT
