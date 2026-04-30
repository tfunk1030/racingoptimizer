"""Tests for damper force estimation (S4.8 / VISION §2)."""
from __future__ import annotations

import numpy as np
import pytest

from racingoptimizer.physics import (
    DIGRESSIVE_KNEE_MM_S,
    damper_coefficient,
    estimate_damper_force_n,
)
from racingoptimizer.physics.damper_force import (
    DAMPER_COEFFICIENT_NS_PER_MM,
    DEFAULT_DAMPER_COEFFICIENT_NS_PER_MM,
)


class TestDamperCoefficient:
    def test_returns_default_for_none(self):
        assert damper_coefficient(None) == DEFAULT_DAMPER_COEFFICIENT_NS_PER_MM

    def test_returns_default_for_unknown_car(self):
        assert damper_coefficient("nissan_skyline") == DEFAULT_DAMPER_COEFFICIENT_NS_PER_MM

    @pytest.mark.parametrize("car", ["bmw", "acura", "cadillac", "ferrari", "porsche"])
    def test_returns_per_car_value(self, car):
        assert damper_coefficient(car) == DAMPER_COEFFICIENT_NS_PER_MM[car]

    def test_case_insensitive(self):
        assert damper_coefficient("BMW") == damper_coefficient("bmw")
        assert damper_coefficient("Acura") == damper_coefficient("acura")

    def test_acura_differs_from_bmw(self):
        # Per-car distinctness — Acura's seeded value differs from BMW's.
        assert damper_coefficient("acura") != damper_coefficient("bmw")


class TestEstimateDamperForce:
    def test_zero_velocity_zero_force(self):
        out = estimate_damper_force_n(np.array([0.0]))
        assert out.shape == (1,)
        assert out[0] == 0.0

    def test_low_velocity_is_linear(self):
        # |v| < knee → F = k * v exactly.
        v = np.array([50.0])
        k = damper_coefficient(None)
        out = estimate_damper_force_n(v)
        assert out[0] == pytest.approx(k * 50.0)

    def test_low_velocity_per_car(self):
        v = np.array([50.0])
        k_bmw = damper_coefficient("bmw")
        out = estimate_damper_force_n(v, car="bmw")
        assert out[0] == pytest.approx(k_bmw * 50.0)

    def test_high_velocity_is_digressive(self):
        # At v >> knee, force grows sub-linearly: should be LESS than k * v.
        # The curve crosses k*v at v = 2*knee; pick well beyond that.
        v = np.array([500.0])
        k = damper_coefficient(None)
        out = estimate_damper_force_n(v)
        linear = k * 500.0
        assert 0.0 < out[0] < linear

    def test_at_knee_is_continuous(self):
        # At |v| == knee, both branches should agree at k * knee.
        # Low-speed branch is < knee strict, so test value just below + just above.
        k = damper_coefficient(None)
        below = estimate_damper_force_n(np.array([DIGRESSIVE_KNEE_MM_S - 1e-6]))
        at_or_above = estimate_damper_force_n(np.array([DIGRESSIVE_KNEE_MM_S]))
        # Both should be close to k * knee.
        assert below[0] == pytest.approx(k * DIGRESSIVE_KNEE_MM_S, rel=1e-3)
        assert at_or_above[0] == pytest.approx(k * DIGRESSIVE_KNEE_MM_S, rel=1e-3)

    def test_sign_preserved_negative(self):
        out = estimate_damper_force_n(np.array([-50.0]))
        assert out[0] < 0.0
        # Linear regime: -k * 50.
        k = damper_coefficient(None)
        assert out[0] == pytest.approx(-k * 50.0)

    def test_sign_preserved_high_velocity(self):
        out = estimate_damper_force_n(np.array([-200.0, 200.0]))
        # Symmetric magnitudes, opposite signs.
        assert out[0] < 0.0
        assert out[1] > 0.0
        assert out[0] == pytest.approx(-out[1])

    def test_acura_force_differs_from_bmw(self):
        # Per-car: same velocity, different cars → different forces.
        v = np.array([50.0])
        out_bmw = estimate_damper_force_n(v, car="bmw")
        out_acura = estimate_damper_force_n(v, car="acura")
        assert out_bmw[0] != out_acura[0]

    def test_array_shape_preserved(self):
        v = np.array([0.0, 25.0, 50.0, -75.0, 150.0, -250.0])
        out = estimate_damper_force_n(v)
        assert out.shape == v.shape
        # Spot-check signs.
        assert out[0] == 0.0
        assert out[1] > 0
        assert out[3] < 0
        assert out[5] < 0

    def test_accepts_python_list(self):
        # asarray conversion should let Python lists through.
        out = estimate_damper_force_n([0.0, 50.0])
        assert out.shape == (2,)
        assert out[0] == 0.0
