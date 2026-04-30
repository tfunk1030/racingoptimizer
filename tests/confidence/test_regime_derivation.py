from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from racingoptimizer.confidence import Confidence


def test_construct_round_trip() -> None:
    c = Confidence(value=1.5, lo=1.4, hi=1.6, n_samples=42, regime="confident")
    assert c.value == 1.5
    assert c.lo == 1.4
    assert c.hi == 1.6
    assert c.n_samples == 42
    assert c.regime == "confident"


def test_dataclass_is_frozen() -> None:
    c = Confidence(value=1.0, lo=0.9, hi=1.1, n_samples=10, regime="sparse")
    with pytest.raises(FrozenInstanceError):
        c.value = 2.0  # type: ignore[misc]


def test_post_init_rejects_lo_greater_than_value() -> None:
    with pytest.raises(ValueError, match="lo <= value <= hi"):
        Confidence(value=1.0, lo=1.5, hi=2.0, n_samples=10, regime="confident")


def test_post_init_rejects_value_greater_than_hi() -> None:
    with pytest.raises(ValueError, match="lo <= value <= hi"):
        Confidence(value=2.5, lo=1.0, hi=2.0, n_samples=10, regime="confident")


def test_post_init_rejects_negative_n_samples() -> None:
    with pytest.raises(ValueError, match="n_samples must be >= 0"):
        Confidence(value=1.0, lo=0.9, hi=1.1, n_samples=-1, regime="sparse")


def test_post_init_rejects_unknown_regime() -> None:
    with pytest.raises(ValueError, match="regime must be one of"):
        Confidence(value=1.0, lo=0.9, hi=1.1, n_samples=10, regime="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("n_samples", "cv_residual_std", "signal_std", "expected_regime"),
    [
        (5, 0.01, 1.0, "sparse"),
        (100, 0.6, 1.0, "noisy"),
        (100, 0.3, 1.0, "confident"),
        (100, 0.1, 1.0, "dense"),
    ],
)
def test_derive_regime_table(
    n_samples: int,
    cv_residual_std: float,
    signal_std: float,
    expected_regime: str,
) -> None:
    c = Confidence.derive(
        value=1.0,
        n_samples=n_samples,
        cv_residual_std=cv_residual_std,
        signal_std=signal_std,
    )
    assert c.regime == expected_regime


def test_derive_band_is_95_pct() -> None:
    # Spec §3 pins (lo, hi) as the 95% bracket. Under a Gaussian residual
    # assumption that is value ± 1.96 * cv_residual_std on either side.
    c = Confidence.derive(value=2.0, n_samples=100, cv_residual_std=0.1, signal_std=1.0)
    assert c.lo == pytest.approx(2.0 - 1.96 * 0.1)
    assert c.hi == pytest.approx(2.0 + 1.96 * 0.1)
    assert c.value == pytest.approx(2.0)


def test_derive_rejects_negative_cv_residual_std() -> None:
    with pytest.raises(ValueError, match="cv_residual_std must be >= 0"):
        Confidence.derive(value=1.0, n_samples=100, cv_residual_std=-0.1, signal_std=1.0)


def test_derive_rejects_negative_signal_std() -> None:
    with pytest.raises(ValueError, match="signal_std must be >= 0"):
        Confidence.derive(value=1.0, n_samples=100, cv_residual_std=0.1, signal_std=-1.0)


def test_derive_handles_zero_signal_std() -> None:
    c = Confidence.derive(value=1.0, n_samples=100, cv_residual_std=0.1, signal_std=0.0)
    assert c.regime == "noisy"
