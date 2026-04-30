"""Confidence dataclass with regime derivation.

Born by slice E (physics fitter, spec §3) as a cross-cutting type. Every
predicted physics-state value travels paired with a Confidence so downstream
consumers (recommender, justification renderer) can phrase aggression vs
caution based on data density and fit quality.

Bracket convention (`lo`, `hi`):
    The physics-fitter spec (`docs/superpowers/specs/2026-04-28-physics-fitter-design.md`
    §3) pins `lo`, `hi` as the **95% bracket** around `value`. `Confidence.derive`
    realises this by widening the K-fold residual std `cv_residual_std` by 1.96
    (the Gaussian 95% multiplier) on either side of `value`. This matches the
    spec's "empirical 2.5/97.5 percentiles of (predicted ± residuals)" promise
    under a Gaussian-residual assumption — the same assumption already implicit
    in summarising fit quality with a single residual std rather than the full
    residual distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Regime = Literal["sparse", "noisy", "confident", "dense"]

_VALID_REGIMES: frozenset[str] = frozenset({"sparse", "noisy", "confident", "dense"})

# Two-sided Gaussian multiplier for a 95% confidence bracket.
_GAUSSIAN_95_MULTIPLIER: float = 1.96


@dataclass(slots=True, frozen=True)
class Confidence:
    value: float
    lo: float
    hi: float
    n_samples: int
    regime: Regime

    def __post_init__(self) -> None:
        if not (self.lo <= self.value <= self.hi):
            raise ValueError(
                f"Confidence requires lo <= value <= hi; got "
                f"lo={self.lo!r}, value={self.value!r}, hi={self.hi!r}"
            )
        if self.n_samples < 0:
            raise ValueError(f"n_samples must be >= 0, got {self.n_samples!r}")
        if self.regime not in _VALID_REGIMES:
            raise ValueError(
                f"regime must be one of {sorted(_VALID_REGIMES)}, got {self.regime!r}"
            )

    @classmethod
    def derive(
        cls,
        *,
        value: float,
        n_samples: int,
        cv_residual_std: float,
        signal_std: float,
    ) -> Confidence:
        if cv_residual_std < 0:
            raise ValueError(f"cv_residual_std must be >= 0, got {cv_residual_std!r}")
        if signal_std < 0:
            raise ValueError(f"signal_std must be >= 0, got {signal_std!r}")

        if n_samples < 30:
            regime: Regime = "sparse"
        else:
            noise_ratio = cv_residual_std / max(signal_std, 1e-12)
            if noise_ratio > 0.5:
                regime = "noisy"
            elif noise_ratio > 0.2:
                regime = "confident"
            else:
                regime = "dense"

        half_width = _GAUSSIAN_95_MULTIPLIER * cv_residual_std
        return cls(
            value=value,
            lo=value - half_width,
            hi=value + half_width,
            n_samples=n_samples,
            regime=regime,
        )
