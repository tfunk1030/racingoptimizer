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

Local-density downgrade (Day 2 of physics-rebuild, Mode 4 closure):
    `Confidence.derive` answers "how good is the FITTER?" Global noise-ratio
    + sample-count is appropriate for that. The recommender, however, also
    needs "how good is the fitter AT THIS RECOMMENDED VALUE?" Even a
    well-fitted model has effectively zero training density at a parameter
    value 10 steps from any observed sample. `with_local_density(...)`
    downgrades the regime by one tier when the recommended value is more
    than `threshold_steps * step` units from the nearest observed value
    for that parameter. See PLAN.md Section 14.2.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

Regime = Literal["sparse", "noisy", "confident", "dense"]

_VALID_REGIMES: frozenset[str] = frozenset({"sparse", "noisy", "confident", "dense"})

# Two-sided Gaussian multiplier for a 95% confidence bracket.
_GAUSSIAN_95_MULTIPLIER: float = 1.96

# Regime-derivation thresholds. Promoted from inline magic numbers per
# audit Slice-10 #5. Values picked against historical fit_quality
# distributions across the BMW + Ferrari corpora; adjust together so
# the four regimes partition the [0, 1] noise_ratio range cleanly.
_SPARSE_MIN_SAMPLES: int = 30
_NOISY_NOISE_RATIO: float = 0.5
_CONFIDENT_NOISE_RATIO: float = 0.2

# Regime hierarchy from most confident -> least. Used by `downgrade(levels=N)`
# to walk the chain.
_REGIME_ORDER: tuple[Regime, ...] = ("dense", "confident", "noisy", "sparse")

# Default local-density threshold (Day 2 of physics-rebuild, Mode 4): a
# recommended value > 3 * step from the nearest observed value gets the
# regime label downgraded by one tier. 3 steps is one click "wider" than a
# typical surrogate's effective interpolation window for fittable parameters
# whose corpus values cluster within a 5-step neighbourhood.
_LOCAL_DENSITY_THRESHOLD_STEPS: float = 3.0


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
        bootstrap_std: float = 0.0,
    ) -> Confidence:
        if cv_residual_std < 0:
            raise ValueError(f"cv_residual_std must be >= 0, got {cv_residual_std!r}")
        if signal_std < 0:
            raise ValueError(f"signal_std must be >= 0, got {signal_std!r}")
        if bootstrap_std < 0:
            raise ValueError(f"bootstrap_std must be >= 0, got {bootstrap_std!r}")

        if n_samples < _SPARSE_MIN_SAMPLES:
            regime: Regime = "sparse"
        else:
            noise_ratio = cv_residual_std / max(signal_std, 1e-12)
            if noise_ratio > _NOISY_NOISE_RATIO:
                regime = "noisy"
            elif noise_ratio > _CONFIDENT_NOISE_RATIO:
                regime = "confident"
            else:
                regime = "dense"

        effective_std = max(float(cv_residual_std), float(bootstrap_std))
        half_width = _GAUSSIAN_95_MULTIPLIER * effective_std
        return cls(
            value=value,
            lo=value - half_width,
            hi=value + half_width,
            n_samples=n_samples,
            regime=regime,
        )

    def downgrade(self, *, levels: int = 1) -> Confidence:
        """Walk the regime hierarchy toward `sparse` by `levels` steps.

        `dense -> confident -> noisy -> sparse`. Already-`sparse` regime is
        the floor; further downgrades stay at `sparse`. `levels=0` is a
        no-op. Negative `levels` is rejected (use `upgrade` semantics
        explicitly if ever needed -- not currently used).
        """
        if levels < 0:
            raise ValueError(f"downgrade levels must be >= 0, got {levels!r}")
        if levels == 0:
            return self
        idx = _REGIME_ORDER.index(self.regime)
        new_idx = min(idx + levels, len(_REGIME_ORDER) - 1)
        if new_idx == idx:
            return self
        return replace(self, regime=_REGIME_ORDER[new_idx])

    def with_local_density(
        self,
        *,
        recommended: float,
        observed_values: tuple[float, ...] | list[float],
        step: float,
        threshold_steps: float = _LOCAL_DENSITY_THRESHOLD_STEPS,
    ) -> Confidence:
        """Downgrade regime by one tier if `recommended` is far from observed.

        "Far" = `min(|recommended - obs| for obs in observed_values) >
        threshold_steps * step`. The intuition: even a well-fitted surrogate
        has effectively zero training density beyond a few steps from any
        observed sample, so the global confidence label overstates trust
        in extrapolated values.

        Defensive returns (self unchanged):
        - `observed_values` empty -- caller decides whether absence of
          training data should mean sparse (the caller has the context).
        - `step <= 0` -- bad input, fail-soft rather than crash.
        - `regime` already at the floor (`sparse`) -- nothing to downgrade.

        Pin contract for Mode 4: PLAN.md Section 14.2 -- a recommended
        value > 3 step units from its nearest observed value gets regime
        label one worse than global, OR stays `noisy`/`sparse` already.
        """
        if not observed_values or step <= 0:
            return self
        if self.regime == "sparse":
            return self
        min_dist = min(abs(recommended - float(v)) for v in observed_values)
        if min_dist > threshold_steps * step:
            return self.downgrade(levels=1)
        return self
