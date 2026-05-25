"""Left/right garage symmetry for DE search and setup rendering.

iRacing GTP garage UI requires LF=RF and LR=RR for camber and per-corner
dampers. The DE search optimizes the left-side (master) parameters only;
``apply_setup_symmetry`` copies each master onto its right-side slave before
scoring or rendering.
"""
from __future__ import annotations

# slave -> master (DE optimizes masters only)
DE_SYMMETRY_MIRRORS: dict[str, str] = {
    "camber_fr_deg": "camber_fl_deg",
    "camber_rr_deg": "camber_rl_deg",
    "damper_lsc_fr": "damper_lsc_fl",
    "damper_hsc_fr": "damper_hsc_fl",
    "damper_hsc_slope_fr": "damper_hsc_slope_fl",
    "damper_lsr_fr": "damper_lsr_fl",
    "damper_hsr_fr": "damper_hsr_fl",
    "damper_lsc_rr": "damper_lsc_rl",
    "damper_hsc_rr": "damper_hsc_rl",
    "damper_hsc_slope_rr": "damper_hsc_slope_rl",
    "damper_lsr_rr": "damper_lsr_rl",
    "damper_hsr_rr": "damper_hsr_rl",
}

DE_SYMMETRY_SLAVES: frozenset[str] = frozenset(DE_SYMMETRY_MIRRORS.keys())

# Per-corner static ride height readout channels checked during DE.
STATIC_RH_READOUT_CHANNELS: tuple[str, ...] = (
    "setup_static_lf_ride_height_mm",
    "setup_static_rf_ride_height_mm",
    "setup_static_lr_ride_height_mm",
    "setup_static_rr_ride_height_mm",
)

# Observation envelope from constraints.md (warn-only for post-hoc checks;
# DE uses the same bounds as a soft penalty during search).
STATIC_RH_ENVELOPE_MM: tuple[float, float] = (30.0, 80.0)

# Base penalty per violating corner plus quadratic mm overrun — large enough
# to dominate a typical hybrid score when RH is illegal (e.g. 26 mm front).
_STATIC_RH_PENALTY_BASE = 3.0
_STATIC_RH_PENALTY_PER_MM2 = 0.15


def apply_setup_symmetry(setup: dict[str, float]) -> dict[str, float]:
    """Return a copy with every slave parameter set to its master value."""
    out = dict(setup)
    for slave, master in DE_SYMMETRY_MIRRORS.items():
        if master in out:
            out[slave] = out[master]
    return out


def static_rh_platform_penalty(
    predicted_readouts: dict[str, float],
) -> float:
    """Penalty when predicted static garage RH falls outside the envelope."""
    if not predicted_readouts:
        return 0.0
    lo, hi = STATIC_RH_ENVELOPE_MM
    total = 0.0
    for channel in STATIC_RH_READOUT_CHANNELS:
        value = predicted_readouts.get(channel)
        if value is None:
            continue
        rh = float(value)
        if rh < lo:
            over = lo - rh
            total += _STATIC_RH_PENALTY_BASE + _STATIC_RH_PENALTY_PER_MM2 * over * over
        elif rh > hi:
            over = rh - hi
            total += _STATIC_RH_PENALTY_BASE + _STATIC_RH_PENALTY_PER_MM2 * over * over
    return total


def static_rh_balance_penalty(
    predicted_readouts: dict[str, float],
) -> float:
    """Penalty for large left/right imbalance in predicted static ride height.
    This catches asymmetric camber, toe, etc. that would unbalance corner weights.
    """
    if not predicted_readouts:
        return 0.0
    front_diff = abs(
        predicted_readouts.get("setup_static_lf_ride_height_mm", 0) -
        predicted_readouts.get("setup_static_rf_ride_height_mm", 0)
    )
    rear_diff = abs(
        predicted_readouts.get("setup_static_lr_ride_height_mm", 0) -
        predicted_readouts.get("setup_static_rr_ride_height_mm", 0)
    )
    # 2mm imbalance is noticeable in garage; penalize quadratically beyond that.
    imbalance = max(0.0, front_diff - 2.0) + max(0.0, rear_diff - 2.0)
    return 0.8 * imbalance * imbalance


__all__ = [
    "DE_SYMMETRY_MIRRORS",
    "DE_SYMMETRY_SLAVES",
    "STATIC_RH_ENVELOPE_MM",
    "STATIC_RH_READOUT_CHANNELS",
    "apply_setup_symmetry",
    "static_rh_platform_penalty",
    "static_rh_balance_penalty",
]
