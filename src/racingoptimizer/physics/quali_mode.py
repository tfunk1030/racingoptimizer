"""Quali-stint phase weight overlay (VISION §4 / §5 short-stint regime).

A 3-lap qualifier and a full race fight different objectives. Race wants
platform consistency (tyres survive), aero efficiency averaged over many
laps, and conservative grip utilization (don't burn the rears stage 1).
Quali wants outright lap time on a single hot stint:

* push aero efficiency on straights harder (qualifying = clean air)
* push grip utilization in mid-corner harder (one-lap tyre window)
* relax platform conservatism (no need to survive 30 laps of bumps)
* keep stability + balance + traction at race weights (still must
  complete the lap cleanly)

Mirrors the wet_mode pattern: returns an adjusted ``PHASE_WEIGHTS``
table that ``aggregate_utilization`` consumes via the
``phase_weights`` override. Race mode is the no-op default — call
``quali_phase_weights()`` only when the CLI's ``--quali`` flag fires.
"""
from __future__ import annotations

from racingoptimizer.corner import Phase
from racingoptimizer.physics.phase_weights import PHASE_WEIGHTS

# Per-sub-utilization scale for quali. >1.0 amplifies the sub-util's
# contribution to the phase total, <1.0 dampens it. The new totals are
# re-normalised so each phase still sums to 1.0 (otherwise the per-corner
# weighted sum drifts).
_QUALI_SCALE: dict[str, float] = {
    "grip": 1.15,         # push the tyre window harder
    "balance": 1.0,
    "stability": 1.0,
    "traction": 1.0,
    "aero_eff": 1.20,     # outright pace on straights, clean air
    "platform": 0.55,     # less mechanical conservatism for one stint
}


def quali_phase_weights() -> dict[Phase, dict[str, float]]:
    """Return PHASE_WEIGHTS reweighted toward outright single-lap pace.

    Per-phase scaling by ``_QUALI_SCALE`` then re-normalised so each
    phase row sums to 1.0 (matching the race-mode invariant in
    ``physics.phase_weights``). Always emits a fresh dict so callers can
    safely mutate.
    """
    out: dict[Phase, dict[str, float]] = {}
    for phase, weights in PHASE_WEIGHTS.items():
        scaled: dict[str, float] = {}
        total = 0.0
        for sub, w in weights.items():
            new_w = float(w) * _QUALI_SCALE.get(sub, 1.0)
            scaled[sub] = new_w
            total += new_w
        if total <= 0.0:
            out[phase] = dict(weights)
            continue
        out[phase] = {sub: w / total for sub, w in scaled.items()}
    return out


__all__ = ["quali_phase_weights"]
