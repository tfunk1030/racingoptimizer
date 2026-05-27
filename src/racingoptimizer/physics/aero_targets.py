"""Per-car aero RH targets and balance-lever priors.

Captures the driver/engineering setup-philosophy values that the
optimizer's scoring and the briefing's explanations should not
contradict. Source-of-truth docs live under ``docs/cars/<car>.md``.

Today this module is populated only for the Acura ARX-06 (the only
car with a published target spec in the working tree). Other cars
fall back to ``None`` -- callers MUST treat ``None`` as "no
preference, defer to the surrogate" rather than inventing defaults.

Structure intentionally narrow: aero targets the optimizer should
gravitate toward, plus a small enum describing which balance lever
to reach for first when phrasing a rebalance move. We do NOT bake
penalty weights here -- the evaluator's per-car weights live in
``physics/evaluator.py``; this module is read-only data that
explanations and the setup-justifier subagent can consult.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Acura's setup philosophy emphasises rake-driven aero balance over
# mechanical (ARB) balance for general rebalance; ARBs are the lever
# of choice ONLY when the user is happy with the aero/dynamic balance
# overall but wants more or less slow-speed rotation. Match this
# ordering when phrasing a recommendation that touches multiple
# families: lead with the aero-trim explanation, fall back to
# mechanical balance language only when the aero levers haven't moved.
BalanceLever = Literal[
    "rear_wing",            # downforce/drag trim, primary balance change
    "rear_pushrod",         # rake adjustment, aero balance without trim
    "arb_size",             # mechanical balance without aero
    "arb_blades",           # finer mechanical balance
    "diff_preload",         # entry / off-throttle rotation
    "diff_coast_drive",     # full diff (advanced)
    "diff_clutch_plates",   # full diff (advanced)
]


@dataclass(frozen=True, slots=True)
class AeroTargets:
    """Per-car aero RH targets at mid-corner / at speed.

    ``front_rh_target_mm`` / ``rear_rh_target_mm`` are the dynamic
    (at-speed) ride heights where peak downforce sits on the car's
    aero map. The optimizer's recommendations should keep the
    surrogate-predicted ``dynamic_*_rh_at_speed_mm`` channels near
    these values, modulated by the lap's drag / downforce trade-off
    (lower-both = less drag at cost of downforce).

    ``rake_orientation`` records the direction the car wants for a
    forward (oversteer) aero shift: ``"more_rake"`` means raising the
    rear RH relative to the front; ``"less_rake"`` means the
    opposite. Used by the briefing to emit the right "rake → balance"
    arrow when phrasing pushrod or perch moves.

    ``balance_levers_priority`` is the user's preferred ordering for
    rebalancing the car (see ``docs/cars/<car>.md``). The explainer
    consults this when choosing which family's move to lead the
    briefing with.
    """

    car: str
    front_rh_target_mm: float
    rear_rh_target_mm: float
    rake_orientation: Literal["more_rake", "less_rake"]
    balance_levers_priority: tuple[BalanceLever, ...] = field(default_factory=tuple)
    notes: str = ""


# Acura ARX-06 GTP. Source: docs/cars/acura_arx06.md (driver spec
# captured 2026-05-25).
ACURA_AERO_TARGETS: AeroTargets = AeroTargets(
    car="acura",
    front_rh_target_mm=15.0,
    rear_rh_target_mm=40.0,
    rake_orientation="more_rake",
    balance_levers_priority=(
        "rear_wing",
        "rear_pushrod",
        "arb_size",
        "arb_blades",
        "diff_preload",
        "diff_coast_drive",
        "diff_clutch_plates",
    ),
    notes=(
        "More rake = aero balance forward (oversteer). "
        "Front heave target ~15 mm mid-corner without bottoming; "
        "rear heave soft = less drag but loses some downforce + "
        "efficiency mid-corner. More diff preload = less off-throttle "
        "rotation, more entry stability."
    ),
)


_BY_CAR: dict[str, AeroTargets] = {
    "acura": ACURA_AERO_TARGETS,
}


def aero_targets_for(car: str) -> AeroTargets | None:
    """Return the per-car aero target spec, or ``None`` if undocumented.

    Callers MUST treat ``None`` as "defer to the surrogate". Inventing
    default targets would silently nudge the optimizer away from
    correctly-fit aero maps for the four cars whose driver spec
    hasn't been captured yet.
    """
    return _BY_CAR.get((car or "").strip().lower())


__all__ = [
    "ACURA_AERO_TARGETS",
    "AeroTargets",
    "BalanceLever",
    "aero_targets_for",
]
