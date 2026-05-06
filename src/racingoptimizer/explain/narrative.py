"""Plain-English narrative renderer for setup recommendations.

Default rendering for `optimize <car> <track>` since 2026-05-06. Replaces
the per-parameter +/-1-click + score-delta blocks with a 2-3 line summary
per change, plus an OVERALL DIRECTION header. Pass `--detailed` to bring
back the legacy block-per-parameter format from
``explain.render_text.render_recommendation_text``.

The narrative re-uses the existing ``SetupJustification`` dataclass
(``corners_helped`` / ``corners_hurt`` lists) so no new physics work is
needed -- only translation: corner-phase + parameter family ->
engineering English.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from racingoptimizer.corner import Phase
from racingoptimizer.explain.justification import (
    CornerPhaseImpact,
    SetupJustification,
)
from racingoptimizer.physics.ontology import ParameterSpec, ontology_for

if TYPE_CHECKING:
    from racingoptimizer.physics import PhysicsModel, SetupRecommendation


# ---------------------------------------------------------------------------
# Translation tables
# ---------------------------------------------------------------------------

_PHASE_LABEL: dict[Phase, str] = {
    Phase.BRAKING: "under braking",
    Phase.TRAIL_BRAKE: "on trail-braking",
    Phase.MID_CORNER: "mid-corner",
    Phase.EXIT: "on power exit",
    Phase.STRAIGHT: "on straights",
}


# What gets OPTIMIZED at each phase, based on which sub-utilities
# dominate the phase-weight table. Used to phrase helps/watch in
# telemetry terms ("braking stability", "mid-corner grip") instead of
# raw corner-phase strings.
_PHASE_THEME: dict[Phase, str] = {
    Phase.BRAKING: "braking stability",
    Phase.TRAIL_BRAKE: "trail-brake balance",
    Phase.MID_CORNER: "mid-corner grip",
    Phase.EXIT: "power-down traction",
    Phase.STRAIGHT: "straight-line speed",
}


# Per-parameter (direction) tag inside parens on each CHANGES row.
# Just the direction verb — the parameter label already names the
# thing being adjusted. (verb_when_value_INCREASES, verb_when_DECREASES).
# Camber + toe values are negative-by-convention; +ve delta = less
# negative; -ve = more negative.
_DIRECTION_VERB: dict[str, tuple[str, str]] = {
    "spring_rate":   ("stiffens",        "softens"),
    "heave_spring":  ("stiffens",        "softens"),
    "heave_slider":  ("stiffens",        "softens"),
    "damper":        ("stiffens",        "softens"),
    "arb":           ("stiffens",        "softens"),
    "torsion_bar":   ("stiffens",        "softens"),
    "perch_offset":  ("lowers car",      "raises car"),
    "pushrod":       ("raises car",      "lowers car"),
    "ride_height":   ("raises",          "lowers"),
    "camber":        ("less negative",   "more negative"),
    "rear_wing":     ("more downforce",  "less drag"),
    "front_wing":    ("more downforce",  "less front"),
    "tyre_pressure": ("higher pressure", "lower pressure"),
    "brake_bias":    ("bias forward",    "bias rearward"),
    "diff":          ("tighter",         "freer"),
    "fuel":          ("heavier",         "lighter"),
    "corner_weight": ("higher",          "lower"),
}


# OVERALL DIRECTION sentence fragments per THEME. Multiple families
# can share a theme (perches + pushrods both move "ride height"); the
# theme dedup picks the net direction (up vs down delta count).
# Stored as complete noun phrases that read fluently when joined with
# "; ".
_OVERALL_FRAGMENT: dict[str, tuple[str, str]] = {
    "platform":       ("stiffer platform",          "softer platform"),
    "front platform": ("stiffer front platform",    "softer front platform"),
    "ride height":    ("higher ride height",        "lower ride height"),
    "damping":        ("stiffer damping",           "softer damping"),
    "anti-roll":      ("stiffer ARBs",              "softer ARBs"),
    "downforce":      ("more downforce",            "less drag"),
    "tire pressure":  ("higher tire pressure",      "lower tire pressure"),
    "brake bias":     ("brake bias forward",        "brake bias rearward"),
    "diff":           ("tighter diff",              "freer diff"),
    "fuel":           ("more fuel",                 "less fuel"),
    "camber":         ("less negative camber",      "more negative camber"),
    "corner weight":  ("higher corner weight",      "lower corner weight"),
}


# Car-feel narrative per (family[/mode][/axis], direction).
# Each entry is (effect_sentence, trade_sentence). Speaks in handling
# vocabulary (pitch / roll / understeer / oversteer / aero stall /
# bottoming / turn-in / throttle traction) instead of phase-aggregated
# corner lists. Matched in this order at lookup time:
#   1. (family, mode, axis, direction)   — most specific (dampers)
#   2. (family, axis, direction)          — front-vs-rear families
#   3. (family, direction)                — axle-agnostic
# A miss falls through to the terse (verb) format.
_CAR_FEEL: dict[tuple, tuple[str, str]] = {

    # --- Springs (front platform) ---
    ("heave_spring", "front", "+"): (
        "Less front pitch on brake apply; sharper turn-in; aero "
        "platform held flatter through Eau Rouge / Pouhon compression.",
        "Harsher over kerbs (T3 chicane, T16). Mid-corner front grip "
        "can drop if oversprung — watch for understeer in T9, T13.",
    ),
    ("heave_spring", "front", "-"): (
        "More front compliance over kerbs and bumps; better front "
        "grip retention through bumpy entries; smoother brake-release.",
        "More pitch dive under heavy braking; slower turn-in; aero "
        "platform less stable on Kemmel compression.",
    ),

    # Front torsion bars behave like front springs for pitch / roll,
    # but with a per-corner anti-roll component; both turns and OD
    # share the same effect axis (stiffer = stiffer).
    ("torsion_bar", "front", "+"): (
        "Stiffer front platform: less roll through fast corners, less "
        "pitch under braking, sharper transient response.",
        "Less single-wheel compliance on one-sided kerbs; "
        "mid-corner understeer risk if rear isn't matched.",
    ),
    ("torsion_bar", "front", "-"): (
        "Softer front platform: more roll compliance, gentler weight "
        "transfer, more front mechanical grip.",
        "More pitch dive on entry; slower turn-in; aero platform "
        "wanders more on bumpy entries.",
    ),

    # --- Springs (rear platform) ---
    ("spring_rate", "rear", "+"): (
        "Less rear squat under throttle; faster on-throttle response; "
        "rear platform settles less deep on Kemmel exit.",
        "Snap-oversteer risk if rear grip is exceeded on power exit; "
        "harsher over rear kerbs (T8, T15).",
    ),
    ("spring_rate", "rear", "-"): (
        "More rear compliance; smoother throttle pickup; better "
        "low-speed traction off slow corners.",
        "Rear platform settles deeper on power; bottoming risk on "
        "long throttle commits; rotation can feel lazy on entry.",
    ),
    # Rear third spring acts like a heave at the rear axle.
    ("spring_rate", "rear-third", "+"): (
        "Stiffer rear-axle heave: less rear pitch in compression, "
        "sharper rotation through high-speed corners.",
        "Less rear traction over kerbs; can promote oversteer on "
        "throttle if rear is too stiff overall.",
    ),
    ("spring_rate", "rear-third", "-"): (
        "More rear-heave compliance; better mechanical grip in "
        "low-speed corners; smoother power-down.",
        "Rear platform less stable in high-speed compression; "
        "aero-platform yaw on Kemmel exit.",
    ),

    # --- Perches (ride height direction; iRacing convention: -ve perch
    #     offset RAISES the car, +ve LOWERS). Same vocabulary front + rear.
    ("perch_offset", "front", "+"): (
        "Lower front static ride height: more downforce up front, "
        "stiffer aero platform.",
        "Bottoming risk on Kemmel compression / Eau Rouge entry; "
        "harsh over big kerbs (T3, T16).",
    ),
    ("perch_offset", "front", "-"): (
        "Higher front static ride height: more bump clearance, "
        "less aero load.",
        "Less front downforce in fast corners; more pitch sensitivity.",
    ),
    ("perch_offset", "rear", "+"): (
        "Lower rear static ride height: more rear downforce, more "
        "stable rear aero platform.",
        "Bottoming risk on T1 exit / Kemmel; rear can stall over "
        "large kerbs.",
    ),
    ("perch_offset", "rear", "-"): (
        "Higher rear static ride height: more rear compliance, "
        "less rear downforce, more rotation under throttle.",
        "Rear aero stall risk in high-speed corners; less rear "
        "stability on Kemmel compression.",
    ),
    # Pushrod has OPPOSITE polarity to perch (longer pushrod = car UP).
    ("pushrod", "front", "+"): (
        "Higher front static ride height: more bump clearance, "
        "less front aero load.",
        "Less front downforce; can stall front aero in mid-corner "
        "pitch.",
    ),
    ("pushrod", "front", "-"): (
        "Lower front static ride height: more front downforce, "
        "stiffer aero platform, sharper turn-in.",
        "Bottoming risk on Kemmel compression / Eau Rouge; harsh "
        "over kerbs.",
    ),
    ("pushrod", "rear", "+"): (
        "Higher rear static ride height: more rear bump clearance.",
        "Less rear downforce, more rear rotation under throttle.",
    ),
    ("pushrod", "rear", "-"): (
        "Lower rear static ride height: more rear downforce, more "
        "stable platform.",
        "Rear bottoming risk on Kemmel and Eau Rouge exit.",
    ),

    # --- Dampers (mode, axis, direction). LSC = brake/throttle apply,
    #     HSC = bumps + kerbs, LSR = recovery from pitch, HSR = kerb
    #     rebound, hsc_slope = transition between LS and HS regimes.
    ("damper", "lsc", "front", "+"): (
        "Sharper front response on brake apply; less initial pitch "
        "dive; quicker pitch transfer to front.",
        "Aggressive brake-release transient; trail-brake balance can "
        "feel snatchy.",
    ),
    ("damper", "lsc", "front", "-"): (
        "Smoother weight transfer to front under braking; gentler "
        "initial bite; more progressive pitch.",
        "More dive on entry; can over-rotate the front platform.",
    ),
    ("damper", "lsc", "rear", "+"): (
        "Sharper rear response on throttle apply; less initial squat.",
        "Less initial rear traction; can feel harsh on bumps under "
        "power.",
    ),
    ("damper", "lsc", "rear", "-"): (
        "Smoother power pickup; more initial rear squat absorbs the "
        "throttle hit.",
        "Rear platform sinks deeper on power; less consistent aero "
        "platform.",
    ),
    ("damper", "hsc", "front", "+"): (
        "Less compliance over bumps and kerbs; sharper kerb response.",
        "Chassis can skip across kerbs (T3, T7); aero platform less "
        "stable on bumpy entries.",
    ),
    ("damper", "hsc", "front", "-"): (
        "More compliance over kerbs and high-frequency bumps; better "
        "front grip retention through Eau Rouge bumps.",
        "More pitch noise mid-corner; aero platform wanders.",
    ),
    ("damper", "hsc", "rear", "+"): (
        "Less rear compliance over bumps; sharper kerb response.",
        "Rear can skip on kerbs and bumps; traction loss at the "
        "moment of impact.",
    ),
    ("damper", "hsc", "rear", "-"): (
        "More rear compliance over kerbs; better traction over "
        "uneven surfaces.",
        "More rear-axle pitch noise; can feel vague on long "
        "compressions.",
    ),
    ("damper", "lsr", "front", "+"): (
        "Slower front rebound after pitch — keeps weight on the "
        "front longer through entry.",
        "Front 'sticks down' after braking; can promote mid-corner "
        "understeer on T9 / T13 release.",
    ),
    ("damper", "lsr", "front", "-"): (
        "Faster front rebound; weight returns to neutral quicker; "
        "sharper rotation as the brake releases.",
        "Front can unload too fast through trail-brake; loss of "
        "front grip mid-corner.",
    ),
    ("damper", "lsr", "rear", "+"): (
        "Slower rear rebound out of the corner; keeps rear platform "
        "settled longer on power.",
        "Rear can stay too low after exit — bottoming risk on the "
        "compression that follows.",
    ),
    ("damper", "lsr", "rear", "-"): (
        "Faster rear rebound on exit; rear lifts back to neutral "
        "quicker.",
        "Rear can unload before throttle is fully applied; on-power "
        "snap-oversteer risk.",
    ),
    ("damper", "hsr", "front", "+"): (
        "Slower front rebound off kerbs and curbing.",
        "Front can hang low after a kerb hit; subsequent corner "
        "compromised.",
    ),
    ("damper", "hsr", "front", "-"): (
        "Faster front rebound off kerbs; chassis returns to neutral "
        "quicker.",
        "Aggressive bounce off kerbing; aero platform unstable in "
        "the lap after a kerb-heavy entry.",
    ),
    ("damper", "hsr", "rear", "+"): (
        "Slower rear rebound off kerbs; rear stays planted longer.",
        "Rear can stay loaded too long; subsequent corner rotation "
        "compromised.",
    ),
    ("damper", "hsr", "rear", "-"): (
        "Faster rear rebound off kerbs; rear returns to neutral "
        "quicker.",
        "Rear bounce off kerbing — traction loss on the next throttle "
        "application.",
    ),
    # HSC slope is the transition rate between LS and HS regimes.
    ("damper", "hsc_slope", "front", "+"): (
        "Earlier transition from LS to HS damping under braking — "
        "stiffer through the meat of the bump.",
        "Bump-velocity threshold lowers; chassis loses compliance "
        "earlier.",
    ),
    ("damper", "hsc_slope", "front", "-"): (
        "Later transition to HS damping; more LS regime through "
        "moderate bumps.",
        "Loses high-velocity bump control on big hits (Kemmel kink).",
    ),
    ("damper", "hsc_slope", "rear", "+"): (
        "Rear transitions to HS damping earlier — stiffer through "
        "moderate bumps and kerbs.",
        "Less rear compliance on bumpy throttle commits.",
    ),
    ("damper", "hsc_slope", "rear", "-"): (
        "Rear stays in LS damping over a wider velocity range.",
        "Loses rear bump control on big hits.",
    ),

    # --- ARBs ---
    ("arb", "front", "+"): (
        "Stiffer front anti-roll: less front roll through fast "
        "corners, sharper transient response, better aero-platform "
        "stability.",
        "More mid-corner understeer; less single-wheel compliance "
        "over one-sided kerbs.",
    ),
    ("arb", "front", "-"): (
        "Softer front anti-roll: more front mechanical grip mid-"
        "corner; better single-wheel compliance.",
        "More front roll through fast corners; aero balance shifts "
        "rearward.",
    ),
    ("arb", "rear", "+"): (
        "Stiffer rear anti-roll: faster rotation, sharper turn-in.",
        "Snap-oversteer risk on throttle and trail-brake; less rear "
        "single-wheel compliance.",
    ),
    ("arb", "rear", "-"): (
        "Softer rear anti-roll: more rear mechanical grip; better "
        "throttle traction.",
        "Lazier rotation; understeer can creep in on entry.",
    ),

    # --- Camber ---
    ("camber", "front", "+"): (  # less negative
        "Less aggressive front camber: more contact patch on "
        "straights; faster tire warm-up; less wear.",
        "Less peak mid-corner front grip; more understeer.",
    ),
    ("camber", "front", "-"): (  # more negative
        "More aggressive front camber: more peak mid-corner grip "
        "from better contact patch in roll.",
        "Less straight-line tire footprint; more wear; longer "
        "warm-up.",
    ),
    ("camber", "rear", "+"): (
        "Less aggressive rear camber: more straight-line traction; "
        "less wear.",
        "Less peak rear cornering grip; rear can lose grip on power "
        "exit.",
    ),
    ("camber", "rear", "-"): (
        "More aggressive rear camber: more peak rear cornering grip.",
        "Less straight-line traction on power; more wear.",
    ),

    # --- Toe (negative = toe-out for front, positive = toe-in;
    # the per-corner sign convention varies but the family + sign +
    # axis maps to handling effect cleanly.)
    ("camber", "toe-front", "+"):  (  # less negative, toward toe-in
        "More front straight-line stability; less darty on Kemmel.",
        "Lazier turn-in response; reduced rotation.",
    ),
    ("camber", "toe-front", "-"): (  # more negative, toward toe-out
        "More turn-in response and rotation.",
        "More straight-line nervousness; faster front tire wear.",
    ),
    ("camber", "toe-rear", "+"): (
        "More rear straight-line stability; better high-speed "
        "directional control.",
        "Lazier mid-corner rotation; more rear drag.",
    ),
    ("camber", "toe-rear", "-"): (
        "More rear rotation; better turn-in feel.",
        "Less straight-line stability; rear can step out on Kemmel "
        "transitions.",
    ),

    # --- Aero ---
    ("rear_wing", None, "+"): (
        "More downforce in fast corners (Eau Rouge, Pouhon, "
        "Blanchimont); higher mid-corner peak grip.",
        "Higher straight-line drag; ~0.05-0.10 s slower on Kemmel "
        "per click.",
    ),
    ("rear_wing", None, "-"): (
        "Less drag = faster top speed on Kemmel and Pit straight; "
        "lower fuel burn.",
        "Less downforce in fast corners; aero-stall risk in pitch; "
        "less rear stability under throttle.",
    ),

    # --- Tire pressure ---
    ("tyre_pressure", None, "+"): (
        "Higher cold pressure: less contact patch but faster heat-up "
        "and more consistent wear over a stint.",
        "Lower peak grip; tires reach optimum window slower.",
    ),
    ("tyre_pressure", None, "-"): (
        "Lower cold pressure: bigger contact patch, higher peak "
        "grip, faster warm-up.",
        "Faster wear; more sensitivity to camber; over-heating risk "
        "in long stints.",
    ),

    # --- Brake bias ---
    ("brake_bias", None, "+"): (  # forward
        "More front bias: stable braking, no rear lock-up, "
        "predictable threshold-brake behaviour.",
        "Lazy turn-in; longer braking distance; front locks first "
        "if pushed past threshold.",
    ),
    ("brake_bias", None, "-"): (  # rearward
        "More rear bias: rotates faster on entry; more turn-in "
        "feel; shorter braking distance if rear has grip.",
        "Snap-oversteer risk under heavy braking; rear can lock at "
        "T1, T11, T18 entries.",
    ),

    # --- Diff ---
    ("diff", "preload", "+"): (
        "More diff lockup at low torque: better initial throttle "
        "traction; sharper turn-in.",
        "More on-throttle understeer; rear can feel locked through "
        "mid-corner.",
    ),
    ("diff", "preload", "-"): (
        "Freer diff at low torque: easier rotation off-throttle; "
        "smoother throttle pickup.",
        "Less initial traction on power; rear can step out "
        "asymmetrically.",
    ),
    ("diff", "ramps", "+"): (
        "More aggressive coast/drive ramps: tighter diff lock under "
        "transition; sharper rotation.",
        "Throttle-off can feel locked; trail-brake balance more "
        "snappy.",
    ),
    ("diff", "ramps", "-"): (
        "Freer coast/drive ramps: smoother transitions; better "
        "low-speed throttle modulation.",
        "Less initial traction; rear can be lazy off slow corners.",
    ),
    ("diff", "plates", "+"): (
        "More clutch plates: tighter diff lock overall; better "
        "traction.",
        "More on-throttle understeer; less rotation.",
    ),
    ("diff", "plates", "-"): (
        "Fewer clutch plates: freer diff overall; easier rotation.",
        "Less peak traction; more wheelspin on slow exits.",
    ),

    # --- Fuel (mass) ---
    ("fuel", None, "+"): (
        "More fuel: more vertical load on tires (slightly more "
        "grip), but more mass to brake and accelerate.",
        "Longer braking distances; slower per-lap pace; more brake "
        "heat.",
    ),
    ("fuel", None, "-"): (
        "Less fuel: faster pace per lap, sharper transitions, less "
        "brake heat.",
        "Less downforce-to-mass ratio in high-speed corners; "
        "shorter race distance.",
    ),
}


def _car_feel(family: str, name: str, delta: float) -> tuple[str, str] | None:
    """Look up the (effect, trade) sentences for a parameter change."""
    sign = "+" if delta > 0 else "-"
    axis = _param_axis(name)
    mode = _param_damper_mode(name) if family == "damper" else None
    sub = _param_subtype(family, name)

    # 1. (family, mode, axis, direction) — dampers
    if mode and axis:
        key = (family, mode, axis, sign)
        if key in _CAR_FEEL:
            return _CAR_FEEL[key]
    # 2. (family, sub, direction) — diff modes (preload/ramps/plates),
    # rear-third spring, toe (front/rear sub-axes)
    if sub:
        key = (family, sub, sign)
        if key in _CAR_FEEL:
            return _CAR_FEEL[key]
    # 3. (family, axis, direction)
    if axis:
        key = (family, axis, sign)
        if key in _CAR_FEEL:
            return _CAR_FEEL[key]
    # 4. (family, None, direction) — single-axis families (wing, fuel,
    # brake bias, tyre pressure)
    key = (family, None, sign)
    if key in _CAR_FEEL:
        return _CAR_FEEL[key]
    return None


def _param_axis(name: str) -> str | None:
    """Classify a parameter as front / rear by inspecting the name.

    Falls back to family-implied axis for parameters that don't
    encode the axle in the name string itself (heave springs are
    front-only on the GTP cars; the third spring is rear-only).
    """
    n = name.lower()
    if any(t in n for t in ("_fl", "_fr", "front", "_f_")):
        return "front"
    if any(t in n for t in ("_rl", "_rr", "rear", "_r_", "third")):
        return "rear"
    # Family-implied axis (param name has no axle marker).
    if "heave_spring" in n:
        return "front"  # GTP heave is front-axle
    return None


def _param_damper_mode(name: str) -> str | None:
    """Pull lsc/hsc/lsr/hsr/hsc_slope from a damper parameter name."""
    n = name.lower()
    if "hsc_slope" in n:
        return "hsc_slope"
    for mode in ("lsc", "hsc", "lsr", "hsr"):
        if f"damper_{mode}_" in n:
            return mode
    return None


def _param_subtype(family: str, name: str) -> str | None:
    """Sub-bucket within a family (e.g. diff preload vs ramps vs plates)."""
    n = name.lower()
    if family == "diff":
        if "preload" in n:
            return "preload"
        if "ramp" in n or "coast" in n:
            return "ramps"
        if "clutch" in n or "plate" in n:
            return "plates"
    if family == "spring_rate":
        if "third" in n:
            return "rear-third"
    if family == "camber" and "toe" in n:
        return "toe-front" if "front" in n or "_fl" in n or "_fr" in n else "toe-rear"
    return None


# Map ParameterSpec.family -> overall theme key. Many families collapse
# into the same theme so the OVERALL paragraph doesn't fragment per
# family (e.g. perches + pushrods are both "ride height").
_FAMILY_TO_THEME: dict[str, str] = {
    "spring_rate":   "platform",
    "heave_spring":  "front platform",
    "heave_slider":  "front platform",
    "damper":        "damping",
    "arb":           "anti-roll",
    "torsion_bar":   "front platform",
    "perch_offset":  "ride height",
    "pushrod":       "ride height",
    "ride_height":   "ride height",
    "camber":        "camber",
    "rear_wing":     "downforce",
    "front_wing":    "downforce",
    "tyre_pressure": "tire pressure",
    "brake_bias":    "brake bias",
    "diff":          "diff",
    "fuel":          "fuel",
    "corner_weight": "corner weight",
}


# Per-parameter human label override.
_PARAM_LABEL: dict[str, str] = {
    "rear_wing_angle_deg":            "Rear wing",
    "tyre_cold_pressure_kpa":         "Tyre cold pressure",
    "heave_spring_rate_n_per_mm":     "Front heave spring",
    "third_spring_rate_n_per_mm":     "Rear third spring",
    "rear_coil_spring_rate_n_per_mm": "Rear coil spring",
    "heave_perch_offset_front_mm":    "Front heave perch",
    "spring_perch_offset_rear_mm":    "Rear spring perch",
    "third_perch_offset_rear_mm":     "Rear third perch",
    "pushrod_length_offset_front_mm": "Front pushrod offset",
    "pushrod_length_offset_rear_mm":  "Rear pushrod offset",
    "fuel_level_l":                   "Fuel level",
    "diff_preload_nm":                "Rear diff preload",
    "front_diff_preload_nm":          "Front diff preload",
    "diff_coast_drive_ramps":         "Diff coast/drive ramps",
    "diff_clutch_friction_plates":    "Diff clutch plates",
    "brake_bias_pct":                 "Brake bias",
    "anti_roll_bar_front":            "Front ARB blade",
    "anti_roll_bar_rear":             "Rear ARB blade",
    "arb_size_front":                 "Front ARB size",
    "arb_size_rear":                  "Rear ARB size",
    "toe_front_mm":                   "Front toe",
    "toe_rl_mm":                      "Rear toe",
    "torsion_bar_turns_fl":           "Front torsion bar turns",
    "torsion_bar_od_fl_mm":           "Front torsion bar OD",
    "torsion_bar_turns_rl":           "Rear torsion bar turns",
    "torsion_bar_od_rl_mm":           "Rear torsion bar OD",
    "camber_fl_deg":                  "FL camber",
    "camber_fr_deg":                  "FR camber",
    "camber_rl_deg":                  "RL camber",
    "camber_rr_deg":                  "RR camber",
}


# Group ordering for the briefing.
_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("AERO", ("rear_wing", "front_wing", "tyre_pressure")),
    (
        "PLATFORM (springs / perches / pushrods / torsion bars)",
        (
            "spring_rate", "heave_spring", "perch_offset", "pushrod",
            "torsion_bar", "ride_height", "fuel",
        ),
    ),
    ("DAMPERS", ("damper",)),
    ("BALANCE (ARBs / camber / toe)", ("arb", "camber")),
    ("BRAKES & DRIVETRAIN", ("brake_bias", "diff")),
)

_FAMILY_TO_GROUP: dict[str, str] = {}
for _label, _families in _GROUPS:
    for _f in _families:
        _FAMILY_TO_GROUP[_f] = _label


_REGIME_RANK = {"sparse": 0, "noisy": 1, "confident": 2, "dense": 3}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_narrative(
    rec: SetupRecommendation,
    model: PhysicsModel,
    justifications: list[SetupJustification],
    *,
    most_recent_setup: dict | str | None = None,
    track_display: str | None = None,
    quali: bool = False,
    pinned: dict[str, float] | None = None,
    warnings: list[str] | None = None,
) -> str:
    """Plain-English briefing -- every parameter that moved, with helps + watch."""
    pinned = pinned or {}
    warnings = list(warnings or [])
    onto = ontology_for(model.car)

    past_value = _resolve_past_values(justifications, onto, most_recent_setup)

    lines: list[str] = []
    lines.append("=" * 72)
    track_label = track_display or rec.track
    fuel_l = rec.parameters.get("fuel_level_l")
    fuel_str = f" ({fuel_l[0]:.1f} L fuel)" if fuel_l else ""
    mode = "quali (3-lap stint)" if quali else "race"
    lines.append(f" {model.car} @ {track_label} -- {mode}{fuel_str}")
    rolled = _rollup_regime(justifications)
    n_med = _median_n_samples(justifications)
    lines.append(
        f" Conditions: {rec.env.air_temp_c:.0f} C ambient / "
        f"{rec.env.track_temp_c:.0f} C track  |  "
        f"Confidence: {rolled} (median n={n_med})"
    )
    lines.append("=" * 72)
    lines.append("")

    # ---- OVERALL DIRECTION ----
    moved = [j for j in justifications if _moved(j, past_value, onto)]
    lines.extend(_overall_direction(moved, past_value, onto))
    lines.append("")

    # ---- CHANGES ----
    skipped = [
        j for j in justifications
        if not _moved(j, past_value, onto) and not j.pinned
    ]
    pinned_js = [j for j in justifications if j.pinned]

    lines.append(f"CHANGES ({len(moved)} of {len(justifications)} parameters moved)")
    lines.append("")

    by_group: dict[str, list[SetupJustification]] = defaultdict(list)
    for j in moved:
        spec = onto.get(j.parameter)
        family = spec.family if spec else "other"
        by_group[_FAMILY_TO_GROUP.get(family, "OTHER")].append(j)

    for group_label, _families in _GROUPS:
        if group_label not in by_group:
            continue
        lines.append(f"-- {group_label} --")
        for j in sorted(by_group[group_label], key=_param_sort_key):
            lines.extend(_render_change(j, past_value.get(j.parameter), onto))
            lines.append("")

    if "OTHER" in by_group:
        lines.append("-- OTHER --")
        for j in sorted(by_group["OTHER"], key=_param_sort_key):
            lines.extend(_render_change(j, past_value.get(j.parameter), onto))
            lines.append("")

    # ---- NOTES (pins / sparse-corpus warnings / untrained / clamp) ----
    note_lines = _notes_block(rec, justifications, pinned_js, skipped, past_value, warnings, onto)
    if note_lines:
        lines.append("NOTES")
        lines.extend(note_lines)

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _render_change(
    j: SetupJustification,
    past: float | None,
    onto: dict[str, ParameterSpec],
) -> list[str]:
    spec = onto.get(j.parameter)
    label = _PARAM_LABEL.get(j.parameter, _humanize(j.parameter))
    family = spec.family if spec else "other"
    delta = (j.value - past) if past is not None else 0.0
    direction = _direction_word(family, delta)

    change_str = _format_value_delta(j, spec, past)
    body: list[str] = [f"{label}: {change_str}  ({direction})"]

    # Car-feel narrative: Effect + Trade in handling vocabulary
    # (pitch / roll / understeer / oversteer / aero stall / bottoming
    # / turn-in / throttle traction). Falls through to phase-themed
    # corner mention if no narrative is registered for this family.
    feel = _car_feel(family, j.parameter, delta)
    if feel is not None:
        effect, trade = feel
        body.append(f"  Effect: {effect}")
        body.append(f"  Trade:  {trade}")
        # Tack on a single most-impactful corner mention so the user
        # knows where on the lap to feel it.
        impact = _dominant_impact_corner(j)
        if impact:
            body.append(f"  Watch most: {impact}")
    else:
        # No car-feel entry for this family/axis -- fall back to the
        # phase-themed list so we still say something meaningful.
        helps = _phase_phrase(j.corners_helped[:5])
        hurts = _phase_phrase(j.corners_hurt[:4])
        if helps:
            body.append(f"  Helps: {helps}")
        if hurts:
            body.append(f"  Watch: {hurts}")
        if not helps and not hurts:
            body.append("  (no per-corner trade-off -- model held this at training baseline)")
    return body


def _dominant_impact_corner(j: SetupJustification) -> str:
    """Single-line `T11 braking apex` style — the corner-phase with the
    biggest absolute score delta across helps + hurts."""
    candidates = list(j.corners_helped) + list(j.corners_hurt)
    if not candidates:
        return ""
    top = max(candidates, key=lambda i: abs(i.score_delta))
    phase_label = _PHASE_LABEL.get(top.phase, top.phase.value)
    return f"T{top.corner_id} {phase_label}"


def _format_value_delta(
    j: SetupJustification, spec: ParameterSpec | None, past: float | None,
) -> str:
    """`50 -> 60 N/mm` or `Soft -> Medium` or `60 N/mm` (no past).

    Numeric values are SNAPPED to the parameter's iRacing UI click
    step so a continuous DE result of 487.4 N/mm displays as 490
    when step=10 (matches the setup-card rendering).
    """
    units = (spec.units if spec else j.unit) or ""
    if spec and spec.choices:
        idx = max(0, min(len(spec.choices) - 1, int(round(j.value))))
        new_label = spec.choices[idx]
        if past is not None:
            past_idx = max(0, min(len(spec.choices) - 1, int(round(past))))
            old_label = spec.choices[past_idx]
            if old_label == new_label:
                return new_label
            return f"{old_label} -> {new_label}"
        return new_label

    step = (spec.step if spec and spec.step else 0.5)
    snapped_val = _snap(j.value, step, spec)
    snapped_past = _snap(past, step, spec) if past is not None else None
    if step >= 1.0:
        fmt = lambda v: f"{v:.0f}"  # noqa: E731
    elif step >= 0.1:
        fmt = lambda v: f"{v:.1f}"  # noqa: E731
    elif step >= 0.01:
        fmt = lambda v: f"{v:.2f}"  # noqa: E731
    else:
        # Torsion bar turns step 0.001 -- 3 decimals so 0.105 doesn't
        # render as 0.10 (loses a digit the user actually has to enter).
        fmt = lambda v: f"{v:.3f}"  # noqa: E731

    val_s = fmt(snapped_val).rstrip()
    unit_s = f" {units}" if units else ""
    if snapped_past is None:
        return f"{val_s}{unit_s}"
    past_s = fmt(snapped_past).rstrip()
    if past_s == val_s:
        return f"{val_s}{unit_s}"
    return f"{past_s} -> {val_s}{unit_s}"


def _snap(value: float, step: float | None, spec: ParameterSpec | None) -> float:
    """Snap to non-uniform discrete_values if set, else uniform step."""
    if spec and spec.discrete_values:
        return min(spec.discrete_values, key=lambda c: abs(c - value))
    if step is None or step <= 0.0:
        return value
    return round(value / step) * step


def _direction_word(family: str, delta: float) -> str:
    if abs(delta) < 1e-9:
        return "no change"
    if family in _DIRECTION_VERB:
        up, down = _DIRECTION_VERB[family]
        return up if delta > 0 else down
    return "increases" if delta > 0 else "decreases"


def _phase_phrase(impacts: tuple[CornerPhaseImpact, ...]) -> str:
    """Telemetry-themed phrase: `mid-corner grip in T9, T13; braking stability in T3`."""
    if not impacts:
        return ""
    by_phase: dict[Phase, list[int]] = defaultdict(list)
    for imp in impacts:
        by_phase[imp.phase].append(imp.corner_id)
    parts: list[str] = []
    for phase in (
        Phase.BRAKING, Phase.TRAIL_BRAKE, Phase.MID_CORNER,
        Phase.EXIT, Phase.STRAIGHT,
    ):
        if phase not in by_phase:
            continue
        corners = sorted(set(by_phase[phase]))
        corner_str = ", ".join(f"T{c}" for c in corners)
        theme = _PHASE_THEME.get(phase, phase.value)
        parts.append(f"{theme} in {corner_str}")
    return "; ".join(parts)


def _overall_direction(
    moved: list[SetupJustification],
    past_value: dict[str, float | None],
    onto: dict[str, ParameterSpec],
) -> list[str]:
    """One paragraph summarising the dominant direction per THEME.

    Multiple families share a theme (perches + pushrods both move
    "ride height"); collapse on the theme so the paragraph doesn't
    fragment. The displayed direction is the net (up vs down delta
    count) per theme. Mixed-direction themes get a "mixed" annotation.
    """
    if not moved:
        return [
            "OVERALL DIRECTION",
            "  No changes from past setup -- already optimal in the model's view.",
        ]
    theme_dirs: dict[str, dict[str, int]] = defaultdict(
        lambda: {"up": 0, "down": 0},
    )
    for j in moved:
        spec = onto.get(j.parameter)
        family = spec.family if spec else "other"
        theme = _FAMILY_TO_THEME.get(family)
        if theme is None:
            continue
        past = past_value.get(j.parameter)
        if past is None:
            continue
        if j.value > past:
            theme_dirs[theme]["up"] += 1
        else:
            theme_dirs[theme]["down"] += 1

    summaries: list[str] = []
    for theme, dirs in theme_dirs.items():
        up_n = dirs["up"]
        down_n = dirs["down"]
        fragment = _OVERALL_FRAGMENT.get(theme)
        if fragment is None:
            continue
        up_phrase, down_phrase = fragment
        if up_n > down_n:
            summaries.append(up_phrase)
        elif down_n > up_n:
            summaries.append(down_phrase)
        else:
            summaries.append(f"mixed {theme} adjustments")

    sentence = (
        "; ".join(summaries) + "."
        if summaries
        else f"Adjusts {len(moved)} parameter(s) without a single dominant theme."
    )
    return [
        "OVERALL DIRECTION",
        f"  {sentence[0].upper()}{sentence[1:]}",
    ]


def _notes_block(
    rec: SetupRecommendation,
    justifications: list[SetupJustification],
    pinned_js: list[SetupJustification],
    skipped: list[SetupJustification],
    past_value: dict[str, float | None],
    warnings: list[str],
    onto: dict[str, ParameterSpec],
) -> list[str]:
    out: list[str] = []
    for j in pinned_js:
        label = _PARAM_LABEL.get(j.parameter, _humanize(j.parameter))
        spec = onto.get(j.parameter)
        val_s = _format_value_delta(j, spec, None)
        out.append(f"  {label}: pinned by user at {val_s}")
    # Truly unchanged params already implicit; only mention "held at past" for params
    # the user might have expected to move
    for j in skipped:
        past = past_value.get(j.parameter)
        if past is None:
            continue
        spec = onto.get(j.parameter)
        step = (spec.step if spec and spec.step else 0.5)
        if abs(j.value - past) >= step / 2:
            continue  # actually moved by a half-step or more -- covered above
    if rec.pinned_to_observed_median:
        for name in sorted(rec.pinned_to_observed_median):
            label = _PARAM_LABEL.get(name, _humanize(name))
            out.append(
                f"  {label}: corpus has only one value -- vary it next session "
                f"for a recommendation"
            )
    if rec.untrained_parameters:
        params_str = ", ".join(sorted(rec.untrained_parameters))
        out.append(f"  Untrained (constraints.md TODO): {params_str}")
    for w in warnings:
        out.append(f"  ! {w}")
    return out


def _resolve_past_values(
    justifications: list[SetupJustification],
    onto: dict[str, ParameterSpec],
    most_recent_setup: dict | str | None,
) -> dict[str, float | None]:
    setup = _coerce_setup(most_recent_setup)
    out: dict[str, float | None] = {}
    if setup is None:
        for j in justifications:
            out[j.parameter] = None
        return out
    from racingoptimizer.explain.full_setup_card import _scalar_from_yaml
    for j in justifications:
        spec = onto.get(j.parameter)
        if spec is None:
            out[j.parameter] = None
            continue
        cur: dict | str | float | int | None = setup
        for seg in spec.json_path:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(seg)
        if cur is None:
            out[j.parameter] = None
            continue
        if spec.choices and isinstance(cur, str):
            raw_norm = cur.strip().lower()
            for idx, label in enumerate(spec.choices):
                if label.strip().lower() == raw_norm:
                    out[j.parameter] = float(idx)
                    break
            else:
                out[j.parameter] = None
        else:
            out[j.parameter] = _scalar_from_yaml(cur)
    return out


def _coerce_setup(setup: dict | str | None) -> dict | None:
    if setup is None:
        return None
    if isinstance(setup, str):
        import json
        try:
            return json.loads(setup)
        except Exception:
            return None
    return setup


def _moved(
    j: SetupJustification,
    past_value: dict[str, float | None],
    onto: dict[str, ParameterSpec],
) -> bool:
    """A parameter "moved" only if its STEP-SNAPPED value differs from past.

    Compares the values the user would actually enter in the iRacing UI
    (snapped to discrete_values or step), not the raw DE output. Without
    this, a torsion-bar turn change of 0.108 -> 0.105 (delta 0.003 with
    step 0.125) would render as "0.10 turns (stiffens)" — same displayed
    value but a "stiffens" verb that contradicts the no-change display.
    """
    if j.pinned:
        return False
    past = past_value.get(j.parameter)
    if past is None:
        return False
    spec = onto.get(j.parameter)
    step = (spec.step if spec and spec.step else 0.5)
    snapped_new = _snap(j.value, step, spec)
    snapped_past = _snap(past, step, spec)
    return abs(snapped_new - snapped_past) > 1e-9


def _param_sort_key(j: SetupJustification) -> tuple[int, str]:
    """Sort within a group: largest absolute trade-off first, then name."""
    helped = sum(abs(i.score_delta) for i in j.corners_helped)
    hurt = sum(abs(i.score_delta) for i in j.corners_hurt)
    return (-int((helped + hurt) * 1e6), j.parameter)


def _humanize(name: str) -> str:
    return name.replace("_", " ").title()


def _rollup_regime(justifications: list[SetupJustification]) -> str:
    if not justifications:
        return "sparse"
    worst_rank = min(
        _REGIME_RANK[j.confidence.regime] for j in justifications
    )
    for k, v in _REGIME_RANK.items():
        if v == worst_rank:
            return k
    return "sparse"


def _median_n_samples(justifications: list[SetupJustification]) -> int:
    if not justifications:
        return 0
    counts = sorted(j.confidence.n_samples for j in justifications)
    return counts[len(counts) // 2]


__all__ = ["render_narrative"]
