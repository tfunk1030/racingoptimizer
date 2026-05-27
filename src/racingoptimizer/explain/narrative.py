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

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from racingoptimizer.constraints import load_constraints
from racingoptimizer.corner import CornerPhaseKey, Phase
from racingoptimizer.explain.justification import (
    CornerPhaseImpact,
    SetupJustification,
)
from racingoptimizer.physics.ontology import ParameterSpec, ontology_for

if TYPE_CHECKING:
    from racingoptimizer.physics import PhysicsModel, SetupRecommendation


# ---------------------------------------------------------------------------
# P3.1 -- per-channel held-out error budget surfaced in the briefing header
# ---------------------------------------------------------------------------

# Keep aligned with ``scripts/holdout_accuracy_gate.py::_PER_CHANNEL_THRESHOLDS``.
# Maps the gate JSON's channel key to ``(label, unit)`` for the briefing
# header. Channels not in this dict aren't surfaced in the header even if
# the gate scores them (they may still drive the gate's pass/fail).
_HEADER_ERROR_BUDGET_CHANNELS: tuple[tuple[str, str, str], ...] = (
    # (channel, label, unit)
    ("accel_lat_g_max", "peak lateral G", "g"),
    ("understeer_angle_mean_rad", "understeer angle", "rad"),
    ("setup_static_lf_ride_height_mm", "static front RH", "mm"),
    ("damper_force_p99_n", "damper force p99", "N"),
)


def _holdout_accuracy_path() -> Path:
    return Path("docs/physics-rebuild/holdout_accuracy_latest.json")


def _load_holdout_rows_for(car: str, track: str) -> list[dict] | None:
    """Return the per-channel rows for ``(car, track)`` from the
    held-out gate JSON, or None if the file is missing, malformed, or
    has no matching row.

    The gate runs on a per-(car, held-out track) pair, so the briefing
    header surfaces the closest match: same car AND same track. When
    the user is recommending for a track the held-out gate doesn't
    cover, the header falls back to the legacy ``Confidence: ...`` line.
    """
    path = _holdout_accuracy_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list):
        return None
    car_norm = (car or "").lower()
    track_norm = (track or "").lower()
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if (
            str(entry.get("car", "")).lower() == car_norm
            and str(entry.get("track", "")).lower() == track_norm
            and isinstance(entry.get("channels"), list)
        ):
            return entry["channels"]
    return None


def _format_error_budget_value(channel: str, mean_abs: float, unit: str) -> str:
    """Render ``+/- X.XX unit`` with channel-aware precision."""
    if not math.isfinite(mean_abs):
        return ""
    if channel == "damper_force_p99_n" or unit == "N":
        return f"+/- {mean_abs:.0f} {unit}"
    if abs(mean_abs) >= 10:
        return f"+/- {mean_abs:.1f} {unit}"
    return f"+/- {mean_abs:.2f} {unit}"


def _render_error_budget_block(car: str, track: str) -> list[str]:
    """Build the ``Predicted error on this car/track (held-out): ...``
    block. Returns a list of lines. Empty list when no held-out row
    matches.
    """
    rows = _load_holdout_rows_for(car, track)
    if not rows:
        return []
    by_channel = {row.get("channel"): row for row in rows if isinstance(row, dict)}
    lines: list[str] = ["Predicted error on this car/track (held-out):"]
    rendered_any = False
    for channel, label, unit in _HEADER_ERROR_BUDGET_CHANNELS:
        row = by_channel.get(channel)
        if not row:
            continue
        try:
            mean_abs = float(row.get("mean_abs"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(mean_abs):
            continue
        value_str = _format_error_budget_value(channel, mean_abs, unit)
        if not value_str:
            continue
        regime = str(row.get("regime") or "").strip()
        regime_str = f" ({regime})" if regime else ""
        # Pad label to 24 chars so columns line up across rows.
        lines.append(f"  {label:<24} {value_str}{regime_str}")
        rendered_any = True
    if not rendered_any:
        return []
    return lines


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
# Just the direction verb -- the parameter label already names the
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
#   1. (family, mode, axis, direction)   -- most specific (dampers)
#   2. (family, axis, direction)          -- front-vs-rear families
#   3. (family, direction)                -- axle-agnostic
# A miss falls through to the terse (verb) format.
_CAR_FEEL: dict[tuple, tuple[str, str]] = {

    # --- Springs (front-axle heave) ---
    # `heave_spring_rate_n_per_mm` has family="spring_rate"; the
    # subtype lookup ("front-heave") is what routes here. Don't be
    # tempted to key on family="heave_spring" -- that's a readout
    # family with no user-settable parameter.
    ("spring_rate", "front-heave", "+"): (
        "Less front pitch on brake apply; sharper turn-in; aero "
        "platform held flatter on long compressions.",
        "Harsher over kerbs. Mid-corner front grip can drop if "
        "oversprung -- watch for understeer.",
    ),
    ("spring_rate", "front-heave", "-"): (
        "More front compliance over kerbs and bumps; better front "
        "grip retention through bumpy entries; smoother brake-release.",
        "More pitch dive under heavy braking; slower turn-in; aero "
        "platform less stable on long compressions.",
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
    # Rear torsion bars behave like rear springs for pitch / roll, with a
    # per-corner anti-roll component on cars that have rear torsion bars
    # (Ferrari all-4 and several GTP rears). Both turns and OD share the
    # stiffer-direction axis.
    ("torsion_bar", "rear", "+"): (
        "Stiffer rear platform: less rear roll, less squat under "
        "throttle, sharper rotation through high-speed corners.",
        "Less rear single-wheel compliance over kerbs; snap-oversteer "
        "risk if rear grip is exceeded on power.",
    ),
    ("torsion_bar", "rear", "-"): (
        "Softer rear platform: more rear mechanical grip, smoother "
        "throttle pickup, better traction off slow corners.",
        "More squat on power; rear platform sinks deeper; rotation "
        "can feel lazy on entry.",
    ),

    # --- Ride height (catch-all when the optimizer changes ride height
    #     directly rather than via perch / pushrod). Same vocabulary as
    #     the perch/pushrod families, normalised to height direction:
    #     "+" = raises, "-" = lowers (matches `_DIRECTION_VERB`).
    ("ride_height", "front", "+"): (
        "Higher front static ride height: more bump clearance, less "
        "front aero load.",
        "Less front downforce in fast corners; more pitch sensitivity.",
    ),
    ("ride_height", "front", "-"): (
        "Lower front static ride height: more downforce up front, "
        "stiffer aero platform, sharper turn-in.",
        "Bottoming risk on long compressions and high-speed crests; "
        "harsh over kerbs.",
    ),
    ("ride_height", "rear", "+"): (
        "Higher rear static ride height: more rear compliance, less "
        "rear downforce, more rotation under throttle.",
        "Rear aero stall risk in high-speed corners; less rear "
        "stability through long compressions.",
    ),
    ("ride_height", "rear", "-"): (
        "Lower rear static ride height: more rear downforce, more "
        "stable rear aero platform.",
        "Bottoming risk on heavy-throttle exits; rear can stall over "
        "large kerbs.",
    ),

    # --- Corner weight (cross-weight redistribution). Front / rear axle
    #     bias from individual corner weights; same vocabulary as the
    #     other axle-bias families.
    ("corner_weight", "front", "+"): (
        "More weight on the front axle: more front mechanical grip "
        "static, sharper turn-in feel.",
        "Less rear traction off corners; can promote lift-off "
        "oversteer on entry.",
    ),
    ("corner_weight", "front", "-"): (
        "Less weight on the front axle: more rear traction static; "
        "smoother power-down.",
        "Mid-corner understeer risk; lazier turn-in.",
    ),
    ("corner_weight", "rear", "+"): (
        "More weight on the rear axle: more rear traction static; "
        "smoother power-down out of slow corners.",
        "Mid-corner understeer; lazier transient response.",
    ),
    ("corner_weight", "rear", "-"): (
        "Less weight on the rear axle: sharper rotation, more turn-in "
        "feel.",
        "Less rear traction on power exit; can promote snap-oversteer.",
    ),

    # --- Springs (rear platform) ---
    ("spring_rate", "rear", "+"): (
        "Less rear squat under throttle; faster on-throttle response; "
        "rear platform settles less deep on long throttle commits.",
        "Snap-oversteer risk if rear grip is exceeded on power exit; "
        "harsher over rear-loaded kerbs.",
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
        "aero-platform yaw on long throttle commits.",
    ),

    # --- Perches (ride height direction; iRacing convention: -ve perch
    #     offset RAISES the car, +ve LOWERS). Same vocabulary front + rear.
    ("perch_offset", "front", "+"): (
        "Lower front static ride height: more downforce up front, "
        "stiffer aero platform.",
        "Bottoming risk on long compressions and high-speed crests; "
        "harsh over high-load kerbs.",
    ),
    ("perch_offset", "front", "-"): (
        "Higher front static ride height: more bump clearance, "
        "less aero load.",
        "Less front downforce in fast corners; more pitch sensitivity.",
    ),
    ("perch_offset", "rear", "+"): (
        "Lower rear static ride height: more rear downforce, more "
        "stable rear aero platform.",
        "Bottoming risk on heavy-throttle exits and long straights; "
        "rear can stall over large kerbs.",
    ),
    ("perch_offset", "rear", "-"): (
        "Higher rear static ride height: more rear compliance, "
        "less rear downforce, more rotation under throttle.",
        "Rear aero stall risk in high-speed corners; less rear "
        "stability through long compressions.",
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
        "Bottoming risk on long compressions and high-speed crests; "
        "harsh over kerbs.",
    ),
    ("pushrod", "rear", "+"): (
        "Higher rear static ride height: more rear bump clearance.",
        "Less rear downforce, more rear rotation under throttle.",
    ),
    ("pushrod", "rear", "-"): (
        "Lower rear static ride height: more rear downforce, more "
        "stable platform.",
        "Rear bottoming risk on long throttle commits and high-speed "
        "compressions.",
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
        "Chassis can skip across kerbs on entry; aero platform less "
        "stable on bumpy entries.",
    ),
    ("damper", "hsc", "front", "-"): (
        "More compliance over kerbs and high-frequency bumps; better "
        "front grip retention through high-speed bumps.",
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
        "Slower front rebound after pitch -- keeps weight on the "
        "front longer through entry.",
        "Front 'sticks down' after braking; can promote mid-corner "
        "understeer on long mid-corner releases.",
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
        "Rear can stay too low after exit -- bottoming risk on the "
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
        "Rear bounce off kerbing -- traction loss on the next throttle "
        "application.",
    ),
    # HSC slope is the transition rate between LS and HS regimes.
    ("damper", "hsc_slope", "front", "+"): (
        "Earlier transition from LS to HS damping under braking -- "
        "stiffer through the meat of the bump.",
        "Bump-velocity threshold lowers; chassis loses compliance "
        "earlier.",
    ),
    ("damper", "hsc_slope", "front", "-"): (
        "Later transition to HS damping; more LS regime through "
        "moderate bumps.",
        "Loses high-velocity bump control on big hits (high-speed kinks).",
    ),
    ("damper", "hsc_slope", "rear", "+"): (
        "Rear transitions to HS damping earlier -- stiffer through "
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
        "More front straight-line stability; less darty on long straights.",
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
        "Less straight-line stability; rear can step out on high-speed "
        "transitions.",
    ),

    # --- Aero ---
    ("rear_wing", None, "+"): (
        "More downforce in fast corners (high-speed sweepers); "
        "higher mid-corner peak grip.",
        "Higher straight-line drag; ~0.05-0.10 s slower per click on "
        "long straights.",
    ),
    ("rear_wing", None, "-"): (
        "Less drag = faster top speed on long straights; lower fuel "
        "burn.",
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
        "heavy-braking entries.",
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

    # 1. (family, mode, axis, direction) -- dampers
    if mode and axis:
        key = (family, mode, axis, sign)
        if key in _CAR_FEEL:
            return _CAR_FEEL[key]
    # 2. (family, sub, direction) -- diff modes (preload/ramps/plates),
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
    # 4. (family, None, direction) -- single-axis families (wing, fuel,
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
    """Pull lsc/hsc/lsr/hsr/hsc_slope/roll_* from a damper parameter name."""
    n = name.lower()
    if "hsc_slope" in n:
        return "hsc_slope"
    for mode in ("roll_lsc", "roll_hsc"):
        if f"damper_{mode}_" in n:
            return mode
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
        if "heave" in n:
            return "front-heave"
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
    "traction_control_gain":          "TC1 gain",
    "traction_control_slip":          "TC2 slip",
    "damper_roll_lsc_front":          "Front roll damper LSC",
    "damper_roll_hsc_front":          "Front roll damper HSC",
    "damper_roll_lsc_rear":           "Rear roll damper LSC",
    "damper_roll_hsc_rear":           "Rear roll damper HSC",
    "camber_fl_deg":                  "Front camber",
    "camber_fr_deg":                  "Front camber",
    "camber_rl_deg":                  "Rear camber",
    "camber_rr_deg":                  "Rear camber",
}


def _param_label(name: str, car: str | None = None) -> str:
    """Human label for a parameter, with per-car garage UI overrides."""
    if (car or "").lower() == "acura" and name == "spring_perch_offset_rear_mm":
        return "Rear heave perch"
    return _PARAM_LABEL.get(name, _humanize(name))


def _track_coverage_headline(model: PhysicsModel, track: str) -> str | None:
    """One-line summary of within-track parameter exploration."""
    per_track = getattr(model, "per_track_parameter_observed", {}) or {}
    track_obs = per_track.get(track, {})
    if not track_obs:
        return None
    n_covered = sum(1 for vals in track_obs.values() if len(set(vals)) >= 3)
    n_total = len(track_obs)
    pct = int(round(100.0 * n_covered / n_total)) if n_total else 0
    return (
        f"Track coverage: {n_covered}/{n_total} parameters with 3+ distinct "
        f"values ({pct}% explored at {track})"
    )


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
    schedule: list | None = None,
) -> str:
    """Plain-English briefing -- every parameter that moved, with helps + watch."""
    pinned = pinned or {}
    warnings = list(warnings or [])
    onto = ontology_for(model.car)

    past_value = _resolve_past_values(justifications, onto, most_recent_setup)
    # Telemetry-evidence cache: predicted output-channel values at the
    # past-setup counterfactual for each (parameter, dominant Helps
    # cpkey) tuple. Computed lazily inside _render_change.
    rec_setup = {name: float(v) for name, (v, _c) in rec.parameters.items()}

    # P0.3: parameters where the surrogate cannot resolve +/-1 click
    # are reset to baseline by ``recommend()`` and listed in
    # ``rec.suppressed_below_sensitivity``. They MUST NOT appear in the
    # "moved" block -- baseline_setup (training median) may still
    # differ from past_setup (most recent IBT), which would otherwise
    # render them as a phantom move. Surfaced under NOTES instead.
    suppressed_names = set(
        getattr(rec, "suppressed_below_sensitivity", ()) or ()
    )

    moved = [
        j for j in justifications
        if _moved(j, past_value, onto)
        and j.parameter not in suppressed_names
    ]
    moved_names = {j.parameter for j in moved}

    lines: list[str] = []
    lines.append("=" * 72)
    track_label = track_display or rec.track
    fuel_l = rec.parameters.get("fuel_level_l")
    fuel_str = f" ({fuel_l[0]:.1f} L fuel)" if fuel_l else ""
    mode = "quali (3-lap stint)" if quali else "race"
    lines.append(f" {model.car} @ {track_label} -- {mode}{fuel_str}")
    from racingoptimizer.physics.aero_targets import aero_targets_for

    targets = aero_targets_for(model.car)
    if targets is not None:
        lines.append(
            f" Aero targets (at speed): front ~{targets.front_rh_target_mm:.0f} mm, "
            f"rear ~{targets.rear_rh_target_mm:.0f} mm peak downforce "
            "(see docs/cars/acura_arx06.md)."
        )
    rolled = _rollup_regime(justifications)
    n_med = _median_n_samples(justifications)
    sparse_moved = sum(
        1 for name, (_v, conf) in rec.parameters.items()
        if conf.regime == "sparse" and name in moved_names
    )
    noisy_moved = sum(
        1 for name, (_v, conf) in rec.parameters.items()
        if conf.regime == "noisy" and name in moved_names
    )
    # P3.1: replace the legacy ``Confidence: rolled (median n=N)`` line
    # (internally inconsistent with per-parameter regime, see PLAN.md
    # 2.2) with a per-channel held-out error budget pulled from the gate
    # JSON. Falls back to the legacy line when no held-out row matches
    # this (car, track) -- e.g. a track the gate doesn't cover.
    error_budget_block = _render_error_budget_block(model.car, track_label)
    if error_budget_block:
        lines.append(
            f" Conditions: {rec.env.air_temp_c:.0f} C ambient / "
            f"{rec.env.track_temp_c:.0f} C track  |  "
            f"Moved params: {sparse_moved} sparse / {noisy_moved} noisy "
            f"of {len(moved)} changes"
        )
        for line in error_budget_block:
            lines.append(f" {line}")
    else:
        lines.append(
            f" Conditions: {rec.env.air_temp_c:.0f} C ambient / "
            f"{rec.env.track_temp_c:.0f} C track  |  "
            f"Confidence: {rolled} (median n={n_med})  |  "
            f"Moved params: {sparse_moved} sparse / {noisy_moved} noisy "
            f"of {len(moved)} changes"
        )
    coverage_line = _track_coverage_headline(model, track_label)
    if coverage_line:
        lines.append(f" {coverage_line}")
    lines.append("=" * 72)
    lines.append("")

    # ---- OVERALL DIRECTION ----
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
            lines.extend(_render_change(
                j, past_value.get(j.parameter), onto,
                model=model, rec_setup=rec_setup, env=rec.env, schedule=schedule,
            ))
            lines.append("")

    if "OTHER" in by_group:
        lines.append("-- OTHER --")
        for j in sorted(by_group["OTHER"], key=_param_sort_key):
            lines.extend(_render_change(
                j, past_value.get(j.parameter), onto,
                model=model, rec_setup=rec_setup, env=rec.env, schedule=schedule,
            ))
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
    *,
    model: PhysicsModel | None = None,
    rec_setup: dict[str, float] | None = None,
    env=None,
    schedule: list | None = None,
) -> list[str]:
    spec = onto.get(j.parameter)
    label = _param_label(j.parameter, model.car if model is not None else None)
    family = spec.family if spec else "other"
    delta = (j.value - past) if past is not None else 0.0
    direction = _direction_word(family, delta)

    change_str = _format_value_delta(j, spec, past)
    body: list[str] = [f"{label}: {change_str}  ({direction})"]

    # Telemetry-backed "Why" line. Queries the model for what the
    # dominant evidence channel for this parameter family read at the
    # past setup, with a concrete number from the corpus. Only fires
    # when the call site supplied model + rec_setup + env (i.e.
    # full-fledged briefing, not unit-test scaffolding).
    if model is not None and rec_setup is not None and env is not None and past is not None:
        why = _telemetry_why(
            j, family, past, model, rec_setup, env, schedule,
        )
        if why:
            body.append(f"  Why:    {why}")

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
        impact = _dominant_impact_corner(j, family=family)
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
    if not j.pinned and (
        abs(j.sensitivity_plus_1_click) > 1e-6
        or abs(j.sensitivity_minus_1_click) > 1e-6
    ):
        body.append(
            f"  Sensitivity: +1 click {j.sensitivity_plus_1_click:+.3f} score, "
            f"-1 click {j.sensitivity_minus_1_click:+.3f} score"
        )
    return body


# ---------------------------------------------------------------------------
# Telemetry-evidence Why line
# ---------------------------------------------------------------------------


# Per-(family[, axis]) ordered list of (channel, friendly_label, unit,
# threshold_hint). The first channel whose past-setup prediction is
# numeric (not None) seeds the Why line. Threshold hint adds a brief
# qualitative anchor when the value is in a known concerning range.
_EVIDENCE_CHANNELS: dict[tuple, list[tuple[str, str, str, str]]] = {
    ("heave_spring", "front"): [
        ("lf_shock_defl_p99_mm", "front-left shock peak compression", "mm", ""),
        ("setup_static_lf_ride_height_mm", "front static garage ride height", "mm", ""),
        ("understeer_angle_mean_rad", "mid-corner understeer angle", "rad", ""),
    ],
    ("torsion_bar", "front"): [
        ("lf_shock_defl_p99_mm", "front-left shock peak compression", "mm", ""),
        ("understeer_angle_mean_rad", "mid-corner understeer angle", "rad", ""),
    ],
    ("torsion_bar", "rear"): [
        ("lr_shock_defl_p99_mm", "rear-left shock peak compression", "mm", ""),
    ],
    ("spring_rate", "rear"): [
        ("lr_shock_defl_p99_mm", "rear-left shock peak compression", "mm", ""),
        ("setup_static_lr_ride_height_mm", "rear static garage ride height", "mm", ""),
        ("dynamic_rear_rh_at_speed_mm", "rear wheel ride height on straights", "mm", ""),
    ],
    ("spring_rate", "rear-third"): [
        ("dynamic_rear_rh_at_speed_mm", "rear wheel ride height on straights", "mm", ""),
        ("lr_shock_defl_p99_mm", "rear shock peak compression", "mm", ""),
    ],
    ("perch_offset", "front"): [
        ("setup_static_lf_ride_height_mm", "front static garage ride height", "mm", ""),
        ("dynamic_front_rh_at_speed_mm", "front wheel ride height on straights", "mm", ""),
    ],
    ("perch_offset", "rear"): [
        ("setup_static_lr_ride_height_mm", "rear static garage ride height", "mm", ""),
        ("dynamic_rear_rh_at_speed_mm", "rear wheel ride height on straights", "mm", ""),
    ],
    ("pushrod", "front"): [
        ("setup_static_lf_ride_height_mm", "front static garage ride height", "mm", ""),
    ],
    ("pushrod", "rear"): [
        ("setup_static_lr_ride_height_mm", "rear static ride height", "mm", ""),
    ],
    ("damper", "front"): [
        ("damper_velocity_p99_mms", "front damper peak velocity", "mm/s", ""),
        ("lf_shock_defl_p99_mm", "front-left shock peak compression", "mm", ""),
    ],
    ("damper", "rear"): [
        ("damper_velocity_p99_mms", "rear damper peak velocity", "mm/s", ""),
        ("lr_shock_defl_p99_mm", "rear-left shock peak compression", "mm", ""),
    ],
    ("arb", "front"): [
        ("understeer_angle_mean_rad", "mid-corner understeer angle", "rad", ""),
        ("accel_lat_g_max", "peak lateral G", "g", ""),
    ],
    ("arb", "rear"): [
        ("accel_lat_g_max", "peak lateral G", "g", ""),
        ("throttle_max", "throttle application", "", ""),
    ],
    ("camber", "front"): [
        ("accel_lat_g_max", "peak lateral G", "g", ""),
        ("understeer_angle_mean_rad", "understeer angle", "rad", ""),
    ],
    ("camber", "rear"): [
        ("accel_lat_g_max", "peak lateral G", "g", ""),
        ("throttle_max", "throttle on exit", "", ""),
    ],
    ("camber", "toe-front"): [
        ("steering_max_rad", "peak steering input", "rad", ""),
        ("accel_lat_g_max", "peak lateral G", "g", ""),
    ],
    ("camber", "toe-rear"): [
        ("understeer_angle_mean_rad", "understeer angle", "rad", ""),
    ],
    ("rear_wing", None): [
        ("dynamic_rear_rh_at_speed_mm", "rear wheel ride height on straights", "mm", ""),
        ("accel_lat_g_max", "peak lateral G", "g", ""),
    ],
    ("tyre_pressure", None): [
        ("accel_lat_g_max", "peak lateral G", "g", ""),
    ],
    ("brake_bias", None): [
        ("brake_max", "peak brake pressure", "", ""),
        ("accel_lat_g_max", "peak lateral G", "g", ""),
    ],
    ("diff", "preload"): [
        ("throttle_max", "throttle application on exit", "", ""),
        ("accel_lat_g_max", "peak lateral G", "g", ""),
    ],
    ("diff", "ramps"): [
        ("throttle_max", "throttle application on exit", "", ""),
    ],
    ("diff", "plates"): [
        ("throttle_max", "throttle application on exit", "", ""),
    ],
    ("fuel", None): [
        ("setup_static_lf_ride_height_mm", "front static garage ride height", "mm", ""),
        ("setup_static_lr_ride_height_mm", "rear static ride height", "mm", ""),
    ],
}


def _telemetry_why(
    j: SetupJustification,
    family: str,
    past_param_value: float,
    model: PhysicsModel,
    rec_setup: dict[str, float],
    env,
    schedule: list | None,
) -> str:
    """Predict telemetry channel value at the parameter's past setup.

    Builds a counterfactual where every parameter is at the optimised
    value EXCEPT the one in question -- that one stays at the past
    value. Queries the model at the dominant Helps corner-phase and
    reports the value of the family's primary evidence channel.
    Reads as: "at T11 mid-corner, predicted front shock p99 was 17.4 mm
    -- close to bottoming."
    """
    name = j.parameter
    axis = _param_axis(name)
    sub = _param_subtype(family, name)

    # Select the evidence channel list. Try most-specific match first.
    channels: list[tuple[str, str, str, str]] | None = None
    for key in (
        (family, sub),
        (family, axis),
        (family, None),
    ):
        if key in _EVIDENCE_CHANNELS:
            channels = _EVIDENCE_CHANNELS[key]
            break
    if channels is None:
        return ""

    # Dominant Helps corner-phase -- where this parameter does the most
    # work. Falls back to the worst Hurts when there are no helps.
    # Apply the same family-preferred-phase filter the Watch-most picker
    # uses (commit 7be3017): without this, raw score-delta max picks the
    # heaviest-weighted corner (T5 at Spa from corner-duration weighting)
    # for EVERY parameter, making every Why line anchor at the same
    # corner regardless of what the parameter mechanically affects.
    raw_pool = list(j.corners_helped) or list(j.corners_hurt)
    if not raw_pool:
        return ""
    # P3.2: count corner spread BEFORE family filter; same rule as
    # ``_dominant_impact_corner`` so the Why line and Watch-most line
    # agree on which corner the parameter is mechanically anchored to.
    corner_counts: dict[int, int] = {}
    for impact in raw_pool:
        corner_counts[int(impact.corner_id)] = (
            corner_counts.get(int(impact.corner_id), 0) + 1
        )
    impacts = raw_pool
    preferred = _FAMILY_PREFERRED_PHASES.get(family or "")
    if preferred:
        in_preferred = [i for i in impacts if i.phase in preferred]
        if in_preferred:
            impacts = in_preferred
    else:
        on_corner = [i for i in impacts if i.phase != Phase.STRAIGHT]
        if on_corner:
            impacts = on_corner
    top = max(
        impacts,
        key=lambda i: abs(float(i.score_delta)) / float(
            max(corner_counts.get(int(i.corner_id), 1), 1),
        ),
    )

    # Counterfactual setup: rec values everywhere except this param.
    counterfactual = dict(rec_setup)
    counterfactual[name] = float(past_param_value)

    cpkey = CornerPhaseKey(
        session_id="<narrative-counterfactual>",
        lap_index=0,
        corner_id=int(top.corner_id),
        phase=top.phase,
    )

    archetype = _archetype_for(schedule, top.corner_id)
    try:
        if int(model.feature_schema_version) >= 4 and archetype is not None:
            state = model.predict(counterfactual, env, cpkey, corner_archetype=archetype)
        else:
            state = model.predict(counterfactual, env, cpkey)
    except Exception:
        return ""

    states = getattr(state, "states", {}) or {}
    for channel, label, unit, _hint in channels:
        conf = states.get(channel)
        if conf is None:
            continue
        value = float(conf.value)
        unit_s = f" {unit}" if unit else ""
        phase_label = _PHASE_LABEL.get(top.phase, top.phase.value)
        # Friendly value formatting per channel range.
        if abs(value) >= 100:
            v_s = f"{value:.0f}"
        elif abs(value) >= 10:
            v_s = f"{value:.1f}"
        else:
            v_s = f"{value:.2f}"
        return (
            f"At T{top.corner_id} {phase_label}, predicted {label} "
            f"was {v_s}{unit_s} under the past setup."
        )
    return ""


def _archetype_for(schedule: list | None, corner_id: int) -> dict | None:
    """Pull the corner_archetype dict for a given corner_id from the schedule."""
    if not schedule:
        return None
    for entry in schedule:
        if int(getattr(entry, "corner_id", -1)) == int(corner_id):
            return getattr(entry, "archetype", None)
    return None


# Per-family "you will feel this in these phases" preference. The
# raw score delta picker chases time-impact (longest corner wins),
# which is duration-biased after the e90e8fd corner-weight refactor.
# Drivers feel parameters at MECHANICALLY relevant phases regardless
# of corner length, so we filter the impact pool by family-preferred
# phases before picking the dominant corner.
_FAMILY_PREFERRED_PHASES: dict[str, frozenset[Phase]] = {
    "rear_wing":     frozenset({Phase.MID_CORNER, Phase.STRAIGHT}),
    "tyre_pressure": frozenset({Phase.MID_CORNER}),
    "perch_offset":  frozenset({Phase.MID_CORNER, Phase.EXIT}),
    "pushrod":       frozenset({Phase.MID_CORNER, Phase.EXIT}),
    "spring_rate":   frozenset({Phase.TRAIL_BRAKE, Phase.MID_CORNER}),
    "torsion_bar":   frozenset({Phase.TRAIL_BRAKE, Phase.MID_CORNER}),
    "arb":           frozenset({Phase.MID_CORNER}),
    "damper":        frozenset({Phase.TRAIL_BRAKE, Phase.MID_CORNER, Phase.EXIT}),
    "camber":        frozenset({Phase.MID_CORNER}),
    "brake_bias":    frozenset({Phase.BRAKING, Phase.TRAIL_BRAKE}),
    "diff":          frozenset({Phase.EXIT}),
    "fuel":          frozenset({Phase.MID_CORNER}),
}


def _dominant_impact_corner(
    j: SetupJustification, family: str | None = None,
) -> str:
    """Single-line `T11 braking apex` style -- the corner-phase the
    driver will actually feel this parameter at.

    Filters the helps + hurts pool by `_FAMILY_PREFERRED_PHASES[family]`
    so a damper change reports trail-brake/mid-corner/exit, a camber
    reports mid-corner, a diff reports exit -- regardless of which
    corner has the biggest raw score delta (which is duration-biased
    by the corner-weight refactor in e90e8fd).

    P3.2: scores impacts by ``|score_delta| / phase_count_in_pool``
    where ``phase_count_in_pool`` is the number of times this candidate's
    corner shows up across phases in the filtered pool. The
    corner-duration weighting makes the longest corner emit a per-phase
    score delta proportional to its share of lap-time, so a long corner
    accumulates impact across many phases. Normalising by the corner's
    phase-spread in the pool penalises that broad-impact pattern and
    rewards corners with mechanically-concentrated impact -- which is
    what the user is actually being told to "Watch most".

    Falls back to non-STRAIGHT phases, then to any phase, when the
    preferred-phase pool is empty.
    """
    raw_pool = list(j.corners_helped) + list(j.corners_hurt)
    if not raw_pool:
        return ""

    # P3.2: count corner spread BEFORE the family-preferred-phase
    # filter, so a long corner that emits impact across all five phases
    # always carries its full duration penalty -- even when the family
    # filter narrows the candidate pool to one phase. Counting
    # post-filter would let a long corner that happens to match a
    # single preferred phase escape the normalisation.
    corner_counts: dict[int, int] = {}
    for impact in raw_pool:
        corner_counts[int(impact.corner_id)] = (
            corner_counts.get(int(impact.corner_id), 0) + 1
        )

    candidates = raw_pool
    preferred = _FAMILY_PREFERRED_PHASES.get(family or "")
    if preferred:
        in_preferred = [i for i in candidates if i.phase in preferred]
        if in_preferred:
            candidates = in_preferred
    else:
        on_corner = [i for i in candidates if i.phase != Phase.STRAIGHT]
        if on_corner:
            candidates = on_corner

    def _normalised(i) -> float:
        spread = corner_counts.get(int(i.corner_id), 1)
        return abs(float(i.score_delta)) / float(max(spread, 1))

    top = max(candidates, key=_normalised)
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
        label = _param_label(j.parameter, rec.car)
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
    if rec.pinned_within_track_thin:
        for name in sorted(rec.pinned_within_track_thin):
            label = _param_label(name, rec.car)
            out.append(
                f"  {label}: pinned at target track (fewer than 3 distinct "
                f"values driven here -- run calibrate probes before trusting "
                f"cross-track moves)"
            )
    global_pinned = [
        n for n in rec.pinned_to_observed_median
        if n not in set(rec.pinned_within_track_thin)
    ]
    if global_pinned:
        for name in sorted(global_pinned):
            label = _param_label(name, rec.car)
            out.append(
                f"  {label}: pinned to observed median (no corpus variance on "
                f"this parameter -- vary it in a session to unlock search)"
            )
    suppressed = tuple(
        getattr(rec, "suppressed_below_sensitivity", ()) or ()
    )
    if suppressed:
        for name in sorted(suppressed):
            label = _param_label(name, rec.car)
            out.append(
                f"  {label}: held at past value (model cannot resolve "
                f"+/-1 click on this corpus -- below sensitivity floor)"
            )
    if rec.untrained_parameters:
        table = load_constraints()
        no_bounds: list[str] = []
        blocked: list[str] = []
        for name in sorted(rec.untrained_parameters):
            spec = onto.get(name)
            if spec is not None and not spec.fittable:
                blocked.append(name)
            elif table.bounds(rec.car, name) is None:
                no_bounds.append(name)
            else:
                blocked.append(name)
        if no_bounds:
            params_str = ", ".join(no_bounds)
            out.append(f"  Untrained (no constraints.md bounds): {params_str}")
        if blocked:
            params_str = ", ".join(blocked)
            out.append(
                f"  Not searched (calculated readout or blocked in ontology): "
                f"{params_str}"
            )
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
    step 0.125) would render as "0.10 turns (stiffens)" -- same displayed
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
