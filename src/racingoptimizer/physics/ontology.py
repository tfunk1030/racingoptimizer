"""Per-car garage parameter ontology.

Maps the bounded parameter names from `constraints.md` to the JSON paths in
slice A's catalog `setup` blob. Five frozen dicts (one per car) cover both
the bounded `fittable=True` families and the CE-gated `fittable=False`
ones (ARBs, dampers, corner weights, brake bias, diff). The CE-gated entries
are emitted so `PhysicsModel.untrained_parameters` can list them.

JSON path resolution is NaN-tolerant: missing keys return `None`, missing
units return `None`. The fitter drops rows where `setup_value` returns `None`
rather than aborting the model build.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from racingoptimizer.constraints import ConstraintsTable

Family = Literal[
    "heave_spring",
    "heave_slider",
    "tyre_pressure",
    "front_wing",
    "rear_wing",
    "ride_height",
    "arb",
    "damper",
    "corner_weight",
    "brake_bias",
    "diff",
    "spring_rate",
    "perch_offset",
    "pushrod",
    "camber",
    "torsion_bar",
    "fuel",
]


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    json_path: tuple[str, ...]
    dtype: type
    units: str
    family: Family
    # `fittable=True` → the model trains a per-(corner, phase, channel)
    # fitter that includes this parameter in its feature vector. Setting
    # this to False excludes the parameter from the joint surrogate
    # entirely (used for fields the model has no use for).
    fittable: bool
    # `user_settable=True` → the driver can type this value into the iRacing
    # garage UI. False marks calculated readouts (ride heights, deflections,
    # corner weights, aero-calculator outputs, last hot pressures, etc.)
    # that the model is allowed to LEARN as targets but the optimizer must
    # not put into its search space and the briefing must not list as
    # "set this". Defaults to True for backward-compat with all the existing
    # entries that pre-date this flag — every legacy entry was a real input.
    user_settable: bool = True
    # `is_discrete=True` → the iRacing garage UI exposes this parameter as
    # an integer click count (e.g. ARB blade index 1..5, damper clicks).
    # The DE search runs continuously over `[lo, hi]`, so the post-clamp
    # step in `cli/recommend.py` rounds discrete recommendations to the
    # nearest integer before rendering. Defaults to False for the
    # continuous parameters (springs, perches, pressures, wing angle, etc.).
    is_discrete: bool = False
    # ----------------------------------------------------------------------
    # IMPORTANT: append-only beyond this point.
    #
    # ParameterSpec is a frozen+slots dataclass; instances are referenced
    # from `ACURA`/`BMW`/`CADILLAC`/etc. module-level dicts that get
    # pickled inside `PhysicsModel.ontology`. Inserting a field in the
    # middle would shift every later slot's pickle position and silently
    # corrupt revives of older models. New fields MUST go at the end.
    # ----------------------------------------------------------------------
    # iRacing garage UI click-step in `units`. The garage exposes every
    # parameter as a discrete adjustment — the driver clicks ◀ / ▶ chevrons
    # to step by this increment. The renderer rounds optimizer-recommended
    # values to the nearest step before displaying so the briefing always
    # shows a value the driver can actually enter. ``None`` disables
    # rounding (used for parameters whose step the loader does not yet know
    # — value is shown raw). Step sizes are ESTIMATES from the user's
    # "general idea" garage-ranges table and may need refinement against
    # the live iRacing UI.
    step: float | None = None
    # Non-uniform discrete legal values. Used by parameters whose iRacing
    # UI exposes a fixed list of options that don't follow a uniform
    # step — e.g. torsion bar OD's 14 diameters from 13.90..18.20 mm.
    # When present, the renderer snaps the optimizer's continuous output
    # to the nearest legal value at display time (overrides ``step``);
    # the DE search itself runs continuously over the constraint range.
    discrete_values: tuple[float, ...] | None = None
    # Categorical (enum-typed) parameter labels, in display order. The
    # underlying YAML value is a string (e.g. ARB Size: "Soft"); the
    # fitter trains on the integer index and the renderer maps the
    # rounded index back to the label. ``None`` means the parameter is
    # numeric.
    choices: tuple[str, ...] | None = None


# --- Path-extraction helpers ----------------------------------------------

# Strip "12 kPa" / "+0.3 mm" / "47.00%" / "10 clicks" → 12.0 / 0.3 / 47.0 / 10.0.
# First number wins, sign retained, fractional optional.
_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _scalar(text: Any) -> float | None:
    """Coerce an IBT setup field to a single float, or `None` if unavailable."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    if not isinstance(text, str):
        return None
    m = _NUMERIC_RE.search(text)
    if m is None:
        return None
    return float(m.group(0))


def _walk(setup: dict, path: tuple[str, ...]) -> Any:
    cur: Any = setup
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


# --- Per-car ontologies ----------------------------------------------------

# All five GTP cars use the same TiresAero / Chassis top-level layout but
# diverge inside Chassis.LeftFront/LeftRear/Rear, especially around damper
# placement (Acura/Porsche have a separate `Dampers` block; BMW/Cadillac
# inline per-corner) and CE-gated families. The dicts below pin one
# representative path per bounded `constraints.md` key plus untyped CE
# entries for graceful enumeration.

_TYRE_FL = ("TiresAero", "LeftFront", "StartingPressure")
_AERO_REAR_WING = ("TiresAero", "AeroSettings", "RearWingAngle")
_RIDE_LF = ("Chassis", "LeftFront", "RideHeight")
_RIDE_LR = ("Chassis", "LeftRear", "RideHeight")
_HEAVE_SPRING_F = ("Chassis", "Front", "HeaveSpringDefl")
_HEAVE_SLIDER_F = ("Chassis", "Front", "HeaveSliderDefl")
_BRAKE_BIAS = ("BrakesDriveUnit", "BrakeSpec", "BrakePressureBias")

# USER-input setup parameters that drive the calculated readouts above.
# Ranges are bounded by `constraints.md` (see the "estimated bounds" note).
# Spring rates and perch offsets sit at the "Front" / "Rear" axle level;
# per-corner spring rates are read off LeftRear (mirrored to RightRear by
# the iRacing UI when the symmetric option is on).
_HEAVE_SPRING_RATE_F = ("Chassis", "Front", "HeaveSpring")
_THIRD_SPRING_RATE_R = ("Chassis", "Rear", "ThirdSpring")
_REAR_COIL_SPRING_RATE = ("Chassis", "LeftRear", "SpringRate")
_HEAVE_PERCH_OFFSET_F = ("Chassis", "Front", "HeavePerchOffset")
_SPRING_PERCH_OFFSET_R = ("Chassis", "LeftRear", "SpringPerchOffset")
_THIRD_PERCH_OFFSET_R = ("Chassis", "Rear", "ThirdPerchOffset")
_PUSHROD_OFFSET_F = ("Chassis", "Front", "PushrodLengthOffset")
_PUSHROD_OFFSET_R = ("Chassis", "Rear", "PushrodLengthOffset")
_PUSHROD_DELTA_F = ("Chassis", "Front", "PushrodLengthDelta")
_PUSHROD_DELTA_R = ("Chassis", "Rear", "PushrodLengthDelta")

# Per-corner camber. iRacing GTP setup YAML stores camber as a string
# (e.g. "-2.5 deg") at `Chassis.<Corner>.Camber`. Bounds in `constraints.md`
# (front -2.9..0, rear -1.9..0) are real numeric ranges, not TODOs.
_CAMBER_LF = ("Chassis", "LeftFront", "Camber")
_CAMBER_RF = ("Chassis", "RightFront", "Camber")
_CAMBER_LR = ("Chassis", "LeftRear", "Camber")
_CAMBER_RR = ("Chassis", "RightRear", "Camber")


def _common_bounded() -> dict[str, ParameterSpec]:
    """Bounded families that every car shares (path conventions vary; see overrides)."""
    return {
        "rear_wing_angle_deg": ParameterSpec(
            json_path=_AERO_REAR_WING, dtype=float, units="deg",
            family="rear_wing", fittable=True, step=1.0,
        ),
        "tyre_cold_pressure_kpa": ParameterSpec(
            json_path=_TYRE_FL, dtype=float, units="kPa",
            family="tyre_pressure", fittable=True, step=0.5,
        ),
        # ------------------------------------------------------------------
        # CALCULATED READOUTS — `user_settable=False`.
        # iRacing's garage UI displays these as outputs of the actual setup
        # work (perch offsets, pushrod lengths, torsion-bar turns, spring
        # rates, corner weights). The driver cannot type them into the
        # garage; they update as a consequence of the inputs.
        #
        # These entries stay in the ontology so:
        #   * `parameters(car)` enumerates them (used by callers that want
        #     a complete YAML-leaf inventory, e.g. coverage reports).
        #   * `setup_value(car, name, blob)` resolves them on ingested
        #     setups for downstream readers (e.g. fit-quality dashboards
        #     comparing observed vs predicted ride height).
        #
        # They are *not* in the optimizer's search space (filtered out by
        # `user_settable=False` inside `fittable_parameters`) and they are
        # *not* trained as fit targets either — `TARGET_OUTPUT_CHANNELS`
        # in `physics/fitter.py` already includes the per-corner-phase
        # *dynamic* ride heights (`lf/rf/lr/rr_ride_height_mean_mm`),
        # which are the surface `_aero_ld_for_state` queries in
        # `physics/score.py`. The static ride height in the YAML is just
        # the iRacing UI's calculated baseline — redundant once the
        # dynamic per-corner ride-height predictions exist.
        #
        # Bounds for the underlying USER inputs (HeavePerchOffset,
        # PushrodLengthOffset, SpringPerchOffset, TorsionBarTurns, spring
        # rates) live alongside as `user_settable=True` entries below.
        # ------------------------------------------------------------------
        "static_ride_height_front_mm": ParameterSpec(
            json_path=_RIDE_LF, dtype=float, units="mm",
            family="ride_height", fittable=True, user_settable=False,
        ),
        "static_ride_height_rear_mm": ParameterSpec(
            json_path=_RIDE_LR, dtype=float, units="mm",
            family="ride_height", fittable=True, user_settable=False,
        ),
        "heave_spring_mm": ParameterSpec(
            json_path=_HEAVE_SPRING_F, dtype=float, units="mm",
            family="heave_spring", fittable=True, user_settable=False,
        ),
        "heave_slider_mm": ParameterSpec(
            json_path=_HEAVE_SLIDER_F, dtype=float, units="mm",
            family="heave_slider", fittable=True, user_settable=False,
        ),
        # USER-input setup parameters that drive the calculated readouts
        # above. The optimizer searches over these; the platform state
        # (ride heights, deflections) is the model's prediction of what
        # results. Ranges in `constraints.md` are estimates. `step` is the
        # iRacing garage UI click increment so the renderer can round
        # recommendations to the nearest enterable value.
        "heave_spring_rate_n_per_mm": ParameterSpec(
            json_path=_HEAVE_SPRING_RATE_F, dtype=float, units="N/mm",
            family="spring_rate", fittable=True, user_settable=True,
            step=5.0,
        ),
        "third_spring_rate_n_per_mm": ParameterSpec(
            json_path=_THIRD_SPRING_RATE_R, dtype=float, units="N/mm",
            family="spring_rate", fittable=True, user_settable=True,
            # iRacing GTP UI exposes third spring in 10 N/mm increments
            # (BMWBounds.md / Cadillacbounds.md both confirm). Coil springs
            # step in 5 N/mm but the third is coarser.
            step=10.0,
        ),
        "rear_coil_spring_rate_n_per_mm": ParameterSpec(
            json_path=_REAR_COIL_SPRING_RATE, dtype=float, units="N/mm",
            family="spring_rate", fittable=True, user_settable=True,
            step=5.0,
        ),
        "heave_perch_offset_front_mm": ParameterSpec(
            json_path=_HEAVE_PERCH_OFFSET_F, dtype=float, units="mm",
            family="perch_offset", fittable=True, user_settable=True,
            step=0.5,
        ),
        "spring_perch_offset_rear_mm": ParameterSpec(
            json_path=_SPRING_PERCH_OFFSET_R, dtype=float, units="mm",
            family="perch_offset", fittable=True, user_settable=True,
            step=0.5,
        ),
        "third_perch_offset_rear_mm": ParameterSpec(
            json_path=_THIRD_PERCH_OFFSET_R, dtype=float, units="mm",
            family="perch_offset", fittable=True, user_settable=True,
            step=0.5,
        ),
        "pushrod_length_offset_front_mm": ParameterSpec(
            json_path=_PUSHROD_OFFSET_F, dtype=float, units="mm",
            family="pushrod", fittable=True, user_settable=True,
            step=0.5,
        ),
        "pushrod_length_offset_rear_mm": ParameterSpec(
            json_path=_PUSHROD_OFFSET_R, dtype=float, units="mm",
            family="pushrod", fittable=True, user_settable=True,
            step=0.5,
        ),
        # Per-corner camber. Direct setup → tire-grip lever. Bounds in
        # `constraints.md` are non-placeholder (front -2.9..0, rear -1.9..0).
        "camber_fl_deg": ParameterSpec(
            json_path=_CAMBER_LF, dtype=float, units="deg",
            family="camber", fittable=True, user_settable=True, step=0.1,
        ),
        "camber_fr_deg": ParameterSpec(
            json_path=_CAMBER_RF, dtype=float, units="deg",
            family="camber", fittable=True, user_settable=True, step=0.1,
        ),
        "camber_rl_deg": ParameterSpec(
            json_path=_CAMBER_LR, dtype=float, units="deg",
            family="camber", fittable=True, user_settable=True, step=0.1,
        ),
        "camber_rr_deg": ParameterSpec(
            json_path=_CAMBER_RR, dtype=float, units="deg",
            family="camber", fittable=True, user_settable=True, step=0.1,
        ),
    }


def _common_ce_gated() -> dict[str, ParameterSpec]:
    """Mixed CE-gated + recently-bounded entries.

    ``fittable=True`` parameters here have legal bounds in
    ``constraints.md`` — the recommender will search over them via
    `fittable_parameters`. ``fittable=False`` parameters are still
    awaiting bounds capture from the iRacing UI; they're kept in the
    ontology so `PhysicsModel.untrained_parameters` can list them.
    """
    return {
        # ARB blade index (1..5 per `constraints.md`). Fittable since
        # bounds landed in `bf2e48b`. ARBs are GP-routed via
        # `_GP_FAMILIES` — DE search emits a continuous click value;
        # the user rounds to the nearest integer at the iRacing garage.
        "anti_roll_bar_front": ParameterSpec(
            json_path=("Chassis", "Front", "ArbBlades"), dtype=float, units="click",
            family="arb", fittable=True, is_discrete=True,
        ),
        "anti_roll_bar_rear": ParameterSpec(
            json_path=("Chassis", "Rear", "ArbBlades"), dtype=float, units="click",
            family="arb", fittable=True, is_discrete=True,
        ),
        # Brake bias front-axle pct (40..60 per `constraints.md`).
        # Fittable since bounds landed in `bf2e48b`.
        "brake_bias_pct": ParameterSpec(
            json_path=_BRAKE_BIAS, dtype=float, units="pct",
            family="brake_bias", fittable=True,
        ),
        # Differential preload Nm (0..150 per `constraints.md`). Coast/
        # power ratios remain CE-gated (separate parameters that aren't
        # in this ontology yet — see `constraints.md` `### Differential`
        # block).
        "diff_preload_nm": ParameterSpec(
            json_path=("BrakesDriveUnit", "RearDiffSpec", "Preload"),
            dtype=float, units="Nm", family="diff", fittable=True,
        ),
        "corner_weight_fl_kg": ParameterSpec(
            json_path=("Chassis", "LeftFront", "CornerWeight"), dtype=float,
            units="N", family="corner_weight", fittable=False,
            user_settable=False, step=10.0,
        ),
        "corner_weight_fr_kg": ParameterSpec(
            json_path=("Chassis", "RightFront", "CornerWeight"), dtype=float,
            units="N", family="corner_weight", fittable=False,
            user_settable=False, step=10.0,
        ),
        "corner_weight_rl_kg": ParameterSpec(
            json_path=("Chassis", "LeftRear", "CornerWeight"), dtype=float,
            units="N", family="corner_weight", fittable=False,
            user_settable=False, step=10.0,
        ),
        "corner_weight_rr_kg": ParameterSpec(
            json_path=("Chassis", "RightRear", "CornerWeight"), dtype=float,
            units="N", family="corner_weight", fittable=False,
            user_settable=False, step=10.0,
        ),
        # Race fuel load. iRacing exposes this as a typed value the user
        # picks pre-session (race default: ~58 L on the BMW M Hybrid V8;
        # quali stints are user-input depending on track length, often
        # 5-15 L for 3 laps + reserve). Fittable + user_settable so:
        #   * the joint surrogate sees fuel weight in the per-(corner,
        #     phase) feature row and can learn fuel→ride-height /
        #     fuel→balance coupling without a textbook formula
        #     (VISION §3 / §10);
        #   * `--fuel N` pins it via the existing pin mechanism;
        #   * `--quali` triggers a phase-weight swap toward outright
        #     pace and the user supplies the matching low-fuel level.
        # Step 1.0 L matches the iRacing UI's integer click. Bounds in
        # constraints.md (1..100 L envelope; quali min ~1 L is the
        # absolute floor, race max sits comfortably below tank cap).
        "fuel_level_l": ParameterSpec(
            # iRacing GTP YAML carries this under the BrakesDriveUnit
            # umbrella, NOT Chassis. Confirmed against BMW M Hybrid V8
            # session blob 0a9321b5: BrakesDriveUnit.Fuel.FuelLevel =
            # "58.0 L"; Chassis.Fuel does not exist. Cars with the diff
            # under Systems.RearDiffSpec (Acura / Ferrari) likely also
            # mirror their fuel under BrakesDriveUnit; verify when those
            # join the per-car path.
            json_path=("BrakesDriveUnit", "Fuel", "FuelLevel"), dtype=float,
            units="L", family="fuel", fittable=True, user_settable=True,
            step=1.0,
        ),
        # Toe-in (mm). Front is an axle-level scalar
        # (Chassis.Front.ToeIn); rear is per-corner with iRacing-UI
        # L=R symmetry — only the LR side is fittable here, the RR
        # YAML leaf gets the LR value mirrored at render time via
        # `_MIRRORED_LEAVES`. Cars with axle-level rear toe (none in
        # the current GTP fleet) would override `toe_rl_mm` to the
        # axle path; per-corner is the GTP norm.
        "toe_front_mm": ParameterSpec(
            json_path=("Chassis", "Front", "ToeIn"), dtype=float,
            units="mm", family="camber",
            fittable=True, user_settable=True, step=0.1,
        ),
        "toe_rl_mm": ParameterSpec(
            json_path=("Chassis", "LeftRear", "ToeIn"), dtype=float,
            units="mm", family="camber",
            fittable=True, user_settable=True, step=0.1,
        ),
        # Throttle/brake mapping path not yet verified against the iRacing
        # garage YAML — `BrakesDriveUnit.TcAndThrottle.ThrottleShape` does
        # not resolve on any of the 5 GTP fixtures. Held as user_settable
        # but `fittable=False` until the real path lands (see
        # `tests/physics/test_ontology_per_car.py`).
        "throttle_brake_mapping": ParameterSpec(
            json_path=("BrakesDriveUnit", "TcAndThrottle", "ThrottleShape"),
            dtype=float, units="click", family="throttle_map",
            fittable=False, user_settable=True, step=1.0,
        ),
    }


def _blocked_like(base: ParameterSpec, *, units: str | None = None) -> ParameterSpec:
    """Return a known user input that is not safe to fit/recommend yet."""
    return ParameterSpec(
        json_path=base.json_path,
        dtype=base.dtype,
        units=units or base.units,
        family=base.family,
        fittable=False,
        user_settable=base.user_settable,
        is_discrete=base.is_discrete,
    )


# (constraint suffix, IBT field name) for the damper modes.
# `hsc_slope` is iRacing's HS-Comp damper crossover-knee click separate
# from the four primary modes; constraints.md has its own section.
_DAMPER_MODES: tuple[tuple[str, str], ...] = (
    ("lsc", "LsCompDamping"),
    ("hsc", "HsCompDamping"),
    ("lsr", "LsRbdDamping"),
    ("hsr", "HsRbdDamping"),
    ("hsc_slope", "HsCompDampSlope"),
)


def _damper_paths(
    corner_to_path: tuple[tuple[str, tuple[str, ...]], ...],
) -> dict[str, ParameterSpec]:
    """Generate `damper_<mode>_<corner>` entries from a (corner_code → path) map.

    Dampers are now ``fittable=True`` since `constraints.md` carries
    estimated 1..15 click bounds per (corner, mode). The values are
    discrete integer clicks; the post-clamp in `cli/recommend.py` rounds
    DE's continuous proposals to the nearest legal integer before
    rendering.
    """
    out: dict[str, ParameterSpec] = {}
    for code, parent_path in corner_to_path:
        for suffix, field_name in _DAMPER_MODES:
            out[f"damper_{suffix}_{code}"] = ParameterSpec(
                json_path=(*parent_path, field_name), dtype=float, units="click",
                family="damper", fittable=True, is_discrete=True, step=1.0,
            )
    return out


# BMW / Cadillac inline per-corner damper clicks under Chassis.<corner>.
_INLINE_DAMPERS = _damper_paths(
    (
        ("fl", ("Chassis", "LeftFront")),
        ("fr", ("Chassis", "RightFront")),
        ("rl", ("Chassis", "LeftRear")),
        ("rr", ("Chassis", "RightRear")),
    )
)
# Acura / Porsche keep a top-level `Dampers` block keyed by axle/role. The two
# front damper paths share the FrontHeave block and the two rear paths share
# RearHeave — the per-corner click is the same scalar today and a future
# refinement can split them when iRacing exposes per-corner Dampers entries.
_SPLIT_DAMPERS = _damper_paths(
    (
        ("fl", ("Dampers", "FrontHeave")),
        ("fr", ("Dampers", "FrontHeave")),
        ("rl", ("Dampers", "RearHeave")),
        ("rr", ("Dampers", "RearHeave")),
    )
)
# Ferrari 499P keeps per-corner damper clicks under a top-level
# ``Dampers`` section but with one block PER corner (LeftFrontDamper /
# RightFrontDamper / LeftRearDamper / RightRearDamper) — confirmed
# against an ingested Ferrari setup blob. Damper click range is also
# wider than the BMW/Cadillac default (0..40 vs 0..11); per-car
# constraint overrides in `constraints.md` carry the wider envelope.
_FERRARI_DAMPERS = _damper_paths(
    (
        ("fl", ("Dampers", "LeftFrontDamper")),
        ("fr", ("Dampers", "RightFrontDamper")),
        ("rl", ("Dampers", "LeftRearDamper")),
        ("rr", ("Dampers", "RightRearDamper")),
    )
)
# Porsche 963 has per-corner rear dampers (`LeftRear`, `RightRear`) plus
# a shared `FrontHeave` for the front axle. The front heave block has
# NO `HsCompDampSlope` field, so the two front-slope entries are dropped
# from the Porsche damper set (vs. Acura which does carry it on both
# front and rear). Verified against an ingested Porsche Algarve setup
# blob (FrontHeave + LeftRear + RightRear + FrontRoll + Rear3Rd).
_PORSCHE_DAMPERS_RAW = _damper_paths(
    (
        ("fl", ("Dampers", "FrontHeave")),
        ("fr", ("Dampers", "FrontHeave")),
        ("rl", ("Dampers", "LeftRear")),
        ("rr", ("Dampers", "RightRear")),
    )
)
_PORSCHE_DAMPERS = {
    k: v for k, v in _PORSCHE_DAMPERS_RAW.items()
    if k not in ("damper_hsc_slope_fl", "damper_hsc_slope_fr")
}


def _build(
    damper_paths: dict[str, ParameterSpec],
    **overrides: ParameterSpec,
) -> dict[str, ParameterSpec]:
    return {**_common_bounded(), **_common_ce_gated(), **damper_paths, **overrides}


# Acura uses HeaveDamperDefl for the heave slider; the rest follow the
# common bounded layout. Cadillac/BMW/Ferrari share the inline damper layout;
# Porsche/Acura share the split layout.
_ACURA_HEAVE_SLIDER = ParameterSpec(
    json_path=("Chassis", "Front", "HeaveDamperDefl"),
    dtype=float, units="mm", family="heave_slider",
    fittable=True, user_settable=False,  # readout, not user input
)

_ACURA_OVERRIDES: dict[str, ParameterSpec] = {
    "heave_slider_mm": _ACURA_HEAVE_SLIDER,
    "third_spring_rate_n_per_mm": ParameterSpec(
        json_path=("Chassis", "Rear", "HeaveSpring"), dtype=float, units="N/mm",
        family="spring_rate", fittable=True, user_settable=True,
    ),
    "spring_perch_offset_rear_mm": ParameterSpec(
        json_path=("Chassis", "Rear", "HeavePerchOffset"), dtype=float, units="mm",
        family="perch_offset", fittable=True, user_settable=True,
    ),
    "pushrod_length_offset_front_mm": ParameterSpec(
        json_path=_PUSHROD_DELTA_F, dtype=float, units="mm",
        family="pushrod", fittable=True, user_settable=True,
    ),
    "pushrod_length_offset_rear_mm": ParameterSpec(
        json_path=_PUSHROD_DELTA_R, dtype=float, units="mm",
        family="pushrod", fittable=True, user_settable=True,
    ),
    "brake_bias_pct": ParameterSpec(
        json_path=("Systems", "BrakeSpec", "BrakePressureBias"),
        dtype=float, units="pct", family="brake_bias", fittable=True,
    ),
    "diff_preload_nm": ParameterSpec(
        json_path=("Systems", "RearDiffSpec", "Preload"),
        dtype=float, units="Nm", family="diff", fittable=True,
    ),
    # Acura has no rear coil / third perch leaves in the canonical setup YAML.
    "rear_coil_spring_rate_n_per_mm": _blocked_like(
        _common_bounded()["rear_coil_spring_rate_n_per_mm"],
    ),
    "third_perch_offset_rear_mm": _blocked_like(_common_bounded()["third_perch_offset_rear_mm"]),
    # Acura fuel is under Systems.Fuel (vs BrakesDriveUnit.Fuel for BMW etc.).
    "fuel_level_l": ParameterSpec(
        json_path=("Systems", "Fuel", "FuelLevel"),
        dtype=float, units="L", family="fuel",
        fittable=True, user_settable=True, step=0.1,
    ),
    # Acura tracks toe at the axle level (`Chassis.Rear.ToeIn`), not
    # per-corner. The per-corner rear toe parameter does not resolve
    # on Acura YAMLs; block it here until per-axle rear toe is added
    # as a first-class ontology parameter.
    "toe_rl_mm": _blocked_like(_common_ce_gated()["toe_rl_mm"]),
}

# iRacing GTP front torsion-bar diameters — 14 discrete values across the
# range. Both Cadillac and BMW (per BMWBounds.md / Cadillacbounds.md)
# expose the same list. The DE search runs continuously over the bound
# envelope; the renderer snaps to the nearest legal diameter at display
# time via `ParameterSpec.discrete_values`.
_TORSION_BAR_OD_VALUES: tuple[float, ...] = (
    13.90, 14.34, 14.76, 15.14, 15.51, 15.86, 16.19,
    16.51, 16.81, 17.11, 17.39, 17.67, 17.94, 18.20,
)


_CADILLAC_OVERRIDES: dict[str, ParameterSpec] = {
    "diff_preload_nm": ParameterSpec(
        json_path=("BrakesDriveUnit", "DiffSpec", "Preload"),
        dtype=float, units="Nm", family="diff", fittable=True,
    ),
    # Cadillac front uses torsion bars instead of coil springs at the
    # corners. iRacing requires LF=RF symmetry (same hardware contract
    # as rear spring rate on BMW), so we train ONE LF parameter per
    # turns + OD; the renderer mirrors to RF via `_MIRRORED_LEAVES`.
    # The iRacing UI exposes 14 discrete OD values from 13.90..18.20
    # mm — DE searches the continuous envelope and the renderer snaps
    # to the nearest legal diameter via `discrete_values`.
    "torsion_bar_turns_fl": ParameterSpec(
        json_path=("Chassis", "LeftFront", "TorsionBarTurns"),
        dtype=float, units="turns", family="torsion_bar",
        fittable=True, user_settable=True, step=0.001,
    ),
    "torsion_bar_od_fl_mm": ParameterSpec(
        json_path=("Chassis", "LeftFront", "TorsionBarOD"),
        dtype=float, units="mm", family="torsion_bar",
        fittable=True, user_settable=True,
        discrete_values=_TORSION_BAR_OD_VALUES,
    ),
}


# ARB-size enum (front + rear). Order is stiffness-ascending: the index
# is meaningful as an ordinal feature for the fitter (Disconnect = no
# anti-roll bar engagement → softest, Stiff = fully engaged).
_ARB_SIZE_CHOICES: tuple[str, ...] = ("Disconnect", "Soft", "Medium", "Stiff")
# Diff coast/drive ramp angle pairs. Ordered ascending in lock-up
# aggressiveness (40°/65° = freest, 50°/75° = most locked).
_DIFF_COAST_DRIVE_RAMP_CHOICES: tuple[str, ...] = ("40/65", "45/70", "50/75")
# Clutch friction plate count. Stored as a numeric value in the iRacing
# YAML (not a string label), so it uses ``discrete_values`` for the
# render-time snap rather than ``choices``.
_DIFF_CLUTCH_PLATES_VALUES: tuple[float, ...] = (2.0, 4.0, 6.0)


_BMW_OVERRIDES: dict[str, ParameterSpec] = {
    # BMW M Hybrid V8 front heave spring steps in 10 N/mm increments
    # in the iRacing UI (per BMWBounds.md "0-900 10n/mm intervals"),
    # not the global 5 N/mm default. Same path, same legal envelope —
    # only the render-time snap step changes.
    "heave_spring_rate_n_per_mm": ParameterSpec(
        json_path=_HEAVE_SPRING_RATE_F, dtype=float, units="N/mm",
        family="spring_rate", fittable=True, user_settable=True,
        step=10.0,
    ),
    # BMW M Hybrid V8 uses front torsion bars on the same pattern as
    # Cadillac (per BMWBounds.md lines 13–19). Identical YAML paths,
    # identical bound envelope, identical 14-diameter OD list. Same
    # LF=RF iRacing-UI symmetry as Cadillac — train LF only and mirror
    # to RF via the renderer.
    "torsion_bar_turns_fl": ParameterSpec(
        json_path=("Chassis", "LeftFront", "TorsionBarTurns"),
        dtype=float, units="turns", family="torsion_bar",
        fittable=True, user_settable=True, step=0.001,
    ),
    "torsion_bar_od_fl_mm": ParameterSpec(
        json_path=("Chassis", "LeftFront", "TorsionBarOD"),
        dtype=float, units="mm", family="torsion_bar",
        fittable=True, user_settable=True,
        discrete_values=_TORSION_BAR_OD_VALUES,
    ),
    # Categorical ARB size selection (front + rear). The iRacing UI
    # exposes these as named options (Disconnect / Soft / Medium /
    # Stiff); we store an integer index for the fitter and the renderer
    # maps the rounded index back to the label via ``ParameterSpec.choices``.
    "arb_size_front": ParameterSpec(
        json_path=("Chassis", "Front", "ArbSize"),
        dtype=float, units="", family="arb",
        fittable=True, user_settable=True, is_discrete=True,
        choices=_ARB_SIZE_CHOICES,
    ),
    "arb_size_rear": ParameterSpec(
        json_path=("Chassis", "Rear", "ArbSize"),
        dtype=float, units="", family="arb",
        fittable=True, user_settable=True, is_discrete=True,
        choices=_ARB_SIZE_CHOICES,
    ),
    # RearDiffSpec categoricals — both stored under
    # ``BrakesDriveUnit.RearDiffSpec`` for BMW. Coast/drive ramp angles
    # are stored as composite strings ("40/65"); clutch plates are
    # numeric (2/4/6).
    "diff_coast_drive_ramps": ParameterSpec(
        json_path=("BrakesDriveUnit", "RearDiffSpec", "CoastDriveRampAngles"),
        dtype=float, units="", family="diff",
        fittable=True, user_settable=True, is_discrete=True,
        choices=_DIFF_COAST_DRIVE_RAMP_CHOICES,
    ),
    "diff_clutch_friction_plates": ParameterSpec(
        json_path=("BrakesDriveUnit", "RearDiffSpec", "ClutchFrictionPlates"),
        dtype=float, units="plates", family="diff",
        fittable=True, user_settable=True, is_discrete=True,
        discrete_values=_DIFF_CLUTCH_PLATES_VALUES,
    ),
}

# Ferrari 499P exposes its torsion bar OD as an integer index (0..18),
# not the 14-discrete-mm list Cadillac/BMW use. Same channel name
# (`TorsionBarOD`) but radically different value space. Treated as a
# continuous integer here so DE searches the index range; renderer
# rounds to nearest int.
_FERRARI_ARB_SIZE_CHOICES: tuple[str, ...] = (
    "Disconnected", "A", "B", "C", "D", "E",
)


_FERRARI_OVERRIDES: dict[str, ParameterSpec] = {
    # Pushrod delta (per-axle, mm) — already named PushrodLengthDelta in
    # Ferrari YAML, distinct from the *Offset path other cars use.
    "pushrod_length_offset_front_mm": ParameterSpec(
        json_path=_PUSHROD_DELTA_F, dtype=float, units="mm",
        family="pushrod", fittable=True, user_settable=True, step=0.5,
    ),
    "pushrod_length_offset_rear_mm": ParameterSpec(
        json_path=_PUSHROD_DELTA_R, dtype=float, units="mm",
        family="pushrod", fittable=True, user_settable=True, step=0.5,
    ),
    "brake_bias_pct": ParameterSpec(
        json_path=("Systems", "BrakeSpec", "BrakePressureBias"),
        dtype=float, units="pct", family="brake_bias", fittable=True,
        step=0.5,
    ),
    # Rear differential preload — same units as default but path is
    # under Systems instead of BrakesDriveUnit.
    "diff_preload_nm": ParameterSpec(
        json_path=("Systems", "RearDiffSpec", "Preload"),
        dtype=float, units="Nm", family="diff", fittable=True, step=5.0,
    ),
    # Front differential preload — Ferrari has a separate front diff
    # (most other GTPs do not). Bound -50..+50 Nm per Ferraribounds.md.
    "front_diff_preload_nm": ParameterSpec(
        json_path=("Systems", "FrontDiffSpec", "Preload"),
        dtype=float, units="Nm", family="diff", fittable=True, step=5.0,
    ),
    # Heave springs — Ferrari stores BOTH front and rear as integer
    # indices, not N/mm. Front 0..8, rear 0..9. Reuses the existing
    # heave/third spring slot names so the per-axle "set this value"
    # row in the briefing still finds the right row, but units flip
    # to "index" so the renderer formats without N/mm.
    "heave_spring_rate_n_per_mm": ParameterSpec(
        json_path=_HEAVE_SPRING_RATE_F, dtype=float, units="index",
        family="spring_rate", fittable=True, user_settable=True, step=1.0,
    ),
    "third_spring_rate_n_per_mm": ParameterSpec(
        json_path=("Chassis", "Rear", "HeaveSpring"), dtype=float,
        units="index", family="spring_rate",
        fittable=True, user_settable=True, step=1.0,
    ),
    # Ferrari has no rear coil spring or rear coil-spring perch — the
    # rear ride is controlled by torsion bars + heave spring + heave
    # perch. Block the inherited entries so the optimizer doesn't
    # search a parameter that doesn't exist.
    "rear_coil_spring_rate_n_per_mm": _blocked_like(
        _common_bounded()["rear_coil_spring_rate_n_per_mm"],
    ),
    "spring_perch_offset_rear_mm": _blocked_like(
        _common_bounded()["spring_perch_offset_rear_mm"],
    ),
    # The "third perch offset" slot maps to Ferrari's REAR HEAVE PERCH
    # (the rear-axle equivalent of HeavePerchOffset). Wider envelope
    # than BMW/Cadillac (-150..+100 mm per Ferraribounds.md).
    "third_perch_offset_rear_mm": ParameterSpec(
        json_path=("Chassis", "Rear", "HeavePerchOffset"), dtype=float,
        units="mm", family="perch_offset",
        fittable=True, user_settable=True, step=0.5,
    ),
    # Torsion bars — Ferrari has them at all 4 corners (BMW/Cadillac
    # only have front). iRacing requires LF=RF and LR=RR symmetry, so
    # train ONE LF + ONE LR per turns + OD; the renderer mirrors RF
    # and RR via `_MIRRORED_LEAVES`. OD is an integer INDEX (0..18) on
    # Ferrari, not a discrete list of mm diameters — no
    # `discrete_values` here, DE searches the continuous 0..18 range
    # and the renderer rounds to int.
    "torsion_bar_turns_fl": ParameterSpec(
        json_path=("Chassis", "LeftFront", "TorsionBarTurns"),
        dtype=float, units="turns", family="torsion_bar",
        fittable=True, user_settable=True, step=0.125,
    ),
    "torsion_bar_od_fl_mm": ParameterSpec(
        json_path=("Chassis", "LeftFront", "TorsionBarOD"),
        dtype=float, units="index", family="torsion_bar",
        fittable=True, user_settable=True, step=1.0,
    ),
    "torsion_bar_turns_rl": ParameterSpec(
        json_path=("Chassis", "LeftRear", "TorsionBarTurns"),
        dtype=float, units="turns", family="torsion_bar",
        fittable=True, user_settable=True, step=0.125,
    ),
    "torsion_bar_od_rl_mm": ParameterSpec(
        json_path=("Chassis", "LeftRear", "TorsionBarOD"),
        dtype=float, units="index", family="torsion_bar",
        fittable=True, user_settable=True, step=1.0,
    ),
    # ARB Size — Ferrari uses letter labels (Disconnected, A, B, C, D, E)
    # at both axles, where BMW uses Disconnect/Soft/Medium/Stiff. Wholly
    # different choice list; the ordinal still ascends in stiffness
    # (Disconnected = 0 = softest, E = 5 = stiffest).
    "arb_size_front": ParameterSpec(
        json_path=("Chassis", "Front", "ArbSize"),
        dtype=float, units="", family="arb",
        fittable=True, user_settable=True, is_discrete=True,
        choices=_FERRARI_ARB_SIZE_CHOICES,
    ),
    "arb_size_rear": ParameterSpec(
        json_path=("Chassis", "Rear", "ArbSize"),
        dtype=float, units="", family="arb",
        fittable=True, user_settable=True, is_discrete=True,
        choices=_FERRARI_ARB_SIZE_CHOICES,
    ),
    # Ferrari fuel YAML carries the canonical value at Systems.Fuel
    # (also mirrored at Chassis.Rear.FuelLevel). Use the Systems path
    # for the optimizer.
    "fuel_level_l": ParameterSpec(
        json_path=("Systems", "Fuel", "FuelLevel"), dtype=float,
        units="L", family="fuel", fittable=True, user_settable=True,
        step=1.0,
    ),
}

_PORSCHE_OVERRIDES: dict[str, ParameterSpec] = {
    "anti_roll_bar_front": ParameterSpec(
        json_path=("Chassis", "Front", "ArbAdj"), dtype=float, units="click",
        family="arb", fittable=True, is_discrete=True,
    ),
    "anti_roll_bar_rear": ParameterSpec(
        json_path=("Chassis", "Rear", "ArbAdj"), dtype=float, units="click",
        family="arb", fittable=True, is_discrete=True,
    ),
}

ACURA: dict[str, ParameterSpec] = _build(_SPLIT_DAMPERS, **_ACURA_OVERRIDES)
BMW: dict[str, ParameterSpec] = _build(_INLINE_DAMPERS, **_BMW_OVERRIDES)
CADILLAC: dict[str, ParameterSpec] = _build(_INLINE_DAMPERS, **_CADILLAC_OVERRIDES)
FERRARI: dict[str, ParameterSpec] = _build(_FERRARI_DAMPERS, **_FERRARI_OVERRIDES)
PORSCHE: dict[str, ParameterSpec] = _build(_PORSCHE_DAMPERS, **_PORSCHE_OVERRIDES)

_BY_CAR: dict[str, dict[str, ParameterSpec]] = {
    "acura": ACURA,
    "bmw": BMW,
    "cadillac": CADILLAC,
    "ferrari": FERRARI,
    "porsche": PORSCHE,
}


def ontology_for(car: str) -> dict[str, ParameterSpec]:
    """Return the per-car ontology dict; raises KeyError on unknown car."""
    key = car.strip().lower()
    if key not in _BY_CAR:
        raise KeyError(
            f"unknown car: {car!r}; expected one of {sorted(_BY_CAR)}"
        )
    return _BY_CAR[key]


def parameters(car: str) -> list[str]:
    """Sorted list of every typed parameter name for `car`."""
    return sorted(ontology_for(car).keys())


def fittable_parameters(car: str, table: ConstraintsTable) -> list[str]:
    """Parameters that the OPTIMIZER may search over.

    Three gates, all required:

    1. ``spec.fittable=True`` — model is allowed to learn correlations on
       this parameter.
    2. ``spec.user_settable=True`` — driver can type this value into the
       iRacing garage UI. Calculated readouts (ride heights, deflections,
       corner weights, etc.) are excluded — the optimizer must not produce
       a value the user cannot enter.
    3. ``constraints.md`` provides bounds — CE-gated parameters whose
       bounds are still ``<TODO: from iRacing UI>`` are excluded until
       captured.
    """
    onto = ontology_for(car)
    out: list[str] = []
    for name, spec in onto.items():
        if not spec.fittable:
            continue
        if not spec.user_settable:
            continue
        bound = table.bounds(car.strip().lower(), name)
        if bound is None:
            continue
        out.append(name)
    return sorted(out)


def setup_value(car: str, parameter: str, setup_json: dict | str) -> float | None:
    """Resolve `parameter` against a session's setup blob.

    Accepts either a parsed dict or a raw JSON string (slice A persists JSON).
    Returns `None` when the JSON path is missing or the value cannot be
    coerced to a float — the fitter drops such rows rather than aborting.

    Categorical parameters (``ParameterSpec.choices`` non-empty) map the
    YAML's string label to its integer index in the choices tuple. The
    fitter then trains on the ordinal-encoded index; the renderer maps
    the rounded index back to a label at display time. Unknown labels
    return None so the fitter can drop the row instead of guessing.
    """
    onto = ontology_for(car)
    if parameter not in onto:
        raise KeyError(f"parameter {parameter!r} not in ontology for car={car!r}")
    spec = onto[parameter]

    if isinstance(setup_json, str):
        try:
            setup = json.loads(setup_json)
        except json.JSONDecodeError:
            return None
    else:
        setup = setup_json
    if not isinstance(setup, dict):
        return None

    raw = _walk(setup, spec.json_path)
    if spec.choices and isinstance(raw, str):
        raw_norm = raw.strip().lower()
        for idx, label in enumerate(spec.choices):
            if label.strip().lower() == raw_norm:
                return float(idx)
        return None
    return _scalar(raw)


__all__ = [
    "ACURA",
    "BMW",
    "CADILLAC",
    "FERRARI",
    "PORSCHE",
    "Family",
    "ParameterSpec",
    "fittable_parameters",
    "ontology_for",
    "parameters",
    "setup_value",
]
