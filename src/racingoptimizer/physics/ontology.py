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
            family="rear_wing", fittable=True,
        ),
        "tyre_cold_pressure_kpa": ParameterSpec(
            json_path=_TYRE_FL, dtype=float, units="kPa",
            family="tyre_pressure", fittable=True,
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
        # results. Ranges in `constraints.md` are estimates.
        "heave_spring_rate_n_per_mm": ParameterSpec(
            json_path=_HEAVE_SPRING_RATE_F, dtype=float, units="N/mm",
            family="spring_rate", fittable=True, user_settable=True,
        ),
        "third_spring_rate_n_per_mm": ParameterSpec(
            json_path=_THIRD_SPRING_RATE_R, dtype=float, units="N/mm",
            family="spring_rate", fittable=True, user_settable=True,
        ),
        "rear_coil_spring_rate_n_per_mm": ParameterSpec(
            json_path=_REAR_COIL_SPRING_RATE, dtype=float, units="N/mm",
            family="spring_rate", fittable=True, user_settable=True,
        ),
        "heave_perch_offset_front_mm": ParameterSpec(
            json_path=_HEAVE_PERCH_OFFSET_F, dtype=float, units="mm",
            family="perch_offset", fittable=True, user_settable=True,
        ),
        "spring_perch_offset_rear_mm": ParameterSpec(
            json_path=_SPRING_PERCH_OFFSET_R, dtype=float, units="mm",
            family="perch_offset", fittable=True, user_settable=True,
        ),
        "third_perch_offset_rear_mm": ParameterSpec(
            json_path=_THIRD_PERCH_OFFSET_R, dtype=float, units="mm",
            family="perch_offset", fittable=True, user_settable=True,
        ),
        "pushrod_length_offset_front_mm": ParameterSpec(
            json_path=_PUSHROD_OFFSET_F, dtype=float, units="mm",
            family="pushrod", fittable=True, user_settable=True,
        ),
        "pushrod_length_offset_rear_mm": ParameterSpec(
            json_path=_PUSHROD_OFFSET_R, dtype=float, units="mm",
            family="pushrod", fittable=True, user_settable=True,
        ),
        # Per-corner camber. Direct setup → tire-grip lever. Bounds in
        # `constraints.md` are non-placeholder (front -2.9..0, rear -1.9..0).
        "camber_fl_deg": ParameterSpec(
            json_path=_CAMBER_LF, dtype=float, units="deg",
            family="camber", fittable=True, user_settable=True,
        ),
        "camber_fr_deg": ParameterSpec(
            json_path=_CAMBER_RF, dtype=float, units="deg",
            family="camber", fittable=True, user_settable=True,
        ),
        "camber_rl_deg": ParameterSpec(
            json_path=_CAMBER_LR, dtype=float, units="deg",
            family="camber", fittable=True, user_settable=True,
        ),
        "camber_rr_deg": ParameterSpec(
            json_path=_CAMBER_RR, dtype=float, units="deg",
            family="camber", fittable=True, user_settable=True,
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
        ),
        "corner_weight_fr_kg": ParameterSpec(
            json_path=("Chassis", "RightFront", "CornerWeight"), dtype=float,
            units="N", family="corner_weight", fittable=False,
        ),
        "corner_weight_rl_kg": ParameterSpec(
            json_path=("Chassis", "LeftRear", "CornerWeight"), dtype=float,
            units="N", family="corner_weight", fittable=False,
        ),
        "corner_weight_rr_kg": ParameterSpec(
            json_path=("Chassis", "RightRear", "CornerWeight"), dtype=float,
            units="N", family="corner_weight", fittable=False,
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


# (constraint suffix, IBT field name) for the four damper modes.
_DAMPER_MODES: tuple[tuple[str, str], ...] = (
    ("lsc", "LsCompDamping"),
    ("hsc", "HsCompDamping"),
    ("lsr", "LsRbdDamping"),
    ("hsr", "HsRbdDamping"),
)


def _damper_paths(
    corner_to_path: tuple[tuple[str, tuple[str, ...]], ...],
) -> dict[str, ParameterSpec]:
    """Generate `damper_<mode>_<corner>` entries from a (corner_code → path) map."""
    out: dict[str, ParameterSpec] = {}
    for code, parent_path in corner_to_path:
        for suffix, field_name in _DAMPER_MODES:
            out[f"damper_{suffix}_{code}"] = ParameterSpec(
                json_path=(*parent_path, field_name), dtype=float, units="click",
                family="damper", fittable=False, is_discrete=True,
            )
    return out


# BMW / Cadillac / Ferrari inline per-corner damper clicks under Chassis.<corner>.
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
}

_CADILLAC_OVERRIDES: dict[str, ParameterSpec] = {
    "diff_preload_nm": ParameterSpec(
        json_path=("BrakesDriveUnit", "DiffSpec", "Preload"),
        dtype=float, units="Nm", family="diff", fittable=True,
    ),
}

_FERRARI_OVERRIDES: dict[str, ParameterSpec] = {
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
    # Ferrari stores heave/torsion controls as UI indices, not N/mm values.
    # Leave them known but blocked until Ferrari-specific legal ranges are captured.
    "heave_spring_rate_n_per_mm": ParameterSpec(
        json_path=_HEAVE_SPRING_RATE_F, dtype=float, units="index",
        family="spring_rate", fittable=False, user_settable=True,
    ),
    "third_spring_rate_n_per_mm": ParameterSpec(
        json_path=("Chassis", "Rear", "HeaveSpring"), dtype=float, units="index",
        family="spring_rate", fittable=False, user_settable=True,
    ),
    "rear_coil_spring_rate_n_per_mm": _blocked_like(
        _common_bounded()["rear_coil_spring_rate_n_per_mm"],
    ),
    "spring_perch_offset_rear_mm": _blocked_like(_common_bounded()["spring_perch_offset_rear_mm"]),
    "third_perch_offset_rear_mm": _blocked_like(_common_bounded()["third_perch_offset_rear_mm"]),
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
BMW: dict[str, ParameterSpec] = _build(_INLINE_DAMPERS)
CADILLAC: dict[str, ParameterSpec] = _build(_INLINE_DAMPERS, **_CADILLAC_OVERRIDES)
FERRARI: dict[str, ParameterSpec] = _build(_INLINE_DAMPERS, **_FERRARI_OVERRIDES)
PORSCHE: dict[str, ParameterSpec] = _build(_SPLIT_DAMPERS, **_PORSCHE_OVERRIDES)

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
