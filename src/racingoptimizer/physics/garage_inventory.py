"""Garage setup inventory and classification helpers.

VISION requires every setup leaf to be either modeled, recommended, or
dropped with an auditable reason. This module provides the small shared
taxonomy used by tests and build reports to make that explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from racingoptimizer.constraints import ConstraintsTable
from racingoptimizer.physics.ontology import ontology_for

Classification = Literal[
    "optimized_user_input",
    "modeled_readout",
    "blocked_user_input",
    "unsupported_readout",
    "unsupported_non_setup",
]


@dataclass(frozen=True, slots=True)
class GarageLeaf:
    path: tuple[str, ...]
    value: Any
    classification: Classification
    parameter: str | None
    reason: str


def flatten_setup(setup: dict[str, Any]) -> dict[tuple[str, ...], Any]:
    """Return every non-dict setup leaf keyed by its tuple path."""
    leaves: dict[tuple[str, ...], Any] = {}

    def walk(obj: Any, prefix: tuple[str, ...]) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(value, (*prefix, str(key)))
            return
        leaves[prefix] = obj

    walk(setup, ())
    return leaves


def inventory_setup(
    car: str,
    setup: dict[str, Any],
    constraints: ConstraintsTable,
) -> list[GarageLeaf]:
    """Classify every garage setup leaf for `car`."""
    onto = ontology_for(car)
    path_to_param = {spec.json_path: name for name, spec in onto.items()}
    leaves: list[GarageLeaf] = []
    for path, value in sorted(flatten_setup(setup).items()):
        parameter = path_to_param.get(path)
        if parameter is not None:
            spec = onto[parameter]
            if not spec.user_settable:
                leaves.append(GarageLeaf(
                    path=path,
                    value=value,
                    classification="modeled_readout",
                    parameter=parameter,
                    reason="calculated setup readout, not enterable by driver",
                ))
            elif spec.fittable and constraints.bounds(car, parameter) is not None:
                leaves.append(GarageLeaf(
                    path=path,
                    value=value,
                    classification="optimized_user_input",
                    parameter=parameter,
                    reason="bounded user-settable input in optimizer search space",
                ))
            else:
                leaves.append(GarageLeaf(
                    path=path,
                    value=value,
                    classification="blocked_user_input",
                    parameter=parameter,
                    reason="known user input but missing safe bounds or fit enablement",
                ))
            continue

        classification, reason = classify_unmapped_path(path)
        leaves.append(GarageLeaf(
            path=path,
            value=value,
            classification=classification,
            parameter=None,
            reason=reason,
        ))
    return leaves


def classify_unmapped_path(path: tuple[str, ...]) -> tuple[Classification, str]:
    """Classify an unmapped setup leaf using conservative garage semantics."""
    dotted = ".".join(path)
    last = path[-1].lower() if path else ""

    if dotted == "UpdateCount":
        return "unsupported_non_setup", "setup blob metadata counter"

    readout_terms = (
        "defl",
        "rideheight",
        "at speed",
        "atspeed",
        "downforce",
        "ld",
        "balance",
        "lasthotpressure",
        "lasttemps",
        "treadremaining",
        "crossweight",
        "cornerweight",
    )
    if any(term in dotted.lower() for term in readout_terms):
        return "unsupported_readout", "calculated readout or telemetry carry-over"

    blocked_terms = (
        "toe",
        "arb",
        "damp",
        "spring",
        "perch",
        "pushrod",
        "torsion",
        "brake",
        "diff",
        "fuel",
        "gear",
        "tractioncontrol",
        "throttle",
        "hybrid",
        "mastercyl",
        "padcompound",
        "tiretype",
        "startingpressure",
        "wing",
        "lighting",
        "headlight",
        "roofid",
    )
    if any(term in dotted.lower() for term in blocked_terms):
        return "blocked_user_input", "user-settable garage input lacks verified optimizer mapping"

    if last in {"speedinfirst", "speedinsecond", "speedinthird", "speedinfourth",
                "speedinfifth", "speedinsixth", "speedinseventh"}:
        return "unsupported_readout", "gear-stack calculated speed readout"

    return "blocked_user_input", "unmapped garage leaf requires explicit classification"


__all__ = [
    "Classification",
    "GarageLeaf",
    "classify_unmapped_path",
    "flatten_setup",
    "inventory_setup",
]
