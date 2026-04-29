"""Disk -> AeroMapData. Pure data parsing; no interpolation, no scipy.

Reads `aero-maps/<car>_wing_<X.X>.json` files and returns an immutable
container with axes and 3D matrices stacked along the wing dimension.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REQUIRED_KEYS = ("car", "wing", "front_rh_mm", "rear_rh_mm", "balance_pct", "ld_ratio")
EXPECTED_FRONT_RH_LEN = 51
EXPECTED_REAR_RH_LEN = 46

_FILENAME_RE = re.compile(r"^(?P<car>[a-z0-9]+)_wing_(?P<wing>\d+\.\d+)\.json$")


class AeroLoadError(ValueError):
    """Raised when an aero JSON fails schema validation or a car has no maps."""


@dataclass(frozen=True)
class AeroMapData:
    """Stacked aero data for one car across all loaded wing angles.

    Wing axis is leading so each per-wing 2D slice is a contiguous (51, 46)
    array — cheap to hand to a per-wing RegularGridInterpolator.
    """

    car: str
    wing_angles: tuple[float, ...]
    front_rh_mm: np.ndarray
    rear_rh_mm: np.ndarray
    balance_pct: np.ndarray
    ld_ratio: np.ndarray


def parse_aero_filename(name: str) -> tuple[str, float]:
    """Pull (car, wing_deg) from `<car>_wing_<X.X>.json`."""
    m = _FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"not an aero-map filename: {name!r}")
    return m["car"], float(m["wing"])


def load_aero_map_data(car: str, *, aero_dir: Path) -> AeroMapData:
    """Load every <car>_wing_*.json under aero_dir and stack into AeroMapData."""
    aero_dir = Path(aero_dir)
    if not aero_dir.is_dir():
        raise FileNotFoundError(f"aero directory not found: {aero_dir}")

    matches: list[tuple[float, Path]] = []
    for entry in sorted(aero_dir.iterdir()):
        if not entry.is_file() or not entry.name.endswith(".json"):
            continue
        try:
            file_car, file_wing = parse_aero_filename(entry.name)
        except ValueError:
            continue
        if file_car == car:
            matches.append((file_wing, entry))

    if not matches:
        raise AeroLoadError(f"no aero maps for car {car!r} under {aero_dir}")

    matches.sort(key=lambda x: x[0])

    front_rh: np.ndarray | None = None
    rear_rh: np.ndarray | None = None
    balance_stack: list[np.ndarray] = []
    ld_stack: list[np.ndarray] = []
    wings: list[float] = []

    for filename_wing, path in matches:
        payload = json.loads(path.read_text())
        for key in REQUIRED_KEYS:
            if key not in payload:
                raise AeroLoadError(f"{path.name}: missing required key {key!r}")

        if payload["car"] != car:
            raise AeroLoadError(
                f"{path.name}: payload car {payload['car']!r} != filename car {car!r}"
            )
        if float(payload["wing"]) != filename_wing:
            raise AeroLoadError(
                f"{path.name}: payload wing {payload['wing']} != filename wing {filename_wing}"
            )

        f_axis = np.asarray(payload["front_rh_mm"], dtype=float)
        r_axis = np.asarray(payload["rear_rh_mm"], dtype=float)
        if f_axis.shape != (EXPECTED_FRONT_RH_LEN,):
            raise AeroLoadError(
                f"{path.name}: front_rh_mm length {f_axis.shape[0]} != {EXPECTED_FRONT_RH_LEN}"
            )
        if r_axis.shape != (EXPECTED_REAR_RH_LEN,):
            raise AeroLoadError(
                f"{path.name}: rear_rh_mm length {r_axis.shape[0]} != {EXPECTED_REAR_RH_LEN}"
            )

        if front_rh is None:
            front_rh, rear_rh = f_axis, r_axis
        elif not (np.array_equal(front_rh, f_axis) and np.array_equal(rear_rh, r_axis)):
            raise AeroLoadError(
                f"{path.name}: rh axis disagrees with sibling wing files for car {car!r}"
            )

        balance = np.asarray(payload["balance_pct"], dtype=float)
        ld = np.asarray(payload["ld_ratio"], dtype=float)
        expected_shape = (EXPECTED_FRONT_RH_LEN, EXPECTED_REAR_RH_LEN)
        if balance.shape != expected_shape:
            raise AeroLoadError(
                f"{path.name}: balance_pct shape {balance.shape} != {expected_shape}"
            )
        if ld.shape != expected_shape:
            raise AeroLoadError(
                f"{path.name}: ld_ratio shape {ld.shape} != {expected_shape}"
            )
        if np.isnan(balance).any() or np.isnan(ld).any():
            raise AeroLoadError(f"{path.name}: NaN cell in balance_pct or ld_ratio")

        wings.append(filename_wing)
        balance_stack.append(balance)
        ld_stack.append(ld)

    assert front_rh is not None and rear_rh is not None

    return AeroMapData(
        car=car,
        wing_angles=tuple(wings),
        front_rh_mm=front_rh,
        rear_rh_mm=rear_rh,
        balance_pct=np.stack(balance_stack, axis=0),
        ld_ratio=np.stack(ld_stack, axis=0),
    )
