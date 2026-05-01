"""Parse `constraints.md` -> ConstraintsTable.

The live `constraints.md` is the source of truth. The parser supports four
table shapes that appear in it: single-row `min|max`, multi-row `parameter`,
multi-row `corner`, and the `parameter|unit|min|max` differential variant.
TODO placeholders (`<TODO: ...>`) become `None` bounds so the recommender can
warn about unbounded parameters rather than emit values for them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

CANONICAL_CARS = ("acura", "bmw", "cadillac", "ferrari", "porsche")
DEFAULT_KEY = "default"


class ConstraintsParseError(ValueError):
    """Raised when constraints.md cannot be parsed."""


@dataclass(frozen=True)
class ConstraintsTable:
    """Per-car bounds table. Outer key is `default` or a canonical car key."""

    _by_car: dict[str, dict[str, tuple[float, float] | None]] = field(default_factory=dict)

    def bounds(self, car: str, parameter: str) -> tuple[float, float] | None:
        car_table = self._by_car.get(car, {})
        if parameter in car_table:
            return car_table[parameter]
        return self._by_car.get(DEFAULT_KEY, {}).get(parameter)

    def parameters(self, car: str | None = None) -> list[str]:
        keys: set[str] = set(self._by_car.get(DEFAULT_KEY, {}))
        if car is not None and car in self._by_car:
            keys.update(self._by_car[car])
        elif car is None:
            for table in self._by_car.values():
                keys.update(table)
        return sorted(keys)


def _default_constraints_path() -> Path:
    return Path(__file__).resolve().parents[3] / "constraints.md"


_TODO_RE = re.compile(r"<TODO[^>]*>")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_cell(cell: str) -> float | None:
    """Return float or None if cell is a TODO placeholder."""
    cell = cell.strip()
    if _TODO_RE.search(cell):
        return None
    m = _NUM_RE.search(cell)
    if m is None:
        raise ConstraintsParseError(f"no numeric value in cell: {cell!r}")
    return float(m.group(0))


def _bounds_from_cells(min_cell: str, max_cell: str) -> tuple[float, float] | None:
    lo = _parse_cell(min_cell)
    hi = _parse_cell(max_cell)
    if lo is None or hi is None:
        return None
    if lo > hi:
        raise ConstraintsParseError(f"min {lo} > max {hi}")
    return (lo, hi)


# --- name derivation ---------------------------------------------------------

DAMPER_SUFFIX = {
    "low speed compression": "lsc",
    "high speed compression": "hsc",
    "low speed rebound": "lsr",
    "high speed rebound": "hsr",
}


def _section_to_param_base(heading: str) -> tuple[str, str | None] | None:
    """Map a section heading to ``(base_key, default_unit_suffix)``.

    The unit suffix is appended only for single-row sections; corner/multi-row
    sections build their own keys via `_compose_param_key`.

    Returns ``None`` for headings the loader does not understand. The caller
    is expected to skip such sections, which lets ``constraints.md`` carry
    annotation-only sub-sections (e.g. parenthetical "(USER input — TODO
    bounds)" notes for params not yet captured from the iRacing UI) without
    breaking the parse. Unknown sections were previously a hard error — that
    made even one-line documentation drift fatal to the optimizer.
    """
    h = heading.strip().lower()

    if h == "rear wing angle":
        return ("rear_wing_angle", "deg")
    if h == "tyre cold pressure":
        return ("tyre_cold_pressure", "kpa")
    if h == "suspension deflections":
        return ("", None)
    if h == "static ride height":
        return ("static_ride_height", "mm")
    # USER-input families that drive the calculated readouts above. Bound
    # ranges are estimates — see the prose annotation in `constraints.md`.
    if h == "heave spring rate":
        return ("heave_spring_rate_n_per_mm", None)
    if h == "rear third spring rate":
        return ("third_spring_rate_n_per_mm", None)
    if h == "rear coil spring rate":
        return ("rear_coil_spring_rate_n_per_mm", None)
    if h == "heave perch offset front":
        return ("heave_perch_offset_front_mm", None)
    if h == "spring perch offset rear":
        return ("spring_perch_offset_rear_mm", None)
    if h == "third perch offset rear":
        return ("third_perch_offset_rear_mm", None)
    if h == "pushrod length offset front":
        return ("pushrod_length_offset_front_mm", None)
    if h == "pushrod length offset rear":
        return ("pushrod_length_offset_rear_mm", None)
    if h.startswith("anti-roll bar"):
        side = h.split("—", 1)[1].strip() if "—" in h else h.rsplit(maxsplit=1)[-1]
        return (f"anti_roll_bar_{side}", None)
    if h.startswith("damper"):
        body = h.split("—", 1)[1].strip() if "—" in h else h
        body = re.sub(r"\(.*\)", "", body).strip()
        suffix = DAMPER_SUFFIX.get(body)
        if suffix is None:
            raise ConstraintsParseError(f"unknown damper section: {heading!r}")
        return (f"damper_{suffix}", None)
    if h.startswith("corner weight"):
        return ("corner_weight", "kg")
    if h == "brake bias":
        return ("brake_bias", "pct")
    if h == "differential":
        return ("diff", None)
    if h == "camber":
        return ("camber", "deg")
    if h == "toe":
        return ("toe", "deg")
    if h.startswith("brake duct opening"):
        side = h.split("—", 1)[1].strip() if "—" in h else h.rsplit(maxsplit=1)[-1]
        return (f"brake_duct_{side}", None)
    if h.startswith("throttle") and "brake" in h and "mapping" in h:
        return ("throttle_brake_mapping", None)
    return None


def _slug(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


_UNIT_SYMBOLS = {"%": "pct", "°": "deg"}


def _normalize_unit(unit: str) -> str:
    u = unit.strip()
    return _UNIT_SYMBOLS.get(u, _slug(u))


def _compose_corner_key(base: str, corner: str, unit: str | None) -> str:
    parts = [base, _slug(corner)]
    if unit:
        parts.append(unit)
    return "_".join(p for p in parts if p)


# --- table parsing -----------------------------------------------------------

def _split_row(line: str) -> list[str]:
    line = line.strip()
    if not line.startswith("|"):
        return []
    inner = line[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [c.strip() for c in inner.split("|")]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(set(c) <= set("-: ") and c for c in cells)


def _parse_table(lines: list[str], start: int) -> tuple[list[str], list[list[str]], int]:
    """Return (header_cells, data_rows, next_index)."""
    header = _split_row(lines[start])
    if not header:
        raise ConstraintsParseError(f"expected table header at line {start + 1}")
    if start + 1 >= len(lines):
        raise ConstraintsParseError("table truncated after header")
    sep = _split_row(lines[start + 1])
    if not _is_separator(sep):
        raise ConstraintsParseError(f"expected table separator at line {start + 2}")
    rows: list[list[str]] = []
    i = start + 2
    while i < len(lines):
        cells = _split_row(lines[i])
        if not cells:
            break
        rows.append(cells)
        i += 1
    return header, rows, i


# --- main loader -------------------------------------------------------------

_HEADING_RE = re.compile(r"^###\s+(.+?)\s*$")
_H2_RE = re.compile(r"^##\s+(.+?)\s*$")
_OVERRIDE_RE = re.compile(
    r"^-\s+\*\*(?P<param>[^:*]+):\*\*\s+"
    r"(?P<lo>-?\d+(?:\.\d+)?)[^\d\n]*?"
    r"(?P<hi>-?\d+(?:\.\d+)?)"
)


def load_constraints(path: Path | None = None) -> ConstraintsTable:
    p = Path(path) if path is not None else _default_constraints_path()
    if not p.is_file():
        raise FileNotFoundError(f"constraints.md not found at {p}")
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()

    by_car: dict[str, dict[str, tuple[float, float] | None]] = {DEFAULT_KEY: {}}
    section: str | None = None  # 'defaults' | 'overrides'
    current_car: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        h2 = _H2_RE.match(line)
        if h2:
            title = h2.group(1).lower()
            if title.startswith("defaults"):
                section = "defaults"
                current_car = None
            elif title.startswith("per-car overrides"):
                section = "overrides"
                current_car = None
            else:
                section = None
            i += 1
            continue

        if section == "defaults":
            h3 = _HEADING_RE.match(line)
            if h3:
                heading = h3.group(1)
                resolved = _section_to_param_base(heading)
                if resolved is None:
                    # Unknown / annotation-only section. Skip the heading and
                    # keep scanning — let other recognised sections continue
                    # to populate the table normally.
                    i += 1
                    continue
                base, default_unit = resolved
                # Find next table
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith("|"):
                    j += 1
                if j >= len(lines):
                    raise ConstraintsParseError(f"no table for section {heading!r}")
                header, rows, after = _parse_table(lines, j)
                _ingest_section(by_car[DEFAULT_KEY], heading, base, default_unit, header, rows)
                i = after
                continue
            i += 1
            continue

        if section == "overrides":
            h3 = _HEADING_RE.match(line)
            if h3:
                car = h3.group(1).strip().lower()
                current_car = car if car in CANONICAL_CARS else None
                if current_car is not None:
                    by_car.setdefault(current_car, {})
                i += 1
                continue
            if current_car is not None:
                m = _OVERRIDE_RE.match(line)
                if m:
                    resolved = _section_to_param_base(m.group("param"))
                    if resolved is not None:
                        base, default_unit = resolved
                        key = base if default_unit is None else f"{base}_{default_unit}"
                        by_car[current_car][key] = (float(m["lo"]), float(m["hi"]))
            i += 1
            continue

        i += 1

    return ConstraintsTable(_by_car=by_car)


def _ingest_section(
    table: dict[str, tuple[float, float] | None],
    heading: str,
    base: str,
    default_unit: str | None,
    header: list[str],
    rows: list[list[str]],
) -> None:
    cols = [h.lower() for h in header]
    h_lower = heading.strip().lower()

    if cols == ["min", "max"]:
        if len(rows) != 1:
            raise ConstraintsParseError(f"{heading}: expected 1 data row, got {len(rows)}")
        key = base if default_unit is None else f"{base}_{default_unit}"
        table[key] = _bounds_from_cells(rows[0][0], rows[0][1])
        return

    if cols == ["parameter", "min", "max"]:
        for param_name, min_c, max_c in (r[:3] for r in rows):
            if h_lower == "suspension deflections":
                key = f"{_slug(param_name)}_mm"
            elif "throttle" in h_lower:
                key = "throttle_brake_mapping"
            else:
                key = f"{base}_{_slug(param_name)}"
                if default_unit:
                    key = f"{key}_{default_unit}"
            table[key] = _bounds_from_cells(min_c, max_c)
        return

    if cols == ["corner", "min", "max"]:
        for corner, min_c, max_c in (r[:3] for r in rows):
            table[_compose_corner_key(base, corner, default_unit)] = _bounds_from_cells(
                min_c, max_c
            )
        return

    if cols == ["parameter", "unit", "min", "max"]:
        for param_name, unit, min_c, max_c in (r[:4] for r in rows):
            unit_slug = _normalize_unit(unit)
            key = f"{base}_{_slug(param_name)}"
            if unit_slug:
                key = f"{key}_{unit_slug}"
            table[key] = _bounds_from_cells(min_c, max_c)
        return

    raise ConstraintsParseError(f"{heading}: unsupported table header {header!r}")
