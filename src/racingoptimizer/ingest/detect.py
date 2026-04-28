"""Car & track detection / normalization.

The IBT YAML header is authoritative for the canonical car name when present;
the filename is used as a fallback. Track names always come from a stringified
source (YAML or filename) and get slugified.
"""
from __future__ import annotations

import re
from pathlib import Path

# Maps known iRacing car identifiers (as they appear in IBT YAML or filenames)
# to the canonical key used in `aero-maps/` and the corpus directory layout.
# Order matters: longest-prefix wins.
CAR_PREFIX_MAP: dict[str, str] = {
    "acuraarx06gtp": "acura",
    "bmwlmdh": "bmw",
    "bmwm4gt3": "bmw",
    "cadillacvseriesrgtp": "cadillac",
    "ferrari499p": "ferrari",
    "porsche963gtp": "porsche",
    "porsche992rgt3": "porsche",
    "amvantageevogt3": "bmw",   # NOTE: not GTP — placeholder mapping until we
                                # confirm where Aston Martin GT3 fits in the model.
}


class UnknownCarError(ValueError):
    """Raised when a car identifier cannot be mapped to a canonical key."""


def normalize_car_key(raw: str) -> str:
    """Map a raw iRacing car identifier to a canonical car key.

    >>> normalize_car_key("acuraarx06gtp")
    'acura'
    """
    raw = raw.strip().lower()
    # longest-prefix match
    for prefix in sorted(CAR_PREFIX_MAP, key=len, reverse=True):
        if raw.startswith(prefix):
            return CAR_PREFIX_MAP[prefix]
    raise UnknownCarError(raw)


def slugify_track(raw: str) -> str:
    """Lowercase, replace whitespace runs and punctuation with single underscores."""
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


_FILENAME_RE = re.compile(
    r"^(?P<car>[a-z0-9]+)_(?P<track>.+?)(?:\s+\d{4}-\d{2}-\d{2}\s+\d{2}-\d{2}-\d{2})?\.ibt$"
)


def detect_track_from_filename(filename: str) -> str | None:
    """Pull a track slug out of an iRacing-style filename, or None on no match."""
    name = Path(filename).name.lower()
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    return slugify_track(m["track"])


def detect_car_from_filename(filename: str) -> str | None:
    """Pull the raw car identifier out of an iRacing-style filename, or None."""
    m = _FILENAME_RE.match(Path(filename).name.lower())
    return m["car"] if m else None


def detect_car(*, yaml_car: str | None, filename_car: str | None) -> str:
    """Pick the canonical car key. YAML wins when present and known."""
    if yaml_car:
        try:
            return normalize_car_key(yaml_car)
        except UnknownCarError:
            pass
    if filename_car:
        return normalize_car_key(filename_car)
    raise UnknownCarError("no car id available from yaml or filename")
