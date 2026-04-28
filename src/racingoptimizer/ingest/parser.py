"""Wrap `pyirsdk` (a.k.a. `irsdk`) into a typed ParseResult.

Reads each scalar telemetry channel into a float32 numpy array and parses the
embedded YAML session info to extract car / track / setup / weather context.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import numpy as np

from racingoptimizer.ingest.segment import LapSpan, detect_lap_boundaries


# Substring patterns that, when present in a channel name, cause us to drop
# that channel from the corpus. Multi-driver arrays (`CarIdx*`) and per-tyre
# per-spot temperature/pressure spreads are the worst offenders for disk/IO.
EXCLUDED_CHANNEL_PATTERNS: tuple[str, ...] = (
    "CarIdx",          # 64-element multi-driver arrays
    "TempCM",          # per-tyre per-spot temp arrays in some IBT versions
    "TempCL",
    "TempCR",
)


class ParseResult(NamedTuple):
    yaml_car: str
    yaml_track: str
    recorded_at: str | None    # ISO timestamp from YAML header if present
    duration_s: float
    channels: dict[str, np.ndarray]      # name -> float32 1-D array, all same length
    setup: dict                          # nested garage setup as parsed from YAML
    weather_summary: dict                # JSON-friendly summary of weather channels
    lap_spans: list[LapSpan]


def _excluded(name: str) -> bool:
    return any(p in name for p in EXCLUDED_CHANNEL_PATTERNS)


def _read_yaml(ibt) -> dict:
    """Pull the YAML session info blob out of an opened IBT and parse it.

    pyirsdk 1.3.5's `IBT` does not expose a YAML helper directly (only the
    live shared-memory `IRSDK` class does). Replicate the relevant logic.
    """
    import yaml as _yaml
    from irsdk import (
        YAML_CODE_PAGE,
        YAML_TRANSLATER,
        CustomYamlSafeLoader,
        YamlReader,
    )

    hdr = ibt._header
    data = bytes(
        ibt._shared_mem[hdr.session_info_offset : hdr.session_info_offset + hdr.session_info_len]
    )
    yaml_src = re.sub(
        YamlReader.NON_PRINTABLE,
        "",
        data.translate(YAML_TRANSLATER).rstrip(b"\x00").decode(YAML_CODE_PAGE),
    )
    yaml_src = re.sub(r"(\w+: )(,.*)", r'\1"\2"', yaml_src)
    parsed = _yaml.load(yaml_src, Loader=CustomYamlSafeLoader)
    return parsed or {}


def _player_car_path(info: dict) -> str:
    """Resolve the driver's car identifier (CarPath) from session info."""
    di = info.get("DriverInfo", {}) or {}
    drivers = di.get("Drivers") or []
    idx = di.get("DriverCarIdx")
    if idx is not None:
        for d in drivers:
            if d.get("CarIdx") == idx:
                return str(d.get("CarPath", "") or "")
    # Fall back: first non-pace-car driver.
    for d in drivers:
        path = str(d.get("CarPath", "") or "")
        if path and "safety" not in path.lower() and "pace" not in path.lower():
            return path
    return ""


def parse_ibt(path: Path | str) -> ParseResult:
    """Parse one .ibt file via pyirsdk and return a ParseResult."""
    try:
        import irsdk
    except ImportError:  # pragma: no cover
        import pyirsdk as irsdk  # type: ignore[import-not-found, no-redef]

    ibt = irsdk.IBT()
    ibt.open(str(path))
    try:
        info = _read_yaml(ibt)
        weekend = info.get("WeekendInfo", {}) or {}
        yaml_car = _player_car_path(info)
        yaml_track = str(weekend.get("TrackName", "") or "")
        recorded_at = weekend.get("WeekendOptions", {}).get("Date") or None

        setup = info.get("CarSetup", {}) or {}

        channels: dict[str, np.ndarray] = {}
        for header in ibt._var_headers:
            name = header.name
            if _excluded(name):
                continue
            if header.count and header.count > 1:
                # Multi-element array channel — skip; we only want scalars.
                continue
            arr = np.asarray(ibt.get_all(name), dtype=np.float32)
            if arr.ndim != 1:
                continue
            channels[name] = arr

        for required in ("LapDistPct", "Lap"):
            if required not in channels:
                raise ValueError(f"required channel {required!r} missing from IBT")

        sample_count = channels["LapDistPct"].shape[0]
        # Sample rate is 60 Hz nominal; duration is samples / 60.
        duration_s = float(sample_count) / 60.0

        lap_spans = detect_lap_boundaries(channels["LapDistPct"], channels["Lap"])

        weather_summary = _summarize_weather(channels)

        return ParseResult(
            yaml_car=str(yaml_car),
            yaml_track=str(yaml_track),
            recorded_at=str(recorded_at) if recorded_at else None,
            duration_s=duration_s,
            channels=channels,
            setup=setup,
            weather_summary=weather_summary,
            lap_spans=lap_spans,
        )
    finally:
        ibt.close()


def _summarize_weather(channels: dict[str, np.ndarray]) -> dict:
    """Reduce per-sample weather channels to a small JSON-friendly summary."""
    summary: dict = {}
    if "AirTemp" in channels:
        summary["AirTemp_c_mean"] = float(channels["AirTemp"].mean())
    if "TrackTempCrew" in channels:
        summary["TrackTempCrew_c_mean"] = float(channels["TrackTempCrew"].mean())
    if "AirDensity" in channels:
        summary["AirDensity_kgm3_mean"] = float(channels["AirDensity"].mean())
    if "WindVel" in channels:
        summary["WindVel_ms_mean"] = float(channels["WindVel"].mean())
    if "TrackWetness" in channels:
        summary["TrackWetness_max"] = float(channels["TrackWetness"].max())
    return summary


def channel_names(result: ParseResult) -> Iterable[str]:
    return result.channels.keys()
