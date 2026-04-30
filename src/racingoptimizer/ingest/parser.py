"""Wrap `pyirsdk` (a.k.a. `irsdk`) into a typed ParseResult.

Reads each scalar telemetry channel into a float32 numpy array and parses the
embedded YAML session info to extract car / track / setup / weather context.

VISION §1 ("use everything, lose nothing"):
- Every channel we drop is recorded in `ParseResult.dropped_channels` with a
  human-readable reason. Silent drops were forbidden because they made it
  impossible to audit what telemetry the optimizer was actually seeing.
- Sample rate is auto-detected (see `_detect_sample_rate`); the old hardcoded
  60 Hz divisor would silently corrupt `t_s`/`duration_s`/`lap_time_s` for any
  IBT recorded at iRacing's higher tick rates (e.g. 360 Hz via app.ini).
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

# Fallback when neither the IBT header nor the YAML SessionInfo expose a tick
# rate. iRacing's default IBT recording rate is 60 Hz; app.ini can raise it
# to 360 Hz, but a 60 Hz default matches the historical assumption.
DEFAULT_SAMPLE_RATE_HZ: float = 60.0


class ParseResult(NamedTuple):
    yaml_car: str
    yaml_track: str
    recorded_at: str | None    # ISO timestamp from YAML header if present
    duration_s: float
    sample_rate_hz: float                # detected from header / YAML / fallback
    channels: dict[str, np.ndarray]      # name -> float32 1-D array, all same length
    setup: dict                          # nested garage setup as parsed from YAML
    weather_summary: dict                # JSON-friendly summary of weather channels
    lap_spans: list[LapSpan]
    dropped_channels: dict[str, str]     # name -> reason (VISION §1 audit trail)


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


def _detect_sample_rate(ibt, info: dict) -> float:
    """Detect the IBT recording rate in Hz.

    Fallback chain (first hit wins):
      1. `ibt._header.tick_rate` — pyirsdk's parsed value from the IBT header.
         This is the canonical recording rate iRacing wrote to disk.
      2. `disk_header.session_record_count / (session_end_time - session_start_time)`
         — back-computed from the disk subheader. Matches `tick_rate` in
         practice (within float-rounding) and is the safety net if the header
         field is ever zero or missing.
      3. `SessionInfo.SessionTickRate` from the YAML blob — covers any IBT
         version that exposes the rate only in YAML.
      4. `DEFAULT_SAMPLE_RATE_HZ` (60.0) — historical default.

    The result is always > 0; non-positive intermediates are skipped so a
    corrupt header field cannot poison the time axis.
    """
    header = getattr(ibt, "_header", None)
    if header is not None:
        rate = getattr(header, "tick_rate", None)
        if rate is not None and rate > 0:
            return float(rate)

    disk_header = getattr(ibt, "_disk_header", None)
    if disk_header is not None:
        count = getattr(disk_header, "session_record_count", None)
        start = getattr(disk_header, "session_start_time", None)
        end = getattr(disk_header, "session_end_time", None)
        if count and start is not None and end is not None and end > start:
            rate = float(count) / float(end - start)
            if rate > 0:
                return rate

    # YAML fallback: SessionInfo.SessionTickRate is the rate iRacing's session
    # config asked for, even if the header value is missing.
    si = info.get("SessionInfo", {}) or {}
    yaml_rate = si.get("SessionTickRate")
    if yaml_rate is not None:
        try:
            rate = float(yaml_rate)
        except (TypeError, ValueError):
            rate = 0.0
        if rate > 0:
            return rate

    return DEFAULT_SAMPLE_RATE_HZ


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
        sample_rate_hz = _detect_sample_rate(ibt, info)

        channels: dict[str, np.ndarray] = {}
        dropped: dict[str, str] = {}
        for header in ibt._var_headers:
            name = header.name
            if _excluded(name):
                dropped[name] = "excluded by EXCLUDED_CHANNEL_PATTERNS"
                continue
            if header.count and header.count > 1:
                # Multi-element array channel. We only consume scalars today;
                # record the drop so it shows up in the catalog audit trail.
                dropped[name] = f"multi-element array (count={header.count})"
                continue
            arr = np.asarray(ibt.get_all(name), dtype=np.float32)
            if arr.ndim != 1:
                dropped[name] = f"unexpected ndim={arr.ndim} after get_all"
                continue
            channels[name] = arr

        for required in ("LapDistPct", "Lap"):
            if required not in channels:
                raise ValueError(f"required channel {required!r} missing from IBT")

        sample_count = channels["LapDistPct"].shape[0]
        duration_s = float(sample_count) / sample_rate_hz

        lap_spans = detect_lap_boundaries(channels["LapDistPct"], channels["Lap"])

        weather_summary = _summarize_weather(channels)

        return ParseResult(
            yaml_car=str(yaml_car),
            yaml_track=str(yaml_track),
            recorded_at=str(recorded_at) if recorded_at else None,
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            setup=setup,
            weather_summary=weather_summary,
            lap_spans=lap_spans,
            dropped_channels=dropped,
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
