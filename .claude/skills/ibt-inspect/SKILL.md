---
name: ibt-inspect
description: Summarize an iRacing .ibt telemetry file - channels present, sample rate, lap count, embedded garage setup, and weather context. Use when first encountering a new IBT file or auditing what a session contains. Auto-detects car and track from filename.
---

# ibt-inspect

When given an `.ibt` path:

1. Parse the YAML header (session info + garage setup) and the channel index.
2. Report:
   - **Filename-derived**: car, track, session datetime
   - **Header**: track surface state, weather (`AirTemp`, `TrackTempCrew`, `Skies`, wind), session type
   - **Garage setup**: full setup as recorded
   - **Channels**: count, names grouped by category (suspension / aero / tyres / driver inputs / GPS / weather)
   - **Time series**: total samples, sample rate (expect 60 Hz), duration, lap count, and best lap time if computable
3. If the file is large (>100 MB), do **not** read all samples — header + a sampled lap is enough for inspection.

The IBT parser library is whatever the project chose (check `pyproject.toml` once it exists). If unset, fail fast with a message asking which parser to use.

Per `CLAUDE.md`, never collapse to session averages — this skill is for discovery, not analysis. Keep the corner-phase invariants downstream of this.
