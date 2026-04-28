---
name: add-constraint
description: Append a new setup-parameter legality entry to constraints.md, preserving the established defaults + per-car overrides shape. Use when the user provides ranges for ARBs, dampers, corner weights, brake bias, diff preload, or any other garage parameter not yet in the file.
disable-model-invocation: true
---

# add-constraint

`constraints.md` enforces the bounds the optimizer must clamp to. It is partial — extending it correctly preserves machine-loadability for the ingestion code that will follow.

**Schema rules:**
- All ranges live under `## Defaults (apply to all cars unless overridden)` unless they vary by car. Per-car deviations go under `## Per-car overrides`, one section per car (`### acura`, `### bmw`, `### cadillac`, `### ferrari`, `### porsche` — match `aero-maps/` filename casing).
- Tables, not bullet lists. Each table heading row carries units in the column titles.
- Always include the unit (`mm`, `kPa`, `°`, `N/mm`, `clicks`, etc.).
- Never reformat existing entries — append only.

**Workflow:**
1. Ask the user for: parameter name, units, default min/max, and any per-car deviations.
2. Read the current `constraints.md`.
3. Insert the new section under the right heading, alphabetized within its group.
4. Show the diff before writing.
