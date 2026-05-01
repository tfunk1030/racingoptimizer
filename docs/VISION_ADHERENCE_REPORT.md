# VISION Adherence Report

Date: 2026-05-01

This report supersedes the older green-only compliance claims in
`docs/VISION_COMPLIANCE.md`. It records what is implemented, what was rebuilt,
and which remaining gaps are blocked by missing external setup data rather than
code wiring.

## What Changed

- Added garage setup inventory classification so every observed setup leaf is
  categorized as optimized, modeled readout, blocked user input, unsupported
  readout, or non-setup metadata.
- Fixed per-car ontology paths for bounded parameters on Acura, Cadillac,
  Ferrari, and Porsche so optimizer-search parameters resolve against real
  setup YAML before entering the model.
- Blocked car-specific parameters whose units or semantics do not match the
  current generic bounds, notably Ferrari spring-rate index controls and
  non-applicable Acura rear-coil / third-perch entries.
- Propagated lap sample rate from the `t_s` time axis into corner/phase timing
  so high-rate recordings do not inherit 60 Hz millisecond thresholds.
- Filtered dirty corner-phase rows from physics fitting using
  `data_quality_clean_frac >= 0.8`.
- Made prediction confidence affect the optimization score, not only the
  rendered output and trust radius.
- Updated stale constraints/status prose so bounded ARB, brake-bias,
  differential-preload, and camber families are not reported as missing.

## Scratch Corpus Build

Scratch corpus: `.scratch/vision-full-corpus-20260501`

Ingest:

- `uv run optimize learn "ibtfiles" --corpus-root ".scratch\\vision-full-corpus-20260501"`
- Result: 125 sessions reported ingested; scratch catalog contains 124 session
  rows for status/model queries.

Forced model builds:

- PASS `acura daytona_2011_road`
- PASS `acura hockenheim_gp`
- FAIL `bmw nurburgring_combined` — insufficient training data: one session,
  one valid lap, no trained fitters.
- PASS `bmw roadatlanta_full`
- PASS `bmw sebring_international`
- PASS `bmw spielberg_gp`
- PASS `cadillac lagunaseca`
- PASS `ferrari algarve_gp`
- PASS `ferrari hockenheim_gp`
- PASS `porsche algarve_gp`
- PASS `porsche lagunaseca`
- PASS `porsche spielberg_gp`

Five-car text + JSON CLI smoke passed against the scratch corpus.

## Parameter Wiring Summary

| Car | Optimized user inputs | Modeled readouts | Blocked user inputs |
| --- | ---: | ---: | ---: |
| acura | 16 | 4 | 22 |
| bmw | 18 | 4 | 20 |
| cadillac | 18 | 4 | 20 |
| ferrari | 13 | 4 | 25 |
| porsche | 18 | 4 | 20 |

Optimized user-input families now include the bounded, resolved subset of:

- rear wing
- tyre cold pressure
- ARB blade index
- brake bias
- differential preload
- camber
- heave/rear spring-rate controls where units match bounds
- perch and pushrod controls where paths and units match bounds

Modeled-only readouts include static ride-height and heave/slider deflection
setup leaves. They are not recommendation outputs because the driver cannot
enter them directly in the iRacing garage.

## Remaining External-Data Blockers

These are not considered complete VISION adherence until real per-car UI bounds
and units are captured:

- Damper click legal ranges and exact per-car damper grouping semantics.
- Corner-weight target legal ranges, if they are truly user-enterable for the
  relevant car.
- Toe units and legal ranges in millimeters, replacing the old degree-based
  TODO.
- Brake duct opening legal ranges.
- Differential coast/power controls and car-specific string/ratio mappings.
- Throttle/brake mapping and traction-control style controls by car.
- Ferrari spring/heave controls, which appear as UI indices in setup YAML rather
  than N/mm values compatible with the generic bounds.
- BMW Nurburgring model training, which needs more valid laps before a fit can
  be trained.

## Verification

- `uv run pytest tests/physics/test_ontology_per_car.py tests/physics/test_garage_inventory.py -q`
  - 14 passed
- `uv run pytest tests/corner/test_segment_lap.py tests/physics/test_fitter.py tests/physics/test_score.py tests/physics/test_ontology.py tests/physics/test_ontology_per_car.py tests/physics/test_garage_inventory.py tests/cli/test_post_clamp_discrete.py -q`
  - 85 passed
- `uv run ruff check src tests`
  - All checks passed
- `uv run pytest -q -m "not slow"`
  - 594 passed, 38 deselected

## Current Adherence Status

The implementation is materially closer to VISION.md, but full adherence is
not yet claimable because the blocked setup families above lack verified
external UI bounds/units. Code gaps closed in this pass include sample-rate
propagation, dirty-section filtering during fitting, confidence-aware scoring,
and bounded per-car ontology resolution.
