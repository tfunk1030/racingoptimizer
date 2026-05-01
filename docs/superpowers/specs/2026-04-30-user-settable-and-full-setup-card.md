# Spec: USER INPUTS vs CALCULATED READOUTS + Full setup card

**Date:** 2026-04-30
**Slice:** cross-cutting (touches `physics/ontology`, `physics/recommend`,
`physics/fitter`, `explain/full_setup_card`, `cli/recommend`)
**Status:** merged in `bf2e48b`; this spec was written after the fact as part
of the audit follow-up (clause 8 of `docs/VISION_COMPLIANCE.md`).

## Problem

Several iRacing GTP garage values look like setup inputs but are actually
calculated readouts that the driver cannot type into the UI:

* **Static ride heights** (`Chassis.<corner>.RideHeight`) — output of perch
  offsets, pushrod lengths, torsion-bar turns, spring rates, corner
  weights.
* **Heave spring deflection** / **slider deflection**
  (`Chassis.Front.HeaveSpringDefl`, `HeaveSliderDefl`) — current/max
  deflection measured at the part.
* **Per-corner weights**, **cross weight**, the entire **AeroCalculator**
  block, **last hot pressures / temps / tread remaining**.

Pre-`bf2e48b` the optimizer's recommendation listed values for these
readouts. That is the same correctness class of bug as a unit mismatch —
the recommendation cannot be entered into the iRacing garage. VISION §7
("justify every click") implicitly requires every recommended click to be
enterable.

The fix introduces a three-flag matrix on `physics.ontology.ParameterSpec`:

| Flag | Meaning |
|---|---|
| `fittable: bool` | The model trains a per-(corner, phase, channel) fitter that includes this parameter in its feature vector. Set False to exclude the parameter from the joint surrogate entirely. |
| `user_settable: bool` | The driver can type this value into the iRacing garage UI. False marks calculated readouts. The optimizer must not put `user_settable=False` parameters into its search space and the briefing must not list them as "set this". |
| `constraints.bounds(car, name) is not None` | `constraints.md` provides legal bounds. CE-gated parameters whose bounds are still `<TODO: from iRacing UI>` are excluded from the search until captured. |

`fittable_parameters(car, table)` (`ontology.py:334-360`) returns the
optimizer's free-variable list — every parameter that satisfies all three
gates.

## Files

### `physics/ontology.py`

* `ParameterSpec` (lines 40-58) gains the `user_settable: bool = True`
  field. Default True for backward-compat with every legacy entry that
  pre-dates this flag — every legacy entry was a real input.
* `_common_bounded()` (lines 123-201) marks four entries
  `user_settable=False`:
  * `static_ride_height_front_mm`, `static_ride_height_rear_mm`
  * `heave_spring_mm` (`HeaveSpringDefl`)
  * `heave_slider_mm` (`HeaveSliderDefl`)
* `_common_bounded()` adds eight new USER-input entries with the JSON
  paths that drive those readouts:
  * `heave_spring_rate_n_per_mm` (`Chassis.Front.HeaveSpring`)
  * `third_spring_rate_n_per_mm` (`Chassis.Rear.ThirdSpring`)
  * `rear_coil_spring_rate_n_per_mm` (`Chassis.LeftRear.SpringRate`)
  * `heave_perch_offset_front_mm` (`Chassis.Front.HeavePerchOffset`)
  * `spring_perch_offset_rear_mm` (`Chassis.LeftRear.SpringPerchOffset`)
  * `third_perch_offset_rear_mm` (`Chassis.Rear.ThirdPerchOffset`)
  * `pushrod_length_offset_front_mm` (`Chassis.Front.PushrodLengthOffset`)
  * `pushrod_length_offset_rear_mm` (`Chassis.Rear.PushrodLengthOffset`)

### `physics/recommend.py`

* `_pin_or_trust_bounds` (lines 271-297) — pin the search window to
  `baseline ± 1e-6 * range` when the per-session observed std is below
  2% of the constraint range. The DE search needs `lo < hi`, so we use a
  tiny but non-zero window. The `pinned_to_observed_median` field of
  `SetupRecommendation` records which parameters were pinned — the CLI
  surfaces them in a top-level warning.
* Rationale: when every observed session ran the same value (e.g. every
  Cadillac fixture had tyre cold P at 152 kPa), the joint surrogate has
  no signal about the response surface. The DE search drifts to whichever
  bound the noise gradient points at, producing absurd constraint-edge
  recommendations.

### `physics/recommendation.py`

* `SetupRecommendation` (line 11) gains `pinned_to_observed_median:
  tuple[str, ...] = ()`.

### `explain/full_setup_card.py` (new)

* `render_full_setup_card(rec, *, car, most_recent_setup)` walks the
  driver's most recent ingested setup blob for this `(car, track)` pair
  and emits one line per garage-panel leaf with one of four tags:
  * `[OPT]` — optimizer-recommended value.
  * `[OPT pin]` — optimizer pinned to observed median.
  * `[past]` — value carried over from the user's most recent session
    (no constraints bounds yet).
  * `[readout]` — calculated by iRacing; informational only.
* `_CALCULATED_LEAF_NAMES` (`full_setup_card.py:37-55`) hard-codes 22
  YAML leaf names that iRacing always treats as readouts on the GTPs.
* The renderer's `_ontology_path_index` (lines 100-125) skips any
  parameter whose ontology entry has `user_settable=False`, even if
  `rec.parameters` contains it. Defence in depth against future leaks.

### `cli/recommend.py`

* `recommend_cmd` (lines 188-191) appends the full-setup-card output
  after the briefing in text mode. JSON mode does not include it (would
  bloat the contract; consumers can call the renderer separately).
* `_most_recent_setup_for(sessions_df)` (lines 432-447) pulls the parsed
  YAML from the latest session, ordered by `recorded_at`.

### `constraints.md`

Three new prose blocks explain the user-input vs readout distinction,
the "estimated bounds" caveat for the new spring/perch/pushrod families,
and why "static ride height" + "suspension deflections" sections stay in
the file (observation envelope record only — the recommender never emits
values for them).

Per-car overrides (`### acura` / `### bmw` / `### cadillac` / `### porsche`)
shadow the spring-rate defaults with values closer to the observed session
range for each car. Ferrari overrides are not yet captured.

## Validation

* `tests/physics/test_ontology.py::test_fittable_parameters_only_returns_bounded_user_settable`
  pins the BMW fittable list (must include the eight new user-input
  parameters; must exclude the four readouts).
* `tests/physics/test_ontology.py::test_ce_gated_families_present_but_unfittable[<car>]`
  parametrised over all 5 cars — asserts dampers + corner-weights stay
  `fittable=False` until bounds land.
* `tests/physics/test_pin_near_constant.py` — five unit tests for
  `_pin_or_trust_bounds` plus one integration test against
  `bmw_model_session`.
* `tests/explain/test_full_setup_card.py` (added in audit follow-up) —
  ten tests covering the four tag paths, the empty-input paths, the JSON-
  string acceptance path, the per-car non-crash check, and the
  `user_settable=False` defence-in-depth guard.
* `tests/physics/test_ontology_per_car.py` (added in audit follow-up) —
  asserts every USER-input parameter resolves against the real-IBT setup
  YAML for every canonical car fixture.

## Non-goals / known follow-ups

* **Per-car spring/perch/pushrod bounds are estimates.** `constraints.md`
  flags this with a `> NOTE — estimated bounds` block. Per-car values
  need to be captured from the iRacing garage UI for each of the 5 GTP
  cars. Until then, the recommender clamps to the wide defaults.
* **Camber and toe.** Bounds exist in `constraints.md` (camber ranges
  per corner; toe is `<TODO: units mismatch>`). The ontology has no
  `camber_*` or `toe_*` entries yet — adding them means new per-car
  ParameterSpec entries plus per-corner JSON paths for each of the 5
  cars. Tracked as a follow-up to this spec.
* **Discrete clicks on ARBs.** `anti_roll_bar_front` / `_rear` are
  integer-valued (1..5) but the DE search treats them as continuous.
  Recommendations like "anti_roll_bar_front: 3.7" reach the user and
  must be rounded by hand. A `dtype=int` enforcement at the briefing
  layer is a follow-up.
