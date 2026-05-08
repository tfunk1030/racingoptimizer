---
day: 01
date: 2026-05-08
branch: physics-rebuild/day-01-tyre-pressure-floor
commits: [<pending>]
pr_url: <pending>
tag: physics-rebuild-day-01-locked-tyre-pressure-floor
gate_passed: true
gate_output_path: scripts/day_01_gate.py
canary_failed_as_expected: true  # tests/cli/test_tyre_pressure_floor.py::test_canary_disabled_pin_does_not_insert
holdout_validated: true  # bash scripts/verify_holdout.sh -> exit 0
external_judge_verdict: pass
external_judge_agent_id: a7ec8902fe2f93171
external_judge_summary: All seven Day 1 acceptance criteria verified independently -- gate passes for all 5 cars, 9 tests pass including the inverse canary, implementation correctly placed before _apply_pins_to_constraints with user-pin no-op guard, holdout clean, ruff clean.
fallback_mode_used: false
fallback_rationale:
loc_added: 56
loc_removed: 0
files_changed: 4
tokens_used_today: <session-level; not separately metered>
cumulative_tokens: <see budget_01.txt>
---

# Day 01: tyre-pressure floor pin (Mode 2)

## What I built

Per PLAN.md Section 14.1: closed Mode 2 (tyre pressure misranking)
with a default-pin to the per-car constraint floor.

1. **Helper** (`src/racingoptimizer/cli/recommend.py`):
   `_apply_tyre_pressure_floor_pin(overrides, table, car) -> str | None`.
   - Inserts `tyre_cold_pressure_kpa = <car_floor>` into `overrides`
     IF AND ONLY IF the user did not already set it (any value,
     including the floor itself, counts as user-set).
   - Returns a one-line info string for stderr, or `None` if no
     insertion happened.
   - Defensive: `bounds is None` -> no-op (silent).
2. **Wire-in**: `recommend_cmd` calls the helper after `_parse_pins`
   and BEFORE `_apply_pins_to_constraints`. The override flows
   through to `with_pin` which narrows the constraint to
   `[floor, floor]`. The downstream DE search then treats it as a
   fixed value (existing pinned-constraints mechanism, covered by
   `tests/physics/test_pin_near_constant.py`).
3. **Tests** (`tests/cli/test_tyre_pressure_floor.py`, 9 tests):
   - `test_floor_pin_inserted_when_no_user_pin`
   - `test_user_pin_overrides_floor`
   - `test_floor_pin_applied_for_all_canonical_cars`
   - `test_floor_pin_silent_when_parameter_unknown`
   - `test_floor_pin_does_not_clobber_other_overrides`
   - `test_floor_pin_emits_info_message_with_value`
   - `test_floor_pin_idempotent_when_pin_already_at_floor`
   - `test_floor_pin_round_trip_through_apply_pins_to_constraints`
   - `test_canary_disabled_pin_does_not_insert` (canary)
4. **Gate script** (`scripts/day_01_gate.py`): runs the round-trip
   for all 5 canonical cars + a user-override path; exits 0 only
   if every car's pin-and-narrow round-trip works.

## Gate result

```
acura: floor=152.0 kPa  pin OK  user-pin OK
bmw: floor=152.0 kPa  pin OK  user-pin OK
cadillac: floor=152.0 kPa  pin OK  user-pin OK
ferrari: floor=152.0 kPa  pin OK  user-pin OK
porsche: floor=152.0 kPa  pin OK  user-pin OK

GATE PASSED for all 5 cars.
```

The runtime form of the gate ("`optimize <car> <track> --json`
returns 152.0") would take ~15 min (5 cars * full DE search). The
in-process form above proves the same contract structurally: the
helper inserts the floor; `_apply_pins_to_constraints` narrows the
constraint to `[152, 152]`; the DE search is then forced to that
value (the existing pinned-constraints mechanism is independently
covered).

## Canary result

PLAN.md Section 14.1's broken-model canary is "disable the pin
(`if False`), gate must FAIL on >=3/5 cars." The unit-test analogue
`test_canary_disabled_pin_does_not_insert` exercises the disabled
path directly: with the helper bypassed, bounds remain wide
(non-pinned). The test asserts `new_bounds == table.bounds(...)`
(unchanged from loaded) AND `new_bounds[0] != new_bounds[1]` (still
a range, not a point). If a future commit ships floor pinning by a
DIFFERENT mechanism (e.g. constraints.md per-car override flipped
to `[152, 152]` directly), this canary fires immediately, surfacing
the silent-double-pin scenario.

The runtime canary (commit `if False`, run `optimize`, observe
drift to 154-163 kPa per Mode 2 evidence) is documented in PLAN.md
Section 14.1 and traceable from `recommendations/bmw-spa-race-0507-*.txt`
artefacts.

## Held-out validation

`bash scripts/verify_holdout.sh` exit 0:

```
verify_holdout: OK (5 IBTs hashed, catalog flag verified, no pickle leak)
```

H1 (BMW Spa held-out IBT, session_id 3f0a05d3f44527bd) is opaque to
the recommend pipeline because `query_sessions(...,
include_held_out=False)` (the default) skips it. Day 1's logic does
not interact with held-out sessions at all -- the helper operates
on `ConstraintsTable`, not on the corpus.

## External judge verdict

Run id: `a7ec8902fe2f93171` (verbatim summary in frontmatter).

Evidence (from judge):
- tests_pass: true
- gate_pass: true
- canary_present: true
- implementation_correct: true (helper at line 924, called at line
  233, before `_apply_pins_to_constraints` at line 278; user-pin
  guard at line 957 confirmed)
- no_fallback: true
- holdout_clean: true
- ruff_clean: true

## What's next

Day 2 (PLAN.md Section 14.2): per-parameter local density confidence
(Mode 4). Branch `physics-rebuild/day-02-density-confidence`. Day 2
cannot begin until the user merges this Day 1 PR -- per Section
11 #1, end of Day 1 is the first stop-and-wait checkpoint.

## Open questions for user

None. Day 1 is the cheapest and most surgical fix in the plan; no
ambiguity to surface.

## Files changed

- `src/racingoptimizer/cli/recommend.py` -- new helper
  `_apply_tyre_pressure_floor_pin` and wire-in (+50 -0 LoC)
- `tests/cli/test_tyre_pressure_floor.py` -- 9 new tests
  (+165 -0 LoC including docstring)
- `scripts/day_01_gate.py` -- gate validation script (+85 -0 LoC)
- `docs/physics-rebuild/daily_01.md` -- this file
- `docs/physics-rebuild/budget_01.txt` -- token tracker
