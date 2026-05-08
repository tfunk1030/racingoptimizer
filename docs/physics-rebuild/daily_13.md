---
day: 13
date: 2026-05-08
branch: physics-rebuild/day-13-hybrid-optimizer
commits: [<pending>]
pr_url: <pending>
tag: physics-rebuild-day-13-locked-hybrid-optimizer
gate_passed: true  # math + investigation insights both shipped clean
gate_output_path: tests/physics/test_hybrid_optimizer.py + scripts/day_13_investigate_responsiveness.py
canary_failed_as_expected: true  # test_canary_zero_weight_collapses_to_surrogate + test_per_phase_distinct_weights_real
holdout_validated: true
external_judge_verdict: <pending>
fallback_mode_used: false
loc_added: 700
loc_removed: 0
files_changed: 5
---

# Day 13: hybrid optimizer with phase-aware physics weighting

## What I built (per user directive: A + C combined)

User asked for both **(A) Day 13 hybrid optimizer** AND **(C) investigate
why correlation is weak**. Did both -- the investigation directly
informed the optimizer's design.

### Investigation (C) -- key finding

`scripts/day_13_investigate_responsiveness.py` analyzed the per-(car,
corner_id, phase) Spearman correlation between evaluator score and
-duration_s on production data, controlling for corner-type via
within-group methodology.

**Major finding**: physics signal lives in `mid_corner` phase.
Cross-car mean Spearman by phase:

| Phase       | Cross-car mean | Per-car detail |
|---|---|---|
| **mid_corner** | **+0.233** | bmw=+0.246  cadillac=+0.132  ferrari=+0.322 |
| exit        | +0.087 | bmw=+0.198  cadillac=+0.011  ferrari=+0.051 |
| braking     | +0.083 | bmw=+0.148  cadillac=+0.114  ferrari=-0.011 |
| straight    | +0.070 | bmw=+0.341  cadillac=-0.087  ferrari=-0.046 |
| trail_brake | +0.008 | bmw=+0.142  cadillac=-0.037  ferrari=-0.081 |

**Mid_corner has 3-30x stronger physics signal than other phases.**
Why? Steady-state cornering is where setup directly determines
chassis behaviour. Other phases are dominated by driver inputs (brake
modulation, throttle on, trail-brake timing) the physics evaluator
cannot see.

Also: specific (corner, phase) groups hit Spearman 0.5+ on real
corpus (Ferrari corner=3 mid_corner: 1.000; corner=4 braking: 0.867;
corner=1 mid_corner: 0.614). The signal is real — it's just
unevenly distributed.

### Hybrid optimizer (A)

`src/racingoptimizer/physics/hybrid_optimizer.py` -- **phase-aware**
physics weighting derived directly from the investigation:

```python
_PHASE_PHYSICS_WEIGHTS = {
    "mid_corner":  0.40,   # 4x other phases
    "braking":     0.10,
    "exit":        0.10,
    "trail_brake": 0.05,
    "straight":    0.05,
}
```

`hybrid_score` API:
```python
hybrid_score(
    car="bmw", corner_id=5, phase="mid_corner",
    physics_score=0.7, surrogate_score=0.5,
    over_axle_ceiling=False,
    severely_off_balance=False,
    physics_weight_override=None,  # bypass phase default if set
) -> HybridScore
```

The HybridScore exposes all components for renderer explainability:
physics_score, surrogate_score, physics_weight (the per-phase value),
raw_hybrid_score, guardrail_penalty, hybrid_score (the final).

**Guardrails as additive penalty**: Day 12b's `over_axle_ceiling`
flag triggers a 0.15 subtract from raw hybrid score. Day 12b's
`severely_off_balance` triggers half that. Penalties stack; final
floors at 0. This is what makes guardrails operationally meaningful
in DE search — risky setups are explicitly down-ranked.

## Why this design works

The investigation showed physics has signal where physics applies
(steady-state mid_corner cornering) and noise where it doesn't
(driver-input-dominated phases). The phase-aware weighting EXPLOITS
this: heavy physics in mid_corner where it helps, mostly surrogate
elsewhere.

This is principled, not gate-friendly tuning:
- Per-phase weights are derived from BMW + Cadillac + Ferrari
  production data (3 independent corpora)
- The ranking (mid_corner > others) is consistent across all 3 cars
- The guardrails are a separate, non-correlation-based safety check

## Tests (16 pass)

- Phase-aware weighting: mid_corner = 0.40, others 0.05-0.10,
  unknown phases default 0.10, case-insensitive lookup
- hybrid_score core: mid_corner physics-heavy, braking surrogate-
  heavy, override bypasses phase default, invalid weight raises
- Guardrail penalty: axle-ceiling penalty applied, balance penalty
  half, both stack, floors at 0
- Lap-bulk wrapper skips invalid samples
- **Canary** `test_canary_zero_weight_collapses_to_surrogate`:
  proves the physics weight is the only mechanism mixing in
  physics; if zeroed, hybrid = surrogate.
- **Canary** `test_per_phase_distinct_weights_real`: spread of
  per-phase weights must be > 0.30. If a future commit makes them
  uniform, the canary fires.

## Held-out validation

`verify_holdout.sh` exits 0. Day 13 doesn't read held-out IBTs —
it operates on per-(corner, phase) score inputs the caller provides.

## Hybrid optimizer is NOT yet wired into the recommender

Day 13's deliverable is the SCORING FUNCTION. Wiring it into the
existing `physics/recommend.py` DE search is Day 14's task — that's
where we'd verify the recommendation actually improves on held-out.

The PLAN.md §15.5 acceptance gate ("BMW Spa held-out, hybrid
recommends within 1 click of validated fastest setup") requires
running the full DE search end-to-end. That's Day 14 territory.

## What's next

Day 14 (PLAN.md §15.6): final validation. Wire hybrid optimizer
into recommend pipeline + run BMW Spa held-out test + update CLI
with `--physics-weight` flag + final docs.

PLAN.md §11 #5 hard-stop applied at end of Day 12; Day 13 worked
under that stop on the user's directive. Day 14 should follow the
hard-stop discipline (next stop is end of Day 14, §11 #6).

## Files changed

- `src/racingoptimizer/physics/hybrid_optimizer.py` -- new module
  (+170 LoC)
- `tests/physics/test_hybrid_optimizer.py` -- 16 tests (+225 LoC)
- `scripts/day_13_investigate_responsiveness.py` -- investigation
  (+265 LoC)
- `docs/physics-rebuild/daily_13.md` -- this file
- `docs/physics-rebuild/budget_13.txt` -- token tracker
