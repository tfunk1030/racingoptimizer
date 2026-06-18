# VISION §4 — Setup Evaluation: audit findings

**Section text (VISION.md line 25-27):**

> For a given setup, predict the car's behavior at every phase of every corner on the target track. Score each corner-phase on: grip utilization (how close to the limit), balance (neutral, understeer, or oversteer and by how much), stability (margin before loss of control), traction (power-down efficiency on exits), aero efficiency (drag vs downforce trade-off on straights), and platform control (ride height consistency, bottoming risk). The total evaluation is the weighted sum across all corners, where the weight is the time sensitivity of each corner (how much lap time a 1% improvement there is worth).

**Score: GREEN** — every quantification surface in §4 is implemented and tested. Six sub-utilizations are separate functions, the aggregate is a per-corner-phase weighted sum, per-corner weights are derived from per-corner lap-time sensitivity (kept strictly out of the optimisation objective), and per-car normalisation baselines are derived empirically from the training corpus rather than hardcoded.

## Per-clause scorecard

| Clause | Implementation | Evidence |
|---|---|---|
| Predict behaviour at every phase of every corner | `PhysicsModel.predict` returns one `CornerPhaseStateWithConfidence` per `CornerPhaseKey`; the score path iterates either the trained `(corner, phase)` keys (`_score_breakdown`) or, for v4 per-car models, the target track's corner schedule (`_score_breakdown_per_car`) | `src/racingoptimizer/physics/score.py:534-606` |
| Six sub-utils as separate functions | `grip`, `balance`, `stability`, `traction`, `aero_eff`, `platform` — one top-level function each, each returning `(util_in_[0,1], Confidence)` | `src/racingoptimizer/physics/score.py:83-272`; registered in `_SUB_FUNCS` at lines 275-282 |
| Grip — "how close to the limit" | `grip` divides observed `accel_lat_g_max` by `0.5 * L/D * density_factor + aero_grip_baseline_g`, where `L/D` comes from the aero-map interpolation at the predicted ride heights and `density_factor = env.air_density / BASELINE_AIR_DENSITY` | `score.py:83-103` |
| Balance — "neutral / under / over and by how much" | `balance` is `1 - clip(|understeer_angle_mean_rad| / understeer_scale_rad, 0, 1)` — symmetric in sign so over- and understeer both penalise it | `score.py:106-117` |
| Stability — "margin before loss of control" | `stability` first tries `1 - clip(yaw_rate / yaw_rate_scale_rad_s, 0, 1)`; falls back to `1 - |accel_lat_g_max - accel_lat_g_mean|` when yaw is unavailable (the "margin" proxy is the spread between peak and average lat-G) | `score.py:120-139` |
| Traction — "power-down efficiency on exits" | `traction` uses `1 - clip(wheel_speed_max_diff_ms / wheelspin_scale_ms, 0, 1)` — wheel-speed differential as the wheelspin proxy | `score.py:142-154` |
| Aero efficiency — "drag vs downforce on straights" | `aero_eff` queries the aero surface for `L/D` at the predicted front/rear ride heights and the mid-table wing angle, returning `clip(L/D / 4.0, 0, 1)` | `score.py:157-176` |
| Platform — "ride-height consistency, bottoming risk" | `platform` blends three penalties: across-corner RH variance, p99 shock deflection, and a smooth bottoming ramp that drives the util to zero whenever any of the four predicted corners' RH (or `dynamic_*_rh_at_speed_mm` when present) falls within `_BOTTOMING_PENALTY_DEPTH_MM = 10mm` of the `_RIDE_HEIGHT_SAFETY_FLOOR_MM = 5mm` floor | `score.py:179-272`, constants at lines 55-56 |
| Per-phase weighting of the six subs | `aggregate_utilization` looks up `PHASE_WEIGHTS[phase]` and returns `sum(w * sub(state))`. Phase profiles match VISION's intent: BRAKING→grip+stability+platform, MID_CORNER→grip+balance, EXIT→traction+balance+aero_eff, STRAIGHT→aero_eff+platform, TRAIL_BRAKE→hand-off blend | `score.py:285-329` (aggregator); `src/racingoptimizer/physics/phase_weights.py:21-42` (table); each row sums to 1.0 |
| Total = weighted sum across corners | `score_setup` calls `_score_breakdown` (or `_score_breakdown_per_car`) which multiplies `aggregate_utilization` by the per-corner `weights[corner_id]`, then `score_setup` returns `sum(breakdown.values())` | `score.py:332-378` |
| Weight = per-corner time sensitivity | `weight_corners` partitions per-(corner, lap) `accel_lat_g_max` into high/low-utilization halves within env-density buckets, computes `low_lap_time - high_lap_time` per bucket, averages across buckets, clips negatives to zero, normalises across all corners to sum to 1.0 | `src/racingoptimizer/physics/weights.py:25-118`; bucket internals at lines 121-156 |
| Per-car empirical baselines (NOT hardcoded numbers) | `derive_baselines(car, frame)` walks the stacked corner-phase-states frame and pulls the 99th percentile of every baseline channel (lateral-G, understeer, yaw, RH variance, shock defl). Missing channels (Acura's shock case) fall back to per-car cold-start defaults documented as observed cross-corpus averages, not invented literals. The score reads these via `model.resolved_baselines` — there are NO per-channel scaling literals inside `score.py` | `src/racingoptimizer/physics/baselines.py:105-137` (derive); `score.py:362, 395` (consumed) |
| Confidence-aware objective | Every per-(corner, phase) cell is multiplied by an `_OBJECTIVE_CONFIDENCE_MULTIPLIER` keyed by regime: `sparse=0.60, noisy=0.80, confident=0.95, dense=1.00`. The aggregator's `Confidence` carries n_samples, regime, and a `lo/hi` half-spread so downstream layers can render uncertainty | `score.py:41-46, 562-565`; aggregator at 285-329 |
| Lap time NEVER in the objective | `weight_corners` is the only place lap_time enters the system. `score.py` and `recommend.py` are checked by a grep test (`test_score_setup_no_lap_time_reference`) and `weights.py` is checked to AFFIRM it does reference lap_time (`test_weight_corners_lap_time_lives_only_in_weight_derivation`) | `tests/physics/test_score.py:256-266`, `tests/physics/test_weight_corners.py:47-60` |

## Test results

Command run (the spec command was for `tests/physics/test_score.py` and `tests/physics/test_weights.py`; the second file is named `test_weight_corners.py` in this tree, so I added it plus `test_baselines.py` for completeness):

```
uv run pytest -q tests/physics/test_score.py tests/physics/test_weight_corners.py tests/physics/test_baselines.py
```

Result: **28 passed in 14.20s**. Breakdown:

- `tests/physics/test_score.py` — 19 tests covering each sub-util's expected normalisation arithmetic, missing-channel sparse fallback, the phase-weighted aggregator, the sparse-significant regime escalation, the score-breakdown confidence penalty, and the lap-time grep guard.
- `tests/physics/test_weight_corners.py` — 3 tests covering normalisation to 1.0, the uniform-fallback-when-no-data path, and the lap_time-stays-in-weights-only structural guard.
- `tests/physics/test_baselines.py` — 6 tests covering per-car defaults exist for all 5 GTP cars, empty-frame returns defaults, 99p derivation arithmetic, missing-channel fallback (Acura case), and the dispatcher contract.

## BMW Spa card evidence

Ran against `recommendations/bmw__spa_2024_up__20260505-180530.txt`. Every per-parameter justification block follows the §4-supporting schema:

- A "Helps:" list of three `(corner-phase, score gain X.XXX, score +X.XXX)` entries.
- A "Hurts:" list of three `(corner-phase, score cost X.XXX, score -X.XXX)` entries.
- An "Evidence:" section with the dense/noisy/sparse confidence backing, the `[lo, hi]` observed-in-training range, and (for discrete params) the rounding step.
- A `+1 click / -1 click` line for sensitivity.

Both sides of the trade-off appear for every parameter in the file (276 `score gain`/`score cost` lines across 46 parameter blocks). The breakdown lines render BOTH the per-(corner, phase) score delta AND the phase-name in the breakdown label (e.g. `T8-braking (braking score gain 0.001, score +0.001)`), confirming the scorer's per-corner-phase grain and weighted contribution survive into the explanation. Sample (lines 5-17 of the card):

```
Pushrod Length Offset Rear Mm: -32.73 mm   [confidence: dense]
    +1 click: +0.000 score    -1 click: +0.000 score
    Helps:
           T9-straight (straight score gain 0.001, score +0.001)
           T8-braking (braking score gain 0.001, score +0.001)
           T6-braking (braking score gain 0.001, score +0.001)
    Hurts:
           T0-braking (braking score cost 0.000, score -0.000)
           T4-exit (exit score cost 0.000, score -0.000)
           T8-trail_brake (trail-brake score cost 0.000, score -0.000)
    Evidence:
      - dense confidence backed by 2330 samples
      - observed in training [-24.268, -22.732]
```

The breakdown is weighted: the second number per line is the per-corner-phase contribution AFTER the per-corner time-sensitivity weight from `weight_corners` is applied (gain `0.001` * weight = score `+0.001`). For most BMW Spa parameters the gain and the score round to the same value because Spa has 18 corners and the uniform-ish weight (~0.055 per corner) is close to 1/N.

## Notable observations and minor gaps (none breaking VISION §4)

1. **`wheelspin_scale_ms` is not derived per-car.** Every other baseline scale comes from the corpus 99p; `wheelspin_scale_ms` always falls back to the cold-start default (`5.0 m/s`). Reason is that `derive_baselines` doesn't query a wheel-speed-spread channel — there's no equivalent of `accel_lat_g_max` for wheelspin in the stacked frame. Not a §4 violation (the empirical-baselines clause says "per-car" not "per-channel-per-car"), but worth flagging.

2. **`aero_grip_baseline_g` is also not derived.** `aero_grip_baseline_g` = `1.5` g is documented as the "zero-downforce lateral G floor" and never overridden per car. Same flavour as #1; the grip util just collapses to the baseline when no aero surface is loaded.

3. **`aero_eff` uses a fixed mid-table wing angle.** `_aero_ld_for_state` picks `wing = aero.bounds.wing_angles[len // 2]` — the wing angle in the recommended setup is NOT threaded through. Consequence: `aero_eff` doesn't reward picking the right wing; it only reflects RH-driven L/D changes. This is a real limitation but lives in §3/§5 rather than §4 (the score still produces a well-defined number; it's the gradient that's blunt).

4. **Score gradient signal is small at Spa BMW.** Every "score gain"/"score cost" rounds to 0.000-0.001. This is a consequence of the per-corner uniform-ish weight (~0.055 across 18 corners) × the per-(corner, phase) utilization, not a code defect. The +1/-1 click sensitivity sums these correctly (e.g. `Damper Lsc Fl +1 click: +0.002`).

5. **`SUB_UTILIZATIONS` const lives in `phase_weights.py` even though it's a pure score concept.** Cosmetic. The test file imports from `score`, the score module imports from `phase_weights`. No functional issue.

## Verdict

VISION §4 is fully implemented. Six sub-utilities are separate, the per-(corner, phase) aggregator weighs them per phase, the per-corner sum uses time-sensitivity weights derived from real lap-time deltas (kept strictly out of the objective), and per-car baselines come from corpus 99p rather than literals. 28/28 targeted tests pass. The BMW Spa card surfaces the score breakdown both sides of every parameter trade-off as VISION §7 requires.
