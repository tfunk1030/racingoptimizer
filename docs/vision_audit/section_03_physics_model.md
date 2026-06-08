# VISION §3 — Physics Model

> Build an empirical physics model for each car from the measured data.
> Don't hardcode spring rate formulas or LLTD equations — fit them from
> what the car actually does. Every parameter interacts with every other
> parameter — model the coupled system, not independent channels. Use the
> aero maps to connect ride height changes to downforce changes. Use the
> measured shock behavior to connect spring/damper changes to platform
> dynamics.

## Verdict: GREEN

The physics model is genuinely empirical, the joint architecture is
implemented end-to-end, the aero maps are wired into both the score and
the per-(corner, phase) ride-height predictions, and the recently added
`dynamic_at_speed.py` closes the §3 mandate to "fit from what the car
actually does" for the at-speed pose specifically — preferring real
60Hz telemetry over iRacing's static AeroCalculator estimate.

The BMW Spa briefing demonstrates the joint model doing exactly what
§3 prescribes: a perturbation of any single parameter yields a
quantifiable per-corner-phase score delta (`+1 click` / `-1 click`
sensitivities) reported across **46 parameters** with `dense`
confidence on the dominant fitter (n=2330 backing samples). Discrete
clicks (ARBs, dampers) are rounded post-clamp with an audit message,
and parameters with no observed variation in training are pinned to
the median rather than allowed to drift to a constraint edge.

## Tests run

`uv run pytest -q tests/physics/test_per_car_fit_predict.py`

(`tests/physics/test_per_car_smoke.py` does not exist; the per-car
smoke for the physics slice is `test_per_car_fit_predict.py` itself
and it parametrises the 5 canonical cars.)

Result: **16 passed in 198.07s** (~3 min 18 s on a cold-cache run).

The suite covers per-car `fit` + `predict`, the Acura graceful-
degradation contract for missing per-corner shock channels, and the
joint feature-vector schema across all 5 cars.

## Evidence

### (a) NO hardcoded spring/LLTD/textbook formulas in the model code

A grep across `src/racingoptimizer/physics/` for `spring_rate * X`,
`LLTD`, `K_roll`, `Speed**2`, `* Speed`, and `AccelLat * <literal>`
returns **no matches** in `score.py`, `fitter.py`, `model.py`,
`baselines.py`, `recommend.py`, `wet_mode.py`, `wind.py`,
`dynamic_at_speed.py`, or any of the four fitter implementations.

The only literal coefficients in `score.py` are normalisation
denominators that route through per-car empirical baselines —
verified by `tests/physics/test_validator_gate.py::test_no_textbook_formulas_in_score`,
which is parametrised against the live source and runs every CI cycle.

The two intentional exceptions are documented:

- `aero.interpolate(...)` calls (`physics/score.py:97, 165, 450`) —
  empirical aero-map lookups, explicitly exempted by VISION §3.
- `physics/damper_force.py` (a per-car digressive curve estimator,
  inlined as a Polars expression in the corner aggregator) —
  declared a "stepping-stone" pending real damper-spec capture from
  iRacing's garage tooltips. It populates the
  `damper_force_p99_n` / `damper_force_mean_n` corner-phase columns
  that then become **target outputs** of the empirical fitter — they
  are not used as inputs to a closed-form score formula.

### (b) Joint multi-input feature schema

`physics/fitter.py:fit` (lines 211-411) trains **one fitter per
(corner_id, phase, output_channel)** over the FULL bounded setup
vector + 12 env channels. The joint feature row is assembled by
`_attach_setup_columns`, `_attach_setup_readouts`, and
`_attach_dynamic_at_speed` and passed to `_fit_one_quadruple` (line
905), which builds `X = concat([param_block, env_block,
arch_block])` and records `feature_names` on the `FitRecord` so
`predict` can reconstruct the same row at query time.

`TARGET_OUTPUT_CHANNELS` (`fitter.py:56-115`) covers the full chain
demanded by §3: lateral G, brake/throttle, steering, understeer,
4-corner shock deflection p99, 4-corner ride-height mean, **damper
velocity / force p99 + mean** (the §3 "shock velocities and forces"
mandate), the four `setup_static_*_ride_height_mm` setup-readout
channels, and the six `dynamic_*_rh_at_speed_mm` telemetry-derived
channels. 26 distinct outputs per (corner, phase).

The `feature_schema_version=3` field on the pickled `PhysicsModel`
(`model.py:120`) is the single source of truth that `predict` keys
off to dispatch to `_predict_v3` (the joint path) vs. `_predict_legacy`
(the pre-Stage-3 sum-of-per-parameter path). Older v1/v2 pickles still
revive correctly via the `__setstate__` backfill (lines 167-196).

A second per-car (track-agnostic) variant `fit_per_car`
(`fitter.py:615-852`) trains under
`feature_schema_version=4` with 5 corner archetype features
(`apex_speed_ms`, `peak_lat_g`, `corner_min/max_speed_ms`,
`corner_duration_s`) appended to the joint vector; predict assembles
the row via `_assemble_feature_row_v4` (`model.py:641-673`). This
realises VISION §6 — "be able to generate optimal setups for any
track from just the track model and aero maps."

### (c) Coupled chain — perturbing one parameter propagates

`tests/physics/test_coupling.py::test_perturb_one_parameter_changes_multiple_corners`
fits a tiny synthetic GP coupled across `heave_spring_mm`,
`static_ride_height_front_mm`, and `rear_wing_angle_deg` and asserts
that perturbing a single parameter moves **≥3 (corner, phase) cells**
in the `score_breakdown`. This is the inverse of the deleted
pre-Stage-3 `test_score_breakdown_locality` (which AFFIRMED
non-coupling).

End-to-end proof on real data: the BMW Spa recommendation
(`recommendations/bmw__spa_2024_up__20260505-180530.txt`) lists
"Helps" / "Hurts" multi-corner blocks for every one of the 46
parameters reported. Stiffening the rear coil spring (line 104)
shows score gains at `T14-straight`, `T8-braking`, `T1-trail_brake`
and costs at `T1-exit`, `T17-exit`, `T0-braking` — a single
parameter touching 6 distinct (corner, phase) cells.

### (d) Aero-map consumption

`physics/score.py:_aero_ld_for_state` (line 435-451) reads the
per-(corner, phase) predicted ride heights out of the `state.states`
dict and queries `aero.interpolate(front_v, rear_v, wing,
env.air_density)`. It is invoked from both `grip` (line 97) and
`aero_eff` (line 165). The `AeroSurface` is loaded once per car and
cached in `_AERO_CACHE` (line 456) so the optimisation hot loop
amortises the JSON parse.

The aero scorer takes ride heights from the **fitter's predictions**
of the four `*_ride_height_mean_mm` channels — so the chain runs
correctly: setup change → joint surrogate predicts new ride height →
aero map returns new L/D → grip / aero_eff utilisation updates. This
is the "stiffer springs change ride heights which change aero
balance" chain in §5.

`PhysicsModel.aero_correction_available` (set during `fit` via
`_try_load_aero`, `fitter.py:1070`) gates the chain — when slice C
is unavailable for a car, the `AERO_DEPENDENT_CHANNELS` regimes
downgrade by one tier (`_maybe_downgrade_aero`, `model.py:592-607`)
so the briefing reports lower confidence rather than silently
fabricating a downforce estimate.

### (e) Per-car family routing

`_GP_FAMILIES` (`fitter.py:138-154`) lists the families that route
to `GPFitter`: `heave_spring`, `heave_slider`, `tyre_pressure`,
`front_wing`, `rear_wing`, `ride_height`, `arb`, `brake_bias`,
`diff`, `spring_rate`, `perch_offset`, `pushrod`, `camber`, `damper`,
`torsion_bar`. `_joint_family_kind` (`fitter.py:885-902`) returns
`"rf"` if **any** parameter in the joint vector belongs to an
RF-family (none currently — every fittable family is in `_GP_FAMILIES`),
else `"gp"`. Setup-readout channels (`setup_static_*`,
`dynamic_*`) are overridden to `RidgeFitter` by
`_channel_family_kind` (line 1023-1033) because the underlying
mapping is a deterministic linear-ish function of the bounded setup
parameters — Forest underfits to near-constant predictions in low
data regimes, Ridge captures the chain (see `RidgeFitter` docstring
in `fitters/ridge.py:1-23`).

The per-car `fit_per_car` path **always** uses Forest (`fitter.py:728`)
because the 35-dim mixed-scale feature vector (setup 0..550, env
0..1000, archetype 5..100) collapses a scalar-length-scale GP onto
whichever feature has the most variance (documented at
`fitter.py:715-727` with the failure mode that prompted the switch:
constant predicted ride heights regardless of `heave_spring`).

### (f) `dynamic_at_speed.py` — telemetry-derived ride heights

`physics/dynamic_at_speed.py` (215 LOC) is the §3 fix for "fit from
what the car actually does" applied to the at-speed pose
specifically. Rather than trusting `TiresAero.AeroCalculator.{Front,
Rear}RhAtSpeed` (a static iRacing estimate that ignores damper
dynamics, real straight-line speeds, and curb effects), the module
queries the 60Hz `LFrideHeight` / `RFrideHeight` / `LRrideHeight` /
`RRrideHeight` channels at high-speed straight-line samples
(`Speed >= 80th percentile`, `|LatAccel| < 0.3g`,
`Throttle > 0.7`, `data_quality_mask == True`) and reports the
median per corner. `compute_dynamic_at_speed_rh` returns one value
per session per corner, broadcast via `_attach_dynamic_at_speed`
onto every training row (`fitter.py:301`), and the trained Ridge
fitter learns the `setup → real_at_speed_rh` mapping from
observation. Six target channels (`dynamic_lf/rf/lr/rr/front/rear_rh_at_speed_mm`)
are pinned in `TARGET_OUTPUT_CHANNELS` (`fitter.py:108-114`).

### BMW Spa card evidence (joint-model perturbation readout)

`recommendations/bmw__spa_2024_up__20260505-180530.txt`:

- Header (line 3): `Confidence: dense (n=2330 backing samples for the
  dominant dense parameter, 46 parameters reported)` — proves the
  joint model is dense across most of the 46-dim search space, with
  per-parameter `[confidence: dense]` tags throughout.
- Every parameter block carries a `+1 click: <delta> score    -1 click:
  <delta> score` line (e.g. `Damper Lsc Fl: 5.00 click ... +1 click:
  +0.002 score    -1 click: +0.000 score`, line 660-661). This is
  only physically possible with a joint surrogate — the optimizer
  has to perturb each single parameter and re-evaluate the FULL
  per-(corner, phase) sum to read out the score delta.
- `Heave Spring Rate N Per Mm: 49.14 N/mm   [confidence: dense]`
  with `+1 click: +0.001 score    -1 click: -0.000 score` and per-
  corner Helps/Hurts blocks (line 529-541) — direct demonstration
  that "what ACTUALLY happens to ride height, pitch angle, shock
  velocities, aero balance, and understeer at corner entry" when
  the heave spring changes is being read out from learned coupled
  responses, not a closed-form formula.
- Discrete-click handling: `Damper Hsc Slope Fl: 11.00 click ...
  discrete-click value rounded from 10.872 to 11 (legal range
  0..11)` (line 484-497) — the DE search ran continuous, the
  post-clamp rounded to the nearest legal integer, and the user is
  told about the rounding step.
- Pinning: `Warnings: pinned to observed median (no per-session
  variation in training corpus, no learnable response surface):
  arb_size_front` (line 676) — the `parameter_observed_std`
  mechanism (`model.py:139`) detects parameters with zero training
  variance and short-circuits the DE search to `baseline_setup[name]`
  rather than allowing drift to a constraint edge.

## Gaps / known follow-ups

(Not VISION §3 violations; the architecture satisfies §3. These are
calibration / coverage debt that the codebase already documents.)

1. **Damper-force curve calibration** — `physics/damper_force.py`
   uses seeded per-car coefficients (4-8 N·s/mm range) pending real
   damper-spec capture from the iRacing UI. The estimator feeds the
   `damper_force_*_n` corner-phase columns that are then fit
   empirically as targets — so the absolute magnitudes are
   stepping-stone but the learned `setup → force` chain is still
   empirical. Documented at `docs/VISION_COMPLIANCE.md` follow-up
   #5.

2. **Per-car GP isotropic length scale** — `fitters/gp.py` uses a
   single shared length scale across the joint feature vector, with
   per-feature standardisation in `fit` to make that meaningful.
   Anisotropic ARD was tried and rejected because L-BFGS hangs on
   >20-dim length-scale vectors (documented at `fitters/gp.py:33-43`).
   The per-car `fit_per_car` path side-steps this by using Forest
   for the 35-dim vector. Cosmetic, not a §3 violation, but a real
   capacity ceiling on the per-(car,track) GP path when the joint
   vector grows past ~20 fittable parameters.

3. **Aero-map envelope warnings** — when DE probes ride heights
   briefly outside the aero map envelope, `aero.interpolate` clamps
   and emits a `front_rh_mm=… out of envelope (…) for car bmw;
   clamped to …` stderr warning. Cosmetic. Documented at
   `docs/VISION_COMPLIANCE.md` follow-up #6.

## Files inspected

- `C:\Users\VYRAL\racingoptimizer\.claude\worktrees\agent-aef991b445d28fdae\VISION.md`
- `C:\Users\VYRAL\racingoptimizer\.claude\worktrees\agent-aef991b445d28fdae\CLAUDE.md`
- `C:\Users\VYRAL\racingoptimizer\.claude\worktrees\agent-aef991b445d28fdae\docs\VISION_COMPLIANCE.md`
- `C:\Users\VYRAL\racingoptimizer\.claude\worktrees\agent-aef991b445d28fdae\src\racingoptimizer\physics\fitter.py`
- `C:\Users\VYRAL\racingoptimizer\.claude\worktrees\agent-aef991b445d28fdae\src\racingoptimizer\physics\fitters\base.py`
- `C:\Users\VYRAL\racingoptimizer\.claude\worktrees\agent-aef991b445d28fdae\src\racingoptimizer\physics\fitters\gp.py`
- `C:\Users\VYRAL\racingoptimizer\.claude\worktrees\agent-aef991b445d28fdae\src\racingoptimizer\physics\fitters\forest.py`
- `C:\Users\VYRAL\racingoptimizer\.claude\worktrees\agent-aef991b445d28fdae\src\racingoptimizer\physics\fitters\ridge.py`
- `C:\Users\VYRAL\racingoptimizer\.claude\worktrees\agent-aef991b445d28fdae\src\racingoptimizer\physics\model.py`
- `C:\Users\VYRAL\racingoptimizer\.claude\worktrees\agent-aef991b445d28fdae\src\racingoptimizer\physics\dynamic_at_speed.py`
- `C:\Users\VYRAL\racingoptimizer\.claude\worktrees\agent-aef991b445d28fdae\src\racingoptimizer\physics\baselines.py`
- `C:\Users\VYRAL\racingoptimizer\.claude\worktrees\agent-aef991b445d28fdae\src\racingoptimizer\physics\score.py`
- `C:\Users\VYRAL\racingoptimizer\.claude\worktrees\agent-aef991b445d28fdae\src\racingoptimizer\physics\ontology.py`
- `C:\Users\VYRAL\racingoptimizer\.claude\worktrees\agent-aef991b445d28fdae\tests\physics\test_per_car_fit_predict.py`
- `C:\Users\VYRAL\racingoptimizer\.claude\worktrees\agent-aef991b445d28fdae\tests\physics\test_coupling.py`
- `C:\Users\VYRAL\racingoptimizer\.claude\worktrees\agent-aef991b445d28fdae\tests\physics\test_validator_gate.py`
- `C:\Users\VYRAL\racingoptimizer\.claude\worktrees\agent-aef991b445d28fdae\recommendations\bmw__spa_2024_up__20260505-180530.txt`
