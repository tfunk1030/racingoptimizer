# AUDIT.md

Audit of `racingoptimizer`, refreshed **2026-06-10** (supersedes the 2026-06-08
pass; original finding IDs are preserved so cross-references in `CLAUDE.md` stay
valid). Findings only — **nothing was fixed**. Severity is engineering/product
risk, ranked High / Med / Low. Every claim carries a `file:line` reference
verified by reading the code at HEAD (`9db6d30`), unless marked *(inferred)* or
*(agent-reported)*.

Headline: the codebase remains clean, modern, and unusually well documented,
with **no injection, secret, or remote-code-execution risks**. But the picture
degraded since 2026-06-08: **CI is red on master**, the **held-out integrity
system was severed** by a doc-directory deletion, and the core product promise —
"fully accurate and correlated physics-based optimizer" — still has **no
committed empirical validation** (held-out per-channel results, lap-time
Spearman) backing it.

---

## New findings since 2026-06-08

### N1 — CI is red on master: two stale tests vs. intentional garage-step snap (High)
- **Where:** `tests/cli/test_post_clamp_discrete.py:110-120` vs
  `src/racingoptimizer/physics/ontology.py:326-328` (`brake_bias_pct` gained
  `step=0.5`) and `ontology.py:346-348` (`diff_preload_nm` gained `step=5.0`),
  both introduced by commit `785b87b` ("belleisle", W6). Same steps duplicated at
  `ontology.py:593-595` and `:605-607`.
- **What:** `_post_clamp` correctly snaps continuous params to their garage step
  (the behaviour the W6 test `tests/physics/test_garage_step_snap.py` asserts),
  but the two older tests still assert step-less precision
  (`assert 47.5 == 47.3`, `assert 75.0 == 75.5`). Verified against the live CI
  run for master merge `9db6d30` (workflow run 27168788022): `2 failed, 1014
  passed, 4 skipped … in 5225.05s`.
- **Risk:** Every push/PR fails the `Pytest (fast)` step, so the **whole CI
  signal is dead** — and because the failing step aborts the job, the
  `Verify held-out integrity` step is *skipped*, masking N2 below.
- **Proposed fix:** Update the two tests to assert the snapped values (47.5,
  75.0) — the snap is the intended product behaviour; the tests are the stale
  half. One-line each.
- **Status: FIXED on this branch (2026-06-10, user-approved):** both tests now
  assert the step-snapped values
  (`test_brake_bias_snaps_to_half_pct_step` / `test_diff_preload_snaps_to_5nm_step`).

### N2 — `docs/physics-rebuild/` was deleted but is still load-bearing (High)
Commit `a4e4f5f` ("belleisle") deleted the entire `docs/physics-rebuild/` tree,
including `holdout.sha256` and `holdout_accuracy_latest.json`. Four consumers
still point at it:

1. **Held-out integrity check is inoperative.**
   `scripts/verify_holdout.sh:24` reads
   `docs/physics-rebuild/holdout.sha256`; exits **4** when missing.
   The actual protection (hash check, catalog-flag check, pickle-leak check)
   has not run since the deletion.
2. ~~Weekly accuracy gate writes into a nonexistent directory~~ —
   **retracted on closer read**: both gate scripts already
   `mkdir(parents=True)` before writing
   (`scripts/holdout_accuracy_gate.py:957`,
   `scripts/lap_time_correlation_gate.py:425`).
3. **Briefing error-budget header silently never renders.**
   `src/racingoptimizer/explain/narrative.py:52` loads the gate JSON; the P3.1
   per-channel error-budget block falls back to the legacy confidence line for
   every briefing, permanently, with no warning.
4. **Weekly "Day 12b evaluator calibration gate" is vacuous.**
   `a4e4f5f` also deleted `scripts/day_12b_calibrate_evaluator.py`;
   `tests/test_calibration_gate.py:23` now skips with
   "day_12b_calibrate_evaluator.py missing", so the cron step
   (`ci.yml:53-54`) passes by skipping everything.

- **Status: LARGELY REPAIRED on this branch (2026-06-10):**
  `docs/physics-rebuild/holdout.sha256` restored from `94ce009` and
  `holdout_accuracy_latest.json` restored from `ff357c8` (the newest revision
  with populated per-channel data; the post-W6 revision had empty channel
  arrays). Verified live: the briefing error budget renders again
  (`bmw @ spa_2024_up` → "peak lateral G +/- 0.78 g (noisy)" …), and
  `verify_holdout.sh` gains a distinct exit **5** + message for
  unmaterialised-LFS-pointer checkouts so a pointer clone reads as "data not
  fetched", not as a bogus tampering MISMATCH.
  `day_12b_calibrate_evaluator.py` is restored from `a4e4f5f^` (lint-fixed;
  all imports verified against the current package). Remaining: the weekly
  job needs the LFS budget back to actually hash the IBTs (N5).

### N3 — The "fast" CI suite takes 87 minutes (Med)
- **Where:** CI run 27168788022, `Pytest (fast)` step: 21:44 → 23:11 (5225 s).
  `CLAUDE.md` documents the `-m "not slow"` suite as "~2 min".
- **What:** With LFS-materialised `ibtfiles/` in CI (checkout uses `lfs: true`,
  `ci.yml:14-16`), corpus-gated "fast" tests parse real multi-MB IBTs. The
  recently added Watkins Glen IBTs (commits `1d5a930`…`f67b764`) plausibly
  worsened this *(inferred — per-test timing not in the log)*.
- **Risk:** A 1.5-hour PR loop kills iteration and makes people merge red.
- **Proposed fix:** Run per-test durations (`pytest --durations=25`) and move
  real-IBT parsing tests behind the `slow` mark or a session-scoped cached
  fixture; or stop materialising LFS in the fast job.

### N4 — ~60 phantom gitlinks committed under `.claude/worktrees/` (Med)
- **Where:** `git ls-files -s | awk '$1==160000'` → 60+ entries like
  `.claude/worktrees/agent-a0a234e7c2d538764`, with **no `.gitmodules`**.
- **What:** Agent worktrees were committed as bare gitlinks (mode 160000).
  Already breaking tooling: CI post-job cleanup logs
  `fatal: No url found for submodule path '.claude/worktrees/agent-…' in
  .gitmodules` (run 27168788022).
- **Risk:** Any `git submodule` operation errors; fresh clones get confusing
  empty dirs; future `git add -A` keeps re-adding them.
- **Proposed fix:** `git rm --cached -r .claude/worktrees/` and add
  `.claude/worktrees/` to `.gitignore`.

### N5 — GitHub LFS budget exhausted: CI cannot even check out (High, blocking)
- **Where:** workflow run 27248186852 (PR #94, 2026-06-10), `lint-and-test` →
  checkout step: `batch response: This repository exceeded its LFS budget. The
  account responsible for the budget should increase it to restore access.` —
  three retries, then job failure ~40 s in. `.github/workflows/ci.yml:14-16`
  checks out with `lfs: true`.
- **What:** Every CI job fetches the full LFS object set (`ibtfiles/`, multi-GB)
  on every run. Combined with N3 (87-minute fast suite, so runs are frequent and
  heavyweight) and the 2026-06-08 Watkins Glen IBT pushes, the account's LFS
  bandwidth budget is now exhausted. Until it resets or is increased, **every CI
  run on every branch fails at checkout** — the N1 test failures aren't even
  reached.
- **Risk:** Total loss of CI signal; merges proceed unverified.
- **Proposed fix (pick one or both):**
  1. *No-cost code fix:* set `lfs: false` in the per-PR `lint-and-test` job.
     `tests/conftest.py:31-60` already skips corpus-gated tests when fixtures
     are unmaterialised LFS pointers, so the fast suite passes legitimately in
     minutes (also fixing N3 for PRs). Keep `lfs: true` only on the weekly
     `calibration-weekly` job. Note the per-PR `verify_holdout.sh` hash check
     needs the real IBT bytes, so it would move to the weekly job too (it is
     currently broken anyway, N2).
  2. *Billing fix:* increase the LFS data-pack budget on the GitHub account —
     restores the status quo, including its multi-GB-per-run bandwidth burn.
- **Status: MITIGATED on this branch (2026-06-10, user-approved):** option 1
  applied — the per-PR `lint-and-test` job no longer fetches LFS and the
  (broken, LFS-dependent) `verify_holdout.sh` step moved out of the per-PR job
  (it already runs in `calibration-weekly`). This also resolves N3 for PRs.
  The weekly job keeps `lfs: true` and still needs the budget restored.
  One latent land mine was fixed to make this viable:
  `tests/physics/test_per_car_recommend_fast.py` gated only on `.exists()`,
  so on a pointer-only checkout it fed the 133-byte LFS pointer to irsdk —
  the documented runaway-allocation failure (`tests/_lfs_util.py:1-9`) —
  and got OOM-killed. It now uses the standard
  `is_unmaterialised_lfs_pointer` skip. Verified locally on a pointer-only
  clone (the exact post-fix CI environment): `917 passed, 103 skipped,
  0 failed in 8.4 s`.

---

## Accuracy & correlation state (the product goal)

The stated goal is a *fully accurate and correlated physics-based optimizer*.
The code infrastructure for validation is merged, but **no empirical evidence
is committed**. Status of the accuracy-rebuild definition-of-done
(`docs/accuracy-rebuild-2026-05-24/PLAN.md` §5):

### Independent read of the committed evidence (2026-06-10)

**Fresh measurement (2026-06-10, this session):** with the LFS budget
restored, the full 192-IBT corpus was pulled, ingested, leak-guarded
(`mark_holdout_sessions.py`, `verify_holdout.sh` exit 0), and
`holdout_accuracy_gate.py` was run — the first measurement since the v5-era
log. As-found at schema v8 the verdict was **per-channel gate 0/5, aggregate
2/5**, with a newly-discovered rear-RH regression; after root-causing and
fixing that regression in this session (schema v9, see below) the gate is
**aggregate 5/5 pass, per-channel still 0/5** but with the failure surface
collapsed. Committed JSON
(`docs/physics-rebuild/holdout_accuracy_latest.json`) reflects the **v9
post-fix** state. The trajectory across all three measurements:

| channel (mean_abs) | budget | v5 era | v8 (aero on) | **v9 (aero off, fix)** |
|---|---|---|---|---|
| accel_lat_g_max — bmw | 0.30 g | 0.789 | 0.365 | **0.296 PASS** |
| accel_lat_g_max — porsche | 0.30 g | 1.126 | 0.419 | **0.294 PASS** |
| accel_lat_g_max — ferrari | 0.30 g | 0.921 | 0.328 | **0.265 PASS** |
| accel_lat_g_max — cadillac | 0.30 g | 0.799 | 0.437 | **0.285 PASS** |
| understeer — cadillac | 0.10 rad | 0.712 | 0.219 | **0.112** |
| understeer — bmw | 0.10 rad | 0.350 | 0.188 | **0.140** |
| rr_ride_height — bmw | 3.0 mm | 4.628 | 10.375 (regr.) | **3.545** |
| lr_ride_height — cadillac | 3.0 mm | 5.858 | 10.490 (regr.) | **2.668 PASS** |

- **The W5/W6 rebuild worked where it aimed:** peak lat-G error halved on
  every car, understeer down 46–69 %, longitudinal-G channels now PASS on
  4/5 cars, and driver-input channels are near-zero error. The remaining
  lat-G/understeer gaps are 1.1–2.6× over budget (was 2.3–7.1×).
- **Rear ride-height regression — ROOT-CAUSED AND FIXED (2026-06-10).** lr/rr
  dynamic RH had doubled in error on BMW/Cadillac with coverage collapsing
  (BMW rr 0.99 → 0.35) while the regime read "confident". A controlled
  held-out ablation (same corpus, same seed) isolated the cause to the **W6
  aero-map fit features** (`physics/aero_fit_features.py`), not the P2.2
  track intercepts:
  - clearing `track_random_intercepts` at predict time moved every RH
    channel by < 0.3 mm (not the cause);
  - refitting with `attach_aero_map_features` disabled moved BMW rr RH
    **10.38 → 3.55 mm** (coverage 0.35 → 0.99) and Cadillac lr **10.49 →
    2.67 mm** (0.27 → 0.94), and pulled peak lat-G under budget on 4/5 cars.

  Root cause: the feature was computed at **observed dynamic platform RH** at
  fit time but approximated from **static garage readouts** at predict time —
  a 10–40 mm train/serve distribution skew (rear static RH even clamps at the
  map's 50 mm ceiling). The Forest followed the skewed feature into
  sparsely-trained regions. Fix: `AERO_MAP_FIT_FEATURES_ENABLED = False`
  (gated at both call sites; helpers kept pure + tested),
  `ENV_FEATURE_SCHEMA_VERSION_PER_CAR` 8 → 9 invalidates every stale pickle.
  Re-running the full gate gives **aggregate 5/5 pass** (was 2/5) and
  per-channel failure counts of bmw 6, cadillac 2, ferrari 4, acura 8,
  porsche 2 (was 7/7/6/8/7). Committed JSON reflects the post-fix state.
- **Remaining per-channel blockers (post-fix), ranked by tractability:**
  1. **`damper_force_p99_n` — gate-classification inconsistency (NEW finding,
     fit-quality-proven).** Blocks 4/5 cars (271–354 N vs ~145–189 budget).
     Measured per-channel fit skill `cv_residual_std / signal_std` (lower =
     better, from a fitted Cadillac model; `physics/model.py:68-69`):

     | channel | gated? | skill |
     |---|---|---|
     | `damper_force_p99_n` | **GATED** | 0.650 |
     | `damper_force_mean_n` | exempt | 0.578 |
     | `damper_velocity_p99_mms` | exempt | 0.755 |
     | `damper_velocity_mean_mms` | exempt | 0.605 |
     | grip + dynamic-RH channels | gated | **0.13–0.31** |
     | throttle / brake / wheel-speed (driver-input) | exempt | 0.17–0.36 |

     The whole damper family clusters at 0.58–0.76 — structurally distinct
     from the genuinely setup-driven channels (0.13–0.31) and *worse*-fit than
     the exempt driver-input channels. `damper_force` is a deterministic
     digressive transform of damper velocity (`physics/damper_force.py:102`),
     which CLAUDE.md already classifies as a structural fit ceiling. Gating the
     p99 force while exempting `damper_force_mean_n`,
     `damper_velocity_p99_mms`, and `damper_velocity_mean_mms` (same family,
     equal-or-better fit) was internally inconsistent. **APPLIED 2026-06-16
     (user decision):** `damper_force_p99_n` moved to informational
     (`scripts/holdout_accuracy_gate.py:97-116`, locked by
     `tests/physics/test_gate_channel_classification.py`). It does NOT make
     any car pass on its own — after the change cadillac and porsche drop to a
     **single** remaining blocker each (understeer), bmw to 5, ferrari to 3,
     acura unchanged at 8 — so it is an honesty fix, not a gate-gaming lever
     (per-channel gate stays 0/5).
  2. **understeer near-misses** — cadillac 0.112, porsche 0.138 vs 0.10 rad.
     Genuinely close; the only blocker on those two cars besides damper.
     **Forest-hyperparameter tuning ruled out (2026-06-16, user-authorized
     investigation):** held-out understeer on cadillac under refit variants —
     baseline (n50/d15) **0.1124**, shallower d10 **0.1163**, d8 **0.1195**.
     Shallower trees make it *worse*, so the model is not overfitting and the
     obvious fitter-tuning lever does not exist. Note understeer
     (`SteeringWheelAngle − k·AccelLat`, `corner/states.py:555`) is partly
     driver-input like damper, but it is the canonical balance metric and the
     single most important gated channel, so it is NOT a reclassification
     candidate. The remaining gap is a genuine model-accuracy limit on a tight
     0.10 rad budget; a real improvement would need a predict-time balance
     feature the surrogate doesn't have (the kinematic slip-angle in
     `physics/diagnostic_state.py` is telemetry-derived, unavailable at
     predict time — dead end for setup recommendation).
  3. **front dynamic RH** — bmw lf 4.27, ferrari lf 4.07 / rf 3.72 vs 3.0 mm.
     Note disabling the aero features slightly *raised* front-lf error on
     bmw/ferrari (3.49 → 4.27) while massively fixing the rear — a residual
     of the same train/serve issue, now front-only.
  4. **acura — thin corpus, not a code bug.** 8 fails incl. accel_lon_g_min
     1.011 (banked Daytona). Axle-ceiling fit needs ≥ 100 corner-phase rows;
     acura has ~36 (CLAUDE.md). Needs more sessions, not more code.

  **Net:** the per-channel gate is 0/5 and is *not* closable by code alone on
  this corpus — #1 is a metric-definition call, #4 is a data-volume problem,
  and #2/#3 are real (tractable) model-accuracy work.

The earlier read of the v5-era log (kept for the trend): 34/34 gated pairs
failed, lat-G 2.3–3.8× over, understeer 3.5–7.1× over; coverage was high
(0.84–1.0) because CIs were wide — the score path consumes the mean, not
the CI.

Independent aero-map analysis (the 33 `aero-maps/*.json` are real data in
this clone): all five cars share the same calibrated envelope (front RH
25–75 mm, rear 5–50 mm); the balance gradient at the front floor is
~0.18–0.19 %/mm on every car. Cadillac's observed ~8.4 mm front RH is
**16.6 mm below the floor**, so every clamped query carries a front-balance
bias of **≥ ~+3 %** (floor-gradient × excursion; true bias below the floor
is unknowable from the map) — large in GTP terms, one-directional, and since
W6 it also contaminates the `aero_map_*` *training* features
(`physics/aero_fit_features.py`). The Acura spec's own 15 mm front-RH target
(`docs/cars/acura_arx06.md`) is also below the floor — this is a fleet-wide
domain mismatch, not a Cadillac quirk.

**Optimizer improvement applied on this branch (2026-06-10, user-directed):**
out-of-domain aero is now detected and made consequential —
`aero/interpolator.py::AeroClampStats` counts clamped queries + max excursion
per `AeroSurface`; `cli/recommend.py::_aero_out_of_domain_warnings` (wired
after `_post_clamp`) emits a briefing/JSON warning quantifying the bias and
downgrades the aero-driven families (`rear_wing`, `pushrod`, `perch_offset`,
`ride_height`) one confidence tier; `physics/score.py` warn-once replaces the
silent constant-default aero fallback (M5). Re-deriving the maps below 25 mm
front RH remains the real fix (H2).

| DoD item | Criterion | Status | Evidence |
|---|---|---|---|
| §5.1 held-out gate | green on all 5 cars, per-channel | **Unproven** — result JSON deleted (N2); `scripts/_holdout_run_latest.log` shows the *aggregate* gate passing all 5 cars (median normed residual 0.56–0.70) but prints no per-channel pass/fail | `scripts/holdout_accuracy_gate.py:86` `_PER_CHANNEL_THRESHOLDS`; log tracked in repo |
| §5.3 lap-time Spearman | ρ ≥ 0.30 per qualifying (car, track) pair | **STILL FAIL, but the aero-feature disable flipped the sign — accuracy & correlation share a root.** As-found (aero features on): cadillac @ lagunaseca ρ = **−0.067** (n=12), bmw @ spa ρ = **−0.222** (n=14) — both anti-correlated. After disabling the W6 aero features (same pairs, same corpus): **cadillac ρ = −0.067 → +0.235** (+0.30) and **bmw ρ = −0.222 → −0.033** (+0.19). Both pairs improved by 0.19–0.30 in the positive direction — the same train/serve skew that was degrading lat-G/rear-RH accuracy was also depressing the score↔pace ranking. But neither clears 0.30: cadillac is now positive-but-short, bmw is ≈zero (uncorrelated). So the accuracy fix *partially* lifts correlation; it does not close it.

**Measurement-fidelity bug found + fixed, and it confirms the gap is real (2026-06-16).** The LOSO loop trained each refit on *same-track sessions only* (`rest = session_ids - held`), but production (`optimize <car> <track>`) pools the car's sessions across *all* tracks — so the gate was scoring a strictly weaker model than ships, biasing correlation downward. Fixed: the refit now pools the full per-car production corpus minus the held session (`lap_time_correlation_gate.py:258-282`). Re-measured cadillac under the production-faithful fit: **ρ = 0.235 → 0.242** — essentially unchanged. This **rules out the measurement-artifact hypothesis**: the ~0.24 correlation ceiling is a genuine property of the objective, not an artifact of thin same-track training. The residual gap is the objective formulation itself (per-corner-phase utilization, VISION §6; evaluator within-group Spearman ~0.19 vs the 0.35 PLAN target, CLAUDE.md) — closing it is a scoring redesign, not a fit fix. (bmw all-track re-measure pending — 53-session pool, ~hours offline.)

**Decision (2026-06-16, user): honor VISION §6 — accept the ~0.24 proxy correlation.** The "correlated ≥ 0.30" target conflicts with the project's foundational rule ("lap time is an outcome, not the optimization signal; never feed lap time into the objective"). Reaching ≥ 0.30 would require either feeding lap time into the score (violates §6) or a multi-day per-car/per-phase evaluator-weight re-search with uncertain payoff (the Day-12b calibration already pushed this to ~0.19). The product posture stays a **guardrailed physics-utilization optimizer**, not a lap-time predictor — the correlation number is a validation signal, not the objective. This is a deliberate, documented limitation, not an open bug. Caveats unchanged: each pair underpowered (95 % CI ≈ ±0.55); LOSO refit uses same-track sessions only (`lap_time_correlation_gate.py:260`) — thinner than the production pooled fit; 4 qualifying pairs (~118 refits) still unmeasured. Committed JSON reflects the post-fix values. First execution also crash-fixed a latent `SessionRow.valid` bug. | `scripts/lap_time_correlation_gate.py:31,161-380` |
| §5.6 in-garage static RH | within 1 mm | **Unvalidated offline** — kinematic fit ships gated on in-sample R² ≥ 0.98 only | `physics/static_rh_kinematic.py` *(agent-verified)* |
| Evaluator lap-time correlation | target 0.35 (fallback 0.20) | **Below target**: BMW +0.189, Cadillac +0.122, Ferrari +0.249 (only Ferrari passes fallback); Porsche undocumented; Acura uncalibrated (default weights) | `src/racingoptimizer/physics/evaluator.py:86-101` |
| Hybrid ≥ surrogate A/B (P1.3) | hybrid not >20 % below surrogate on H1–H5 | Wired into weekly cron only; no committed results | `tests/physics/test_hybrid_heldout_ab.py` assert `total_h >= total_s * 0.80` *(agent-verified)*; `ci.yml:59-60` |

Structural blockers documented in-repo (verified locations):
- Driver-input channels plateau at fit-quality ~0.50 — signal == noise
  (`CLAUDE.md` "Known regressions / gaps"). No model fix possible without
  driver-input labels the IBT format lacks.
- Per-car cross-track confounding: parameters constant within the dominant
  track inherit its philosophy at under-sampled tracks (`CLAUDE.md` "Per-car v4
  cross-track confounding"); P2.2 random intercepts shipped but no before/after
  delta committed.
- Acura corpus thinnest (~33 of 192 IBTs by filename prefix *(agent-counted)*),
  no evaluator calibration (`evaluator.py:99-100`).

---

## Carried findings (2026-06-08), current status

### H1 — Accuracy unvalidated on the full corpus and not PR-gated — **PARTIALLY FIXED on this branch**
- `.github/workflows/ci.yml`: the holdout gate, hybrid A/B, and lap-time
  gate run only `if: github.event_name == 'schedule'` (weekly). Per-PR CI runs
  lint and fast pytest. A change that degrades recommendation accuracy can
  still merge green.
- **Worse than first reported (verified 2026-06-10):** the weekly job had
  **never actually measured anything**. Every gate needs
  `corpus/catalog.sqlite` + parquet sessions, and the workflow never built a
  corpus (no `optimize learn` step): `tests/test_calibration_gate.py:23-25,43`
  skips both calibration tests without a catalog,
  `lap_time_correlation_gate.py:127-158` returns an empty pair dict ("no
  qualifying pairs … skipping", exit 0), and the holdout gate short-circuits
  the same way. Gate JSONs were also never uploaded, so even a hypothetical
  run's evidence evaporated with the runner.
- **Fixed on this branch (2026-06-10):** `calibration-weekly` now (1) builds
  the corpus (`uv run optimize learn ./ibtfiles`), (2) re-applies the
  gate-only flags via the new `scripts/mark_holdout_sessions.py` — a fresh
  ingest writes `held_out=0`, so without this step the per-car fits would
  *train on the held-out IBTs* (leak), (3) runs `verify_holdout.sh` before
  the gates, and (4) uploads `docs/physics-rebuild/*.json` as a run artifact
  with `if: always()`. `day_12b_calibrate_evaluator.py` is restored from
  `a4e4f5f^` so the calibration step is no longer vacuous. The job is
  capped at `timeout-minutes: 350`.
- **Still open:** the weekly job needs the LFS budget restored to fetch
  telemetry (N5) — until then the ingest step has only pointer files; and a
  cheap per-PR accuracy smoke remains unbuilt.

### H2 — Cadillac ride heights clamped out of the aero-map envelope — **STILL OPEN**
- Clamp logic unchanged: `aero/interpolator.py:46-52` (`_clamp`), `:150-172`
  (DEBUG-level log only — deliberately demoted from WARNING to avoid spam). No
  confidence downgrade anywhere when the query point is out of domain. Historic
  evidence (`err.log`, 31k lines of `front_rh_mm=8.43 out of envelope (25.0,
  75.0) for car cadillac`) was deleted in `a4e4f5f`, but the *mechanism* is
  untouched, so Cadillac aero balance / L-D is still evaluated at the 25 mm map
  floor while the car runs ~8 mm. `docs/watkins-glen-runbook.md:79-80`
  acknowledges the issue without remedy.
- **Fix:** re-derive/extend the Cadillac map below 25 mm front RH, or apply an
  explicit out-of-domain confidence downgrade + briefing warning when clamping
  fires during scoring.
- **Status: PARTIALLY MITIGATED on this branch (2026-06-10, user-directed):**
  the downgrade-and-warn half is implemented (`AeroClampStats` accounting in
  `aero/interpolator.py`; `_aero_out_of_domain_warnings` in
  `cli/recommend.py` warns with a quantified floor-gradient bias estimate and
  downgrades `rear_wing`/`pushrod`/`perch_offset`/`ride_height` one tier).
  Map re-derivation below the 25 mm front floor remains open — see the
  "Independent read" section for why this is fleet-wide.

### M1 — Generated artifacts committed — **PARTIALLY FIXED**
- Cleaned since the last audit: `err.log` and all `recommendations/*.txt` are no
  longer tracked (`git ls-files recommendations/` → 0; both removed in
  `a4e4f5f`).
- Still tracked: `_status.txt`, `_status_filtered.txt`, `status.md` (generated
  status dumps at repo root), `scripts/_holdout_run.log`,
  `scripts/_holdout_run_latest.log`. `.gitignore` covers none of them.
- **Fix:** `git rm --cached` the five files; gitignore `_status*`, `status.md`,
  `scripts/_holdout_run*.log`, `recommendations/`, `err.log`. (Note:
  `_holdout_run_latest.log` is currently the *only* record of held-out gate
  results — capture its content into a committed dated JSON before deleting.)

### M2 — Orphaned exploratory scripts at repo root — **STILL OPEN**
- `categorize_13.py`, `telemetry_discovery.py` still tracked; both import
  `irsdk` directly and duplicate `ingest/parser.py::_read_yaml`; imported
  nowhere in `src/` or `tests/` *(agent-verified)*.

### M3 — Model-cache load bypassed the type guard — **FIXED** (`1c30b33`)
- Both cache-load sites now route through `physics.io.load` (isinstance
  `PhysicsModel` guard) and echo
  "ignoring stale/incompatible model cache … refitting" on stderr:
  `cli/recommend.py:1296-1312` (per-car) and `:1358-1374` (per-track). Guard
  contract test added at `tests/physics/test_io_guard.py`.

### M4 — Data-protection hook over-blocks read-only commands — **STILL OPEN**
- `.claude/hooks/protect-data.sh:16-18` unchanged; `>[[:space:]]*[^|]` still
  matches `2>/dev/null`, blocking read-only `ls ibtfiles … 2>/dev/null`.

### M5 — Silent aero fallback defaults in the DE objective — **FIXED on this branch (2026-06-10)**
- Was: `physics/score.py:62-64` defaults (`balance=50%`, `L/D=3.5`) used
  silently when the aero surface is `None`. Now
  `_aero_surface_or_none` warns once per car
  (`physics/score.py::_warn_aero_defaults_once`) that aero terms carry no
  signal for that car.

### L1 — `per_track_residuals` retired but still written — **STILL OPEN**
- Computed-as-empty and stored: `physics/fitter.py:1302,1351`; slot kept at
  `physics/model.py:181`, backfilled at `:271`, explicitly not read (`:592`
  comment).

### L2 — `segment_lap` dead `track_model` kwarg — **STILL OPEN**
- `corner/states.py:215-218` still raises `NotImplementedError`.

### L3 — Long-G is a hardcoded phase constant — **STILL OPEN**
- `physics/score.py:810-818` (`mid_corner=0.0`, `braking/trail=-0.5`,
  `exit=0.3`), used at `:874,:1000`. Under-allocates rear Fz (documented safe
  direction).

### L4 — Cold-start TrackModel silently treats all samples as clean — **STILL OPEN**
- `track/builder.py:355-357` returns empty mask frames for <3 sessions; no
  briefing note. Newly relevant: **every car at Watkins Glen has exactly one
  session** (commits `1d5a930`…`f67b764`), so the upcoming Watkins Glen
  recommendations will train without curb/off-track masking.

### L5 — Fragile sklearn pickle round-trips — **STILL OPEN**
- `physics/fitters/ridge.py:121-136` triple round-trip unchanged; CI resolved
  scikit-learn 1.9.0 (run 27168788022 install log) vs the `>=1.5` floor.

### L6 — Loose dependency floors, no upper bounds — **STILL OPEN**
- `pyproject.toml` unchanged; CI now resolves numpy 2.4.6 / sklearn 1.9.0 /
  polars 1.41.2. `uv.lock` committed, so reproducible via `uv`; bare
  `pip install -e` would float.

---

## Verified safe (re-checked or carried; no action needed)

- **SQL:** parameterised throughout (`ingest/catalog.py`); the only f-string SQL
  interpolates a hardcoded additive-column constant.
- **Secrets:** none (grep for key/token/password patterns — no hits).
- **Dangerous calls:** no `eval`/`exec`/`os.system` in `src/`.
- **YAML:** safe loader (`ingest/parser.py` `CustomYamlSafeLoader`).
- **Pickle:** only locally-generated model caches under gitignored
  `corpus/models/`; production loads now go through the `physics.io.load`
  type guard (M3 fixed).
- **VISION §6 integrity restored:** commit `ccc0dee` removed
  `_track_fastest_observed_value` and the `track_best_value` pin branch from
  `physics/recommend.py` — lap time no longer selects setup values (it remains,
  legitimately, the corner time-sensitivity weight at fit time).
- **Ontology integrity:** `f16d0a8` set unverified brake-duct / throttle-map
  params to `fittable=False, user_settable=False` (no CarSetup YAML leaves
  exist for them), unbreaking `test_per_car_setup_yaml_resolves_every_user_input`.

---

## Suggested first actions (cheap, high-leverage, in order)

0. **N5** — unblock CI entirely: either drop `lfs: true` from the per-PR job
   (tests already skip unmaterialised fixtures) or raise the LFS budget;
   nothing else in CI matters until checkout succeeds (15 min / billing).
1. **N1** — fix the two stale asserts in
   `tests/cli/test_post_clamp_discrete.py:110-120` → CI signal restored (10 min).
2. **N2** — ~~restore the holdout manifest + gate JSON~~ **done on this branch
   (2026-06-10)**; remaining: restore or retire
   `scripts/day_12b_calibrate_evaluator.py` (the weekly calibration step is
   vacuous without it).
3. **N4 + M1** — drop the `.claude/worktrees/` gitlinks and the five generated
   artifacts; extend `.gitignore` (15 min).
4. **H1/§5.1** — run `scripts/holdout_accuracy_gate.py` offline on the full
   corpus, commit the dated per-channel JSON; that is the single biggest step
   toward an evidence-backed "accurate and correlated" claim.
5. **§5.3** — implement/run the LOSO lap-time Spearman offline for the two or
   three densest (car, track) pairs; commit results even if they fail the 0.30
   target — knowing the number beats a placeholder gate.
6. **H2** — Cadillac aero-map extension or out-of-domain downgrade (scoping).
