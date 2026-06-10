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

### N2 — `docs/physics-rebuild/` was deleted but is still load-bearing (High)
Commit `a4e4f5f` ("belleisle") deleted the entire `docs/physics-rebuild/` tree,
including `holdout.sha256` and `holdout_accuracy_latest.json`. Four consumers
still point at it:

1. **Held-out integrity check is inoperative and will hard-fail CI.**
   `scripts/verify_holdout.sh:24` reads
   `docs/physics-rebuild/holdout.sha256`; `:27-30` exits **4** when missing.
   It runs on every push/PR (`.github/workflows/ci.yml:34-35`). Today it is
   *skipped* because the pytest step fails first — fix N1 and every CI run will
   fail here instead. Meanwhile the actual protection (hash check, catalog-flag
   check, pickle-leak check) has not run since the deletion.
2. **Weekly accuracy gate writes into a nonexistent directory.**
   `scripts/holdout_accuracy_gate.py:942` writes
   `docs/physics-rebuild/holdout_accuracy_latest.json`; with the parent dir gone
   the write raises `FileNotFoundError` *(inferred from `Path.write_text`
   semantics; not executed)*.
3. **Briefing error-budget header silently never renders.**
   `src/racingoptimizer/explain/narrative.py:52` loads the same JSON; the P3.1
   per-channel error-budget block falls back to the legacy confidence line for
   every briefing, permanently, with no warning.
4. **Weekly "Day 12b evaluator calibration gate" is vacuous.**
   `a4e4f5f` also deleted `scripts/day_12b_calibrate_evaluator.py`;
   `tests/test_calibration_gate.py:23` now skips with
   "day_12b_calibrate_evaluator.py missing", so the cron step
   (`ci.yml:53-54`) passes by skipping everything.

- **Proposed fix:** Restore `docs/physics-rebuild/holdout.sha256` (recoverable
  via `git show 94ce009:docs/physics-rebuild/holdout.sha256`) or move the
  manifest to `scripts/`/`docs/holdout/` and update the four references. Decide
  whether the day-12b calibration gate is retired (then delete the cron step and
  test) or restore the script.

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

---

## Accuracy & correlation state (the product goal)

The stated goal is a *fully accurate and correlated physics-based optimizer*.
The code infrastructure for validation is merged, but **no empirical evidence
is committed**. Status of the accuracy-rebuild definition-of-done
(`docs/accuracy-rebuild-2026-05-24/PLAN.md` §5):

| DoD item | Criterion | Status | Evidence |
|---|---|---|---|
| §5.1 held-out gate | green on all 5 cars, per-channel | **Unproven** — result JSON deleted (N2); `scripts/_holdout_run_latest.log` shows the *aggregate* gate passing all 5 cars (median normed residual 0.56–0.70) but prints no per-channel pass/fail | `scripts/holdout_accuracy_gate.py:86` `_PER_CHANNEL_THRESHOLDS`; log tracked in repo |
| §5.3 lap-time Spearman | ρ ≥ 0.30 per qualifying (car, track) pair | **Never computed** — the LOSO per-pair refit is an explicit placeholder writing an empty result list; the cron step (`ci.yml:62-63`) gates on nothing | `scripts/lap_time_correlation_gate.py:31` (`_SPEARMAN_TARGET=0.30`), module docstring *(agent-verified)* |
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

### H1 — Accuracy unvalidated on the full corpus and not PR-gated — **STILL OPEN**
- `.github/workflows/ci.yml:37-39`: the holdout gate, hybrid A/B, and lap-time
  gate run only `if: github.event_name == 'schedule'` (weekly). Per-PR CI runs
  lint, fast pytest, and `verify_holdout.sh` (integrity only). A change that
  degrades recommendation accuracy merges green — and currently *everything*
  merges red (N1), which is worse.
- **Fix:** as before — add a cheap per-PR accuracy smoke (one car, held-out
  channels, loose threshold); commit a dated results JSON and assert freshness.
  Now additionally blocked on N2 (the results path no longer exists).

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

### M5 — Silent aero fallback defaults in the DE objective — **STILL OPEN**
- `physics/score.py:62-64`: `_DEFAULT_AERO_BALANCE_PCT=50.0`,
  `_DEFAULT_AERO_LD=3.5` used when the aero surface is `None`, with no warning
  or confidence downgrade. Note W6 (`physics/aero_fit_features.py`) added
  aero-map features *at fit time*, which reduces but does not remove the
  exposure at score time.

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

1. **N1** — fix the two stale asserts in
   `tests/cli/test_post_clamp_discrete.py:110-120` → CI signal restored (10 min).
2. **N2** — restore `docs/physics-rebuild/holdout.sha256` (or relocate + update
   the 4 references) → held-out integrity protection live again, weekly gate
   can write results, briefing error budget renders (1 hr).
3. **N4 + M1** — drop the `.claude/worktrees/` gitlinks and the five generated
   artifacts; extend `.gitignore` (15 min).
4. **H1/§5.1** — run `scripts/holdout_accuracy_gate.py` offline on the full
   corpus, commit the dated per-channel JSON; that is the single biggest step
   toward an evidence-backed "accurate and correlated" claim.
5. **§5.3** — implement/run the LOSO lap-time Spearman offline for the two or
   three densest (car, track) pairs; commit results even if they fail the 0.30
   target — knowing the number beats a placeholder gate.
6. **H2** — Cadillac aero-map extension or out-of-domain downgrade (scoping).
