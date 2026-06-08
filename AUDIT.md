# AUDIT.md

Onboarding audit of `racingoptimizer`, 2026-06-08. Findings only — **nothing was
fixed**. Severity is engineering/product risk, ranked High / Med / Low. Every
claim carries a `file:line` reference verified by reading the code, unless marked
*(inferred)*. Fixes are **proposed, not applied**.

Headline: the codebase is clean, modern, well-structured, and unusually well
documented. There are **no injection, secret, or remote-code-execution risks**.
The real risks are (1) recommendation **accuracy is unvalidated on the full corpus
and not gated on PRs**, with at least one car (Cadillac) showing systematic
out-of-envelope predictions, and (2) **repo hygiene** — large generated artifacts
are committed.

---

## High

### H1 — Recommendation accuracy is unvalidated on the full corpus and not gated on PRs
- **Where:** `.github/workflows/ci.yml:37-39` (accuracy gates run only `if:
  github.event_name == 'schedule'`); `CLAUDE.md` "Known accuracy gap" + accuracy
  rebuild plan `docs/accuracy-rebuild-2026-05-24/PLAN.md` (§5.1/§5.3/§5.6
  definition-of-done "require offline runs … not gated by these code changes").
- **What:** Every push/PR runs only `ruff`, the fast (`not slow`) pytest subset,
  and `verify_holdout.sh` (hash/flag/leak integrity — *not* accuracy). The held-out
  accuracy gate, hybrid-vs-surrogate A/B, lap-time Spearman gate, and evaluator
  calibration run **weekly on cron only**. A change that degrades recommendation
  accuracy can merge green. The lap-time gate (`scripts/lap_time_correlation_gate.py`,
  P1.2) is additionally a heavy offline LOSO (~2.5 hr/pair) not realistically run
  per-PR.
- **Risk:** The product's core promise ("physics-based and calibrated") is not
  continuously verified. `CLAUDE.md` itself warns "READ BEFORE TRUSTING
  RECOMMENDATIONS."
- **Proposed fix:** Add a lightweight per-PR accuracy smoke (e.g. the BMW-Sebring
  held-out channels at the observed setup with a loose threshold) so regressions
  surface on PRs; keep the full corpus gate on cron. Commit a dated
  `holdout_accuracy_latest.json` and assert it is fresh.

### H2 — Cadillac static ride-height predictions are systematically out of envelope
- **Where:** `err.log` — 31,008 lines, **all** Cadillac: e.g.
  `front_rh_mm=8.43 out of envelope (25.0, 75.0) for car cadillac; clamped to 25.0`.
  Clamp logic in `aero/interpolator.py:132-173` (front-rh clamped below the
  aero-map floor) and the static-RH envelope warnings in `cli/recommend.py`.
- **What:** The static/at-speed RH pipeline predicts front ride heights far below
  the calibrated aero-map envelope for Cadillac, clamped on every interpolate call.
  This is not cosmetic — clamping at the envelope floor means the aero balance / L/D
  the optimiser sees is evaluated at the boundary, not the true (lower) RH, biasing
  the score for that car. The same pattern likely affects other GTPs that run below
  the aero-map's calibrated front floor (`aero/interpolator.py:142-149` notes GTPs
  run below the floor generally).
- **Risk:** Cadillac recommendations rest on out-of-domain aero lookups. Med-High
  correctness for that car specifically; the aero maps may need re-derivation at
  realistic GTP geometry.
- **Proposed fix:** Re-derive/extend the Cadillac aero map down to the observed
  front-RH range, or add an explicit out-of-domain confidence downgrade when the
  predicted RH is clamped, rather than silently clamping. (Also delete `err.log`
  from the repo — see M1.)

---

## Med

### M1 — Generated artifacts committed to the repo
- **Where (all tracked in git):** `err.log` (2.9 MB / 31,008 lines, H2 evidence),
  `status.md` (0 bytes, empty), `recommendations/*.txt` (**149** generated briefing
  files), `scripts/_holdout_run.log`, `scripts/_holdout_run_latest.log`.
- **What:** Run output and an empty placeholder are version-controlled. `corpus/` is
  correctly gitignored (`.gitignore:18`) but `recommendations/` and the logs are not.
- **Risk:** Repo bloat, noisy diffs, stale data masquerading as source. Low security
  risk; real maintainability cost.
- **Proposed fix:** `git rm` the logs and empty `status.md`; add
  `recommendations/`, `err.log`, `*.log` to `.gitignore`. Keep one sample briefing
  under `docs/` if an example is wanted.

### M2 — Orphaned exploratory scripts at repo root
- **Where:** `categorize_13.py`, `telemetry_discovery.py` (both import `irsdk`
  directly, duplicating the YAML-read logic in `ingest/parser.py:_read_yaml`; neither
  is imported anywhere in `src/` *(inferred from grep)*).
- **What:** One-off Acura Belle Isle discovery scripts left at the top level. They
  re-implement parsing the package already owns.
- **Risk:** Confusion about the real entry points; drift from the canonical parser.
- **Proposed fix:** Move to `scripts/exploratory/` or delete; if kept, have them call
  `ingest.parser` rather than re-deriving YAML reads.

### M3 — Production model-cache load bypasses the type guard and swallows all errors
- **Where:** `cli/recommend.py:1298-1302` (per-car) and `:1351-1355` (per-track):
  ```python
  try:
      with cache_path.open("rb") as fh:
          return pickle.load(fh)
  except Exception:
      pass
  ```
- **What:** The hardened loader `physics/io.py:20-24` (isinstance `PhysicsModel`
  guard, `# noqa: S301 trusted offline artefact`) is **not** used here; the
  production path calls raw `pickle.load`. The bare `except Exception: pass` also
  swallows the `TypeError` that P1.4's `_validate_pickle_slots` (`model.py:880-938`)
  raises on a corrupt/wrong-typed slot — so the "point the user at `--no-cache`"
  protection silently degrades to "refit instead." Refit-on-failure is desirable, but
  any genuinely malformed cache is hidden with no log line.
- **Risk:** Low security (corpus is local + gitignored, trust boundary is the local
  filesystem), but the slot-validation safety net is effectively neutralised at the
  one site that matters, and failures are invisible.
- **Proposed fix:** Route both cache loads through `physics.io.load`, and on
  exception emit a stderr note ("stale/incompatible model cache — refitting") before
  falling through to `fit_per_car`.

### M4 — Data-protection hook over-blocks read-only commands
- **Where:** `.claude/hooks/protect-data.sh:16-18`.
- **What:** The block fires when the command string *merely contains* `ibtfiles` or
  `aero-maps` **and** matches the destructive-verb regex — which includes any output
  redirect `>`. Because `2>/dev/null` matches `>[[:space:]]*[^|]`, a read-only
  `ls ibtfiles ... 2>/dev/null` (or `git ls-files | grep ibtfiles`) is blocked even
  though it touches nothing. Verified live during this audit — a read-only
  `git ls-files`/`ls` command was blocked.
- **Risk:** Genuine friction; encourages `--no-verify`-style workarounds the hook is
  meant to prevent.
- **Proposed fix:** Scope the destructive-verb match to the redirect *target* (only
  block when the path after `>`/`rm`/`mv` is inside the protected dirs), and exclude
  `2>`/`2>&1` stderr redirects. Keep the intent (block writes/deletes into
  `ibtfiles/` & `aero-maps/`).

### M5 — Silent fallbacks in the DE objective mask missing aero data
- **Where:** `physics/score.py:62-64` (`_DEFAULT_AERO_BALANCE_PCT=50%`,
  `_DEFAULT_AERO_LD=3.5`), used when the aero surface is `None`; aero load can fall
  through to `None` silently (`physics/fitter.py` aero-cache path *(inferred)* +
  `aero_correction_available` set without a guaranteed map). Reported by the physics
  mapping pass.
- **What:** If an aero map fails to load, the optimiser scores against constant
  balance/L-D defaults with no warning, so DE can optimise on fabricated aero.
- **Risk:** Wrong recommendations that still report "dense" confidence (confidence is
  track-wide, not aero-aware). Med correctness, low likelihood (maps are present in
  the repo).
- **Proposed fix:** Emit a one-time stderr warning + downgrade confidence when the
  aero surface is unavailable for a car the optimiser is actively using.

---

## Low

### L1 — `per_track_residuals` is retired but still written and carried
- **Where:** computed-as-empty and stored at `physics/fitter.py:1278,1327`; kept on
  the model (`model.py:181`) and explicitly **not read** by predict (`model.py:566`
  comment). Superseded by `track_random_intercepts`.
- **Risk:** Dead slot; harmless but confusing. Keep only the pickle-compat default.
- **Fix:** Drop the empty-dict computation; keep the slot default in `__setstate__`.

### L2 — `segment_lap` has a reserved-but-unimplemented `track_model` kwarg
- **Where:** `corner/states.py:215-218` raises
  `NotImplementedError("track_model integration deferred to Wave 3 U8")`.
- **What:** The live P2.1 masking is in `corner_phase_states` via
  `_attach_cleanliness_masks` *before* calling `segment_lap(df)` (verified — not a
  bug). The lower-level kwarg is dead API surface.
- **Fix:** Remove the unused `track_model` parameter from `segment_lap`, or wire it
  through and delete the raise.

### L3 — Long-G is a hardcoded phase constant, not trained
- **Where:** `physics/score.py:810 _long_g_for_phase`, used at `:874,:1000`.
- **What:** Mid-corner long-G is approximated as 0 (and fixed constants per phase),
  under-allocating rear Fz (documented as a "safe" failure mode in `CLAUDE.md`). The
  `accel_lon_g_*` channels exist in the corpus but scoring doesn't consume the
  surrogate's prediction.
- **Fix:** Use the surrogate's predicted long-G per phase in the axle-margin
  computation; validate the rear-margin shift on held-out laps before shipping.

### L4 — Cold-start TrackModel silently treats all samples as clean
- **Where:** `track/builder.py` cold-start (<3 sessions) returns zero masks; fitter
  filter becomes a no-op (`physics/fitter.py` P2.1 filter). Reported by the corner/
  track mapping pass.
- **Risk:** On thin tracks, curb/off-track telemetry is trained on without warning.
  Documented, low impact.
- **Fix:** Surface "quality masks inactive (cold-start track)" in the briefing NOTES.

### L5 — Fragile sklearn pickle round-trips
- **Where:** `physics/fitters/ridge.py:121-136` (triple round-trip),
  `forest.py:73`, `gp.py:132` — needed for byte-identical re-pickling per spec.
- **Risk:** Brittle across sklearn versions (`scikit-learn>=1.5`, resolved 1.8.0);
  guarded only by the determinism tests.
- **Fix:** Pin a tighter sklearn upper bound, or add a CI canary that fails loudly on
  a non-idempotent pickle rather than relying on the slow suite.

### L6 — Dependency floors are loose (`>=`) with no upper bounds
- **Where:** `pyproject.toml:6-14` — `numpy>=1.26` (resolves 2.x), `scipy`, `polars`,
  `scikit-learn` all unbounded-above; `uv.lock` is committed so installs are
  reproducible, but `uv pip install -e` without the lock would float to new majors.
- **Risk:** Low (lockfile present); a fresh `pip install` could pull a breaking
  sklearn/numpy major. No unmaintained or known-vulnerable deps found.
- **Fix:** Add upper bounds for the numerics stack (numpy/scipy/sklearn) or document
  that installs must use `uv sync`/the lockfile.

---

## Verified safe (checked, no action needed)

- **SQL:** all queries parameterised with `?` (`ingest/catalog.py:120-196,222-238`);
  the only f-string SQL (`catalog.py:98` `ALTER TABLE … ADD COLUMN`) interpolates a
  hardcoded constant from `_ADDITIVE_SESSION_COLUMNS`, not user input. No injection.
- **Secrets:** none in source *(grep for API_KEY/SECRET/TOKEN/PASSWORD — no hits)*.
- **Dangerous calls:** no `eval`/`exec`/`os.system`/`subprocess` in `src/`; the only
  `subprocess.run` is in `tests/test_calibration_gate.py:26-32` (script automation).
- **YAML:** `ingest/parser.py:132` uses `CustomYamlSafeLoader` (safe loader), not
  `yaml.load` with the default loader.
- **Pickle (deserialisation):** the only unpickled data is locally-generated model
  caches under gitignored `corpus/models/`; not loaded from any untrusted/remote
  source. Trust boundary is the local FS. (See M3 for the bypassed type guard.)
- **Tests:** deterministic seeds (`0xC0FFEE`, `tests/physics/conftest.py:70,163`);
  robust git-lfs-pointer detection to skip unmaterialised fixtures
  (`tests/conftest.py:31-60`); no time- or network-dependent tests *(inferred)*.

---

## Suggested first actions (cheap, high-leverage)
1. **M1** — stop committing `err.log` / `recommendations/*.txt` / logs (5 min).
2. **H1** — add a per-PR accuracy smoke so regressions surface (half day).
3. **M3 + M4** — route cache loads through `io.load` with a log line; tighten the
   protect-data hook to the redirect target (1–2 hrs).
4. **H2** — decide Cadillac aero-map remediation vs explicit out-of-domain
   confidence downgrade (scoping needed).
