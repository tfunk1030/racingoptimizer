# ARCHITECTURE.md

Map of the `racingoptimizer` codebase produced during the 2026-06-08 onboarding
audit, **refreshed 2026-06-10** to fold in the "belleisle"/W6 changes
(`785b87b`/`a4e4f5f`: aero-map fit features, garage-step ontology, deletion of
the `day_NN` gate scripts and `docs/physics-rebuild/`). Pairs with `AUDIT.md`
(findings) and `CLAUDE.md` (working memory + the deep domain notes). Read
`VISION.md` for the design intent this implements.

Everything below is grounded in `file:line` references verified by reading the
code. Where a claim is inferred rather than read, it is marked *(inferred)*.

---

## 1. Stack & runtime

| Aspect | Value | Source |
|---|---|---|
| Language | Python `>=3.12` | `pyproject.toml:5`, `.python-version` |
| Package/build | `uv` + `hatchling` (wheel packages `src/racingoptimizer`) | `pyproject.toml:21-27` |
| CLI framework | `click>=8.1` (resolved 8.3.3) | `pyproject.toml`, `uv.lock` |
| Dataframes | `polars>=1.0` (columnar pipeline, the spine of the codebase) | `pyproject.toml` |
| Numerics/ML | `numpy>=1.26`, `scipy>=1.13`, `scikit-learn>=1.5` | `pyproject.toml` |
| IBT parsing | `pyirsdk>=1.3.5` (iRacing telemetry SDK) | `pyproject.toml` |
| Storage | parquet (`pyarrow`, zstd) + SQLite catalog + pickled models | `ingest/writer.py`, `ingest/catalog.py`, `physics/io.py` |
| Tests/lint | `pytest`, `pytest-cov`, `ruff` | `pyproject.toml:16-19` |
| Entry point | `optimize = racingoptimizer.cli:main` | `pyproject.toml:29-30` |

There is **no server, no network I/O, no database server** — it is a local CLI
that reads `.ibt` files and `aero-maps/*.json`, builds a local corpus, and prints
setup recommendations. The only external "service" is the iRacing telemetry
format, parsed offline.

---

## 2. Directory map

```
src/racingoptimizer/
  cli/        Slice F entry — `optimize` group, recommend/calibrate commands
  ingest/     Slice A — .ibt -> parquet + SQLite catalog (the data layer)
  corner/     Slice B — segment a lap into corners then braking/trail/mid/exit/straight phases
  track/      Slice D — compounding per-track model: curbs, bumps, off-track masks, geometry
  aero/       Slice C — load + interpolate aero-maps/*.json (balance %, L/D vs ride height & wing)
  physics/    Slice E — the core: fit surrogate models, score corner-phases, run DE optimisation
    fitters/  forest / gp / ridge sklearn wrappers (FITTERS_LAYOUT_VERSION gate)
  explain/    Slice F rendering — narrative (default), detailed, JSON, setup card, status, compare
  context/    cross-cutting — EnvironmentFrame (12 atmospheric channels)
  confidence/ cross-cutting — Confidence (value, lo, hi, n, regime)
  constraints/cross-cutting — parse constraints.md, clamp values to legal bounds

scripts/      holdout_accuracy_gate.py, lap_time_correlation_gate.py,
              compare_cars_at_track.py (cross-car ranking), validate_ontology_paths.py,
              verify_holdout.sh  (the old day_NN_*.py gates were deleted in a4e4f5f)
tests/        145 files mirroring each slice; per_car smoke tests + slow/calibration marks
docs/         VISION_COMPLIANCE, physics-rebuild plan, accuracy-rebuild plan, slice specs
aero-maps/    parsed aero surfaces, one JSON per (car, wing angle)   [git-lfs / data asset]
ibtfiles/     raw iRacing .ibt telemetry binaries                    [data asset, hook-protected]
corpus/       generated: catalog.sqlite + sessions/*.parquet + models/*.pickle  [gitignored]
```

Cars covered (5 GTP): BMW M Hybrid V8, Porsche 963, Cadillac V-Series.R, Acura
ARX-06, Ferrari 499P. **No unified setup schema across cars** — each has its own
suspension architecture and YAML setup-blob shape (`constraints.md`,
`physics/ontology.py`).

---

## 3. Entry points & command routing

`optimize` is a custom `click.Group` (`_OptimizeGroup`, `cli/__init__.py:32-45`)
that rewrites positional shorthands into the `recommend` subcommand:

- `optimize bmw spa` → first arg in `CANONICAL_CARS` → `["recommend", "bmw", "spa"]`
  (`cli/__init__.py:38-39`).
- `optimize ./x.ibt` → first arg is an existing `.ibt` file → recommend with
  auto-detect (`cli/__init__.py:41-44`).
- Subcommands registered at `cli/__init__.py:56-60`: `learn` (ingest), `compare`,
  `status`, `recommend`, `calibrate`.

`main()` (`cli/__init__.py:48-53`) prints help when no subcommand resolves.

---

## 4. Data flow A — ingestion (`optimize learn ./ibtfiles`)

```
learn_command (ingest/cli.py:45)
  -> ingest.api.learn (ingest/api.py:27)
       resolve_corpus_root -> catalog_path = <corpus>/catalog.sqlite (api.py:55-56)
       _iter_ibt_paths (recursive walk) (api.py:58)
       for each .ibt: _process_one (api.py:257-368)
         Stage 0  read bytes, session_id = sha256(bytes)[:16]   (api.py:295-300)
         Stage 1  parse_ibt -> ParseResult                       (api.py:310, parser.py)
                    channels{name->float32 np.array}, setup{nested YAML},
                    weather_summary, lap_spans, dropped_channels, sample_rate_hz
         Stage 2  detect_car / detect_track_from_filename        (api.py:319-329, detect.py)
                    status ok | partial (unknown car/track/no laps)
         Stage 3  write_session (writer.py:89)
                    df -> sessions/<car>/<track>/<session_id>.parquet (zstd)
                    upsert sessions row + insert laps rows
       _apply_masks_for_session_ids  (Slice D quality-mask pass) (api.py:67)
```

**Storage layout** (`<corpus_root>` = `--corpus-root` > `$RACINGOPTIMIZER_CORPUS`
> `<repo>/corpus`, `ingest/paths.py:23`):

```
corpus/
  catalog.sqlite                                  sessions + laps tables
  sessions/<car>/<track>/<session_id>.parquet     t_s, lap_index, lap_dist_pct,
                                                  data_quality_mask, + all channels
  models/<car>__per-car__<digest>.pickle          fitted PhysicsModel cache (v4)
  models/<car>__<track>__<digest>.pickle          legacy per-(car,track) cache (v3)
```

### Catalog schema (`ingest/catalog.py:17-49`)

- **sessions**: `session_id` PK, `car`, `track`, `recorded_at`, `duration_s`,
  `lap_count`, `weather_summary`(JSON), `setup`(JSON), `source_path`,
  `ingested_at`, `parquet_path`, `status`(ok|partial|failed), `error`,
  `dropped_channels`(JSON), `sample_rate_hz`, `held_out`(0/1).
- **laps**: `session_id` FK, `lap_index`, `lap_time_s`, `start_sample`,
  `end_sample`, `valid`, `best`.
- Indices on `(car, track)` and `laps.session_id`. Schema migrations are additive
  (`_ADDITIVE_SESSION_COLUMNS`, `catalog.py:86-99`).

`recorded_at` is parsed from the **filename** (`parser.py::_filename_recorded_at`),
not the YAML `WeekendOptions.Date` (which is identical across a race weekend) — this
drives the most-recent-setup picker.

---

## 5. Data flow B — recommendation (`optimize bmw spa`)

```
recommend_cmd (cli/recommend.py:199)
  _resolve_car_track_or_exit              (recommend.py:231)  car/track or sniff from .ibt
  _safe_sessions(car)                     (recommend.py:233)  -> catalog query_sessions
  load_constraints(constraints.md)        (recommend.py:250)
  _apply_tyre_pressure_floor_pin          (recommend.py:251)  auto-pin 152 kPa unless --pin
  race-fuel auto-pin block                (recommend.py:268-302) anchor fuel to last session
  _build_per_car_pipeline                 (recommend.py:325)  (PER_CAR_MODEL_CARS = all 5)
     build_corner_schedule(target track)            (TrackModel + corner archetypes)
     _build_or_load_per_car_model                    cache: models/<car>__per-car__<digest>.pickle
        pickle.load OR fit_per_car(...)              (recommend.py:1296-1320)
  _is_thin_corpus_for_recommend           (recommend.py:339)  n_prod<20 OR no axle ceilings -> REFUSE
  model.recommend(...)                    (recommend.py:357)  differential evolution
  _post_clamp(...)                        (recommend.py:387)  clamp to constraints
  model.predict_setup_readouts(...)       (recommend.py:393)  static RH + readouts at new setup
  build_justifications(...)               (recommend.py:428)  per-param corners helped/hurt + sensitivity
  render (narrative default | --detailed | --json) + full_setup_card  (recommend.py:435-512)
  write recommendations/<car>-<track>-<mode>[-NL]-MMDD-HHMM.txt        (recommend.py:514-560)
```

### Model cache key (`cli/recommend.py::_model_cache_parts`, ~1326-1410)

Digest folds: pooled `session_ids` + ontology fingerprint (incl. `json_path`) +
`constraints.md` content hash + `FITTERS_LAYOUT_VERSION` (**12**,
`physics/fitters/__init__.py:72`) + `ENV_FEATURE_SCHEMA_VERSION_PER_CAR` (**8**,
`physics/fitter.py:215`). v12 / schema-8 correspond to W6's aero-map fit
features (`physics/aero_fit_features.py`). Editing `constraints.md` invalidates
**every** per-car cache → ~15-min refit per car.

---

## 6. The physics core (Slice E)

### Fit (`physics/fitter.py::fit_per_car`, starts :1030)

1. Pull each session's setup JSON; collect **corner-phase training frames**
   (`_collect_training_frames`) — one row per `(corner_id, phase)` with ~60 physics
   columns, **quality-filtered** (P2.1: drops rows where `curb_frac_mean>0.5` or
   `off_track_frac_mean>0.0` once the track model is in the compounding regime).
2. Attach setup columns, **corner archetypes** (apex speed, peak lat-G, duration,
   `phase_duration_s`), static-RH readouts, dynamic-at-speed RH, P2.3
   inverse-track-sample-count weights (`1/sqrt(n_track_rows)`), and (W6,
   `physics/aero_fit_features.py`) per-row **aero-map features**
   (`aero_map_ld_ratio`, `aero_map_balance_pct`) queried at the observed
   platform RH + wing + air density, so grip-balance channels don't have to
   learn downforce implicitly from setup alone.
3. Per `(phase, channel)` quadruple, fit a model via `_fit_one_quadruple`
   (:1435). Per-car v4 always uses **ForestFitter** (mixed 35-dim feature space
   defeats the scalar-scale GP). Static/aero/dynamic readout channels route to
   **Ridge**. K-fold CV gives `cv_residual_std`; forest also yields a bootstrap std.
4. Assemble a `PhysicsModel` (`physics/model.py:104-227`): `fitters{(phase,channel)
   ->FitRecord}`, `baseline_setup` (lap-time-weighted median), `axle_grip_ceilings`,
   `bayes_posteriors`, `static_rh_kinematic`, `track_random_intercepts`,
   `per_track_parameter_observed`, etc. Pickled to the cache.

Fitter families: `fitters/forest.py` (RF 50 trees, per-tree std), `fitters/gp.py`
(Matérn→RBF fallback, ARD disabled — L-BFGS hangs on 20+ dims), `fitters/ridge.py`
(α=0.1, deterministic readouts). `FITTERS_LAYOUT_VERSION` gates pickle compat.

### Predict (`physics/model.py`)

`_predict_v4` (model.py:513) keys on `(phase, channel)`, assembles a feature row
[setup params | 12 env channels | corner archetype], adds the P2.2
`track_random_intercepts` correction when a `track=` is supplied, widens CI in
quadrature. `_predict_v3`/`_predict_legacy` retained for rollback. At predict time the W6
aero features are approximated from the deterministic static-RH readouts
(telemetry RH unavailable — `physics/aero_fit_features.py` module docstring).
`__setstate__` (model.py:242) backfills slots, runs `_repair_legacy_slot_shift`
then `_validate_pickle_slots` (type-safety, P1.4).

### Score (`physics/score.py`)

Per corner-phase: `grip / balance / stability / traction / aero_eff / platform`
utilisation (score.py:91-312), aggregated and **phase-time-weighted** (never
lap-time — VISION §6). Default path is **hybrid**: `hybrid_score()` blends the
physics evaluator with the surrogate (mid-corner has the strongest physics signal)
plus additive guardrail penalties (over-axle-ceiling, off-balance, grip
inconsistency). `--surrogate-only` reverts to surrogate + axle-guardrail penalty.
Long-G is a **hardcoded phase constant** (`_long_g_for_phase`, score.py:810), not
trained. Conditions branch dry/damp/wet/full-rain (`wet_mode`) and race/quali
(`quali_mode`) by swapping baselines + phase weights.

### Optimise (`physics/recommend.py`)

`differential_evolution` (maxiter=15, popsize=20, seeded), minimising the negated
score sum. `_pin_or_trust_bounds` (~886-1041) clips each parameter to the **global
corpus envelope** (union of all observed values), pins near-constant params,
narrows the trust radius by confidence regime, and honours `--explore N` / `--reset`.
P0.3 **sensitivity floor**: post-DE, any parameter whose ±1-step move shifts the
objective by < `_SENSITIVITY_FLOOR=0.005` is reverted to baseline and surfaced under
NOTES. Static-RH feasibility is gated by the kinematic fit (R²≥0.98) or the legacy
k-NN repair.

---

## 7. Slices B / C / D / F (supporting)

- **Corner (B)** — `corner/detect.py:detect_corners` (Schmitt-trigger on lateral G),
  `corner/boundaries.py:assign_phases` (forward-only state machine), aggregated by
  `corner/states.py:corner_phase_states` into one row per `(corner_id, phase)` with
  ~60 columns incl. the 12 env channels and (P2.1) `curb_frac_mean`/`off_track_frac_mean`.
- **Aero (C)** — `aero/loader.py` validates & stacks `aero-maps/{car}_wing_{X}.json`
  (front_rh × rear_rh grids of balance % and L/D); `aero/interpolator.py` does
  per-wing `RegularGridInterpolator` + wing-axis linear blend (no density scaling —
  L/D is dimensionless); `aero/residual_correction.py` fits a per-car scalar bias
  with a ±30% fallback guard.
- **Track (D)** — `track/builder.py:build_track_model` builds a compounding per-track
  model (cold-start <3 sessions → zero masks). `track/masks.py` detects curbs
  (shock-velocity p99 > 350 mm/s, cross-session agreement ≥0.6, Acura 0.3) and
  off-track (grip-loss + wheel-speed-spike detectors). `track/geometry.py` derives
  elevation/camber proxies. Persisted as per-session + per-bin zstd parquet.
- **Explain (F)** — `explain/narrative.py` (default, handling-vocabulary 2–3 line
  summaries), `explain/render_text.py` (`--detailed` legacy blocks),
  `explain/render_json.py` (`--json`), `explain/full_setup_card.py` (every garage
  line, tagged `[OPT]/[OPT pin]/[OPT mirror]/[past]/[readout]/[predicted]`),
  `explain/status.py`, `explain/comparison.py`, `explain/justification.py`
  (VISION §7: every click justified with corners helped/hurt + ±1-click sensitivity).

Cross-cutting: `context/environment.py` (12-channel `EnvironmentFrame`),
`confidence/confidence.py` (`Confidence.derive`: <30 samples→sparse, noise ratio
thresholds→noisy/confident/dense), `constraints/` (parse `constraints.md`, `clamp`).

---

## 8. Build / test / lint / CI (what is actually run)

From `.github/workflows/ci.yml` and `pyproject.toml`:

```bash
uv venv && uv pip install -e ".[dev]"      # setup
uv run ruff check src tests                 # lint (CI: lint-and-test)
uv run pytest -q -m "not slow"              # fast suite (CI on every push/PR)
bash scripts/verify_holdout.sh              # held-out hash/flag/leak integrity (CI)
uv run pytest -q                            # full suite, ~15 min (slow tests local)
```

**Weekly cron only** (`calibration-weekly`, `if: schedule`): day-12b evaluator
calibration, `holdout_accuracy_gate.py`, hybrid-vs-surrogate A/B, lap-time Spearman
gate. → **Accuracy is NOT gated on PRs** (see `AUDIT.md` H1).

**CI state as of 2026-06-10 (verified against workflow run 27168788022):**
master is **red** — two stale tests in `tests/cli/test_post_clamp_discrete.py`
fail against the W6 garage-step ontology (`AUDIT.md` N1); the "fast" suite took
**87 minutes** in CI, not the documented ~2 (`AUDIT.md` N3); and because pytest
fails first, the `verify_holdout.sh` step is skipped — once N1 is fixed it will
exit 4 anyway because `docs/physics-rebuild/holdout.sha256` was deleted
(`AUDIT.md` N2). The weekly day-12b calibration step is vacuous: its script was
deleted and the test skips.

---

## 9. Known design constraints (do not "fix" without understanding)

- Corner-phase is the atomic unit, never the lap (VISION §1-2).
- Empirical physics, not textbook formulas; aero maps are the one lookup exception.
- Lap time is an outcome, never the optimisation signal (VISION §6).
- The track model masks dirty samples instead of dropping whole laps.
- Every sample carries environmental context (12 channels) through fit & predict.
- Output must justify every click; confidence is a first-class output.

See `CLAUDE.md` for the exhaustive gotcha list (cache invalidation, per-car
confounding, pin denominator, renderer tag contract, accuracy-rebuild status).
