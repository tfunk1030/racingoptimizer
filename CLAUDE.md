# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Onboarding quick-reference (added 2026-06-08)

New to this repo? Read this first, then `ARCHITECTURE.md` (the Phase-1 map +
request/data-flow walkthrough) and `AUDIT.md` (ranked findings). The rest of this
file is deep domain memory accumulated over the accuracy rebuild — skim it once,
return to it when touching a specific subsystem.

- **What it is:** a local Python `>=3.12` CLI (`optimize`) that turns iRacing
  `.ibt` telemetry into GTP setup recommendations. No server, no network, no DB
  server — just files. Stack: `click` (CLI), `polars` (columnar spine), `numpy`/
  `scipy`/`scikit-learn` (fitting), `pyirsdk` (IBT parse), parquet + SQLite +
  pickled models for storage. Package/build via `uv` + `hatchling`.
- **Entry point:** `optimize = racingoptimizer.cli:main` (`pyproject.toml:29`).
  `optimize <car> <track>` and `optimize <file.ibt>` both route to `recommend`
  (`cli/__init__.py:32-45`). `optimize learn ./ibtfiles` ingests.
- **Where data lives:** `<corpus_root>/catalog.sqlite` (sessions+laps),
  `<corpus_root>/sessions/<car>/<track>/<id>.parquet` (telemetry),
  `<corpus_root>/models/*.pickle` (fitted `PhysicsModel` caches). `corpus/` is
  gitignored; `<corpus_root>` = `--corpus-root` > `$RACINGOPTIMIZER_CORPUS` >
  `<repo>/corpus` (`ingest/paths.py:23`).
- **The core (Slice E, `physics/`):** `fit_per_car` → pickle cache → `_predict_v4`
  → `score.py` per-corner-phase hybrid utilisation → `recommend.py` differential
  evolution. Cache invalidates on `constraints.md` edits, ontology changes,
  `FITTERS_LAYOUT_VERSION` (12), or `ENV_FEATURE_SCHEMA_VERSION_PER_CAR` (8).
- **Commands (verified in CI):** see "Commands" below; CI runs `ruff` + fast
  pytest (`-m "not slow"`) + `verify_holdout.sh` on every PR. Accuracy gates run
  **weekly on cron only** — accuracy is not gated per-PR (`AUDIT.md` H1).
- **Top audit risks (2026-06-10 refresh):** master CI was red (stale tests vs
  the W6 garage-step ontology — fixed on the audit branch, `AUDIT.md` N1) and
  the GitHub **LFS budget is exhausted** (`AUDIT.md` N5 — per-PR CI no longer
  fetches LFS; the weekly full-data job stays blocked until the budget
  returns); recommendation accuracy is **measured-failing**: 34/34 gated
  (car, channel) pairs miss the P1.1 per-channel thresholds (`AUDIT.md`
  "Independent read") and lap-time correlation has never been run (`AUDIT.md`
  H1); all five aero maps stop at a 25 mm front-RH floor the cars run below
  (`AUDIT.md` H2 — out-of-domain queries now warn + downgrade confidence).

## Repository state

`racingoptimizer` is the `optimize` Python CLI for iRacing GTP setup recommendations (VISION.md §8). All six VISION slices (A–F) plus three cross-cutting modules are merged. End-user walkthrough is in `GETTING_STARTED.md`; design spec is `VISION.md` (read it first); per-clause audit is `docs/VISION_COMPLIANCE.md`.

## Known accuracy gap (2026-05-24) — READ BEFORE TRUSTING RECOMMENDATIONS

The post-physics-rebuild work (2026-05-09 → 2026-05-24) wired hybrid scoring, axle ceilings, static-RH co-optimization, garage symmetry, snap-to-step, and a per-track residual correction. Several of those changes broke the "physics-based and calibrated" claim. The current plan is in `docs/accuracy-rebuild-2026-05-24/PLAN.md`.

**Closed in W1-W5 (2026-05-24 → 2026-05-25):** every plan item (P0.1, P0.2, P0.3, P0.4, P1.1, **P1.2**, **P1.3**, P1.4, **P2.1**, **P2.2**, P2.3, P2.4, P3.1, P3.2, P3.3). The legacy k-NN static-RH repair is bypassed when the kinematic fit ships (W2 cleanup). Bold items closed in W5 (2026-05-25); the remainder closed in W1-W4. Definition-of-done validation (held-out gate green on all 5 cars, lap-time Spearman ≥ 0.30 per pair, in-garage RH within 1 mm) requires offline runs on the full corpus.

- **P0.1 — `per_track_residuals` retired.** The slot is preserved on `PhysicsModel` for pickle compat but the predict path no longer reads it and `fit_per_car` no longer computes it. `FITTERS_LAYOUT_VERSION = 10` invalidates every pre-2026-05-24 pickle so they refit clean.
- **P0.2 — Deterministic kinematic static-RH fit ships.** `physics/static_rh_kinematic.py` is a per-car closed-form ridge-regularised OLS gated on R² ≥ 0.98 across the four `setup_static_*_ride_height_mm` channels. Wired into `PhysicsModel.predict_setup_readouts` (kinematic wins for those four channels; surrogate fallback for everything else). Slot validated by `_validate_pickle_slots`. `physics/recommend.py::_kinematic_static_rh_ready` skips the legacy k-NN `enforce_static_rh_feasible` repair when the kinematic fit shipped (would degrade an already-correct prediction); falls back to the legacy k-NN path when kinematic refused (R² < 0.98 or thin corpus).
- **P0.3 — Sensitivity floor enforced.** `physics/recommend.py` probes every moved parameter at ±1 garage step against an order-independent `recommended_snapshot`; moves below `_SENSITIVITY_FLOOR = 0.005` on both sides are reverted to `model.baseline_setup` and surfaced under NOTES via `SetupRecommendation.suppressed_below_sensitivity`. The narrative renderer excludes suppressed params from the "moved" block so they appear only in NOTES.
- **P0.4 — Phantom corner-0 + per-corner dedupe.** `physics/corner_schedule.is_real_corner_archetype` (peak lat-G ≥ 0.40 and max−apex slowdown ≥ 5 m/s) gates both `_axle_guardrail_penalty` and `guardrail_warnings_for_setup`. The latter additionally collapses per-(corner, phase) hits to one summary line per corner.
- **P1.1 — Per-channel held-out gate.** `scripts/holdout_accuracy_gate.py::_PER_CHANNEL_THRESHOLDS` carries the per-channel pass criteria from PLAN §3 P1.1 (peak lat/long G 0.30 / 0.5; understeer 0.10 rad / 0.5; wheel RH 3.0 mm / 0.5; static RH 1.0 mm / 0.2; damper force p99 = 30 % of channel std / 0.5). `main()` returns 1 on ANY car failing ANY non-driver-input channel; the aggregate gate stays informational. JSON output gains `per_channel_pass` + `per_channel_failed` keys per car.
- **P1.4 — Slot type-safety.** `_validate_pickle_slots` invoked from `__setstate__` after `_repair_legacy_slot_shift`. Refuses revives with wrong-typed slots and points the user at `--no-cache`.
- **P2.3 — Inverse-track-sample-count training weights.** `physics/fitter.py::_attach_track_balance_weights` adds a `_track_balance_weight = 1 / sqrt(n_track_rows)` column piped through to the Forest fitter via `sample_weight`. Ridge / GP silently ignore the kwarg (their channels are session-invariant).
- **P2.4 — `phase_duration_s` injected at fit time.** `CORNER_ARCHETYPE_COLUMNS` extended with the per-phase duration so the surrogate can learn within-corner phase asymmetry. `ENV_FEATURE_SCHEMA_VERSION_PER_CAR` bumped 5 → 6.
- **P3.1 — Briefing header carries channel-level error budget.** `explain/narrative.py::_render_error_budget_block` loads `holdout_accuracy_latest.json` and renders one line per `_HEADER_ERROR_BUDGET_CHANNELS` entry. Falls back to the legacy `Confidence: <regime> (median n=N)` line when no row matches.
- **P3.2 — Watch-most picker normalisation.** `_dominant_impact_corner` and `_telemetry_why` divide each impact's `|score_delta|` by the corner's pre-filter spread in the candidate pool, so a long corner with broad-impact loses to a corner with concentrated impact.
- **P3.3 — Thin-corpus refusal banner.** `cli/recommend.py::_is_thin_corpus_for_recommend` (n_prod < 20 OR `axle_grip_ceilings is None`) gates `recommend_cmd`; refusal emits a banner pointing at `optimize calibrate <car> <track>` and returns before DE.
- **P1.2 — Real LOSO orchestration (W5, 2026-05-25).** `scripts/lap_time_correlation_gate.py::_compute_loso_pairs_for_track` walks every qualifying pair, holds each session out, refits per-car on the rest, and pairs the score against the held session's median lap time. Heavy (~2.5 hr per 10-session pair); intended to run offline on a workstation and the JSON committed for CI consumption.
- **P1.3 — CI flip + asymmetric A/B assertion (W5, 2026-05-25).** `tests/physics/test_hybrid_heldout_ab.py` now carries `total_hybrid >= total_surrogate * (1 - 0.20)` per-car -- catches hybrid specifically falling >20% below surrogate on a driver-validated setup (guardrail penalties firing on real-world driving). `.github/workflows/ci.yml::calibration-weekly` invokes the A/B test + lap-time gate on the cron schedule.
- **P2.1 — Curb / off-line row masking before fit (W5, 2026-05-25).** `corner.states.corner_phase_states` gained a `track_model=` kwarg; `_attach_cleanliness_masks` pulls per-sample curb + off-track masks from the TrackModel and the aggregator emits `curb_frac_mean` + `off_track_frac_mean` per (corner, phase). `physics/fitter.py::_collect_training_frames` lazily builds a per-track TrackModel from the catalog and filters rows where `curb_frac_mean > 0.5` (the plan's "median sample on a curb" cutoff) or `off_track_frac_mean > 0.0` before training. Cold-start TrackModels return zero masks; the filter is a no-op until the corpus reaches the compounding regime. `ENV_FEATURE_SCHEMA_VERSION_PER_CAR` 6 → 7.
- **P2.2 — Per-track random intercepts (W5, 2026-05-25).** `physics/track_random_intercepts.py` ports the conjugate-Gaussian random-intercept math from `bayes_retrofit.py` and centres the prior on zero (residuals already have the surrogate's mean removed). `fit_per_car` populates `PhysicsModel.track_random_intercepts` with `(channel, track) -> TrackIntercept` posteriors (empirical-Bayes shrinkage; pruned to ≥10 residuals per channel). `_predict_v4` accepts a new `track=` kwarg (threaded by the score path) and adds the intercept to mu; CI widens in quadrature by `intercept_std`. The setup gradient is preserved because `alpha_t` does not depend on setup. `FITTERS_LAYOUT_VERSION` 10 → 11.

Full plan: `docs/accuracy-rebuild-2026-05-24/PLAN.md`. Definition-of-done items §5.1 / §5.3 / §5.6 (empirical validation on populated corpus + in-garage RH check) require offline runs and are not gated by these code changes. **As of 2026-06-10 none of the DoD evidence is committed** — see `AUDIT.md` "Accuracy & correlation state".

## W6 "belleisle" + post-audit fixes (2026-05-26 → 2026-06-08)

Changes that landed after the W5 wave and around the 2026-06-08 onboarding audit
(commits `785b87b`, `a4e4f5f`, `ccc0dee`, `1c30b33`, `f16d0a8`, `1f64966`):

- **W6 aero-map fit features** (`physics/aero_fit_features.py`): each training
  row gains `aero_map_ld_ratio` + `aero_map_balance_pct` queried at observed
  platform RH + wing + air density; at predict time approximated from the
  deterministic static-RH readouts. `ENV_FEATURE_SCHEMA_VERSION_PER_CAR` 7 → **8**,
  `FITTERS_LAYOUT_VERSION` 11 → **12**.
- **Garage steps for previously step-less params**: `brake_bias_pct` `step=0.5`,
  `diff_preload_nm` `step=5.0` (`physics/ontology.py:326-348`); `_post_clamp`
  snaps to them. Two stale tests in `tests/cli/test_post_clamp_discrete.py:110-120`
  still assert step-less behaviour → **CI red on master** (`AUDIT.md` N1).
- **Deletions**: the `scripts/day_NN_*.py` gate scripts and the whole
  `docs/physics-rebuild/` tree (incl. `holdout.sha256` and
  `holdout_accuracy_latest.json`) were deleted, severing the held-out
  integrity check (`AUDIT.md` N2; manifest + results JSON restored 2026-06-10
  — see "Held-out IBT system" below). `recommendations/*.txt` and `err.log`
  were untracked (good). `day_12b_calibrate_evaluator.py` is still gone, so
  the weekly calibration cron step skips vacuously.
- **VISION §6 restored** (`ccc0dee`): `_track_fastest_observed_value` and the
  `track_best_value` pin branch were removed from `physics/recommend.py` — lap
  time no longer selects setup values (it remains, legitimately, the corner
  time-sensitivity weight at fit time).
- **AUDIT M3 fixed** (`1c30b33`): both model-cache loads route through
  `physics.io.load` (type guard) with a stderr "refitting" note
  (`cli/recommend.py:1296-1312`, `:1358-1374`).
- **Ontology integrity** (`f16d0a8`): brake-duct / throttle-map params set
  `fittable=False, user_settable=False` (no CarSetup YAML leaves exist).
- **Cross-car ranking** (`1f64966`): `scripts/compare_cars_at_track.py` ranks
  the 5 cars at a track by `0.6*LapScore + 0.4*PhysScore` (reads recommend
  `--json` `score_total` + catalog best laps; does not touch the DE objective).
  Runbook: `docs/watkins-glen-runbook.md`. Watkins Glen has exactly **one IBT
  per car** → cold-start TrackModel, no curb/off-track masking there yet.

## Commands

```bash
uv venv && uv pip install -e ".[dev]"

uv run optimize learn ./ibtfiles            # ingest a directory of .ibt files
uv run optimize learn ./ibtfiles --reparse  # force re-parse already-ok sessions
uv run optimize bmw sebring                 # recommend by (car, track) — race default
uv run optimize bmw spa --quali --fuel 8    # quali stint, 8 L pinned
uv run optimize bmw spa --explore 10        # widen empirical envelope by 10% per side
uv run optimize bmw spa --reset             # full-envelope search; "fundamentally different" setup
uv run optimize bmw spa --detailed          # legacy per-param block format (vs default narrative)
uv run optimize ./my_session.ibt            # recommend by IBT path (auto-detect)
uv run optimize compare a.ibt b.ibt         # diff two setups per (corner, phase)
uv run optimize status bmw                  # coverage + fit-quality trend
uv run optimize calibrate bmw spa           # active-learning probes for thin-variance params
uv run optimize calibrate bmw spa --status  # just the per-parameter coverage table
uv run optimize bmw spa --physics           # Day 14: prepend physics-view banner (per-car evaluator weights, geometry, tyre floor)

uv run pytest -q                            # full suite (~15 min)
uv run pytest -q -m "not slow"              # fast suite (~2 min locally; ~87 min in CI with LFS ibtfiles — AUDIT.md N3)
uv run ruff check src tests                 # lint (must stay clean)
```

## Tooling quirks

- Click 8.3+: `CliRunner(mix_stderr=False)` is gone — stderr always separated, access via `result.stderr`. Don't use `%%` in option help text either; Click no longer printf-formats it.
- `from racingoptimizer.physics import recommend` returns the FUNCTION (re-exported via `physics/__init__.py`), not the module. For `inspect.getsource` / module attrs, use `import racingoptimizer.physics.recommend as rec_mod` or `importlib.import_module("racingoptimizer.physics.recommend")`.
- Multi-line Python on Windows: use `uv run python << 'PY' ... PY` heredoc; `python -c "..."` truncates at first newline. `grep -P` (Perl regex) is also unavailable on the bundled grep — use the `Grep` tool's regex mode or a small Python snippet.
- Background `optimize` runs buffer Python stdout until exit; the output file stays at 0 bytes the entire run. Don't poll for progress — wait for the completion notification.

## Active build

VISION.md is decomposed into six slices plus three cross-cutting modules. Status reflects what is **merged** AND what has been **verified across all 5 GTP cars** (BMW M Hybrid V8, Porsche 963, Cadillac V-Series.R, Acura ARX-06, Ferrari 499P) versus only single-car (BMW Sebring fixture) smoke. Per VISION.md "do not assume a unified setup schema across cars" — the five cars have different suspension architectures, IBT YAML setup-blob shapes, and aero-map step sizes. **A green BMW test is not a "works" claim.**

**Two recommend code paths.** `optimize <car> <track>` routes by `PER_CAR_MODEL_CARS` (top of `src/racingoptimizer/cli/recommend.py`):

- **v4 (per-car, track-agnostic)** — all five GTP cars (`PER_CAR_MODEL_CARS` in `cli/recommend.py`). `fit_per_car()` pools every session for the car across every track into one fitter; cache file `corpus/models/<car>__per-car__<digest>.pickle`. For tracks the car has never been driven on, the CLI's cross-car schedule fallback (`_maybe_borrow_cross_car_track`) borrows corner geometry from any other car's sessions on that track — so Ferrari@Spa works even though Ferrari has no Spa IBTs.
- **v3 (per-(car, track))** — legacy branch retained for rollback; not the default production path since 2026-05-23.

| Slice | Module | Code merged | Per-car verification scope |
|---|---|---|---|
| **A — IBT ingestion** | `racingoptimizer.ingest` | ✅ | Detect/normalize: ✓ all 5 (`tests/test_detect.py`). Per-car parser end-to-end: ✓ all 5 (`tests/test_parser_per_car.py`). |
| **B — Corner-phase decomposition** | `racingoptimizer.corner` | ✅ | ✓ all 5 (`tests/corner/test_per_car_smoke.py` loops the canonical car fixtures and asserts ≥1 corner detected). VISION §2 damper-velocities-vs-forces fully wired: `damper_force_p99_n` / `damper_force_mean_n` columns derived per-car via `physics.damper_force.estimate_damper_force_n` (digressive curve, inlined as a Polars expression to keep the pipeline columnar). |
| **C — Aero-map loader & interpolator** | `racingoptimizer.aero` | ✅ | ✓ all 5 (`tests/aero/test_loader.py::test_load_real_corpus_per_car` and `tests/aero/test_smoke.py::test_load_aero_maps_per_car_smoke`). |
| **D — Track model** | `racingoptimizer.track` | ✅ | ✓ all 5 (`tests/track/test_per_car_real_ibt.py`). Per-car curb-agreement threshold lives in `_PER_CAR_CURB_AGREEMENT_FRACTION` (Acura uses 0.3 instead of the 0.6 four-corner default — its heave/roll-shock signal aggregates more symmetrically and produces lower cross-session agreement). Per-IBT sample rate is threaded through `TrackModel.sample_rate_hz` and `_lap_length_from_speed_fallback` so high-rate (e.g. 360 Hz) recordings don't 6× the lap length. |
| **E — Physics fitter** | `racingoptimizer.physics` | ✅ — both halves merged (U9 train + U10 score/recommend). | ✓ all 5 (`tests/physics/test_per_car_fit_predict.py`). Acura gracefully degrades when shock-deflection channels are missing — flagged in `untrained_parameters`, fit still runs. |
| **F — CLI / recommendation rendering** | `racingoptimizer.cli`, `racingoptimizer.explain` | ✅ — `optimize <car> <track>`, `optimize <ibt_path>` (auto-detect), `optimize compare`, `optimize status` (U11). `optimize learn` (slice A) preserved. | ✓ all 5 (`tests/cli/test_per_car_smoke.py` loops the canonical car fixtures, runs `optimize <car> <track>` text + JSON, and asserts an exit-0 briefing with confidence + parameter blocks for every car). |

**Verification convention:** before claiming a slice "works" or marking it ✅, the slice must have a `tests/<slice>/test_per_car_smoke.py` (or equivalent) that loops the five canonical car fixtures (skipping when missing) and asserts the slice's contract holds. Single-car smoke tests are gaps to fill before the slice is called done.

**Repo layout:**

- `src/racingoptimizer/<slice>/` — implementation (`ingest`, `corner`, `aero`, `track`, `physics`, `cli`, `explain`, plus cross-cutting `context`, `confidence`, `constraints`).
- `tests/<slice>/` — pytest suites mirroring each module; per-car smoke tests live as `test_per_car_*.py`.
- `docs/superpowers/specs/` — the design spec for each slice (read before touching the code).
- `docs/VISION_COMPLIANCE.md` — per-clause file:line audit of how the code satisfies VISION.
- `GETTING_STARTED.md` / `README.md` — end-user docs (don't duplicate them here).

Cross-cutting modules (master-plan §2) — all merged:

- `racingoptimizer.context.EnvironmentFrame` — per-corner-phase atmospheric snapshot (12 channels: `AirTemp`, `AirDensity`, `AirPressure`, `RelativeHumidity`, `WindVel`, `WindDir`, `FogLevel`, `TrackTempCrew`, `TrackWetness`, `WeatherDeclaredWet`, `PrecipType`, `Skies`).
- `racingoptimizer.confidence.Confidence` — frozen `(value, lo, hi, n_samples, regime)` with `Confidence.derive(...)` regime-derivation classmethod (sparse short-circuits noisy at `n_samples < 30`). `Confidence.with_local_density(recommended, observed, step)` downgrades regime by one tier when value is more than 3*step from nearest observed (Mode 4 closure, 2026-05-08).
- `racingoptimizer.constraints.{ConstraintsTable, load_constraints, clamp}` — markdown parser for `constraints.md` plus per-car-shadowing `clamp(value, parameter, car)`.

## Physics-rebuild modules (2026-05-08, Days 8-13 of the 14-day plan)

Eight new modules across `physics/` and `aero/` shipped via the
physics-rebuild plan (the plan docs `docs/physics-rebuild/PLAN.md` and
`COMPLETE.md` were deleted in `a4e4f5f`; recover via
`git show 94ce009:docs/physics-rebuild/PLAN.md` if needed). Production wiring
varies by module:

- **`physics.diagnostic_state`** — body slip β = atan2(Vy, Vx),
  per-axle kinematic slip angles (bicycle model with per-car wheelbase
  + steering ratio), chassis-level axle force decomposition. Per-car
  geometry registry `_CAR_GEOMETRY` for all 5 GTP cars. *Used by*:
  `physics.axle_grip`, future Mode 5 features. *Not yet wired into
  recommend pipeline.*
- **`physics.axle_grip`** — per-(car, axle) grip-margin model. ONE
  parameter per axle (`AxleGripCeiling.mu_peak`), NOT Pacejka. Per
  Reviewer Agent 1's veto: tire-model fit from telemetry alone is
  circular without measured forces. `mu_peak` is an *axle utilization
  ratio* (Fy/Fz), not tire mu — observed values 2.5-3.0 are normal
  because chassis-level Fz in the denominator excludes aero downforce.
  *Used by*: `physics.evaluator`, `physics.hybrid_optimizer`.
- **`physics.bayes_retrofit`** — closed-form empirical-Bayes hierarchical
  retrofit (NOT MCMC; conjugate-Gaussian one-way random-effects model
  has identical math). Per-(parameter, track) `BayesPosterior` with
  `mean_std` (uncertainty in central tendency, used by recommender
  trust-radius) + `predictive_std` (uncertainty in next observation,
  used by held-out coverage tests). Wired into `fit_per_car` →
  `PhysicsModel.bayes_posteriors`. **`FITTERS_LAYOUT_VERSION = 5`**
  (was 4 pre-hybrid-wire-in, was 3 pre-guardrail-wire-in, was 2
  pre-Day-4) so existing pickles invalidate. v4 added
  `PhysicsModel.axle_grip_ceilings`; v5 corresponds to the
  hybrid-blend + guardrail wiring landed on 2026-05-23.
- **`physics.evaluator`** — per-corner-phase composite physics score:
  `axle_utilization * w + aero_balance * w + grip_headroom * w`.
  **Per-car CALIBRATED weights** (Day 12b): BMW (0.2, 0.8, 0.0),
  Cadillac (0.2, 0.3, 0.5), Ferrari (0.0, 0.0, 1.0), Porsche
  (0.0, 0.5, 0.5). Acura falls back to default (0.5, 0.3, 0.2)
  because the corpus only has ~36 corner-phase rows (axle-ceiling
  fit needs >=100); refresh once more sessions land.
  `get_weights_for_car(car)`.
  **PRIMARY VALUE IS GUARDRAILS**, not lap-time prediction
  (Spearman within-group only 0.12-0.25 on this corpus).
  `guardrail_check(score, front_margin, rear_margin) → GuardrailReport`
  flags axle_util > 1.0, severe imbalance, surrogate divergence.
- **`physics.hybrid_optimizer`** — phase-aware physics+surrogate
  combination: `score = w_phase * physics + (1-w_phase) * surrogate`.
  Per-phase weights from Day 13 investigation:
  `mid_corner=0.40, braking=0.10, exit=0.10, trail_brake=0.05,
  straight=0.05`. **Mid_corner has 3-30x stronger physics signal**
  (steady-state cornering vs driver-input-dominated phases).
  **Hybrid scoring WIRED INTO DE** (2026-05-23): default recommend path uses
  `physics/score.py::_corner_phase_objective_value` → `hybrid_score()` —
  phase-aware blend of physics evaluator + surrogate, plus additive guardrail
  penalties (`over_axle_ceiling = -0.15`, `severely_off_balance = -0.075`,
  `grip_inconsistency = -0.25` quarter penalty). Pass `--surrogate-only` to
  revert to surrogate + `_axle_guardrail_penalty` only (legacy mid-corner
  axle-ceiling check in `physics/recommend.py`). Long-G is approximated as 0
  in mid-corner (steady-state) since `accel_long_g_*` isn't in
  `TARGET_OUTPUT_CHANNELS`; the approximation under-allocates rear Fz, so the
  rear margin is slightly inflated (safe failure mode). `None` axle ceilings
  (legacy pickles, or cars with insufficient mid-corner samples) → hybrid falls
  back to surrogate-only scoring for guardrail terms.
- **`physics.damper_force.DamperCurve`** + `fit_damper_curve_from_corpus(car)`
  — per-car (k_low_speed, knee_mm_s) refit from corpus shock-velocity
  p30/p95 percentiles (`*shockVel` is in m/s; multiply by 1000).
  Backward-compat: `estimate_damper_force_n(velocity, *, car=None,
  curve=None)` — `curve` overrides seeded values when provided.
- **`aero.residual_correction`** — per-car scalar correction on
  aero-map peak-lat-G prediction. **Authorized fallback triggered
  on all 3 v4 cars** on this corpus; ships as infrastructure for
  future refinement.
- **CLI `--physics` flag** (`cli/recommend.py::_render_physics_banner`):
  prepends a "PHYSICS VIEW" banner to the briefing. Recommendation
  values unchanged — banner is informational. Surfaces per-car
  evaluator weights, geometry, tyre floor pin status.

## Held-out IBT system (2026-05-08, Day 0 prep)

**Repaired 2026-06-10 (was broken since `a4e4f5f`):** the manifest
`docs/physics-rebuild/holdout.sha256` (restored from `94ce009`) and the gate
results `holdout_accuracy_latest.json` (restored from `ff357c8`, the newest
revision with populated channels) are back in-tree; the briefing error-budget
header renders again. `verify_holdout.sh` now exits **5** with a clear message
on unmaterialised-LFS-pointer checkouts (pointer ≠ tampering); a real
integrity pass still needs `git lfs pull` (blocked on the LFS budget,
`AUDIT.md` N5).

5 hash-pinned IBTs in `docs/physics-rebuild/holdout.sha256` are
flagged `held_out=1` in the catalog and excluded from production
queries by default. Used by gate scripts (which opt in via
`include_held_out=True`):
- H1 BMW Spa (3f0a05d3f44527bd)
- H2 Cadillac Laguna (d236a089300fc0ea)
- H3 Ferrari Hockenheim (fc96805e3b1a27cc)
- H4 Acura Daytona, banked (72f43fa4527c4260)
- H5 Porsche Algarve (a3d43056a952ff99)

`scripts/verify_holdout.sh` runs three checks: hash match, catalog
flag, no pickle leak. Run before any work that depends on held-out
isolation.

`scripts/holdout_accuracy_gate.py` is the per-car generalisation gate:
refits each car from production-only sessions, predicts every
corner-phase channel of the held-out IBT at the OBSERVED setup, and
reports per-channel `mean_abs`, `normed_residual` (residual / channel
signal std), and `coverage` (fraction of actuals inside the predicted
CI). Per-car pass criteria: median coverage >= 0.50, dense-regime
mean coverage >= 0.85, median normed residual <= 2.0. Output JSON at
`docs/physics-rebuild/holdout_accuracy_latest.json` (directory currently
deleted — the write will fail until the path is restored or relocated;
`explain/narrative.py:52` reads the same JSON for the briefing error budget).

`cat.set_held_out_sessions(conn, ids, held_out=True)` is the
helper to mark sessions gate-only; `upsert_session`'s ON CONFLICT
clause omits `held_out` so re-ingesting an IBT cannot silently
flip a gate-only row back to production.

## Tyre pressure floor pin (2026-05-08, Day 1)

`cli/recommend.py::_apply_tyre_pressure_floor_pin` defaults
`tyre_cold_pressure_kpa` to the per-car constraint floor (152 kPa
for all 5 cars per `constraints.md`) unless the user passes
`--pin tyre_cold_pressure_kpa=`. Override an explicit value
(e.g. `--pin tyre_cold_pressure_kpa=180` for tyre warming) and
the helper is a no-op. Closes Mode 2 (surrogate cannot see
peak-grip-vs-Fz curvature, drifts to 154-163 kPa otherwise).

Info message printed on stderr when the auto-pin fires:
`Tyre cold pressure auto-pinned to constraint floor: 152.0 kPa
(override with --pin tyre_cold_pressure_kpa=N).`

**Static ride height envelope warnings** (`cli/recommend.py::
_static_ride_height_envelope_warnings`): after recommend, compares
`predict_setup_readouts()` static RH against the observation envelopes
in `constraints.md` (30–80 mm). Warn-only — user validates in iRacing
garage; does not clamp perch/pushrod/spring inputs.

## Stint mode + conditions branching (`physics/wet_mode.py`, `physics/quali_mode.py`)

Two orthogonal axes feed `physics.score._conditions_adjusted_baselines`:

- **Dry vs wet** (VISION §10): `physics.wet_mode.classify_conditions(env)` returns `dry / damp / wet / full_rain` from `track_wetness` + `weather_declared_wet` + `precip_type`. Non-dry regimes swap baselines (lower max grip + aero baseline, higher wheelspin tolerance) **and** phase weights (less aero_eff, more platform + grip).
- **Race vs quali** (VISION §4 / §5): `--quali` swaps to `physics.quali_mode.quali_phase_weights` — `grip` x1.15, `aero_eff` x1.20, `platform` x0.55 (re-normalised so each phase still sums to 1.0). Quali takes precedence over the wet phase-weight pick (a wet quali still wants outright pace on a wet-adjusted baseline). Quali requires `--fuel N` (no auto fuel; quali fuel is per-track).

**Race-mode fuel auto-pin** (`cli/recommend.py::recommend_cmd` race-fuel block, search `Race fuel auto-pinned`): without `--quali` AND without `--fuel`, the CLI anchors `fuel_level_l` to the most-recent past-session value for the target track (typically ~58 L on the BMW M Hybrid V8). Substring-matches the user-typed track to the catalog slug (so `optimize bmw spa` finds `spa_2024_up`); falls back to all-car sessions only when the target track has none. Without this, the optimizer would treat fuel as a freely-fittable input and minimize mass for one-lap pace -- recommending fuel loads that won't cover a race.

Wind enters `physics.score.aero_eff` via `physics.wind.aero_wind_modifier` as a *magnitude* downforce penalty (treats `wind_vel_ms` as a tailwind worst case). Directional decomposition is deferred per `physics/wind.py` docstring — needs per-corner heading data the corner schedule doesn't carry yet.

## Trust radius + `--explore N` + `--reset` (`physics/recommend.py::_pin_or_trust_bounds`)

Per-car DE search is clipped to the **global corpus envelope** -- the union of every value the user has ever run this car at, on any track. Per VISION §3 the search will not extrapolate outside what the surrogate has been trained near, but per-target-track strictness was relaxed in `318d91d` because it left good setups off the table at every track the driver hadn't fully swept. The model is trained on the global corpus too, so the trust envelope matches training density.

`--explore N` widens the envelope by N% of each parameter's constraint span on each side, clipped to legal bounds. `0` (default) is strict empirical; 5-10 is modest exploration; 20-30 is aggressive. Recommendations landing in the widened territory carry weaker confidence by design.

`--reset` opens the search to `[corpus_min - 30%, corpus_max + 30%]` of constraint span on each side, skips the corpus-density pin check (so observed-constants can move), downgrades every parameter's confidence regime to `noisy`, and prints a `RESET MODE` banner to stderr. Use when the current setup feels fundamentally broken and small +/-1-click tweaks aren't moving the car. The optimizer is intentionally extrapolating beyond corpus density; verify on track before pushing.

Pin denominator uses the **empirical training range** (max−min observed across all pooled sessions), not the constraint span. Wide legal envelopes per BMWBounds.md (e.g. heave 0..900 N/mm) would otherwise mask real corpus variation as "near constant" and pin everything (regression test in `tests/physics/test_pin_near_constant.py`). Defensive guard: `target_observed` is clipped to the constraint envelope BEFORE the empirical-window math, so a user constraint pin (e.g. `--fuel 8` collapses to `(8, 8)`) outside the in-corpus values doesn't produce an inverted bound that crashes DE.

DE budget: `maxiter=15, popsize=20` (bumped from 5/10 on 2026-05-06). ~15 min per run on the 47-param BMW per-car search.

## Per-car model cache key (`cli/recommend.py::_model_cache_parts`)

Cache files at `corpus/models/<car>__per-car__<digest>.pickle` (or `<car>__<track>__<digest>.pickle` for v3). Digest folds:

1. `session_ids` — adding a session = new training data
2. `ontology` fingerprint — `(name, family, fittable, user_settable, json_path)` per spec. **The `json_path` is critical** — without it, a leaf-path correction (e.g. moving `fuel_level_l` from `Chassis.Fuel` to `BrakesDriveUnit.Fuel`) silently reuses the OLD pickle that trained against the wrong YAML field, masking the fix.
3. `constraints.md` content hash — bounds are baked into the pickle at fit time; editing them must invalidate the cache so DE doesn't search against stale bounds.
4. `FITTERS_LAYOUT_VERSION` (in `physics.fitters.__init__`, **currently 12** as of 2026-05-26) — bump when class names / module paths under `physics.fitters` change so old pickles don't fail to revive (`ModuleNotFoundError`), OR when PhysicsModel gains a new field that production paths read (forces refit so the field populates rather than default-empty). v11 (W5) adds `track_random_intercepts`; v12 (W6, `785b87b`) corresponds to the aero-map fit features.
5. `ENV_FEATURE_SCHEMA_VERSION[_PER_CAR]` — pre-S2.2 (v1), S2.2 env-12 (v2), Stage-3 coupled (v3), per-car (v4); bumped to 6 on 2026-05-24 with P2.4's `phase_duration_s` injection; to 7 on 2026-05-25 (W5) with P2.1's curb/off-track row masking; to **8 on 2026-05-26 (W6)** with `physics/aero_fit_features.py` adding `aero_map_ld_ratio`/`aero_map_balance_pct` to the joint feature vector.

Editing `constraints.md` invalidates EVERY per-car cache (constraint content hash is a cache-key ingredient). Next recommend per car triggers a ~15-min refit. Plan constraint edits in batches.

## Confidence is track-wide, NOT parameter-specific

The `dense` / `noisy` / `sparse` label in the briefing header reflects overall corpus density at the target track, not specific evidence at the recommended value. A polluted corpus (e.g. GT3 sessions misrouted to BMW pre-1a8c9a3) will report "dense" while recommending values that have zero physics anchor. When a recommendation feels off, cross-check `model.per_track_parameter_observed[track][param]` for actual driven values before trusting it.

## Per-car v4 cross-track confounding

The joint surrogate pools every track for the car. If a parameter was held CONSTANT within a high-sample-count track (e.g. wing=17 across 24 Hockenheim Ferrari sessions) and varied only at a low-sample-count track (wing=14-15 across 6 Spa sessions), the regressor is dragged by sample weight regardless of corner-archetype features. Recommendations at the under-sampled track silently inherit the over-sampled track's setup philosophy. Mitigation: check whether the parameter has WITHIN-TRACK variance at the target before trusting recommendations; if not, the surrogate has no signal to learn track-specific interactions.

## IBT picker fields (`ingest/parser.py::_filename_recorded_at`)

`recorded_at` is parsed from the IBT filename (`<car>_<track> YYYY-MM-DD HH-MM-SS.ibt`), with the YAML's `WeekendInfo.WeekendOptions.Date` as fallback. The YAML field is iRacing's SCHEDULED race date for series events — identical across an entire weekend — so using it as a per-session timestamp made the most-recent-setup picker pick arbitrary winners. Filename-derived datetimes are unique per session.

Already-ingested sessions short-circuit on `status="ok"`. Use `optimize learn --reparse` to force re-processing after a parser change to refresh stale `recorded_at` / setup fields without manual catalog surgery.

The `_most_recent_setup_for(sessions_df)` picker sorts by `recorded_at desc`, with `ingested_at desc` as tiebreaker. Picker also drives the `(was X)` deltas in the renderer — wrong picker = misleading deltas.

## Pin denominator (`physics/recommend.py::_pin_or_trust_bounds`)

Decides whether to pin a parameter to its observed median or let DE search a trust-radius around it. The denominator for `observed_std / x < _NEAR_CONSTANT_FRACTION` is the **empirical training range** (max−min observed across all pooled sessions), not the constraint span. Wide legal envelopes per BMWBounds.md (e.g. heave 0..900 N/mm) would otherwise mask real corpus variation as "near constant" and pin everything. Falls back to the constraint span only when no per-track observed values exist (truly-constant params still pin correctly).

Defensive guard: `target_observed` is clipped to the constraint envelope BEFORE the empirical-window math runs, so a user constraint pin (e.g. `--fuel 8` collapses to `(8, 8)`) outside the in-corpus values (corpus only has 58 L) doesn't produce an inverted `(57, 8)` bound that crashes DE's seed_population.

## Setup-card renderer contract (`explain/full_setup_card.py`)

Mirror precedence: `_MIRRORED_LEAVES` is checked BEFORE `opt_match` in `_render_panel` (commit 06f0c23). Adding a mirror entry overrides any per-corner DE recommendation for that path. Required so right-side dampers (which have their own per-corner ParameterSpec) can mirror from the left rather than render an independent DE result.

Watch-most + Why-line corner picker filter impacts by `_FAMILY_PREFERRED_PHASES` before max-by-score-delta (commits 7be3017, 5f305b4). Without this, the corner-duration weighting from e90e8fd makes the longest corner dominate every parameter's "Watch most" line.

Every garage line carries one tag. The set is closed:

| Tag | Meaning |
|---|---|
| `[OPT]` | Optimizer recommendation, post-clamp, rounded to iRacing UI step. |
| `[OPT pin]` | Optimizer pinned to observed median (no per-session variance to learn from). |
| `[OPT mirror]` | Per-axle parameter mirrored onto the symmetric corner. iRacing UI requires LR=RR for: rear coil spring rate, rear spring perch offset, front + rear torsion bar turns/OD, rear toe-in, and all right-side damper clicks (5 modes × 2 axles). |
| `[past]` | Most-recent session value; no constraint bounds yet to optimize against. |
| `[readout]` | iRacing-calculated, past-session value. Driver cannot type these. |
| `[predicted]` | Same readout but evaluated by `PhysicsModel.predict_setup_readouts()` at the new setup vector — what iRacing will display after the user enters the `[OPT]` values. |

Static ride heights, corner weights, deflections, and the `AeroCalculator` block are all readouts; the `[predicted]` path covers `setup_static_*_ride_height_mm` and falls back to `[readout]` for the rest.

Per-axle mirroring lives in `_MIRRORED_LEAVES` (6 entries today: rear coil rate, rear spring perch offset, front + rear torsion bar turns/OD, rear toe-in); predicted-readout path mapping in `_PREDICTED_READOUT_PATHS`. Add to either dict to extend coverage. The renderer also snaps `[OPT]` numeric values to either the parameter's uniform `step` OR the non-uniform `discrete_values` list (torsion bar OD's 14 diameters; clutch plates {2, 4, 6}).

## Plain-English narrative (`explain/narrative.py`)

DEFAULT briefing renderer since 2026-05-06. Replaces the legacy per-parameter `Helps:`/`Hurts:`/score-delta blocks with a 2–3 line summary per change in handling vocabulary (pitch / roll / understeer / oversteer / aero stall / bottoming / turn-in / throttle traction). Per change:

```
Front heave spring: 60 -> 50 N/mm  (softens)
  Effect: More front compliance over kerbs; better front grip
    retention through bumpy entries; smoother brake-release.
  Trade:  More pitch dive under heavy braking; slower turn-in;
    aero platform less stable on Kemmel compression.
  Watch most: T9 mid-corner
  Sensitivity: +1 click -0.012 score; -1 click +0.008 score
```

OVERALL DIRECTION at the top aggregates moves by THEME (perches + pushrods both → "ride height"; spring_rate + heave_spring → "platform stiffness"), with `(mixed direction)` annotation when a theme has both up and down moves. A **NOTES** block at the end separates pinned parameters, untrained (no bounds), and blocked readouts.

`--detailed` reverts to the legacy block format for engineering drill-down + the `setup-justifier` validator agent. Translation tables: `_DIRECTION_VERB` (per-row terse verb), `_OVERALL_FRAGMENT` (fluent OVERALL phrase per theme), `_CAR_FEEL` (Effect + Trade per `(family, mode, axis, direction)`). New parameter families need entries in all three (and a label in `_PARAM_LABEL`) to render the full vocabulary; missing families fall through to a phase-themed `Helps:`/`Watch:` corner-list line.

Output is ASCII-only (Windows cp1252 console can't encode Unicode arrows / em-dashes). Use `->`, `--`, `!`, `+/-`, etc.

Specs live under `docs/superpowers/specs/`; plans under `docs/superpowers/plans/`. Read the active slice's spec before touching its code.

`constraints.md` covers most legal-range parameters:
- Wing, tyre pressure, heave spring/slider, static ride heights (shared defaults)
- BMW per-car overrides per BMWBounds.md (heave 0–900, third 0–900, coil 105–280, ±100 mm rear perches)
- Cadillac, Ferrari, Acura, Porsche per-car overrides similarly grounded in their respective bounds files
- Fuel level (1–100 L, fittable, with race auto-pin in CLI)
- ARB Size front + rear (categorical, BMW = Disconnect/Soft/Medium/Stiff, Ferrari = Disconnected/A..E)
- Diff coast/drive ramps + clutch friction plates (categorical / discrete numeric)
- Front diff preload (Ferrari only; -50..+50 Nm)
- Toe-in, mm-based (front axle scalar, rear per-corner with LR=RR mirror)
- Per-corner damper bounds with per-car-override fan-out (Ferrari overrides global 0–11 to 0–40)
- Per-corner torsion bar turns + OD (BMW/Cadillac front; Ferrari all 4)

Brake duct openings now have provisional ontology entries (2026 integration, paths to be verified on a duct-swept session). Still `<TODO: from iRacing UI>` for some cars: exact json_path tuning + full per-car coverage. Corner weights are
**calculated readouts** (`fittable=False`, `user_settable=False` in ontology;
`constraints.md` carries observation envelopes for validation only). Slice E's
`fit` gracefully degrades — lists CE-gated unbounded parameters in
`untrained_parameters` and does not refuse to run.

Categorical params (ARB size, diff coast/drive ramps) are encoded as ordinal indices via `ParameterSpec.choices`; the renderer maps the rounded index back to the label. Non-uniform numeric discrete sets (torsion bar OD's 14 diameters on BMW/Cadillac; clutch's `{2, 4, 6}`) use `ParameterSpec.discrete_values` and snap to the nearest legal value at render time.

## Active-learning probes (`cli/calibrate.py`)

`optimize calibrate <car> <track>` surfaces parameters with thin observed variance on the target track and proposes a value in the largest unsampled gap of each one's legal range. Drives the loop: probe -> drive a clean stint with the proposed values -> `optimize learn` -> re-fit -> the next recommend has slope/curvature where the parameter used to pin to its observed median.

* Default mode: coverage table + top-N (default 3, controlled by `--targets N`) thin-variance probes + a synthetic setup card with probes tagged `[OPT]` and everything else `[OPT pin]` from the past setup.
* `--status`: coverage table only. Use to decide whether more variance is the bottleneck on accuracy.
* `--output-file PATH` / `--output-file -`: same convention as recommend; default writes to `recommendations/<car>-<short-track>-cal[-status]-<MMDD>-<HHMM>.txt`.

The picker reads `model.per_track_parameter_observed[track]` (populated by `fit_per_car`); proposals are step-snapped to the parameter's UI step or `discrete_values` set; targets that would step-snap onto the past value get re-routed to the second-largest gap.

## Recommendation filename convention (`cli/recommend.py::recommend_cmd` output-file block; `cli/calibrate.py::_maybe_save`)

Default output path: `recommendations/<car>-<short-track>-<mode>[-<fuel>L]-<MMDD>-<HHMM>.<ext>`. `<short-track>` strips variant suffixes via `_short_track` (`spa_2024_up` -> `spa`, `hockenheim_gp` -> `hockenheim`, `sebring_international` -> `sebring`, `daytona_2011_road` -> `daytona`). Mode tag is one of `race` / `quali` / `reset` / `cal` / `cal-status`. Fuel suffix only for quali stints (race auto-pins, would clutter every name). Pass `--output-file -` to suppress file output (e.g. when piping JSON).

**Known regressions / gaps:**

- **Out-of-domain aero now warns + downgrades (2026-06-10).** All five aero
  maps share a 25 mm front-RH floor while the cars run below it (Cadillac
  ~8 mm → ≥ ~+3 % front-balance bias at the floor gradient of ~0.19 %/mm).
  `AeroSurface` counts clamped queries (`aero/interpolator.py::AeroClampStats`);
  `cli/recommend.py::_aero_out_of_domain_warnings` warns and downgrades the
  `rear_wing`/`pushrod`/`perch_offset`/`ride_height` families one confidence
  tier when ≥ 50 % of a run's queries clamped on the front axis. The real fix
  (re-deriving maps below 25 mm) is still open — AUDIT.md H2.

- **`per_track_residuals` flattens setup gradient — FIXED 2026-05-24 (P0.1).** Predict path no longer reads it; fitter no longer computes it; cache key bumped to `FITTERS_LAYOUT_VERSION=10` so pre-2026-05-24 pickles refit clean.
- **Sensitivity floor enforced — FIXED 2026-05-24 (P0.3).** Moves below `_SENSITIVITY_FLOOR = 0.005` on ±1 step are reverted to baseline and listed in `SetupRecommendation.suppressed_below_sensitivity`; narrative renderer excludes them from the "moved" block.
- **Phantom corner 0 + 5-line spam — FIXED 2026-05-24 (P0.4).** `is_real_corner_archetype` gates both guardrail paths; `guardrail_warnings_for_setup` dedupes per corner.
- **Slot-shift type corruption — FIXED 2026-05-24 (P1.4).** `_validate_pickle_slots` refuses pickles with wrong slot types; existing `_repair_legacy_slot_shift` continues to rescue the 2026-05-08 case.
- **Static RH `[predicted]` — FIXED 2026-05-24 (P0.2 + W2 cleanup).** Per-car kinematic linear fit (`physics/static_rh_kinematic.py`) gated on R² ≥ 0.98 ships for the four `setup_static_*_ride_height_mm` channels; `physics/recommend.py::_kinematic_static_rh_ready` skips the legacy k-NN repair when the fit is present. Falls back to k-NN when R² < 0.98 / corpus too thin.
- **Held-out per-channel gate — FIXED 2026-05-24 (P1.1).** `scripts/holdout_accuracy_gate.py::_PER_CHANNEL_THRESHOLDS` carries per-channel pass criteria; gate hard-fails on ANY car failing ANY non-driver-input channel.
- **Watch-most picker — FIXED 2026-05-24 (P3.2).** `_dominant_impact_corner` and `_telemetry_why` divide each impact's `|score_delta|` by the corner's pre-filter spread in the candidate pool, so a long broad-impact corner loses to a corner with concentrated impact.
- **Thin-corpus refusal banner — FIXED 2026-05-24 (P3.3).** `_is_thin_corpus_for_recommend` (n_prod < 20 OR axle_grip_ceilings is None) gates `recommend_cmd`; refusal points the user at `optimize calibrate <car> <track>`.
- **Inverse-track-sample-count training weights — FIXED 2026-05-24 (P2.3).** Forest fitter honours `sample_weight = 1 / sqrt(n_track_rows)` so a Sebring-heavy corpus doesn't drown Spa rows.
- **Corner archetype `phase_duration_s` at fit time — FIXED 2026-05-24 (P2.4).** `CORNER_ARCHETYPE_COLUMNS` extended; `ENV_FEATURE_SCHEMA_VERSION_PER_CAR` 5 → 6.
- **Briefing header per-channel error budget — FIXED 2026-05-24 (P3.1).** `_render_error_budget_block` reads `holdout_accuracy_latest.json` and renders per-channel `mean_abs` lines; falls back to the legacy `Confidence: ...` line on missing rows.
- **Lap-time Spearman gate (P1.2 — code real, results missing).** `scripts/lap_time_correlation_gate.py` carries the full LOSO orchestration (`_compute_loso_pairs_for_track`, lines 161-364) but it has never been run on a populated corpus (~2.5 hr/car-track offline) — no `lap_time_correlation_latest.json` exists, so correlation is still unmeasured.
- **Hybrid vs surrogate-only A/B in CI (PARTIAL — P1.3).** `tests/physics/test_hybrid_heldout_ab.py` gates non-regression invariants (key-set match, finite totals, 50 % relative-delta bound). CI YAML flip + per-car asymmetric "hybrid doesn't lose" assertion still pending.
- **Curb / off-line row masking (DEFERRED — P2.1).** Needs `TrackModel.bump_map` API addition; deferred to next pass.
- `optimize <car> <track> --json` emits valid JSON to stdout; pin/fuel/reset/static-RH warnings populate the `warnings` array (stderr save line still present unless `--output-file -`).
- Brake duct openings still `<TODO>` in `constraints.md`.
- Wind decomposition uses tailwind-worst-case magnitude only; per-corner directional headwind/crosswind correction needs heading data the corner schedule doesn't carry yet (deferred per `physics/wind.py` docstring).
- `fuel_level_l` is fittable but most cars' corpora have thin fuel variance — `predict_setup_readouts` learns whatever pattern exists, which may not be physically pure fuel→RH.
- Driver-input output channels (throttle, brake, damper velocity) plateau at fit_quality ~0.50 (signal == noise). Structural ceiling — the model can't fully resolve channels that depend more on driver input than setup.
- BMW corpus is Sebring-dominated (37/53 sessions). Spa-specific predictions (11 sessions) are weaker than Sebring's by design.
- Evaluator Spearman gate (~0.19 mean on corpus) below PLAN target 0.35 — product posture is guardrailed surrogate, not lap-time physics predictor.

## Project automations (`.claude/`)

- **Hook — PreToolUse on Bash:** blocks destructive ops (`rm`, `mv`, `truncate`, `shred`, `dd`, output redirect) targeting `ibtfiles/` or `aero-maps/`. These dirs hold irreplaceable training data. If a hook fires against a real need, fix the hook or run the command outside Claude Code — do not work around it with `--no-verify` style hacks.
- **Hook — PostToolUse on Edit/Write:** sanity-checks `constraints.md` keeps required headings (`## Defaults`, `## Per-car overrides`).
- **Skill `ibt-inspect`:** summarizes channels / setup / weather for any `.ibt` path. Use during exploratory work on telemetry.
- **Skill `add-constraint`** (user-invocable only): appends new parameter ranges to `constraints.md` in the established schema.
- **Subagent `setup-justifier`:** gates VISION §7 — every recommended parameter must have corner trade-offs, telemetry evidence, sensitivity, and confidence.
- **Subagent `physics-fit-validator`:** verifies VISION §3/§6 — residuals on held-out laps, confidence-vs-density alignment, leaking-formula detection.
- **MCP — context7** (user-scoped): live docs for numpy/scipy/pandas/pyirsdk. Prefer it when uncertain about library APIs instead of guessing.

## Data assets (the only inputs that exist today)

- `ibtfiles/` — raw iRacing telemetry binaries (`.ibt`). 60 Hz, 100+ channels per lap, with the session's garage setup embedded in the YAML header. Filenames encode car and track, e.g. `acuraarx06gtp_daytona 2011 road 2026-04-03 20-31-55.ibt`. Filenames may contain spaces — quote paths. The directory mixes flat files with subfolders (e.g. `ibtfiles/cadillac/`); ingestion must walk recursively. Files range from ~1 MB to ~210 MB; do not load all telemetry into memory at once.
- `aero-maps/` — parsed aerodynamic surfaces, one JSON per `(car, wing_angle)` pair. Five cars (`acura`, `bmw`, `cadillac`, `ferrari`, `porsche`) × discrete wing angles. Schema (per file):
  - `car`, `wing` — identifiers
  - `front_rh_mm`, `rear_rh_mm` — 1D axis grids (front and rear ride height in mm)
  - `balance_pct` — 2D matrix indexed `[front_rh][rear_rh]`, aero balance %
  - `ld_ratio` — 2D matrix, lift/drag ratio
  - Note wing-angle granularity differs per car (e.g. acura indexes at 0.5° steps; others at 1.0°). Do not assume a uniform step.
- `constraints.md` — hard legal bounds for setup parameters (wing, tyre cold pressure, heave spring/slider deflection, static ride heights so far; more incoming). Defaults apply to all cars; per-car overrides shadow them (e.g. Acura wing 6.0–10.0°). The optimizer MUST clamp every recommended value to these bounds before output.

## Architectural commitments from VISION.md (non-obvious — read before designing)

These are deliberate design constraints, not suggestions. Diverging from them defeats the project's premise:

- **Corner-phase is the atomic unit, not the lap.** Segment every lap into corners, then into braking / trail-brake / mid-corner / exit / straight phases. All physics fitting and scoring happens at the corner-phase grain. Never collapse to session averages.
- **Empirical physics, not textbook formulas.** Spring-rate effects, LLTD, aero balance vs ride height — fit them from observed data. No hardcoded engineering equations as the primary model. Aero maps are the one exception (they're empirical lookups already).
- **Lap time is an outcome, not the optimization signal.** Score setups by per-corner-phase physics utilization weighted by each corner's time sensitivity, then sum. Never feed lap time directly into the objective.
- **Track model is compounding and load-bearing for data quality.** It identifies curbs, bumps, rumble strips, and off-track excursions per track position so the physics fitter can mask out non-clean samples. Do not throw away whole laps — mask the dirty sections.
- **Every data point carries environmental context.** Air density, track temp, wind, wetness are 60 Hz channels in the IBT and must travel with each sample through fitting and prediction. The aero maps need an air-density correction; do not treat them as condition-invariant.
- **Output explains every click.** Setup recommendations must justify each parameter with the corners that benefit, the corners that compromise, the telemetry evidence, and the ±1–2 click sensitivity. A bare list of numbers is not an acceptable output format.
- **Confidence is a first-class output.** Sparse/noisy parameter interactions → conservative recommendations; dense/consistent → aggressive. Report confidence alongside every value.

## CLI UX target (VISION.md §8)

Top-level command is `optimize`, with auto-detection of car and track from IBT filenames. Avoid long flag chains and module-path invocations. The default path is "drop in an IBT, get a setup out." Power-user flags are fine as overrides, not as the primary interface.

Two recommend invocations are supported and routed by `_OptimizeGroup.parse_args` in `cli/__init__.py`:

- `optimize <car> <track>` — explicit pair. Track input is slugified before catalog lookup so `laguna-seca`, `Laguna Seca`, and `lagunaseca` all match.
- `optimize <ibt_path>` — first positional is an existing `.ibt` file; car & track are sniffed from the filename via `detect_car_from_filename` / `detect_track_from_filename`.

Both forms route to the same `recommend_cmd`. The end-user-facing walkthrough lives in `GETTING_STARTED.md`; do not duplicate it here.

## Cars covered

BMW M Hybrid V8, Porsche 963, Cadillac V-Series.R, Acura ARX-06, Ferrari 499P. Each has its own suspension architecture and garage parameter set — do not assume a unified setup schema across cars.

**Per-car driver/engineering setup specs** live under `docs/cars/`. Today only `acura_arx06.md` is captured: front RH ~15 mm + rear RH ~40 mm peak-downforce targets at mid-corner / at speed, rake-driven aero balance (more rake → oversteer), preferred balance-lever order (rear wing → rear pushrod → ARB → diff). When this spec exists for a car, the recommendation briefing and the `setup-justifier` subagent should use its vocabulary (rake, downforce trim, entry stability, on-/off-throttle rotation) and must not contradict its targets. Structured as `physics/aero_targets.aero_targets_for(car)` for code consumers; returns `None` for the four cars whose spec hasn't been captured — callers MUST defer to the surrogate rather than invent defaults.

## When to override the optimizer

The score function is per-corner-phase utilization, NOT lap time (VISION §6). For most parameters this correlates well with stopwatch pace. Three known disconnects:

- **Tyre pressure**: surrogate rewards platform-stability (cleaner ride-height telemetry) which higher cold P delivers, but doesn't see the peak-grip loss from smaller contact patch. Community wisdom (152 floor for GTP) wins on lap time.
- **Stiff vs soft setup philosophy**: when a user's recent corpus drifts toward conservative setups (rear toe-in, soft ARB, tight diff), the surrogate inherits that bias even when the same user has a faster validated setup in their IBT history. Compare optimizer recommendation to the user's fastest IBT before applying.
- **Pinned-outside-corpus parameters**: `--pin <param>=<value>` outside the global corpus envelope produces curve-fit extrapolation, not engineered compensation. Confidence still reads "dense" because density is track-wide. Verify on track.
