# VISION §6 — Learning ("get smarter with every lap")

## Clause text

> Every new IBT file is more training data. Re-fit the physics models. Track
> how prediction accuracy improves. Identify which parameter interactions are
> well-understood (many data points, consistent behavior) vs uncertain
> (sparse data, noisy). When the model is uncertain, be conservative. When
> it's confident, be aggressive. Report confidence alongside every
> recommendation. Over time, the system should converge on a complete
> understanding of each car's behavior and be able to generate optimal
> setups for any track from just the track model and aero maps.

## Score: 🟡 partial — load-bearing wiring is real, but the user-facing "trend / confidence" surface routinely lies under load

The architectural pieces VISION §6 mandates are all merged: every fit appends
one row per `(corner, phase, channel)` to `corpus/models/accuracy_log.parquet`,
the cache key folds the session-id set + ontology fingerprint + feature schema
so a new IBT really does invalidate the cached pickle, every parameter on
every recommendation carries a `Confidence(value, lo, hi, n_samples, regime)`,
and the trust radius shrinks for sparse parameters in the DE search
(spec §10). However, the two displays the user actually sees disagree with
the per-fitter numerics that back them, and the `optimize status` "Fit sigma"
column has been collapsing to `0.000` on real corpora because every
`noise_ratio` is ≥ 1.0 and `fit_quality = max(0, 1 - noise_ratio)` saturates.
None of these break the recommend path, but they each make "the system that
gets smarter every lap" visibly indistinguishable from a stuck system.

## Per-clause evidence

| Sub-clause | Code | Verdict | Notes |
|---|---|---|---|
| Every new IBT is more training data | `cli/__init__.py` exposes `optimize learn`; `cli/recommend.py:_build_or_load_model` (l. 586-629) refits when the cache key misses | 🟢 | New session id changes the digest at `_model_cache_path` (l. 632-654). |
| Cache invalidation when new sessions arrive | `_model_cache_path` folds `sorted(session_ids)` + per-car ontology fingerprint `(name → family/fittable/user_settable)` + `ENV_FEATURE_SCHEMA_VERSION` (l. 645-653) | 🟢 | Tested by `tests/cli/test_model_cache_path.py`; also catches stale pickles after an `ontology.py` mutation. |
| Re-fit appends to a longitudinal accuracy log | `physics/io_log.append_accuracy_log` (l. 52-106) writes one row per `(timestamp, car, track, sessions_hash, corner_id, phase, channel)` to `<corpus>/models/accuracy_log.parquet`; called from `physics/fitter.fit` at l. 297-309 inside a `try / except: pass` | 🟢 wiring | Wrapping in `except` is documented as "telemetry-side persistence; never block the model build on it" but it also silently swallows write failures. Worth a `logging.warning` to surface I/O regressions. |
| Per-(corner, phase, channel) granularity tracked over time | `accuracy_log.parquet` schema (`io_log.py:1-27`) carries `corner_id`, `phase`, `channel`, `cv_residual_std`, `signal_std`, `noise_ratio`, `n_samples`, `regime` per row | 🟢 | Confirmed live: `accuracy_log.parquet` in this repo holds 16 200 rows across 14 (car, track) pairs. |
| `Confidence.derive` regime classifier | `confidence/confidence.py:52-84` — `n_samples<30 → "sparse"`; else `noise_ratio = cv_residual_std / max(signal_std, 1e-12)`; `>0.5 → "noisy"`, `>0.2 → "confident"`, `else "dense"` | 🟢 | Single source of truth; tested in `tests/confidence/test_regime_derivation.py` (table-driven across all four regimes plus 95 % bracket invariant). |
| Conservative when uncertain (trust radius) | `physics/recommend._pin_or_trust_bounds` (l. 322-348) routes via `_TRUST_FRACTION = {"sparse": 0.30, "noisy": 0.50}` (l. 245). Sparse → 30 % of constraint range around baseline; noisy → 50 %; confident/dense → full range. Plus near-constant pinning at `_NEAR_CONSTANT_FRACTION = 0.02` (l. 312) | 🟢 | Real protective behaviour: bmw Spa briefing's `arb_size_front` was pinned because per-session std fell below 2 % of the bound range. |
| Aggressive when confident | Same `_TRUST_FRACTION` table — confident/dense parameters have `fraction = None`, so `_trust_bounds` returns the full `(lo, hi)` (l. 289-293). | 🟢 | Symmetric with the conservative case. |
| Report confidence alongside every recommendation | `physics/recommend._parameter_confidence` (l. 374-396) attaches a `Confidence` to each parameter via the median over fitters that depend on the parameter; renderer surfaces `[confidence: <regime>]` per parameter (`explain/render_text.py:172`) plus rolled-up header (`render_text.py:151-166`) | 🟢 wiring, 🟡 reported numbers | See finding 1 below — the per-parameter regime in the BMW Spa briefing disagrees with what the same model's per-fitter regimes actually compute. |
| `optimize status` renders the trend | `cli/recommend.status_cmd` (l. 294-366) calls `load_latest_fit_quality` per (car, track) and emits `track fit_quality: 0.847 (+0.012 vs prior fit)` (l. 331-340); table column `Fit sigma` at `explain/render_text.py:118-127` | 🟢 wiring, 🔴 numerics broken on real corpus | See finding 2 below — saturation at 0. |
| Per-track regime in the status table | `_coverage_regime(n_sessions, n_laps)` (`cli/recommend.py:1016-1025`) — `n_sessions≥4 ∧ n_laps≥100 → "dense"`, etc. | 🟡 | Heuristic over **lap counts**, not the per-fitter `Confidence.derive` regime. Two fundamentally different regime classifiers coexist with the same name. |
| Per-track regime persists in the accuracy log | `io_log.append_accuracy_log` writes `regime = Confidence.derive(...)` per fitter (l. 72-77, 91) | 🟢 |  |

## Findings (in severity order)

### 🔴 1. Status `Fit sigma` column saturates to `0.000` on real telemetry

`load_latest_fit_quality` (`physics/io_log.py:121-169`) collapses to one
number per fit run via `fit_quality = max(0.0, 1.0 - median(noise_ratio))`.
On the live `corpus/models/accuracy_log.parquet`, every (car, track) pair
except `porsche / spielberg_gp` has a median `noise_ratio ≥ 1.0`, which
clamps `fit_quality` to exactly 0.000. The BMW status table I just rendered
shows it explicitly:

```
Track                       Sessions  Valid laps  Clean CP  Fit sigma     Regime
nurburgring_combined               1           1        30          -     sparse
roadatlanta_full                   2          13       390      0.000      noisy
sebring_international             37         356     10680      0.000      dense
spa_2024_up                        9          43      1290      0.000  confident
spielberg_gp                       2           6       180      1.000      noisy
```

VISION §6 demands "Track how prediction accuracy improves." A column whose
output is 0.000 across 4 of 5 tracks for the most-data car in the corpus
gives the user no signal at all about whether yesterday's session helped or
hurt the model. The renderer header even calls it "Fit sigma" but the
underlying scalar is `1 - median(noise_ratio)`, which is neither a sigma
nor monotonic in any quantity a race engineer would recognise.

Two contributing root causes:

  * `noise_ratio = cv_residual_std / signal_std` exceeding 1.0 means the
    K-fold residual std is larger than the training-data std for the output
    channel. Stage-3 fits one model per `(corner, phase, channel)` over the
    full setup vector + 12 env channels with `n_samples` typically in the
    3–22 range per fitter (the spa parquet shows median `n_samples = 11`).
    With <30 samples and a 14+ feature joint vector, GP/RF CV residuals
    routinely exceed in-sample variance — that's expected, not a bug.
  * The 1-minus-median squashes that into "0", losing the fact that one fit
    might be at noise=1.05 and another at noise=2.5. A monotone transform
    that stays informative across the full range (e.g. `1 / (1 + median)`
    → 0.49 vs 0.29, or just reporting the median noise ratio directly with
    a "lower is better" header) would preserve the trend signal.

This is the headline VISION §6 surface and it has effectively no
information content right now.

### 🟡 2. Recommendation header confidence and per-parameter `[confidence: dense]` tags disagree with the persisted per-fitter regimes for the same model

The BMW Spa briefing in `recommendations/bmw__spa_2024_up__20260505-180530.txt`
(line 3) reads:

```
Confidence: dense (n=2330 backing samples for the dominant dense parameter, 46 parameters reported)
```

…and every one of the 46 parameter blocks carries `[confidence: dense]` and
"dense confidence backed by 2330 samples". The persisted accuracy log for
the same `(bmw, spa_2024_up)` pair, however, contains 2 065 fitter rows of
which **every single row** is `regime = sparse` (max `n_samples = 22`,
median = 11). The 2 330-sample claim cannot be reproduced from any fitter
in the current accuracy log.

The likely path that produced the discrepancy:

  * The briefing was rendered from a stale on-disk pickle
    (`corpus/models/bmw__spa_2024_up__67f5c76c0533ed5d.pickle`) created
    before a `physics.fitters` package re-layout — loading it in the
    current tree fails with `ModuleNotFoundError: No module named
    'racingoptimizer.physics.fitters.ridge'`. So the briefing carries
    `n_samples` from a much older fit (per-parameter sum across all
    `(corner, phase)` — the pre-Stage-3 v1/v2 feature schema, where
    "samples backing parameter X" was the row count over all per-X
    fitters) — and `_parameter_confidence` taking the median over per-X
    fitters in the v1 schema returned a 4-digit number.
  * The cache-invalidation key (`_model_cache_path`) folds the ontology
    fingerprint and `ENV_FEATURE_SCHEMA_VERSION` (which is correct), but
    nothing folds in the *fitters package layout*, so a refactor that
    renames a fitter module silently breaks the cached pickle without
    invalidating its on-disk filename. The next read attempt raises and the
    `_build_or_load_model` exception swallow at l. 596-600 falls through to
    a fresh fit — which is fine for this run, but the briefing the user is
    reading right now was produced before the rename and never refreshed.

VISION §6 says "Report confidence alongside every recommendation." The
confidence tag is reported, but for at least the BMW Spa briefing the user
holds it is reporting a statistic from a no-longer-loadable pickle that
disagrees by ~250× with what the live fitter records say. The user has no
way to tell.

### 🟡 3. Two unrelated regime classifiers both named "regime"

`cli/recommend._coverage_regime` (l. 1016-1025) classifies a (car, track)
slot by **session and lap counts** — `n_sessions ≥ 4 ∧ n_laps ≥ 100 →
"dense"`. `Confidence.derive` (`confidence/confidence.py:66-75`) classifies
a single fitter by `n_samples` + the `cv_residual_std / signal_std` ratio.

The `optimize status` table prints the former in its rightmost column.
`load_latest_fit_quality` returns the latter (per row in the parquet) but
only the median noise ratio is rolled up into the table; the per-track
regime column is the lap-count heuristic. Result: the status table can
show `spa_2024_up confident` (lap-count heuristic) at the same time the
per-fitter accuracy log holds 2 065 sparse rows for that exact (car, track)
pair (live data, see finding 1).

VISION §6 only specifies one classifier — "well-understood (many data
points, consistent behavior) vs uncertain (sparse data, noisy)". Having
the status command report a session-count proxy under the same word that
the recommend command uses for the residual-driven `Confidence.derive`
regime is the kind of soft drift the audit is supposed to catch.

### 🟡 4. `append_accuracy_log` failures are silently swallowed inside `fit()`

`physics/fitter.py:300-309`:

```python
try:
    append_accuracy_log(...)
except Exception:
    # Telemetry-side persistence; never block the model build on it.
    pass
```

Bare `except: pass`. A perms / disk-full / parquet-schema-drift failure
here would mean the trend line in `optimize status` starts reading stale
or empty data with zero feedback to the user. Not a VISION violation per
se, but it means clause "Track how prediction accuracy improves" can quietly
stop working. A `warnings.warn(...)` here would be a one-line fix.

### 🟢 5. Cache-invalidation behaviour matches the spec for ontology / schema mutations

`_model_cache_path` (`cli/recommend.py:632-654`) folds the per-car ontology
fingerprint plus `ENV_FEATURE_SCHEMA_VERSION` into the digest. The five
tests in `tests/cli/test_model_cache_path.py` pin: ontology mutation,
schema-version mutation, identical-input stability, session-id order
independence, and per-`(car, track)` isolation. This is solid.

The one gap (covered under finding 2) is that the fitters package
**layout** isn't part of the digest — renaming a fitter submodule will
make pre-rename pickles unloadable without invalidating their on-disk
filenames.

## BMW Spa card cross-check (per-prompt step 4)

Prompt asked for: "look for `[confidence: dense]` etc., n_samples backing
each (header says n=2330)". The briefing
`recommendations/bmw__spa_2024_up__20260505-180530.txt` shows:

  * Header: `Confidence: dense (n=2330 backing samples for the dominant
    dense parameter, 46 parameters reported)` ✓ (line 3)
  * All 46 parameter blocks carry `[confidence: dense]` ✓
  * Each parameter's evidence section reads `dense confidence backed by
    2330 samples` ✓ — but see finding 2: the 2 330 number is not
    reproducible from the live accuracy log for this (car, track) pair,
    where every fitter row is sparse with `n_samples ≤ 22`.

`optimize status bmw` (run live during this audit):

```
Track                       Sessions  Valid laps  Clean CP  Fit sigma     Regime
nurburgring_combined               1           1        30          -     sparse
roadatlanta_full                   2          13       390      0.000      noisy
sebring_international             37         356     10680      0.000      dense
spa_2024_up                        9          43      1290      0.000  confident
spielberg_gp                       2           6       180      1.000      noisy
Overall regime: dense
```

Per-Track / Sessions / ValidLaps / CleanCP / FitSigma / Regime columns all
render correctly per the prompt's contract. The CleanCP value
(`n_clean_corner_phases`) is `n_valid_laps × 30` per
`_approximate_clean_corner_phases` (l. 997-1013) — explicitly documented
as a placeholder until corner-phase parquets land on disk. Not a VISION
violation; flagged here so the next §1 / §2 audit doesn't claim CleanCP is
load-bearing.

## Test results

```
$ uv run pytest -q tests/cli/test_status_cmd.py tests/physics/test_accuracy_log.py tests/confidence/
.......................                                                  [100%]
23 passed in 14.86s
```

23 / 23 pass. The slow `test_fit_writes_accuracy_log` was collected and
skipped because the BMW Sebring fixture is not git-lfs materialised in
this sandbox; the other three accuracy-log tests run synthetically and
pass.

## What's tested vs. what's implemented

| Capability | Implemented | Tested |
|---|---|---|
| `Confidence.derive` regime table | ✓ | ✓ table-driven (`tests/confidence/test_regime_derivation.py`) |
| 95 % bracket invariant | ✓ | ✓ |
| Append-and-load round trip | ✓ | ✓ (`test_append_and_load_round_trip`) |
| Real `fit()` persists a log row | ✓ | ✓ (slow, LFS-skipped in sandbox; covered by `test_fit_writes_accuracy_log`) |
| Cache key folds ontology + schema | ✓ | ✓ (`tests/cli/test_model_cache_path.py`) |
| Status renders trend line | ✓ | ⚠ tested only for happy-path single-fit ("first fit" branch); no test asserts the prior-vs-latest delta line renders correctly |
| Trust-radius narrowing for sparse params | ✓ | ✓ (`tests/physics/test_recommend.py`) |
| Sparse-by-sample-count → regime sparse | ✓ | ✓ (`tests/physics/test_validator_gate.py::test_confidence_aligns_with_data_density`) |
| Per-fitter accuracy log noise_ratio is a useful trend signal | partial | ✗ no test asserts `fit_quality` stays >0 on a typical Stage-3 fit; current behaviour is saturated (finding 1) |
| Recommendation header `n_samples` matches the live fitter records | partial | ✗ no test asserts the briefing's "n=N backing samples" matches what `_parameter_confidence` returns for the same model |

## Suggested follow-ups (out of audit scope, but flagged)

1. **Replace `1 - median(noise_ratio)` with a transform that stays
   informative across `noise_ratio ∈ [0, ∞)`** (finding 1). Either
   `1 / (1 + median)` or simply rename the column and let it report
   `median(noise_ratio)` directly with a "lower is better" header. Keep the
   trend-vs-prior delta line either way — that's the load-bearing VISION §6
   surface.
2. **Surface `append_accuracy_log` failures** with `warnings.warn` instead
   of `except: pass` (finding 4).
3. **Fold a fitters-package fingerprint** (e.g. `frozenset of importable
   `racingoptimizer.physics.fitters.*` module names) into
   `_model_cache_path` (finding 2). Today an ontology mutation invalidates
   cached pickles; a fitters-package rename does not.
4. **Pick one regime classifier** for `optimize status` and document the
   other (finding 3). Two functions both named "regime" returning different
   answers from different inputs is a footgun.
5. **Add a golden test** that asserts the briefing's per-parameter
   `n_samples` matches what `_parameter_confidence(model, name).n_samples`
   returns for the same in-memory model (finding 2). Catches silent drift
   between cached pickle and renderer.
