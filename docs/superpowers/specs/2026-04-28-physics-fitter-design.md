# Physics Fitter — Design Spec

**Date:** 2026-04-28
**Slice:** E — Physics fitter (the heart of the optimizer)
**Module:** `racingoptimizer.physics` (also births `racingoptimizer.confidence`)

## 1. Context

`racingoptimizer` is a physics-based setup optimizer for iRacing GTP cars (see `VISION.md`). Slice A persists every IBT as parquet plus a session-level garage setup JSON. Slice B decomposes laps into corner-phases and exposes per-phase physics state. Slice C interpolates aerodynamic surfaces. Slice D builds compounding track models and writes the per-sample `data_quality_mask` that says which samples are clean. None of those slices makes a recommendation; they prepare the substrate.

Slice E is the substrate-consumer. It fits empirical physics — *learned from measured car behaviour, never from textbook formulas* (VISION §3, CLAUDE.md commitment) — between every relevant garage parameter and every per-corner-phase physics state, on a per-car basis. Once fit, it predicts how a candidate setup will behave at every corner-phase under given conditions, scores it, and searches the constrained parameter space for the score-maximizing setup. Confidence is a first-class output: every predicted value carries an uncertainty bracket and a sample-density tag (VISION §6).

This spec is the design that the future implementation session follows. It also births the cross-cutting `racingoptimizer.confidence` module (master plan §2). It does *not* render recommendations to the user — that is slice F. It does *not* run the corner-phase decomposition — that is slice B. It does *not* compute aero-map values — that is slice C. The boundaries are firm.

Two non-negotiables that drive every design choice in this spec:

1. **Lap time NEVER appears in the optimization objective.** Lap time is an outcome, not a signal (VISION §5, CLAUDE.md). It is permitted only as a held-out diagnostic (residual check, weight derivation source, confidence calibration). The `physics-fit-validator` subagent (`.claude/agents/physics-fit-validator.md`) rejects any fit that leaks lap time into the objective — this is gated as a CI/release check.
2. **Empirical, not analytical.** Spring rates, LLTD, aero balance vs ride height — every relationship is fit from observed `(setup, environment, measured-physics-state)` tuples. The aero map is the single permitted lookup (it is empirical itself, just precomputed by iRacing). No hardcoded engineering equations as the primary model. The `physics-fit-validator` flags fits whose residuals are sign-correct but magnitude-off by a near-constant factor — the tell of a leaking textbook formula.

## 2. Public API

```python
from racingoptimizer.physics import fit, PhysicsModel, SetupRecommendation
from racingoptimizer.confidence import Confidence

fit(car: str, session_ids: list[str], track_model: TrackModel) -> PhysicsModel
    # Empirically fits a per-car physics model from the named sessions. Each
    # session_id resolves via slice A's catalog; lap data and per-sample
    # data_quality_mask are pulled from parquet, segmented into corner-phases
    # by slice B, and joined against the embedded garage setup JSON. Returns
    # a PhysicsModel populated with one fitter per (parameter, corner-phase)
    # tuple. Fitter family is chosen per parameter (§5). Random seed pinned
    # for determinism (§12). Failures on individual fitters mark that fitter
    # 'untrained' but do not abort the model build (§11).

PhysicsModel.predict(
    setup: dict,
    env: EnvironmentFrame,
    corner_phase_key: CornerPhaseKey,
) -> CornerPhaseStateWithConfidence
    # Predict the per-corner-phase physics state (understeer angle, load
    # transfer asymmetry, traction utilization, aero platform state, etc.)
    # for a hypothetical setup under a hypothetical environment at a known
    # (corner, phase). Each output channel arrives wrapped in Confidence.
    # Conditions on env (does NOT marginalize). If any required fitter is
    # untrained, raises UntrainedError for that channel; the caller decides
    # whether to fall back. Aero-related channels delegate to slice C's
    # AeroSurface.interpolate when available; without C, returned with a
    # downgraded Confidence.regime ('sparse' or 'noisy', §9).

PhysicsModel.recommend(
    track: str,
    env: EnvironmentFrame,
    constraints: ConstraintsTable,
) -> SetupRecommendation
    # Search the constraint-clamped parameter space for the setup that
    # maximises score_setup under the target environment on the target track.
    # Gradient-free optimization (§10). Returns SetupRecommendation: a dict
    # of {parameter -> (value, Confidence)} plus a per-corner-phase score
    # breakdown (consumed by slice F to render SetupJustification). Every
    # value is clamped to constraints both pre- and post-optimization.

PhysicsModel.score_setup(
    setup: dict,
    track: str,
    env: EnvironmentFrame,
) -> float
    # Sum of per-corner-phase utilization weighted by corner time-sensitivity
    # (§6). Pure scalar, no Confidence — confidence belongs on predictions,
    # not on the aggregate score. Lap time NEVER feeds this function.

# Born here (cross-cutting; master plan §2):
@dataclass(frozen=True)
class Confidence:
    value: float
    lo: float                 # 95% lower bracket
    hi: float                 # 95% upper bracket
    n_samples: int
    regime: Literal['sparse', 'noisy', 'confident', 'dense']
```

`SetupRecommendation` is a typed dataclass (not a bare dict) so slice F can iterate `.parameters` and `.score_breakdown` deterministically:

```python
@dataclass(frozen=True)
class SetupRecommendation:
    car: str
    track: str
    env: EnvironmentFrame
    parameters: dict[str, tuple[float, Confidence]]   # value + uncertainty per param
    score_breakdown: dict[CornerPhaseKey, float]      # per (corner, phase) contribution
    untrained_parameters: list[str]                   # families that fell back to median
    aero_correction_available: bool                   # tracks slice C presence
```

Polars frames where slice E reads from slice A/B, native dataclasses where slice E exposes typed records. Match slice A's stack: `polars`, `pyarrow`, plus `scipy` (interpolation, optimization) and `scikit-learn` (GPs, tree ensembles).

## 3. Confidence contract

`racingoptimizer.confidence` is born here. The minimum contract from master plan §2 is pinned to the dataclass above. Two implementation rules slice E enforces:

- `lo <= value <= hi` always. Violations are programmer error (raise on construction).
- `regime` is **derived**, never user-supplied. Construction takes `value, lo, hi, n_samples, cv_residual_std, signal_std` and computes `regime` per the table below.

### Regime derivation

Let `noise_ratio = cv_residual_std / signal_std` (residual std from K-fold CV divided by std of the underlying physics-state signal in the training data).

| Condition | `regime` |
|---|---|
| `n_samples < 10` | `sparse` |
| `noise_ratio > 0.5` | `noisy` |
| `noise_ratio < 0.1 AND n_samples > 100` | `dense` |
| else | `confident` |

Order matters: `sparse` short-circuits before `noisy`; a parameter with five samples is reported sparse even if its tiny residual implies dense. This is intentional — five samples cannot ground a "dense" claim regardless of fit quality.

`Confidence.regime` is the lever slice F uses when phrasing recommendation justifications ("we are confident in this value because…" vs "this value is provisional — only six laps support it"). It is also what `PhysicsModel.recommend` examines when selecting the gradient-free optimiser's trust radius (§10).

## 4. §Setup ontology

Slice E is the first slice that reasons about garage parameters as typed values. Every prior slice has carried the setup as opaque JSON in the catalog `setup` column (master plan §3 A↔E handshake). Slice E pins the typed subset.

The ontology is a per-car mapping `parameter_name -> (json_path, dtype, units, fittable)`. Untyped parameters (cosmetics, throttle-map slot, telemetry export options, anything the optimizer should not touch) **pass through unchanged** when `recommend` returns a setup; the fitter never reads them and never writes them. The aim is "use everything, lose nothing" without committing to "fit everything" — the parameters that are not in the ontology are not fitted, but they are not stripped either.

### Fittable families (slice E reasons over these)

Available as soon as slice A lands (covered by today's `constraints.md`):

- **Heave springs:** front, rear (N/mm)
- **Heave sliders:** front, rear (mm static deflection)
- **Tyre cold pressures:** four corners — LF, RF, LR, RR (kPa)
- **Front wing:** where applicable per car (deg) — *acura only currently exposes a front wing knob; spec keeps the column nullable for the others*
- **Rear wing:** all cars (deg)
- **Static ride heights:** four corners — LF, RF, LR, RR (mm)

Available after Constraint Expansion (CE — master plan §5):

- **ARBs:** front, rear stiffness (clicks or N/mm depending on car)
- **Dampers:** LSC, HSC, LSR, HSR per corner (clicks)
- **Corner weights:** FL, FR, RL, RR (N or kg, per car convention)
- **Brake bias:** front:rear percent
- **Differential:** preload (Nm), coast ramp (deg or %), power ramp (deg or %)

Slice E's implementation **cannot start until CE is complete for every family it intends to fit.** This is a hard precondition copied from master plan §5: CE is scheduled before slice E's spec; slice E's plan and implementation block on it. The implementation session must verify `constraints.md` covers the families enumerated above before writing any fitter code.

### CE-gated families and graceful degradation

If at implementation time CE is incomplete (e.g., dampers landed but differential is still TODO), slice E **fits the bounded subset and skips the rest**. Skipped families are reported in `SetupRecommendation.untrained_parameters`; their values fall through from a configurable baseline setup (the user's prior session's setup, or a per-car canonical baseline if none). This mirrors §11's failure handling for individual fitter exceptions: the model degrades gracefully rather than refusing to run. The implementation session must enumerate which families are CE-gated in the plan, and `physics-fit-validator` must accept "untrained, fell back" as a valid status (it is not a leaking-formula condition).

### Per-car wrinkles

The five cars share families but not field paths or units inside the IBT YAML setup blob. The ontology lives in `racingoptimizer/physics/ontology.py` as five frozen dictionaries (one per car) and a `setup_value(car, parameter, setup_json) -> float` helper that resolves the path. A typed-but-untrained parameter (e.g., a car has dampers in its YAML but CE has not bounded them) is enumerable from the ontology and clearly distinguished from an unreasoned-about parameter (e.g., a cosmetic toggle).

## 5. Fitter family per parameter

Each `(car, parameter, corner-phase, output-channel)` quadruple gets its own fitter. Output channels are the per-phase physics-state outputs from slice B's `corner_phase_states` (understeer angle, load transfer asymmetry, traction utilization, aero platform state, roll angle/rate, damper velocities). The fitter takes `(parameter_value, env_features)` as input and predicts the output channel.

Two fitter families are pinned, chosen per the **dimensionality and continuity** of the input parameter:

### Continuous low-dim parameters → Gaussian process

Parameters: wing angle (front, rear), ride heights (4 corners), heave spring rates (front, rear), heave slider deflections (front, rear), ARB stiffness (front, rear), tyre cold pressures (4 corners).

These are smooth, continuous, low-dimensional (single scalar per parameter) and we will be data-sparse in many regions of the corpus (especially per-car-per-track). Gaussian processes:

- Give a posterior mean **and** posterior variance natively — the variance becomes `Confidence.lo / hi` directly (no bootstrap loop needed for the brackets, although bootstrap CV still runs for `cv_residual_std`).
- Handle sparse data well via the kernel prior. With six laps spanning two spring values, a GP says "we don't know what the curve looks like off-grid" by widening its variance; a tree ensemble would silently extrapolate.
- Are cheap at our scale (low hundreds of training points per fitter, scalar input + ~5 env features).

Kernel choice is an open question (§15) — Matérn-2.5 is the default recommendation pending sensitivity testing; RBF is a fallback if Matérn produces overly smooth fits.

### High-dim or discrete parameters → Random forest / gradient-boosted trees

Parameters: dampers (LSC/HSC/LSR/HSR per corner — together a 16-dimensional click space whose cells interact), corner weights (4-dim, near-constrained-sum), brake bias (discrete clicks in iRacing UI), differential (3 strongly-coupled scalars), gear ratios where exposed.

- Tree ensembles handle nonlinearity and feature interactions natively; we expect damper interactions to be the dominant fit complexity.
- Robust to sparse data without producing wild extrapolations — they predict the median of the relevant leaf, which is conservative.
- Uncertainty is **not** native; uses **bootstrap CI** (resample sessions K times, refit, take the K predictions' 2.5/97.5 percentiles) as `Confidence.lo / hi`.

`scikit-learn`'s `RandomForestRegressor` for the discrete-and-coupled families; `GradientBoostingRegressor` reserved for cases where RF underfits (deferred to plan).

### Per-corner-phase localization

Every fitter is local to one `(corner_id, phase)`. The recommendation step (§10) sums across all corner-phases on the target track. This is the CLAUDE.md "corner-phase is the atomic unit" commitment as code: there is no global "spring rate effect on the car" model — there is a "spring rate effect on understeer angle at Sebring T1 mid-corner" model, and the optimiser sums those.

The per-phase localization is what makes the trade-off visible: a setup change improves utilization at T1 mid-corner by 3% but worsens it at T17 trail-brake by 2%. Slice F renders that. Slice E *produces* it.

### Empirical-only enforcement

No fitter is allowed to internally call a textbook formula as a feature transform. The `physics-fit-validator` subagent (§13) checks the residual signature of each fit; a constant-factor magnitude error across the parameter range is flagged as suspected formula leakage. The implementation must keep the feature pipeline transparent and source-controlled.

## 6. Score function

```
score(setup) = sum over (corner, phase) in track.corner_phases of [
    utilization(corner, phase, setup) * weight(corner)
]
```

Two functions to define: `utilization` and `weight`.

### Utilization

`utilization(corner, phase, setup) ∈ [0, 1]`. A 0 means "the car is wasting most of its capability at this corner-phase under this setup"; a 1 means "the car is at the limit of the physics envelope". Higher is better. Per VISION §4, six sub-utilizations combine:

| Sub-utilization | Definition | Source channels |
|---|---|---|
| `grip` | Lateral G achieved / lateral G achievable given downforce + tyre + condition | `AccelLat`, predicted downforce, tyre pressure model |
| `balance` | 1 - normalised |understeer_angle - target_balance| | Predicted understeer angle |
| `stability` | Margin to loss-of-control: how far from yaw-rate divergence | Predicted yaw rate, predicted understeer slope |
| `traction` | Power-down efficiency on exits: `1 - wheelspin_excess` | Predicted wheel speed differential |
| `aero_eff` | `ld_ratio` from slice C at predicted ride heights | Slice C's `AeroSurface.interpolate` |
| `platform` | 1 - normalised ride-height variance (penalises bottoming and excursions) | Predicted ride heights |

Combination: weighted sum with phase-specific weights baked into a small lookup. Braking phase weights `grip + stability + platform` heavily; mid-corner weights `grip + balance`; exit weights `traction + balance + aero_eff`; straight weights `aero_eff + platform`. Trail-brake is the hand-off: a blend of braking and mid-corner weights. The phase-weight table lives at `racingoptimizer/physics/phase_weights.py` as a frozen constant; the implementation session may tune the numbers but the table's *shape* is pinned here.

Each sub-utilization is computed from a `PhysicsModel.predict` output; that means each carries `Confidence`. The aggregate utilization for a `(corner, phase)` is a confidence-weighted blend (sub-utilizations with `regime='sparse'` contribute less). This is how VISION §6's "be conservative when uncertain" surfaces in the score — the aggregator naturally down-weights uncertain components.

### Weight (corner time-sensitivity)

`weight(corner) ∈ [0, 1]`, normalised to sum to 1 across the track's corners. Definition: how much lap-time delta a 1% utilization improvement at this corner is worth.

**Derivation (held-out, never feeds the objective):**

1. Across all sessions on the target track, partition laps into "high utilization at corner C" and "low utilization at corner C" using the median of measured per-corner utilization.
2. Compute the difference of *mean lap times* between the two groups, controlling for environment via stratification (group within env-similarity buckets) — this gives `lap_time_sensitivity[C]` in seconds per utilization unit.
3. Normalise across all corners: `weight[C] = lap_time_sensitivity[C] / sum(lap_time_sensitivity)`.

This pre-computation produces a per-track `weight` table that is fixed for the lifetime of a `PhysicsModel`. Lap time appears here exactly once: as the source of weights, *outside* the objective. The score function itself takes lap time as a black box; lap times of candidate setups are never queried.

The `physics-fit-validator` subagent verifies this separation: it grep-checks that `score_setup` and the inner objective passed to the optimiser do not import or read any lap-time channel. The implementation must keep weight derivation in a clearly-named precompute (`weight_corners(track) -> dict[corner_id, float]`) that runs once and then is immutable for the rest of the model's life.

### The non-negotiable

CLAUDE.md and VISION §5 elevate this to a design rule: **lap time NEVER appears in the optimization objective.** Repeating because it has been the most-violated rule in setup tooling: lap-time-as-objective is a flagged risk in master plan §6 and the `physics-fit-validator` subagent rejects any fit whose objective signature consumes a lap-time channel. The implementation session is required to invoke `physics-fit-validator` against the trained model before merging slice E.

## 7. Confidence reporting

Slice E reports `Confidence` from K-fold cross-validation across `session_ids`.

### Procedure

1. Partition `session_ids` into K folds (default K=5, configurable; minimum K=3 for tiny corpora). Stratify on `(track, env-bucket)` so each fold sees diverse conditions.
2. For each fold, fit a `PhysicsModel` on K-1 folds, predict the held-out fold's per-corner-phase physics state.
3. Collect residuals per fitter: `residual = predicted - measured` for every held-out lap × corner-phase × output channel.
4. Per fitter, compute:
   - `cv_residual_std`: std of held-out residuals
   - `signal_std`: std of the measured output channel across the training data
   - `Confidence.lo, hi`: empirical 2.5/97.5 percentiles of `(predicted ± residuals)` evaluated at the prediction point
   - `Confidence.regime`: derived per §3 from `n_samples` (the training-data point count for this fitter) and `noise_ratio = cv_residual_std / signal_std`

### Calibration check

Slice E's integration test (§13) asserts `Confidence.lo <= observed <= Confidence.hi` for ≥ 90% of held-out predictions on the BMW Sebring fixture. If the calibration is below 90%, the bracket-derivation procedure widens (raise the bootstrap percentile range, or fall back to GP posterior variance for GP-fitted parameters). The `physics-fit-validator` subagent inspects calibration alignment vs data density and flags any fit that reports `confident` or `dense` on a region where held-out residuals exceed `signal_std * 0.5`.

### Cold-start regime

When `len(session_ids) < K`, K is reduced to `len(session_ids) - 1` (LOO CV); when `len(session_ids) < 3`, all fits are tagged `regime='sparse'` regardless of measured residuals — the sample size cannot ground a stronger claim. This matches §3's `n_samples < 10` short-circuit applied at the model level.

## 8. Weather / environment as feature

Per CLAUDE.md ("every data point carries environmental context") and master plan §2 (the `racingoptimizer.context` module is born in slice B), `EnvironmentFrame` is a per-corner-phase aggregate of the per-sample weather/track-surface channels (`AirTemp`, `AirDensity`, `AirPressure`, `RelativeHumidity`, `WindVel`, `WindDir`, `TrackTempCrew`, `TrackWetness`).

Slice E uses `EnvironmentFrame` as **a feature**, not as a marginalisation axis. Concretely:

- During fit, every training tuple is `(parameter_value, env_features, output_channel)`. `env_features` is the `EnvironmentFrame` for that corner-phase, vectorized to scalars (mean per channel over the phase's sample range).
- The fitter learns the joint sensitivity `f(parameter, env)` rather than averaging across environments and reporting a condition-blind sensitivity.
- During prediction, `env` arrives from the caller (the user-supplied target environment) and is fed to the fitter alongside the candidate `parameter_value`.
- During recommendation, `env` is passed once to `recommend(track, env, constraints)` and held constant during the optimiser's search — the recommender finds the setup that maximises score *for that environment*.

**Anti-pattern (forbidden):** marginalising environment by averaging the fit across all observed weather. That collapses tyre-temp sensitivity into "the average effect of tyre temp on grip" and loses the reason the user supplied `env` in the first place.

When an `env_features` vector for a prediction sits well outside the training-data envelope (e.g., predicting at 50 °C track when the corpus tops out at 35 °C), the GP variance widens automatically and `Confidence.regime` goes to `sparse`. Tree-ensemble fits cannot do this naturally; the implementation must add an explicit envelope check (compare each `env_features` dimension to its training-data range) and degrade `regime` accordingly.

## 9. Aero correction integration

Slice C exposes `AeroSurface.interpolate(front_rh_mm, rear_rh_mm, wing_deg, air_density) -> (balance_pct, ld_ratio)`. Slice E calls it from inside the GP fitters whose output channel is downforce-related (specifically the `aero_eff` and `platform` sub-utilizations and the predicted-downforce input to `grip`). This keeps the empirical aero map as the single source of truth for ride-height-to-downforce mapping; the fitter only learns *what ride heights this setup produces*, not *what downforce those ride heights deliver*.

### Soft-prereq behaviour

Per master plan §4 (slice E's hard prereqs are A, B, D, CE; **soft on C**), slice E must run when slice C is unavailable:

- Aero-related sub-utilizations fall back to a no-aero-correction mode: `aero_eff = 0.5` (a neutral constant), `grip` is computed without the downforce-derived contribution, `platform` is computed from ride-height variance only.
- `Confidence.regime` for high-speed corners is downgraded one tier (e.g., `confident` → `noisy`, `dense` → `confident`). High-speed defined as predicted phase-mean speed > 200 km/h.
- `SetupRecommendation.aero_correction_available = False`.

This degradation is loud, not silent: `recommend` logs a warning that aero correction is unavailable and the user is told (via slice F's rendering) that high-speed-corner recommendations are coarser. The fitter still produces a recommendation rather than refusing — VISION §6 ("over time, the system should converge … and be able to generate optimal setups for any track") permits running on partial information so long as confidence reflects it.

### Air-density sensitivity

Slice C handles air-density correction internally per its spec's open question (per-call multiplier on `ld_ratio`). Slice E supplies the air density from the `EnvironmentFrame` to slice C; it does not re-derive density correction itself.

## 10. Optimization (the recommendation step)

`PhysicsModel.recommend(track, env, constraints)` searches the constrained setup space for the score-maximizing setup.

### Search algorithm

Gradient-free, because (a) tree-ensemble fitters are non-differentiable, (b) several parameters are integer-valued (clicks), and (c) the score function is a confidence-weighted aggregate that is messy to differentiate end-to-end.

Two candidates: **CMA-ES** (`cma` package) or **scipy.optimize.differential_evolution**. Recommendation: **differential_evolution** for the first implementation — it is in scipy (no new dependency), handles bounded variables natively, and its budget is straightforward to cap (`maxiter`, `popsize`). CMA-ES is a fallback if convergence is poor on high-dim damper subspaces; deferred to plan.

### Constraint handling

Constraints are loaded from `constraints.md` via a typed `ConstraintsTable` class (open question in §15: structure pinned in plan). For each parameter the table exposes `bounds(car, parameter) -> (low, high)` with per-car overrides shadowing defaults — this matches `constraints.md`'s existing format.

Clamping happens **twice**:

1. **Pre-optimization:** the optimiser's variable bounds are set from `ConstraintsTable.bounds(car, p)`. The optimiser cannot propose out-of-bounds candidates.
2. **Post-optimization:** the returned setup is re-clamped per parameter as a defensive check; out-of-bounds values trigger a hard error (programmer bug, not a recommendation issue).

CE-gated parameters that are typed in the ontology but unbounded in `constraints.md` are **excluded from the search space**. They fall through from the baseline setup with `Confidence.regime='sparse'` (see §4 graceful degradation, §11 failure handling).

### Trust-radius from confidence

Per §3, `Confidence.regime` informs the search. Specifically: parameters whose fit reports `regime='sparse'` get a narrower per-iteration step size (the optimiser is told to explore conservatively near the baseline value); `regime='dense'` parameters get the full bound range. This is how VISION §6's "be conservative when uncertain, aggressive when confident" reaches the optimiser: not as a separate signal, but as a constraint on how far each parameter is allowed to walk per iteration.

### Output

`SetupRecommendation` per the dataclass in §2:

- `parameters`: every fittable parameter mapped to `(value, Confidence)`. Untyped parameters not present here (they pass through the user's baseline setup unchanged).
- `score_breakdown`: per `(corner_id, phase)` the score contribution. Slice F consumes this to render the "corners helped / hurt" justification.
- `untrained_parameters`: families that had to fall back (CE incomplete, fit failed, etc.).
- `aero_correction_available`: set to slice C's presence.

## 11. Failure handling

Slice E faces two distinct failure surfaces: per-fitter and whole-model.

### Per-fitter failure (singular matrix, all-NaN training data, GP fails to converge, etc.)

| Failure mode | Action |
|---|---|
| Singular feature matrix (e.g., parameter never varied across training set) | Mark the fitter `untrained`; its `predict` raises `UntrainedError`. Do not abort the model build. |
| All-NaN training data after quality masking | Same as above. |
| GP optimiser fails to converge in N iterations | Retry with a stiffer initial length-scale; if still failing, mark `untrained`. |
| Tree ensemble OOM (unlikely at our scale, defensive) | Halve `n_estimators` and retry once; then mark `untrained`. |

`PhysicsModel.recommend` skips untrained fitters during the search — their parameters fall through to a baseline value with `Confidence.regime='sparse'`. Baseline source: the median of training-data values for that parameter. The result lists every untrained parameter in `SetupRecommendation.untrained_parameters` so slice F can flag them in the user-facing output ("we don't have enough data on rear ARB to recommend a value; falling back to the corpus median of 4 clicks").

### Whole-model failure

If `fit` cannot produce any working fitter (e.g., zero clean laps after applying `data_quality_mask`), it raises `InsufficientDataError` with a structured message naming the missing prerequisites. The caller (slice F) renders this to the user with actionable guidance ("ingest more sessions on this car-track pair").

### Untrained vs unbounded vs out-of-ontology

Three distinct states a parameter can be in, exposed differently:

- **Out-of-ontology** (e.g., a cosmetic toggle): never read, never written, never mentioned in output.
- **In-ontology, unbounded by CE:** in `untrained_parameters`, falls through from baseline, no Confidence reported.
- **In-ontology, bounded, fit failed at runtime:** in `untrained_parameters`, falls through from training median, Confidence carries `regime='sparse'`.

## 12. Idempotency / determinism

`fit` is deterministic given the same inputs:

- Random seed pinned globally inside `fit` (single `seed=int` argument with default `0xC0FFEE`).
- Numpy RNG, scikit-learn `random_state`, scipy's `differential_evolution` `seed` all wired from the same root.
- `session_ids` are sorted before fold assignment so caller order does not affect results.
- Phase-weight table and corner-weight derivation are pure functions of inputs.

Re-running `fit(car='bmw', session_ids=[ids], track_model=tm)` twice produces byte-identical model artefacts. Re-running `recommend(track, env, constraints)` twice produces byte-identical `SetupRecommendation`s.

The implementation must include a determinism unit test (§13) that invokes `fit` twice and asserts the produced `PhysicsModel`s pickle to identical bytes.

## 13. Testing

### Unit

- **Synthetic relationship recovery (per fitter family):** generate `output = 2 * spring_rate + noise(σ=0.05)` for 200 samples spanning the constraint range; fit a GP; assert recovered slope at the median ≈ 2 within ±0.1 and `Confidence.regime='dense'`. Repeat with `output = damper_lsc * 0.3 + damper_hsc * 0.7 + noise` for the random forest.
- **Confidence dataclass invariants:** assert `lo <= value <= hi`, regime derivation table coverage, regime ordering (sparse short-circuits noisy).
- **Score-function locality:** modify a fitter for a single corner-phase, assert the per-corner-phase delta in `score_breakdown` matches and other corners are byte-identical.
- **Constraint clamping:** propose an out-of-bounds setup directly to `score_setup`, assert it is internally clamped (or rejected, per chosen contract); assert post-optimisation re-clamping is a no-op when the optimiser respects bounds.
- **Determinism:** §12 — fit twice, assert pickle bytes match.

### Integration

- **Held-out lap residuals (master plan slice E e2e):** `pytest tests/physics/test_fitter.py::test_held_out_lap_residuals`. Fit on N-1 BMW Sebring laps from the canonical fixture, predict the Nth lap, assert per-corner-phase residuals below `signal_std * 0.3` for each output channel and that `Confidence.lo / hi` brackets the observed value for ≥ 90% of held-out predictions. Uses the BMW Sebring fixture from master plan §1.
- **Soft-prereq fall-back:** monkeypatch slice C's `AeroSurface` to raise `ImportError` at fit time, assert `SetupRecommendation.aero_correction_available == False`, high-speed corner Confidence regimes are downgraded one tier, and the model still produces a setup.
- **CE-gated graceful degradation:** with `constraints.md` covering only the today-bounded families, fit on the BMW fixture, assert ARB / damper / corner-weight families appear in `untrained_parameters` and that `recommend` still returns.
- **Empirical-only enforcement:** introduce a deliberate textbook-formula leak in a fixture fitter (e.g., add a hardcoded `f = k*x` transform), assert `physics-fit-validator` flags it. This proves the validator is wired correctly, not just present.

### `physics-fit-validator` subagent gate

The subagent at `.claude/agents/physics-fit-validator.md` is invoked as a CI / pre-merge release gate. Per its workflow (loaded from the agent file):

1. It loads the slice-E-produced `PhysicsModel` artefact and a held-out lap set.
2. It computes residual mean / RMSE / p95 per modeled relationship at corner-phase grain.
3. It cross-checks confidence reporting against data density (sparse coverage must carry lower confidence).
4. It flags fits whose residuals are sign-correct but magnitude-off by a near-constant factor — leaking textbook formulas.

Slice E is **not merged** until the validator's output for the BMW Sebring held-out test passes (no leaking-formula flags, confidence-vs-density alignment within tolerance). The implementation session must include the invocation in the slice's plan (a discrete task, not a buried script).

### Performance budget

Fit on the gold corpus (3 sessions per car-track where 3+ exist; per master plan §1) in **< 60 s per car** on the developer's reference machine. The integration test asserts wall-clock; if the budget is missed, the plan triggers a sparse-GP swap (Nyström approximation) or a smaller K in K-fold CV. Recommendation latency budget: **< 5 s** for `recommend(track, env, constraints)` on a fitted model.

## 14. Out of scope

Explicit non-goals for this slice:

- **Recommendation rendering / explanation.** Slice F. Slice E produces the `SetupRecommendation` dataclass; slice F formats it as engineering-briefing prose (VISION §7) via `racingoptimizer.explain.SetupJustification`.
- **Corner-phase decomposition.** Slice B. Slice E consumes `corner_phase_states` rather than computing them.
- **Aero-map computation.** Slice C. Slice E calls `AeroSurface.interpolate` rather than recomputing aero balance / drag.
- **Track model construction.** Slice D. Slice E consumes `TrackModel` (uses `data_quality_mask` and corner identification) but does not build it.
- **CLI surface.** Slice F. Slice E exposes a Python API only.
- **Constraint expansion.** A discrete pre-spec task (master plan §5) that runs before slice E's plan and implementation. Slice E's spec assumes CE is complete or gracefully degrades when it is not.
- **A unified setup schema across cars.** Each car's ontology is a separate frozen dict (§4). The anti-recommendation in master plan §2 stands.
- **Online / streaming fits.** Refit-on-every-new-session is too costly. The user invokes `fit` explicitly when ready; results are cached. Cache eviction policy is a future concern.

## 15. Open questions / future work

- **GP kernel family.** Matérn-2.5 is the default recommendation; RBF is a fallback when Matérn produces overly smooth fits and a discrete-derivative-aware kernel (rational quadratic) is a candidate when the ARB-clicks-as-continuous approximation breaks. Pin during implementation by sweeping the three on the BMW Sebring fixture and picking the one that minimises calibration error (the 90% bracket-coverage target in §13).
- **`ConstraintsTable` structure.** Whether `constraints.md` is parsed live every call to `recommend` or cached at `fit` time, and whether the parser is a markdown-table reader or expects a side-by-side machine-readable file. Recommendation: parse once at `fit` time and embed in the `PhysicsModel`; live-reload not required for the offline use case. Pin in plan.
- **CE phasing.** Can slice E's implementation start on the bounded subset (heave springs, sliders, ride heights, tyre pressures, wings) before CE fully completes the damper / ARB / diff bounds? Recommendation: **yes**, gated on the §4 graceful degradation. The slice's first-merge target is the bounded subset; CE-gated families slot in as their bounds land. Pin in plan as a phased rollout.
- **Random-forest hyperparameters.** Whether `n_estimators=100` and `max_depth=None` are sufficient for the damper subspace, or whether a per-family hyperparameter sweep is needed. Defer to plan; default is scikit-learn defaults.
- **Bootstrap CI vs GP posterior** for confidence brackets when both are technically available. Current spec uses GP posterior for GP-fit parameters and bootstrap CI for tree ensembles; whether to add bootstrap CI on top of GP posterior for triangulation is a follow-on study.
- **Confidence persistence.** Whether `Confidence` objects round-trip through pickle without losing precision. They should (frozen dataclass of native floats), but a deserialisation test should land in slice E's plan to be sure.
- **Multi-track recommendation.** `recommend` takes one track. Recommending a setup that performs well across two tracks (e.g., a championship round robin) is out of scope here; a future slice can wrap `recommend` with a multi-track aggregator.
- **Driver-style as a feature.** Multiple drivers logging into the same car produce setup-irrelevant variance. Today's corpus is single-driver; if that changes, driver becomes a feature (or a fold-stratification axis). Defer to data reality at fit time.
