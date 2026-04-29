# Recommendation CLI + Output Rendering — Design Spec

**Date:** 2026-04-28
**Slice:** F — Recommendation CLI + Output Rendering (last of the VISION subsystems)
**Modules:** `racingoptimizer.cli` (extending), `racingoptimizer.explain` (born here)

## 1. Context

`racingoptimizer` is a physics-based setup optimizer for iRacing GTP cars (see `VISION.md`). Slices A–E build the data pipeline, corner-phase decomposer, aero loader, track model, and empirical physics fitter. Slice F is the seam between the engine and the user: it exposes the setup-recommendation surface as a CLI, formats `SetupRecommendation` outputs into human-readable briefings (and JSON), and is gated on output completeness by the `setup-justifier` subagent.

Slice F is **not "the CLI" wholesale.** Master plan (`~/.claude/plans/zazzy-snacking-lighthouse.md` §4 Slice F):

> Each prior slice owns its own subcommand registration. F is **only** the recommendation + compare + status + rendering surface — not "the CLI" wholesale.

Slice A already owns `optimize learn <path>`. Slice F adds three subcommands (`<car> <track>`, `compare`, `status`) and births `racingoptimizer.explain` — the module that carries per-parameter justifications from `PhysicsModel.recommend` into the rendered output.

This slice satisfies VISION §7 ("Justify every click") and §8 ("Simple commands, powerful output") end-to-end, and the matching CLAUDE.md commitments ("Output explains every click", "Confidence is a first-class output"). It does no physics, no fitting, no segmentation, no aero math, no track inference — those are slices B/C/D/E, and F consumes their public APIs.

## 2. Public API

### CLI surface

```text
optimize <car> <track> [--wing N] [--air-temp N] [--track-temp N] [--wind N] [--wetness N] [--json] [--pin KEY=VAL]
    Recommend a setup for the given car-track pair. Auto-detects car/track
    against slice A's catalog when an IBT is omitted. --wing pins the wing
    angle (treated as a constraint, see §6). --air-temp / --track-temp / --wind
    / --wetness override the median environment used in prediction. --json
    swaps the renderer. --pin lets the user freeze any parameter; repeatable.

optimize compare <ibt1> <ibt2> [--json] [--no-auto-learn]
    Diff two setups by per-corner-phase score. Auto-`learn`s either IBT if
    not yet ingested. --no-auto-learn requires both already in the catalog.

optimize status <car> [--json]
    Print what the model knows about a car: per-(car, track) coverage
    counts, fit quality, confidence regime summary.

optimize learn <path>           # already implemented in slice A. NOT redefined here.
```

### Python surface (`racingoptimizer.explain` + `racingoptimizer.cli`)

```python
from racingoptimizer.explain import (
    SetupJustification,
    CornerPhaseImpact,
    SetupComparison,
    ModelStatus,
    render_recommendation,
    render_recommendation_json,
)
from racingoptimizer.cli.recommend import compare_setups, model_status

def render_recommendation(rec: SetupRecommendation) -> str:
    """Human-text renderer. Multi-section briefing per §4 layout."""

def render_recommendation_json(rec: SetupRecommendation) -> dict:
    """Machine-readable renderer; mirrors §4 fields exactly."""

def compare_setups(ibt_a: Path, ibt_b: Path, *, auto_learn: bool = True) -> SetupComparison:
    """Score both setups with the same PhysicsModel; emit per-corner-phase deltas."""

def model_status(car: str) -> ModelStatus:
    """Coverage map for one car: sessions, laps, tracks, fit quality, regimes."""
```

`SetupRecommendation` and `PhysicsModel` are slice E exports. F imports them; F does not redefine them.

## 3. `SetupJustification` contract

Per master plan §2 minimum contract, this slice births `racingoptimizer.explain`. The dataclass shape below is **the public contract**; slice E populates it inside `PhysicsModel.recommend`, slice F renders it. A future subclass / extension is allowed only if every field below remains.

```python
from dataclasses import dataclass
from racingoptimizer.confidence import Confidence
from racingoptimizer.corner import Phase

@dataclass(frozen=True)
class CornerPhaseImpact:
    corner_id: int          # 1-indexed corner number from slice B / D
    phase: Phase            # BRAKING | TRAIL_BRAKE | MID_CORNER | EXIT | STRAIGHT
    score_delta: float      # this parameter's contribution to total score at
                            # this (corner, phase). Positive = helps; negative = hurts.
    note: str               # short physics rationale, e.g. "understeer reduced 0.4°"

@dataclass(frozen=True)
class SetupJustification:
    parameter: str                              # e.g. "front_heave_spring_rate_n_per_mm"
    value: float
    unit: str                                   # e.g. "N/mm", "kPa", "mm", "°"
    confidence: Confidence                      # from slice E (regime, n_samples, lo/hi)
    corners_helped: list[CornerPhaseImpact]     # non-empty iff helped score
    corners_hurt:   list[CornerPhaseImpact]     # non-empty iff cost score
    sensitivity_minus_1_click: float            # predicted total-score delta at -1 click
    sensitivity_plus_1_click:  float            # predicted total-score delta at +1 click
    telemetry_evidence: list[str]               # short bullets, channel + condition + value
                                                # e.g. "T1 trail-brake understeer reduced 0.4°"
                                                # also: warning lines from §5 clamping go here
```

The four required `setup-justifier` criteria (`.claude/agents/setup-justifier.md`) map onto this dataclass:

| `setup-justifier` requirement | Field |
|---|---|
| Corners that benefit (named corner-phase + physics) | `corners_helped` (list of `CornerPhaseImpact`, with `note`) |
| Corners that compromise | `corners_hurt` |
| Telemetry evidence (channel + condition) | `telemetry_evidence` |
| Sensitivity at ±1–2 clicks | `sensitivity_minus_1_click`, `sensitivity_plus_1_click` |
| Confidence reported and consistent with density | `confidence` |
| Value inside `constraints.md` bounds | enforced upstream of render — see §5 |

If any field above is missing or empty for a parameter, the renderer raises `IncompleteJustificationError` before any bytes hit stdout. Empty-list emptiness for `corners_helped` / `corners_hurt` is allowed only if the *other* list is non-empty (a parameter that strictly helps is fine; a parameter that helps and hurts nothing is invalid because then it would not be a recommendation, it would be a default).

```python
@dataclass(frozen=True)
class SetupComparison:
    car: str
    track: str
    setup_a_id: str          # session_id of ibt_a
    setup_b_id: str          # session_id of ibt_b
    total_score_a: float
    total_score_b: float
    per_corner_phase: list["CornerPhaseDelta"]   # per (corner, phase) score delta
    notes: list[str]         # warnings: missing setup keys, mismatched cars/tracks, etc.

@dataclass(frozen=True)
class CornerPhaseDelta:
    corner_id: int
    phase: Phase
    score_a: float
    score_b: float
    delta: float             # b - a
    drivers: list[str]       # short bullets: which parameters / channels drove the delta

@dataclass(frozen=True)
class ModelStatus:
    car: str
    coverage: list["TrackCoverage"]   # per-track summary
    overall_regime: str               # 'sparse' | 'dense' | 'confident' | 'noisy'
    notes: list[str]                  # CE-incomplete params, missing aero wing angles, etc.

@dataclass(frozen=True)
class TrackCoverage:
    track: str
    n_sessions: int
    n_valid_laps: int
    n_clean_corner_phases: int        # post-quality-mask
    fit_quality: float | None         # cross-validation residual std from slice E, None if unfit
    regime: str                       # 'sparse' | 'dense' | 'confident' | 'noisy'
```

## 4. Output format

### Human-text layout (default)

```
<car> @ <track> — recommended setup
Conditions: AirTemp 24.0 °C  TrackTemp 32.5 °C  Wind 3.0 m/s  Wetness 0.0
Confidence: dense (n=187 corner-phase samples across 4 sessions, fit residual σ=0.08)
Pinned by user: rear_wing_angle_deg = 17.0

Front Heave Spring: 45.0 N/mm   [confidence: dense]
    +1 click: -0.18 score    -1 click: +0.04 score
    Helps: T1-MID_CORNER (-0.5° understeer, score +0.21)
           T7-EXIT (+1.2% traction utilization, score +0.09)
    Hurts: T3-BRAKING (+8 mm bottoming risk, score -0.04)
    Evidence:
      - shock velocity p99 = 220 mm/s in T3-BRAKING
      - aero balance shifts +1.4% rearward at high speed
      - 142 clean samples backing this fit

Rear Wing Angle: 17.0°   [confidence: pinned by user]
    (no sensitivity reported — pinned)
    Helps: T6-MID_CORNER (+2.1% downforce, score +0.31)
    Hurts: straight-section drag (score -0.12)
    Evidence:
      - user override; recommend would have chosen 14.0°
      - aero map L/D ratio 4.2 at this configuration

...

Warnings:
  - front_arb_blade_position: skipped (constraint bounds TODO in constraints.md)
```

Layout rules:

- Header: `<car> @ <track> — recommended setup`. Title-case track display; lowercase slug stays internal.
- Conditions line: render the `EnvironmentFrame` used for prediction (median of training data unless overridden by `--air-temp` / `--track-temp` / `--wind` / `--wetness`).
- Top-level Confidence line: roll-up across all parameter confidences. The renderer picks the *least confident* regime tag (sparse > noisy > dense > confident, in order of caution) and reports the dominant `n_samples`.
- Per-parameter block: parameter name in human form (`Front Heave Spring`, not `front_heave_spring_rate_n_per_mm`), value + unit, regime tag, ±1-click line, helps list, hurts list, evidence list.
- "Pinned by user" parameters omit the ±1-click line and instead show a `(no sensitivity reported — pinned)` placeholder.
- Warnings block at the end: clamping events, skipped parameters (CE-incomplete), and any `notes` from `SetupRecommendation`.

Order of parameters: most-impactful first (highest `abs(sum(corners_helped.score_delta) - sum(corners_hurt.score_delta))`). User-pinned values float to the top of their tier.

### JSON layout (`--json`)

Mirror the dataclass fields one-to-one. Snake-case keys. Floats not rounded; let downstream consumers format. No trailing-comma fragility — emit valid JSON via `json.dumps(..., indent=2, sort_keys=False)`.

```json
{
  "car": "bmw",
  "track": "sebring_international",
  "environment": {
    "air_temp_c": 24.0,
    "track_temp_c": 32.5,
    "wind_vel_ms": 3.0,
    "wind_dir_deg": 90.0,
    "wetness": 0.0,
    "air_density_kg_m3": 1.205
  },
  "confidence": {
    "regime": "dense",
    "n_samples": 187,
    "n_sessions": 4,
    "fit_residual_sigma": 0.08
  },
  "pinned": {"rear_wing_angle_deg": 17.0},
  "parameters": [
    {
      "parameter": "front_heave_spring_rate_n_per_mm",
      "value": 45.0,
      "unit": "N/mm",
      "confidence": {"regime": "dense", "n_samples": 142, "lo": 42.0, "hi": 48.0},
      "sensitivity_minus_1_click": 0.04,
      "sensitivity_plus_1_click": -0.18,
      "corners_helped": [
        {"corner_id": 1, "phase": "MID_CORNER", "score_delta": 0.21, "note": "understeer reduced 0.5°"},
        {"corner_id": 7, "phase": "EXIT", "score_delta": 0.09, "note": "traction utilization +1.2%"}
      ],
      "corners_hurt": [
        {"corner_id": 3, "phase": "BRAKING", "score_delta": -0.04, "note": "bottoming risk +8 mm"}
      ],
      "telemetry_evidence": [
        "shock velocity p99 = 220 mm/s in T3-BRAKING",
        "aero balance shifts +1.4% rearward at high speed",
        "142 clean samples backing this fit"
      ]
    }
  ],
  "warnings": [
    "front_arb_blade_position skipped: constraint bounds TODO in constraints.md"
  ]
}
```

## 5. Constraint clamping

Every recommended value passes through `racingoptimizer.constraints.clamp(value, parameter, car) -> (clamped: float, was_clamped: bool, bound: tuple[float, float] | None)` before it reaches the renderer.

```python
clamped, was_clamped, bound = clamp(rec.value, rec.parameter, car)
if was_clamped:
    rec = replace(rec,
        value=clamped,
        telemetry_evidence=rec.telemetry_evidence + [
            f"value clamped from {rec.value} to {clamped} (legal bounds: {bound[0]}–{bound[1]})"
        ],
    )
```

The clamping is **the renderer's last preflight**, not the fitter's responsibility. Slice E may emit a value outside the legal envelope (it is fitting empirically, not enforcing legality); slice F clamps and warns. The warning becomes an entry in `SetupJustification.telemetry_evidence` so the `setup-justifier` subagent sees it inline with the parameter.

The clamp implementation lives in `racingoptimizer.constraints` (a tiny module that loads `constraints.md`, parses defaults + per-car overrides, and answers bounds queries). This module is **born here** if not already present from CE work; otherwise extended.

`constraints.md` `<TODO: from iRacing UI>` placeholders (CE incomplete) are handled per §13: the parameter is **skipped from the recommendation entirely**, with a warning line appended to the top-level `warnings` list. The renderer never emits a value for a parameter it cannot legally bound.

Master plan §5 (CE prereq) is the source of truth for which parameters CE must fill before slice E's spec; until then, slice F skips them gracefully.

## 6. Auto-detection

When the user runs `optimize bmw sebring`:

1. CLI normalizes `bmw` against the canonical car keys (`acura, bmw, cadillac, ferrari, porsche` per slice A `detect.py`).
2. CLI normalizes `sebring` against existing track slugs in slice A's catalog (`sebring` → `sebring_international` via prefix/substring match; ambiguous matches list candidates and exit 2).
3. CLI calls `sessions(car='bmw', track='sebring_international')` from slice A; if zero rows → exit with "model has no data on (bmw, sebring_international); run `optimize learn <ibt>` first" message.
4. Otherwise: select all valid sessions on (car, track), build (or reuse cached) `PhysicsModel` via slice E's `fit(...)`, derive median `EnvironmentFrame` from training data (overridable by CLI flags), call `PhysicsModel.recommend(...)`, render.

When the user runs `optimize compare a.ibt b.ibt`:

1. For each IBT path, hash the bytes (slice A's `session_id`); query catalog.
2. If either hash is absent and `--no-auto-learn` is not set → call `learn(path)` for the missing IBTs.
3. Reject if cars or tracks differ between A and B (write a `notes` entry, exit 2).
4. Fit `PhysicsModel` for the (car, track) pair (or reuse cache).
5. Score both setups via `PhysicsModel.score_setup(...)`, build `SetupComparison`, render.

When the user runs `optimize status bmw`:

1. Query `sessions(car='bmw')`; group by track.
2. For each track group: count valid laps, count post-quality-mask corner-phases (this requires slice D for the mask; if D absent, fall back to all-valid-mask), call slice E's coverage / cv-residual API to get `fit_quality` and `regime`.
3. Roll up `overall_regime`. Render.

### Caching

Fitted `PhysicsModel` objects are cached at `corpus/models/<car>__<track>__<sorted-session-ids-sha>.pickle`. Cache key is the sha256 of the sorted `session_id` list — when the catalog gains a new session, the key changes, the cache misses, and the model refits. Cache is invalidated on `optimize learn` if the new session matches the (car, track) of any cached model. Slice E owns serialization; slice F owns the file path layout under `corpus/models/`.

## 7. `status` subcommand

Coverage map per (car, track) pair:

```
bmw — coverage report
─────────────────────────────────────────────────────────────────────────
Track                       Sessions  Valid laps  Clean CP  Fit σ   Regime
sebring_international              4         126     11,832   0.08   dense
hockenheim_gp                      6         203     19,471   0.06   dense
algarve_gp                         2          47      4,201   0.18   sparse
laguna_seca                        1          12      1,008      —   sparse
─────────────────────────────────────────────────────────────────────────
Overall regime: dense (weighted by n_clean_corner_phases)

Notes:
  - laguna_seca: only 1 session — recommendations will be conservative.
  - constraints.md is missing bounds for: ARBs, dampers, corner_weights,
    brake_bias, differential, camber, toe, brake_ducts.
```

JSON variant maps to the `ModelStatus` dataclass in §3.

`status` is more useful when slice D has run (the `n_clean_corner_phases` column meaningfully reflects the mask). Without D, every sample is clean, and `n_clean_corner_phases` is just `n_valid_laps × mean_corner_phases_per_lap`. Document this in the `--help` output.

## 8. Subcommand registration pattern

`racingoptimizer.cli` is a `click` (or `typer`) group composed by each slice that contributes commands. Slice A registers `learn`. Slice F registers `recommend` (the `<car> <track>` default), `compare`, `status`. Future slices can add their own.

```python
# src/racingoptimizer/cli/__init__.py
import click
from racingoptimizer.ingest.cli import register_ingest_commands
from racingoptimizer.cli.recommend import register_recommend_commands

@click.group(invoke_without_command=False)
def optimize():
    """Racing setup optimizer."""

register_ingest_commands(optimize)     # adds: learn  (slice A)
register_recommend_commands(optimize)  # adds: recommend (default), compare, status  (slice F)

# pyproject.toml entry point: optimize = "racingoptimizer.cli:optimize"
```

```python
# src/racingoptimizer/cli/recommend.py
import click

def register_recommend_commands(group: click.Group) -> None:
    group.add_command(_recommend_cmd)
    group.add_command(_compare_cmd)
    group.add_command(_status_cmd)
```

The `<car> <track>` default-recommend command is registered as `recommend` internally and as the root invocation handler — `optimize bmw sebring` dispatches to `recommend(car='bmw', track='sebring')` via a `click.pass_context` shim that tests `args[0]` against known car keys before falling through to subcommand dispatch. Pin this shim in the implementation; the spec only requires "users type `optimize bmw sebring` and it works."

The pattern is documented so slice B/C/D/E can register their own commands later (e.g. `optimize segment <ibt> <lap>`, `optimize aero <car> --rh-front N`, `optimize track <track>`, `optimize fit <car> <track>`) without touching slice F's code.

## 9. `setup-justifier` subagent gating

`.claude/agents/setup-justifier.md` defines the gatekeeper: it walks every recommended parameter and verifies four required justifications (corners-that-benefit, corners-that-compromise, telemetry evidence, sensitivity at ±1–2 clicks) plus two cross-checks (value inside `constraints.md` bounds, confidence reported and consistent with data density).

Slice F's e2e test invokes the subagent against rendered output:

```python
# tests/cli/test_e2e.py
def test_recommendation_satisfies_setup_justifier(tmp_path, bmw_sebring_fixture):
    # 1. learn the fixture
    runner.invoke(optimize, ['learn', str(bmw_sebring_fixture)])

    # 2. recommend
    out_path = tmp_path / 'out.txt'
    result = runner.invoke(optimize, ['bmw', 'sebring', '--wing', '17'])
    out_path.write_text(result.output)

    # 3. assertions BEFORE subagent
    assert result.exit_code == 0
    assert _every_param_has_required_fields(result.output)   # local check
    assert _every_value_inside_constraints(result.output)    # local check

    # 4. subagent gate
    verdict = run_subagent('setup-justifier', input_path=out_path)
    assert verdict.passes, verdict.failure_table
```

`run_subagent('setup-justifier', input_path=...)` shells out to whatever Claude Code subagent invocation harness is wired up (tests skip with `pytest.skip` if the harness is unavailable in CI; the local-check assertions remain). The subagent rejects bare-list-of-numbers outputs — the renderer in §4 is structured to never emit one.

## 10. Failure handling

| Failure mode | Behaviour |
|---|---|
| Unknown car (not in 5 canonical keys) | exit 2, message: "unknown car '<x>'; expected one of acura, bmw, cadillac, ferrari, porsche" |
| Track slug ambiguous | exit 2, list candidate slugs |
| Track slug unknown to catalog | exit 2, message + suggestion to `optimize learn <ibt>` |
| Model untrained on (car, track) — zero sessions | exit 0; emit "untrained recommendation" using **median of training-data values per parameter across other tracks for the same car**, every parameter `Confidence.regime = 'sparse'`, every parameter's `corners_helped` / `corners_hurt` lists are populated from the most-similar fitted track's `SetupRecommendation` with a `note: "extrapolated from <other_track>"`, and a top-level warning: `untrained on (<car>, <track>); recommendations are conservative defaults` |
| `compare`: cars / tracks differ between A and B | exit 2, message + per-IBT car/track in the output |
| `compare`: one or both IBTs unhashable | exit 2, message names the offending file |
| Renderer-detected missing justification field | `IncompleteJustificationError` → exit 3 with the parameter + missing-field name |
| Constraint clamp `<TODO>` placeholder hit | parameter skipped, top-level warning emitted, exit 0 |
| Constraint clamp out-of-bounds | value clamped, evidence line appended, exit 0 |
| `--pin KEY=VAL` references unknown parameter | exit 2, list known parameter keys |
| Filesystem missing (`corpus/` does not exist) | exit 4, suggestion to `optimize learn <path>` first |

Untrained-recommendation format mirrors §4 with three deltas: a `(untrained — sparse defaults)` suffix on the header, every parameter forced to `regime: sparse`, and each `CornerPhaseImpact.note` prefixed with `extrapolated from <other_track>`. The top-level warning is `untrained on (<car>, <track>); recommendations are conservative defaults`.

## 11. Idempotency / determinism

Same `(car, track, environment, model_state, pinned_overrides)` input yields the same rendered output, byte-for-byte, modulo a single timestamped header line which is opt-out via `--no-timestamp`.

Determinism caveats:

- **Slice E may be non-deterministic** without a fixed seed (GP / bootstrap CI / tree ensemble all have stochastic components). F inherits this. F does not introduce its own randomness; if E pins a seed (recommended), F's outputs are byte-stable.
- The model cache key (sorted-session-ids sha) keeps the same training set producing the same cached model.
- Float formatting in the renderer: pin the format strings — values to one decimal place, scores to two, sensitivities to two — so trivial display drift does not break golden-file tests.
- JSON output never re-orders fields; `sort_keys=False` enforced.

## 12. Testing

### Unit tests

- `test_explain.py` — `SetupJustification` field validation (raises on empty `corners_helped` AND `corners_hurt`; raises on missing `confidence`; etc.). `CornerPhaseImpact` ordering by abs(score_delta).
- `test_render_text.py` — golden-file test on a hand-built `SetupRecommendation` with three parameters; assert exact string match against `tests/cli/golden/bmw_sebring.txt`.
- `test_render_json.py` — same `SetupRecommendation`, assert `json.loads(render_json(rec))` round-trips and matches `tests/cli/golden/bmw_sebring.json`.
- `test_clamp.py` — value above max → clamped + evidence line; value below min → clamped + evidence line; CE-TODO parameter → skipped + warning.
- `test_status.py` — synthetic catalog with 3 (car, track) groups → expected `ModelStatus` shape.
- `test_pin.py` — `--pin rear_wing_angle_deg=17.0` propagates as a constraint into `PhysicsModel.recommend`; output renders `pinned by user` and omits ±1-click for that param.

### Integration tests

- **Untrained path** — fresh `corpus/`, run `optimize bmw daytona_2011_road`, assert exit 0, output contains "untrained" warning, every parameter has `regime: sparse`.
- **Trained path (the e2e from master plan §4)** — ingest BMW Sebring fixture (`ibtfiles/bmwlmdh_sebring international 2026-03-22 14-47-42.ibt`, master plan §1 canonical fixture), run `optimize bmw sebring --wing 17 > out.txt`, assert:
  - exit code 0
  - stdout contains every required `SetupJustification` field for every parameter
  - every value lies inside `constraints.md` bounds for car=bmw
  - `setup-justifier` subagent invocation passes
- **`--json` parity** — same trained path with `--json`, assert `json.loads(stdout)` succeeds and contains the same fields as the human-text version.
- **`compare` over two real IBT fixtures** — compare BMW Sebring fixture against itself (degenerate case, all deltas zero); compare BMW Sebring against a synthetic-modified copy with one parameter changed, assert per-corner-phase delta breakdown is non-empty and the changed parameter shows up in `drivers`.
- **`status` against canonical 5 cars** — run `optimize status acura`, `optimize status bmw`, `optimize status cadillac`, `optimize status ferrari`, `optimize status porsche`. Each must exit 0 and emit a sensible regime tag (or `sparse` when only one fixture is ingested).

### Fixtures (master plan §1 canonical set)

| Test | Fixture |
|---|---|
| Trained-path e2e | `ibtfiles/bmwlmdh_sebring international 2026-03-22 14-47-42.ibt` |
| `compare` self vs self | same as above |
| `compare` cross-car (negative test) | acura fixture vs porsche fixture; expect exit 2 |
| `status` per car | one canonical fixture per car from master plan §1 table |

Goldens live under `tests/cli/golden/`. Regenerate by running `pytest --regen-goldens` (a one-flag override in `conftest.py`); review diffs before committing.

### Subagent gating in CI

The `setup-justifier` subagent is gated locally and on CI when the Claude Code harness is available. CI pipelines without the harness skip the subagent step but still run the local-check assertions (`_every_param_has_required_fields`, `_every_value_inside_constraints`); these enforce the same contract structurally even in absence of the LLM gatekeeper. The subagent is the higher-quality check and runs locally on every PR.

## 13. Out of scope

- **Physics fitting** — slice E. F never fits, never opens an aero map, never computes a score.
- **Aero-map computation** — slice C. F reads `AeroSurface` only via slice E's `PhysicsModel`.
- **Corner segmentation** — slice B. F renders `CornerPhaseKey` outputs but never decomposes a lap.
- **Track model** — slice D. F consumes `data_quality_mask` indirectly via slice E; F never builds a curb / bump / grip map.
- **IBT parsing / catalog writes** — slice A. F calls `learn(...)` but never re-implements parsing.
- **Constraint expansion (CE)** — pre-slice-E task per master plan §5. F handles missing parameters gracefully (§5 of this spec) but does not author bounds.
- **Setup ontology** — slice E owns it (master plan §2). F reads parameter names; F does not enumerate them.
- **Multi-driver / lobby comparison features** — out of slice F. CLI is single-user.
- **GUI / web UI / dashboard** — out of slice F. CLI only.
- **Setup file export to iRacing** — out. (Future slice could add `optimize export --format irsdk`.)
- **Live telemetry streaming** — out. We work on persisted `.ibt` files only.

## 14. Open questions

- **`compare` on raw IBTs.** Default: auto-`learn` first for UX (drop in two IBTs, get a diff). Future flag `--no-auto-learn` requires both already in catalog. **Pinned: auto-learn by default.**
- **User pin vs model preference.** When the user supplies `--wing 17` but the model thinks 14 is optimal, the recommendation **respects the user's pin** as a constraint and recomputes the rest of the setup around it. The pinned parameter's block reports `confidence: pinned by user` and the evidence list includes a note `recommend would have chosen <model_value>`. **Pinned: user pin wins; model recomputes everything else.**
- **CE placeholders.** When `constraints.md` has a `<TODO: from iRacing UI>` entry for a parameter, **skip recommendations for that parameter entirely** and emit a top-level warning. Do not guess bounds. Do not emit an unbounded value. **Pinned: skip + warn.**
- **Confidence regime roll-up function.** The top-level confidence on the briefing is the *least confident* per-parameter regime, weighted by `n_samples`. **Pinned: min-regime by sample-density rule.** Future spec may revisit if users complain that a single sparse parameter masks an otherwise dense fit.
- **Track-slug ambiguity.** `optimize bmw sebring` matches `sebring_international` unambiguously today; if iRacing later adds `sebring_short`, the matcher returns both candidates and exits with a chooser. **Pinned: substring match → unique → use; non-unique → exit 2 with candidates.**
- **Cache eviction.** Model pickles in `corpus/models/` accumulate. No eviction policy in slice F. **Open**: future cleanup task or LRU-by-mtime in a follow-up.
- **`--pin` semantics for non-numeric parameters.** Categorical parameters (e.g. tyre compound) are out of slice F's initial scope; CE will reveal whether any setup parameter is categorical. **Open**: revisit when CE lands.
- **Untrained extrapolation source selection.** When (car, track) is untrained, F extrapolates from the "most similar" trained track. The similarity metric (corner-count proximity? speed-profile correlation?) is not defined. **Open: defer to slice E**; F just calls a slice-E API like `PhysicsModel.nearest_trained_track(car, track)`.

## 15. Module layout

```
src/racingoptimizer/
  cli/
    __init__.py            # composes the click group; registers all slices' commands
    recommend.py           # slice F: <car> <track>, compare, status command implementations
  explain/
    __init__.py            # public re-exports
    justification.py       # SetupJustification, CornerPhaseImpact dataclasses + validators
    comparison.py          # SetupComparison, CornerPhaseDelta dataclasses
    status.py              # ModelStatus, TrackCoverage dataclasses
    render_text.py         # render_recommendation, _render_status_text, _render_compare_text
    render_json.py         # render_recommendation_json, _render_status_json, _render_compare_json
  constraints/             # may already exist from CE work; F extends if needed
    __init__.py
    loader.py              # parse constraints.md → bounds table
    clamp.py               # clamp(value, parameter, car) -> (clamped, was_clamped, bound)
tests/
  cli/
    test_e2e.py            # the master-plan §4 e2e + setup-justifier gate
    test_recommend_cmd.py  # CLI-level recommend command unit tests
    test_compare_cmd.py
    test_status_cmd.py
    test_render_text.py
    test_render_json.py
    test_pin.py
    test_clamp.py
    test_untrained.py
    golden/
      bmw_sebring.txt
      bmw_sebring.json
      bmw_sebring_untrained_daytona.txt
```

## 16. Success criteria

This slice is done when the §12 test suite passes, the `setup-justifier` subagent passes against the trained-path BMW Sebring output, and `optimize learn <ibt>` (slice A) continues to work unmodified.
