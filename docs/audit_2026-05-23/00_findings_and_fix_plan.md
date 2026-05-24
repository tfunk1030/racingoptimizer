# Full audit findings and fix plan — 2026-05-23

Audit goal: physics-based iRacing GTP setup optimizer that recommends per-click
setups from telemetry — **fully accurate, completely built, correlated, and
calibrated**.

Sources: full-code review (2026-05-23), `docs/VISION_ADHERENCE_REPORT.md`,
`docs/physics-rebuild/COMPLETE.md`, `docs/physics-rebuild/daily_12b.md`,
`AGENTS.md` (continual-learning), uncommitted working-tree diff.

**Verdict:** The end-to-end pipeline works for all five GTP cars and ships a
usable recommend path. The bar in the goal statement is **not met** today.
The honest product posture is **guardrailed surrogate optimizer with physics
checks**, not a lap-time-correlated physics predictor.

---

## 1. Executive summary

| Dimension | Status | Notes |
|-----------|--------|-------|
| Ingest → corner-phase → fit → recommend → briefing | **Shipped** | All 5 cars; fast CI smoke (`test_per_car_smoke.py`) |
| Per-corner-phase grain (VISION §2) | **Shipped** | Lap time not in DE objective |
| Physics hybrid in DE (default) | **Shipped** | `use_hybrid = not surrogate_only` → `score.py` → `hybrid_score()` |
| Evaluator Spearman vs corner duration | **Failed gate** | Mean within-group **~0.19**; PLAN target **≥0.35** (fallback **≥0.20**) |
| Per-click explainability (VISION §7) | **Partial** | Default narrative; ±1-click sensitivity behind `--detailed` |
| Full garage optimizable | **Partial** | ~20–25 blocked leaves/car; corner-weight policy contradictory |
| Calibration CI enforcement | **Missing** | `test_calibration_gate.py` accepts script exit 1 |
| User validation bar (`AGENTS.md`) | **High** | `[OPT]` must match iRacing garage; static RH mismatch (~3 mm) = failure |

---

## 2. Findings by severity

### P0 — Critical (trust / correctness)

#### P0-1: Corner weights are contradictory

Three layers disagree on whether corner weights are user-enterable:

| Layer | Position |
|-------|----------|
| `constraints.md` CRITICAL block (lines 7–22) | `CornerWeight` is a **calculated readout** — never recommend |
| `constraints.md` bounds table (lines 222–231) | Legal **target** ranges for all four corners |
| `ontology.py` | `corner_weight_*` → `fittable=True`, `user_settable=True` |
| `full_setup_card.py` | `CornerWeight` in `_CALCULATED_LEAF_NAMES` → `[readout]` |

**Risk:** DE spends dimensions on non-actionable outputs; user cannot apply
`[OPT]` for those leaves. Breaks per-click contract.

**Files:** `constraints.md`, `src/racingoptimizer/physics/ontology.py`,
`src/racingoptimizer/explain/full_setup_card.py`

---

#### P0-2: Evaluator correlation gate failed; CI does not enforce

Offline calibration (`scripts/day_12b_calibrate_evaluator.py`,
`docs/physics-rebuild/daily_12b.md`):

| Car | Weights (util, balance, head) | Within-group Spearman |
|-----|-------------------------------|------------------------|
| BMW | (0.2, 0.8, 0.0) | +0.189 |
| Cadillac | (0.2, 0.3, 0.5) | +0.122 |
| Ferrari | (0.0, 0.0, 1.0) | +0.249 |
| Porsche | (0.0, 0.5, 0.5) | Not documented in evaluator header |
| Acura | Default (0.5, 0.3, 0.2) | Insufficient mid-corner samples |

Mean **~0.187** — below 0.20 fallback. `tests/test_calibration_gate.py`
allows exit code 1 (“does not fail CI hard yet”).

**Risk:** Cannot claim “calibrated” in the statistical sense the project
defined. Hybrid DE runs on weak physics correlation without blocking regression.

---

#### P0-3: Headroom component circular in DE hot loop

Production hybrid path (`score.py`):

```python
surrogate_lat_g_ceiling=lat_g  # lat_g from surrogate accel_lat_g_max
```

For Ferrari/Porsche weights emphasizing headroom, physics scoring partially
compares the surrogate to itself. Day 12b rejected a speed-anchored proxy as
tautological; this is a softer form of the same problem.

**File:** `src/racingoptimizer/physics/score.py` (~line 886)

---

#### P0-4: Static ride height mismatch vs user validation bar

From `AGENTS.md` and prior sessions:

- User validates by applying `[OPT]` in iRacing; static ride height mismatch
  (even ~3 mm) counts as optimizer failure.
- Static `RideHeight` is readout-only; DE scores dynamic
  `*_ride_height_mean_mm` telemetry, not static YAML readouts.
- `constraints.md` static RH rows are **observation envelopes**, not DE
  feasibility constraints — but nothing warns when predicted static RH would
  violate those envelopes.

**Risk:** Recommendations can look correct in briefing while failing the
user’s garage readout check.

**Files:** `constraints.md`, `physics/score.py`, `explain/full_setup_card.py`,
`physics/model.py` (`predict_setup_readouts`)

---

### P1 — High (major VISION / product gaps)

| ID | Finding | Evidence |
|----|---------|----------|
| P1-1 | Default briefing omits VISION §7 ±1-click sensitivity | **Resolved 2026-05-23** — compact sensitivity in `narrative.py` |
| P1-2 | Hybrid DE live without held-out A/B validation | No test: `recommend(hybrid=True)` vs `--surrogate-only` on H1–H5 |
| P1-3 | Mode 1 cross-track confounding structurally open | v4 pools all tracks; Bayes trust anchor only; BMW Spa held-out MAE -7.3% |
| P1-4 | Acura/Porsche lack Day-12b-calibrated evaluator weights | `evaluator.py::_CALIBRATED_WEIGHTS`; Acura uses defaults |
| P1-5 | Per-car recommend excluded from fast CI | **Partially resolved** — `test_per_car_recommend_fast.py`; full DE still `@pytest.mark.slow` |
| P1-6 | Split guardrail paths; `grip_inconsistency` warnings-only | **Resolved 2026-05-23** — quarter penalty in `hybrid_score()`; dual paths documented |
| P1-7 | Docs stale on hybrid wiring | **Resolved 2026-05-23** — `COMPLETE.md`, `CLAUDE.md`, `VISION_COMPLIANCE.md`, etc. |
| P1-8 | Setup card LR-only OPT for rear coil/perch | **Resolved 2026-05-23** — RR `SpringPerchOffset` in `_MIRRORED_LEAVES` |

---

### P2 — Moderate (completeness / validation)

| ID | Finding |
|----|---------|
| P2-1 | ~20–25 garage leaves blocked per car; brake ducts have bounds but no ontology |
| P2-2 | Held-out tests measure fitter residuals, not evaluator Spearman or recommend quality |
| P2-3 | Aero residual correction not passed into DE hybrid evaluator path |
| P2-4 | `diagnostic_state` not consumed in recommend pipeline |
| P2-5 | Wind: magnitude tailwind worst-case only |
| P2-6 | Driver-input channels ~0.50 fit_quality ceiling |
| P2-7 | CI: weekly calibration has no corpus; holdout verify needs LFS IBTs |
| P2-8 | Heave slider 45 mm tech rule not enforced at recommend time |
| P2-9 | JSON pin banners moved to `warnings[]` — contract undocumented |

---

### P3 — Low (docs / polish)

- Stale v3 routing comment (`cli/recommend.py:60–61`; all 5 cars on v4)
- Stale `_status_notes()` (dampers/toe listed as missing)
- `test_clamp_unbounded` always skips — `ClampStatus.unbounded` untested
- `test_hybrid_de_integration.py` misnamed (unit test, not DE)
- Golden files CRLF noise; `Cadillachints.md` Acura copy-paste in diff section

---

## 3. What is solid (do not regress)

- All five GTP cars: ingest, corner, aero, track, fit, CLI smoke
- Modes **2, 3, 4** closed: tyre floor pin, lap-time-weighted samples,
  local-density confidence
- Corner-phase atomic unit; no lap time in DE objective
- Trust envelope, `--explore`, `--reset`, `optimize calibrate`
- Setup card tag contract (`[OPT]`, `[OPT pin]`, `[OPT mirror]`, `[predicted]`)
- Held-out IBT isolation (`held_out` catalog + `verify_holdout.sh`)
- Axle grip ceilings + hybrid guardrail penalties (when ceilings present)
- ~800+ tests including physics-rebuild suite

---

## 4. Fix plan (phased)

Phases ordered by **user validation impact** first, then **trust/correctness**,
then **calibration**, then **completeness**.

### Phase 0 — Policy decisions (1 session, no code until decided)

Resolve contradictions that block implementation:

| Decision | Options | Recommendation |
|----------|---------|----------------|
| Corner weights | (A) readout only — remove from ontology fittable set; (B) user target — remove from CRITICAL block and render `[OPT]` | **(A) readout only** — matches `full_setup_card.py` and iRacing UI |
| Product success criteria | (A) guardrails + surrogate primary; (B) require Spearman ≥0.20 in CI | **(A) for ship**, **(B) as stretch** — document in README/CLAUDE |
| Static RH enforcement | (A) post-DE feasibility check + warning; (B) informational only | **(A)** — matches `AGENTS.md` user bar |

**Exit criteria:** Written decision in this doc’s “Decisions” subsection (below).

---

### Phase 1 — Trust-breaking fixes (1–2 weeks)

| # | Work item | Files | Acceptance |
|---|-----------|-------|------------|
| 1.1 | **Corner-weight policy** — pick readout-only; remove `fittable=True` from `corner_weight_*` OR remove bounds table; align constraints header | `ontology.py`, `constraints.md`, tests | DE dimension count drops by 4; setup card shows `[readout]` only; no `[OPT]` on CornerWeight |
| 1.2 | **Static RH feasibility check** — after DE, run `predict_setup_readouts()`; if static RH outside observation envelope, emit stderr + JSON `warnings` | `cli/recommend.py`, `physics/model.py`, `explain/full_setup_card.py` | Test: pinned setup known to violate envelope → warning present |
| 1.3 | **Fix headroom circularity** — use corpus percentile of observed `accel_lat_g_max` at (corner, phase) as ceiling, not surrogate self-reference | `score.py`, `evaluator.py` | Unit test: headroom varies when surrogate lat-G fixed but setup changes aero |
| 1.4 | **Unify guardrail docs** — update `COMPLETE.md`, `CLAUDE.md`: hybrid default on; dual paths documented | docs | No “deferred blend” claim where code wires `hybrid_score()` |
| 1.5 | **Rear coil/perch mirror** — extend `_MIRRORED_LEAVES` or document LR-only intent | `full_setup_card.py` | Cadillac/BMW rear perch shows `[OPT mirror]` on RR where UI requires symmetry |

**Phase 1 gate:** Run `optimize cadillac lagunaseca` (or user’s fixture IBT);
user applies `[OPT]`; static RH within stated tolerance OR explicit warning
printed.

---

### Phase 2 — Explainability and JSON contract (3–5 days)

| # | Work item | Files | Acceptance |
|---|-----------|-------|------------|
| 2.1 | Add ±1-click sensitivity to default narrative (compact form) OR document `--detailed` as VISION §7 mode in `GETTING_STARTED.md` | `explain/narrative.py`, docs | Each changed param shows sensitivity line or doc states requirement |
| 2.2 | JSON regression: pin banners in `warnings[]` | `tests/cli/test_recommend_cmd.py` | `test_json_pin_warnings_in_payload` green |
| 2.3 | Distinguish untrained vs pinned in briefing header/footer | `cli/recommend.py`, `explain/narrative.py` | User sees separate counts/lists |

---

### Phase 3 — Calibration and validation hardening (1–2 weeks)

| # | Work item | Files | Acceptance |
|---|-----------|-------|------------|
| 3.1 | **Harden calibration gate** — `test_calibration_gate` fails on exit 1 OR pins minimum Spearman floor with `@pytest.mark.calibration` | `tests/test_calibration_gate.py`, CI | Weekly job fails when script fails |
| 3.2 | **CI corpus for weekly job** — cache `corpus/` from `optimize learn ./ibtfiles` or skip with explicit message | `.github/workflows/ci.yml` | Weekly run produces non-empty Day 12b output or documented skip |
| 3.3 | **Held-out recommend A/B** — H1–H5: hybrid vs `--surrogate-only`; define metric (score delta, readout MAE, guardrail count) | `tests/physics/test_hybrid_heldout_ab.py` (new) | Non-regression assertion on chosen metric |
| 3.4 | **Re-run Day 12b for Acura/Porsche** — document Spearman or mark surrogate-primary | `evaluator.py`, `scripts/day_12b_calibrate_evaluator.py` | Comments match script output |
| 3.5 | **Wire or drop `grip_inconsistency` penalty** in `hybrid_score()` | `hybrid_optimizer.py`, `score.py` | Behavior matches docs |
| 3.6 | **Promote one fast per-car recommend smoke** (T2.5) | `tests/physics/test_per_car_recommend.py` | BMW Sebring case in `-m "not slow"` |

**Phase 3 gate:** Calibration script exit 0 on developer machine with full
corpus; CI weekly job meaningful; held-out A/B test green.

---

### Phase 4 — Garage completeness (ongoing, blocked on UI verification)

| # | Work item | Blocker |
|---|-----------|---------|
| 4.1 | Brake duct ontology paths + bounds verification | Per-car YAML path audit |
| 4.2 | Heave slider 45 mm validation in `predict_setup_readouts` | Channel availability in readout predictor |
| 4.3 | Refresh `PARAMETER_VALIDATION.md`, `Cadillachints.md`, `Porschebounds.md` | Manual garage session |
| 4.4 | Ferrari index-based heave/torsion semantics vs N/mm bounds | Ferrari bounds file |
| 4.5 | Directional wind (deferred) | Per-corner heading in corner schedule |

Track in `docs/audit_2026-05-06/99_punch_list.md` Tier 4; extend as items close.

---

### Phase 5 — Uncommitted work to land (immediate)

| Item | Action |
|------|--------|
| `.github/workflows/ci.yml` | Commit; verify LFS + holdout on clean checkout |
| `tests/test_calibration_gate.py` | Commit with Phase 3.1 hardening |
| `tests/physics/test_hybrid_de_integration.py` | Rename or extend to real DE call |
| `tests/constraints/test_clamp.py` | Synthetic unbounded fixture — stop always-skipping |
| `cli/recommend.py` JSON warnings | Commit + Phase 2.2 test |

---

## 5. Suggested execution order (single sprint)

```
Week 1:  Phase 0 decisions → 1.1 corner weights → 1.2 static RH warnings → 1.3 headroom fix
Week 2:  1.4 docs → 1.5 mirror → Phase 2 explainability → commit Phase 5 CI
Week 3:  Phase 3 calibration + held-out A/B + fast recommend smoke
Ongoing: Phase 4 as garage UI data arrives
```

---

## 6. Success metrics

| Metric | Current | Target |
|--------|---------|--------|
| Within-group evaluator Spearman (mean v4) | ~0.19 | ≥0.20 fallback; stretch ≥0.35 |
| User garage readout match (static RH) | Unverified | Warning if outside envelope; user sign-off on 1 car/track |
| Parameters with `[OPT]` in setup card | ~25–30/car | All user-settable bounded params |
| Fast CI per-car recommend | Skipped | 1 case promoted |
| Calibration CI | Soft pass | Hard fail on regression |
| VISION §7 sensitivity in default output | No | Yes (compact) or documented `--detailed` requirement |

---

## 7. Decisions (fill in during Phase 0)

| Question | Decision | Date |
|----------|----------|------|
| Corner weights: readout or target? | **(A) readout only** — `fittable=False`, observation envelope in constraints | 2026-05-23 |
| Formal product mode: guardrails-primary vs correlation-gated? | **(A) guardrails + surrogate primary**; Spearman gate stretch goal | 2026-05-23 |
| Static RH: warn-only vs reject recommendation? | **(A) warn-only** via `_static_ride_height_envelope_warnings` | 2026-05-23 |

---

## 8. Related documents

| Doc | Role |
|-----|------|
| `docs/VISION_ADHERENCE_REPORT.md` | Prior adherence assessment |
| `docs/physics-rebuild/COMPLETE.md` | Physics rebuild honest assessment |
| `docs/physics-rebuild/daily_12b.md` | Evaluator calibration failure detail |
| `docs/audit_2026-05-06/99_punch_list.md` | Open punch-list items |
| `AGENTS.md` | User validation preferences (continual learning) |
| `CLAUDE.md` | Agent operational guide |

---

## 9. Out of scope for this plan

- Replacing surrogate with pure physics (corpus evidence says insufficient)
- Pacejka tire model from telemetry (explicitly vetoed in physics rebuild)
- Lap time as DE objective (violates VISION §6)
- One-off recommendation file tuning (`recommendations/*.txt` artifacts)

---

## 10. Implementation status (2026-05-23)

| Item | Status |
|------|--------|
| 1.1 Corner weights → readout only | **Done** — ontology `fittable=False`; constraints relabeled |
| 1.2 Static RH envelope warnings | **Done** — `_static_ride_height_envelope_warnings` in CLI |
| 1.3 Headroom circularity fix | **Done** — uses `baselines.max_lateral_g` reference |
| 1.4 Docs (hybrid default, v4 all cars, corner weights, static RH) | **Done** — `README.md`, `GETTING_STARTED.md`, `CLAUDE.md`, `VISION_COMPLIANCE.md`, `AGENTS.md`, `COMPLETE.md`, `PARAMETER_VALIDATION.md`, punch list |
| 1.5 Rear spring perch mirror | **Done** — `_MIRRORED_LEAVES` RR SpringPerchOffset |
| 2.1 Compact ±1-click sensitivity in narrative | **Done** |
| 2.3 Untrained vs pinned distinction | **Done** — `_notes_block` |
| 3.1 Calibration gate hardening | **Done** — fails when corpus present |
| 3.5 grip_inconsistency DE penalty | **Done** — quarter penalty in `hybrid_score` |
| 3.6 Fast per-car recommend smoke | **Done** — `test_per_car_recommend_fast.py` |
| 3.3 Held-out hybrid A/B | **Deferred** |
| Phase 4 garage completeness | **Deferred** (UI verification) |
| 2.2 JSON pin warnings regression test | **Deferred** (JSON warnings path already in `recommend.py`) |
