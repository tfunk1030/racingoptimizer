# PLAN: 2-week empirical-fixes-first + scoped-physics rebuild

**Plan-of-record.** Authored 2026-05-08. Synthesized from a 6-agent
adversarial review of an earlier "physics-first" proposal (see
`docs/physics-rebuild/REVIEWS_2026-05-08.md` for raw findings; this
plan is the conclusion).

After authorization (Section 17), this file is **immutable**. Every
loop iteration, the agent re-reads PLAN.md and asserts the invariants
in Section 4 verbatim. If a numeric, file path, or rule below appears
to drift in implementation, the implementation is wrong, not the plan.

> Status: DRAFT, awaiting user sign-off (see Section 17).

---

## Table of contents

1. [Background and current state](#1-background-and-current-state)
2. [The Five Failure Modes (with evidence)](#2-the-five-failure-modes-with-evidence)
3. [Why the original "physics-first" proposal was wrong-shape](#3-why-the-original-physics-first-proposal-was-wrong-shape)
4. [Hard rules (DO and DO-NOT)](#4-hard-rules-do-and-do-not)
5. [Permissions matrix](#5-permissions-matrix)
6. [Branch / tag / commit conventions](#6-branch--tag--commit-conventions)
7. [Held-out validation set](#7-held-out-validation-set)
8. [Daily workflow](#8-daily-workflow)
9. [Token / wall-clock budget](#9-token--wall-clock-budget)
10. [External-judge protocol](#10-external-judge-protocol)
11. [Stop-and-wait checkpoints](#11-stop-and-wait-checkpoints)
12. [Fallback-mode discipline](#12-fallback-mode-discipline)
13. [Failure recovery](#13-failure-recovery)
14. [Week 1 plan -- empirical fixes (Days 1-7)](#14-week-1-plan--empirical-fixes-days-1-7)
15. [Week 2 plan -- scoped physics (Days 8-14)](#15-week-2-plan--scoped-physics-days-8-14)
16. [Hand-off / completion criteria](#16-hand-off--completion-criteria)
17. [Sign-off (user authorizes here)](#17-sign-off-user-authorizes-here)
18. [Appendix A -- file paths and key functions](#appendix-a--file-paths-and-key-functions)
19. [Appendix B -- review-agent findings (one-liner each)](#appendix-b--review-agent-findings-one-liner-each)
20. [Appendix C -- glossary](#appendix-c--glossary)

---

## 1. Background and current state

`racingoptimizer` is the `optimize` Python CLI for iRacing GTP setup
recommendations. Five cars covered: BMW M Hybrid V8, Cadillac V-Series.R,
Acura ARX-06, Ferrari 499P, Porsche 963. Per-car suspension architectures
differ; setup schemas are per-car.

Today's recommend pipeline routes by `PER_CAR_MODEL_CARS`
(`src/racingoptimizer/cli/recommend.py`):

- **v4 (per-car, track-agnostic)** -- BMW, Cadillac, Ferrari. Pools every
  session for the car across every track into one Forest fitter; cache
  at `corpus/models/<car>__per-car__<digest>.pickle`.
- **v3 (per-(car, track))** -- Acura, Porsche. Trains per pair; donor-
  track extrapolation when target unseen.

Per VISION.md: corner-phase is the atomic unit; physics is empirical
not formula-based; lap time is an outcome not the optimization signal;
every data point carries environmental context; every recommendation
carries confidence.

Six VISION slices (A-F) plus three cross-cutting modules (`context`,
`confidence`, `constraints`) are merged. Per-car verification matrix
is documented in `CLAUDE.md` (Active build section). The May-2026
audit (`docs/audit_2026-05-06/00_summary.md`, `99_punch_list.md`)
ranked remaining gaps; this plan's Week 1 closes 4 of the top 5
unsolved user-visible problems.

---

## 2. The Five Failure Modes (with evidence)

Each mode lists: definition; verdict from review; concrete evidence
(file:line or artefact); cheapest possible fix; whether a physics
rebuild fixes it.

### Mode 1 -- Cross-track confounding

A parameter held constant on a high-sample-count track (e.g.
Hockenheim wing=17 across 24 Ferrari sessions) drags recommendations
on a low-sample-count track (Spa wing=14-15 across 6 sessions),
because the joint surrogate is dragged by sample weight regardless
of corner-archetype features.

- **Verdict**: SUPPORTED.
- **Evidence**: `CLAUDE.md` lines 112-118 documents the mechanism
  verbatim. `physics/fitter.py:734-747` chose Forest-only because
  GP collapsed under cross-track variance.
  `physics/model.py:150` `per_track_parameter_observed` exists but
  is consumed only as a *trust envelope*, not as a feature
  interaction.
- **Cheapest fix**: 2-4 person-days. Hierarchical Bayesian regression
  with track / car / session as random effects, OR per-track residual
  Forest for the high-variance ~10 parameters with sample-weight
  rebalancing. Stays inside VISION SS3's "empirical, not textbook
  formulas" mandate.
- **Physics rebuild fixes**: NO -- a physics model still needs
  parameters fit from data; unless every parameter gets a
  from-physics prior, the same data-imbalance confound returns.

### Mode 2 -- Tyre pressure misranking

Surrogate rewards platform stability (cleaner ride-height telemetry,
which higher cold pressure delivers) but cannot see the peak-grip
drop from smaller contact patch. Optimizer recommends pressures
above the community-known floor.

- **Verdict**: SUPPORTED.
- **Evidence**: `recommendations/bmw-spa-race-0507-2255.txt` recommends
  163.5 kPa, `0507-2301.txt` 160.5 kPa, `0507-2308.txt` 154.0 kPa.
  Floor per community wisdom (and `constraints.md` per-car overrides
  capped at 152). `CLAUDE.md` "When to override the optimizer"
  documents this disconnect.
- **Cheapest fix**: 0.5 person-day. In `cli/recommend.py`, after
  `_apply_pins_to_constraints`, if `tyre_cold_pressure_kpa` is
  fittable AND user did not pass `--pin tyre_cold_pressure_kpa=`,
  force-pin to the per-car constraint floor.
- **Physics rebuild fixes**: YES, but a 0.5-day pin closes the
  user-visible gap.

### Mode 3 -- Driver-bias inheritance

Conservative recent stints pull recommendations conservative because
`baseline_setup` and `parameter_observed_std` are corpus medians /
pstdevs without lap-time weighting.

- **Verdict**: WEAKLY SUPPORTED (mechanism plausible; no quantitative
  artefact found).
- **Evidence**: `CLAUDE.md` "When to override the optimizer" notes the
  pattern; no `recommendations/` artefact directly demonstrates it.
- **Cheapest fix**: 1-2 person-days. Lap-time-weighted sample
  reweighting in the fitter -- weight rows by
  `1 / (lap_time - track_min_lap_time + epsilon)`. Pure feature
  engineering, no architecture change.
- **Physics rebuild fixes**: NO -- a physics model also inherits
  whatever sample-weighting choice is made.

### Mode 4 -- "Confidence reads dense" while extrapolating

Confidence regime label (sparse / noisy / confident / dense) is
computed from global model statistics; it does not check parameter-
specific or value-specific corpus density. A polluted corpus reports
"dense" while recommending values that have zero physics anchor.

- **Verdict**: SUPPORTED.
- **Evidence**: `confidence/confidence.py:60-92` -- `Confidence.derive`
  uses (n_samples, cv_residual_std, signal_std), all global. No per-
  value density check. `CLAUDE.md` lines 108-110 documents this.
- **Cheapest fix**: 1 person-day. When `_pin_or_trust_bounds` opens
  the trust radius beyond the empirical envelope (the `--explore` /
  `--reset` paths), downgrade that parameter's regime. `--reset`
  already force-downgrades to noisy; extend to per-parameter local
  density: `abs(recommended - nearest_observed) / step` exceeds N
  -> noisy or sparse. Pure bookkeeping.
- **Physics rebuild fixes**: NO -- bookkeeping bug, not physics bug.

### Mode 5 -- Cannot reason about new cars/tracks

A new car or track without a dense corpus produces extrapolation,
not engineered compensation, and the trust radius today only
mitigates by clipping to corpus envelope.

- **Verdict**: SUPPORTED, but bites a small fraction of usage.
- **Evidence**: Cross-car schedule fallback
  (`_maybe_borrow_cross_car_track`) borrows corner geometry but not
  setup priors; verified by Ferrari@Spa working despite zero Ferrari
  Spa IBTs.
- **Cheapest fix**: not really fixable empirically. This is the ONE
  mode where a physics layer is the right answer.
- **Physics rebuild fixes**: YES (and this is the only mode where the
  rebuild premise is actually correct).

### Summary

| Mode | Verdict | Cheapest fix (days) | Physics rebuild fixes? |
|---|---|---|---|
| 1 cross-track confounding | SUPPORTED | 2-4 (Bayesian or per-track) | NO |
| 2 tyre pressure | SUPPORTED | 0.5 (floor pin) | YES, but 0.5d pin closes it |
| 3 driver bias | WEAK | 1-2 (lap-time-weighted samples) | NO |
| 4 confidence label | SUPPORTED | 1 (per-parameter density) | NO |
| 5 new car/track | SUPPORTED | not fixable empirically | YES |

**3 of 5 modes are orthogonal to a physics rebuild.** All four of
Modes 1-4 are total <8 person-days of bookkeeping + better surrogate
structure. Mode 5 is the only physics-required mode.

---

## 3. Why the original "physics-first" proposal was wrong-shape

The original proposal (8-stage Pacejka tires + bicycle model + lap-
time integrator + hybrid optimizer, 14 days unattended) was reviewed
by 6 specialized agents. Convergent findings:

1. **Pacejka fit is circular from iRacing telemetry alone.** Slip
   angle has no ground truth; Fz is model-derived; F_lat is
   m * a_lat divided by an assumed front/rear split. The fitter
   minimizes residuals but the parameters are not identifiable. This
   is the textbook "two-fit problem" VISION SS3 explicitly warned
   against.

2. **Engine torque is unobservable on a hybrid powertrain.** GTP cars
   have ICE + MGU-K. `TorqueMGU_K` covers only the motor. ICE torque
   is not a channel. Backing engine torque out of `m * a_long`
   requires subtracting drag (poorly known), brake torque (not a
   channel), MGU assist, AND tire rolling resistance -- five unknowns
   and one equation.

3. **Cd vs (RH_f, RH_r) is data-starved.** Pure-coast samples
   (no throttle, no brake, straight) are about 50/session. Even
   pooled across 37 BMW Sebring sessions that's ~2k -- enough for
   one Cd, not enough for a 2D function.

4. **Every acceptance gate could pass while directionally wrong.**
   Single-track lap-time gates overfit; "within 1 click on 3 of 47
   parameters of a setup that's *in* the training corpus" is
   train-on-test leakage; "moves toward direction" on 2 parameters
   passes ~25% by random chance.

5. **Sim racing community evidence**: nobody ships a Pacejka-based
   setup optimizer. VRS, GarageHive, Coach Dave Academy, MoTeC, and
   academic literature (Veneri/Massaro F3; the 2024 ScienceDirect
   "AI-enabled prediction of sim racing performance") are
   overwhelmingly empirical/iterative or surrogate-based.

6. **14-day unattended autonomous run = ~4% chance all stages clean**
   (compounding 0.67^8 from Devin's reported per-task merge rate;
   ~30% chance of an incident based on Cursor/PocketOS production-DB-
   deletion precedent). Compaction context loss across long runs
   documented in GitHub anthropics/claude-code issues #23620, #23966.

The conclusion: **build the cheap empirical fixes first; add physics
only where telemetry honestly supports it; do not run unattended.**

---

## 4. Hard rules (DO and DO-NOT)

### 4.1 Forbidden operations

The agent MUST NOT, under any circumstance:

- F1. Push to `master` directly. All work goes via PR (Section 6).
- F2. Force-push to any branch matching `physics-rebuild-*` or
  `master`.
- F3. Delete or rewrite any tag (especially
  `physics-rebuild-day-NN-locked-*`).
- F4. Run `git filter-branch`, `git reset --hard` on `master`,
  `git rebase -i`, or any history-rewriting operation on a published
  branch.
- F5. Touch `ibtfiles/`, `aero-maps/`, or `corpus/` *except* its own
  `corpus/models/*.pickle` cache files. The pre-existing
  PreToolUse-on-Bash hook (`.claude/hooks/`) already blocks
  destructive ops on these paths; the agent MUST NOT attempt to
  bypass via subagent or shell escape.
- F6. Load any IBT in the held-out set (Section 7) into any fit
  pipeline. Held-out IBTs are **gate-only**.
- F7. Skip pre-commit hooks (`--no-verify`), bypass GPG signing
  (`--no-gpg-sign`), or disable any test or lint check.
- F8. Delete more than 500 lines of existing code in a single commit
  without a daily-snapshot rationale entry approved by the external
  judge.
- F9. Modify this PLAN.md after authorization. The plan is the spec.
- F10. Use any tool, MCP server, or subagent to circumvent F1-F9.
- F11. Begin the next day's work before the current day's external-
  judge verdict is recorded.
- F12. Mark a gate "passed" when the external judge said "fail" or
  "unsure".

### 4.2 Required operations

The agent MUST, every single day:

- R1. Re-read this PLAN.md before starting work. Compare invariants
  in Section 4 to current behavior; halt if drift detected.
- R2. Re-read the previous day's `daily_NN-1.md` snapshot.
- R3. Verify held-out set unmodified (`scripts/verify_holdout.sh`,
  Section 7).
- R4. Work on a `physics-rebuild/day-NN-<topic-slug>` branch.
- R5. Run `uv run pytest -m "not slow"` and `uv run ruff check src
  tests` before any commit. Both must be clean.
- R6. Write a `daily_NN.md` snapshot at end of day with mandatory
  fields (Section 8.4).
- R7. Spawn external-judge subagent BEFORE tagging the day locked.
- R8. Tag the day locked ONLY if external judge returns `pass`.
- R9. Set `fallback_mode_used: true` in the snapshot if any gate was
  met via degraded path. No silent fallbacks.
- R10. Stop-and-wait per Section 11 at the listed checkpoints.

### 4.3 Plan-integrity invariants

Every loop iteration, the agent asserts these literally before
beginning work. Drift = STOP and write `BLOCKED_dayNN.md`.

- I1. Held-out set: 5 IBTs in Section 7. Hash-pinned. NEVER loaded
  for fitting.
- I2. Branch protection on `master`: requires PR and passing CI. The
  agent has push to `physics-rebuild/*` only.
- I3. Token budget: 2M soft / 4M hard per day (Section 9).
- I4. Stop-and-wait checkpoints: 9 listed in Section 11. ALL ARE
  MANDATORY.
- I5. `fallback_mode_used: true` for 2 consecutive days = HARD STOP.
- I6. Stuck-loop heuristic: 3 identical tool calls or A-B-A-B pattern
  = HARD STOP.
- I7. >500 LoC deletion in single commit = HARD STOP.
- I8. The 5 modes' verdicts in Section 2 are the operating diagnosis.
  The plan does not pivot mid-execution.

---

## 5. Permissions matrix

| Action | Authorization | Method |
|---|---|---|
| Create `physics-rebuild/*` branch | Pre-auth | `git checkout -b` |
| Commit on feature branch | Pre-auth | `git commit` (signed) |
| Push feature branch to `origin` | Pre-auth | `git push -u origin` |
| Create PR | Pre-auth | `gh pr create` |
| Tag `physics-rebuild-day-NN-locked-*` | After gate pass + judge `pass` | `git tag -as` |
| Merge PR to `master` | **User-only** | (manual via `gh` web UI) |
| Delete any tag | **FORBIDDEN** | -- |
| Push to `master` | **FORBIDDEN** | -- |
| Modify PLAN.md | **FORBIDDEN** after auth | -- |
| Modify CLAUDE.md | Allowed | only inside daily snapshot scope |
| Touch `ibtfiles/` | **READ-ONLY** | (hook-enforced) |
| Touch `aero-maps/` | **READ-ONLY** | (hook-enforced) |
| Touch `corpus/` | RW for own caches; READ-ONLY for parquet/catalog | -- |
| Modify held-out IBT set | **FORBIDDEN** | (Section 7 list is immutable) |
| Spawn subagent | Allowed | per Section 10 only |
| Run `optimize learn` | **FORBIDDEN** -- writes to `corpus/` | -- |

`gh` CLI permissions: agent has `repo` scope only. No org/team
admin, no settings edits, no webhook changes.

---

## 6. Branch / tag / commit conventions

### 6.1 Branch naming

`physics-rebuild/day-NN-<topic-slug>`

Examples:

- `physics-rebuild/day-01-tyre-pressure-floor`
- `physics-rebuild/day-02-density-confidence`
- `physics-rebuild/day-03-bayes-track-effects`

Rules:

- One branch per day, even if work spans two consecutive sessions.
- Topic slug is lowercase, hyphenated, <= 5 words.
- Branch is created from the prior day's locked tag (or `master` for
  Day 1).

### 6.2 Tag naming

`physics-rebuild-day-NN-locked-<topic-slug>`

- Tag is signed (`-s`), annotated (`-a`), and applied to the daily
  PR's head commit *after* external-judge `pass`.
- Server-side hook to be configured (Section 17 sign-off) rejects:
  - Force-push to any tag matching `physics-rebuild-*`.
  - Tag deletion of `physics-rebuild-day-*-locked-*`.

### 6.3 Commit message format

Title: `<type>(scope): <72 char summary>` (matches existing repo
convention).

Body: WHY first, evidence (file:line / artefact path), gate criteria
that the change pass against.

Footer: `Co-Authored-By: Claude Opus 4.7 (1M context)
<noreply@anthropic.com>`

Example:

```
feat(cli): pin tyre_cold_pressure_kpa to constraint floor (Mode 2)

Per PLAN.md Mode 2: surrogate rewards platform stability without
seeing peak-grip-vs-Fz. Community wisdom is "stay at the floor."
Evidence: recommendations/bmw-spa-race-0507-2308.txt recommends
154.0 kPa; constraint floor for BMW is 152.0 kPa.

Implementation: in `recommend_cmd`, after `_apply_pins_to_constraints`,
if `tyre_cold_pressure_kpa` is fittable AND user did not pass
`--pin tyre_cold_pressure_kpa=`, force-pin to the per-car constraint
floor. Print info line.

Gate: `optimize bmw spa --json` must show 152.0 kPa.
Canary: with the pin disabled, gate fails (would recommend 154-163).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

### 6.4 PR template

```
## Day NN: <topic>

**Plan reference**: PLAN.md Section 14.NN (or 15.NN).

**Gate (from PLAN.md)**:
> <verbatim acceptance gate>

**External-judge verdict**: <pass|fail|unsure>
**External-judge run id**: <agent id>

**Files changed**: <list>
**LoC delta**: +<N> / -<M>
**Tests added**: <count>
**Held-out validation**: <pass|N/A>
**Fallback mode used**: <true|false>

## Test plan
- [ ] `uv run pytest -m "not slow"` -- pass
- [ ] `uv run ruff check src tests` -- clean
- [ ] held-out unmodified: `scripts/verify_holdout.sh`
- [ ] gate validation script: <path>
- [ ] canary test: <path>

## Notes
<anything user should see when they return>

Generated under PLAN.md authorization. Do not merge before reading
`docs/physics-rebuild/daily_NN.md`.
```

### 6.5 Master protection (user enables Day 0)

Before authorization (Section 17), user enables on GitHub:

- Require PR review before merging to `master` (1 approving review).
- Require status checks: `pytest`, `ruff`, plus a custom check
  `physics-rebuild-judge-passed` that the daily PR adds via
  workflow.
- Restrict who can push to `master` to user account only.
- Restrict tag deletion: branch protection rule on
  `refs/tags/physics-rebuild-day-*-locked-*`.

If branch protection cannot be enabled (e.g. private fork without
that feature), the agent is **not authorized to begin** -- this is a
mandatory pre-condition.

---

## 7. Held-out validation set

These five IBTs are reserved as a held-out gate set. They are NEVER
loaded into any fit, train, or recommend pipeline during the build.
They are loaded ONLY by gate validation scripts.

| # | Path | Purpose |
|---|------|---------|
| H1 | `ibtfiles/bmwlmdh_spa 2024 up 2026-05-07 11-59-06.ibt` | BMW Spa under-corpus victim (Mode 1 documented case) |
| H2 | `ibtfiles/cadillacvseriesrgtp_lagunaseca 2026-04-29 23-48-56.ibt` | Cadillac as v4 path |
| H3 | `ibtfiles/ferrari499p_hockenheim gp 2026-03-31 12-32-40.ibt` | Ferrari Hockenheim (the documented Mode 1 dragger) |
| H4 | `ibtfiles/acuraarx06gtp_daytona 2011 road 2026-04-03 20-39-40.ibt` | Banked-track sign-error canary + Acura degraded-channel case |
| H5 | `ibtfiles/porsche963gtp_algarve gp 2026-04-04 12-30-44.ibt` | Porsche v3 path coverage |

### 7.1 Verification

`scripts/verify_holdout.sh` (to be created Day 0) runs every morning
before work begins. It:

1. Computes SHA-256 of each held-out IBT path.
2. Compares to the manifest at `docs/physics-rebuild/holdout.sha256`
   (committed Day 0; never modified).
3. Greps the catalog for held-out session IDs and confirms they are
   marked `held_out=true` (catalog flag added Day 0).
4. Greps `corpus/models/*.pickle` for any pickle whose `session_ids`
   tuple includes a held-out session id; FAIL on any hit.

If any check fails, the day's work is HALTED until the user re-
authorizes.

### 7.2 Catalog flag

Day 0 task: add a `held_out` BOOLEAN column to the catalog table
(default false). For the five held-out sessions, set to true. The
ingestion pipeline (`ingest/api.py`) skips `held_out=true` rows in
`sessions(...)` query. The fitter therefore never sees held-out
sessions.

---

## 8. Daily workflow

### 8.1 Pre-work checklist

Run before the first commit each day. Each item is mandatory; failure
of any item halts.

1. `git fetch origin && git checkout master && git pull --ff-only origin master`
2. Read `docs/physics-rebuild/PLAN.md` (this file).
3. Read `docs/physics-rebuild/daily_NN-1.md` if N >= 2.
4. Run `scripts/verify_holdout.sh` -- must exit 0.
5. Run `uv run pytest -m "not slow"` on master -- must be clean.
6. Run `uv run ruff check src tests` -- must be clean.
7. Re-read the day's section in PLAN.md; copy gate criteria verbatim
   into a scratch note.
8. Create branch: `git checkout -b physics-rebuild/day-NN-<slug>`.
9. Confirm token-budget tracker exists at
   `docs/physics-rebuild/budget_NN.txt` (created from template if
   absent); record start tokens.

### 8.2 Work checklist

- TDD: write the gate test FIRST, confirm it fails, then implement.
- Implement in the minimal scope per the day's section.
- Run targeted tests + `ruff` after each meaningful change.
- Document non-obvious decisions in code comments only when the WHY
  is not derivable from the code.
- Track token spend; if soft cap reached, finish current change and
  stop spawning subagents.

### 8.3 Gate evaluation

1. Run the day's gate validation script.
2. Run the day's broken-model canary test -- canary MUST FAIL when
   the change is reverted.
3. Run the held-out validation script if the day says so.
4. Run external-judge subagent (Section 10) with: gate criteria,
   gate output, canary output, held-out output, snapshot draft.
5. If judge returns `pass`: commit snapshot, push branch, create PR,
   tag locked.
6. If judge returns `fail`: halt; write `BLOCKED_dayNN.md`; do not
   tag or merge.
7. If judge returns `unsure`: HARD STOP (Section 11 #9); do not tag.

### 8.4 Daily snapshot format (`docs/physics-rebuild/daily_NN.md`)

Mandatory frontmatter; missing fields = invalid snapshot.

```yaml
---
day: NN
date: YYYY-MM-DD
branch: physics-rebuild/day-NN-<slug>
commits: [<sha>, ...]
pr_url: https://github.com/<repo>/pull/<n>
tag: physics-rebuild-day-NN-locked-<slug>
gate_passed: true|false
gate_output_path: <relative path>
canary_failed_as_expected: true|false
holdout_validated: true|false|n/a
external_judge_verdict: pass|fail|unsure
external_judge_agent_id: <id>
external_judge_summary: <one line>
fallback_mode_used: true|false
fallback_rationale: <if true, one paragraph>
loc_added: <int>
loc_removed: <int>
files_changed: <int>
tokens_used_today: <int>
cumulative_tokens: <int>
---

# Day NN: <topic>

## What I built

## Gate result
- Numeric: <metric: target / observed>
- Pass/fail: <result>

## Canary result

## Held-out validation (if applicable)

## External judge verdict
> Verbatim copy of judge response.

## What's next

## Open questions for user
```

### 8.5 End-of-day checklist

1. Snapshot committed.
2. Branch pushed.
3. PR created.
4. Locked tag pushed (only if judge `pass`).
5. Token tracker updated.
6. CLAUDE.md updated only if architecture changed (rare; needs
   judge approval as "architecture-affecting" change).

---

## 9. Token / wall-clock budget

| Limit | Value | Action when reached |
|---|---|---|
| Soft cap per day | 2,000,000 tokens | Stop spawning subagents; finish current change |
| Hard cap per day | 4,000,000 tokens | Pause until next calendar day (UTC midnight) |
| Subagent token cap | 200,000 per spawn | Subagent prompt enforces this |
| External-judge cap | 100,000 per run | Single response only |
| Wall-clock per day | 14 hours | Hard stop |
| Wall-clock per stage | 1.5 days | Stop-and-wait if exceeded |

Tracker: `docs/physics-rebuild/budget_NN.txt`. One line per material
operation: `HH:MM tokens_delta cumulative description`. Auto-summed
into the daily snapshot.

---

## 10. External-judge protocol

The agent both writes the model AND grades it. To prevent self-grade
bias, EVERY day's gate evaluation goes through an external judge.

### 10.1 Spawn

```
Agent(
  description="External judge -- Day NN gate review",
  subagent_type="general-purpose",
  prompt=<built from template below>,
)
```

### 10.2 Prompt template

```
You are an external validation judge for racingoptimizer's physics-
rebuild Day NN. You have NO context from prior conversations. You
only see what's in this prompt.

PLAN: PLAN.md Section <14|15>.<NN> says the day's gate is:
<verbatim gate>

GATE OUTPUT:
<paste gate validation script output verbatim>

CANARY OUTPUT (canary should FAIL when feature is removed):
<paste canary output>

HELD-OUT OUTPUT (if applicable):
<paste held-out script output>

DAILY SNAPSHOT DRAFT:
<paste snapshot draft>

YOUR JOB: read the gate criterion, read the outputs, decide ONE OF:
- "pass": the gate is rigorously met, canary fails as expected, no
  evidence of fallback or shortcut.
- "fail": the gate is not met OR canary did not fail OR snapshot
  contradicts outputs.
- "unsure": something is ambiguous; explain.

Constraints:
- You do NOT see the implementation code.
- You do NOT see prior days' work except the gate criterion.
- "pass" requires ALL of: gate met, canary failed as expected,
  snapshot fields consistent with outputs.
- If `fallback_mode_used: true`, you MUST return "unsure" or "fail"
  unless the day's plan explicitly authorized fallback.

Output exactly:
{verdict: pass|fail|unsure}
{summary: <single sentence rationale>}
```

### 10.3 Judge binding

- Judge `pass` -> tag locked, PR published.
- Judge `fail` -> halt; `BLOCKED_dayNN.md` written.
- Judge `unsure` -> HARD STOP for user re-auth.

The agent SHALL NOT re-run the judge with a different prompt to
"flip" a verdict.

---

## 11. Stop-and-wait checkpoints

At each of these the agent halts work, writes a status note, and
waits for user re-authorization:

| # | Trigger |
|---|---------|
| 1 | End of Day 1 (first ship: Mode 2 closed) |
| 2 | End of Day 7 (Week 1 -> Week 2 transition; Modes 1-4 closed) |
| 3 | End of Day 9 (first scoped-physics piece: damper refit) |
| 4 | End of Day 12 (first physics evaluator) |
| 5 | End of Day 14 (final ship) |
| 6 | Two consecutive `fallback_mode_used: true` days |
| 7 | Stuck-loop heuristic fires (Section 4.3 I6) |
| 8 | Single commit deletes >500 LoC |
| 9 | External judge returns `unsure` |

Halting writes a `STOP_dayNN_<reason>.md` file at
`docs/physics-rebuild/`. Resumption requires the user to update the
file with `resume: true` and (where applicable) corrected
parameters.

---

## 12. Fallback-mode discipline

A fallback mode is a degraded numerical method, simpler model, or
cached fixture used because the primary path failed. Examples:

- Hierarchical Bayesian fit didn't converge in 30 min -> use a
  point-estimate per-track Forest instead.
- Damper curve fit had insufficient samples for digressive shape ->
  use a linear curve with the slope from peak observed velocities.

Rules:

- Allowed only if the day's section explicitly lists a fallback
  option.
- Required: `fallback_mode_used: true` + `fallback_rationale` +
  recovery plan (when can the primary path be retried?).
- Forbidden: silent fallback (any path that ships without the
  snapshot flag set).
- Two consecutive fallback days = HARD STOP (Section 11 #6).

---

## 13. Failure recovery

| Failure | Recovery |
|---|---|
| Daily branch breaks unrelated tests | STOP, file `BLOCKED_dayNN.md`, do not merge. |
| Gate fails | Document in snapshot; decide degraded path (if authorized) vs STOP. |
| Canary doesn't fail when expected | STOP -- means canary is wrong OR change is no-op; do not tag. |
| External judge `fail` or `unsure` | STOP per Section 11. |
| Held-out IBT modified or loaded into fit | HARD STOP. Reset working tree to last locked tag. Notify user. |
| Catastrophic corruption (e.g. parquet schema break) | `git checkout <last locked tag> -- <path>` for the affected file; STOP. Do NOT `git reset --hard master`. |
| Stuck loop detected | HARD STOP. Write `STOP_dayNN_stuck-loop.md`. |
| Token hard cap reached | Pause until next UTC day; resume from last tag. |
| `gh` rate limit reached | Pause 1 hour; if persistent, STOP. |

---

## 14. Week 1 plan -- empirical fixes (Days 1-7)

Each day below has the same template:

- **Background**: which mode this closes and why this approach
- **Files to modify**: specific paths
- **Tests to add**: specific paths
- **Acceptance gate**: numeric, specific
- **Broken-model canary**: how we know the gate is real
- **Held-out validation**: which of H1-H5 must pass
- **Fallback mode**: authorized degraded path (or "none")
- **Risk**: LOW / MEDIUM / HIGH
- **Estimated LoC**: rough size
- **Stop-and-wait?**: yes/no

### 14.1 Day 1 -- Tyre pressure floor pin (Mode 2)

- **Background**: Mode 2 evidence shows BMW Spa runs recommend
  154-163 kPa against the 152 floor; surrogate cannot see peak-grip
  drop. A 0.5-day pin in `cli/recommend.py` closes the gap with no
  architecture change.
- **Files to modify**: `src/racingoptimizer/cli/recommend.py`
  (function `recommend_cmd`, after `_apply_pins_to_constraints`).
- **Tests to add**: `tests/cli/test_tyre_pressure_floor.py` -- 6
  tests: per-car floor matches `constraints.md`; user `--pin`
  override is honored; race vs quali both pinned; floor pin shows
  in recommendation output.
- **Acceptance gate**: For all 5 cars on a representative track
  (BMW Sebring, Cadillac Laguna, Ferrari Spa, Acura Daytona,
  Porsche Algarve), `optimize <car> <track> --json` returns
  `tyre_cold_pressure_kpa` equal to the per-car constraint floor
  +/- 0.01 kPa.
- **Broken-model canary**: Disable the pin (commit it as a
  one-line change with `if False`). Re-run gate. Gate MUST fail
  on at least 3 of 5 cars (current model is documented to drift
  off-floor).
- **Held-out validation**: H1 (BMW Spa held-out) -- run
  `optimize` against synthetic copy of catalog without H1
  ingested, confirm pin still 152.
- **Fallback mode**: none. This is a 1-line guard; no degraded
  path.
- **Risk**: LOW.
- **Estimated LoC**: +30 / -2.
- **Stop-and-wait**: YES (Section 11 #1).

### 14.2 Day 2 -- Per-parameter local density confidence (Mode 4)

- **Background**: Mode 4 evidence: `confidence/confidence.py:60-92`
  uses only global stats. Per-parameter local density check is
  pure bookkeeping; closes the "dense while extrapolating" lie.
- **Files to modify**: `src/racingoptimizer/confidence/confidence.py`,
  `src/racingoptimizer/physics/recommend.py::_pin_or_trust_bounds`.
- **Tests to add**: `tests/confidence/test_local_density.py` -- 8
  tests: in-cluster -> regime same as global; N steps from
  nearest -> downgrade by one; outside corpus envelope ->
  downgrade to noisy; `--reset` already-noisy stays noisy.
- **Acceptance gate**: For 5 representative recommendations,
  every parameter whose recommended value is more than 3 *step*
  units from its nearest observed value gets regime label one
  worse than its global label, OR `noisy`/`sparse` already.
- **Broken-model canary**: Set the local-density threshold to a
  huge number (effectively disable). Re-run gate; must FAIL --
  some recommendations should drift but no labels change.
- **Held-out validation**: H1, H4 -- both have parameters known
  to be sparse (BMW Spa wing variance; Acura degraded channels).
- **Fallback mode**: none.
- **Risk**: LOW.
- **Estimated LoC**: +80 / -10.
- **Stop-and-wait**: NO (continue to Day 3 same session if budget
  allows).

### 14.3 Days 3-5 -- Hierarchical Bayesian retrofit (Mode 1)

- **Background**: Mode 1 evidence: cross-track confounding
  documented in `CLAUDE.md`. Reviewer Agent 3 ranked
  hierarchical Bayes #1 alternative: 3-5 days, direct hit on the
  top documented user-visible failure mode, calibrated posteriors
  also fix Mode 4 secondarily.
- **Approach**: PyMC or NumPyro model:
  ```
  y ~ Normal(mu, sigma)
  mu = alpha + beta_track[track] + beta_car[car] + beta_session[session] + ...
  alpha ~ Normal(0, 10)
  beta_track ~ Normal(0, sigma_track)
  sigma_track ~ HalfNormal(...)
  ```
  Track / car / session as random effects with shrinkage; per-
  parameter posterior std replaces `parameter_observed_std`.
- **Files to add**: `src/racingoptimizer/physics/bayes_retrofit.py`.
- **Files to modify**: `src/racingoptimizer/physics/fitter.py::fit_per_car`
  (call retrofit after Forest fit; store posteriors in PhysicsModel).
- **Tests to add**: `tests/physics/test_bayes_retrofit.py` -- 12
  tests: posterior recovers known-true mean on synthetic data;
  shrinkage occurs across tracks; held-out track posterior std
  larger than in-corpus; per-car retrofit deterministic on fixed
  seed.
- **Acceptance gate**: For BMW with H1 (Spa) held out, the
  Bayesian retrofit's posterior mean prediction on H1 must beat
  the current v4 surrogate prediction on H1 by >= 5% in MAE on
  setup-readout target columns. Plus posterior 95% interval must
  cover >= 80% of held-out setup readouts.
- **Broken-model canary**: Replace the hierarchical model with
  pooled regression (no random effects). Re-run gate; must FAIL
  -- pooled regression underestimates the BMW Spa posterior std.
- **Held-out validation**: H1 (BMW Spa) -- the documented Mode 1
  victim. Canary day: H3 (Ferrari Hockenheim) -- the documented
  dragger.
- **Fallback mode**: AUTHORIZED. If MCMC doesn't converge in 30 min,
  fall back to point-estimate per-track Forest (one Forest per
  track for the high-variance ~10 parameters). Set
  `fallback_mode_used: true`. Day 4 retries primary; if still no
  convergence, ship fallback and STOP at Day 5 boundary.
- **Risk**: MEDIUM (MCMC convergence risk on small per-track corpora).
- **Estimated LoC**: +400 / -30.
- **Stop-and-wait**: at end of Day 5 (Section 11 #2 is end-of-week,
  but this single 3-day component warrants its own check before
  proceeding).

### 14.4 Day 6 -- Lap-time-weighted sample reweighting (Mode 3)

- **Background**: Mode 3 evidence: weak; mechanism plausible. Cheap
  fix: weight training rows by `1 / (lap_time - track_min + epsilon)`.
- **Files to modify**: `src/racingoptimizer/physics/fitter.py`
  (`_collect_training_frames`, sample-weight pipeline).
- **Tests to add**: `tests/physics/test_lap_weighted.py` -- 6 tests:
  fast laps weighted higher; outlier laps clipped; weight
  determinism; effect on baseline.
- **Acceptance gate**: For the BMW Sebring corpus (37 sessions),
  the fitter's per-parameter `baseline_setup` shifts toward the
  values used in the user's known-fast laps (top quartile by
  lap time) by >= 0.3 *step* units on at least 5 fittable parameters.
- **Broken-model canary**: Set all weights to 1.0; gate must FAIL.
- **Held-out validation**: H1 -- BMW Spa baseline must shift
  toward the known-fastest BMW Spa setup.
- **Fallback mode**: none.
- **Risk**: LOW.
- **Estimated LoC**: +60 / -10.
- **Stop-and-wait**: NO.

### 14.5 Day 7 -- Week 1 validation gate

- **Background**: Cumulative validation that Modes 1-4 are
  closed. This is the Week 1 -> Week 2 transition.
- **Files to add**: `scripts/week1_gate.py`.
- **Tests to add**: `tests/integration/test_week1_gate.py` -- 12
  tests calling `optimize` end-to-end on the 5 representative
  (car, track) pairs; assertions per below.
- **Acceptance gate**: ALL of:
  1. Mode 2 closed: tyre pressure pinned to floor on all 5 cars
     (carry-over from Day 1).
  2. Mode 4 closed: regime labels per-parameter on all 5 cars
     show `noisy` for parameters >3 steps from corpus density.
  3. Mode 1 closed: BMW H1 held-out MAE improves by >=5% vs
     pre-Week-1 baseline.
  4. Mode 3 closed: BMW baseline shifts toward fast-lap quartile
     on >=5 parameters.
  5. No regressions: full test suite (slow excluded) passes; ruff
     clean; full per-car smoke matrix passes.
  6. Numeric beat: weighted score on the held-out set (Bayesian
     posterior log-likelihood + readout MAE) BEATS the pre-Week-1
     v4 baseline by >= 7% (composite metric).
- **Broken-model canary**: Revert all Day 1-6 commits on the gate
  branch; re-run gate; must FAIL on items 1-4 + item 6.
- **Held-out validation**: ALL of H1-H5.
- **Fallback mode**: none. This is the gate.
- **Risk**: LOW (validation only).
- **Estimated LoC**: +200 (test scripts).
- **Stop-and-wait**: YES (Section 11 #2). User authorization
  required to begin Week 2.

---

## 15. Week 2 plan -- scoped physics (Days 8-14)

Per the review, the Week 2 scope is:

- DROP: Pacejka tire fit (circular), engine torque map (unobservable),
  2D Cd vs (RH) surface (data-starved), lap-time integrator
  (unnecessary; corner-by-corner scoring works).
- KEEP: per-sample diagnostic state (high-confidence channels),
  damper refit (HIGH confidence, T4.4 punch-list win), per-axle
  grip-margin model (one fit per axle per car), aero-map residual
  refinement (don't rebuild aero maps; refine), per-corner-phase
  physics evaluator, hybrid optimizer.

### 15.1 Day 8 -- Telemetry-derived diagnostic state

- **Background**: From Reviewer Agent 1: HIGH-confidence channels
  in iRacing IBT support body slip beta, per-tire kinematic slip
  angles, and total chassis force decomposition. NOT used as
  Pacejka fitting inputs; used as diagnostic outputs (briefing
  side info: "front axle ran at 95% of measured grip ceiling at
  T7 mid-corner").
- **Files to add**: `src/racingoptimizer/physics/diagnostic_state.py`.
- **Files to modify**: `src/racingoptimizer/corner/states.py` to
  expose β, axle slip, axle force per phase.
- **Tests to add**: `tests/physics/test_diagnostic_state.py` -- 10
  tests including unit checks (β bounded, sign matches steering).
- **Acceptance gate**: On all 5 cars on the SEEN corpus,
  diagnostic state computes for >=80% of clean samples; chassis
  force decomposition residual on Fz balance < 5%; β sign
  correlates with steering on >=80% of mid-corner samples.
- **Broken-model canary**: Invert the sign on the steering ratio.
  Re-run gate; β-vs-steering correlation must FAIL.
- **Held-out validation**: H4 (Acura Daytona, banked) -- the
  banked-track sign-error case. Force decomposition must stay
  within 5% on held-out banked corners; if it doesn't, sign error
  exists; STOP.
- **Fallback mode**: AUTHORIZED. If a per-car wheelbase / steering
  ratio is unknown, fall back to a published-spec value with a
  +/-10% sensitivity check.
- **Risk**: MEDIUM (banked-track sign-error scenario).
- **Estimated LoC**: +250.
- **Stop-and-wait**: NO.

### 15.2 Day 9 -- Damper curve refit (T4.4)

- **Background**: T4.4 from the punch list: "per-car damper
  coefficients are seeded estimates." HIGH-confidence telemetry:
  `*shockVel` AND `*shockDefl` channels are direct.
- **Files to modify**: `src/racingoptimizer/physics/damper_force.py`
  (replace digressive curve seeded estimates with per-car fits).
- **Tests to add**: `tests/physics/test_damper_refit.py` -- 8 tests.
- **Acceptance gate**: Per-car damper curve fit residual < 8% on
  held-out laps for all 5 cars; refit baseline beats seeded
  baseline on residual MAE.
- **Broken-model canary**: Use the seeded curves; gate must
  FAIL on residual MAE comparison.
- **Held-out validation**: H1, H2, H3 (BMW, Cadillac, Ferrari --
  the v4 cars).
- **Fallback mode**: none.
- **Risk**: LOW.
- **Estimated LoC**: +180.
- **Stop-and-wait**: YES (Section 11 #3).

### 15.3 Days 10-11 -- Per-axle grip-margin + aero-map residual

- **Background**: From Reviewer Agent 1's recommendation:
  axle-grip-margin replaces Pacejka. One fit per axle per car
  using observed (axle force, axle Fz) extremes. Plus refine
  aero-map Cl_f / Cl_r via residual on observed lat-G at high
  speed (don't rebuild aero maps; refine).
- **Files to add**: `src/racingoptimizer/physics/axle_grip.py`.
- **Files to modify**: `src/racingoptimizer/aero/loader.py` to add
  residual-correction interpolation.
- **Tests to add**:
  `tests/physics/test_axle_grip.py` -- 10 tests.
  `tests/aero/test_residual_correction.py` -- 6 tests.
- **Acceptance gate**: On held-out laps, per-axle grip-margin
  predicts whether a corner exceeded 90% of axle ceiling with
  >=70% accuracy; aero residual correction reduces lat-G
  prediction MAE by >=10% on the v4 cars.
- **Broken-model canary**: Per-axle ceiling = infinity; gate must
  FAIL (every corner reads <90%).
- **Held-out validation**: H1, H2, H3.
- **Fallback mode**: AUTHORIZED. If aero residual correction
  doesn't beat the raw aero-map, ship without correction; mark
  fallback.
- **Risk**: MEDIUM.
- **Estimated LoC**: +400.
- **Stop-and-wait**: NO.

### 15.4 Day 12 -- Per-corner-phase physics evaluator

- **Background**: Reviewer Agent 2's specific recommendation:
  S4 can score corner-by-corner without lap-time integration.
  This stage assembles the Day 8-11 outputs into a single
  evaluator: given (setup, corner_state, env), produce a
  physics-based axle-utilization score per (corner, phase).
- **Files to add**: `src/racingoptimizer/physics/evaluator.py`.
- **Tests to add**: `tests/physics/test_evaluator.py` -- 12 tests.
- **Acceptance gate**: Evaluator score correlates (Spearman) with
  empirical observed lap-time-per-corner-phase by >= 0.35 across
  the v4 corpus on held-out laps.
- **Broken-model canary**: Constant score regardless of setup.
  Spearman -> 0; gate must FAIL.
- **Held-out validation**: H1, H2, H3.
- **Fallback mode**: AUTHORIZED. If evaluator misses the 0.35
  Spearman, ship at >= 0.20 with `fallback_mode_used: true` and
  document the cause.
- **Risk**: HIGH. This is where the architecture has to actually
  work end-to-end.
- **Estimated LoC**: +350.
- **Stop-and-wait**: YES (Section 11 #4).

### 15.5 Day 13 -- Hybrid optimizer

- **Background**: DE searches setup space using
  `score = w * physics_evaluator + (1-w) * surrogate_residual`.
  `w` defaults to 0.6 (physics-leaning); user-tunable via
  `--physics-weight`.
- **Files to modify**: `src/racingoptimizer/physics/recommend.py`
  to call the hybrid score.
- **Tests to add**: `tests/physics/test_hybrid_optimizer.py` -- 12
  tests.
- **Acceptance gate**: On the BMW Spa held-out scenario (H1), the
  hybrid optimizer recommends within 1 click of the user's
  validated fastest setup on at least 5 of the 8 highest-impact
  parameters (heave_spring, rear_wing, brake_bias, front
  perches, ARB front, ARB rear, rear ride height, rear toe).
  Note: H1 is held out from training; the optimizer has never
  seen its setup.
- **Broken-model canary**: Set w=0 (surrogate-only); on at least
  3 of the 5 representative cars the recommendation must DRIFT
  off the held-out fastest setup by >= 2 clicks on >= 3
  parameters.
- **Held-out validation**: H1, H4 (banked + degraded-channel
  worst case).
- **Fallback mode**: AUTHORIZED. If hybrid doesn't beat surrogate
  alone on the gate, ship with `--physics` opt-in only (no
  default change); set fallback flag.
- **Risk**: HIGH.
- **Estimated LoC**: +200.
- **Stop-and-wait**: NO (proceed to Day 14 validation).

### 15.6 Day 14 -- Final validation + docs

- **Acceptance gate**: ALL of:
  1. Full per-car smoke matrix passes under `--physics` AND
     default mode.
  2. Held-out gate (H1-H5) passes the Week 2 composite metric.
  3. Numeric beat: hybrid mode improves on Week 1 final by
     >= 5% on the composite metric, OR matches Week 1 within
     2% with explicit `--physics` opt-in (not default).
  4. VISION_COMPLIANCE.md updated.
  5. CLAUDE.md updated with new flags.
  6. Full slow suite passes.
- **Broken-model canary**: Revert all Week 2 commits; per-car
  smoke under `--physics` must FAIL (flag undefined).
- **Held-out validation**: H1-H5.
- **Fallback mode**: none.
- **Risk**: LOW (validation only).
- **Estimated LoC**: +100 (docs).
- **Stop-and-wait**: YES (Section 11 #5).

---

## 16. Hand-off / completion criteria

The plan is COMPLETE when ALL of the following hold:

1. Tags `physics-rebuild-day-01-locked-*` through
   `physics-rebuild-day-14-locked-*` exist and are signed.
2. Daily snapshots `daily_01.md` through `daily_14.md` exist with
   judge `pass` recorded.
3. PRs days 1-14 are open (or merged) on `master` with PR template
   filled.
4. Full test suite passes (slow + fast).
5. Held-out validation passes for all 5 IBTs.
6. CLAUDE.md updated with `--physics` and `--physics-weight` flags.
7. VISION_COMPLIANCE.md updated with a 2026-05-22 follow-up
   section.
8. `docs/physics-rebuild/COMPLETE.md` written summarizing what
   shipped, what fell to fallback, what's deferred to a future
   pass.

The plan is INCOMPLETE-BUT-SHIPPABLE if Week 1 (Days 1-7) finished
clean even if Week 2 partially failed. In that case:

- Week 1 PRs merged.
- Week 2 PRs left open with `BLOCKED_*` notes for user review.
- `COMPLETE.md` says "Week 1 shipped; Week 2 partial; <N>/7 days
  reached."

---

## 17. Sign-off (user authorizes here)

Before any agent work begins, the user authorizes by editing this
section ONLY (no other section may be edited after authorization).

User confirms each item below with "y" or replaces the default
with their preferred value:

```
authorized: y
authorization_date: 2026-05-08
authorization_signal: user "Start" command in chat 2026-05-08

# Plan shape
empirical_first_then_scoped_physics: y
drop_pacejka_engine_cd_integrator: y

# Held-out IBTs (Section 7) -- defaults confirmed by user via "Start"
holdout_h1_bmw_spa: ibtfiles/bmwlmdh_spa 2024 up 2026-05-07 11-59-06.ibt
holdout_h2_cadillac_laguna: ibtfiles/cadillacvseriesrgtp_lagunaseca 2026-04-29 23-48-56.ibt
holdout_h3_ferrari_hockenheim: ibtfiles/ferrari499p_hockenheim gp 2026-03-31 12-32-40.ibt
holdout_h4_acura_daytona: ibtfiles/acuraarx06gtp_daytona 2011 road 2026-04-03 20-39-40.ibt
holdout_h5_porsche_algarve: ibtfiles/porsche963gtp_algarve gp 2026-04-04 12-30-44.ibt

# Branch protection enabled on master
branch_protection_master: y  # via GitHub Rulesets id 16127333 ("Claude")
tag_deletion_protection: pending  # user may add tag-protection rule for physics-rebuild-day-*-locked-*

# Token / wall-clock budgets (Section 9)
token_soft_cap_per_day: 2000000
token_hard_cap_per_day: 4000000

# Stop-and-wait granularity (defaults applied; user may amend later via STOP files)
stop_after_day_1: y
stop_after_day_5: y
stop_after_day_7: y
stop_after_day_9: y
stop_after_day_12: y
stop_after_day_14: y

# External judge
external_judge_required_per_day: y

# Pre-execution prerequisites the agent verified before authorizing
prereq_branch_protection_enabled: y  # verified via rulesets API 2026-05-08
prereq_holdout_manifest_committed: pending  # written by Day 0 prep
prereq_holdout_catalog_flag: pending  # added by Day 0 prep
prereq_verify_holdout_script: pending  # written by Day 0 prep

# User signature
deferral_or_amendment_notes:
```

After this section is filled, the agent treats PLAN.md as immutable
(this Section 17 edit was the one allowed edit) and proceeds with
Day 0 prep.

After this section is filled with `authorized: y` and a date, the
agent treats PLAN.md as immutable and begins Day 0 prep (which is
strictly: write `holdout.sha256`, mark catalog `held_out=true` for
H1-H5, write `verify_holdout.sh`, commit those alone, tag
`physics-rebuild-day-00-locked-prep`, request user to merge that
prep PR before Day 1 begins).

---

## Appendix A -- file paths and key functions

### Existing code the plan touches

| Path | Why |
|---|---|
| `src/racingoptimizer/cli/recommend.py` | Day 1 tyre pin, Day 13 hybrid score wire-in |
| `src/racingoptimizer/confidence/confidence.py` | Day 2 local density |
| `src/racingoptimizer/physics/recommend.py::_pin_or_trust_bounds` | Day 2 wire local density |
| `src/racingoptimizer/physics/fitter.py::fit_per_car` | Days 3-5 Bayes; Day 6 lap-time weights |
| `src/racingoptimizer/physics/damper_force.py` | Day 9 damper refit |
| `src/racingoptimizer/aero/loader.py` | Day 11 aero residual correction |
| `src/racingoptimizer/corner/states.py` | Day 8 diagnostic state |
| `src/racingoptimizer/physics/score.py` | Day 12 physics evaluator integration |

### New files the plan creates

| Path | Purpose |
|---|---|
| `docs/physics-rebuild/PLAN.md` | This file |
| `docs/physics-rebuild/REVIEWS_2026-05-08.md` | Raw 6-agent review findings |
| `docs/physics-rebuild/holdout.sha256` | Held-out IBT hash manifest |
| `docs/physics-rebuild/daily_NN.md` | Daily snapshots (14 of these) |
| `docs/physics-rebuild/budget_NN.txt` | Token tracking (14 of these) |
| `docs/physics-rebuild/COMPLETE.md` | Final hand-off note |
| `scripts/verify_holdout.sh` | Pre-work integrity check |
| `scripts/week1_gate.py` | Day 7 cumulative gate |
| `src/racingoptimizer/physics/bayes_retrofit.py` | Days 3-5 |
| `src/racingoptimizer/physics/diagnostic_state.py` | Day 8 |
| `src/racingoptimizer/physics/axle_grip.py` | Days 10-11 |
| `src/racingoptimizer/physics/evaluator.py` | Day 12 |

### Existing automations the plan extends

- `.claude/hooks/` -- pre-existing destructive-op block on
  `ibtfiles/`, `aero-maps/`. The plan adds a hook to refuse any
  edit to `PLAN.md` Sections 1-16 and 18-20 after authorization.
- `.claude/agents/setup-justifier` -- already gates VISION SS7 on
  per-parameter justification; runs as part of Day 7, Day 14
  gates.
- `.claude/agents/physics-fit-validator` -- already gates VISION
  SS3, SS6 on residual quality + density alignment; runs as part of
  Day 7 and Day 14 gates.

---

## Appendix B -- review-agent findings (one-liner each)

The 6 reviewer agents, summarized:

1. **Telemetry feasibility (Agent 1)**: Pacejka fit is circular;
   engine torque is unobservable on a hybrid; Cd is data-starved.
   Drop those; keep dampers + aero-map refinement + axle-grip-
   margin + diagnostic kinematic state. 14d -> ~9d.
2. **Staging cliffs (Agent 2)**: Self-graded gates pass-while-
   directionally-wrong; banked-track Daytona is the worst-case
   sign-error scenario; demote lap-time integrator off critical
   path; ship damper refit standalone Day 3 as T4.4 punch-list win.
3. **Alternative paradigms (Agent 3)**: Hierarchical Bayesian #1
   alternative; active-learning DOE #2; PINNs and CasADi rejected
   as 14-day-killers. Recommend hybrid: Bayes Days 1-3, scoped
   physics Days 4-12, calibrate-loop Days 13-14.
4. **Diagnosis sanity + community (Agent 4)**: 3 of 5 modes
   orthogonal to physics; total <8d of bookkeeping closes Modes
   1-4. Sim-racing community is 100% empirical; nobody ships
   Pacejka.
5. **Validation methodology (Agent 5)**: Original gates fail train-
   test isolation; need leave-one-track-out CV; need broken-model
   canary; need held-out-OOD setup tests. Adopted in Section 14-15.
6. **Autonomous 14-day execution (Agent 6)**: ~4% chance all 8
   stages clean unattended; ~30% incident probability per
   precedent. Recommend external-judge subagent, signed tags +
   deletion protection, branch protection on master, hard token
   caps, fallback flag discipline, immutable plan-of-record. All
   adopted in Sections 4, 5, 6, 9, 10, 11, 12.

Raw findings preserved at
`docs/physics-rebuild/REVIEWS_2026-05-08.md` (to be written
alongside this file).

---

## Appendix C -- glossary

- **Mode 1-5**: the five failure modes diagnosed in Section 2.
- **v3 / v4**: per-(car, track) and per-car (track-agnostic)
  fitting paths, respectively. Documented in `CLAUDE.md`.
- **Axle-grip-margin**: a per-axle ceiling on lateral force
  fitted from observed extremes, used as a one-fit-per-axle
  replacement for the rejected Pacejka model.
- **Held-out**: the 5 IBTs in Section 7. Reserved gate-only;
  never loaded for fit/training.
- **External judge**: a fresh Claude Code subagent with no shared
  context, used as a binding gate verdict per Section 10.
- **Broken-model canary**: a test where a critical component is
  intentionally disabled or corrupted; the gate MUST fail under
  the canary, otherwise the gate is measuring noise.
- **Stop-and-wait**: a checkpoint where the agent halts work and
  waits for the user to re-authorize, per Section 11.
- **Fallback mode**: an authorized degraded path; sets
  `fallback_mode_used: true` in the daily snapshot. Two
  consecutive days of fallback = HARD STOP.
- **Plan-of-record**: this file. Immutable after authorization.

---

End of PLAN.md.
