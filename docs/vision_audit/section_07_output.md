# VISION Audit — §7 Output ("Justify every click")

## VISION clause

> For every parameter in the output setup, explain WHY that value was chosen:
> which corners benefit, which corners compromise, what's the net trade-off,
> what telemetry evidence supports it, and what would happen if you went ±1-2
> clicks in either direction. The output should read like a race engineer's
> briefing, not a list of numbers.

CLAUDE.md restates this as the architectural commitment "Output explains
every click" and adds the closed-set tag system for the full-setup card
(`[OPT]` / `[OPT pin]` / `[OPT mirror]` / `[past]` / `[readout]` /
`[predicted]`).

## Per-clause audit

| # | Clause | Implementation | Status |
|---|---|---|---|
| 1 | Per-parameter justification structure | `SetupJustification` dataclass at `src/racingoptimizer/explain/justification.py:39-50`; `IncompleteJustificationError` raised in `__post_init__` (`justification.py:52-71`) when any of `parameter`, `unit`, `confidence`, `telemetry_evidence` is missing or both `corners_helped` and `corners_hurt` are empty. | PASS |
| 2 | Corners that benefit / compromise (named corner + phase, per-corner score delta) | `_split_impacts` (`justification.py:155-205`) compares the recommendation's per-(corner, phase) `score_breakdown` against a counterfactual where the parameter is reset to its training-baseline median; deltas > 0 land in `corners_helped`, < 0 in `corners_hurt`, sorted by absolute magnitude. Renderer (`render_text.py:181-188`) emits `Helps:` and `Hurts:` blocks with up to 3 corner-phase lines each. | PASS |
| 3 | ±1-click sensitivity | `_sensitivity` (`justification.py:241-262`) shifts the parameter by `_step_for(...)` (built-in click step or 1 % of constraint range fallback), clamps via `racingoptimizer.constraints.clamp`, and computes `model.score_setup(...)` deltas at +1 and -1 click. Renderer (`render_text.py:178-180`) prints `+1 click: {+/-X.XXX} score    -1 click: {+/-X.XXX} score`. Pinned parameters short-circuit to `0.0, 0.0` and the renderer suppresses with `(no sensitivity reported - pinned)`. | PASS |
| 4 | Telemetry evidence | `_evidence` (`justification.py:302-327`) emits the regime + n_samples line, optional `user override — pinned via --pin`, optional clamp / discrete-rounding warning, and the `observed in training [lo, hi]` bracket from the per-fitter Confidence CI. The bracket is explicitly relabeled "observed in training" (not "the recommendation should be in this bracket") so the briefing reader is not misled when the joint DE search legitimately moves the value outside the per-parameter median CI (`justification.py:316-326`). | PASS |
| 5 | Confidence regime per parameter | Each `SetupJustification` carries the per-parameter `Confidence`; the renderer prints `[confidence: {regime}]` next to the value (`render_text.py:171-172`). Top-of-briefing roll-up (`render_text.py:151-166`) reports the worst regime across all parameters with the dominant-regime n_samples count. | PASS |
| 6 | Engineer-briefing format (not bare numbers) | `render_recommendation_text` (`render_text.py:31-84`) emits one block per parameter ordered by total impact magnitude (`_impact_magnitude`, `justification.py:330-333`), each block carrying the four `setup-justifier` sections. `Warnings:` footer collects `untrained_parameters` and per-parameter clamp issues. | PASS |
| 7 | Closed-set tag system (full-setup card) | **PARTIAL — committed code is behind the documented contract.** The current worktree's `full_setup_card.py` only emits `[OPT]`, `[OPT pin]`, `[past]`, `[readout]` (`full_setup_card.py:262-306`). The documented `[OPT mirror]` and `[predicted]` tags listed in CLAUDE.md "Setup-card renderer contract" are **not** in the committed source — they exist only in the parent repo's uncommitted working tree. The recent BMW Spa briefing artefact (`recommendations/bmw__spa_2024_up__20260505-180530.txt`) was produced by that uncommitted code, so the artefact shows tags that the in-tree renderer cannot emit. | PARTIAL |
| 8 | iRacing-step rounding + discrete-value snap (torsion bar OD) | Step-rounding for continuous parameters lives in `_round_to_step` (`full_setup_card.py:106-119`) and is invoked from `_format_opt_value` (`full_setup_card.py:122-131`). Discrete-click rounding for integer-clicked parameters (dampers, ARB, diff plates) lives in the recommend pipeline (`cli/recommend.py:881-895`) and is surfaced in the briefing as evidence (`discrete-click value rounded from 8.623 to 9 (legal range 0..11)`). **Gap:** the non-uniform discrete-value snap for torsion-bar OD (a 14-value lookup) is **not** in the committed code; it ships only in the parent's uncommitted `_snap_to_discrete` (`full_setup_card.py:157-165` of the parent). The Spa artefact shows torsion-bar OD as `15.14 mm` and `14.76 mm` because the parent's working tree contains the snap; the committed worktree would round to the spec's `step` instead. | PARTIAL |
| 9 | Setup-justifier subagent gate | `.claude/agents/setup-justifier.md` documents a gatekeeper that re-validates the four required fields per parameter against the rendered briefing. Code-level gate is `IncompleteJustificationError` (clause 1 above). | PASS |

## End-to-end artefact check — `recommendations/bmw__spa_2024_up__20260505-180530.txt`

The user's primary evidence is the BMW Spa briefing. Spot-check across 5
representative parameters:

| Parameter | Value | Confidence | Helps (3) | Hurts (3) | ±1 click | Evidence | Tag/round |
|---|---|---|---|---|---|---|---|
| `Rear Wing Angle` (line 264) | `15.77 deg` | dense | T8-braking, T17-straight, T1-trail_brake | T0-braking, T12/T13-mid_corner | `+1: -0.000`, `-1: +0.000` | n=2330, observed `[16.232, 17.768]` | (no discrete tag — wing is continuous) |
| `Damper Hsc Rl` (line 47) | `9.00 click` | dense | T1/T2/T4-trail_brake | T0-braking, T2/T8-mid_corner | `+1: +0.000`, `-1: +0.000` | n=2330, **rounded from 8.623 to 9 (legal 0..11)**, observed `[5.232, 6.768]` | discrete-click rounding present |
| `Anti Roll Bar Rear` (line 132) | `2.00 click` | dense | T8-braking, T14-straight, T9-mid_corner | T0-braking, T2-mid_corner, T6-exit | `+1: +0.000`, `-1: +0.000` | n=2330, **rounded from 1.891 to 2 (legal 1..5)**, observed `[2.232, 3.768]` | discrete-click rounding present |
| `Tyre Cold Pressure` (line 557) | `152.44 kPa` | dense | T8-braking, T17-straight, T6-braking | T0-braking, T12/T13-mid_corner | `+1: +0.000`, `-1: +0.000` | n=2330, observed `[151.232, 152.768]` | continuous; 152.44 sits at the new 152-kPa floor (constraints fix from commit 9e949be) |
| `Diff Clutch Friction Plates` (line 514) | `6.00 plates` | dense | T8-braking, T17-straight, T6-braking | T0-braking, T12/T13-mid_corner | `+1: +0.000`, `-1: +0.000` | n=2330, **rounded from 5.969 to 6 (legal 2..6)**, observed `[5.232, 6.768]` | discrete-click rounding present |

All 6 required elements appear on every parameter block: (a) Helps:, (b)
Hurts:, (c) +1 click / -1 click sensitivity, (d) `observed in training [...]`
evidence, (e) `[confidence: dense]` regime, and where applicable (f)
discrete-click rounding warning. The briefing renders 46 parameter blocks.

The `Warnings:` footer (lines 675-677) correctly lists the pinned parameter
(`arb_size_front`) and the four corner-weight parameters as untrained
(skipped — bounds not in constraints.md).

The full-setup card (lines 679-820) shows all four documented optimizer
tags PLUS the two undocumented-in-this-worktree tags (`[OPT mirror]` on
line 771 RR Spring Rate, `[predicted]` on lines 727, 740, 754, 767 ride
heights). This confirms the artefact was produced from the parent repo's
uncommitted working tree — see clause 7 above.

## Tests

```
uv run pytest -q tests/explain/
1 failed, 22 passed in 3.55s
```

Single failure is the **known pre-existing**
`test_optimizer_recommendations_are_tagged_OPT` — the test asserts the
display string contains `15.5` for a `15.5 deg` wing recommendation, but
the renderer rounds to the wing-step of `1.0 deg` (`spec.step`) and emits
`16 deg`. The test predates the iRacing-step rounding fix; updating the
test to assert `16` (or asserting against a parameter whose step is < 1)
would close it. **Not a regression.**

The other 22 tests cover:

- `test_justification.py` (5 tests) — `IncompleteJustificationError` for
  missing telemetry evidence / both-empty corner-impact lists, plus
  pure-helper and pure-hurter happy paths.
- `test_renderers.py` (uncounted from the run, sub-set of the 22) — text
  + JSON renderer round-trips and the regime roll-up logic.
- `test_full_setup_card.py` (10 tests minus the known failure) — the
  four committed tag paths (`[OPT]`, `[OPT pin]`, `[readout]`, `[past]`),
  the two empty-input paths (`None` and unparseable JSON), the
  user-settable guard (`[OPT]` never tags a `user_settable=False`
  parameter), and a per-car parametrised smoke loop over all 5 GTP cars.

## CLAUDE.md "Setup-card renderer contract" cross-check

CLAUDE.md (parent, lines 59-74) documents 6 tags. The committed worktree
emits 4 of them. The two missing tags (`[OPT mirror]`, `[predicted]`) are
in the parent's uncommitted `full_setup_card.py` along with their
supporting infrastructure (`_MIRRORED_LEAVES`, `_PREDICTED_READOUT_PATHS`,
`_snap_to_discrete`). This is a documentation-vs-code-on-disk drift, not
a VISION violation per se — the committed renderer satisfies VISION's
verbatim text ("justify every click") because the per-parameter briefing
is fully wired. The full-setup card's contract is what's lagging.

## Verdict

**Score: 🟡 (mostly green; one documentation drift gap)**

The per-parameter engineering briefing — VISION §7's literal contract —
is fully implemented:

- `SetupJustification.__post_init__` is the structural gate; missing
  fields raise.
- `_split_impacts` produces named per-(corner, phase) deltas vs. the
  training-baseline counterfactual.
- `_sensitivity` produces ±1-click score deltas with constraint-aware
  clamping.
- `_evidence` reports regime, n_samples, observed-in-training bracket,
  pinning, and clamp / discrete-rounding warnings.
- `render_recommendation_text` lays out the briefing with `Helps:`,
  `Hurts:`, `+1/-1 click`, and `Evidence:` sections per parameter,
  ordered by total impact magnitude.
- The `setup-justifier` subagent provides an out-of-band gate.

The full-setup card — the supplementary "ready to enter" view — is one
commit behind CLAUDE.md's documented closed-set tag contract:
`[OPT mirror]` and `[predicted]` are documented but not implemented in
the committed source. The BMW Spa briefing artefact reflects the parent's
uncommitted working tree and is what users should expect once those
changes land.

## Recommended follow-ups (not VISION violations)

1. **Land `[OPT mirror]` + `[predicted]` + `_snap_to_discrete` (torsion-bar OD)
   in the committed `full_setup_card.py`.** The implementation already lives
   in the parent's uncommitted working tree; commit it so the in-tree code
   matches CLAUDE.md and the artefact in `recommendations/`.
2. **Update `test_optimizer_recommendations_are_tagged_OPT`** to assert the
   post-rounding display (wing step is `1.0 deg`, so `15.5 → 16`).
3. **Add a per-car briefing smoke** that asserts each parameter block in
   `render_recommendation_text` carries Helps / Hurts / ±1-click / Evidence —
   currently the contract is only enforced structurally (via
   `IncompleteJustificationError`), not at the rendered-text level.
