# Audit -- Docs + automations (2026-05-06)

## Summary
- Grade: PARTIAL
- One-sentence verdict: VISION/README/GETTING_STARTED reflect today's CLI, but `CLAUDE.md` is materially stale on the work landed in the last 24 hours (`--reset`, `optimize calibrate`, always-global trust envelope, new recommendation filename), `docs/VISION_COMPLIANCE.md` has the same omissions despite its 2026-05-06 currency note, and the historical 2026-04-28 specs are not labelled as historical.

## Doc-by-doc accuracy

### `CLAUDE.md`
**Correct claims**
- `## Commands` block runs against today's CLI as written (every flag listed exists in `cli/recommend.py` / `ingest/cli.py`).
- `PER_CAR_MODEL_CARS = {"cadillac", "bmw", "ferrari"}` matches `src/racingoptimizer/cli/recommend.py:57`.
- Setup-card tag legend matches the rendered legend in `recommendations/ferrari-spa-race-0506-1026.txt:214`.
- Plain-English narrative as DEFAULT, `--detailed` for legacy block format -- matches `cli/recommend.py:370-388`.
- ASCII-only output rule enforced (`->` and `--` used in artefacts).
- Race-mode fuel auto-pin behaviour and quali requirement of `--fuel` -- matches `cli/recommend.py:196-242`.
- Per-car cache-key digest ingredients accurate.
- Cross-cutting modules (`EnvironmentFrame` 12 channels, `Confidence`, `ConstraintsTable`) accurate.
- Hooks listed (PreToolUse Bash, PostToolUse Edit/Write) match `.claude/settings.json` and the two scripts under `.claude/hooks/`.

**Stale / missing**
- `## Trust radius + --explore N` (line 76) says "Per-car DE search is clipped to `[min(target_observed), max(target_observed)]` -- never extrapolates outside the values the driver has actually tried on **the target track**." That is the pre-`318d91d` rule. As of 2026-05-06 the trust envelope is **always pooled across every track** for the car (see `physics/recommend.py:494-505` docstring: "this used to be per-target-track strict, but that left good setups off the table at every track that hadn't seen full sweeps"). The whole paragraph -- including "VISION §3 honesty rule" framing -- needs rewriting.
- No mention of `--reset` mode (`cli/recommend.py:111-122`) with `RESET MODE` banner and confidence downgrade (`physics/recommend.py:310-314`).
- No mention of `optimize calibrate <car> <track>` subcommand (`cli/calibrate.py`, registered `cli/__init__.py:60`).
- No mention of the new recommendation filename convention `<car>-<short-track>-<mode>[-<fuel>L]-<MMDD>-<HHMM>.<ext>` (`cli/recommend.py:407-424`, `cli/calibrate.py:354-371`).
- Stale file:line cite: "Race-mode fuel auto-pin (`cli/recommend.py:160+`)" -- actual block lives at `cli/recommend.py:213-244`.
- Stale file:line cite: `PER_CAR_MODEL_CARS` at "`recommend.py:54`" -- actual line 57.

### `VISION.md`
- Aspirational design contract; unchanged. Every clause referenced from CLAUDE.md / VISION_COMPLIANCE.md exists. §8 PowerShell examples map to real subcommands. §10 12-channel weather list matches `EnvironmentFrame` exactly. Not a freshness defect by design.

### `GETTING_STARTED.md`
**Correct**
- `## Use it` and "Other commands" blocks match the live CLI (including `calibrate` with default and `--status`).
- Tag legend matches the renderer + artefacts.
- All flags (`--detailed`, `--fuel`, `--quali`, `--explore`, `--wetness`, `--no-cache`, `--corpus-root`, `--reset`) exist as documented.
- Active-learning loop with `calibrate` matches `cli/calibrate.py` behaviour.
- Per-car v4 vs per-(car, track) v3 split correct.

**Stale / missing**
- `--reset` description (line 69) says "abandon the per-track trust radius entirely". After `318d91d` there is no per-track trust radius -- envelope is always-global.
- `--output-file` flag and `--output-file -` to suppress are not mentioned in the user-facing options list.
- `--air-temp` / `--track-temp` / `--wind` only mentioned in passing inside the `--wetness` bullet.

### `README.md`
- Status table mirrors CLAUDE.md and is accurate per-slice. Per-car v4 cars (`bmw`, `cadillac`, `ferrari`) listed correctly.
- No mention of `--reset`, `optimize calibrate`, `--explore`, `--quali`, narrative-vs-detailed split. Likely intentional (minimal README), but a one-line pointer to GETTING_STARTED for those flags would close the gap.

### `docs/VISION_COMPLIANCE.md`
**Correct**
- "2026-05-06 follow-up audit" header section enumerates many post-2026-05-01 changes (per-car v4 enabled for BMW/Ferrari, narrative renderer, race-fuel auto-pin, `--explore`, `--reparse`, `--detailed`, picker fix, torsion bar L/R symmetry, toe-in mm, BMW heave step, DE budget bump, wet/wind wiring) -- match the git log.
- 2026-05-01 per-clause scorecard preserved as historical evidence and clearly framed as such.

**Stale / missing**
- Currency note promises "the summary at the top of this file has been refreshed with every change landed since" but the follow-up section omits `--reset` (commit `87289a8`), `optimize calibrate` (commit `7e15db0`), the always-global trust envelope (commit `318d91d`), the cross-track envelope intermediate (commit `d64fd7b`), the new mode-tagged short filename (commit `0a3f256`), and the telemetry-backed Why line (commit `2355127`). All six landed before the document's "as of `7e3c172` (2026-05-06)" cutoff.

### `docs/PARAMETER_VALIDATION.md`
- Generated by `scripts/validate_ontology_paths.py`; per-car ontology paths resolving against the canonical IBT fixture. Format and per-car summary row counts (42 params each, OK/MISMATCH/BLOCKED/READOUT sums) consistent.
- Finding 0 (`brake_bias_pct` vs `brake_duct_*`) correctly cross-references `constraints.md` line ranges.
- Document is undated and carries no "regenerated against `<commit>`" stamp -- freshness not auditable.

### `docs/VISION_ADHERENCE_REPORT.md`
- Dated 2026-05-01; explicit "supersedes the older green-only compliance claims in `docs/VISION_COMPLIANCE.md`".
- Same blind spot as VISION_COMPLIANCE: no mention of `--reset`, `--explore`, `--quali`, `optimize calibrate`, narrative-default renderer, or always-global trust envelope.
- "Supersedes" framing creates ambiguity with VISION_COMPLIANCE.md (which was also updated 2026-05-06). One should be retired or they should be merged.

### `docs/superpowers/specs/`
- Six 2026-04-28 specs are pre-implementation design docs. The CLI spec in particular predates per-car v4, narrative renderer, `--quali`, `--reset`, `--explore`, `optimize calibrate`. CLAUDE.md positions specs as "read before touching the slice's code", which doesn't match the live surface anymore. Specs are not labelled as historical.
- `2026-04-30-user-settable-and-full-setup-card.md` -- explicit post-hoc spec ("written after the fact as part of the audit follow-up"); accurate against current `physics/ontology.py` three-flag matrix and `explain/full_setup_card.py`.

### `docs/superpowers/plans/`
- `2026-04-28-aero-loader.md`, `2026-04-28-ibt-ingestion.md` -- historical execution plans for merged slices.
- `2026-05-04-percar-correctness.md` -- the per-track-observed cap that this plan delivered was relaxed by `318d91d` two days later. Plan now reads as historical context but is not labelled as superseded.

## Automations review

### Hook -- `.claude/hooks/protect-data.sh` (PreToolUse on Bash)
Reads tool-call JSON, blocks (exit 2) when command targets `ibtfiles/` or `aero-maps/` AND uses a destructive verb (`rm`, `mv`, `truncate`, `shred`, `dd`) or output redirect. jq with grep fallback. Documented in CLAUDE.md `## Project automations`. Works.

### Hook -- `.claude/hooks/validate-constraints.sh` (PostToolUse on Edit|Write|MultiEdit)
After any Edit/Write to `*constraints.md`, greps for `## Defaults` and `## Per-car overrides` headings. Both headings are present (lines 31 and 328). Works. CLAUDE.md is honest that table-row validation is left to the loader.

### Skill -- `.claude/skills/ibt-inspect/SKILL.md`
Frontmatter-described summary skill consistent with `ingest/parser.py` capabilities. Stale wording at line 19: "check `pyproject.toml` once it exists" -- pyproject.toml exists and pins `pyirsdk>=1.3.5`.

### Skill -- `.claude/skills/add-constraint/SKILL.md`
User-invocable-only (`disable-model-invocation: true`). Schema rules consistent with current `constraints.md` shape. Per-car heading casing (`### bmw`, `### acura`) matches reality. Documented in CLAUDE.md.

### Subagent -- `.claude/agents/setup-justifier.md`
Read-only (`Read, Grep, Glob`); verifies four §7 fields per parameter. Matches the contract enforced in `explain/justification.py::SetupJustification` and rendered by `explain/render_text.py` (the `--detailed` block format). Agent description does NOT make explicit that it must consume `--detailed` output rather than the default narrative; the narrative compresses sensitivity into prose ("Effect:" / "Trade:" / "Watch most:") that this gate cannot parse.

### Subagent -- `.claude/agents/physics-fit-validator.md`
Tools `Read, Bash, Grep, Glob`. Relies on `physics/io_log.accuracy_log.parquet` and `Confidence.derive` which exist. Documented.

### MCP -- context7
User-scoped (no project-scope config to verify). Documented in CLAUDE.md.

## Gaps

1. **MAJOR** -- `CLAUDE.md:74-78` -- "Trust radius + --explore N" describes the pre-`318d91d` per-target-track empirical envelope. Update to "global pooled envelope across every track for the car" and re-examine the "VISION §3 honesty rule" framing.
2. **MAJOR** -- `CLAUDE.md` (no current line; new section needed) -- `--reset` mode is undocumented. Document the full-envelope search, the 30% widen factor, the confidence downgrade, on-track verification expectation. Source: `cli/recommend.py:111-122`, `physics/recommend.py:529-549`.
3. **MAJOR** -- `CLAUDE.md` (new section) -- `optimize calibrate` subcommand is undocumented. Add to `## Commands` and `## Project automations`. Source: `cli/calibrate.py`, registered `cli/__init__.py:60`.
4. **MINOR** -- `CLAUDE.md` (Commands block + known-regression blurb at line 162) -- new recommendation filename convention `<car>-<short-track>-<mode>[-<fuel>L]-<MMDD>-<HHMM>.<ext>` not described.
5. **MINOR** -- `CLAUDE.md:33` -- `PER_CAR_MODEL_CARS` is at `cli/recommend.py:57`, not `:54`.
6. **MINOR** -- `CLAUDE.md:70` -- "Race-mode fuel auto-pin (`cli/recommend.py:160+`)" -- block actually at `cli/recommend.py:213-244`.
7. **MAJOR** -- `docs/VISION_COMPLIANCE.md:11-15` -- 2026-05-06 follow-up audit omits `--reset`, `optimize calibrate`, always-global trust envelope (`318d91d`), cross-track fallback intermediate (`d64fd7b`), mode-tagged filename (`0a3f256`), telemetry-backed Why (`2355127`).
8. **MINOR** -- `GETTING_STARTED.md:69` -- `--reset` says "the per-track trust radius". Reword to "the trust radius" or "the corpus trust radius".
9. **MINOR** -- `GETTING_STARTED.md` Useful options -- `--output-file` and `-` to suppress not documented.
10. **MINOR** -- `docs/VISION_ADHERENCE_REPORT.md` vs `docs/VISION_COMPLIANCE.md` -- ambiguous which is the live status surface; both updated 2026-05-06. Pick one or merge.
11. **MINOR** -- `docs/superpowers/specs/2026-04-28-cli-design.md` (and the other five 2026-04-28 specs) -- add a `Status: historical` header.
12. **MINOR** -- `docs/superpowers/plans/2026-05-04-percar-correctness.md` -- add `Status: superseded by 318d91d (2026-05-06)`.
13. **MINOR** -- `docs/PARAMETER_VALIDATION.md:1-3` -- add a "Generated YYYY-MM-DD against `<sha>`" line.
14. **MINOR** -- `.claude/skills/ibt-inspect/SKILL.md:19` -- "check `pyproject.toml` once it exists" -- pyproject.toml exists; pin `pyirsdk` per pyproject.
15. **MINOR** -- `.claude/agents/setup-justifier.md` -- add "Invoke against `--detailed` output; the default narrative compresses sensitivity into prose and will fail this gate."
16. **MINOR** -- `cli/calibrate.py:165-168` (Click `--output-file` help) says default is `<car>_<track>_calibrate[_status]_<YYYY-MM-DD>_<HHMM>.txt`; `_maybe_save` (line 367-370) actually writes `<car>-<short-track>-cal[-status]-<MMDD>-<HHMM>.txt`. Internal code/help-text inconsistency, not a doc gap, but surfaces to users via `--help`.

## Evidence

**Code -> doc mismatches (claim vs reality):**
- `CLAUDE.md:33` "PER_CAR_MODEL_CARS (`recommend.py:54`)" -> actual `recommend.py:57`.
- `CLAUDE.md:70` "(`cli/recommend.py:160+`)" -> actual `cli/recommend.py:213-244`.
- `CLAUDE.md:76` "values the driver has actually tried on the target track" -> actual: global pooled across every track per `physics/recommend.py:494-505` docstring.
- `CLAUDE.md` no `--reset` / `calibrate` / new filename -> all merged before doc's stated 2026-05-06 cutoff (commits `87289a8`, `7e15db0`, `0a3f256`).
- `GETTING_STARTED.md:69` "per-track trust radius" -> global since `318d91d`.
- `cli/calibrate.py:165` Click help vs `cli/calibrate.py:367` actual filename -> divergent.
- `recommendations/` directory contains both legacy `bmw__spa_2024_up__20260506-053645.txt` and new `bmw-spa-reset-0506-1029.txt` / `ferrari-spa-race-0506-1026.txt` -- confirms cutover.

**pyproject.toml cross-check:**
- `[project.scripts] optimize = "racingoptimizer.cli:main"` -> matches all docs ("`uv run optimize ...`").
- `[tool.pytest.ini_options] markers = ["slow: ..."]` -> matches CLAUDE.md `pytest -q -m "not slow"` recipe.
- `requires-python = ">=3.12"` -> not contradicted by any doc.
- Dependencies (`pyirsdk`, `polars`, `pyarrow`, `click`, `numpy`, `scipy`, `scikit-learn`) -> all imported and exercised.
- Single console script `optimize` -> no doc claims another.

## Recommended next actions

- Patch `CLAUDE.md`:
  1. Rewrite `## Trust radius + --explore N` to describe the global pooled envelope (cite `physics/recommend.py:494-505`).
  2. Add `## Reset mode` covering `--reset`, the 30% widen, the confidence downgrade, on-track verification.
  3. Add `## Active-learning probes` covering `optimize calibrate` (default + `--status` + `--targets`).
  4. Update the `## Commands` block to include `calibrate` and `--reset`.
  5. Refresh both file:line cites (PER_CAR_MODEL_CARS line and the Race-mode fuel block line).
  6. Document the new recommendation filename in the renderer-contract section (and update the known-regression blurb).
- Patch `GETTING_STARTED.md:69` -- replace "the per-track trust radius" with "the trust radius".
- Patch `docs/VISION_COMPLIANCE.md` follow-up audit -- append `--reset`, `calibrate`, always-global trust envelope, telemetry-backed Why under §3 / §5 / §7 / §8 buckets.
- Add `Status: superseded by 318d91d (2026-05-06)` header to `docs/superpowers/plans/2026-05-04-percar-correctness.md`.
- Add `Status: historical -- see GETTING_STARTED.md for the live CLI surface` to the six 2026-04-28 specs.
- Decide between `docs/VISION_COMPLIANCE.md` and `docs/VISION_ADHERENCE_REPORT.md` as the live status surface; mark the other historical or merge.
- Add "Generated YYYY-MM-DD against commit `<sha>`" header to `docs/PARAMETER_VALIDATION.md`.
- Reconcile `cli/calibrate.py:165-168` Click help-text default-filename description with `_maybe_save` (line 370).
- Add a one-line note to `.claude/agents/setup-justifier.md` directing the agent to consume `--detailed` output.
- Drop the "once it exists" wording in `.claude/skills/ibt-inspect/SKILL.md:19`.
