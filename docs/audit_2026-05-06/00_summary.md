# Codebase audit summary -- 2026-05-06

12 parallel audit units across slices A through F + cross-cutting + tests +
docs/automations. All workers ran in worktree-isolated sandboxes that
denied Bash/PowerShell/Write, so the per-unit reports were captured inline
and consolidated here in the main session. No code changes in this audit
batch -- findings only.

## Per-slice grades

| # | Unit | Grade | High-leverage finding |
|---|------|-------|----------------------|
| 1 | Slice A -- Ingest | PASS | Aston Martin GT3 silently mapped to BMW (`detect.py:23`) |
| 2 | Slice B -- Corner-phase | PASS | `segment_lap(track_model=...)` still raises `NotImplementedError` despite slice D merged |
| 3 | Slice C -- Aero maps | PASS | `AeroSurface(baseline_air_density=)` constructor arg never read by `interpolate()` |
| 4 | Slice D -- Track model | PARTIAL -- 2 HIGH | **`apply_quality_mask` never called in production**; `fit(track_model=...)` ignored |
| 5 | Slice E1 -- Fitter / Model | PARTIAL -- 1 MAJOR | `fit_per_car()` (BMW/Cadillac/Ferrari path) has zero direct test coverage |
| 6 | Slice E2 -- Recommend / DE | PARTIAL | `--reset` and `--explore` untested; per-car recommend smoke is `slow`-marked |
| 7 | Slice E3 -- Score + conditions | PARTIAL -- 5 MAJOR | `weight_corners` dead for v4 cars; `wet_baselines` ignores corpus baselines; `decompose_wind` stranded |
| 8 | Slice F1 -- Explain / Render | PARTIAL -- 2 CRITICAL | **Telemetry-backed Why line dead in production for v4 cars**; `_CAR_FEEL["heave_spring"]` unreachable |
| 9 | Slice F2 -- CLI surface | PARTIAL -- 5 MAJOR | `optimize calibrate` zero tests; `--json` stderr-mixing live; slug-resolution duped 4x |
| 10 | Cross-cutting modules | PARTIAL -- 6 MAJOR | 5 unbounded constraints still TODO; 3 duplicate IBT-channel <-> field mappings; field-order coupling untested |
| 11 | Test coverage + suite | PARTIAL -- 3 CRITICAL | 11 recently landed features ship with zero direct tests |
| 12 | Docs + automations | PARTIAL -- 4 MAJOR | CLAUDE.md missing `--reset`, `calibrate`, new filename; trust-radius section describes pre-`318d91d` rule |

## Convergent themes (cross-slice)

1. **Track-quality mask is a no-op in production** -- Slice D's `apply_quality_mask` has zero callers in `src/`; Slice B's `corner/states.py` confirms `data_quality_clean_frac` averages an all-True placeholder. The "Track model is load-bearing for data quality" claim in CLAUDE.md is currently false. Same root cause from two angles.

2. **Telemetry-backed Why line dead for BMW/Cadillac/Ferrari** -- `cli/recommend.py:381` calls `render_narrative(...)` without `schedule=schedule`, so every v4 `_telemetry_why` prediction silently raises and is swallowed.

3. **`weight_corners` skipped for v4 cars** -- BMW/Cadillac/Ferrari (3 of 5) silently use uniform weights -- VISION §6 time-sensitivity weighting only fires for Acura + Porsche.

4. **`fit_per_car()` (the v4 path that ships to BMW/Cadillac/Ferrari users) has zero direct test coverage** -- `test_per_car_fit_predict.py` is named for v4 but its conftest factory calls `fit()` (v3).

5. **Aston Martin GT3 -> BMW silent routing** would pollute the BMW corpus on ingest. Surfaced independently by Slice A and Slice F2.

6. **`--reset`, `--explore`, `optimize calibrate`, narrative renderer, race-fuel auto-pin, `_short_track`, `predict_setup_readouts`** all ship without direct unit tests.

7. **Unicode em-dashes in narrative + setup card** cause `?` corruption on Windows cp1252; visible in `bmw-spa-cal-0506-1015.txt:72`.

8. **CLAUDE.md and VISION_COMPLIANCE.md** are missing the last 24h of feature work despite the doc claiming 2026-05-06 currency.

## Severity breakdown

- **CRITICAL** (3): Telemetry-backed Why line dead in production (F1); `optimize calibrate` zero tests (Tests, F2); `--reset` zero tests (Tests, E2).
- **HIGH** (2): `apply_quality_mask` never invoked in production (D); `fit(track_model=...)` parameter unused (D).
- **MAJOR** (~30): see per-unit reports.
- **MINOR** (~80): see per-unit reports.

## Suggested follow-up sequence

1. **Wire the track-quality mask into production** (`apply_quality_mask` at end of `optimize learn` or in `cli/recommend.py:1110+`). Decides whether `physics/fitter.py:217` `track_model=` parameter is consumed or dropped. Single change unlocks VISION §1's data-quality contract.
2. **Pass `schedule=schedule` to `render_narrative`** (`cli/recommend.py:381`). One-line fix; restores the telemetry-backed Why line for v4 cars. Add an integration test asserting at least one `Why:` line in a BMW briefing.
3. **Fix Aston Martin -> BMW silent routing** (`ingest/detect.py:23`). Either remove the placeholder or raise `UnknownCarError` so misrouted IBTs land as `partial`/`unknown` instead of poisoning the BMW corpus.
4. **Wire `weight_corners` for v4 cars**. BMW/Cadillac/Ferrari currently ignore VISION §6; threading `CornerScheduleEntry.archetype` into `_cached_weights` closes the gap.
5. **Refresh CLAUDE.md** with `--reset`, `optimize calibrate`, the always-global trust envelope, and the new recommendation filename. Source-of-truth for future agents.
6. **Add tests for `--reset`, `--explore`, `optimize calibrate`, race-fuel auto-pin, narrative renderer**. Each of these is small and self-contained.
7. **ASCII sweep** of `narrative.py` and `full_setup_card.py` -- replace em-dashes / middle-dots with `--` / `|` to fix cp1252 corruption.
8. **Capture the 5 unbounded `constraints.md` rows** from the iRacing UI: corner weights (4), diff coast/power, brake duct front+rear, throttle/brake mapping.

## Per-unit findings

See `01_ingest.md` through `12_docs_tooling.md` for full per-slice file:line citations, wiring diagrams, and recommended actions.

## Status update -- 2026-05-06 evening

Followup commits landed against the 8-step "Suggested follow-up sequence" above:

| # | Item | Status | Commit |
|---|------|--------|--------|
| 1 | Wire `apply_quality_mask` into production | done | `e90e8fd` |
| 2 | Pass `schedule=` to `render_narrative` (telemetry Why-line for v4 cars) | done | `f7fe058` |
| 3 | Fix Aston Martin -> BMW silent routing (and drop all GT3 routing -- GTP only) | done | `1a8c9a3` |
| 4 | Wire `weight_corners` for v4 cars | done (corner-duration weighting) | `e90e8fd` |
| 5 | Refresh CLAUDE.md (`--reset`, `optimize calibrate`, always-global trust envelope, new filename convention) | done | `f7fe058` |
| 6 | Add tests for `--reset`, `--explore`, calibrate, race-fuel auto-pin, narrative renderer | mostly done -- pin/reset/explore tests green; CLI tests in flight | (phase 3 commit) |
| 7 | ASCII sweep on `narrative.py` and `full_setup_card.py` | done | `f7fe058` |
| 8 | Capture the 5 unbounded `constraints.md` rows from the iRacing UI | pending -- needs user data |

Bonus fixes:
* `_CAR_FEEL["heave_spring"]` keys were unreachable -- front heave-spring rendered with phase-themed defaults instead of the rich Effect/Trade table. Fixed via `("spring_rate", "front-heave", +/-)` keys + `_param_subtype` recognition (commit `f7fe058`).
* `tests/explain/test_full_setup_card.py:94+` was asserting `15.5` against a wing-angle field that step-snaps to `16` (step=1.0) -- pre-existing brittle test, updated to assert post-snap value (commit `f7fe058`).
* Renamed/short filename convention now also applies to `optimize calibrate` artefacts (`<car>-<short-track>-cal[-status]-<MMDD>-<HHMM>.txt`).

Still open:
* Slice F1 #4: `tests/cli/test_per_car_smoke.py:61` asserts `[confidence:` which only exists in `--detailed` output -- silent test gap (low-priority).
* Slice E1 #1: `fit_per_car()` direct test coverage -- conftest factory still calls `fit()` (v3) for the per-car fixture.
* Slice F2 #2: `--json` stderr-mixing live in production code -- saver block still writes `[saved to ...]` to stderr under JSON mode.
* Slice E3 #2-#5: directional wind decomposition stranded; `_score_breakdown_per_car` returns 0.0 for empty states (indistinguishable from real disaster).
* Slice 12 doc cleanups: 2026-04-28 specs unmarked as historical; `docs/VISION_COMPLIANCE.md` follow-up audit still missing 6 commits despite its 2026-05-06 currency note.

## Notes on this audit batch

The 12 audit workers all hit a sandbox lockdown that denied Bash, PowerShell, and Write. Each worker performed full read-only static analysis (Read + Grep + Glob) and returned its findings as inline text rather than committing files. The findings here are consolidated from those reports, written from the main session that retained Write access.

No commands that write to `corpus/` (`optimize learn`, `optimize <car> <track>`) were invoked by any worker -- this audit is read-only against the corpus.
