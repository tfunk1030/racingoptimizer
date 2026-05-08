# Punch list -- post-session synthesis (2026-05-07)

Consolidated leverage-ranked roadmap derived from the 12-unit audit
(`docs/audit_2026-05-06/01_ingest.md` through `12_docs_tooling.md`).
Items already closed by session commits (`a48b686..3c098bc`) have been
removed; what remains is what's still open.

Closed by this session (for the record):
* GT3 routing pollution -> `1a8c9a3` (CAR_PREFIX_MAP cleaned)
* `apply_quality_mask` never called in production -> `e90e8fd` (wired into `learn` with `--no-quality-masks` opt-out)
* `weight_corners` dead for v4 cars -> `e90e8fd` (corner-duration weighting consumed)
* Telemetry Why-line dead in production -> `f7fe058` (`schedule=` passed to `render_narrative`)
* `_CAR_FEEL["heave_spring"]` unreachable -> `f7fe058` (subtype="front-heave")
* Em-dash / middle-dot cp1252 corruption -> `f7fe058`, `06f0c23` (ASCII sweep, legend `?` fix)
* Mirror precedence -> `06f0c23` (mirror checked before opt_match)
* Watch-most + Why-line all-T5 dominance -> `7be3017`, `5f305b4` (`_FAMILY_PREFERRED_PHASES` filter)
* `--reset` / `--explore` / `optimize calibrate` / race-fuel auto-pin / `--detailed` test gaps -> `85b64f8`
* CLAUDE.md trust-radius / `--reset` / `calibrate` / filename docs -> `f7fe058`, `3c098bc`
* `--staged` 5-stage DE driver shipped -> `00ca849`
* `ConstraintsTable.with_pin` factory (closes `_by_car` leak) -> `00ca849`

## Tier 1 -- highest leverage (active production gaps)

| # | Item | File:line | Why this matters | Effort |
|---|------|-----------|------------------|--------|
| T1.1 | `fit(track_model=...)` parameter still unused even after `apply_quality_mask` wiring | `physics/fitter.py:217-243` | `cli/recommend.py:1115` builds the TrackModel and threads it in; the fitter ignores it. Quality mask now reaches disk via ingest but the fitter doesn't read TrackModel directly. Decide: drop the param OR have it consume curb / off-track masks for the fit-time filter. | 1 day |
| T1.2 | `--json` stderr-mixing bug latent in production | `cli/recommend.py:407-434` | Auto-save banner unconditionally written to stderr even under `--json`. Users piping JSON to `jq` see the banner appended to JSON. Fix: when `as_json AND output_file is None`, default `output_file = Path("-")`. | 30 min |
| T1.3 | `wet_baselines` ignores per-car corpus baselines | `physics/wet_mode.py:74` | Wet-mode swaps to `baselines_for(car)` with no frame -> always cold-start defaults. The corpus-derived baselines never reach wet predictions. | 2 hr |
| T1.4 | `_score_breakdown_per_car` returns 0.0 on empty `state.states` | `physics/score.py:709-711` | Same value as a maximally-bad real prediction. Optimizer can't distinguish "no model coverage at this corner-phase" from "actively predicted disaster" -> may pick setups whose only merit is that the model has no signal. | 1 day |
| T1.5 | `tests/cli/test_per_car_smoke.py:61` asserts on stale `[confidence:` tag | `tests/cli/test_per_car_smoke.py:61` | The assertion only fires under `--detailed`. Default narrative emits `Confidence: <regime>` once at top with no per-parameter `[confidence:` tag. Either silent test gap (passing for the wrong reason) or about-to-fail. Fix: assert on `OVERALL DIRECTION` or `CHANGES (`. | 15 min |

## Tier 2 -- coverage gaps for shipped features

| # | Item | Suggested test path |
|---|------|---------------------|
| T2.1 | `fit_per_car` no direct test coverage | `tests/physics/test_fit_per_car.py` (new) -- conftest factory currently calls `fit()` instead |
| T2.2 | `render_narrative` no test file | `tests/explain/test_narrative.py` (new) -- ASCII guard, OVERALL DIRECTION aggregation, `_CAR_FEEL` coverage per active family |
| T2.3 | `_maybe_borrow_cross_car_track` untested | `tests/cli/test_cross_car_schedule_fallback.py` (new) |
| T2.4 | `predict_setup_readouts` untested | `tests/explain/test_full_setup_card.py::test_predicted_static_rh_lines_get_predicted_tag` (extend) |
| T2.5 | `tests/physics/test_per_car_recommend.py` is `pytest.mark.slow` -> merge gate skips it | Promote one BMW Sebring case to fast suite; keep full per-car loop as slow |

## Tier 3 -- code-health (reduce drift surface)

| # | Item | Approach |
|---|------|----------|
| T3.1 | ~120-line copy-paste between `fit` (`fitter.py:214-422`) and `fit_per_car` (`fitter.py:626-868`) | Extract `_train_joint(...)` helper. Three real differences (group-by key, family_kind="rf" force, per-track observed build) become parameters. |
| T3.2 | Slug-resolution duplicated 4x in `cli/recommend.py` | Single `_match_track_slug(needle, available)` helper. Drift already started (one site sorts alphabetically before substring scan, others don't). |
| T3.3 | Three duplicate IBT-channel <-> field mappings | `context/environment.py:28-45`, `cli/recommend.py:1249-1264`, `physics/fitter.py:162-178`, `explain/render_json.py:112-128` -- consolidate to one source. |
| T3.4 | `Confidence` regime thresholds hardcoded magic numbers | `confidence/confidence.py:65-75` -- promote `30`, `0.5`, `0.2` to named module constants. |
| T3.5 | `_CAR_FEEL` hardcodes Spa landmarks ("Eau Rouge / Pouhon", "Kemmel compression", "T9, T13") | Genericize phrasing or drive from `_dominant_impact_corner`. Currently misleading at non-Spa tracks. |
| T3.6 | `_CAR_FEEL` missing per-axle entries (rear torsion, ride_height, corner_weight) | Add the missing keys; verify via "every fittable user-settable param resolves to a hit" test. |
| T3.7 | `bare except Exception` in `justification.py:251, 292` | Narrow to specific exception types so legitimate scoring failures aren't masked as "neutral baseline". |
| T3.8 | `EnvironmentFrame.from_partial_row` dead production code | Delete or wire up. CLI uses `_env_from_overrides` instead. |
| T3.9 | Implicit field-order coupling between `EnvironmentFrame` and `_env_to_array` | Add a "field order is part of v4 fitter contract" comment + round-trip regression test. |
| T3.10 | `CANONICAL_CARS` defined three places | Single source of truth (`cli/recommend.py:47`); `tests/cli/conftest.py` and `_maybe_borrow_cross_car_track` both import from it. |

## Tier 4 -- blocked / long-term (need external data or deeper work)

| # | Item | Blocker |
|---|------|---------|
| T4.1 | 5 unbounded `constraints.md` families (corner weights, diff coast/power %, brake duct F+R, throttle/brake mapping) | Needs iRacing UI capture from user. |
| T4.2 | Slider deflection prediction + 45 mm tech-rule validation | Needs `setup_static_heave_slider_defl_mm` as a fitted readout channel; ~half-day of telemetry scaffolding. |
| T4.3 | Directional wind decomposition (`decompose_wind`, `balance_shift_pct`) stranded | Needs per-corner heading data the corner schedule doesn't carry yet. |
| T4.4 | Per-car damper coefficients are seeded estimates | Stage-4 physics work; needs real damper-curve capture from iRacing UI. |
| T4.5 | Archetype provenance asymmetry (training per-session window vs predict per-track group) | `physics/fitter.py::_attach_corner_archetypes` vs `physics/corner_schedule.py::build_corner_schedule`. Deeper refactor; reach unanimous semantics. |
| T4.6 | Per-corner-LEVERAGE-not-DURATION weighting for Watch-most / Why-line | Picker still lands on T5 because corner-duration dominates. Normalizing impact by corner-weight inside `_dominant_impact_corner` would route more parameters to non-T5 corners; needs careful design to avoid breaking score function intent. |
| T4.7 | `--symmetric-dampers` CLI flag for ovals / asymmetric tracks | Today damper L/R mirroring is unconditional in the renderer. Per-corner DE search runs anyway, so the underlying values exist; need a flag to surface them when banking actually warrants asymmetry. |

## Order of operations recommended

1. **T1.5** (15 min) -- silent test gap; cheapest to close
2. **T1.2** (30 min) -- production JSON-pipe bug
3. **T2.5** (1 hr) -- promote one fast per-car-recommend case to merge gate
4. **T1.3, T1.4** (2-3 hr each) -- wet baselines + score-vs-empty distinction
5. **T1.1** (1 day) -- decide fit(track_model=...) consume-or-drop
6. **Tier 2 tests** in parallel as time allows
7. **Tier 3** when touching the relevant code anyway
8. **Tier 4** parked until external data arrives

## Verification rule

After closing any item: cross-check the per-slice audit doc that flagged it, mark it [resolved by `<commit>`] in this file, and update the status row in `00_summary.md`.
