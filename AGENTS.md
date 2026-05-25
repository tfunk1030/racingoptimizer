## Learned User Preferences

- Primary product bar: a physics-based GTP setup optimizer that is fully accurate, completely built, correlated, and calibrated—not merely a working empirical recommender.
- User validates recommendations by applying `[OPT]` values in iRacing; garage readout mismatch (especially static ride height, even a few mm) is treated as the optimizer being wrong.
- When parameters are skipped or pinned, distinguish untrained (no `constraints.md` bounds), global pinned (no corpus variance), and thin-track pinned (fewer than 3 distinct values at target track)—user expects the full garage to be optimizable.
- Reject pinning as a substitute for fixing scoring; prefer data-driven bounds (within-track lap evidence at N=2, local min/max) over global corpus pins or cross-track extrapolation.
- Garage requires LF=RF and LR=RR camber symmetry; asymmetric camber recommendations are wrong and skew corner weights.
- Static ride height mismatch is surfaced via **warn-only envelope checks** on predicted readouts after recommend—not hard clamping of user inputs.
- Run `optimize learn` before `optimize recommend` when new IBTs exist on disk for the target (car, track); unlearned files make the past setup, coverage table, and `(was X)` deltas wrong.
- When within-track coverage is thin, prefer `optimize calibrate` probes over trusting bulk race `[OPT]` moves; validate platform changes against garage readouts incrementally.

## Active accuracy gap (2026-05-24)

Current state — W1 through W4 of `docs/accuracy-rebuild-2026-05-24/PLAN.md` shipped 2026-05-24. P0/P1/P3 items closed (with P1.2 / P1.3 partial); P2.1 deferred and P2.2 skipped in favour of P2.3.

Closed:

- **P0.1 — `per_track_residuals` retired.** Predict path no longer reads it; fitter no longer computes it. `FITTERS_LAYOUT_VERSION = 10` invalidates pre-2026-05-24 pickles.
- **P0.2 — Deterministic kinematic static-RH fit.** `physics/static_rh_kinematic.py` per-car closed-form ridge OLS gated on R² ≥ 0.98. Wired into `predict_setup_readouts` (kinematic wins for the four `setup_static_*_ride_height_mm` channels; surrogate fallback for everything else). `physics/recommend.py::_kinematic_static_rh_ready` skips the legacy k-NN `enforce_static_rh_feasible` repair when the kinematic fit ships; legacy k-NN remains as fallback for cars whose fit refused. `cooptimize_tb_for_static_rh` (post-DE TB trim) still applied unconditionally.
- **P0.3 — Sensitivity floor.** Below `_SENSITIVITY_FLOOR = 0.005` on both ±1 step → reset to baseline; surfaced under NOTES via `SetupRecommendation.suppressed_below_sensitivity`.
- **P0.4 — Phantom corner-0 + per-corner dedupe.** `is_real_corner_archetype` gates both guardrail paths; `guardrail_warnings_for_setup` collapses to one line per corner.
- **P1.1 — Per-channel held-out gate.** `scripts/holdout_accuracy_gate.py::_PER_CHANNEL_THRESHOLDS` carries the PLAN §3 P1.1 thresholds; `main()` returns 1 on ANY car failing ANY non-driver-input channel (aggregate gate stays informational).
- **P1.4 — `_validate_pickle_slots`.** Refuses revives with wrong-typed slots; points at `--no-cache`.
- **P2.3 — Inverse-track-sample-count training weights.** Forest fitter honours `sample_weight = 1 / sqrt(n_track_rows)` joined via session_id → track. Ridge / GP silently accept the kwarg (no-op for session-invariant channels).
- **P2.4 — `phase_duration_s` at fit time.** `CORNER_ARCHETYPE_COLUMNS` extended; `ENV_FEATURE_SCHEMA_VERSION_PER_CAR` 5 → 6.
- **P3.1 — Briefing header carries per-channel error budget.** Loads `holdout_accuracy_latest.json` and renders peak lateral G / understeer / static front RH / damper force p99 lines. Falls back to legacy `Confidence: ...` line on missing rows.
- **P3.2 — Watch-most picker normalisation.** `_dominant_impact_corner` and `_telemetry_why` divide each impact by the corner's pre-filter spread in the candidate pool.
- **P3.3 — Thin-corpus refusal banner.** `cli/recommend.py::_is_thin_corpus_for_recommend` (n_prod < 20 OR `axle_grip_ceilings is None`) gates `recommend_cmd`; refusal points at `optimize calibrate <car> <track>` and returns before DE.

Still open / partial:

- **P1.2 — Lap-time Spearman gate (PARTIAL).** Helpers + qualifying-pair filter ship in `scripts/lap_time_correlation_gate.py`; LOSO orchestration is a placeholder pending offline run (~2.5 hr/car-track on this corpus).
- **P1.3 — Hybrid vs surrogate-only A/B (PARTIAL).** Existing `tests/physics/test_hybrid_heldout_ab.py` gates non-regression invariants (identical key sets, finite positive totals, |hybrid - surrogate| / surrogate ≤ 50 %). CI YAML flip + per-car asymmetric "hybrid doesn't lose" assertion still pending.
- **P2.1 — Curb / off-line row masking (DEFERRED).** Needs `TrackModel.bump_map` API addition; bigger than W3 budget. Picked up in next pass.
- **P2.2 — Per-track random intercepts.** Skipped in favour of P2.3 (cheap weights). Math still lives in `physics/bayes_retrofit.py` for a future upgrade.

## Learned Workspace Facts

- Static garage `RideHeight` is `user_settable=False`; the DE path scores dynamic `*_ride_height_mean_mm` telemetry, not static YAML readouts.
- Static RH: observation envelopes in `constraints.md`; post-hoc `_static_ride_height_envelope_warnings()` warn-only; DE also applies soft penalty via `static_rh_platform_penalty()` when predicted static RH outside 30–80 mm (guardrail, not mm-perfect physics readout model).
- Three ride-height semantics in output: wheel dynamic RH (telemetry), static garage RH (`[predicted]` envelope check), and AeroCalculator readout (different reference—do not conflate with wheel telemetry).
- Corner weights are calculated readouts (`fittable=False`); setup card shows `[readout]`, not `[OPT]`.
- Setup card `[OPT]`/`[OPT mirror]` by ontology `json_path`; LF/LR camber and per-corner dampers mirror to RF/RR at recommend (`setup_symmetry.py`); ingest still reads all four camber corners independently from IBT YAML.
- Parameters without finite bounds in `constraints.md` are excluded from fit and DE search via the `fittable_parameters()` bounds gate.
- Default DE uses hybrid scoring (`hybrid_score()`); `--surrogate-only` reverts to surrogate + axle guardrail penalty only.
- Race DE within-track evidence gate (N≥3): N=1 pins; N=2 caps to locally observed min/max (lap-time anchor when available); N≥3 uses global corpus envelope.
- `pinned_within_track_thin` (local coverage) and `pinned_to_observed_median` (global corpus variance) are distinct pin reasons in briefing NOTES and stderr warnings.
- Briefing header confidence is track-wide, not per-parameter; the track coverage line reports how many parameters have 3+ distinct values at the target track.
- Acura: HF/TR/FROLL/RROLL shock deflection aliases; `spring_perch_offset_rear_mm` → HeavePerchOffset ("Rear heave perch"); IBT YAML ARB `Disconnected` aliases to ontology `Disconnect` (index 0).
- `optimize calibrate` reads coverage live from ingested sessions (not stale fit pickle); excludes brake ducts, throttle map, tyre cold pressure (152 kPa pin), and brake bias from probes.
- `PhysicsModel` slots are append-only (frozen+slots dataclass); inserting a field shifts every later slot's pickle position and silently corrupts revives. `_repair_legacy_slot_shift` in `model.py` rescues pre-v7 pickles where `axle_grip_ceilings` landed in `per_track_residuals` etc.; type-safety on slot revive is P1.4 in the accuracy-rebuild plan.
- Acura at < 20 production sessions has `axle_grip_ceilings=None` → hybrid scoring silently collapses to surrogate-only; the user is currently shown a "physics-based" briefing with no physics anchor. Plan P3.3 adds an explicit thin-corpus refusal banner.
- Corner 0 leaks into `_axle_guardrail_penalty` on multiple tracks as "T0 ... physics-vs-surrogate divergence" across all five phases — schedule artifact (start/finish straight or pit-out), not a real corner. Plan P0.4 filters at the guardrail step.
