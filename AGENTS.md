## Learned User Preferences

- Primary product bar: a physics-based GTP setup optimizer that is fully accurate, completely built, correlated, and calibrated—not merely a working empirical recommender.
- User validates recommendations by applying `[OPT]` values in iRacing; garage readout mismatch (especially static ride height, even a few mm) is treated as the optimizer being wrong.
- When parameters are skipped or pinned, distinguish untrained (no `constraints.md` bounds), global pinned (no corpus variance), thin-track pinned (fewer than 3 distinct values at target track), and Bayes near-constant pin (track `mean_std` vs global empirical range below `_NEAR_CONSTANT_FRACTION` despite multiple track values)—user expects the full garage to be optimizable; do not label Bayes pins as "no corpus variance".
- Reject pinning as a substitute for fixing scoring; prefer data-driven bounds (within-track lap evidence at N=2, local min/max) over global corpus pins or cross-track extrapolation.
- Garage requires LF=RF and LR=RR camber symmetry; asymmetric camber recommendations are wrong and skew corner weights.
- Static ride height mismatch is surfaced via **warn-only envelope checks** on predicted readouts after recommend—not hard clamping of user inputs.
- Run `optimize learn` before `optimize recommend` when new IBTs exist on disk for the target (car, track); unlearned files make the past setup, coverage table, and `(was X)` deltas wrong.
- When within-track coverage is thin, prefer `optimize calibrate` probes over trusting bulk race `[OPT]` moves; validate platform changes against garage readouts incrementally.

## Learned Workspace Facts

- Static garage `RideHeight` is `user_settable=False`; the DE path scores dynamic `*_ride_height_mean_mm` telemetry, not static YAML readouts. Post-recommend `[predicted]` static RH uses warn-only envelope checks; DE applies soft `static_rh_platform_penalty()` when predicted static RH is outside 30–80 mm.
- Three ride-height semantics in output: wheel dynamic RH (telemetry), static garage RH (`[predicted]` envelope check), and AeroCalculator readout (different reference—do not conflate with wheel telemetry or static `[predicted]`).
- Corner weights are calculated readouts (`fittable=False`); setup card shows `[readout]`, not `[OPT]`.
- Setup card `[OPT]`/`[OPT mirror]` by ontology `json_path`; LF/LR camber and per-corner dampers mirror to RF/RR at recommend (`setup_symmetry.py`); ingest still reads all four camber corners independently from IBT YAML.
- Parameters without finite bounds in `constraints.md` are excluded from fit and DE search via the `fittable_parameters()` bounds gate.
- Default DE uses hybrid scoring (`hybrid_score()`); `--surrogate-only` reverts to surrogate + axle guardrail penalty only.
- Race DE within-track evidence gate (N≥3): N=1 pins; N=2 caps to locally observed min/max (lap-time anchor when available); N≥3 uses global corpus envelope.
- `pinned_within_track_thin` (local coverage) and `pinned_to_observed_median` (global corpus variance) are distinct pin reasons in briefing NOTES and stderr warnings.
- Briefing header confidence is track-wide, not per-parameter; the track coverage line reports how many parameters have 3+ distinct values at the target track.
- Acura: HF/TR/FROLL/RROLL shock deflection aliases; `spring_perch_offset_rear_mm` → HeavePerchOffset ("Rear heave perch"); IBT YAML ARB `Disconnected` aliases to ontology `Disconnect` (index 0). Setup spec in `docs/cars/acura_arx06.md` / `physics/aero_targets.aero_targets_for("acura")` — peak-downforce RH targets, rake/balance-lever vocabulary; defer to surrogate for other cars.
- `optimize calibrate` reads coverage live from ingested sessions (not stale fit pickle); excludes brake ducts, throttle map, tyre cold pressure (152 kPa pin), and brake bias from probes.
- `scripts/holdout_accuracy_gate.py` refits in memory each run (does not load or update the recommend pickle); rerun `optimize` after `learn` or cache invalidation, not because the gate finished. Use `--car` for single-car runs; `--eval-track belleisle --in-sample` scores Belle Isle with the full corpus training pool (matches recommend); without `--in-sample` trains excluding Belle Isle (cross-track extrapolation stress test).
