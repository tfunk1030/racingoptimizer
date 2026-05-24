## Learned User Preferences

- Primary product bar: a physics-based GTP setup optimizer that is fully accurate, completely built, correlated, and calibrated—not merely a working empirical recommender.
- User validates recommendations by applying `[OPT]` values in iRacing; garage readout mismatch (especially static ride height, even a few mm) is treated as the optimizer being wrong.
- When parameters are skipped or pinned, distinguish untrained (no `constraints.md` bounds) from pinned (no corpus variance)—user expects the full garage to be optimizable.
- Static ride height mismatch is surfaced via **warn-only envelope checks** on predicted readouts after recommend—not hard clamping of user inputs.

## Learned Workspace Facts

- Static garage `RideHeight` is `user_settable=False`; the DE path scores dynamic `*_ride_height_mean_mm` telemetry, not static YAML readouts.
- Static ride height rows in `constraints.md` are observation envelopes only; `_static_ride_height_envelope_warnings()` flags predicted readouts outside the envelope.
- Corner weights are calculated readouts (`fittable=False`); setup card shows `[readout]`, not `[OPT]`.
- Setup card `[OPT]` replaces rows by exact ontology `json_path`; rear coil rate and spring perch OPT appear on `Chassis.LeftRear` with `RightRear` tagged `[OPT mirror]` for perch offset and coil rate.
- Parameters without finite bounds in `constraints.md` are excluded from fit and DE search via the `fittable_parameters()` bounds gate.
- Default DE uses hybrid scoring (`hybrid_score()`); `--surrogate-only` reverts to surrogate + axle guardrail penalty only.
