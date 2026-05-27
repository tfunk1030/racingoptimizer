The last 14 days shipped a lot of scaffolding (hybrid blend wiring, axle guardrails, per-car evaluator weights, static RH co-optimization, within-track-thin pins, garage symmetry, snap-to-step, calibrate live-coverage, brake-duct/TC ontology). The scaffolding is good engineering. But the underlying surrogate is not predicting setup→channel relationships any better than predicting the channel mean for several key chassis-balance channels, and one of the new modules (per_track_residuals) is mathematically broken in a way that suppresses setup sensitivity. That's why your most recent Belleisle race briefing shows +1 click -0.000 score on virtually every parameter while still moving 22 of 43 of them.

You don't have a calibrated physics optimizer right now. You have a well-instrumented noisy surrogate with multiple repair layers that hide the underlying signal-to-noise problem rather than fix it.

What the latest run actually tells us
From recommendations/acura-belleisle-race-0524-1808.txt:

Sensitivity is ±0.000 on essentially every move. The DE optimizer cannot distinguish ±1 click in either direction.
Header reads sparse (median n=194) which is internally contradictory — Confidence.derive short-circuits noisy at n<30. n=194 should be at minimum noisy, more likely dense. The header label is computing something different from the per-parameter regime.
Moved params: 0 sparse / 2 noisy of 22 changes — meaning 20 of 22 moves are tagged confident/dense by per-parameter regime, yet each has zero sensitivity. "Confident the param doesn't matter" is the only consistent reading, which means either the confidence label is wrong or the optimizer is searching parameters it cannot resolve. Both are problems.
Watch most: T18 mid-corner on virtually every parameter — the corner-duration weighting is still dominating the _FAMILY_PREFERRED_PHASES filter that was supposed to fix this.
T0 ... physics-vs-surrogate divergence in all five phases. Corner 0 is leaking into guardrail evaluation; this is likely the start/finish straight slot or pit-out, not a real corner.
Front pushrod -24.5 → -18.5 mm [OPT] but the [predicted] static LF ride height moves 30.0 → 30.1 mm. A 6 mm pushrod move that produces 0.1 mm of predicted static RH change means the readout fitter has a near-zero pushrod gradient. That violates simple suspension geometry.
Hard accuracy floor: docs/physics-rebuild/holdout_accuracy_latest.json
Every channel on every car is noisy. The damning numbers (normed_residual = RMSE/channel-std; ≥1.0 means worse than predicting the channel mean):

Car	Channel	normed_residual	Reading
Cadillac
understeer_angle_mean_rad
2.24
model is 2.24× worse than predict the mean
Cadillac
wheel_speed_max_diff_ms
3.18
catastrophic
Acura
steering_max_rad
1.18
worse than mean
Acura
understeer_angle_mean_rad
1.07
worse than mean
Ferrari
accel_lat_g_max
1.23
worse than mean on peak grip
Porsche
accel_lat_g_max
1.14
worse than mean on peak grip
BMW
accel_lat_g_max
1.00
no skill above mean
These channels feed axle_grip, understeer-driven Why-lines, and the entire balance evaluator. Coverage (CI containment) sits at 0.85–0.99 only because the CI is wide enough to swallow anything — that's not accuracy, that's calibrated pessimism.

Per CLAUDE.md the pass criteria are median coverage ≥0.50 and median normed_residual ≤2.0. That gate passes worse-than-mean predictions on grip-balance channels. The gate is too lax to detect the actual problem.

The most damaging recent change: per_track_residuals is not a residual
In physics/fitter.py:1023-1054 (the new "cross-track de-confounding" layer):


diff_fitter.txt
Lines 86-120
+    per_track_residuals: dict[str, dict[str, float]] = {}
...
+                track_med = float(np.median(vals))
+                track_res[ch] = track_med - global_medians[ch]
And the predict side in physics/model.py:_predict_v4:


diff_model.txt
Lines 86-96
+            if track:
+                track_res = getattr(self, "per_track_residuals", {}).get(track, {})
+                if channel in track_res:
+                    mean_value += float(track_res[channel])
This is wrong in two ways:

It is not a residual. track_median(actual) − global_median(actual) is the track-output bias relative to the corpus mean. The surrogate is already trained on data containing those rows, so its natural prediction at Belleisle features already shifts toward Belleisle's central tendency. Adding the track-vs-global-median offset double-counts the track effect.
It erases setup gradient. The correction is constant per (track, channel) regardless of setup inputs, so when DE evaluates many candidates at the same track, the only thing changing is the surrogate's setup-dependent piece (which on this corpus is small) — but the inflated additive constant dominates the signal. The objective gets flatter; sensitivity falls to zero. This is the proximate cause of +1 click −0.000 score.
The own comment even confesses the right formulation:


diff_fitter.txt
Lines 55-57
+    # More sophisticated version would re-predict every row with the
+    # final fitters and take mean(actual - pred).
That's the actual residual. Or better: hold-out-fold residuals so the fitter isn't being corrected against its own training fit.

The slot-shift repair tells us tests didn't catch a real type-corruption bug
In model.py:


diff_model.txt
Lines 104-128
+def _repair_legacy_slot_shift(slot_values: dict[str, object]) -> None:
+    """Fix pickles saved before ``per_track_residuals`` was append-only.
+    Pre-v7 pickles serialised a positional slot list whose tail was one
+    entry short: ``axle_grip_ceilings`` landed in ``per_track_residuals``,
+    ``aero_residual_correction`` landed in ``axle_grip_ceilings``, ...
This means production was running with AxleGripCeiling instances stored in the per_track_residuals dict and AeroResidualCorrection instances stored in the axle_grip_ceilings slot. Hybrid scoring getattr(model, "axle_grip_ceilings", None) or {} would have returned an AeroResidualCorrection, hit .get() and crashed — except _corner_phase_objective_value was hardened with isinstance(..., dict) (see diff_score.txt:20-22) which now silently treats wrong-typed ceilings as empty. So guardrails went dark instead of crashing. No regression test caught the type mismatch, which is why this only got noticed when somebody looked at a stale pickle. A PhysicsModel.__post_init__ type-check would prevent recurrence; the _repair_legacy_slot_shift is correct rescue but doesn't prevent the next slot-order accident.

Static RH is being treated as a surrogate target when it is deterministic geometry
The chain in physics/recommend.py:430-540 plus static_rh_knn.py runs FOUR sequential repair passes:

DE search with TB turns held at baseline, soft penalty on out-of-envelope predicted static RH
enforce_static_rh_feasible lerps perch/pushrod/heave toward a corpus k-NN session
cooptimize_tb_for_static_rh grid-searches TB turns 1-D against predict_setup_readouts
snap_to_garage_step snaps to legal clicks
This whole apparatus exists because predict_setup_readouts cannot accurately predict static garage RH from perch/pushrod/heave/TB. But static RH is a deterministic kinematic function of those inputs given the car's installation ratios. iRacing computes it the same way every time from the same inputs. Fitting it with a Ridge/Forest surrogate is the wrong tool — the relationship has no environmental dependence, no driver-input noise, and one true installation-ratio matrix per car.

What you should be doing: regress static RH on the platform inputs alone, with all other features dropped, on a per-car basis. The regression should be near-perfect (R² > 0.99) because the relationship is mechanical. If it isn't, the corpus has noise in the recorded YAML (unlikely) or the ontology paths are wrong (likely — pushrod_length_offset_*_mm per car needs validation against a swept session). The Belleisle output's 6 mm pushrod move → 0.1 mm static RH proves this regression is broken on Acura today.

The physics evaluator is calibrated to the wrong target
Per CLAUDE.md:

Spearman(physics_score, lap_time) within-group sits at 0.12–0.25 (PLAN target 0.35).
Per-car weights were hand-set after Day 12b: BMW (0.2, 0.8, 0.0), Ferrari (0.0, 0.0, 1.0), Acura default fallback (0.5, 0.3, 0.2) because Acura has only ~36 corner-phase rows.
Hybrid is wired into DE by default since 2026-05-23 (hybrid_score()).
So on the car you're optimizing today (Acura at Belleisle):

Axle ceilings = None (corpus too thin) → hybrid guardrail terms collapse to surrogate-only
Evaluator weights = generic default (not Acura-calibrated)
Per-track residual correction is the broken one above
13 production sessions for the whole car
The user's stated requirement — "physics-based, fully accurate, completely built, correlated, and calibrated" — is unmet on Acura by every observable metric.

What "physics-based, optimal, calibrated" actually requires
I'll order this by leverage: highest-impact, hardest-to-skip first.

P0 — Stop emitting moves the model cannot justify
P0.1 — Remove or correctly implement per_track_residuals (1 day)

Either delete the addition in _predict_v4 and the construction in fit_per_car, OR replace with proper held-out-fold residuals:

# For each fold (one track held out per fold):
#   refit the per-car surrogate on the remaining tracks
#   predict the held-out track's rows
#   residual_track[ch] = mean(actual − holdout_pred)
Until this is fixed, run with --surrogate-only is mathematically saner than hybrid because hybrid inherits the bug through predict(). Add a test that pins setup variance to zero and asserts predictions vary only by setup, not by track-identity-bias.

P0.2 — Suppress moves with sub-noise sensitivity (0.5 day)

If |score(+step) − score(0)| < 0.005 AND |score(−step) − score(0)| < 0.005, the optimizer cannot tell apart ±1 click. Drop the move from the briefing entirely, list it under NOTES as pinned at past (model cannot resolve ±1 click on this corpus). The user will trust the remaining recommendations more.

P0.3 — Replace static-RH surrogate with deterministic kinematic fit (1 day)

Per car, fit static_rh_mm = A · platform_vec + b (linear LS) against per-session YAML readouts. Constrain features to ONLY the kinematic inputs (perch, pushrod, heave, third spring, TB turns + OD, fuel, camber). Refuse to ship the fitter unless R² > 0.98. Replace predict_setup_readouts(...)[static_rh channel] and physics_static_rh_readouts with this. Delete the k-NN + bisection + grid-search-TB stack — none of it should be necessary.

P0.4 — Detect and skip phantom corner 0 (0.5 day)

The five "T0 divergence" lines in the briefing point at a schedule artifact. Either the track-model includes the start/finish straight as a corner, or the corner detector emits a synthetic head-of-lap entry. Strip phase guardrails for any corner with entry_speed_kmh > N or radius_m > X, or whose corner_archetype features fail a "is a corner" sanity check.

P1 — Tighten the gate so it actually measures physics
P1.1 — Replace lax holdout pass criteria (1 day)

Today's median-coverage≥0.50 and median-normed-residual≤2.0 lets through models that are worse than predicting the mean on grip-balance channels. Replace with per-channel thresholds:

Channel	mean_abs target	normed_residual target
accel_lat_g_max
< 0.3 g
< 0.5
accel_lon_g_min/max
< 0.3 g
< 0.5
understeer_angle_mean_rad
< 0.10 rad
< 0.5
lf/rf/lr/rr_ride_height_mean_mm
< 3 mm
< 0.5
setup_static_*_ride_height_mm
< 1 mm
< 0.2 (deterministic — P0.3)
Make the gate hard in CI. Any per-car holdout result failing these targets refuses to release the model — fall back to surrogate-only OR refuse to recommend, don't silently ship.

P1.2 — Lap-time Spearman gate per (car, track) (2 days)

For each (car, track) with ≥10 sessions, leave-one-session-out: compute score(observed_setup) vs observed median_lap_time_s, take Spearman. Fail any (car, track) where ρ < 0.30. This is the actual physics-vs-lap-time correlation; if the score doesn't rank setups by lap time, no amount of evaluator-weight tuning will rescue downstream recommendations.

P1.3 — Held-out hybrid-vs-surrogate-only A/B in CI (already in test_holdout_a_b_regression.py, not gating)

Make it gate. If hybrid loses on >=1 car's held-out IBT, refuse to release the hybrid weights.

P2 — Fix the modeling regime to escape "noisy"
P2.1 — Mask non-clean rows before fit (1-2 days)

TrackModel identifies curbs, bumps, off-track. corner/states.py aggregates everything that's labeled corner-phase. Add a clean_row flag from track-model evidence and drop the dirty rows from the joint surrogate's training set. Per VISION §4 this is the whole point of the track model. Today it informs nothing about fit.

P2.2 — Per-track random intercepts (mixed-effects) (2-3 days)

Closed-form Bayes random-effects (you already have the math in bayes_retrofit.py for parameters) extended to channels would give you proper per-track output offsets fit by maximum-likelihood, not median-of-medians. Replace per_track_residuals with BayesPosterior per (track, channel) and add as a feature, not a post-hoc add-on to predictions.

P2.3 — Inverse-track-sample-count weighting (0.5 day)

If full mixed-effects is too much: weight training rows by 1/sqrt(n_track) so Sebring's 37 sessions don't drown Spa's 11. This is a cheap proxy for partial pooling.

P2.4 — Bake corner-archetype into the fit, not just predict (1 day)

corner_archetype (radius, entry speed, duration, banking) is computed for prediction but is not in the feature matrix at fit time. Adding it lets the surrogate learn wing × radius interactions instead of pooling all corner shapes for the same parameter.

P3 — Honest user-facing accuracy receipts
P3.1 — Replace Confidence: sparse (median n=N) with channel-level error budget (1 day)

The header should read:

Predicted accuracy on this car/track (held-out): 
  peak lateral G  ±0.42 g (noisy)     <- mean_abs from gate
  understeer angle ±0.18 rad (noisy)
  static front RH ±0.4 mm (dense)     <- deterministic P0.3
Lap-time correlation: ρ=0.22 (below 0.35 target)
That puts the user in a position to know when to trust the model. Today the user has no signal.

P3.2 — Per-recommendation sensitivity gate (already half-built — see P0.2)

If you cannot resolve a click, don't move the click. The current behavior — moving 22 of 43 params with zero sensitivity — is the optimizer making cosmetic noise moves.

P4 — Long-tail
Refit damper curves per car when corpus grows past N=100 per-axle samples. Acura specifically.
Tighten installation-ratio sweeps: drive a single session per car that varies ONLY perch by ±10 mm across stints to give the static-RH regression real signal. This is what optimize calibrate is for — make it the user-flow.
The Watch most picker (_FAMILY_PREFERRED_PHASES in full_setup_card.py) is still failing on Belleisle. With 24 corners, T18 dominates. The fix is to score each (family, phase) pair by |impact|/phase_count rather than absolute impact — normalize for the number of corners in that phase.
Acura's corpus (13 prod sessions) is below "trust this model" threshold for any car. Add a hard refusal: if n_prod_sessions < 20, recommend optimize calibrate only, no race briefing.
What I would do this week (priority order)
Delete per_track_residuals add-on in _predict_v4 (or feature-flag it OFF by default). Re-run the latest Acura/Belleisle recommendation and confirm sensitivity values come back nonzero. This single change unblocks every other diagnostic — until it's done, you cannot tell what the actual surrogate signal looks like.
Replace static-RH surrogate with a deterministic linear fit per car. Validate on the Acura corpus: 6 mm of pushrod should move static RH by roughly 6 mm × installation_ratio. If R² < 0.95, fix the ontology paths before fixing anything else.
Tighten the holdout gate and run it in CI. Hard-fail per-channel normed_residual > 0.6 on accel_lat_g_*, understeer_angle_*, dynamic ride heights. The "gate passing" today is providing false safety.
Add a sensitivity floor to recommend — don't ship +1 click ±0.000 moves.
Mask off-line/curb rows from the fit using existing TrackModel evidence.
Strip phantom corner 0 from guardrail evaluation.
Items 1, 2, and 4 alone will visibly change the next Belleisle recommendation — fewer moves, real sensitivity numbers, accurate static RH [predicted] values. Items 3 and 5 will quietly raise the floor under future runs. Item 6 is cosmetic but it's been polluting NOTES for a while.

