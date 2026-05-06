# Audit -- Slice C: Aero maps (2026-05-06)

## Summary
- Grade: PASS
- Small, well-scoped slice: per-car JSON loader with strict schema validation, per-wing 2D `RegularGridInterpolator` + linear blend on the wing axis, an empirically-justified "no rho correction" decision on `ld_ratio`, and the most material gap is dead state rather than a functional bug.

## Implementation quality
- `src/racingoptimizer/aero/__init__.py:33` -- `_default_aero_dir()` resolves `aero-maps/` via `Path(__file__).resolve().parents[3]`. Hard-coded `[3]` hop count is fragile to repo layout moves; only indirectly exercised by `tests/aero/test_smoke.py:93::test_default_aero_dir_resolves_to_repo_root`.
- `src/racingoptimizer/aero/loader.py:16-17` -- `EXPECTED_FRONT_RH_LEN = 51` / `EXPECTED_REAR_RH_LEN = 46` hardcoded; spec and CLAUDE.md flag wing granularity as per-car but RH-axis is silently strict across all 5 cars.
- `src/racingoptimizer/aero/loader.py:106` -- `np.array_equal` on float axes is exact; corpus axes are integer-mm so the strict check works in practice.
- `src/racingoptimizer/aero/loader.py:122` -- NaN guard but no `np.isfinite` (would also catch +/- inf).
- `src/racingoptimizer/aero/interpolator.py:35` -- `BASELINE_AIR_DENSITY = 1.225` is ISA sea-level; spec §6 calls for per-car corpus-mean overrides.
- `src/racingoptimizer/aero/interpolator.py:67-101` -- Constructor accepts `baseline_air_density=` and exposes a `baseline_air_density` property, but `interpolate()` body never reads it. **Dead state.**
- `src/racingoptimizer/aero/interpolator.py:73-93` -- One `RegularGridInterpolator` per (wing, channel); 18 instances for acura. `fill_value=None` enables scipy extrapolation but the upstream `_clamp` ensures it's never invoked OOB.
- `src/racingoptimizer/aero/interpolator.py:142-173` -- Per-axis clamp warnings demoted to DEBUG with a thorough why-comment about DE-search log volume; correct call.
- `src/racingoptimizer/aero/interpolator.py:178-188` -- Wing-axis bracket via `searchsorted(side='right') - 1`, clipped to `[0, n-2]`, with `if w_hi == w_lo` t-guard. Clean.
- `src/racingoptimizer/aero/interpolator.py:191-195` -- Final ld is the dimensionless ratio; no rho factor; module/inline docstrings document the S2.9 audit and point downstream consumers to `physics.score.grip`.

## Wiring
- `src/racingoptimizer/aero/__init__.py:39` -- sole public entry `load_aero_maps(car, *, aero_dir=None) -> AeroSurface`.
- `src/racingoptimizer/physics/score.py:15` imports `BASELINE_AIR_DENSITY, AeroSurface`.
- `src/racingoptimizer/physics/score.py:559` -- `_AERO_CACHE: dict[str, AeroSurface | None]`.
- `src/racingoptimizer/physics/score.py:562` -- `_aero_surface_or_none(model)`: short-circuits on `model.aero_correction_available`, lazy-loads via `load_aero_maps(model.car)`, memoises by car (incl. None on failure).
- `src/racingoptimizer/physics/score.py:538` -- `_aero_ld_for_state` composes `(front_rh_mean, rear_rh_mean, mid-wing, env.air_density)` into `aero.interpolate(...)`. Mid-wing fallback uses `aero.bounds.wing_angles[len(...) // 2]`.
- `src/racingoptimizer/physics/score.py:98` -- only consumer of `BASELINE_AIR_DENSITY` for an actual rho correction: `density_factor = float(env.air_density) / BASELINE_AIR_DENSITY` then `max_g = 0.5 * ld * density_factor + baseline`. This honours the CLAUDE.md commitment.
- `src/racingoptimizer/physics/fitter.py:1086-1098` -- `_try_load_aero(car)` detects whether slice C is reachable; result flows into `PhysicsModel.aero_correction_available` (`fitter.py:389,839`) and onto the JSON payload (`explain/render_json.py:52`).
- `src/racingoptimizer/explain/full_setup_card.py:64` -- AeroCalculator block (`FrontRhAtSpeed`, `RearRhAtSpeed`, `DownforceBalance`, `LD`) renders as `[readout]`. Slice C is NOT on the rendering path for this block -- it's iRacing's own static calculator from the past-session setup blob.

## Gaps
1. **MEDIUM | `src/racingoptimizer/aero/interpolator.py:67,70,100-101`** -- `baseline_air_density` ctor argument is stored but `interpolate()` never reads it. The density correction lives entirely at `physics/score.py:98` and hard-codes the module-level `BASELINE_AIR_DENSITY`. Dead state misleadingly suggests per-car configurability that doesn't exist. Either delete or actually plumb through.
2. **MEDIUM | `src/racingoptimizer/physics/score.py:547-548`** -- `bounds.front_rh_mm[0] + bounds.front_rh_mm[1]) / 2.0` indexes a `(lo, hi)` tuple. The arithmetic happens to compute the bounds midpoint correctly, but `[0]/[1]` on a tuple-typed bound reads like an axis-array index and is a re-read footgun. Refactor to a `_bounds_midpoint(b)` helper.
3. **LOW | `src/racingoptimizer/aero/loader.py:122`** -- NaN guard but no +/- inf check; tighten to `np.isfinite` for symmetry.
4. **LOW | `src/racingoptimizer/aero/loader.py:16-17`** -- Hardcoded RH-axis lengths (51, 46) shared across all 5 cars today; promote to per-car overrides ahead of any aero re-extraction.
5. **LOW | `src/racingoptimizer/aero/__init__.py:36`** -- `parents[3]` is fragile; add an `assert is_dir` or env-var override (`RACINGOPTIMIZER_AERO_DIR`).
6. **INFO | `src/racingoptimizer/aero/interpolator.py:35`** -- Per-car baseline air-density still a follow-up per spec §6. BMW Sebring corpus mean rho is closer to 1.18 than 1.225 ISA; absolute-grip term scales against ISA reference rather than corpus central tendency.
7. **INFO | `src/racingoptimizer/physics/score.py:552`** -- Mid-wing fallback hardcodes `wing_angles[len // 2]` rather than the optimizer's recommended wing. Slice-C is innocent; flagged for the call-site.

## Evidence
- Test suite: NOT EXECUTED (sandbox denied `uv run pytest tests/aero -m "not slow"`). Static review of `tests/aero/`:
  - `test_loader.py` -- 12 tests (filename parsing happy + 5 invalid forms; per-car real-corpus shape × 5 cars; unknown-car/missing-dir; missing-required-key; balance-shape mismatch; axis disagreement across wings; NaN cell; sort order; payload/filename wing match).
  - `test_interpolator.py` -- 12 tests (bounds for porsche + acura; exact node lookup; midway-on-wing; bilinear at fixed wing; density baseline; **density invariance per S2.9** double + half BASELINE; zero/negative density raises; clamp behaviour on front_rh + wing with caplog assert; OOB doesn't raise; `car` attribute).
  - `test_smoke.py` -- 4 tests (per-car smoke loop across all 5 cars; full-envelope hand-precomputed trilinear reference @ porsche (42.5, 22.5, 14.5°, 1.225) within 1e-9; OOB negative query; `_default_aero_dir` resolves to repo root).
  - `test_ld_ratio_units.py` -- 3 tests (every cell across 33 files in [1.0, 10.0]; corpus min/max in [2.5, 5.0] documented band; per-car parametrize over 5 cars).
  Coverage matches the per-car verification convention in CLAUDE.md.
- Lint: NOT EXECUTED. Static review found no obvious ruff trip-wires (no unused imports, all `from __future__ import annotations`, no star-imports).
- Latest artefact with AeroCalculator block: `recommendations/ferrari-spa-race-0506-1026.txt`. Block renders four iRacing-calculator readouts (`Front Rh At Speed 15.0 mm`, `Rear Rh At Speed 40.0 mm`, `Downforce Balance 48.98%`, `LD 3.855`), all tagged `[readout]` -- confirms slice C does NOT feed this block.

## Recommended next actions
- Decide whether `AeroSurface(baseline_air_density=...)` is real config or dead state. If real, plumb through to `physics/score.py::grip`'s rho factor; if not, remove the parameter and property. Single most misleading thing in this slice.
- Refactor `physics/score.py:547-548` bounds-midpoint into a `_bounds_midpoint(b)` helper to remove the `[0]+[1]` ambiguity.
- Tighten `loader.py:122` NaN guard to `np.isfinite`.
- Promote `EXPECTED_FRONT_RH_LEN` / `EXPECTED_REAR_RH_LEN` to per-car overrides ahead of any aero re-extraction.
- Track per-car baseline-air-density override (spec §6) as the follow-up that retires gap #6.
