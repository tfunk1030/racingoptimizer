# Aero-Map Loader & Interpolator — Design Spec

**Date:** 2026-04-28
**Slice:** C — Aero-map loader & interpolator
**Module:** `racingoptimizer.aero`

## 1. Context

`racingoptimizer` is a physics-based setup optimizer for iRacing GTP cars (see `VISION.md`). The aero corpus in `aero-maps/` is one of two empirical truth sources (the other is the IBT telemetry parsed by slice A). Each JSON parses a (car, wing-angle) aerodynamic surface as a function of front and rear ride height: aero balance % and lift/drag ratio at every node of a 51 × 46 grid. Five cars × six-or-nine wing angles = 33 files. The data was extracted offline from a higher-fidelity source; we treat it as ground truth and only interpolate, never re-fit.

VISION.md §3 names this slice as the bridge between ride-height changes and downforce changes — every physics-fitter and setup-evaluator query that asks "what would aero do at front_rh=42 mm, rear_rh=20 mm, wing=14°?" has to land here. VISION.md §10 names the air-density correction: aero forces scale with air density, and IBT files carry per-sample `AirDensity` (kg/m³). The loader must accept that environmental context per call, not bake it into the cached surface.

CLAUDE.md "Architectural commitments" pin two constraints onto this slice: **empirical physics, not formulas** (aero maps are the explicit one exception — they're empirical lookups already), and **every data point carries environmental context** (the air-density argument is mandatory on every call, even if the caller passes the standard atmosphere).

The deliverable is a Python module `racingoptimizer.aero` that:

- Reads the 33 aero-map JSONs and constructs a per-car `AeroSurface` cache
- Interpolates `(balance_pct, ld_ratio)` at any `(front_rh_mm, rear_rh_mm, wing_deg)` inside the documented envelope
- Applies an air-density correction to `ld_ratio` per call
- Clamps out-of-envelope queries to the nearest grid edge with a warning, never raises

This slice is fully isolatable: it consumes nothing from slices A, B, D, E, and produces a module that C, D, E, F all consume. It is the smallest well-bounded unit in the master plan.

## 2. Public API

```python
from racingoptimizer.aero import load_aero_maps, AeroSurface, AeroBounds

load_aero_maps(car: str, *, aero_dir: Path | None = None) -> AeroSurface
    # Loads every aero-maps/<car>_wing_*.json under `aero_dir` (default:
    # repo-relative `aero-maps/`). Returns a single AeroSurface cached over
    # all wing angles for that car.

AeroSurface.interpolate(
    front_rh_mm: float,
    rear_rh_mm: float,
    wing_deg: float,
    air_density: float,
) -> tuple[float, float]
    # Returns (balance_pct, ld_ratio_corrected). All four args are required.
    # Out-of-envelope inputs are clamped to the nearest grid edge and a
    # warning is logged via the stdlib logger; the call never raises.

AeroSurface.bounds -> AeroBounds
    # The (front_rh, rear_rh, wing) envelope of the loaded car. Callers can
    # inspect this to clamp their own inputs upstream if they want a hard
    # rejection rather than the loader's clamp-and-warn.

AeroSurface.car -> str
    # The canonical car key ("acura", "bmw", "cadillac", "ferrari", "porsche").

AeroBounds  # immutable dataclass
    front_rh_mm: tuple[float, float]   # (25.0, 75.0) for all current cars
    rear_rh_mm:  tuple[float, float]   # (5.0, 50.0) for all current cars
    wing_deg:    tuple[float, float]   # car-dependent
    wing_angles: tuple[float, ...]     # full sorted list of loaded angles
```

`AeroSurface.interpolate` returns plain Python floats. Vectorized batch interpolation is **out of scope** for this slice — the physics fitter and the optimizer call this point-by-point, and the dominant cost is the wing-axis bracket lookup, not the kernel evaluation. If a future slice needs batched calls we add an `interpolate_batch` method without breaking the scalar API.

## 3. On-disk layout

This slice introduces no new on-disk artefacts. The aero JSONs in `aero-maps/` are the authoritative inputs and are read-only. No caching to disk; the in-memory `AeroSurface` is reconstructed per process.

The aero-maps directory mixes the 33 files flat at the top level (no subfolders today, unlike `ibtfiles/`). The loader walks the directory non-recursively and matches `<car>_wing_<X.X>.json`.

## 4. Schema (verified from corpus, 2026-04-28)

Every JSON shares the same shape. Keys, in the order they appear in each file:

| Key | Type | Shape / range | Notes |
|---|---|---|---|
| `car` | str | one of `acura, bmw, cadillac, ferrari, porsche` | canonical key |
| `wing` | float | car-dependent (see below) | wing angle in degrees |
| `front_rh_mm` | list[int\|float] | length 51, `[25, 75]` step 1 | strictly increasing |
| `rear_rh_mm` | list[int\|float] | length 46, `[5, 50]` step 1 | strictly increasing |
| `balance_pct` | list[list[float]] | `[51][46]` | aero balance %, `[front_rh][rear_rh]` |
| `ld_ratio` | list[list[float]] | `[51][46]` | lift/drag ratio, `[front_rh][rear_rh]` |

Wing-angle granularity differs per car:

| Car | Wing range | Step | Count |
|---|---|---|---|
| acura | 6.0 – 10.0° | 0.5° | 9 |
| bmw | 12.0 – 17.0° | 1.0° | 6 |
| cadillac | 12.0 – 17.0° | 1.0° | 6 |
| ferrari | 12.0 – 17.0° | 1.0° | 6 |
| porsche | 12.0 – 17.0° | 1.0° | 6 |

**Verified absence of NaN cells** across all 33 files (2 × 51 × 46 × 33 = 155 826 scalar samples). The interpolator therefore does not need a NaN-skip path; an explicit assertion at load time catches any future NaNs.

The internal cache representation, after loading every wing for one car:

```python
@dataclass(frozen=True)
class AeroMapData:
    car: str
    wing_angles: tuple[float, ...]            # sorted ascending
    front_rh_mm: np.ndarray                   # shape (51,)
    rear_rh_mm:  np.ndarray                   # shape (46,)
    balance_pct: np.ndarray                   # shape (n_wing, 51, 46)
    ld_ratio:    np.ndarray                   # shape (n_wing, 51, 46)
```

Storing the wing axis as the **leading** dimension (rather than trailing) keeps the per-wing 2D `RegularGridInterpolator` cheap to construct: each wing slice is a contiguous `(51, 46)` view.

## 5. Interpolation kernel

**Inside the envelope.** Three-step procedure:

1. **Bracket the wing axis.** Find indices `i, i+1` such that `wing_angles[i] <= wing_deg <= wing_angles[i+1]`. Compute `t = (wing_deg - wing_angles[i]) / (wing_angles[i+1] - wing_angles[i])`. This is the *only* place the non-uniform wing spacing matters; the lookup uses `np.searchsorted` and the math is local.
2. **2D interpolate at each bracketing wing.** For wing slice `i`, build a `scipy.interpolate.RegularGridInterpolator((front_rh_mm, rear_rh_mm), balance_pct[i, :, :], method='linear', bounds_error=False, fill_value=None)`. Same for `ld_ratio[i]`. Evaluate at `(front_rh_mm, rear_rh_mm)`. Repeat for slice `i+1`. Cache these interpolators per `AeroSurface` instance — the constructor builds one per wing slice and reuses them across calls.
3. **Linear interp across wing.** `value = (1 - t) * value_i + t * value_{i+1}`, applied independently to balance and ld_ratio.

`RegularGridInterpolator` with `method='linear'` performs bilinear interpolation on the rectangular grid; combined with the manual linear blend on the wing axis, the kernel is **trilinear** overall. This is exactly what the master plan §4 slice C card recommends ("scipy `RegularGridInterpolator` per-wing then linear interp across wing — keeps it pure-data, no fitted surface").

**Out-of-envelope.** `bounds_error=False` and `fill_value=None` make `RegularGridInterpolator` return the **clamped** value at the nearest edge for out-of-grid `(front_rh, rear_rh)` inputs. We mirror this for the wing axis manually: if `wing_deg < wing_angles[0]`, snap to `wing_angles[0]`; if `wing_deg > wing_angles[-1]`, snap to `wing_angles[-1]`. Each clamp emits one `logging.warning(...)` per call (the caller can suppress via the standard logging API).

**No exceptions on bad ranges.** A setup that strays out of the aero envelope is a real-world occurrence (e.g. an aggressive ride-height drop bottoms the car; the optimizer should down-rank such setups, not crash). The clamp-and-warn behaviour matches that requirement — slice C card §4.

## 6. Air-density correction

Aerodynamic forces scale linearly with air density. The aero JSONs were extracted at an unstated baseline; until slice A's IBT corpus gives us a corpus-mean reference, we use **ISA sea-level standard atmosphere = 1.225 kg/m³** as the baseline. This is documented as `BASELINE_AIR_DENSITY` and pinned per-car (initially the same for all five cars; per-car overrides land when their reference session is identified).

Formula:

```python
ld_ratio_corrected = ld_ratio_raw * (air_density / baseline_air_density)
```

`balance_pct` is **not corrected for air density** — it is already a ratio (front share of total downforce), and the air-density factor cancels in numerator and denominator.

The baseline is a per-car field on `AeroSurface` (`baseline_air_density: float`) with a module-level default of 1.225. A future slice can replace this with a per-car constant computed from the median `AirDensity` channel across that car's corpus; that change is local to the loader and does not affect the public API.

The `air_density` argument is **required**, not defaulted. Forgetting to supply it would silently bake in an incorrect correction; making it required forces the caller to thread environmental context through their pipeline (CLAUDE.md commitment).

## 7. Failure handling

| Failure mode | Behaviour |
|---|---|
| `aero-maps/` directory missing | Raise `FileNotFoundError` at `load_aero_maps` call time |
| No JSONs match `<car>_wing_*.json` | Raise `AeroLoadError` with "no aero maps for car '<car>'" |
| One JSON parses but lacks a required key | Raise `AeroLoadError` with the offending file path + missing key |
| `front_rh_mm` / `rear_rh_mm` axes disagree across wings of one car | Raise `AeroLoadError` with both filenames |
| `balance_pct` or `ld_ratio` shape inconsistent with axis lengths | Raise `AeroLoadError` |
| NaN cell in `balance_pct` or `ld_ratio` | Raise `AeroLoadError` (verified absent today; defensive check) |
| Out-of-envelope `interpolate(...)` query | Clamp to nearest edge, log warning, return clamped value |
| `air_density <= 0` | Raise `ValueError` (would produce non-physical correction) |

All loader-side failures raise; all per-call failures clamp-and-warn. The split is intentional: a missing JSON is a deployment problem the user can fix once; a query that strays out of the envelope is a hot-path event during physics fitting.

## 8. Module layout

```
src/racingoptimizer/
  __init__.py            # one-line module docstring; shared with slice A
  aero/
    __init__.py          # public API re-exports: load_aero_maps, AeroSurface, AeroBounds
    loader.py            # disk -> AeroMapData; schema validation
    interpolator.py      # AeroMapData -> AeroSurface; bracket + interp + clamp + log
  py.typed
pyproject.toml           # minimal; numpy, scipy + pytest dev (see filename note in §13)
tests/
  aero/
    __init__.py
    test_loader.py       # disk parsing, schema validation, error paths
    test_interpolator.py # interpolation correctness, clamp behaviour, air-density correction
    test_smoke.py        # end-to-end on porsche fixture
```

`loader.py` is **pure data parsing** — it produces the `AeroMapData` immutable container and validates schema. `interpolator.py` is **pure compute** — it consumes `AeroMapData` and exposes the `AeroSurface` query API. The split keeps the loader testable without scipy and the interpolator testable without disk I/O. (The single-file alternative was considered and rejected for that reason: a disk-mocking test of bilinear math is awkward to write.)

`AeroSurface` is the public class; `AeroMapData` is internal and not re-exported.

## 9. Testing

- **Unit (`test_loader.py`):**
  - `_parse_filename("acura_wing_6.5.json")` returns `("acura", 6.5)`; mismatched names raise.
  - Loader produces an `AeroMapData` with the expected shapes for each of the 5 cars.
  - Loader rejects: missing key, shape mismatch, axis disagreement across wings, NaN cell. Each gets a separate parametrized test with a synthetic in-temp-dir JSON.
  - Wing angles in the loaded surface are **sorted ascending** even when the on-disk filename order is not.
- **Unit (`test_interpolator.py`):**
  - Interpolating exactly on a grid node returns the JSON's stored value (within `1e-12`).
  - Interpolating between two adjacent wings at the rh-grid centre matches a hand-computed `0.5 * (v_i + v_{i+1})` (within `1e-12`).
  - Interpolating between two adjacent rh nodes at a fixed wing matches the bilinear hand-computation (within `1e-12`).
  - **Air-density correction:** `interpolate(... , air_density=BASELINE)` returns raw `ld_ratio`; `interpolate(... , air_density=2*BASELINE)` returns `2 * ld_ratio`; `balance_pct` is unchanged across both.
  - **Clamp behaviour:** out-of-envelope `front_rh=200 mm` returns the value at `front_rh=75 mm`; one warning is logged. Same for `rear_rh`, `wing_deg`. The call **does not raise**.
  - `air_density <= 0` raises `ValueError`.
- **Smoke (`test_smoke.py`):**
  - Load every car's aero maps in turn (5 cars × full wing fan); assert `bounds.front_rh_mm == (25.0, 75.0)`, `bounds.rear_rh_mm == (5.0, 50.0)`, and `bounds.wing_angles` matches the §4 table.
  - Porsche end-to-end: `interpolate(42.5, 22.5, 14.5, 1.225)` returns finite numbers, balance_pct in `[0, 100]`, ld_ratio > 0, and the call matches a hand-precomputed reference (built from the four bracketing JSON corners) within `1e-9`.
  - Acura's 0.5° wing step: interpolating at `wing=6.25°` (between 6.0 and 6.5) on otherwise-on-grid rh produces the average of those two slices' values.

The smoke test against the real `aero-maps/` JSONs is the e2e gate the master plan §4 slice C card mandates.

## 10. Out of scope

- **Vectorized batch interpolation.** Scalar API only this slice. Add `interpolate_batch` if a profiled hot path needs it.
- **Per-car baseline air densities derived from the IBT corpus.** Today: ISA standard 1.225 kg/m³ for all cars. The hook is in place (`baseline_air_density` is a per-`AeroSurface` attribute); slice E or a follow-up can wire in measured baselines.
- **Re-fitting the aero surfaces.** The JSONs are empirical; we look up, we don't fit.
- **Caching across processes.** Loading 33 small JSONs takes well under a second; no on-disk cache yet.
- **CLI integration.** Slice F wires the optimizer top-level command; this slice exposes only library API.
- **Wind / yaw asymmetry.** VISION.md §10 names this; the current aero JSONs do not parametrize on wind direction or yaw, so this slice cannot model it. Punt to a future "wind correction" slice.
- **Tyre temperature effects on aero balance.** Same — out of corpus scope.

## 11. Open questions / future work

- **Per-car baseline air density.** Once slice A lands and the IBT corpus' median `AirDensity` per car is computable, replace the 1.225 default with a per-car constant. One-line change in the loader.
- **Setup ontology integration.** When slice E's setup ontology lands, the aero loader could expose a convenience method `interpolate_for_setup(setup, environment)` that pulls `wing` from the setup and `front_rh_mm`/`rear_rh_mm` from the predicted ride heights. Postponed until slice E's typed setup exists.
- **Wing-angle interpolation accuracy.** Linear interp on the wing axis may be coarser than physics warrants in the 1°-step regime; if slice E's residuals show wing-related noise, revisit with cubic splines on wing.
- **Confidence reporting.** Slice E owns the `Confidence` type. When a future caller wants to know "how trustworthy is this aero query?" the answer is essentially "exact at grid nodes, trilinear-interpolated between." That can be wrapped in a `Confidence` later without breaking the scalar API.

## 12. Cross-slice handshakes

This slice introduces no persisted state and no new schema. It is consumed by:

- **Slice E (Physics fitter):** Calls `AeroSurface.interpolate` once per `(corner-phase, sample)` to convert the predicted ride heights at that phase into balance / drag terms entering the residual.
- **Slice F (CLI):** Will call `load_aero_maps(car)` once at startup for the active car and pass the surface into the optimizer's evaluation loop.
- **Slice D (Track model):** Indirectly — track-model output (clean-section masks) doesn't touch this slice; aero queries happen on already-masked data downstream.

No reservation in slice A's parquet is needed. No catalog changes.

## 13. Greenfield notes

Other slices' workers are running in parallel. This slice's worktree forks from master HEAD, so:

- `pyproject.toml` does not yet exist in this worktree. We create a minimal one containing only this slice's runtime deps (`numpy`, `scipy`) and dev deps (`pytest`). Slice A's worker will create their own version with `pyirsdk, polars, pyarrow, click, pytest, pytest-cov, ruff`. The user will reconcile both PRs by hand at merge time.
- `src/racingoptimizer/__init__.py` is shared with slice A. To keep the two PRs trivially mergeable, we write exactly the one-line module docstring `"""racingoptimizer — physics-based setup optimizer for iRacing GTP cars."""` and nothing else.
- `src/racingoptimizer/ingest/` is slice A's territory and is not touched by this slice.
- The `aero-maps/` directory is **read-only** (CLAUDE.md hook blocks destructive ops). Tests use the real files in place; no fixture copies are made.
