# Corner-Phase Decomposition — Design Spec

**Date:** 2026-04-28
**Slice:** B — Corner-phase decomposition (second of the VISION subsystems)
**Modules:** `racingoptimizer.corner`, `racingoptimizer.context`

## 1. Context

`racingoptimizer` scores setups at the **corner-phase grain**, not the lap (CLAUDE.md and `VISION.md` §2). Slice A (`racingoptimizer.ingest`, spec at `docs/superpowers/specs/2026-04-28-ibt-ingestion-design.md`) lands raw 60 Hz telemetry per session into parquet plus a SQLite catalog. Nothing in slice A knows what a corner is. Slice B is the layer that turns a flat lap into a labelled sequence of corners and corner-phases so that the physics fitter (slice E) and the optimizer (slice F) can reason about each phase independently.

This is its own slice, separate from A, for three reasons:

1. **Different change cadence.** Channel selection and parquet layout (slice A) settle once. Corner detection rules (this slice) are heuristic and will be refined as the track model (slice D) matures.
2. **Different test surface.** Slice A is verified by smoke-ingesting a fixture and reading channels back. Slice B requires a synthetic-signal harness plus per-track corner-count sanity checks.
3. **Born-in-B cross-cutting work.** Slice B is also where `racingoptimizer.context.EnvironmentFrame` is born (master plan §2). Putting that with ingestion would have leaked phase-aware aggregation into slice A; putting it with the fitter would have made slices C and D cite contracts that did not exist yet.

The deliverable of this spec is **two** Python modules:

- `racingoptimizer.corner` — `Phase` enum, `CornerPhaseKey` NamedTuple, `segment_lap()`, `corner_phase_states()`.
- `racingoptimizer.context` — `EnvironmentFrame` dataclass with the minimum contract pinned in master plan §2.

This is one slice. Track-position-based segmentation (slice D), aero-density correction (slice C), confidence reporting (slice E), and recommendation rendering (slice F) are explicit non-goals here and get their own specs later.

## 2. Public API

```python
from racingoptimizer.corner import (
    Phase,
    CornerPhaseKey,
    segment_lap,
    corner_phase_states,
)
from racingoptimizer.context import EnvironmentFrame


class Phase(StrEnum):
    BRAKING = "braking"
    TRAIL_BRAKE = "trail_brake"
    MID_CORNER = "mid_corner"
    EXIT = "exit"
    STRAIGHT = "straight"
    # The four in-corner phases plus the connecting STRAIGHT. Ordering is the
    # natural traversal direction through one corner: STRAIGHT precedes BRAKING,
    # EXIT yields back to STRAIGHT.


class CornerPhaseKey(NamedTuple):
    session_id: str   # 16-hex sha256 prefix from slice A
    lap_index: int    # slice A's lap_index; never -1 here
    corner_id: int    # 0-based, ascending in track order; -1 reserved for "not in a corner"
    phase: Phase
    # The atomic unit (CLAUDE.md). Hashable, comparable, JSON-stable. Every
    # downstream training point, prediction, and recommendation cites one of
    # these.


def segment_lap(lap_df: pl.DataFrame) -> pl.DataFrame:
    """Label every sample in a single lap with its corner_id and phase.

    Input: the wide Polars frame returned by slice A's `lap_data(session_id,
    lap_index)`. Required columns: `t_s`, `lap_dist_pct`, `Speed`, `AccelLat`,
    `Brake`, `Throttle`, `Steering`, `YawRate`. Returns a copy with two columns
    appended: `corner_id: Int32` (-1 = outside any corner) and `phase: Utf8`
    matching `Phase` values. Pure function: no I/O, no global state, no side
    effects. Calling twice on the same input yields identical output. Samples
    that the algorithm cannot classify (e.g. mid-lap data gap) get
    corner_id=-1, phase=STRAIGHT — never raised."""


def corner_phase_states(
    session_id: str,
    lap_index: int,
    *,
    corpus_root: Path | None = None,
) -> pl.DataFrame:
    """Aggregate one row per (corner_id, phase) for one lap of one session.

    Loads the lap via slice A's `lap_data`, runs `segment_lap`, then groups by
    (corner_id, phase) and computes the derived physics state defined in §6
    plus the per-phase mean of every `EnvironmentFrame` field. Returns a
    Polars frame keyed by (corner_id, phase) with one row per group. Skips
    corner_id=-1 (samples outside any corner) by design — STRAIGHT samples
    inside a labelled corner stay; STRAIGHT samples between corners do not
    appear in the output. `corpus_root` overrides the default
    `RACINGOPTIMIZER_CORPUS` lookup; threaded through to slice A's loaders."""
```

`EnvironmentFrame` lives in `racingoptimizer.context`:

```python
@dataclass(frozen=True, slots=True)
class EnvironmentFrame:
    """Per-sample atmospheric + track-surface context.

    Master plan §2 minimum contract. Frozen so it is hashable and safe to
    cache. `from_row()` builds one from a single Polars row (raw IBT channel
    names) — the canonical adapter slice B owns. Built from raw IBT channels
    stored in slice A's parquet; never re-fetched from the IBT itself.

    Per-phase means are computed inside `corner_phase_states` (one
    EnvironmentFrame-shaped block of columns per row). Raw per-sample
    EnvironmentFrame data stays in slice A's parquet — slice B never
    persists per-sample environment.
    """
    air_density: float       # kg/m^3   from `AirDensity`
    track_temp_c: float      # °C       from `TrackTempCrew`
    wind_vel_ms: float       # m/s      from `WindVel`
    wind_dir_deg: float      # degrees  from `WindDir`
    track_wetness: float     # unitless from `TrackWetness`

    @classmethod
    def from_row(cls, row: dict[str, float]) -> "EnvironmentFrame": ...
```

Polars (not pandas) for return types — same rationale as slice A.

## 3. Module layout

```
src/racingoptimizer/
  corner/
    __init__.py          # re-exports: Phase, CornerPhaseKey, segment_lap, corner_phase_states
    phase.py             # Phase enum, CornerPhaseKey NamedTuple
    detect.py            # corner detection (lateral-G threshold + hysteresis)
    boundaries.py        # phase-boundary rule engine; threshold table; per-car overrides
    states.py            # corner_phase_states aggregation (derived physics state)
    config.py            # PhaseThresholds dataclass + DEFAULT_THRESHOLDS + PER_CAR overrides
  context/
    __init__.py          # re-exports: EnvironmentFrame
    environment.py       # EnvironmentFrame dataclass + from_row adapter

tests/
  corner/
    test_phase_enum.py
    test_detect_synthetic.py     # constructed lateral-G profiles → expected corner counts
    test_boundaries.py           # synthetic per-phase signals → expected phase labels
    test_states.py               # derived physics state arithmetic on a known fixture frame
    test_sebring_fixture.py      # BMW Sebring fixture: ≥ 12 corners, all phases populated
    test_per_car_smoke.py        # one canonical fixture per car: corner count > 0
  context/
    test_environment.py          # from_row mapping; frozen invariants
```

No `racingoptimizer.utils` (master plan §2 forbids it). No setup-ontology touch (deferred to slice E). No `racingoptimizer.quality` work — slice A reserves the column, slice D fills it; slice B reads it (`data_quality_mask`) when present and falls back to "all-clean" if absent.

## 4. Corner detection algorithm

**Pinned choice: lateral-G threshold with hysteresis.**

Concretely: a sample is "in a corner" iff `|AccelLat|` exceeds an entry threshold; it remains in-corner until `|AccelLat|` drops below an exit threshold for at least `T_EXIT_HOLD_MS` continuous milliseconds. This is a Schmitt-trigger style detector with two thresholds preventing flutter at the boundary.

Pinned defaults (universal, override per car if needed — see §5):

```python
G = 9.80665                          # m/s^2
LAT_G_ENTRY  = 0.50 * G              # |AccelLat| above this opens a corner
LAT_G_EXIT   = 0.30 * G              # below this closes it
T_EXIT_HOLD_MS = 200                 # ms below LAT_G_EXIT before close
MIN_CORNER_DURATION_MS = 400         # discard candidate corners shorter than this
```

`AccelLat` is signed (left-positive in iRacing); the detector uses `abs()`. `corner_id` increments monotonically per lap starting at 0 in track order. Samples that never met the entry threshold get `corner_id = -1`.

### Alternatives considered

- **Speed-trough.** Detect each local minimum of `Speed`. Simple and corner-count-robust on circuits where every corner has a clear speed dip. Fails on near-flat-throttle corners (Sebring T17 long right-hander before the front straight; Eau Rouge analogues) and on banked oval-style segments where speed is monotone-rising through the corner. Rejected.
- **Yaw-rate-onset.** Detect rising `YawRate` magnitude. Catches turn-in well but is noisy under steering correction and conflates curb-induced yaw spikes (which are exactly what slice D's track model is supposed to mask out — and slice D is not yet available to B). Rejected as primary; useful as a future cross-check.
- **Lateral-G threshold (pinned).** Sustained lateral G is the most direct telemetry-side proxy for "the car is loaded laterally"; it survives flat-throttle corners (the load is still there) and is robust to brief yaw transients. Hysteresis kills flutter. Trades off some boundary precision (the exact turn-in moment is fuzzy at 0.5 G) but slice B is not the place where that precision matters — phase boundaries inside a detected corner do that work.

### Forward compatibility with slice D

When slice D (`racingoptimizer.track`) lands, it will provide track-position-anchored corner boundaries that are sharper and consistent across laps. Slice B's `segment_lap` accepts an optional `track_model: TrackModel | None = None` keyword (reserved now, accepted in the signature, ignored when `None`). When non-`None`, the lateral-G detector is replaced by a lookup of corner intervals from `track_model.corner_intervals(track)`. The reserved keyword is part of B's API today so slice D can land without breaking B's callers; the implementation shipped in slice B raises `NotImplementedError` if the keyword is non-`None`.

## 5. Phase-boundary signals

Inside a detected corner the four in-corner phases (`BRAKING → TRAIL_BRAKE → MID_CORNER → EXIT`) are walked in order using the rules below. Outside a detected corner the phase is `STRAIGHT`. The walker is a one-pass state machine; transitions are forward-only within a corner (no `EXIT → MID_CORNER`).

| Transition | Signal (universal default) |
|---|---|
| `STRAIGHT → BRAKING` | `Brake > 0.05` (5% pedal) AND inside a detected corner |
| `BRAKING → TRAIL_BRAKE` | `Brake > 0` AND `\|Steering\| > 0.05 rad` AND `\|AccelLat\| > 0.6 * LAT_G_ENTRY` |
| `TRAIL_BRAKE → MID_CORNER` | `Brake < 0.02` (off-pedal) for ≥ 50 ms |
| `MID_CORNER → EXIT` | `Throttle > 0.10` AND `\|AccelLat\|` strictly decreasing over a 100 ms window |
| `EXIT → STRAIGHT` | `\|Steering\| < 0.05 rad` AND `Throttle > 0.50` AND outside the detected corner |

**Brake values are normalized 0..1** (slice A keeps the IBT raw `Brake` channel which iRacing emits as 0..1). Steering is in radians (`SteeringWheelAngle`). Throttle is 0..1 (`Throttle`). `AccelLat` is m/s² and the rules quote multiples of `LAT_G_ENTRY` so they scale with the §4 constant.

### Universal vs per-car

**Default: universal.** The thresholds above apply to all five GTP cars unless overridden. This is the right default because the cars are close in performance envelope and their pedal/steering normalisation is identical (iRacing normalises by car-config max). A single set of numbers also makes the detector behaviour independent of car-detection success — a session that fails car detection in slice A still gets phase-decomposed.

**Override mechanism: a config dict, keyed by canonical car key.** Pinned in `corner/config.py`:

```python
@dataclass(frozen=True)
class PhaseThresholds:
    brake_on: float = 0.05
    brake_off: float = 0.02
    steering_in_corner_rad: float = 0.05
    steering_straight_rad: float = 0.05
    throttle_exit: float = 0.10
    throttle_straight: float = 0.50
    lat_g_entry_frac: float = 1.0    # multiplier on §4 LAT_G_ENTRY
    accel_lat_corner_frac: float = 0.6
    exit_lat_g_window_ms: int = 100
    brake_off_hold_ms: int = 50

DEFAULT_THRESHOLDS: PhaseThresholds = PhaseThresholds()

PER_CAR: dict[str, PhaseThresholds] = {
    # empty by default; entries added only with telemetry evidence in the
    # commit message that justifies divergence from the default.
}

def thresholds_for(car: str) -> PhaseThresholds:
    return PER_CAR.get(car, DEFAULT_THRESHOLDS)
```

Rationale for "config dict, not registered function and not per-car class": the `add-constraint` and `setup-justifier` automations already treat per-car parameters as data, not code. Mirroring that here keeps the override surface inspectable without grepping for class hierarchies. A config dict also serialises trivially when the eventual recommendation output (slice F) explains why a given threshold was used.

## 6. Per-corner-phase derived state

`corner_phase_states` returns one row per `(corner_id, phase)` with the columns below. All values are scalars per row. Sources are raw IBT channel names persisted by slice A; aggregation rule (`mean`, `max`, `p99`, `range`) is fixed per column.

| Column | Aggregation | Source / Definition |
|---|---|---|
| `session_id, lap_index, corner_id, phase` | key | from inputs |
| `t_start_s, t_end_s, duration_s` | first / last / diff | `t_s` |
| `lap_dist_pct_in, lap_dist_pct_out` | first / last | `lap_dist_pct` |
| `speed_min_ms, speed_max_ms, speed_mean_ms` | min / max / mean | `Speed` |
| `accel_lat_max` | max(abs) | `AccelLat` |
| `accel_lon_min, accel_lon_max` | min / max | `AccelLon` |
| `brake_max, brake_mean` | max / mean | `Brake` |
| `throttle_max, throttle_mean` | max / mean | `Throttle` |
| `steering_max_rad` | max(abs) | `SteeringWheelAngle` |
| `understeer_angle_mean_rad` | mean | `SteeringWheelAngle - (steering_geom * AccelLat / max(Speed^2, eps))` — coefficient `steering_geom` is per-car and lives in `corner/config.py` next to thresholds; for slice B it is a placeholder constant 1.0, refined by slice E |
| `load_transfer_asymmetry_mean` | mean | `(LFshockDefl + RRshockDefl) - (RFshockDefl + LRshockDefl)`, sign convention: positive = right-front + left-rear loaded |
| `traction_util_mean` | mean | `max(LFspeed, RFspeed, LRspeed, RRspeed) - min(...) ` divided by `Speed`, clipped at 1 |
| `aero_platform_front_rh_mean_mm` | mean | `LFrideHeight, RFrideHeight` mean |
| `aero_platform_rear_rh_mean_mm` | mean | `LRrideHeight, RRrideHeight` mean |
| `aero_platform_pitch_mean_mm` | mean | `(rear_rh) - (front_rh)` per sample, mean across phase |
| `roll_angle_max_rad, roll_angle_mean_rad` | max(abs) / mean | `Roll` |
| `roll_rate_max_rps` | max(abs) | `RollRate` |
| `damper_velocity_p99_mms` | p99(abs) | per-corner: derivative of `LFshockDefl, RFshockDefl, LRshockDefl, RRshockDefl`; output is the largest p99 across the four corners |
| `damper_velocity_mean_mms` | mean(abs) | as above, mean of the four corners |
| `air_density_mean` | mean | `AirDensity` |
| `track_temp_c_mean` | mean | `TrackTempCrew` |
| `wind_vel_ms_mean` | mean | `WindVel` |
| `wind_dir_deg_mean` | mean (circular) | `WindDir` — circular mean (atan2 of summed unit vectors) |
| `track_wetness_mean` | mean | `TrackWetness` |
| `n_samples` | count | rows in group |
| `data_quality_clean_frac` | mean | `data_quality_mask` (bool→float). Defaults to 1.0 if column absent (slice D not yet run) |

The five `*_mean` env columns are the per-phase materialisation of `EnvironmentFrame` (CLAUDE.md commitment: every sample carries env context, and the per-phase aggregation is the form the fitter consumes).

Edge cases for derived state:

- A phase with `n_samples == 0` (the walker passed through it instantaneously, e.g. a corner with no detectable trail-brake) is **omitted** from the output, not emitted with NaNs. Downstream consumers must handle absent phases per-corner.
- `understeer_angle_mean_rad` is undefined when `Speed^2` near zero; the `eps = 1.0 m/s` clip keeps it finite at pit speeds and the sample is masked out by phase boundaries before it matters in practice.

## 7. `EnvironmentFrame` contract

The full minimum contract is pinned in §2 and master plan §2. Restated for completeness:

- **Fields:** `air_density: float`, `track_temp_c: float`, `wind_vel_ms: float`, `wind_dir_deg: float`, `track_wetness: float`.
- **Sources (raw IBT channel names, persisted by slice A):** `AirDensity`, `TrackTempCrew`, `WindVel`, `WindDir`, `TrackWetness`.
- **Per-sample storage:** none. Slice B reads them sample-by-sample from the parquet via slice A's `lap_data` and aggregates them into the per-phase columns described in §6. The parquet remains the single source of truth for raw per-sample environment.
- **Adapter:** `EnvironmentFrame.from_row(row)` accepts a `dict[str, float]` with the five raw IBT channel keys above and returns a frozen `EnvironmentFrame`. KeyError on a missing key — never silent zero-fill (slice E's confidence path needs to know when env was unavailable).
- **Hashable & frozen:** so a per-phase `EnvironmentFrame` value can key a fitter cache.
- **No correction here.** Air-density correction belongs to slice C's aero loader; slice B emits the raw `AirDensity` mean. Track-temp tyre-stiffness correction is a slice E concern.

`AirTemp`, `AirPressure`, `RelativeHumidity` are intentionally **out** of `EnvironmentFrame`'s minimum contract. They are present in slice A's parquet (channel-allow-listed) and a future slice can add them if needed; the master plan reserves the minimum, not the maximum, so adding a field is a non-breaking change but removing one is breaking.

## 8. Failure handling

| Failure mode | Behaviour |
|---|---|
| Lap with `lap_index == -1` (slice A's pre-grid sentinel) | `segment_lap` accepts and processes; `corner_phase_states` skips it (out-laps and warm-ups are not training data). Caller gets an empty Polars frame, not an exception. |
| Lap where the lateral-G detector finds zero corners (e.g. oval) | `segment_lap` returns the input frame with `corner_id = -1` everywhere and `phase = STRAIGHT` everywhere. `corner_phase_states` returns an empty frame. Not an error — just an honest "this lap had no detectable cornering." See §12 open question on ovals. |
| Lap where the detector finds one giant corner (banked oval) | `segment_lap` labels the entire sustained-G region as `corner_id = 0`. The phase walker still runs. `MID_CORNER` will dominate; `BRAKING` and `EXIT` may be empty rows that get omitted per §6 edge-case rule. |
| Corner that spans a lap boundary (rare; possible on a botched out-lap) | Skipped. `segment_lap` only emits corners fully bracketed within the lap it was given. The caller orchestrates per-lap calls; cross-lap stitching is **not** in slice B. |
| Required input column missing (`AccelLat` not in frame) | `KeyError` raised explicitly, naming the missing channel. Never a silent zero. Slice A's allow-list guarantees these channels exist; a missing column means the caller passed something other than slice A's `lap_data` output. |
| Lap with internal sample gap (e.g. 3 missing 60 Hz frames) | `segment_lap` runs unchanged; the hysteresis in §4 absorbs gaps shorter than `T_EXIT_HOLD_MS`. Larger gaps may close a corner prematurely; this is acceptable — slice D will subsequently re-segment when track-position-based detection lands. |
| `data_quality_mask` column absent on the input frame | Treat as all-`True`. Logged once per session at `INFO` level. Not an error. |

`segment_lap` and `corner_phase_states` never silently mutate global state and never write to disk. All persistence is slice A or (eventually) slice D.

## 9. Idempotency / determinism

Both public functions are pure:

- `segment_lap(df)` returns a deterministic function of `df`. No clocks, no RNG, no global lookups (the threshold table is immutable). Calling twice on identical input returns frame-equal output (`assert_frame_equal`).
- `corner_phase_states(session_id, lap_index)` is deterministic given the parquet on disk. The aggregation rules in §6 are commutative-and-associative-friendly (mean, max, p99 with deterministic tie-breaking).

The only mutable state in the module is the `PER_CAR` config dict (§5), which is module-level and built once at import time; runtime never mutates it. A per-test override mechanism is provided via `corner.config.thresholds_for` accepting an explicit `PhaseThresholds` argument in tests, but the function-of-record (`segment_lap`) never accepts one — keeping the public API surface idempotent.

## 10. Testing

- **Unit — Phase enum & key.** Construct `CornerPhaseKey`, assert hashability, JSON-roundtrip via `dataclasses.asdict`-equivalent, equality.
- **Unit — synthetic lateral-G detection.** Construct a 60 Hz time series with a programmed `AccelLat` profile yielding a known number of corners (e.g. 5 sinusoidal humps each crossing 0.5 g). Run `segment_lap`. Assert detected corner count equals the constructed count, hysteresis suppresses a 0.49-g flutter hump, and `MIN_CORNER_DURATION_MS` rejects a sub-300-ms spike.
- **Unit — phase walker.** Hand-construct sample-level `(Brake, Throttle, Steering, AccelLat)` traces for a single corner that exercises every transition rule from §5. Assert each sample lands in the expected phase. Cover the "no trail-brake" degenerate path (BRAKING → MID_CORNER skipping TRAIL_BRAKE).
- **Unit — `EnvironmentFrame.from_row`.** Round-trip a dict, assert frozen, assert KeyError when a key is missing. Assert `air_density` of the constructed frame matches the input float exactly.
- **Fixture — BMW Sebring.** Use `ibtfiles/bmwlmdh_sebring international 2026-03-22 14-47-42.ibt` (the canonical BMW fixture from master plan §1, also the slice A primary fixture). Sebring International has 17 reference corners. Run `segment_lap` on the session's first valid lap and assert:
  - `≥ 12` distinct `corner_id` values (12 is the floor; some short corners may merge under the 0.5-g threshold).
  - For each detected corner, all four in-corner phases (`BRAKING, TRAIL_BRAKE, MID_CORNER, EXIT`) are populated with at least one sample. (Empty `EXIT` is allowed only if the corner runs into the very end of the lap-sample window.)
  - `corner_id` values form a contiguous 0..N-1 sequence.
  - Re-running `segment_lap` on the same frame returns a frame-equal result (idempotency check piggybacked on the fixture test).
- **Fixture — `corner_phase_states` round-trip.** On the same Sebring lap, assert `corner_phase_states(sid, 1)` returns a frame with one row per (corner_id, phase) actually present in the segmented lap, every column from §6 populated and finite (no NaN inside in-corner phases), and the five `*_mean` env columns equal to the per-phase mean of the corresponding raw IBT channels (recomputed independently in the test).
- **Per-car spot-check.** For each canonical fixture in master plan §1 — `acura/hockenheim_gp`, `bmw/sebring_international`, `cadillac/laguna_seca`, `ferrari/algarve_gp`, `porsche/algarve_gp` — assert `segment_lap` on the first valid lap returns at least one detected corner. Do **not** pin per-track corner counts: the master plan §6 risk row about over-constraining the algorithm is honoured here. The point is "it runs end-to-end on every car" not "it produces a specific count."
- **Determinism.** Same lap, two calls, `assert_frame_equal`. Same lap, two processes (subprocess), bytewise-identical parquet of the result frame.

## 11. Out of scope

Explicitly **not** in slice B:

- **Track-position-based corner segmentation.** Slice D. B uses lateral-G hysteresis as the standalone detector and reserves a `track_model=` keyword for D to plug into.
- **Curb / bump / off-track masking.** Slice D. B reads `data_quality_mask` if present and trusts it; B never computes it.
- **Aero-density correction.** Slice C. B emits raw mean `AirDensity` per phase; the aero loader applies the correction.
- **Confidence on derived state.** Slice E. The columns in §6 are point estimates; uncertainty wraps them later.
- **Recommendation rendering.** Slice F. `corner_phase_states` is a numerical surface, not a human report.
- **Setup ontology and per-car suspension parameter mapping.** Slice E. B reads channel names that are universal (`LFshockDefl` etc.), not setup parameters.
- **Cross-lap or cross-session aggregation.** B operates per-lap. Stitching is the fitter's job.
- **Persisting corner-phase output.** B is read-side only. No new parquet columns written by this slice. (Future slice D may decide to persist a `corner_id, phase` pair into the parquet for cache reasons; that is a slice-D decision, not slice B's.)
- **Adding `AirTemp, AirPressure, RelativeHumidity` to `EnvironmentFrame`.** Reserved for a later, additive slice. Master plan §2 minimum contract is the floor.

## 12. Open questions

1. **Ovals and near-ovals.** A pure oval has sustained lateral G but no distinct mid-corner phase in the GTP-road-course sense. Slice B's current behaviour: the entire banked turn becomes one `corner_id` and `MID_CORNER` consumes most of it; `BRAKING` and `EXIT` may be empty (and are then omitted per §6). Is this what slice E wants, or should ovals be detected as a distinct case and labelled `phase=BANKED` (a sixth enum value)? Defer to the first oval session that lands. **Working answer: keep five phases; let `MID_CORNER` dominate.**
2. **Sebring T17.** The long-radius right-hander before the front straight at Sebring is sometimes considered one corner (T17) and sometimes two (T16+T17 in iRacing's numbering). The lateral-G detector with the §4 thresholds will likely call it one corner because the G-load never drops below `LAT_G_EXIT` in between. The fixture test (§10) uses `≥ 12` deliberately to absorb this ambiguity. **Working answer: one corner under B's algorithm; slice D's track-position model will resolve it definitively.**
3. **Per-car threshold mechanism.** `PER_CAR` config dict (pinned §5). Compared against (a) a per-car class with method overrides and (b) a registry of functions keyed by car. The dict was picked because it (i) mirrors the `add-constraint` skill's data-not-code convention, (ii) serialises trivially for slice F's per-parameter justification, and (iii) keeps the override surface inspectable in one file. If a future car needs *behaviour* (not just numbers) different from the default state machine, that car's deviation deserves its own design discussion — not a class hierarchy added pre-emptively.
4. **`steering_geom` placeholder.** §6's `understeer_angle_mean_rad` uses a per-car constant for steering geometry that slice B sets to 1.0 across all cars. Slice E will fit this from data. Should slice B emit the column with the placeholder, or omit it until E provides the constant? **Working answer: emit it. The column existing keeps slice E's API stable; the placeholder coefficient is documented and obvious.**
5. **`brake_off_hold_ms` interaction with sample dropouts.** A 50-ms hold corresponds to ~3 samples at 60 Hz. A single dropped frame is enough to trigger spurious phase advance. Should the hold be sample-counted (3 samples) instead of time-counted (50 ms)? **Working answer: time-counted, because slice A's parquet is regular 60 Hz by construction; if slice A ever changes sample rate, the threshold needs revisiting and a time unit makes that obvious.**
