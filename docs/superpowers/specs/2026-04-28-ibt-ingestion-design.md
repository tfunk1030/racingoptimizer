# IBT Ingestion — Design Spec

**Date:** 2026-04-28
**Slice:** A — IBT ingestion (first of the VISION subsystems)
**Module:** `racingoptimizer.ingest`

## 1. Context

`racingoptimizer` is a physics-based setup optimizer for iRacing GTP cars (see `VISION.md`). The repo currently holds a vision document, hard-bounds (`constraints.md`), parsed aero-map JSONs, and ~5 GB of raw `.ibt` telemetry — and zero code. Every other planned subsystem (corner-phase decomposition, physics fitter, evaluator, optimizer, track model) reads from telemetry. Ingestion is therefore the gateway slice and its choices constrain everything downstream.

The deliverable of this spec is a Python module `racingoptimizer.ingest` that:

- Parses `.ibt` files using `pyirsdk`
- Persists each session as a parquet file plus a row in a SQLite catalog
- Exposes a small Polars-typed query API (`learn`, `sessions`, `laps`, `lap_data`)
- Is idempotent, recursive over directories, and never silently drops a failure

This is one slice. Corner-phase decomposition, aero-map interpolation, fitter, optimizer, track model, and CLI scaffolding are explicit non-goals here and get their own specs later.

## 2. Public API

```python
from racingoptimizer.ingest import learn, sessions, laps, lap_data

learn(path: str | Path) -> list[str]
    # Idempotent. Walks directories recursively. Returns session_ids for every
    # .ibt file processed, regardless of status (ok / partial / failed).
    # Caller can join against `sessions(...)` to inspect outcomes.
    # Adds nothing on a re-run of files that previously ingested cleanly.

sessions(car=None, track=None, since=None, valid_only=True) -> pl.DataFrame
    # Catalog query. One row per session: metadata + setup + weather summary
    # + lap_count + ingest status.

laps(session_id=None, car=None, track=None, valid_only=True) -> pl.DataFrame
    # One row per lap: session_id, lap_index, lap_time_s, weather snapshot,
    # mean fuel, valid/best flags.

lap_data(session_id: str, lap_index: int, channels: list[str] | None = None) -> pl.DataFrame
    # Bulk 60 Hz read for one lap. Looks up (start_sample, end_sample) in the
    # catalog `laps` table, then slices that range from the session's parquet
    # via column-projected scan. channels=None returns everything.
    # Wide format: rows = samples, columns = channel names.
```

Polars (not pandas) for return types — faster on this size of data, lazy frames where useful, native parquet/arrow stack.

## 3. On-disk layout

```
corpus/
  catalog.sqlite
  sessions/
    <car>/<track>/<session_id>.parquet
```

`session_id = sha256(ibt_bytes)[:16]`. Hash-based id is the mechanism that makes `learn` idempotent — same content → same id, regardless of filename or path. Also handles the "same session got copied to two locations" case.

`corpus/` location is configurable via env var `RACINGOPTIMIZER_CORPUS`. Default: `<repo>/corpus/`.

Parquet shape: wide, one row per 60 Hz sample, columns = raw IBT channel names plus three derived columns prepended:

- `t_s` — float, seconds since session start
- `lap_index` — int, lap number (`-1` for out-laps / pit / pre-grid)
- `lap_dist_pct` — already in IBT, retained at the front for convenience

Compression: ZSTD. Telemetry compresses well; expect 3–5× reduction. Disk budget ~5 GB total, well within tolerable.

## 4. Catalog schema (SQLite, single file)

```sql
CREATE TABLE sessions (
  session_id      TEXT PRIMARY KEY,
  car             TEXT NOT NULL,
  track           TEXT NOT NULL,
  recorded_at     TEXT,            -- ISO timestamp from IBT YAML
  duration_s      REAL,
  lap_count       INTEGER,
  weather_summary TEXT,            -- JSON: AirTemp/TrackTemp/Skies/wind means
  setup           TEXT,            -- JSON: full nested garage setup as parsed
  source_path     TEXT,            -- original .ibt path at first ingest
  ingested_at     TEXT NOT NULL,
  parquet_path    TEXT,            -- nullable when status = 'failed'
  status          TEXT NOT NULL CHECK(status IN ('ok','partial','failed')),
  error           TEXT             -- nullable, populated when status != 'ok'
);

CREATE TABLE laps (
  session_id   TEXT NOT NULL,
  lap_index    INTEGER NOT NULL,
  lap_time_s   REAL,
  start_sample INTEGER NOT NULL,
  end_sample   INTEGER NOT NULL,
  valid        INTEGER NOT NULL,   -- 0/1: completed clean lap
  best         INTEGER NOT NULL,   -- 0/1: session-best valid lap
  PRIMARY KEY (session_id, lap_index),
  FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX idx_sessions_car_track ON sessions(car, track);
CREATE INDEX idx_laps_session ON laps(session_id);
```

Garage setup as a JSON column rather than a typed schema per car: locks in the "use everything, lose nothing" rule without committing to a setup ontology this early. Setup-aware code can `json_extract` into specific paths.

## 5. Channel selection

Default: keep every scalar numeric channel from the IBT.

**Reject** by default:

- 64-element multi-driver arrays (`CarIdx*`) — huge and unrelated to setup physics.
- Per-tyre per-spot temperature/pressure arrays where the IBT exposes them as full arrays at 60 Hz. Per-tyre scalars stay; per-spot arrays drop.

This is "lose nothing physics-relevant," not "literally every byte." VISION explicitly cares about: shocks (4 corners), ride heights (4 corners), G-forces, speeds, brake/throttle/steering, yaw/pitch/roll rates, wheel speeds, tyre temps + pressures (per-tyre scalars), fuel, downforce, weather. All of those are scalars and stay.

The reject list lives as an explicit constant `EXCLUDED_CHANNEL_PATTERNS` in `parser.py` so it is auditable and easy to widen.

Channel names: kept as raw IBT names. No normalization layer until something concrete forces one (e.g., a future iRacing build renames a channel and a fitter cares).

## 6. Car / track detection

YAML header is authoritative. Filename used only as a fallback hint when YAML is missing or unparseable.

- Car key normalized lowercase to match `aero-maps/` filenames: `acura`, `bmw`, `cadillac`, `ferrari`, `porsche`. Mapping table from full iRacing car names (e.g. `acuraarx06gtp`) to canonical key lives in `detect.py`.
- Track name slugified: lowercase, spaces -> `_`, drop punctuation. Examples: `Daytona 2011 road` -> `daytona_2011_road`, `Algarve GP` -> `algarve_gp`.

If neither YAML nor filename yields a known car key, `learn` records the session with `status='failed'` and `error='unknown car: <raw>'` — the file is registered (not silently skipped), and the user can extend the mapping table and re-run.

## 7. Lap segmentation

A lap boundary is detected where `LapDistPct` rolls back: `prev > 0.9 and curr < 0.1`.

A lap is `valid=1` iff:

1. It both starts and ends with such a rollover, AND
2. The IBT `Lap` channel monotonically increases by exactly 1 across it.

Out-laps, in-laps, and incomplete first/last laps are recorded with `valid=0` but still indexed — VISION may want them later for weather / setup-change detection. `lap_index = -1` is reserved for samples before the first valid lap rollover (pre-grid / pit-out warm-up).

`best=1` is set on the single fastest `valid=1` lap per session. Ties resolved by lower `lap_index`.

## 8. Failure handling

| Failure mode | Catalog row? | `status` | Parquet written? |
|---|---|---|---|
| File doesn't open / corrupt header | Yes | `failed` | No |
| Header parses, no completed laps | Yes | `partial` | Yes (whatever samples exist) |
| Header + some laps, parser exception mid-stream | Yes | `partial` | Yes (samples up to failure point) |
| Completes cleanly | Yes | `ok` | Yes |

`learn` never silently drops a file. If asked to ingest 30 files and 2 fail, it logs a summary, registers the failures in the catalog, and returns success ids for the other 28.

## 9. Idempotency

`learn(path)` reads the IBT, hashes the bytes, queries `sessions` for that hash:

- Present -> return existing id, do nothing.
- Absent -> parse, write parquet, insert catalog row.

Re-running `learn` on the entire `ibtfiles/` tree is therefore cheap (one hash per file, no parsing).

If a session previously ingested as `failed` or `partial` is presented again with the same hash, `learn` re-attempts ingestion (treats it as new). This lets the user fix a bug in `parser.py` and re-run without manual catalog cleanup.

## 10. Module layout

```
src/racingoptimizer/
  __init__.py
  ingest/
    __init__.py        # public API re-exports: learn, sessions, laps, lap_data
    catalog.py         # SQLite open/migrate, CRUD helpers, query builders
    parser.py          # pyirsdk wrapper: yields setup, weather, samples, lap boundaries
    writer.py          # given parser output -> parquet + catalog row
    detect.py          # car/track inference, slugification, name maps
    paths.py           # corpus root, parquet path layout
    cli.py             # `optimize learn <path>` entry point
  py.typed
pyproject.toml         # uv-managed; deps: pyirsdk, polars, pyarrow, click
tests/
  test_detect.py
  test_catalog.py
  test_lap_segmentation.py
  test_ingest_smoke.py # end-to-end on one small real IBT
fixtures/
  small.ibt            # copy of ferrari499p_algarve...17-58-04.ibt (~1.5 MB)
```

## 11. Testing

- **Unit:** `detect.py` (filename -> car/track; YAML conflict -> YAML wins), `catalog.py` (open / migrate / insert / query against in-memory SQLite), lap-boundary detection on synthetic `LapDistPct` arrays.
- **Integration smoke:** ingest the small fixture IBT, assert: `status='ok'`, `lap_count > 0`, parquet readable, `lap_data(sid, 0)` returns a frame containing `Speed`, `Brake`, `Throttle`, `LFshockDefl` columns and >100 rows.
- **Idempotency (clean case):** ingest a known-good fixture twice, assert second call returns the same id, adds zero new catalog rows, and writes zero new parquet bytes (parquet mtime unchanged).
- **Retry of partial (§9 retry path):** seed the catalog with a `status='partial'` row whose `session_id` matches the fixture's content hash, then call `learn` on the fixture; assert the row is updated to `status='ok'` and a parquet is now present.
- **Failure:** truncate a fixture mid-file, assert `status='partial'` and `error` is non-null. Pass a non-IBT file, assert `status='failed'` and no parquet written.

## 12. Out of scope

- Corner segmentation / phase decomposition. Slice **B**.
- Aero-map loader / air-density correction. Slice **C**. Ingestion stores raw weather channels; downstream consumers correct.
- Track model. Slice **D**.
- Any fitting or optimization. Later slices.
- A schema-migration system for the catalog. The catalog is rebuildable from raw IBTs via `learn` — if the schema changes, drop the SQLite and re-ingest.

## 13. Open questions / future work

- **Setup ontology.** We persist the garage setup as opaque JSON now. Eventually a typed per-car schema will be useful (for the fitter and the optimizer to reason about parameter interactions). That's a separate spec.
- **Weather as time series vs summary.** The catalog stores a weather summary; the parquet retains raw per-sample weather channels. The fitter (slice E) will decide whether it wants per-sample resolution everywhere or aggregated-per-corner-phase.
- **Channel allow-list discovery.** As more cars / tracks land, the reject list may need extending. Plan to surface a `learn --report-channels` flag that prints kept-vs-rejected counts after the fact.
