# Watkins Glen — per-car setup runbook

Goal: produce a setup recommendation for **every GTP car at Watkins Glen**, then
reason out the **best car** for the track. Written 2026-06-08; execute once the
telemetry lands.

## Status / blocker (as of 2026-06-08)

- **No Watkins Glen telemetry exists** in the corpus — `ibtfiles/` has zero `.ibt`
  matching `*glen*`/`*watkins*` (verified). Tracks present: Belle Isle, Daytona
  road, Hockenheim GP, Sebring, Spa 2024, Laguna Seca, Algarve.
- The optimizer is **track-specific**: `optimize <car> <track>` builds a per-track
  corner schedule from that track's sessions (`cli/recommend.py:1197`). With no Glen
  sessions for any car, the cross-car geometry borrow (`_maybe_borrow_cross_car_track`,
  `recommend.py:1250`) has nothing to borrow → it refuses for all five cars.
- **Plan:** driver runs ~10 clean laps per car at the Glen tomorrow → 5 `.ibt`
  files (one per car) → ingest → per-car recommend → compare.

## Step 0 — drop the telemetry

Put the 5 `.ibt` files in `ibtfiles/` (the dir is hook-protected against deletes,
not adds). For car/track auto-detection to work, keep iRacing's default filename
shape: `<carid>_<track> YYYY-MM-DD HH-MM-SS.ibt`, e.g.
`bmwlmdh_watkinsglen 2026-06-09 18-20-05.ibt`. Car ids are matched by
`ingest/detect.py:CAR_PREFIX_MAP` (`bmwlmdh`, `porsche963gtp`,
`cadillacvseriesrgtp`, `acuraarx06gtp`, `ferrari499p`). Quote paths — filenames
contain spaces.

> ~10 laps in one `.ibt` = **one session per car**. That's fine for the per-car
> surrogate (each car already has 20+ sessions across other tracks, so the
> thin-corpus refusal `_is_thin_corpus_for_recommend` won't fire — it gates on the
> car's *total* production sessions, not per-track). But Watkins Glen *within-track*
> coverage is 1 session, so expect sparse/noisy per-track confidence and many
> parameters pinned to their observed value. Pooling all 5 cars' Glen sessions
> gives a better shared corner schedule via the cross-car borrow.

## Step 1 — ingest

```bash
uv venv && uv pip install -e ".[dev]"      # fresh container only
uv run optimize learn ./ibtfiles           # already-ok sessions short-circuit; only new ones parse
```

Confirm the five new sessions landed and note the **exact track slug** the catalog
assigned (Watkins Glen has layout variants — e.g. Grand Prix "the Boot" vs Classic):

```bash
uv run optimize status bmw                  # shows tracks/coverage per car
# or inspect the catalog slug directly:
uv run python -c "from racingoptimizer.ingest import api, paths; import polars as pl; \
print(api.sessions(corpus_root=paths.resolve_corpus_root(None)).select(['car','track','lap_count']).filter(pl.col('track').str.contains('glen')))"
```

Use the slug it reports (likely `watkins_glen` or similar) for the recommend calls.
The CLI slugifies user input, so `optimize bmw "watkins glen"`, `watkins-glen`, and
`watkinsglen` all normalise — but verify against the catalog value first.

## Step 2 — per-car recommend (all five)

Race setup is the default. Run each car:

```bash
uv run optimize acura    <glen-slug> --output-file recommendations/acura-glen-race.txt
uv run optimize bmw      <glen-slug> --output-file recommendations/bmw-glen-race.txt
uv run optimize cadillac <glen-slug> --output-file recommendations/cadillac-glen-race.txt
uv run optimize ferrari  <glen-slug> --output-file recommendations/ferrari-glen-race.txt
uv run optimize porsche  <glen-slug> --output-file recommendations/porsche-glen-race.txt
```

Each run is a ~15-min per-car refit the first time (cache miss). Add `--json` for a
machine-readable copy if needed. Optional: `--quali --fuel N` for a one-lap stint.

**Gotchas to expect (and how to read them):**
- **Cold-start TrackModel** (<3 Glen sessions) returns zero curb/off-track masks, so
  no dirty-sample filtering yet — fine for a first pass; more laps later sharpen it.
- **Within-track thin pin**: params with <3 distinct Glen values pin to the observed
  value (`recommend.py:168-211`). That's expected with one session.
- **Cadillac static-RH clamp (AUDIT.md H2)**: Cadillac front ride-height predictions
  tend to fall below the aero-map envelope and get clamped — treat Cadillac's static
  RH / aero-balance readouts with suspicion (cross-check on track).
- **Auto-pins**: tyre cold pressure auto-pins to 152 kPa; race fuel auto-pins to the
  last Glen session's value. Override with `--pin`/`--fuel`.
- If a car refuses with a thin-corpus banner, run `optimize calibrate <car> <glen>`
  and/or add more laps for that car (Acura has the smallest corpus).

## Step 3 — determine the best car (weighted physics + lap time)

The optimizer does **not** rank cars natively (per-car-calibrated evaluator weights
make raw scores non-comparable, and VISION §6 forbids lap time as the objective). So
compute an explicit **weighted composite** per car from two cross-car-comparable
sub-scores, each min-max normalised across the five cars to `0..1`:

- **LapScore** — from the best clean lap of the 10 Glen laps (`laps` table,
  `valid=1`, take `min(lap_time_s)` per car). Normalise inverse lap time so the
  fastest car = 1.0, slowest = 0.0. Lap time is the one genuinely comparable axis.
- **PhysScore** — the car's mean per-corner-phase **utilization at the recommended
  setup** expressed as a fraction of its *own* achievable ceiling (so it reads as
  "how fully this setup exploits the car," which is comparable across cars even though
  absolute utilization is not), **minus a guardrail penalty** for any axle-ceiling /
  balance flags (`evaluator.guardrail_check`). Normalise to `0..1`.

```
Composite(car) = W_LAP * LapScore(car) + W_PHYS * PhysScore(car)
default weights: W_LAP = 0.6, W_PHYS = 0.4   # tune to taste; state the weights used
```

Rationale for the default tilt toward lap time: it's the only strictly comparable
metric, while PhysScore guards against picking a car that's quick on one thin session
but riding its grip ceiling (no headroom, likely to fall off with fuel/tyre wear).

**Supporting context** (not in the weighted score, but report it so the pick is
explainable): per-car telemetry traits from `corner_phase_states` (understeer angle,
traction utilisation, platform stability across the Glen's fast esses + heavy stops
into T1 / the Bus Stop chicane) and known car characteristics (`evaluator.py` weights,
geometry, tyre floor).

This is automated by **`scripts/compare_cars_at_track.py`** (added 2026-06-08), which
runs `optimize <car> <track> --json` per car (production path), reads the new
`score_total` field, pulls each car's best clean lap from the catalog, normalises both
axes, and prints the ranked table:

```bash
uv run python scripts/compare_cars_at_track.py "watkins glen"          # default 0.6/0.4
uv run python scripts/compare_cars_at_track.py "watkins glen" --w-lap 0.7 --w-phys 0.3
```

It emits: `rank | car | best lap | LapScore | PhysScore | Composite | notes`, sorted by
Composite, and lists cars excluded for thin corpus / no clean laps. The `score_total`
field it depends on was added to the recommend JSON in `explain/render_json.py`
(aggregate per-corner-phase utilization). **Caveat:** one session per car is thin
evidence; the ranking is provisional — more laps (esp. Acura, the smallest corpus)
firm it up.

## Where outputs land

`recommendations/<car>-<short-track>-<mode>-<MMDD>-<HHMM>.txt` by default
(`--output-file` overrides). `corpus/` (catalog + parquet + model caches) is
gitignored and regenerable from `ibtfiles/`.
