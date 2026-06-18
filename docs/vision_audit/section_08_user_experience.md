# VISION §8 — User Experience audit

## Section under audit

> **§8 User Experience — Simple commands, powerful output.**
>
> Running the optimizer should be dead simple from PowerShell. No complex
> flags, no remembering module paths, no wrapper scripts.
>
> ```powershell
> optimize bmw sebring --wing 17
> optimize learn "path\to\session.ibt"
> optimize bmw laguna-seca --wing 16
> optimize compare "session1.ibt" "session2.ibt"
> optimize status bmw
> ```
>
> The CLI should auto-detect the car from the IBT filename, auto-detect
> the track, and default to sensible options. Power users can override
> with flags, but the default path should be: drop an IBT file in, get a
> setup out. Install should be `pip install .` and the command should
> just work.

## Verdict

**Faithful — Green.** Every example command in §8 maps to a working
subcommand exposed by a single `optimize` console-script entry point.
Auto-detection of car & track from IBT filename is implemented and
tested per-car. Power-user flags (`--wing`, env overrides, `--pin`,
`--json`, `--no-cache`) exist as overrides, not as the primary
interface. All 50 `tests/cli/` tests pass on a clean install. The
"drop in an IBT, get a setup out" path is wired end-to-end.

## Evidence

### Console-script install (`pip install .` ⇒ `optimize` works)

`pyproject.toml:23-24` registers a single console script:

```toml
[project.scripts]
optimize = "racingoptimizer.cli:main"
```

After `uv pip install -e ".[dev]"` the `optimize` command resolves
without `python -m` or wrapper scripts. `uv run optimize --help`
returns the top-level help and lists the four real subcommands:
`compare`, `learn`, `recommend`, `status`.

### Top-level group + positional shorthands

`src/racingoptimizer/cli/__init__.py:31-44` defines `_OptimizeGroup`,
a `click.Group` whose `parse_args` rewrites the argv before Click sees
it:

- If the first positional is a member of `CANONICAL_CARS` (`acura`,
  `bmw`, `cadillac`, `ferrari`, `porsche`), prepend `recommend` so
  `optimize bmw sebring` becomes `optimize recommend bmw sebring`.
- If the first positional is an existing `.ibt` file, also prepend
  `recommend` — that triggers the auto-detect branch in
  `recommend_cmd`.

This is the mechanism that lets `optimize bmw sebring` and
`optimize ./session.ibt` both work without the user typing
`recommend`.

### Each VISION example maps to a subcommand

| VISION example | Implementation | Verified |
|---|---|---|
| `optimize bmw sebring --wing 17` | `recommend_cmd` (`cli/recommend.py:54-191`) — `--wing` is option `cli/recommend.py:60` | `uv run optimize bmw --help` lists `--wing FLOAT` |
| `optimize learn "path\to\session.ibt"` | `learn_command` (`ingest/cli.py:11-25`) — accepts file or directory via `_iter_ibt_paths` (`ingest/api.py:37`) | `uv run optimize learn --help` lists `PATH` |
| `optimize bmw laguna-seca --wing 16` | Same `recommend_cmd`. Hyphenated track is normalised through `slugify_track` (`ingest/detect.py:46-50`) before catalog lookup in `_resolve_track_or_extrapolate` (`cli/recommend.py:461-512`) | `slugify_track("laguna-seca") == "laguna_seca"`, validated by `tests/test_detect.py` (18 passed) |
| `optimize compare "session1.ibt" "session2.ibt"` | `compare_cmd` (`cli/recommend.py:199-276`) — auto-learns missing IBTs by default (`--no-auto-learn` opt-out) | `uv run optimize compare --help` shows `IBT_A IBT_B` |
| `optimize status bmw` | `status_cmd` (`cli/recommend.py:284-366`) — coverage per-track + fit-quality trend | `uv run optimize status --help` shows `CAR` arg |

### Auto-detect car & track from IBT filename ("drop in an IBT, get a setup out")

`src/racingoptimizer/ingest/detect.py` exposes the three primitives
named in the spec:

- `detect_car_from_filename(filename)` (`detect.py:67-70`) regex-matches
  the iRacing filename pattern `^<car>_<track> YYYY-MM-DD HH-MM-SS.ibt`.
- `detect_track_from_filename(filename)` (`detect.py:58-64`) extracts
  the track segment and runs it through `slugify_track`.
- `slugify_track(raw)` (`detect.py:46-50`) lowercases and replaces any
  run of non-alphanumerics with `_`, so `"Laguna Seca"`, `"laguna-seca"`,
  `"laguna_seca"` all collapse to the same slug.
- `normalize_car_key(raw)` (`detect.py:32-43`) maps raw iRacing prefixes
  (`bmwlmdh`, `bmwm4gt3`, `acuraarx06gtp`, …) onto canonical keys via
  `CAR_PREFIX_MAP` (`detect.py:15-25`), longest-prefix wins.

The auto-detect routing for `optimize <ibt_path>` lives in
`_resolve_car_track_or_exit` (`cli/recommend.py:388-429`). When the
first positional ends in `.ibt` and the file exists, both detection
helpers are invoked, the car prefix is canonicalised, and the resolved
`(car_key, track_slug)` are forwarded into the same code path explicit
`<car> <track>` invocations use — exactly the "drop in an IBT, get a
setup out" behaviour §8 calls for.

### Track-name flexibility

`_resolve_track_or_extrapolate` (`cli/recommend.py:461-512`) handles
three real-world inputs:

1. Slugified exact match (`"sebring_international"`).
2. Bare-alphanum fallback (`"sebringinternational"`).
3. Substring match against catalog slugs — so `optimize bmw sebring`
   resolves against a catalog entry stored as `sebring_international`.

When the car has IBT data on a different track but not the requested
one, the code picks the most-data donor track and flags the recommendation
as extrapolated with a `sparse` regime override
(`_force_sparse_regime`, `cli/recommend.py:515-530`). Test coverage:
`tests/cli/test_untrained_track.py`.

### Power-user flags exist but aren't the primary interface

`recommend_cmd` exposes the documented overrides (`cli/recommend.py:60-92`):

- `--wing` (pin rear wing angle in degrees) — directly mirrors
  `optimize bmw sebring --wing 17`.
- `--air-temp`, `--track-temp`, `--wind`, `--wetness` — environmental
  overrides that pass through to `_env_from_overrides`
  (`cli/recommend.py:657-681`).
- `--pin KEY=VAL` (repeatable) — pin an arbitrary parameter.
- `--json` — machine-readable output.
- `--corpus-root`, `--no-cache` — operational knobs.

The default path needs none of these — `optimize bmw sebring` runs to
completion with sensible defaults pulled from the corpus medians
(VISION §8: "default to sensible options").

### Sample output proves the chain produces the engineer briefing §7 mandates

`recommendations/bmw__spa_2024_up__20260505-180530.txt` (in the repo
root, generated by the CLI in normal use) shows the output format:
heading with car, track, conditions, confidence; then per-parameter
blocks listing value, +1/-1 click sensitivity, helps/hurts, and
evidence. This confirms §8's "powerful output" half — the simple
command produces the engineer-briefing format §7 requires.

## Tests run

```
uv run pytest -q tests/cli/
# 50 passed in 289.97s (0:04:49)
```

All 50 CLI tests pass on this worktree's fresh install
(`uv pip install -e ".[dev]"`). The known
`test_per_car_smoke.py::test_recommend_per_car_json` failure flagged in
the task brief did **not** reproduce in this run — the suite is
currently green.

```
uv run pytest -q tests/test_detect.py
# 18 passed in 0.23s
```

`detect_car_from_filename` / `detect_track_from_filename` /
`normalize_car_key` are exercised across all five canonical car
filenames.

## Gaps / observations (informational, not violations)

- `optimize bmw --help` shows the *recommend* help text rather than a
  car-scoped help page. This is correct behaviour given the
  `_OptimizeGroup` rewrite, but it means power users who actually want
  `recommend`'s help under its real name need `optimize recommend
  --help`. Both work; `bmw --help` is the friendlier surface.
- `CAR_PREFIX_MAP` (`detect.py:23-24`) maps `amvantageevogt3` to
  `bmw` "until we confirm where Aston Martin GT3 fits in the model" —
  the comment explicitly flags this as a placeholder. Doesn't affect
  §8 compliance for the five GTP cars.
- The task brief mentioned a known `test_recommend_per_car_json`
  failure due to `--json` echoing `[saved to ...]` to stderr. This
  did **not** reproduce in the current `tests/cli/` run — all 50
  tests passed cleanly. Either the regression was already fixed or
  the side-channel mixing is environment-specific. Worth noting but
  no action required for this audit.

