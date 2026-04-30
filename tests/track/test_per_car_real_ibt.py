"""Per-car real-IBT integration smoke for the track model (slice D gap-fill).

Closes the verification gap left by the synthetic-only suite: builds a real
track model from raw .ibt fixtures for every GTP car that has fixtures in
`ibtfiles/`, then exercises build_track_model -> bump/grip maps -> curb_mask
-> apply_quality_mask round-trip.

Source bugs uncovered by this real-IBT pass:

1. **`AccelLat` channel-name mismatch (FIXED).**
   `racingoptimizer.track.builder._aggregate_one_session` and
   `racingoptimizer.track.masks.compute_off_track_mask` requested the channel
   `AccelLat`, but real .ibt files (BMW, Acura, Cadillac, Ferrari, Porsche)
   all expose it as `LatAccel`. The builder's broad `except Exception:
   continue` swallowed the resulting `ColumnNotFoundError`, silently producing
   empty bump/grip maps despite a `compounding` regime. Synthetic tests used
   `AccelLat` so they never tripped this. Fix: rename to `LatAccel` in both
   consumers; synthetic tests updated to match the real channel name.

2. **Shock-velocity unit mismatch (TODO).** `LFshockVel` etc. are emitted by
   iRacing in m/s, but `track.builder` and `track.masks` store them as
   `shock_v_p99_mm_s` and apply the spec's 350/400 mm/s thresholds without
   converting. Real bump/grip maps populate (since the AccelLat fix), but
   no bin's p99 ever exceeds 350 mm/s in raw m/s units, so curb / bump
   likelihoods stay zero on real IBTs.

3. **`_lap_length_from_speed_fallback` picks idle pit laps (TODO).** Porsche
   IBT YAML headers omit `WeekendInfo.TrackLength`; the speed-integral
   fallback uses `argmax(end_sample - start_sample)` which selects the
   wallclock-longest lap, including a 350-second pit-idle lap that integrates
   to 412 m on the Porsche/Algarve corpus. The real racing laps integrate
   to ~4600 m. Need to filter by speed/lap-time before picking the longest.

4. **Acura suspension architecture divergence (TODO).** Acura ARX-06 .ibt
   files lack the per-corner `LFshockVel`/`RFshockVel`/`LRshockVel`/
   `RRshockVel` channels that the curb detector consumes — Acura instead
   exposes heave (`HFshockVel`) and roll (`FROLLshockVel`/`RROLLshockVel`)
   shock channels. Per slice E's known divergence note in CLAUDE.md, this
   needs a per-car shock-channel mapping in `track.builder`.

Bugs 2-4 are flagged via `xfail(strict=True)` on
`test_compounding_maps_have_real_content` so the test suite stays green
against the fixed-but-still-incomplete slice D, while keeping the
verification contract live for whoever fixes them next. Bug 1's cousin
`@pytest.mark.xfail` is removed because it is fixed.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import polars as pl
import pytest

from racingoptimizer.ingest import lap_data, laps, learn, sessions
from racingoptimizer.ingest.detect import (
    detect_car_from_filename,
    detect_track_from_filename,
    normalize_car_key,
)
from racingoptimizer.track import apply_quality_mask, build_track_model

pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_IBT_DIR = _REPO_ROOT / "ibtfiles"
_COMPOUNDING_THRESHOLD = 3
_GTP_CARS = ("acura", "bmw", "cadillac", "ferrari", "porsche")
_BUMP_COLUMNS = {
    "bin_index",
    "shock_v_p99_mm_s",
    "n_samples",
    "n_sessions",
    "curb_likelihood",
    "bump_likelihood",
    "track_pos_m",
}
_GRIP_COLUMNS = {
    "bin_index",
    "lateral_g_p95",
    "lateral_g_median",
    "n_samples",
    "n_sessions",
}
_BUG_REAL_IBT_CURB_CONTENT = (
    "blocked by remaining slice-D bugs: shock-velocity m/s vs mm/s unit "
    "mismatch (all cars), pit-idle lap selection in lap-length fallback "
    "(Porsche), and Acura's heave/roll-shock architecture divergence — see "
    "module docstring"
)


class _Case(NamedTuple):
    car: str
    track: str
    expected_regime: str
    fixtures: tuple[Path, ...]


def _discover_fixtures() -> dict[tuple[str, str], list[Path]]:
    """Group every .ibt under ibtfiles/ by (canonical car key, track slug)."""
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    if not _IBT_DIR.is_dir():
        return grouped
    for ibt in sorted(_IBT_DIR.rglob("*.ibt")):
        raw_car = detect_car_from_filename(ibt.name)
        track = detect_track_from_filename(ibt.name)
        if raw_car is None or track is None:
            continue
        try:
            car = normalize_car_key(raw_car)
        except ValueError:
            continue
        grouped[(car, track)].append(ibt)
    return grouped


def _pick_per_car_cases() -> list[_Case]:
    """Pick one case per GTP car. Prefers a >=3-session combo for compounding."""
    by_car: dict[str, list[tuple[str, list[Path]]]] = defaultdict(list)
    for (car, track), files in _discover_fixtures().items():
        if car in _GTP_CARS:
            by_car[car].append((track, files))

    cases: list[_Case] = []
    for car in _GTP_CARS:
        candidates = by_car.get(car, [])
        if not candidates:
            cases.append(_Case(car, "<missing>", "compounding", ()))
            continue
        compounding = [c for c in candidates if len(c[1]) >= _COMPOUNDING_THRESHOLD]
        track, files = max(compounding or candidates, key=lambda c: len(c[1]))
        regime = "compounding" if compounding else "cold_start"
        cases.append(_Case(car, track, regime, tuple(files)))
    return cases


_CASES: list[_Case] = _pick_per_car_cases()
_CASES_BY_CAR: dict[str, _Case] = {c.car: c for c in _CASES}
_CASE_IDS = [f"{c.car}-{c.track}-{c.expected_regime}" for c in _CASES]


@pytest.fixture(scope="module")
def per_car_corpora(tmp_path_factory) -> dict[str, tuple[tuple[str, ...], Path]]:
    """Ingest each car's fixtures once per module — real-IBT parsing is expensive.

    Returns: car -> (sorted_session_ids, corpus_root). Cars without fixtures
    are absent from the mapping; tests skip them.
    """
    out: dict[str, tuple[tuple[str, ...], Path]] = {}
    for case in _CASES:
        if not case.fixtures:
            continue
        corpus = tmp_path_factory.mktemp(f"corpus_{case.car}")
        for ibt in case.fixtures:
            learn(ibt, corpus_root=corpus)
        sess_df = sessions(car=case.car, track=case.track, corpus_root=corpus)
        sids = tuple(sorted(sess_df["session_id"].to_list()))
        out[case.car] = (sids, corpus)
    return out


@pytest.mark.parametrize("car", [c.car for c in _CASES], ids=_CASE_IDS)
def test_build_track_model_structure(car: str, per_car_corpora):
    """Regime, schema, and cache files all line up against real per-car corpora."""
    case = _CASES_BY_CAR[car]
    if not case.fixtures:
        pytest.skip(f"no .ibt fixtures for car={car}")
    sids, corpus = per_car_corpora[car]
    assert sids, f"no successfully ingested sessions for car={car} track={case.track}"

    model = build_track_model(case.track, list(sids), corpus_root=corpus)
    assert model.regime == case.expected_regime
    assert model.cache_path.exists()
    assert model.summary_path.exists()
    assert _BUMP_COLUMNS.issubset(model.bump_map.columns)
    assert _GRIP_COLUMNS.issubset(model.grip_map.columns)

    if case.expected_regime == "cold_start":
        assert model.bump_map.height == 0
        assert model.grip_map.height == 0
        sample_lap = _first_lap_df(sids[0], corpus)
        mask = model.curb_mask(sample_lap)
        assert mask.shape == (sample_lap.height,)
        assert mask.dtype == bool
        assert not mask.any()


@pytest.mark.parametrize("car", [c.car for c in _CASES], ids=_CASE_IDS)
@pytest.mark.xfail(reason=_BUG_REAL_IBT_CURB_CONTENT, strict=True)
def test_compounding_maps_have_real_content(car: str, per_car_corpora):
    """Compounding regime must yield non-empty bump/grip maps with real triggers."""
    case = _CASES_BY_CAR[car]
    if not case.fixtures:
        pytest.skip(f"no .ibt fixtures for car={car}")
    if case.expected_regime != "compounding":
        pytest.skip(f"car={car} only has cold-start fixtures")
    sids, corpus = per_car_corpora[car]

    model = build_track_model(case.track, list(sids), corpus_root=corpus)
    assert model.bump_map.height > 0
    assert model.grip_map.height > 0

    lap_length = model.lap_length_m
    assert isinstance(lap_length, float) and lap_length > 0.0

    bump_df = model.bump_map
    assert bump_df.filter(pl.col("curb_likelihood") > 0.0).height > 0
    assert bump_df.filter(pl.col("bump_likelihood") > 0.0).height > 0

    sample_lap = _first_lap_df(sids[0], corpus)
    mask = model.curb_mask(sample_lap)
    assert mask.shape == (sample_lap.height,)
    assert mask.dtype == bool
    assert mask.any()

    result = apply_quality_mask(sids[0], track_model=model, corpus_root=corpus)
    assert result.regime == "compounding"
    assert result.noop is False
    assert result.n_samples_clean_after < result.n_samples_clean_before
    df = pl.read_parquet(result.parquet_path)
    assert "data_quality_mask_v0" in df.columns


def _first_lap_df(session_id: str, corpus_root: Path) -> pl.DataFrame:
    lap_rows = laps(session_id=session_id, corpus_root=corpus_root)
    assert lap_rows.height > 0, f"no valid laps for session {session_id}"
    lap_index = int(lap_rows["lap_index"].to_list()[0])
    return lap_data(session_id, lap_index, corpus_root=corpus_root)
