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

3. **`_lap_length_from_speed_fallback` picks idle pit laps (FIXED).** Porsche
   IBT YAML headers omit `WeekendInfo.TrackLength`; the old speed-integral
   fallback used `argmax(end_sample - start_sample)` which selected the
   wallclock-longest lap, including a 350-second pit-idle lap that integrated
   to 412 m on the Porsche/Algarve corpus instead of the real ~4600 m. Fix:
   filter out non-racing laps by minimum mean Speed (>= 30 m/s for GTP
   cars), then pick the candidate with the highest mean Speed.

4. **Acura suspension architecture divergence (FIXED — S1.3).** Acura ARX-06
   .ibt files lack the per-corner `LFshockVel`/`RFshockVel`/`LRshockVel`/
   `RRshockVel` channels that the curb detector consumes — Acura instead
   exposes heave (`HFshockVel`/`TRshockVel`) and roll (`FROLLshockVel`/
   `RROLLshockVel`) shock channels. Pre-fix the bare
   `except Exception: continue` in `_aggregate_one_session` swallowed the
   resulting `ColumnNotFoundError` and silently dropped every Acura lap.
   Fix: `track.masks.shock_vel_channels(car)` resolves the right channel
   names per car; the builder threads the catalog's `car` field down and
   uses a typed `ColumnNotFoundError` catch with the spec §9 warning.
   Synthetic-only coverage lives in
   `tests/track/test_per_car_channel_mapping.py`.

Bugs 2-3 remain flagged via `xfail(strict=True)` on
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
_BUG_REAL_IBT_ACURA_AGREEMENT = (
    "acura-only: with S1.3's per-car heave/roll-shock channel mapping, "
    "Acura sessions DO produce non-zero curb_likelihood values, but the "
    "0.6 cross-session agreement threshold (calibrated against per-corner "
    "shock signals) is too strict for the heave/roll signal. Curb-mask is "
    "all-False on the sample lap because no Acura bin clears 60% agreement. "
    "Needs per-car or per-signal threshold recalibration as a follow-up."
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
def test_compounding_maps_have_real_content(car: str, per_car_corpora, request):
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

    # Some sessions have zero laps (e.g. a session that crashed before the
    # first lap completed); skip past them to find one with a real flying
    # lap so the mask intersection actually has bins to flag.
    sample_lap = None
    sample_sid = None
    for sid in sids:
        try:
            sample_lap = _first_lap_df(sid, corpus)
        except AssertionError:
            continue
        sample_sid = sid
        break
    assert sample_lap is not None and sample_sid is not None, (
        f"no session in {sids} has a usable lap"
    )
    mask = model.curb_mask(sample_lap)
    assert mask.shape == (sample_lap.height,)
    assert mask.dtype == bool
    assert mask.any()

    result = apply_quality_mask(sample_sid, track_model=model, corpus_root=corpus)
    assert result.regime == "compounding"
    assert result.noop is False
    assert result.n_samples_clean_after < result.n_samples_clean_before
    df = pl.read_parquet(result.parquet_path)
    assert "data_quality_mask_v0" in df.columns


def _first_lap_df(session_id: str, corpus_root: Path) -> pl.DataFrame:
    """Return a representative full racing lap for a session.

    Earlier this helper picked ``lap_index[0]`` unconditionally — but iRacing
    sessions begin with a short pit-out lap (e.g. 9 s for the Acura corpus)
    that barely moves and visits only a handful of track bins. Some sessions
    also include 300+ s aggregate laps (multi-pit-stops collapsed into one
    lap_index by iRacing's lap detection). Curb-mask intersection tests need
    a full *flying* lap, so:

    1. Take ``best=1`` when iRacing flagged a fastest lap (it is by definition
       a complete on-pace lap).
    2. Else filter valid laps to plausible GTP lap times (30–200 s) and take
       the one with the most samples — the longest clean lap.
    3. Else fall back to whatever the catalog lists first.
    """
    lap_rows_valid = laps(session_id=session_id, corpus_root=corpus_root, valid_only=True)
    lap_rows_all = laps(session_id=session_id, corpus_root=corpus_root)
    lap_rows = lap_rows_valid if lap_rows_valid.height > 0 else lap_rows_all
    assert lap_rows.height > 0, f"no laps at all for session {session_id}"
    plausible = lap_rows.filter(
        (pl.col("lap_time_s") >= 30.0) & (pl.col("lap_time_s") <= 200.0)
    )
    plausible_best = plausible.filter(pl.col("best") == 1)
    if plausible_best.height > 0:
        candidates = plausible_best
    elif plausible.height > 0:
        candidates = plausible
    else:
        candidates = lap_rows
    candidates = candidates.with_columns(
        (pl.col("end_sample") - pl.col("start_sample")).alias("_n_samples")
    ).sort("_n_samples", descending=True)
    lap_index = int(candidates["lap_index"].to_list()[0])
    return lap_data(session_id, lap_index, corpus_root=corpus_root)
