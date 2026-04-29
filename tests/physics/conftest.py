"""Shared session-scoped fixtures for the physics test package.

The BMW Sebring fit takes ~6 minutes; sharing a single fitted model across
the U10 score / recommend / weights tests keeps the suite under 15 minutes.

The per-car module-scoped factory below does the same for the per-car
gap-fill suite — fitting a model once per (car, track) and re-using it
for fit + predict + recommend assertions.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from racingoptimizer.ingest.api import learn, sessions
from racingoptimizer.ingest.detect import (
    detect_car_from_filename,
    detect_track_from_filename,
    normalize_car_key,
)
from racingoptimizer.physics import PhysicsModel, fit
from racingoptimizer.track import build_track_model

REPO_ROOT = Path(__file__).resolve().parents[2]
BMW_SEBRING_IBT = REPO_ROOT / "ibtfiles" / "bmwlmdh_sebring international 2026-03-22 14-52-24.ibt"
_IBT_DIR = REPO_ROOT / "ibtfiles"

GTP_CARS = ("acura", "bmw", "cadillac", "ferrari", "porsche")
# Hand-picked densest (car, track) combos. Falls back to the largest available
# combo when the preferred track has no fixtures.
PREFERRED_TRACK: dict[str, str] = {
    "acura": "hockenheim_gp",
    "bmw": "sebring_international",
    "cadillac": "lagunaseca",
    "ferrari": "hockenheim_gp",
    "porsche": "algarve_gp",
}
# Some (car, track) combos (e.g. Porsche/Algarve has 35) hold far more
# fixtures than needed to exercise the multi-session fit path. Cap so each
# per-car fit stays in the 30-60s envelope per spec §13.
_MAX_FIXTURES_PER_CASE = 6


@pytest.fixture(scope="session")
def bmw_model_session(tmp_path_factory):
    """Fit the BMW Sebring model once per pytest session.

    Returns (model, track, corpus_root). The fixture is shared across
    test_recommend.py / test_recommend_clamp.py / test_weight_corners.py /
    test_score_locality.py — none of these tests mutate the model.
    """
    if not BMW_SEBRING_IBT.exists():
        pytest.skip(f"missing BMW Sebring fixture at {BMW_SEBRING_IBT}")
    root = tmp_path_factory.mktemp("u10_corpus") / "corpus"
    root.mkdir()
    sids = learn(BMW_SEBRING_IBT, corpus_root=root)
    sess_df = sessions(corpus_root=root)
    car = sess_df.row(0, named=True)["car"]
    track = sess_df.row(0, named=True)["track"]
    tm = build_track_model(track, sids, corpus_root=root)
    model = fit(car, sids, tm, corpus_root=root, k_folds=2, seed=0xC0FFEE)
    return model, track, root


# --- per-car gap-fill helpers ---------------------------------------------


def _discover_fixtures() -> dict[tuple[str, str], list[Path]]:
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


def per_car_cases() -> list[tuple[str, str, tuple[Path, ...]]]:
    """One (car, track, fixtures) tuple per GTP car (empty fixtures = skip)."""
    grouped = _discover_fixtures()
    by_car: dict[str, list[tuple[str, list[Path]]]] = defaultdict(list)
    for (car, track), files in grouped.items():
        if car not in GTP_CARS:
            continue
        by_car[car].append((track, files))

    cases: list[tuple[str, str, tuple[Path, ...]]] = []
    for car in GTP_CARS:
        candidates = by_car.get(car, [])
        if not candidates:
            cases.append((car, "<missing>", ()))
            continue
        preferred = PREFERRED_TRACK.get(car)
        chosen: tuple[str, list[Path]] | None = None
        if preferred:
            for track, files in candidates:
                if track == preferred:
                    chosen = (track, files)
                    break
        if chosen is None:
            chosen = max(candidates, key=lambda c: len(c[1]))
        capped = chosen[1][:_MAX_FIXTURES_PER_CASE]
        cases.append((car, chosen[0], tuple(capped)))
    return cases


PER_CAR_CASES = per_car_cases()
PER_CAR_CASE_IDS = [f"{car}-{track}" for car, track, _ in PER_CAR_CASES]


# Module-level cache so the (car, track, fixtures) tuple maps deterministically
# to a single fit result per test session — avoids re-ingesting and re-fitting
# for every test that exercises the same car.
_MODEL_CACHE: dict[tuple[str, str, tuple[Path, ...]], tuple[PhysicsModel, Path]] = {}


@pytest.fixture(scope="session")
def per_car_model_factory(tmp_path_factory):
    """Return a callable that fits (and caches) one PhysicsModel per car.

    Sharing across fit + predict + pickle + recommend tests cuts per-car
    fit cost (~30s real-corpus) from ~4× to 1×.
    """

    def factory(car: str, track: str, fixtures: tuple[Path, ...]) -> tuple[PhysicsModel, Path]:
        key = (car, track, fixtures)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        root = tmp_path_factory.mktemp(f"per_car_{car}") / "corpus"
        root.mkdir()
        for ibt in fixtures:
            learn(ibt, corpus_root=root)
        sess_df = sessions(car=car, track=track, corpus_root=root)
        sids = sorted(sess_df["session_id"].to_list())
        assert sids, f"no successfully ingested sessions for car={car} track={track}"
        tm = build_track_model(track, sids, corpus_root=root)
        k_folds = max(2, min(3, len(sids)))
        model = fit(car, sids, tm, corpus_root=root, k_folds=k_folds, seed=0xC0FFEE)
        _MODEL_CACHE[key] = (model, root)
        return model, root

    return factory
