"""Per-car smoke: corner_phase_states runs end-to-end on each canonical car."""
from __future__ import annotations

from pathlib import Path

import pytest

from racingoptimizer.corner import corner_phase_states
from racingoptimizer.ingest.api import laps, learn

REPO_ROOT = Path(__file__).resolve().parents[2]
IBT_DIR = REPO_ROOT / "ibtfiles"

# Filename prefixes per the iRacing CarPath convention. `normalize_car_key`
# (slice A) maps these to the canonical car keys.
CAR_PREFIXES: dict[str, tuple[str, ...]] = {
    "bmw": ("bmwlmdh",),
    "acura": ("acuraarx06gtp",),
    "cadillac": ("cadillacvseriesrgtp",),
    "ferrari": ("ferrari499p",),
    "porsche": ("porsche963gtp",),
}


def _first_fixture_for(car_key: str) -> Path | None:
    if not IBT_DIR.is_dir():
        return None
    for prefix in CAR_PREFIXES[car_key]:
        for path in sorted(IBT_DIR.glob(f"{prefix}*.ibt")):
            return path
    return None


@pytest.mark.parametrize("car_key", sorted(CAR_PREFIXES))
def test_corner_phase_states_runs_for_each_canonical_car(
    car_key: str, tmp_corpus: Path
) -> None:
    fixture = _first_fixture_for(car_key)
    if fixture is None:
        pytest.skip(f"no {car_key} fixture present in ibtfiles/")

    sids = learn(fixture, corpus_root=tmp_corpus)
    assert sids
    sid = sids[0]

    valid = laps(session_id=sid, valid_only=True, corpus_root=tmp_corpus)
    if valid.height == 0:
        pytest.skip(f"{fixture.name} has no valid laps")

    out = corner_phase_states(sid, int(valid["lap_index"][0]), corpus_root=tmp_corpus)
    assert out.height > 0, f"no corners detected for {car_key} fixture {fixture.name}"
    assert (out["corner_id"] >= 0).all()
