"""Shared session-scoped fixtures for the physics test package.

The BMW Sebring fit takes ~6 minutes; sharing a single fitted model across
the U10 score / recommend / weights tests keeps the suite under 15 minutes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from racingoptimizer.ingest.api import learn, sessions
from racingoptimizer.physics import fit
from racingoptimizer.track import build_track_model

REPO_ROOT = Path(__file__).resolve().parents[2]
BMW_SEBRING_IBT = REPO_ROOT / "ibtfiles" / "bmwlmdh_sebring international 2026-03-22 14-52-24.ibt"


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
