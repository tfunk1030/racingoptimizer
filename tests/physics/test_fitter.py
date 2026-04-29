from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from racingoptimizer.confidence import Confidence
from racingoptimizer.ingest.api import learn, sessions
from racingoptimizer.physics import PhysicsModel, fit
from racingoptimizer.physics.exceptions import InsufficientDataError
from racingoptimizer.track import build_track_model

REPO_ROOT = Path(__file__).resolve().parents[2]
_IBT_DIR = REPO_ROOT / "ibtfiles"
# Spec names this BMW Sebring file as the canonical smoke fixture.
BMW_SEBRING_IBT = _IBT_DIR / "bmwlmdh_sebring international 2026-03-22 14-47-42.ibt"
# A multi-lap session lets us exercise the per-(corner, phase) fit path,
# which the 14-47 single-lap fixture cannot. Auto-skip when missing.
BMW_SEBRING_RICH_IBT = _IBT_DIR / "bmwlmdh_sebring international 2026-03-22 14-52-24.ibt"


@pytest.fixture
def bmw_sebring_corpus(tmp_path: Path) -> tuple[Path, list[str]]:
    if not BMW_SEBRING_RICH_IBT.exists():
        pytest.skip(f"missing BMW Sebring fixture at {BMW_SEBRING_RICH_IBT}")
    root = tmp_path / "corpus"
    root.mkdir()
    sids = learn(BMW_SEBRING_RICH_IBT, corpus_root=root)
    assert sids
    return root, sids


@pytest.fixture
def bmw_sebring_minimal_corpus(tmp_path: Path) -> tuple[Path, list[str]]:
    if not BMW_SEBRING_IBT.exists():
        pytest.skip(f"missing BMW Sebring fixture at {BMW_SEBRING_IBT}")
    root = tmp_path / "corpus"
    root.mkdir()
    sids = learn(BMW_SEBRING_IBT, corpus_root=root)
    assert sids
    return root, sids


def test_fit_orchestration_smoke(bmw_sebring_corpus: tuple[Path, list[str]]) -> None:
    root, sids = bmw_sebring_corpus
    sess_df = sessions(corpus_root=root)
    car = sess_df.row(0, named=True)["car"]
    track = sess_df.row(0, named=True)["track"]
    tm = build_track_model(track, sids, corpus_root=root)
    model = fit(car, sids, tm, corpus_root=root, k_folds=2, seed=0xC0FFEE)
    assert isinstance(model, PhysicsModel)
    assert model.car == car
    assert model.session_ids == tuple(sorted(sids))
    # At least one fitter must be trained on the BMW Sebring fixture.
    assert any(rec.fitter.is_trained for rec in model.fitters.values())
    # Confidence.regime must be derivable from at least one record (n>=0).
    rec = next(iter(model.fitters.values()))
    conf = Confidence.derive(
        value=0.0,
        n_samples=rec.n_samples,
        cv_residual_std=rec.cv_residual_std,
        signal_std=max(rec.signal_std, 1e-9),
    )
    assert conf.regime in {"sparse", "noisy", "confident", "dense"}


def test_fit_pickle_round_trip(bmw_sebring_corpus: tuple[Path, list[str]]) -> None:
    root, sids = bmw_sebring_corpus
    sess_df = sessions(corpus_root=root)
    car = sess_df.row(0, named=True)["car"]
    track = sess_df.row(0, named=True)["track"]
    tm = build_track_model(track, sids, corpus_root=root)
    model_a = fit(car, sids, tm, corpus_root=root, k_folds=2, seed=0xC0FFEE)
    model_b = fit(car, sids, tm, corpus_root=root, k_folds=2, seed=0xC0FFEE)
    blob_a = pickle.dumps(model_a, protocol=pickle.HIGHEST_PROTOCOL)
    blob_b = pickle.dumps(model_b, protocol=pickle.HIGHEST_PROTOCOL)
    assert blob_a == blob_b


def test_fit_cold_start_tags_sparse(
    bmw_sebring_corpus: tuple[Path, list[str]],
) -> None:
    root, sids = bmw_sebring_corpus
    sess_df = sessions(corpus_root=root)
    car = sess_df.row(0, named=True)["car"]
    track = sess_df.row(0, named=True)["track"]
    tm = build_track_model(track, sids, corpus_root=root)
    model = fit(car, sids, tm, corpus_root=root, k_folds=2, seed=0xC0FFEE)
    # Cold-start corpus: every fitter has at most ~17 corner-phase rows per
    # fitter (a handful of laps), so Confidence.derive(...) produces `sparse`
    # for any n<30 record. The model's untrained list still exists.
    assert isinstance(model.untrained_parameters, tuple)
    assert all(isinstance(p, str) for p in model.untrained_parameters)
    # At least one fitter is sparse-grade by sample count.
    sparse_count = sum(1 for r in model.fitters.values() if r.n_samples < 30)
    assert sparse_count >= 1


def test_fit_session_car_mismatch_raises(
    bmw_sebring_corpus: tuple[Path, list[str]],
) -> None:
    root, sids = bmw_sebring_corpus
    sess_df = sessions(corpus_root=root)
    track = sess_df.row(0, named=True)["track"]
    tm = build_track_model(track, sids, corpus_root=root)
    with pytest.raises(ValueError, match="does not match"):
        fit("ferrari", sids, tm, corpus_root=root, k_folds=2)


def test_fit_empty_session_list_raises() -> None:
    with pytest.raises(InsufficientDataError):
        # Track model is unused along the empty-list short-circuit.
        fit("bmw", [], None, k_folds=2)  # type: ignore[arg-type]
