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


def test_fit_emits_v2_feature_schema(
    bmw_sebring_corpus: tuple[Path, list[str]],
) -> None:
    """S2.2 expanded the env feature vector from 5 to 12 channels.

    Newly trained models must advertise schema version 2 so
    `PhysicsModel.predict` knows to feed the full 12-feature env vector.
    Older pickles deserialise as v1 (see backward-compat test below).
    """
    from racingoptimizer.physics.fitter import ENV_FEATURE_SCHEMA_VERSION
    root, sids = bmw_sebring_corpus
    sess_df = sessions(corpus_root=root)
    car = sess_df.row(0, named=True)["car"]
    track = sess_df.row(0, named=True)["track"]
    tm = build_track_model(track, sids, corpus_root=root)
    model = fit(car, sids, tm, corpus_root=root, k_folds=2, seed=0xC0FFEE)
    assert model.feature_schema_version == ENV_FEATURE_SCHEMA_VERSION
    assert model.feature_schema_version == 2


def test_fit_per_quadruple_x_has_thirteen_features(
    bmw_sebring_corpus: tuple[Path, list[str]],
) -> None:
    """Each fitter trains on (1 param + 12 env) = 13 features under v2.

    Re-running the fitter on a single quadruple is the cheapest way to
    assert the per-row width — we just look at the trained estimator's
    `n_features_in_`. GP wraps it in `_gp`, RF wraps it in `_rf`.
    """
    root, sids = bmw_sebring_corpus
    sess_df = sessions(corpus_root=root)
    car = sess_df.row(0, named=True)["car"]
    track = sess_df.row(0, named=True)["track"]
    tm = build_track_model(track, sids, corpus_root=root)
    model = fit(car, sids, tm, corpus_root=root, k_folds=2, seed=0xC0FFEE)
    for record in model.fitters.values():
        if not record.fitter.is_trained:
            continue
        est = (
            getattr(record.fitter, "_gp", None)
            or getattr(record.fitter, "_rf", None)
        )
        if est is None:
            continue
        assert est.n_features_in_ == 13, (
            f"expected 1 param + 12 env = 13 features, got {est.n_features_in_}"
        )
        return
    pytest.skip("no trained fitter to inspect")


def test_v1_pickle_revives_with_default_schema_version() -> None:
    """Pre-S2.2 pickles lacked `feature_schema_version`; revive must backfill v1.

    Frozen+slots dataclasses pickle as a positional list ordered by
    `__slots__`. A pre-S2.2 pickle is shorter by one element (no
    `feature_schema_version` slot at the tail), and `__setstate__` must
    backfill v1 rather than leaving the slot uninitialised.
    """
    from racingoptimizer.physics.model import PhysicsModel

    # Pre-S2.2 slot list: 10 elements (no feature_schema_version).
    legacy_state = [
        "bmw",                       # car
        ("legacy_sid",),             # session_ids
        {"legacy_sid": "sebring"},   # track_models_used
        {},                          # fitters
        {},                          # ontology
        None,                        # constraints
        (),                          # untrained_parameters
        False,                       # aero_correction_available
        {},                          # baseline_setup
        0xC0FFEE,                    # seed
    ]
    instance = PhysicsModel.__new__(PhysicsModel)
    instance.__setstate__(legacy_state)
    assert instance.feature_schema_version == 1, (
        "v1 pickle (no feature_schema_version slot) must backfill to 1"
    )
    assert instance.car == "bmw"


def test_pickle_round_trip_preserves_v2_schema() -> None:
    """Round-trip through pickle preserves `feature_schema_version=2` for new models.

    Catches regressions in `__setstate__` — the slot must be populated
    from the positional list pickle emits.
    """
    import pickle as _pickle

    from racingoptimizer.physics.model import PhysicsModel

    m = PhysicsModel(car="bmw", session_ids=("s1",))
    revived = _pickle.loads(  # noqa: S301 — controlled fixture
        _pickle.dumps(m, protocol=_pickle.HIGHEST_PROTOCOL)
    )
    assert revived.car == "bmw"
    assert revived.feature_schema_version == 2
