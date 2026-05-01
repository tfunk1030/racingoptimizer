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
    from tests._lfs_util import is_unmaterialised_lfs_pointer, lfs_skip_message

    if not BMW_SEBRING_RICH_IBT.exists():
        pytest.skip(f"missing BMW Sebring fixture at {BMW_SEBRING_RICH_IBT}")
    if is_unmaterialised_lfs_pointer(BMW_SEBRING_RICH_IBT):
        pytest.skip(lfs_skip_message(BMW_SEBRING_RICH_IBT))
    root = tmp_path / "corpus"
    root.mkdir()
    sids = learn(BMW_SEBRING_RICH_IBT, corpus_root=root)
    assert sids
    return root, sids


@pytest.fixture
def bmw_sebring_minimal_corpus(tmp_path: Path) -> tuple[Path, list[str]]:
    from tests._lfs_util import is_unmaterialised_lfs_pointer, lfs_skip_message

    if not BMW_SEBRING_IBT.exists():
        pytest.skip(f"missing BMW Sebring fixture at {BMW_SEBRING_IBT}")
    if is_unmaterialised_lfs_pointer(BMW_SEBRING_IBT):
        pytest.skip(lfs_skip_message(BMW_SEBRING_IBT))
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


def test_fit_emits_v3_feature_schema(
    bmw_sebring_corpus: tuple[Path, list[str]],
) -> None:
    """Stage 3 advances the feature schema to v3 (joint multi-input model).

    Newly trained models must advertise schema version 3 so
    `PhysicsModel.predict` dispatches to the joint-fit predict path.
    Older pickles deserialise as v1/v2 (see backward-compat tests below).
    """
    from racingoptimizer.physics.fitter import ENV_FEATURE_SCHEMA_VERSION
    root, sids = bmw_sebring_corpus
    sess_df = sessions(corpus_root=root)
    car = sess_df.row(0, named=True)["car"]
    track = sess_df.row(0, named=True)["track"]
    tm = build_track_model(track, sids, corpus_root=root)
    model = fit(car, sids, tm, corpus_root=root, k_folds=2, seed=0xC0FFEE)
    assert model.feature_schema_version == ENV_FEATURE_SCHEMA_VERSION
    assert model.feature_schema_version == 3


def test_fit_per_quadruple_x_uses_full_setup_vector(
    bmw_sebring_corpus: tuple[Path, list[str]],
) -> None:
    """Stage-3 fitters consume the joint setup vector + 12 env channels.

    Each trained estimator's `n_features_in_` must equal
    ``len(feature_names) == n_setup_params + 12``. The fitter's stored
    `feature_names` is the source of truth for which slots are which.
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
        assert len(record.feature_names) == est.n_features_in_, (
            f"FitRecord.feature_names ({len(record.feature_names)}) must "
            f"match estimator.n_features_in_ ({est.n_features_in_})"
        )
        # Stage-3: the joint vector has at least the 12 env columns and
        # at least one bounded setup parameter (BMW Sebring fixture has
        # several). Total feature count > 12.
        assert est.n_features_in_ > 12, (
            f"Stage-3 fitter must include >=1 setup parameter on top of "
            f"the 12 env channels; got n_features_in_={est.n_features_in_}"
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


def test_pickle_round_trip_preserves_v3_schema() -> None:
    """Round-trip through pickle preserves `feature_schema_version=3` for new models.

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
    assert revived.feature_schema_version == 3


def test_legacy_fitrecord_pickle_revives_with_empty_feature_names() -> None:
    """Pre-Stage-3 FitRecord pickles lacked the `feature_names` slot.

    `FitRecord.__setstate__` must backfill the slot to `()` so revive
    succeeds and `_predict_legacy` keeps the per-parameter sum semantics.
    """
    from racingoptimizer.physics.fitters import GPFitter
    from racingoptimizer.physics.model import FitRecord

    legacy_state = [GPFitter(), 100, 0.1, 0.5]  # 4 slots, no feature_names
    rec = FitRecord.__new__(FitRecord)
    rec.__setstate__(legacy_state)
    assert rec.feature_names == ()
    assert rec.n_samples == 100
    assert rec.cv_residual_std == 0.1
    assert rec.signal_std == 0.5


def test_v2_pickle_revives_and_predict_dispatches_legacy() -> None:
    """Pre-Stage-3 pickles round-trip via the legacy per-parameter predict.

    A v2 model pickle contains 4-tuple keys (param, corner, phase, channel)
    and `feature_schema_version=2`. After revive, `__setstate__` keeps
    those slots intact and `predict` routes through `_predict_legacy`.
    """
    from racingoptimizer.physics.model import PhysicsModel

    # Synthesise a v2-shaped slot list (no feature_names on FitRecord, no
    # Stage-3 fitters; feature_schema_version=2).
    legacy_state = [
        "bmw",                          # car
        ("legacy_sid",),                # session_ids
        {"legacy_sid": "sebring"},      # track_models_used
        {},                             # fitters (empty, but 4-tuple-keyed when populated)
        {},                             # ontology
        None,                           # constraints
        (),                             # untrained_parameters
        False,                          # aero_correction_available
        {},                             # baseline_setup
        0xC0FFEE,                       # seed
        None,                           # car_baselines
        2,                              # feature_schema_version (v2)
    ]
    instance = PhysicsModel.__new__(PhysicsModel)
    instance.__setstate__(legacy_state)
    assert instance.feature_schema_version == 2
    # An empty-fitters predict call routes to legacy path without raising.
    from racingoptimizer.context import EnvironmentFrame
    from racingoptimizer.corner import CornerPhaseKey, Phase
    env = EnvironmentFrame(
        air_density=1.18, track_temp_c=24.0, wind_vel_ms=2.5,
        wind_dir_deg=120.0, track_wetness=0.0,
    )
    cpkey = CornerPhaseKey(
        session_id="legacy_sid", lap_index=1,
        corner_id=1, phase=Phase.MID_CORNER,
    )
    out = instance.predict({}, env, cpkey)
    # Legacy dispatch returns an empty states dict for an empty model.
    assert out.states == {}
