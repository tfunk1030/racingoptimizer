from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from racingoptimizer.confidence import Confidence
from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner import CornerPhaseKey, Phase
from racingoptimizer.ingest.api import learn, sessions
from racingoptimizer.physics import PhysicsModel, fit
from racingoptimizer.physics.exceptions import UntrainedError
from racingoptimizer.track import build_track_model

REPO_ROOT = Path(__file__).resolve().parents[2]
BMW_SEBRING_IBT = REPO_ROOT / "ibtfiles" / "bmwlmdh_sebring international 2026-03-22 14-52-24.ibt"


@pytest.fixture
def bmw_model(tmp_path: Path) -> tuple[PhysicsModel, dict[str, float]]:
    if not BMW_SEBRING_IBT.exists():
        pytest.skip(f"missing BMW Sebring fixture at {BMW_SEBRING_IBT}")
    root = tmp_path / "corpus"
    root.mkdir()
    sids = learn(BMW_SEBRING_IBT, corpus_root=root)
    sess_df = sessions(corpus_root=root)
    car = sess_df.row(0, named=True)["car"]
    track = sess_df.row(0, named=True)["track"]
    tm = build_track_model(track, sids, corpus_root=root)
    model = fit(car, sids, tm, corpus_root=root, k_folds=2, seed=0xC0FFEE)
    return model, dict(model.baseline_setup)


def test_predict_returns_typed_state(bmw_model) -> None:
    model, baseline = bmw_model
    env = EnvironmentFrame(
        air_density=1.18, track_temp_c=24.0, wind_vel_ms=2.5,
        wind_dir_deg=120.0, track_wetness=0.0,
    )
    # Find any (corner, phase) that has trained fitters.
    keys = sorted(model.fitters.keys())
    assert keys, "model must have at least one trained fitter"
    _, corner_id, phase_str, _ = keys[0]
    cpkey = CornerPhaseKey(
        session_id=model.session_ids[0],
        lap_index=1,
        corner_id=corner_id,
        phase=Phase(phase_str),
    )
    out = model.predict(baseline, env, cpkey)
    assert out.corner_phase_key == cpkey
    # At least one channel should resolve.
    assert len(out.states) >= 1
    for ch, conf in out.states.items():
        assert isinstance(ch, str)
        assert isinstance(conf, Confidence)


def test_predict_unknown_corner_returns_empty_states(bmw_model) -> None:
    model, baseline = bmw_model
    env = EnvironmentFrame(
        air_density=1.18, track_temp_c=24.0, wind_vel_ms=2.5,
        wind_dir_deg=120.0, track_wetness=0.0,
    )
    cpkey = CornerPhaseKey(
        session_id=model.session_ids[0],
        lap_index=1,
        corner_id=99999,
        phase=Phase.MID_CORNER,
    )
    out = model.predict(baseline, env, cpkey)
    assert out.states == {}
    assert out.untrained_channels == ()


def test_predict_pickle_round_trip(bmw_model) -> None:
    model_a, _ = bmw_model
    blob = pickle.dumps(model_a, protocol=pickle.HIGHEST_PROTOCOL)
    revived = pickle.loads(blob)  # noqa: S301 — controlled fixture
    assert revived.car == model_a.car
    assert revived.session_ids == model_a.session_ids
    assert sorted(revived.fitters.keys()) == sorted(model_a.fitters.keys())
    assert revived.untrained_parameters == model_a.untrained_parameters
    assert revived.aero_correction_available == model_a.aero_correction_available
    assert revived.baseline_setup == model_a.baseline_setup


def test_predict_untrained_error_path() -> None:
    """Predict raises UntrainedError on a fitter whose state is untrained.

    Calling predict on an untrained fitter directly (rather than via the model
    orchestrator) is the deterministic invariant we exercise here.
    """
    from racingoptimizer.physics.fitters import GPFitter
    fitter = GPFitter()
    with pytest.raises(UntrainedError):
        fitter.predict([[0.0]])
