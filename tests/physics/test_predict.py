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
    from tests._lfs_util import is_unmaterialised_lfs_pointer, lfs_skip_message
    if not BMW_SEBRING_IBT.exists():
        pytest.skip(f"missing BMW Sebring fixture at {BMW_SEBRING_IBT}")
    if is_unmaterialised_lfs_pointer(BMW_SEBRING_IBT):
        pytest.skip(lfs_skip_message(BMW_SEBRING_IBT))
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
    # Stage-3 keys are (corner_id, phase, channel); legacy keys are
    # (param, corner_id, phase, channel).
    first = keys[0]
    if len(first) == 3:
        corner_id, phase_str, _ = first
    else:
        _, corner_id, phase_str, _ = first
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


def test_predict_works_with_v2_full_environment_frame(bmw_model) -> None:
    """v3 model + 12-channel EnvironmentFrame: predict consumes the full vector.

    Stage-3 fits emit `feature_schema_version=3`; the joint multi-input
    fitter still consumes the same 12-channel env vector under the hood.
    """
    model, baseline = bmw_model
    assert model.feature_schema_version == 3
    env = EnvironmentFrame(
        air_temp_c=22.5,
        air_density=1.18,
        air_pressure_mbar=1013.0,
        relative_humidity=0.5,
        wind_vel_ms=2.5,
        wind_dir_deg=120.0,
        fog_level=0.0,
        track_temp_c=24.0,
        track_wetness=0.0,
        weather_declared_wet=False,
        precip_type=0,
        skies=1,
    )
    keys = sorted(model.fitters.keys())
    assert keys
    first = keys[0]
    if len(first) == 3:
        corner_id, phase_str, _ = first
    else:
        _, corner_id, phase_str, _ = first
    cpkey = CornerPhaseKey(
        session_id=model.session_ids[0],
        lap_index=1,
        corner_id=corner_id,
        phase=Phase(phase_str),
    )
    out = model.predict(baseline, env, cpkey)
    assert len(out.states) >= 1


def test_predict_env_dispatch_picks_right_vector_for_schema_version() -> None:
    """v1 schema -> 5-feature env vector; v2 schema -> 12-feature env vector.

    Direct unit test on the dispatch helpers. The integration test against
    a real v1 model is impractical (we don't ship a pre-S2.2 pickle), but
    the dispatch is the single byte of logic that wires the right vector
    to the per-fitter `predict` call.
    """
    from racingoptimizer.physics.model import _env_to_array, _env_to_array_v1

    env = EnvironmentFrame(
        air_temp_c=22.5,
        air_density=1.18,
        air_pressure_mbar=1013.0,
        relative_humidity=0.5,
        wind_vel_ms=2.5,
        wind_dir_deg=120.0,
        fog_level=0.0,
        track_temp_c=24.0,
        track_wetness=0.0,
        weather_declared_wet=False,
        precip_type=0,
        skies=1,
    )
    v1 = _env_to_array_v1(env)
    v2 = _env_to_array(env)
    assert v1.shape == (5,)
    assert v2.shape == (12,)
    # v1 prefix: (air_density, track_temp_c, wind_vel_ms, wind_dir_deg,
    # track_wetness) — NOT a slice of v2 because v2 reorders to put air_temp
    # and pressure first (matching VISION section 10 ordering). This is
    # why predict builds the v1 vector explicitly rather than slicing v2.
    assert v1[0] == 1.18    # air_density
    assert v1[1] == 24.0    # track_temp_c
    assert v1[2] == 2.5     # wind_vel_ms
    assert v1[3] == 120.0   # wind_dir_deg
    assert v1[4] == 0.0     # track_wetness


def test_predict_with_partial_env_frame_returns_state(bmw_model) -> None:
    """A partial env frame (missing -> sentinels) is a valid construction.

    Strictness lives in `EnvironmentFrame.from_row`, not in `predict`.
    Slice F's CLI synthesises env frames from session medians and may end
    up with NaN/-1 sentinels for IBT versions that omit a channel; predict
    must accept the frame as input without raising at the dispatch level.
    Underlying sklearn fitters MAY raise if they reject NaN — that's
    swallowed via the existing UntrainedError / ValueError handling.
    """
    model, baseline = bmw_model
    env = EnvironmentFrame.from_partial_row({
        "AirDensity": 1.18, "TrackTempCrew": 24.0, "WindVel": 2.5,
        "WindDir": 120.0, "TrackWetness": 0.0,
    })
    keys = sorted(model.fitters.keys())
    assert keys
    first = keys[0]
    if len(first) == 3:
        corner_id, phase_str, _ = first
    else:
        _, corner_id, phase_str, _ = first
    cpkey = CornerPhaseKey(
        session_id=model.session_ids[0],
        lap_index=1,
        corner_id=corner_id,
        phase=Phase(phase_str),
    )
    # Five known floats + sentinels for the rest. The dispatch picks
    # _env_to_array (12 floats) and feeds it to the v2/v3 fitters; some
    # may raise on NaN inside, but the dispatch level must not raise.
    out = model.predict(baseline, env, cpkey)
    assert isinstance(out.states, dict)
