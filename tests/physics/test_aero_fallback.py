from __future__ import annotations

from pathlib import Path

import pytest

from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.corner import CornerPhaseKey, Phase
from racingoptimizer.ingest.api import learn, sessions
from racingoptimizer.physics import fit
from racingoptimizer.physics.model import (
    _REGIME_DOWNGRADE,
    AERO_DEPENDENT_CHANNELS,
)
from racingoptimizer.track import build_track_model

REPO_ROOT = Path(__file__).resolve().parents[2]
BMW_SEBRING_IBT = REPO_ROOT / "ibtfiles" / "bmwlmdh_sebring international 2026-03-22 14-52-24.ibt"


def test_fit_without_aero_marks_correction_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if not BMW_SEBRING_IBT.exists():
        pytest.skip(f"missing BMW Sebring fixture at {BMW_SEBRING_IBT}")
    root = tmp_path / "corpus"
    root.mkdir()
    sids = learn(BMW_SEBRING_IBT, corpus_root=root)
    sess_df = sessions(corpus_root=root)
    car = sess_df.row(0, named=True)["car"]
    track = sess_df.row(0, named=True)["track"]
    tm = build_track_model(track, sids, corpus_root=root)

    # Force load_aero_maps to behave as if slice C is unavailable.
    def _raise(*_a, **_kw):
        raise ImportError("simulated missing slice C")

    import racingoptimizer.physics.fitter as fitter_mod
    monkeypatch.setattr(
        "racingoptimizer.aero.load_aero_maps", _raise, raising=True
    )
    # Re-import inside fitter is via the closure in _try_load_aero — patch the
    # module-level import as well.
    monkeypatch.setattr(fitter_mod, "_try_load_aero", lambda _car: False)

    model = fit(car, sids, tm, corpus_root=root, k_folds=2, seed=0xC0FFEE)
    assert model.aero_correction_available is False


def test_predict_downgrades_aero_channel_regime() -> None:
    """Verify the regime-downgrade table mirrors spec §9 (one-tier reduction)."""
    # dense → confident, confident → noisy, noisy → sparse, sparse → sparse.
    assert _REGIME_DOWNGRADE["dense"] == "confident"
    assert _REGIME_DOWNGRADE["confident"] == "noisy"
    assert _REGIME_DOWNGRADE["noisy"] == "sparse"
    assert _REGIME_DOWNGRADE["sparse"] == "sparse"


def test_aero_dependent_channels_cover_ride_heights() -> None:
    """Spec §9: aero-derived sub-utilizations rely on ride-height predictions."""
    expected = {
        "lf_ride_height_mean_mm",
        "rf_ride_height_mean_mm",
        "lr_ride_height_mean_mm",
        "rr_ride_height_mean_mm",
    }
    assert AERO_DEPENDENT_CHANNELS >= expected


def test_predict_aero_unavailable_downgrades_one_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not BMW_SEBRING_IBT.exists():
        pytest.skip(f"missing BMW Sebring fixture at {BMW_SEBRING_IBT}")
    root = tmp_path / "corpus"
    root.mkdir()
    sids = learn(BMW_SEBRING_IBT, corpus_root=root)
    sess_df = sessions(corpus_root=root)
    car = sess_df.row(0, named=True)["car"]
    track = sess_df.row(0, named=True)["track"]
    tm = build_track_model(track, sids, corpus_root=root)

    import racingoptimizer.physics.fitter as fitter_mod
    monkeypatch.setattr(fitter_mod, "_try_load_aero", lambda _car: False)
    model = fit(car, sids, tm, corpus_root=root, k_folds=2, seed=0xC0FFEE)
    assert model.aero_correction_available is False
    env = EnvironmentFrame(
        air_density=1.18, track_temp_c=24.0, wind_vel_ms=2.5,
        wind_dir_deg=120.0, track_wetness=0.0,
    )
    keys = sorted(model.fitters.keys())
    assert keys
    # Pick a key whose channel is aero-dependent if one exists in this fixture.
    aero_keys = [k for k in keys if k[3] in AERO_DEPENDENT_CHANNELS]
    if not aero_keys:
        pytest.skip("fixture produced no aero-dependent channel fitters")
    _, corner_id, phase_str, _ = aero_keys[0]
    cpkey = CornerPhaseKey(
        session_id=model.session_ids[0], lap_index=1,
        corner_id=corner_id, phase=Phase(phase_str),
    )
    out = model.predict(dict(model.baseline_setup), env, cpkey)
    # Every emitted aero channel must NOT be `dense` (downgrade applied).
    for ch, conf in out.states.items():
        if ch in AERO_DEPENDENT_CHANNELS:
            assert conf.regime != "dense"
