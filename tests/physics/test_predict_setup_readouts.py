from __future__ import annotations

import json
from pathlib import Path

import pytest

from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.ingest.api import learn, sessions
from racingoptimizer.physics import fit
from racingoptimizer.physics.ontology import setup_value
from racingoptimizer.track import build_track_model

REPO_ROOT = Path(__file__).resolve().parents[2]
BMW_SEBRING_IBT = (
    REPO_ROOT / "ibtfiles" / "bmwlmdh_sebring international 2026-03-22 14-52-24.ibt"
)
_STATIC_RH_CHANNELS = (
    "setup_static_lf_ride_height_mm",
    "setup_static_rf_ride_height_mm",
    "setup_static_lr_ride_height_mm",
    "setup_static_rr_ride_height_mm",
)
_ONTOLOGY_READOUT_BY_CHANNEL = {
    "setup_static_lf_ride_height_mm": "static_ride_height_front_mm",
    "setup_static_rf_ride_height_mm": "static_ride_height_front_mm",
    "setup_static_lr_ride_height_mm": "static_ride_height_rear_mm",
    "setup_static_rr_ride_height_mm": "static_ride_height_rear_mm",
}
_STATIC_RH_TOLERANCE_MM = 3.0


@pytest.mark.slow
def test_predict_setup_readouts_matches_observed_static_ride_heights(
    tmp_path: Path,
) -> None:
    """Readout predictor should reproduce garage static RH near observed setup."""
    from tests._lfs_util import is_unmaterialised_lfs_pointer, lfs_skip_message

    if not BMW_SEBRING_IBT.exists():
        pytest.skip(f"missing BMW Sebring fixture at {BMW_SEBRING_IBT}")
    if is_unmaterialised_lfs_pointer(BMW_SEBRING_IBT):
        pytest.skip(lfs_skip_message(BMW_SEBRING_IBT))

    root = tmp_path / "corpus"
    root.mkdir()
    session_ids = learn(BMW_SEBRING_IBT, corpus_root=root)
    sess_df = sessions(corpus_root=root)
    row = sess_df.row(0, named=True)
    car = str(row["car"])
    track = str(row["track"])
    tm = build_track_model(track, session_ids, corpus_root=root)
    model = fit(car, session_ids, tm, corpus_root=root, k_folds=2, seed=0xC0FFEE)

    setup_blob_raw = row.get("setup")
    if not isinstance(setup_blob_raw, str) or not setup_blob_raw.strip():
        pytest.skip("catalog row has no setup blob")
    setup_blob = json.loads(setup_blob_raw)

    observed_setup = dict(model.baseline_setup)
    for name in model.baseline_setup:
        val = setup_value(car, name, setup_blob)
        if val is None:
            continue
        try:
            observed_setup[name] = float(val)
        except (TypeError, ValueError):
            continue

    env = EnvironmentFrame(
        air_temp_c=22.0,
        air_density=1.18,
        air_pressure_mbar=1013.0,
        relative_humidity=0.50,
        wind_vel_ms=0.0,
        wind_dir_deg=0.0,
        fog_level=0.0,
        track_temp_c=28.0,
        track_wetness=0.0,
        weather_declared_wet=False,
        precip_type=0,
        skies=1,
    )
    predicted = model.predict_setup_readouts(observed_setup, env)
    assert predicted, "predict_setup_readouts should emit setup readout channels"

    compared = 0
    for channel in _STATIC_RH_CHANNELS:
        actual_param = _ONTOLOGY_READOUT_BY_CHANNEL[channel]
        actual = setup_value(car, actual_param, setup_blob)
        if actual is None:
            continue
        assert channel in predicted, f"missing predicted readout for {channel}"
        delta = abs(float(predicted[channel]) - float(actual))
        assert delta <= _STATIC_RH_TOLERANCE_MM, (
            f"{channel} delta={delta:.3f}mm exceeds "
            f"{_STATIC_RH_TOLERANCE_MM:.1f}mm tolerance"
        )
        compared += 1
    assert compared >= 2, "fixture missing expected static RH channels"
