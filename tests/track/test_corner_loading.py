"""Per-corner loading classifier (S4.5 / VISION §9).

Synthetic 3-session corpus: each session contributes one lap whose
`corner_phase_states` table is hand-crafted so each of the three corners
is engineered to fire a different heuristic:

- corner 0 → ``front_limited``    : high understeer + low rear-shock spread
- corner 1 → ``rear_limited``     : high yaw-rate at EXIT + low front-shock spread
- corner 2 → ``aero_limited``     : high mean speed + lat-G correlated with speed

The test patches the catalog API and `corner_phase_states` so the
classifier's heuristic — not the IBT pipeline — is what's under test.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from racingoptimizer.corner.phase import Phase
from racingoptimizer.track import classify_corner_loading
from racingoptimizer.track.corner_loading import _CLASSIFY_OUTPUT_SCHEMA

_TRACK = "synthetic_track"
_SIDS: tuple[str, ...] = ("sess_0", "sess_1", "sess_2")


def _front_limited_phases(*, session_seed: float) -> list[dict[str, object]]:
    """Corner 0: very high understeer + tightly-clustered rear-shock p99 (low spread).

    `understeer_angle_mean_rad` ≈ 0.18 (>> 0.05 threshold).
    Rear-shock p99 across phases lives in [9.9, 10.3] mm — std-dev ≈ 0.15 mm.
    """
    base = [
        # phase, understeer, lf_p99, rf_p99, lr_p99, rr_p99
        (Phase.BRAKING.value,     0.16, 30.0, 32.0, 10.0, 10.2),
        (Phase.TRAIL_BRAKE.value, 0.18, 28.0, 31.0, 10.1, 10.3),
        (Phase.MID_CORNER.value,  0.20, 25.0, 27.0,  9.9, 10.0),
        (Phase.EXIT.value,        0.18, 24.0, 26.0, 10.0, 10.1),
    ]
    rows: list[dict[str, object]] = []
    for phase, understeer, lf, rf, lr, rr in base:
        rows.append(
            {
                "corner_id": 0,
                "phase": phase,
                "understeer_angle_mean_rad": understeer + session_seed * 0.005,
                "lf_shock_defl_p99_mm": lf,
                "rf_shock_defl_p99_mm": rf,
                "lr_shock_defl_p99_mm": lr,
                "rr_shock_defl_p99_mm": rr,
                "yaw_rate_max_rad_s": 0.05,
                "traction_util_mean": 0.005,
                "speed_mean_ms": 30.0,
                "accel_lat_g_max": 1.4,
            }
        )
    return rows


def _rear_limited_phases(*, session_seed: float) -> list[dict[str, object]]:
    """Corner 1: very high yaw-rate at EXIT + tight front-shock p99 (low spread).

    EXIT `yaw_rate_max_rad_s` ≈ 0.95 rad/s (>> 0.35 threshold).
    Front-shock p99 across phases sits in [10.0, 10.4] — low spread.
    """
    base = [
        # phase, yaw_rate, lf_p99, rf_p99, lr_p99, rr_p99
        (Phase.BRAKING.value,     0.10, 10.0, 10.2, 25.0, 27.0),
        (Phase.TRAIL_BRAKE.value, 0.20, 10.1, 10.3, 28.0, 30.0),
        (Phase.MID_CORNER.value,  0.30, 10.0, 10.4, 30.0, 32.0),
        (Phase.EXIT.value,        0.95, 10.2, 10.3, 32.0, 34.0),
    ]
    rows: list[dict[str, object]] = []
    for phase, yaw, lf, rf, lr, rr in base:
        rows.append(
            {
                "corner_id": 1,
                "phase": phase,
                "understeer_angle_mean_rad": 0.01,
                "lf_shock_defl_p99_mm": lf,
                "rf_shock_defl_p99_mm": rf,
                "lr_shock_defl_p99_mm": lr,
                "rr_shock_defl_p99_mm": rr,
                "yaw_rate_max_rad_s": yaw + session_seed * 0.01,
                "traction_util_mean": 0.005,
                "speed_mean_ms": 28.0,
                "accel_lat_g_max": 1.5,
            }
        )
    return rows


def _aero_limited_phases(*, session_seed: float) -> list[dict[str, object]]:
    """Corner 2: high-speed corner whose lat-G rises with speed across sessions.

    Mean speed ≈ 70 m/s (>> 55 m/s threshold). Per-session pairs of
    (speed, lat-G) trace a strongly positive line so corr → ~1.
    """
    # Each session shifts speed/lat-G together; correlation across the
    # stacked rows is strongly positive → triggers aero_limited.
    speed_offset = 5.0 * session_seed
    base = [
        (Phase.BRAKING.value,     65.0 + speed_offset, 1.4 + 0.10 * session_seed),
        (Phase.TRAIL_BRAKE.value, 68.0 + speed_offset, 1.5 + 0.10 * session_seed),
        (Phase.MID_CORNER.value,  70.0 + speed_offset, 1.6 + 0.10 * session_seed),
        (Phase.EXIT.value,        72.0 + speed_offset, 1.7 + 0.10 * session_seed),
    ]
    rows: list[dict[str, object]] = []
    for phase, speed, lat_g in base:
        rows.append(
            {
                "corner_id": 2,
                "phase": phase,
                "understeer_angle_mean_rad": 0.02,
                "lf_shock_defl_p99_mm": 20.0,
                "rf_shock_defl_p99_mm": 22.0,
                "lr_shock_defl_p99_mm": 21.0,
                "rr_shock_defl_p99_mm": 23.0,
                "yaw_rate_max_rad_s": 0.10,
                "traction_util_mean": 0.005,
                "speed_mean_ms": speed,
                "accel_lat_g_max": lat_g,
            }
        )
    return rows


def _build_synthetic_cps(sid: str) -> pl.DataFrame:
    """Compose one (session, lap) corner-phase-states frame with all 3 corners."""
    seed = float(_SIDS.index(sid))
    rows = (
        _front_limited_phases(session_seed=seed)
        + _rear_limited_phases(session_seed=seed)
        + _aero_limited_phases(session_seed=seed)
    )
    df = pl.DataFrame(rows)
    return df.with_columns(
        pl.lit(sid, dtype=pl.Utf8).alias("session_id"),
        pl.lit(0, dtype=pl.Int32).alias("lap_index"),
        pl.col("corner_id").cast(pl.Int32),
    )


@pytest.fixture
def patched_cps(monkeypatch, tmp_corpus: Path):
    """Patch the catalog/CPS surface so `classify_corner_loading` sees the synthetic corpus."""
    sessions_df = pl.DataFrame(
        {
            "session_id": list(_SIDS),
            "car": ["bmw"] * len(_SIDS),
            "track": [_TRACK] * len(_SIDS),
        }
    )
    laps_by_sid = {sid: pl.DataFrame({"lap_index": [0]}) for sid in _SIDS}

    def fake_sessions(track=None, valid_only=True, corpus_root=None, **_):
        if track is not None and track != _TRACK:
            return pl.DataFrame({"session_id": []})
        return sessions_df

    def fake_laps(session_id=None, **_):
        return laps_by_sid.get(
            session_id, pl.DataFrame({"lap_index": pl.Series([], dtype=pl.Int64)})
        )

    def fake_corner_phase_states(session_id, lap_index, *, corpus_root=None, **_):
        if session_id not in _SIDS:
            raise KeyError(session_id)
        return _build_synthetic_cps(session_id)

    monkeypatch.setattr(
        "racingoptimizer.track.corner_loading.ingest_api.sessions", fake_sessions
    )
    monkeypatch.setattr(
        "racingoptimizer.track.corner_loading.ingest_api.laps", fake_laps
    )
    monkeypatch.setattr(
        "racingoptimizer.track.corner_loading.corner_phase_states",
        fake_corner_phase_states,
    )
    return tmp_corpus


def test_classifier_assigns_each_synthetic_corner_to_its_target_label(patched_cps):
    df = classify_corner_loading(_TRACK, list(_SIDS), corpus_root=patched_cps)

    assert df.height == 3
    expected = {0: "front_limited", 1: "rear_limited", 2: "aero_limited"}
    seen = dict(
        zip(df["corner_id"].to_list(), df["classification"].to_list(), strict=True)
    )
    assert seen == expected


def test_classifier_reports_observation_count_and_confidence(patched_cps):
    df = classify_corner_loading(_TRACK, list(_SIDS), corpus_root=patched_cps)

    # Each corner has 4 phase rows × 3 sessions = 12 observations stacked.
    for n in df["n_observations"].to_list():
        assert n == 12
    # Confidence is in [0, 1] for every classified corner; non-mixed
    # classifications must report > 0.
    confidences = df["confidence"].to_list()
    assert all(0.0 <= c <= 1.0 for c in confidences)
    assert all(c > 0.0 for c in confidences)


def test_classifier_returns_empty_when_no_sessions(patched_cps):
    """Empty session list short-circuits to the schema-only frame."""
    df = classify_corner_loading(_TRACK, [], corpus_root=patched_cps)
    assert df.height == 0
    assert set(df.columns) == set(_CLASSIFY_OUTPUT_SCHEMA.keys())


def test_classifier_returns_empty_when_track_unknown(patched_cps):
    """A track the catalog does not know about yields no observations → empty frame."""
    df = classify_corner_loading(
        "unknown_track", list(_SIDS), corpus_root=patched_cps
    )
    assert df.height == 0


def test_track_model_corner_loading_property_uses_classifier(monkeypatch):
    """`TrackModel.corner_loading` is a thin lazy passthrough to `classify_corner_loading`."""
    from racingoptimizer.track.builder import TrackModel

    sentinel = pl.DataFrame(
        {"corner_id": [7], "classification": ["mixed"], "confidence": [0.0],
         "n_observations": [1]},
        schema=_CLASSIFY_OUTPUT_SCHEMA,
    )

    def fake_classify(track, session_ids, *, corpus_root=None):
        assert track == "fake"
        assert session_ids == ["a", "b"]
        return sentinel

    monkeypatch.setattr(
        "racingoptimizer.track.corner_loading.classify_corner_loading", fake_classify
    )

    tm = TrackModel(
        track="fake",
        regime="compounding",
        session_ids=("a", "b"),
        bin_size_m=5.0,
        bump_map=pl.DataFrame(),
        grip_map=pl.DataFrame(),
        speed_envelope=pl.DataFrame(),
        cache_path=Path("/dev/null/cache"),
        summary_path=Path("/dev/null/summary"),
    )
    out = tm.corner_loading
    assert out.equals(sentinel)
