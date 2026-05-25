"""Static garage RH k-NN corpus: predict, DE feasibility, post-DE repair."""

from __future__ import annotations

from pathlib import Path

import pytest

from racingoptimizer.context import EnvironmentFrame
from racingoptimizer.physics.static_rh_knn import (
    StaticRhCorpusEntry,
    build_static_rh_corpus,
    cooptimize_tb_for_static_rh,
    enforce_static_rh_feasible,
    has_static_rh_physics,
    physics_static_rh_readouts,
    predict_static_rh_knn,
    static_rh_de_infeasible,
    static_rh_within_envelope,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

_ENV = EnvironmentFrame(
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


class _MockFitRecord:
    def __init__(self) -> None:
        self.fitter = type("F", (), {"is_trained": True})()


class _MockStaticRhModel:
    """Linear TB -> static RH for physics co-opt unit tests."""

    car = "acura"
    feature_schema_version = 4

    def __init__(self) -> None:
        self.fitters = {
            ("c", "mid_corner", ch): _MockFitRecord()
            for ch in (
                "setup_static_lf_ride_height_mm",
                "setup_static_rf_ride_height_mm",
                "setup_static_lr_ride_height_mm",
                "setup_static_rr_ride_height_mm",
            )
        }

    def predict_setup_readouts(
        self, setup: dict[str, float], env: EnvironmentFrame,
    ) -> dict[str, float]:
        tb_fl = float(setup.get("torsion_bar_turns_fl", 0.08))
        lf = 20.0 + tb_fl * 100.0
        tb_rl = float(setup.get("torsion_bar_turns_rl", -0.11))
        lr = 60.0 + tb_rl * 50.0
        return {
            "setup_static_lf_ride_height_mm": lf,
            "setup_static_rf_ride_height_mm": lf,
            "setup_static_lr_ride_height_mm": lr,
            "setup_static_rr_ride_height_mm": lr,
        }


def _legal_readouts(lf: float = 45.0, rf: float = 46.0) -> dict[str, float]:
    return {
        "setup_static_lf_ride_height_mm": lf,
        "setup_static_rf_ride_height_mm": rf,
        "setup_static_lr_ride_height_mm": 44.0,
        "setup_static_rr_ride_height_mm": 45.0,
    }


def _platform(**overrides: float) -> dict[str, float]:
    base = {
        "heave_spring_rate_n_per_mm": 140.0,
        "third_spring_rate_n_per_mm": 180.0,
        "heave_perch_offset_front_mm": 95.0,
        "spring_perch_offset_rear_mm": 100.0,
        "pushrod_length_offset_front_mm": 35.0,
        "pushrod_length_offset_rear_mm": 37.5,
        "fuel_level_l": 58.0,
        "camber_fl_deg": -2.5,
        "camber_rl_deg": -1.8,
        "toe_front_mm": 0.0,
        "toe_rl_mm": 1.0,
    }
    base.update(overrides)
    return base


def test_build_static_rh_corpus_skips_incomplete_sessions() -> None:
    sid_to_params = {
        "a": _platform(),
        "b": _platform(heave_spring_rate_n_per_mm=160.0),
    }
    sid_to_readouts = {
        "a": _legal_readouts(),
        "b": {"setup_static_lf_ride_height_mm": 40.0},  # too few channels
    }
    corpus = build_static_rh_corpus(sid_to_params, sid_to_readouts)
    assert len(corpus) == 1
    assert dict(corpus[0].params)["heave_spring_rate_n_per_mm"] == 140.0


def test_predict_static_rh_knn_exact_at_corpus_point() -> None:
    setup = _platform()
    readouts = _legal_readouts(lf=42.0, rf=43.0)
    corpus = (
        StaticRhCorpusEntry(
            params=tuple(sorted(setup.items())),
            readouts=tuple(sorted(readouts.items())),
        ),
    )
    result = predict_static_rh_knn(corpus, setup, car="acura", constraints=None)
    assert result is not None
    assert result.neighbor_distance == 0.0
    assert not result.extrapolated
    assert result.readouts["setup_static_lf_ride_height_mm"] == 42.0
    assert result.readouts["setup_static_rf_ride_height_mm"] == 43.0


def test_static_rh_de_infeasible_rejects_illegal_and_extrapolated() -> None:
    legal = predict_static_rh_knn(
        (
            StaticRhCorpusEntry(
                params=tuple(sorted(_platform().items())),
                readouts=tuple(sorted(_legal_readouts().items())),
            ),
        ),
        _platform(),
        car="acura",
        constraints=None,
    )
    assert legal is not None
    assert not static_rh_de_infeasible(legal)

    illegal_readouts = _legal_readouts(lf=26.0, rf=27.0)
    illegal = predict_static_rh_knn(
        (
            StaticRhCorpusEntry(
                params=tuple(sorted(_platform().items())),
                readouts=tuple(sorted(illegal_readouts.items())),
            ),
        ),
        _platform(heave_spring_rate_n_per_mm=999.0),
        car="acura",
        constraints=None,
    )
    assert illegal is not None
    assert static_rh_de_infeasible(illegal) or not static_rh_within_envelope(
        illegal.readouts,
    )


def test_enforce_static_rh_feasible_blends_toward_legal_neighbor() -> None:
    legal_setup = _platform()
    legal_readouts = _legal_readouts()
    bad_setup = _platform(
        heave_spring_rate_n_per_mm=50.0,
        heave_perch_offset_front_mm=10.0,
    )
    bad_readouts = _legal_readouts(lf=22.0, rf=23.0)
    corpus = (
        StaticRhCorpusEntry(
            params=tuple(sorted(legal_setup.items())),
            readouts=tuple(sorted(legal_readouts.items())),
        ),
        StaticRhCorpusEntry(
            params=tuple(sorted(bad_setup.items())),
            readouts=tuple(sorted(bad_readouts.items())),
        ),
    )
    trial = _platform(
        heave_spring_rate_n_per_mm=55.0,
        heave_perch_offset_front_mm=15.0,
    )
    repaired, still_bad = enforce_static_rh_feasible(
        trial, corpus, car="acura", constraints=None,
    )
    assert not still_bad
    knn = predict_static_rh_knn(corpus, repaired, car="acura", constraints=None)
    assert knn is not None
    assert static_rh_within_envelope(knn.readouts)
    assert not knn.extrapolated


def test_cooptimize_tb_trims_front_turns_with_physics() -> None:
    from racingoptimizer.constraints import load_constraints

    constraints = load_constraints(REPO_ROOT / "constraints.md")
    model = _MockStaticRhModel()
    assert has_static_rh_physics(model)

    coarse = _platform(
        heave_perch_offset_front_mm=49.0,
        pushrod_length_offset_front_mm=-24.5,
    )
    trial = {**coarse, "torsion_bar_turns_fl": 0.081, "torsion_bar_turns_rl": -0.110}
    before = physics_static_rh_readouts(model, trial, _ENV)
    assert before["setup_static_lf_ride_height_mm"] < 30.0

    tuned, ok = cooptimize_tb_for_static_rh(
        trial, model, _ENV, constraints=constraints,
    )
    assert ok
    after = physics_static_rh_readouts(model, tuned, _ENV)
    assert static_rh_within_envelope(after)
    assert after["setup_static_lf_ride_height_mm"] == pytest.approx(30.0, abs=0.15)
    assert tuned["torsion_bar_turns_fl"] > trial["torsion_bar_turns_fl"]
