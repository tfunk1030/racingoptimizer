from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from racingoptimizer.aero.loader import (
    AeroLoadError,
    AeroMapData,
    load_aero_map_data,
    parse_aero_filename,
)


# --- filename parser ---------------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        ("acura_wing_6.0.json", ("acura", 6.0)),
        ("acura_wing_6.5.json", ("acura", 6.5)),
        ("acura_wing_10.0.json", ("acura", 10.0)),
        ("bmw_wing_15.0.json", ("bmw", 15.0)),
        ("porsche_wing_17.0.json", ("porsche", 17.0)),
    ],
)
def test_parse_aero_filename_known(name: str, expected: tuple[str, float]) -> None:
    assert parse_aero_filename(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "wing_6.0.json",
        "acura_wing.json",
        "acura_wing_6.json",
        "acura_wing_6.0.txt",
        "acura_6.0.json",
    ],
)
def test_parse_aero_filename_invalid(name: str) -> None:
    with pytest.raises(ValueError):
        parse_aero_filename(name)


# --- real-corpus loader ------------------------------------------------------

@pytest.mark.parametrize(
    "car, expected_n_wing, expected_first_wing, expected_last_wing",
    [
        ("acura", 9, 6.0, 10.0),
        ("bmw", 6, 12.0, 17.0),
        ("cadillac", 6, 12.0, 17.0),
        ("ferrari", 6, 12.0, 17.0),
        ("porsche", 6, 12.0, 17.0),
    ],
)
def test_load_real_corpus_per_car(
    aero_dir: Path,
    car: str,
    expected_n_wing: int,
    expected_first_wing: float,
    expected_last_wing: float,
) -> None:
    data = load_aero_map_data(car, aero_dir=aero_dir)
    assert isinstance(data, AeroMapData)
    assert data.car == car
    assert len(data.wing_angles) == expected_n_wing
    assert data.wing_angles[0] == expected_first_wing
    assert data.wing_angles[-1] == expected_last_wing
    assert data.front_rh_mm.shape == (51,)
    assert data.rear_rh_mm.shape == (46,)
    assert data.balance_pct.shape == (expected_n_wing, 51, 46)
    assert data.ld_ratio.shape == (expected_n_wing, 51, 46)
    assert np.all(np.diff(data.front_rh_mm) > 0)
    assert np.all(np.diff(data.rear_rh_mm) > 0)
    assert np.all(np.diff(np.array(data.wing_angles)) > 0)
    assert not np.isnan(data.balance_pct).any()
    assert not np.isnan(data.ld_ratio).any()


def test_load_unknown_car_raises(aero_dir: Path) -> None:
    with pytest.raises(AeroLoadError, match="no aero maps"):
        load_aero_map_data("teslamodels", aero_dir=aero_dir)


def test_load_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_aero_map_data("acura", aero_dir=tmp_path / "does-not-exist")


# --- synthetic schema-violation tests ----------------------------------------

def _write_minimal_aero_json(path: Path, car: str, wing: float, **overrides) -> None:
    payload: dict = {
        "car": car,
        "wing": wing,
        "front_rh_mm": list(range(25, 76)),
        "rear_rh_mm": list(range(5, 51)),
        "balance_pct": [[50.0] * 46 for _ in range(51)],
        "ld_ratio": [[3.5] * 46 for _ in range(51)],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload))


def test_missing_required_key_raises(tmp_path: Path) -> None:
    bad = tmp_path / "synth_wing_6.0.json"
    payload = {
        "car": "synth",
        "wing": 6.0,
        "front_rh_mm": list(range(25, 76)),
        "balance_pct": [[50.0] * 46 for _ in range(51)],
        "ld_ratio": [[3.5] * 46 for _ in range(51)],
    }
    bad.write_text(json.dumps(payload))
    with pytest.raises(AeroLoadError, match="rear_rh_mm"):
        load_aero_map_data("synth", aero_dir=tmp_path)


def test_balance_shape_mismatch_raises(tmp_path: Path) -> None:
    _write_minimal_aero_json(
        tmp_path / "synth_wing_6.0.json",
        "synth",
        6.0,
        balance_pct=[[50.0] * 46 for _ in range(50)],
    )
    with pytest.raises(AeroLoadError, match="balance_pct"):
        load_aero_map_data("synth", aero_dir=tmp_path)


def test_axis_disagreement_across_wings_raises(tmp_path: Path) -> None:
    _write_minimal_aero_json(tmp_path / "synth_wing_6.0.json", "synth", 6.0)
    _write_minimal_aero_json(
        tmp_path / "synth_wing_7.0.json",
        "synth",
        7.0,
        front_rh_mm=list(range(20, 71)),
    )
    with pytest.raises(AeroLoadError, match="axis"):
        load_aero_map_data("synth", aero_dir=tmp_path)


def test_nan_cell_raises(tmp_path: Path) -> None:
    bad_balance = [[50.0] * 46 for _ in range(51)]
    bad_balance[10][10] = float("nan")
    _write_minimal_aero_json(
        tmp_path / "synth_wing_6.0.json", "synth", 6.0, balance_pct=bad_balance
    )
    with pytest.raises(AeroLoadError, match="NaN"):
        load_aero_map_data("synth", aero_dir=tmp_path)


def test_wing_angles_returned_sorted(tmp_path: Path) -> None:
    for w in [9.0, 6.0, 7.5, 8.0]:
        _write_minimal_aero_json(tmp_path / f"synth_wing_{w}.json", "synth", w)
    data = load_aero_map_data("synth", aero_dir=tmp_path)
    assert list(data.wing_angles) == [6.0, 7.5, 8.0, 9.0]


def test_filename_wing_must_match_payload_wing(tmp_path: Path) -> None:
    _write_minimal_aero_json(tmp_path / "synth_wing_6.0.json", "synth", 7.0)
    with pytest.raises(AeroLoadError, match="wing"):
        load_aero_map_data("synth", aero_dir=tmp_path)
