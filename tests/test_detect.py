from __future__ import annotations

import pytest

from racingoptimizer.ingest.detect import (
    UnknownCarError,
    detect_car,
    detect_track_from_filename,
    normalize_car_key,
    slugify_track,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("acuraarx06gtp", "acura"),
        ("bmwlmdh", "bmw"),
        ("bmwm4gt3", "bmw"),
        ("cadillacvseriesrgtp", "cadillac"),
        ("ferrari499p", "ferrari"),
        ("porsche963gtp", "porsche"),
        ("porsche992rgt3", "porsche"),
    ],
)
def test_normalize_car_key_known(raw: str, expected: str) -> None:
    assert normalize_car_key(raw) == expected


def test_normalize_car_key_unknown_raises() -> None:
    with pytest.raises(UnknownCarError):
        normalize_car_key("teslamodels")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Daytona 2011 road", "daytona_2011_road"),
        ("Algarve GP", "algarve_gp"),
        ("Hockenheim Gp", "hockenheim_gp"),
        ("Laguna Seca", "laguna_seca"),
        ("Sebring International", "sebring_international"),
        ("Road Atlanta - Full", "road_atlanta_full"),
    ],
)
def test_slugify_track(raw: str, expected: str) -> None:
    assert slugify_track(raw) == expected


def test_detect_track_from_filename_uses_underscore_separator() -> None:
    name = "porsche963gtp_lagunaseca 2026-04-26 18-25-49.ibt"
    assert detect_track_from_filename(name) == "lagunaseca"


def test_detect_track_from_filename_strips_timestamp_suffix() -> None:
    name = "ferrari499p_algarve gp 2026-04-09 17-58-04.ibt"
    assert detect_track_from_filename(name) == "algarve_gp"


def test_detect_car_prefers_yaml_over_filename() -> None:
    yaml_car = "porsche963gtp"
    filename_car = "ferrari499p"
    assert detect_car(yaml_car=yaml_car, filename_car=filename_car) == "porsche"


def test_detect_car_falls_back_to_filename_when_yaml_missing() -> None:
    assert detect_car(yaml_car=None, filename_car="bmwlmdh") == "bmw"
