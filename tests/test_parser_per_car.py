"""Per-car parser/writer/api smoke for slice A.

Slice A's other end-to-end test (`tests/test_ingest_smoke.py`) only covers the
BMW Sebring fixture. This loops the five canonical GTP car keys through a full
`parse_ibt -> learn -> sessions -> laps -> lap_data` cycle so per-car YAML
shape, channel coverage, and writer behaviour are all exercised. Cars without
a fixture in `ibtfiles/` skip cleanly.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from racingoptimizer.ingest.api import lap_data, laps, learn, sessions
from racingoptimizer.ingest.detect import CAR_PREFIX_MAP, normalize_car_key
from racingoptimizer.ingest.parser import parse_ibt

REPO_ROOT = Path(__file__).resolve().parent.parent
IBT_DIR = REPO_ROOT / "ibtfiles"

CANONICAL_CARS = ("bmw", "acura", "cadillac", "ferrari", "porsche")

SHOCK_DEFL_CHANNELS = ("LFshockDefl", "RFshockDefl", "LRshockDefl", "RRshockDefl")


def _find_fixture(car: str) -> Path | None:
    prefix = next((k for k, v in CAR_PREFIX_MAP.items() if v == car), None)
    if prefix is None:
        return None
    if not IBT_DIR.is_dir():
        return None
    matches = sorted(IBT_DIR.glob(f"{prefix}_*"))
    matches = [m for m in matches if "m4gt3" not in m.name.lower()]
    return matches[0] if matches else None


def _parse_recorded_at(value: str | None) -> datetime:
    assert value is not None, "recorded_at missing"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(value)


@pytest.mark.parametrize("car", CANONICAL_CARS)
def test_parse_write_query_per_car(car: str, tmp_corpus: Path) -> None:
    fixture = _find_fixture(car)
    if fixture is None:
        pytest.skip(f"no {car} fixture present in ibtfiles/")

    parsed = parse_ibt(fixture)

    assert isinstance(parsed.yaml_car, str) and parsed.yaml_car
    assert isinstance(_parse_recorded_at(parsed.recorded_at), datetime)
    assert parsed.duration_s > 0
    assert len(parsed.channels) >= 30, f"only {len(parsed.channels)} channels for {car}"
    assert parsed.lap_spans, "no lap spans detected"
    assert any(s.valid == 1 for s in parsed.lap_spans), "no valid laps detected"
    assert isinstance(parsed.setup, dict) and parsed.setup
    for key in ("AirTemp_c_mean", "TrackTempCrew_c_mean"):
        assert key in parsed.weather_summary, f"weather missing {key} for {car}"

    sids = learn(fixture, corpus_root=tmp_corpus)
    assert sids, "learn() returned no session_ids"
    sid = sids[0]

    sess = sessions(corpus_root=tmp_corpus, valid_only=False)
    assert isinstance(sess, pl.DataFrame) and sess.height > 0
    row = sess.filter(pl.col("session_id") == sid)
    assert row.height == 1
    assert row["status"][0] == "ok"
    assert row["car"][0] == normalize_car_key(parsed.yaml_car or car) == car

    lap_rows = laps(session_id=sid, corpus_root=tmp_corpus, valid_only=True)
    assert lap_rows.height > 0, f"no valid laps for {car}"

    best = lap_rows.sort("lap_time_s").row(0, named=True)
    df = lap_data(session_id=sid, lap_index=int(best["lap_index"]), corpus_root=tmp_corpus)

    assert df.height >= 100, f"{df.height} samples for best lap of {car}"
    assert list(df.columns)[:3] == ["t_s", "lap_index", "lap_dist_pct"]
    assert "data_quality_mask" in df.columns
    assert bool(df["data_quality_mask"].all())


@pytest.mark.parametrize("car", CANONICAL_CARS)
def test_shock_deflection_per_car(car: str, tmp_corpus: Path) -> None:
    fixture = _find_fixture(car)
    if fixture is None:
        pytest.skip(f"no {car} fixture present in ibtfiles/")

    sids = learn(fixture, corpus_root=tmp_corpus)
    assert sids
    sid = sids[0]
    valid = laps(session_id=sid, corpus_root=tmp_corpus, valid_only=True)
    if valid.height == 0:
        pytest.skip(f"{fixture.name} has no valid laps")
    lap_idx = int(valid["lap_index"][0])
    df = lap_data(session_id=sid, lap_index=lap_idx, corpus_root=tmp_corpus)

    cols = set(df.columns)
    if car == "acura":
        for ch in SHOCK_DEFL_CHANNELS:
            assert ch not in cols, f"acura unexpectedly has shock channel {ch}"
    else:
        for ch in SHOCK_DEFL_CHANNELS:
            assert ch in cols, f"{car} missing shock channel {ch}"
