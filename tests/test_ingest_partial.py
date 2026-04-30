"""S2.7 — Partial salvage on parse failure.

VISION §1 "use everything, lose nothing": if YAML parses but a few channels
fail, store what we got. If lap segmentation fails but channels parse, store
the channels as a single anonymous "lap". Write `status="partial"` for these
cases. Only stamp `status="failed"` when there is genuinely nothing left to
keep (filesystem error, YAML/channels parse blew up, or writer failed
mid-stream).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from racingoptimizer.ingest import api, learn, sessions
from racingoptimizer.ingest.parser import ParseResult
from racingoptimizer.ingest.segment import LapSpan


def _fake_parse_result(
    *,
    yaml_car: str = "bmwlmdh",
    yaml_track: str = "Sebring International",
    n_samples: int = 600,
    samples_per_lap: int = 200,
    sample_rate_hz: float = 60.0,
    lap_spans: list[LapSpan] | None = None,
) -> ParseResult:
    pct = np.tile(
        np.linspace(0.0, 1.0, samples_per_lap, endpoint=False),
        n_samples // samples_per_lap,
    ).astype(np.float32)
    lap = np.repeat(
        np.arange(n_samples // samples_per_lap), samples_per_lap
    ).astype(np.float32)
    speed = np.linspace(0, 80.0, n_samples).astype(np.float32)
    if lap_spans is None:
        lap_spans = [
            LapSpan(i, i * samples_per_lap, (i + 1) * samples_per_lap, 1)
            for i in range(n_samples // samples_per_lap)
        ]
    return ParseResult(
        yaml_car=yaml_car,
        yaml_track=yaml_track,
        recorded_at="2026-04-29T00:00:00",
        duration_s=n_samples / sample_rate_hz,
        sample_rate_hz=sample_rate_hz,
        channels={
            "LapDistPct": pct,
            "Lap": lap,
            "Speed": speed,
            "Brake": np.zeros(n_samples, dtype=np.float32),
            "Throttle": np.ones(n_samples, dtype=np.float32),
        },
        setup={"chassis": {"front": {"wing": 16.0}}},
        weather_summary={"AirTemp_c_mean": 22.0},
        lap_spans=lap_spans,
        dropped_channels={},
    )


def _fake_ibt(tmp_path: Path, name: str = "fake.ibt") -> Path:
    """Write a non-empty bytes blob so api._process_one can hash it for sid."""
    p = tmp_path / name
    p.write_bytes(b"non-empty placeholder bytes for an .ibt fixture")
    return p


def test_no_laps_detected_writes_partial(
    tmp_path: Path, tmp_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Channels parsed, but lap segmentation produced no spans → partial."""
    ibt = _fake_ibt(tmp_path, "bmwlmdh_sebring international.ibt")
    pr = _fake_parse_result(lap_spans=[])
    monkeypatch.setattr(api, "parse_ibt", lambda _path: pr)

    ids = learn(ibt, corpus_root=tmp_corpus)
    assert len(ids) == 1

    s = sessions(corpus_root=tmp_corpus, valid_only=False)
    assert s.height == 1
    assert s["status"][0] == "partial"
    assert s["car"][0] == "bmw"
    assert s["track"][0] == "sebring_international"
    assert s["lap_count"][0] == 0
    assert s["error"][0] is not None
    # Channels MUST be persisted on partial — VISION §1 lose-nothing.
    assert s["parquet_path"][0]
    assert (tmp_corpus / s["parquet_path"][0]).exists()


def test_unknown_car_writes_partial(
    tmp_path: Path, tmp_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """YAML/filename both fail to identify a known car → partial, car=unknown."""
    # Filename does not match the iRacing pattern (no underscore separator
    # before track + datetime) so detect_car_from_filename returns None.
    ibt = _fake_ibt(tmp_path, "weirdfile.ibt")
    pr = _fake_parse_result(yaml_car="totally_made_up_car_xyz")
    monkeypatch.setattr(api, "parse_ibt", lambda _path: pr)

    ids = learn(ibt, corpus_root=tmp_corpus)
    assert len(ids) == 1

    s = sessions(corpus_root=tmp_corpus, valid_only=False)
    assert s.height == 1
    assert s["status"][0] == "partial"
    assert s["car"][0] == "unknown"
    # Channels persisted, lap_count > 0 because lap_spans were valid.
    assert s["parquet_path"][0]
    assert (tmp_corpus / s["parquet_path"][0]).exists()
    assert s["lap_count"][0] >= 1


def test_yaml_parse_failure_writes_failed_no_parquet(
    tmp_path: Path, tmp_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corrupt header / unparseable YAML → failed, no parquet written."""
    ibt = _fake_ibt(tmp_path, "bmwlmdh_sebring.ibt")

    def _boom(_path: Path) -> ParseResult:
        raise ValueError("corrupt YAML header")

    monkeypatch.setattr(api, "parse_ibt", _boom)

    ids = learn(ibt, corpus_root=tmp_corpus)
    assert len(ids) == 1

    s = sessions(corpus_root=tmp_corpus, valid_only=False)
    assert s.height == 1
    assert s["status"][0] == "failed"
    assert s["car"][0] == "unknown"
    assert s["track"][0] == "unknown"
    assert s["parquet_path"][0] is None
    assert s["error"][0]
    # No parquet for any car/track combo should exist.
    assert list((tmp_corpus / "sessions").rglob("*.parquet")) == []


def test_oserror_on_read_writes_failed_no_parquet(
    tmp_path: Path, tmp_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A filesystem error on read_bytes → failed, no parquet."""
    # File technically exists so _iter_ibt_paths picks it up, but
    # Path.read_bytes raises when api._process_one tries to consume it.
    ibt = _fake_ibt(tmp_path, "bmwlmdh_sebring.ibt")

    real_read_bytes = Path.read_bytes

    def _maybe_raise(self: Path) -> bytes:
        if self == ibt:
            raise OSError("simulated filesystem fault")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _maybe_raise)

    ids = learn(ibt, corpus_root=tmp_corpus)
    assert len(ids) == 1

    s = sessions(corpus_root=tmp_corpus, valid_only=False)
    assert s.height == 1
    assert s["status"][0] == "failed"
    assert "OSError" in (s["error"][0] or "")
    assert s["parquet_path"][0] is None
    assert list((tmp_corpus / "sessions").rglob("*.parquet")) == []


def test_partial_can_be_upgraded_to_ok_on_reingest(
    tmp_path: Path, tmp_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-ingest of a partial session that now parses cleanly → status=ok."""
    ibt = _fake_ibt(tmp_path, "bmwlmdh_sebring international.ibt")

    # First pass: lap segmentation produces nothing → partial.
    monkeypatch.setattr(api, "parse_ibt", lambda _p: _fake_parse_result(lap_spans=[]))
    ids1 = learn(ibt, corpus_root=tmp_corpus)
    s1 = sessions(corpus_root=tmp_corpus, valid_only=False)
    assert s1["status"][0] == "partial"

    # Second pass: full ParseResult, still the same session_id (same bytes).
    monkeypatch.setattr(api, "parse_ibt", lambda _p: _fake_parse_result())
    ids2 = learn(ibt, corpus_root=tmp_corpus)
    assert ids1 == ids2

    s2 = sessions(corpus_root=tmp_corpus, valid_only=False)
    assert s2.height == 1
    assert s2["status"][0] == "ok"
    assert s2["error"][0] is None
    assert s2["lap_count"][0] >= 1
