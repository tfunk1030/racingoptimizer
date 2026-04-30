"""Untrained-track extrapolation (S2.6 / spec §10).

Running `optimize <car> <untrained-track>` against a corpus where the car
has been driven on at least one other track must NOT exit 2 — it must
extrapolate from the trained track with the most sessions, force every
parameter's confidence regime to `sparse`, and emit a top-level warning.
"""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from racingoptimizer.cli import main


def test_untrained_track_extrapolates_from_donor(
    multi_lap_ibt: Path, tmp_corpus: Path,
) -> None:
    runner = CliRunner()
    learn = runner.invoke(
        main,
        ["learn", str(multi_lap_ibt), "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert learn.exit_code == 0, learn.output

    result = runner.invoke(
        main,
        [
            "bmw", "daytona_2011_road",
            "--json",
            "--corpus-root", str(tmp_corpus),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    warnings = payload["warnings"]
    assert any(
        "untrained" in w and "extrapolated from sebring_international" in w
        for w in warnings
    ), f"missing untrained/extrapolated warning in: {warnings}"

    parameters = payload["parameters"]
    assert parameters, "no parameter blocks rendered"
    for param in parameters:
        regime = param["confidence"]["regime"]
        assert regime == "sparse", (
            f"parameter {param['parameter']!r} has regime {regime!r}; "
            f"every parameter must be sparse on an untrained track"
        )


def test_untrained_track_with_no_other_data_exits_2(tmp_corpus: Path) -> None:
    """When the car has zero sessions, fall through to the original exit-2."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["bmw", "daytona_2011_road", "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
    assert "no data" in result.output.lower() or "learn" in result.output.lower()
