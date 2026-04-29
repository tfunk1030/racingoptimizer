"""`optimize <car> <track>` recommend command tests."""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from racingoptimizer.cli import main


def test_unknown_car_exits_2(tmp_corpus: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["recommend", "lambo", "sebring", "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
    assert "unknown car" in result.output.lower()


def test_unknown_track_exits_2(multi_lap_ibt: Path, tmp_corpus: Path) -> None:
    runner = CliRunner()
    runner.invoke(main, ["learn", str(multi_lap_ibt), "--corpus-root", str(tmp_corpus)])
    result = runner.invoke(
        main,
        ["recommend", "bmw", "monza_unknown", "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
    assert "monza_unknown" in result.output


def test_recommend_sebring_succeeds(multi_lap_ibt: Path, tmp_corpus: Path) -> None:
    runner = CliRunner()
    runner.invoke(main, ["learn", str(multi_lap_ibt), "--corpus-root", str(tmp_corpus)])
    result = runner.invoke(
        main,
        ["bmw", "sebring", "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "bmw" in result.output.lower()
    assert "sebring" in result.output.lower()
    assert "Confidence:" in result.output


def test_pin_propagates(multi_lap_ibt: Path, tmp_corpus: Path) -> None:
    runner = CliRunner()
    runner.invoke(main, ["learn", str(multi_lap_ibt), "--corpus-root", str(tmp_corpus)])
    result = runner.invoke(
        main,
        [
            "bmw", "sebring",
            "--pin", "rear_wing_angle_deg=14.0",
            "--corpus-root", str(tmp_corpus),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "Pinned by user" in result.output


def test_invalid_pin_exits_2(multi_lap_ibt: Path, tmp_corpus: Path) -> None:
    runner = CliRunner()
    runner.invoke(main, ["learn", str(multi_lap_ibt), "--corpus-root", str(tmp_corpus)])
    result = runner.invoke(
        main,
        [
            "bmw", "sebring", "--pin", "not_a_real_param=1.0",
            "--corpus-root", str(tmp_corpus),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 2


def test_json_output_is_valid(multi_lap_ibt: Path, tmp_corpus: Path) -> None:
    runner = CliRunner()
    runner.invoke(main, ["learn", str(multi_lap_ibt), "--corpus-root", str(tmp_corpus)])
    result = runner.invoke(
        main,
        ["bmw", "sebring", "--json", "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["car"] == "bmw"
    assert payload["track"] == "sebring_international"
    assert "parameters" in payload
    assert "confidence" in payload
