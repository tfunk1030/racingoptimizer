"""`optimize <car> <track>` recommend command tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
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


def test_unknown_track_extrapolates_from_donor(multi_lap_ibt: Path, tmp_corpus: Path) -> None:
    """After S2.6 (gap #21), an unknown track no longer exits 2 — it extrapolates
    from the most-similar trained track and renders with a warning. Exit 2 is
    reserved for cases where the car has zero training data on ANY track.
    """
    runner = CliRunner()
    runner.invoke(main, ["learn", str(multi_lap_ibt), "--corpus-root", str(tmp_corpus)])
    result = runner.invoke(
        main,
        ["recommend", "bmw", "monza_unknown", "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # Warning must mention the unknown track + the donor track.
    assert "monza_unknown" in result.output or "untrained" in result.output.lower()


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
    assert "pinned by user" in result.output.lower()


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


def test_quali_without_fuel_exits_2(multi_lap_ibt: Path, tmp_corpus: Path) -> None:
    runner = CliRunner()
    runner.invoke(main, ["learn", str(multi_lap_ibt), "--corpus-root", str(tmp_corpus)])
    result = runner.invoke(
        main,
        ["bmw", "sebring", "--quali", "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
    assert "--fuel" in result.output


@pytest.mark.slow
def test_race_mode_auto_pins_fuel_to_past_session(
    multi_lap_ibt: Path, tmp_corpus: Path,
) -> None:
    """Without `--quali` AND without `--fuel`, race mode should anchor
    `fuel_level_l` to the most-recent past-session value at the target
    track, with a stderr banner. Slow because it runs the full DE search."""
    runner = CliRunner()
    runner.invoke(main, ["learn", str(multi_lap_ibt), "--corpus-root", str(tmp_corpus)])
    result = runner.invoke(
        main,
        ["bmw", "sebring", "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "Race fuel auto-pinned to past-session value" in result.stderr


@pytest.mark.slow
def test_reset_mode_emits_banner_and_runs(
    multi_lap_ibt: Path, tmp_corpus: Path,
) -> None:
    """`--reset` prints a banner to stderr and produces a recommendation.
    Slow because it runs the full DE search with widened envelope."""
    runner = CliRunner()
    runner.invoke(main, ["learn", str(multi_lap_ibt), "--corpus-root", str(tmp_corpus)])
    result = runner.invoke(
        main,
        ["bmw", "sebring", "--reset", "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "RESET MODE" in result.stderr


@pytest.mark.slow
def test_detailed_flag_emits_legacy_block_format(
    multi_lap_ibt: Path, tmp_corpus: Path,
) -> None:
    """`--detailed` selects render_recommendation_text -- which emits the
    `[confidence: <regime>]` per-parameter tag the validator agent uses.
    The default narrative emits no such tag."""
    runner = CliRunner()
    runner.invoke(main, ["learn", str(multi_lap_ibt), "--corpus-root", str(tmp_corpus)])
    result_default = runner.invoke(
        main,
        ["bmw", "sebring", "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    result_detailed = runner.invoke(
        main,
        ["bmw", "sebring", "--detailed", "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result_default.exit_code == 0
    assert result_detailed.exit_code == 0
    assert "[confidence:" not in result_default.output
    assert "[confidence:" in result_detailed.output
