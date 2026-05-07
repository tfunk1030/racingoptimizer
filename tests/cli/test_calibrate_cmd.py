"""`optimize calibrate <car> <track>` subcommand tests."""
from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from racingoptimizer.cli import main


def test_calibrate_status_runs_after_ingest(
    multi_lap_ibt: Path, tmp_corpus: Path,
) -> None:
    """`--status` renders the per-parameter coverage table and exits 0."""
    runner = CliRunner()
    runner.invoke(main, ["learn", str(multi_lap_ibt), "--corpus-root", str(tmp_corpus)])
    result = runner.invoke(
        main,
        [
            "calibrate", "bmw", "sebring",
            "--status",
            "--corpus-root", str(tmp_corpus),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "calibration probe" in result.output.lower()
    assert "PARAMETER" in result.output
    assert "COVERAGE" in result.output
    # Status mode should NOT render the proposal block.
    assert "CALIBRATION TARGETS" not in result.output


def test_calibrate_default_proposes_targets(
    multi_lap_ibt: Path, tmp_corpus: Path,
) -> None:
    """Default mode emits a CALIBRATION TARGETS block + setup card."""
    runner = CliRunner()
    runner.invoke(main, ["learn", str(multi_lap_ibt), "--corpus-root", str(tmp_corpus)])
    result = runner.invoke(
        main,
        [
            "calibrate", "bmw", "sebring",
            "--targets", "2",
            "--output-file", "-",  # suppress file output
            "--corpus-root", str(tmp_corpus),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "CALIBRATION TARGETS" in result.output
    assert "WHAT THIS TEACHES" in result.output


def test_calibrate_unknown_car_exits_2(tmp_corpus: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "calibrate", "lambo", "sebring",
            "--corpus-root", str(tmp_corpus),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 2


def test_calibrate_no_corpus_exits_2(tmp_corpus: Path) -> None:
    """Asking for a car with zero ingested sessions should exit 2 with a
    helpful message."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "calibrate", "bmw", "sebring",
            "--corpus-root", str(tmp_corpus),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
    assert "no sessions" in result.output.lower() or "run `optimize learn`" in result.output.lower()
