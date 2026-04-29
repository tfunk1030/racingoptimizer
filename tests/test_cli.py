from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from racingoptimizer.cli import main


def test_optimize_learn_on_one_ibt(small_ibt: Path, tmp_corpus: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["learn", str(small_ibt), "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "ingested" in result.output.lower()


def test_optimize_learn_directory(tmp_path: Path, small_ibt: Path, tmp_corpus: Path) -> None:
    nested = tmp_path / "subdir"
    nested.mkdir()
    (nested / small_ibt.name).write_bytes(small_ibt.read_bytes())
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["learn", str(tmp_path), "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0


def test_optimize_help_lists_learn() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "learn" in result.output
