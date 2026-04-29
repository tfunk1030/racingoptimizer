"""`optimize status` command tests."""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from racingoptimizer.cli import main


def test_status_no_data_emits_warning(tmp_corpus: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["status", "bmw", "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "no sessions ingested" in result.output


def test_status_after_learn(small_ibt: Path, tmp_corpus: Path) -> None:
    runner = CliRunner()
    runner.invoke(main, ["learn", str(small_ibt), "--corpus-root", str(tmp_corpus)])
    result = runner.invoke(
        main,
        ["status", "bmw", "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "sebring_international" in result.output
    assert "Overall regime" in result.output


def test_status_json_after_learn(small_ibt: Path, tmp_corpus: Path) -> None:
    runner = CliRunner()
    runner.invoke(main, ["learn", str(small_ibt), "--corpus-root", str(tmp_corpus)])
    result = runner.invoke(
        main,
        ["status", "bmw", "--json", "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["car"] == "bmw"
    assert any(c["track"] == "sebring_international" for c in payload["coverage"])


def test_status_unknown_car_exits_2(tmp_corpus: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["status", "lambo", "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
