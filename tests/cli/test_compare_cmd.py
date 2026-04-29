"""`optimize compare` command tests."""
from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from racingoptimizer.cli import main


def test_compare_self_vs_self(multi_lap_ibt: Path, tmp_corpus: Path) -> None:
    runner = CliRunner()
    runner.invoke(main, ["learn", str(multi_lap_ibt), "--corpus-root", str(tmp_corpus)])
    result = runner.invoke(
        main,
        [
            "compare", str(multi_lap_ibt), str(multi_lap_ibt),
            "--corpus-root", str(tmp_corpus),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # Self-vs-self -> total delta exactly 0
    assert "+0.000" in result.output


def test_compare_cross_car_exits_2(
    multi_lap_ibt: Path, tmp_corpus: Path,
) -> None:
    """Compare BMW Sebring against an Acura fixture; reject mismatch."""
    repo_root = Path(__file__).resolve().parents[2]
    acura = repo_root / "ibtfiles" / (
        "acuraarx06gtp_daytona 2011 road 2026-04-03 20-31-55.ibt"
    )
    if not acura.exists():
        import pytest
        pytest.skip(f"acura fixture missing: {acura}")
    runner = CliRunner()
    runner.invoke(main, ["learn", str(multi_lap_ibt), "--corpus-root", str(tmp_corpus)])
    runner.invoke(main, ["learn", str(acura), "--corpus-root", str(tmp_corpus)])
    result = runner.invoke(
        main,
        ["compare", str(multi_lap_ibt), str(acura), "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
    assert "compare across cars" in result.output.lower()
