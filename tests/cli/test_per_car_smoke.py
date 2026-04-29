"""Per-car CLI smoke test (HARD MERGE GATE per CLAUDE.md / U11 brief).

`optimize <car> <track>` must exit 0 and produce a recognisable briefing
for every canonical GTP car whose fixture is present in `ibtfiles/`.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from racingoptimizer.cli import main

CARS = ["bmw", "acura", "cadillac", "ferrari", "porsche"]


@pytest.mark.parametrize("per_car_fixture", CARS, indirect=True)
def test_recommend_per_car_smoke(
    per_car_fixture: tuple[str, Path, str],
    tmp_corpus: Path,
) -> None:
    car, ibt, track_sub = per_car_fixture
    runner = CliRunner()

    learn_result = runner.invoke(
        main,
        ["learn", str(ibt), "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert learn_result.exit_code == 0, learn_result.output

    rec_result = runner.invoke(
        main,
        [car, track_sub, "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    if rec_result.exit_code != 0:
        msg = (
            f"recommend failed for {car}@{track_sub}: "
            f"exit={rec_result.exit_code}\n--- stdout ---\n{rec_result.output}"
        )
        if rec_result.exception is not None:
            import traceback
            tb = "".join(
                traceback.format_exception(
                    type(rec_result.exception),
                    rec_result.exception,
                    rec_result.exception.__traceback__,
                )
            )
            msg += f"\n--- traceback ---\n{tb}"
        pytest.fail(msg)

    out = rec_result.output
    assert car in out.lower(), f"car {car} missing from briefing"
    assert "confidence" in out.lower(), "no confidence line in briefing"
    # At least one parameter block (any human-formatted parameter line ending
    # in a unit). Sebring/etc briefings always emit at least one.
    has_param_block = any(
        "[confidence:" in line for line in out.splitlines()
    )
    assert has_param_block, f"no parameter block in briefing for {car}:\n{out}"


@pytest.mark.parametrize("per_car_fixture", CARS, indirect=True)
def test_recommend_per_car_json(
    per_car_fixture: tuple[str, Path, str],
    tmp_corpus: Path,
) -> None:
    """JSON output must be valid JSON for every car."""
    import json

    car, ibt, track_sub = per_car_fixture
    runner = CliRunner()
    runner.invoke(
        main,
        ["learn", str(ibt), "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    result = runner.invoke(
        main,
        [car, track_sub, "--json", "--corpus-root", str(tmp_corpus)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["car"] == car
    assert "parameters" in payload
    assert "confidence" in payload
