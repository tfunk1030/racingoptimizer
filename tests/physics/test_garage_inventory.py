from __future__ import annotations

from pathlib import Path

import pytest

from racingoptimizer.constraints import load_constraints
from racingoptimizer.physics.garage_inventory import (
    classify_unmapped_path,
    flatten_setup,
    inventory_setup,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
IBT_DIR = REPO_ROOT / "ibtfiles"

PER_CAR_FIXTURES: dict[str, str] = {
    "bmw": "bmwlmdh_sebring international 2026-03-22 14-52-24.ibt",
    "acura": "acuraarx06gtp_daytona 2011 road 2026-04-03 20-31-55.ibt",
    "cadillac": "cadillacvseriesrgtp_lagunaseca 2026-04-27 19-50-46.ibt",
    "ferrari": "ferrari499p_hockenheim gp 2026-03-31 15-49-42.ibt",
    "porsche": "porsche963gtp_algarve gp 2026-04-07 15-49-17.ibt",
}


def test_flatten_setup_returns_tuple_paths() -> None:
    setup = {"Chassis": {"Front": {"ToeIn": "-1.0 mm"}}, "UpdateCount": 7}

    leaves = flatten_setup(setup)

    assert leaves[("Chassis", "Front", "ToeIn")] == "-1.0 mm"
    assert leaves[("UpdateCount",)] == 7


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (("TiresAero", "AeroCalculator", "LD"), "unsupported_readout"),
        (("Chassis", "Front", "ToeIn"), "blocked_user_input"),
        (("UpdateCount",), "unsupported_non_setup"),
    ],
)
def test_classify_unmapped_path_has_no_silent_unknowns(
    path: tuple[str, ...],
    expected: str,
) -> None:
    classification, reason = classify_unmapped_path(path)

    assert classification == expected
    assert reason


@pytest.mark.parametrize("car", sorted(PER_CAR_FIXTURES))
def test_canonical_setup_leaves_are_all_classified(car: str) -> None:
    from racingoptimizer.ingest.parser import _read_yaml
    from tests._lfs_util import is_unmaterialised_lfs_pointer, lfs_skip_message

    try:
        import irsdk
    except ImportError:  # pragma: no cover
        import pyirsdk as irsdk  # type: ignore[import-not-found, no-redef]

    fixture = IBT_DIR / PER_CAR_FIXTURES[car]
    if not fixture.exists():
        pytest.skip(f"missing canonical fixture for {car}: {fixture}")
    if is_unmaterialised_lfs_pointer(fixture):
        pytest.skip(lfs_skip_message(fixture))

    ibt = irsdk.IBT()
    ibt.open(str(fixture))
    try:
        info = _read_yaml(ibt)
    finally:
        ibt.close()

    setup = info.get("CarSetup", {}) or {}
    inventory = inventory_setup(car, setup, load_constraints())

    assert len(inventory) == len(flatten_setup(setup))
    assert all(leaf.reason for leaf in inventory)
    assert all(leaf.classification != "blocked_user_input" or leaf.reason for leaf in inventory)
