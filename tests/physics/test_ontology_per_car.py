"""Per-car JSON-path resolution for every USER-input parameter.

CLAUDE.md warns that the 5 GTP cars diverge inside `Chassis.LeftFront /
LeftRear / Rear` — particularly around damper placement (Acura/Porsche use
a separate `Dampers` block; BMW/Cadillac/Ferrari inline). The 8 USER-input
parameters added in `bf2e48b` (heave/third/rear-coil spring rates, perch
offsets, pushrod offsets) hard-code one schema across all five cars.

This test parametrises over the 5 canonical IBT fixtures (one per GTP car)
and asserts that every parameter `fittable_parameters(car)` will hand to
the optimizer resolves against the real-IBT setup YAML for that car. A
failing car indicates either:

  * the per-car ontology dict needs an override entry (different YAML
    leaf name on that car), or
  * the iRacing YAML schema has drifted on that car.

Memory note: the test pulls only the YAML header — no channel decoding —
to keep the per-test footprint tiny. Each car is exercised in a single
pytest function so the fixture cost is paid once per car (5 functions
total, not 5 × N parameters).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from racingoptimizer.constraints import load_constraints
from racingoptimizer.physics.ontology import (
    fittable_parameters,
    ontology_for,
    setup_value,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
IBT_DIR = REPO_ROOT / "ibtfiles"

# One canonical fixture per car. Same fixtures as `tests/cli/conftest.py`
# but inlined so this test stays independent of the CLI fixture's lifecycle.
PER_CAR_FIXTURES: dict[str, str] = {
    "bmw": "bmwlmdh_sebring international 2026-03-22 14-52-24.ibt",
    "acura": "acuraarx06gtp_daytona 2011 road 2026-04-03 20-31-55.ibt",
    "cadillac": "cadillacvseriesrgtp_lagunaseca 2026-04-27 19-50-46.ibt",
    "ferrari": "ferrari499p_hockenheim gp 2026-03-31 15-49-42.ibt",
    "porsche": "porsche963gtp_algarve gp 2026-04-07 15-49-17.ibt",
}

# Parameters introduced in `bf2e48b` + cambers added in the second-pass
# audit's gap-A remediation. Plus the long-standing user-input parameters
# as a regression check.
_USER_INPUT_PARAMS_TO_CHECK: tuple[str, ...] = (
    "rear_wing_angle_deg",
    "tyre_cold_pressure_kpa",
    "heave_spring_rate_n_per_mm",
    "third_spring_rate_n_per_mm",
    "rear_coil_spring_rate_n_per_mm",
    "heave_perch_offset_front_mm",
    "spring_perch_offset_rear_mm",
    "third_perch_offset_rear_mm",
    "pushrod_length_offset_front_mm",
    "pushrod_length_offset_rear_mm",
    "camber_fl_deg",
    "camber_fr_deg",
    "camber_rl_deg",
    "camber_rr_deg",
)


_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _is_lfs_pointer(path: Path) -> bool:
    """Detect a git-lfs pointer file standing in for an unfetched IBT.

    `git lfs pull` materialises the real binary at the same path, but in CI
    or sandboxed checkouts the file may still be the ~130-byte text pointer.
    Trying to parse that pointer through irsdk's binary protocol allocates
    runaway memory (the parser misinterprets the `oid sha256:` text as a
    huge `session_info_len`). Skip cleanly instead.
    """
    if path.stat().st_size > 4096:
        return False
    try:
        with path.open("rb") as fh:
            return fh.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def _read_setup_yaml(ibt_path: Path) -> dict:
    """Open an IBT and pull only the embedded session YAML's `CarSetup` block.

    Avoids the full `parse_ibt` channel loop so each test stays in the
    ~50 MB-per-process envelope.
    """
    try:
        import irsdk
    except ImportError:  # pragma: no cover
        import pyirsdk as irsdk  # type: ignore[import-not-found, no-redef]

    from racingoptimizer.ingest.parser import _read_yaml

    ibt = irsdk.IBT()
    ibt.open(str(ibt_path))
    try:
        info = _read_yaml(ibt)
    finally:
        ibt.close()
    setup = info.get("CarSetup", {}) or {}
    if not isinstance(setup, dict):
        raise RuntimeError(f"unexpected CarSetup shape in {ibt_path}: {type(setup)}")
    return setup


@pytest.mark.parametrize("car", sorted(PER_CAR_FIXTURES.keys()))
def test_per_car_setup_yaml_resolves_every_user_input(car: str) -> None:
    """All USER-input parameters resolve against this car's real setup YAML.

    Combines the user-input-by-name check and the
    `fittable_parameters(car)` exhaustive check into one function so each
    car opens its IBT exactly once per pytest run (the suite runs near
    the OOM ceiling otherwise).
    """
    fixture_path = IBT_DIR / PER_CAR_FIXTURES[car]
    if not fixture_path.exists():
        pytest.skip(f"no real-IBT fixture available for {car}")
    if _is_lfs_pointer(fixture_path):
        pytest.skip(
            f"{car} fixture at {fixture_path.name} is a git-lfs pointer; "
            "run `git lfs pull` to materialise the real IBT before running this test"
        )

    setup = _read_setup_yaml(fixture_path)
    onto = ontology_for(car)

    # 1. Spot-check the named user-input parameters (gives sharper failure
    #    messages when one specific path is wrong).
    failures: list[str] = []
    for name in _USER_INPUT_PARAMS_TO_CHECK:
        value = setup_value(car, name, setup)
        if value is None:
            failures.append(
                f"{name} (path={onto[name].json_path})"
            )
    assert not failures, (
        f"{car}: {len(failures)} user-input parameter(s) missing from real "
        f"setup YAML. Adjust the per-car ontology dict or the JSON path:\n"
        + "\n".join(f"  - {f}" for f in failures)
    )

    # 2. Exhaustive: every parameter the optimizer would search over for
    #    this car must resolve. Catches regressions where adding a new
    #    bound to constraints.md silently activates a parameter whose
    #    JSON path hasn't been verified.
    constraints = load_constraints()
    fittable = fittable_parameters(car, constraints)
    unresolved = [
        name for name in fittable if setup_value(car, name, setup) is None
    ]
    assert not unresolved, (
        f"{car}: {len(unresolved)} fittable parameter(s) failed to resolve "
        f"against real setup YAML: {unresolved}. Either ontology JSON path "
        f"is wrong, or constraints.md should not list these as bounded."
    )
